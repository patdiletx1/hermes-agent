"""AES-256-CBC encryption/decryption for connection-string storage.

Provides encrypt/decrypt functions that use a master key stored in the
``HERMES_MASTER_ENCRYPTION_KEY`` environment variable.  The key must be a
base64-encoded 32-byte value (44-character base64 string).

The combined ciphertext format is URL-safe base64 of::

    IV (16 bytes) ‖ ciphertext (N bytes)

Wire format (one string, no delimiters):
    ``base64_urlsafe(IV + ciphertext)``

Because AES-CBC uses a fixed 16-byte IV, the receiver splits off the first
16 decoded bytes as the IV and decrypts the remainder.

Usage::

    import os
    from hermes_crypto import (
        encrypt_connection_string,
        decrypt_connection_string,
        generate_encryption_key,
    )

    os.environ["HERMES_MASTER_ENCRYPTION_KEY"] = generate_encryption_key()
    encrypted = encrypt_connection_string("postgresql://user:pass@host/db")
    decrypted = decrypt_connection_string(encrypted)
    assert decrypted == "postgresql://user:pass@host/db"
"""

from __future__ import annotations

import base64
import os
from typing import Final

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
except ImportError as _crypto_import_error:
    import warnings as _warnings

    _warnings.warn(
        "cryptography package not available; hermes_crypto is disabled. "
        "Install with: uv pip install cryptography",
        ImportWarning,
        stacklevel=2,
    )
    Cipher = None  # type: ignore[assignment]
    algorithms = None  # type: ignore[assignment]
    modes = None  # type: ignore[assignment]


# ── Constants ─────────────────────────────────────────────────────────────

_KEY_ENV_VAR: Final[str] = "HERMES_MASTER_ENCRYPTION_KEY"
_AES_BLOCK_SIZE: Final[int] = 16  # AES block size in bytes
_AES_KEY_LENGTH: Final[int] = 32  # AES-256 requires 32 bytes
_IV_LENGTH: Final[int] = 16  # AES-CBC IV is always 16 bytes

# Module-level cache for the decoded key
_cached_key: bytes | None = None


# ── Exception Hierarchy ───────────────────────────────────────────────────


class CryptoError(Exception):
    """Base exception for all crypto operations."""


class InvalidKeyError(CryptoError):
    """Raised when the master key is missing, wrong length, or corrupt."""


class DecryptError(CryptoError):
    """Raised when decryption fails (corrupted data, wrong key, padding)."""


class EncryptError(CryptoError):
    """Raised when encryption fails."""


# ── PKCS7 Padding ─────────────────────────────────────────────────────────


class _PKCS7Padding:
    """PKCS7 padding for 16-byte AES block size."""

    block_size: Final[int] = _AES_BLOCK_SIZE

    @classmethod
    def pad(cls, data: bytes) -> bytes:
        """Apply PKCS7 padding to *data*."""
        amount = cls.block_size - (len(data) % cls.block_size)
        if amount == 0:
            amount = cls.block_size
        return data + bytes([amount]) * amount

    @classmethod
    def unpad(cls, data: bytes) -> bytes:
        """Remove and validate PKCS7 padding from *data*.

        Raises:
            DecryptError: If the padding is invalid.
        """
        if not data:
            raise DecryptError("empty decrypted payload")
        pad = data[-1]
        if pad < 1 or pad > cls.block_size:
            raise DecryptError(
                f"invalid PKCS7 padding byte: {pad} "
                f"(expected 1-{cls.block_size})"
            )
        if data[-pad:] != bytes([pad]) * pad:
            raise DecryptError("malformed PKCS7 padding")
        return data[:-pad]


# ── Key Management ────────────────────────────────────────────────────────


def _check_crypto_available() -> None:
    """Raise if the cryptography library is not installed."""
    if Cipher is None:
        raise CryptoError(
            "cryptography package is not installed. "
            "Run: uv pip install cryptography"
        )


def _get_master_key() -> bytes:
    """Load and validate the master AES-256 key from the environment.

    Loads the key once and caches it in the module-level ``_cached_key``
    variable.  Subsequent calls return the cached value without re-reading
    the environment.

    Returns:
        The raw 32-byte AES-256 key.

    Raises:
        InvalidKeyError: If the env var is unset, empty, not valid
            base64, or does not decode to exactly 32 bytes.
    """
    _check_crypto_available()

    global _cached_key
    if _cached_key is not None:
        return _cached_key

    encoded = os.environ.get(_KEY_ENV_VAR)
    if not encoded:
        raise InvalidKeyError(
            f"{_KEY_ENV_VAR} is not set or is empty. "
            "Set it to a base64-encoded 32-byte value "
            "(generate one with generate_encryption_key())."
        )

    try:
        key = base64.b64decode(encoded)
    except Exception as exc:
        raise InvalidKeyError(
            f"{_KEY_ENV_VAR} is not valid base64: {exc}"
        ) from exc

    if len(key) != _AES_KEY_LENGTH:
        raise InvalidKeyError(
            f"{_KEY_ENV_VAR} must decode to exactly "
            f"{_AES_KEY_LENGTH} bytes "
            f"(got {len(key)}). Generate one with generate_encryption_key()."
        )

    _cached_key = key
    return key


def generate_encryption_key() -> str:
    """Generate a new random AES-256 key as a base64 string.

    Returns:
        A 44-character base64-encoded 32-byte key suitable for use as
        ``HERMES_MASTER_ENCRYPTION_KEY``.
    """
    return base64.b64encode(os.urandom(_AES_KEY_LENGTH)).decode("ascii")


# ── Public API ────────────────────────────────────────────────────────────


def encrypt_connection_string(plaintext: str) -> str:
    """Encrypt a connection string with AES-256-CBC + PKCS7 padding.

    Generates a random 16-byte IV for each call, encrypts the UTF-8-encoded
    plaintext, and returns a single URL-safe base64 string combining the IV
    and the ciphertext.

    Args:
        plaintext: The connection string to encrypt (e.g.
            ``"postgresql://user:pass@host:5432/dbname"``).

    Returns:
        A URL-safe base64 string encoding ``IV (16 bytes) ‖ ciphertext``
        suitable for storage in the ``encrypted_connection_string`` column.

    Raises:
        InvalidKeyError: If ``HERMES_MASTER_ENCRYPTION_KEY`` is not set or
            invalid.
        EncryptError: If encryption fails for any other reason.
    """
    key = _get_master_key()
    iv = os.urandom(_IV_LENGTH)

    try:
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        padded = _PKCS7Padding.pad(plaintext.encode("utf-8"))
        ciphertext = encryptor.update(padded) + encryptor.finalize()
    except Exception as exc:
        raise EncryptError(f"encryption failed: {exc}") from exc

    combined = base64.urlsafe_b64encode(iv + ciphertext).decode("ascii")
    return combined


def decrypt_connection_string(combined: str) -> str:
    """Decrypt a combined string produced by :func:`encrypt_connection_string`.

    Args:
        combined: The URL-safe base64 string returned by
            :func:`encrypt_connection_string`.

    Returns:
        The original plaintext connection string.

    Raises:
        InvalidKeyError: If ``HERMES_MASTER_ENCRYPTION_KEY`` is not set or
            invalid.
        DecryptError: If the data is corrupt, the key is wrong, or padding
            validation fails.
    """
    key = _get_master_key()

    try:
        raw = base64.urlsafe_b64decode(combined)
    except Exception as exc:
        raise DecryptError(f"invalid base64 input: {exc}") from exc

    if len(raw) < _IV_LENGTH + 1:
        raise DecryptError(
            f"input too short: expected at least {_IV_LENGTH + 1} bytes, "
            f"got {len(raw)}"
        )

    iv = raw[:_IV_LENGTH]
    ciphertext = raw[_IV_LENGTH:]

    try:
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        padded = decryptor.update(ciphertext) + decryptor.finalize()
        plaintext = _PKCS7Padding.unpad(padded)
    except CryptoError:
        raise
    except Exception as exc:
        raise DecryptError(f"decryption failed: {exc}") from exc

    return plaintext.decode("utf-8")


# ── Public API surface ────────────────────────────────────────────────────

__all__ = [
    "CryptoError",
    "InvalidKeyError",
    "DecryptError",
    "EncryptError",
    "encrypt_connection_string",
    "decrypt_connection_string",
    "generate_encryption_key",
]
