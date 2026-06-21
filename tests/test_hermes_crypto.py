"""Tests for the hermes_crypto AES-256-CBC encryption/decryption module."""

from __future__ import annotations

import base64
import os

import pytest

from hermes_crypto import (
    CryptoError,
    DecryptError,
    EncryptError,
    InvalidKeyError,
    decrypt_connection_string,
    encrypt_connection_string,
    generate_encryption_key,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def master_key() -> str:
    """Generate a valid master key and set it in the environment.

    Also resets the module-level cache so each test gets a fresh key read.
    """
    key = generate_encryption_key()
    os.environ["HERMES_MASTER_ENCRYPTION_KEY"] = key
    import hermes_crypto

    hermes_crypto._cached_key = None
    return key


# ── Round-trip tests ──────────────────────────────────────────────────────


def test_roundtrip(master_key: str) -> None:
    """Encrypt then decrypt returns the original plaintext."""
    original = "postgresql://user:pass@host:5432/dbname?sslmode=require"
    encrypted = encrypt_connection_string(original)
    assert isinstance(encrypted, str)
    assert encrypted != original
    decrypted = decrypt_connection_string(encrypted)
    assert decrypted == original


def test_roundtrip_empty_string(master_key: str) -> None:
    """Empty string round-trips correctly."""
    original = ""
    encrypted = encrypt_connection_string(original)
    assert encrypted  # Not empty — carries IV + encrypted empty string
    decrypted = decrypt_connection_string(encrypted)
    assert decrypted == original


def test_roundtrip_unicode(master_key: str) -> None:
    """Unicode characters (CJK, emoji) survive round-trip."""
    original = "host=数据库; password=🔐secret🔑; db=测试"
    encrypted = encrypt_connection_string(original)
    decrypted = decrypt_connection_string(encrypted)
    assert decrypted == original


def test_roundtrip_long_string(master_key: str) -> None:
    """Multi-kilobyte string round-trips correctly."""
    original = "postgresql://user:" + ("x" * 4096) + "@host:5432/db"
    encrypted = encrypt_connection_string(original)
    decrypted = decrypt_connection_string(encrypted)
    assert decrypted == original


# ── IV randomness ─────────────────────────────────────────────────────────


def test_different_iv_each_call(master_key: str) -> None:
    """Same plaintext with same key produces different ciphertext each call."""
    plaintext = "postgresql://user:pass@host/db"
    encrypted1 = encrypt_connection_string(plaintext)
    encrypted2 = encrypt_connection_string(plaintext)
    assert encrypted1 != encrypted2, (
        "Consecutive encryptions must produce different output "
        "(random IV per call)"
    )


# ── Decryption error cases ───────────────────────────────────────────────


def test_wrong_key_on_decrypt(master_key: str) -> None:
    """Decrypting with a different key raises DecryptError."""
    original = "postgresql://user:pass@host/db"
    encrypted = encrypt_connection_string(original)

    # Set a different key
    wrong_key = generate_encryption_key()
    os.environ["HERMES_MASTER_ENCRYPTION_KEY"] = wrong_key
    import hermes_crypto

    hermes_crypto._cached_key = None

    with pytest.raises(DecryptError):
        decrypt_connection_string(encrypted)


def test_corrupted_base64(master_key: str) -> None:
    """Invalid base64 input raises DecryptError."""
    with pytest.raises(DecryptError):
        decrypt_connection_string("not valid base64!!!")


def test_truncated_data(master_key: str) -> None:
    """Input shorter than 16 bytes raises DecryptError."""
    with pytest.raises(DecryptError):
        decrypt_connection_string("AAA")  # decodes to < 16 bytes


def test_tampered_ciphertext(master_key: str) -> None:
    """A single bit-flip in the ciphertext raises DecryptError."""
    original = "postgresql://user:pass@host/db"
    encrypted = encrypt_connection_string(original)
    # Flip the last character
    tampered = encrypted[:-1] + (
        "A" if encrypted[-1] != "A" else "B"
    )
    with pytest.raises(DecryptError):
        decrypt_connection_string(tampered)


# ── Key validation ────────────────────────────────────────────────────────


def test_missing_env_var() -> None:
    """Unset HERMES_MASTER_ENCRYPTION_KEY raises InvalidKeyError."""
    import hermes_crypto

    hermes_crypto._cached_key = None

    # Remove the env var set by the hermetic fixture (conftest unsets
    # credential env vars). We also explicitly pop it here.
    os.environ.pop("HERMES_MASTER_ENCRYPTION_KEY", None)

    with pytest.raises(InvalidKeyError):
        encrypt_connection_string("test")


def test_empty_env_var(master_key: str) -> None:
    """Empty HERMES_MASTER_ENCRYPTION_KEY raises InvalidKeyError."""
    os.environ["HERMES_MASTER_ENCRYPTION_KEY"] = ""
    import hermes_crypto

    hermes_crypto._cached_key = None

    with pytest.raises(InvalidKeyError):
        encrypt_connection_string("test")


def test_invalid_base64_key(master_key: str) -> None:
    """Non-base64 master key raises InvalidKeyError."""
    os.environ["HERMES_MASTER_ENCRYPTION_KEY"] = "not base64!!!"
    import hermes_crypto

    hermes_crypto._cached_key = None

    with pytest.raises(InvalidKeyError):
        encrypt_connection_string("test")


def test_wrong_key_length(master_key: str) -> None:
    """A 16-byte key (instead of 32) raises InvalidKeyError."""
    short_key = base64.b64encode(os.urandom(16)).decode("ascii")
    os.environ["HERMES_MASTER_ENCRYPTION_KEY"] = short_key
    import hermes_crypto

    hermes_crypto._cached_key = None

    with pytest.raises(InvalidKeyError):
        encrypt_connection_string("test")


# ── Key generation ────────────────────────────────────────────────────────


def test_generate_encryption_key_output() -> None:
    """Output is a 44-character base64 string that decodes to 32 bytes."""
    key = generate_encryption_key()
    assert isinstance(key, str)
    assert len(key) == 44
    decoded = base64.b64decode(key)
    assert len(decoded) == 32


def test_generated_key_usable() -> None:
    """A freshly generated key works for a real round-trip."""
    key = generate_encryption_key()
    os.environ["HERMES_MASTER_ENCRYPTION_KEY"] = key
    import hermes_crypto

    hermes_crypto._cached_key = None

    original = "mysql://admin:s3cret@10.0.1.5:3306/erp_prod"
    encrypted = encrypt_connection_string(original)
    decrypted = decrypt_connection_string(encrypted)
    assert decrypted == original


# ── Key caching ──────────────────────────────────────────────────────────


def test_key_is_cached_after_first_use(master_key: str) -> None:
    """After first encrypt, the key is cached — changing the env var
    does not affect subsequent calls."""
    # First call loads and caches the key
    original = "postgresql://a:b@h/d"
    encrypted1 = encrypt_connection_string(original)
    assert decrypt_connection_string(encrypted1) == original

    # Change the env var to an invalid value — cache should still be used
    os.environ["HERMES_MASTER_ENCRYPTION_KEY"] = "not base64!!!"
    # decrypt still works because the cached key is used
    assert decrypt_connection_string(encrypted1) == original

    # encrypt also still works
    encrypted2 = encrypt_connection_string("host=other")
    assert decrypt_connection_string(encrypted2) == "host=other"


# ── Cryptography not installed ────────────────────────────────────────────


def test_crypto_not_installed(monkeypatch) -> None:
    """When Cipher is None, operations raise CryptoError."""
    import hermes_crypto

    original_cipher = hermes_crypto.Cipher
    try:
        monkeypatch.setattr(hermes_crypto, "Cipher", None)
        hermes_crypto._cached_key = None
        os.environ["HERMES_MASTER_ENCRYPTION_KEY"] = generate_encryption_key()

        with pytest.raises(CryptoError):
            encrypt_connection_string("test")
    finally:
        monkeypatch.setattr(hermes_crypto, "Cipher", original_cipher)
        hermes_crypto._cached_key = None


# ── Exception hierarchy ───────────────────────────────────────────────────


def test_exception_hierarchy() -> None:
    """Verify the exception hierarchy for correct except clause ordering."""
    assert issubclass(InvalidKeyError, CryptoError)
    assert issubclass(DecryptError, CryptoError)
    assert issubclass(EncryptError, CryptoError)
    assert issubclass(CryptoError, Exception)
