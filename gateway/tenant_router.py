"""Dynamic Multi-Tenant Connection Router.

Resolves ``channel_user_id`` → tenant → encrypted connection string →
decrypted DB connection, injecting tenant-specific business rules and
semantic mappings into the agent's context prompt.

**Feature-gated**: activated only when ``HERMES_CENTRAL_DB_URL`` is set.
When unset, the entire module is a no-op and the gateway runs in
single-user (SQLite-only) mode with zero overhead.

Usage (from gateway/run.py)::

    from gateway.tenant_router import (
        TenantRouter, TenantState, initialize_tenant_system,
        shutdown_tenant_system, is_tenant_mode_enabled,
    )

    if is_tenant_mode_enabled():
        await initialize_tenant_system()
        router = TenantRouter()
        ...
        ctx = await router.resolve_tenant(channel_user_id, platform)
        ...
        await router.close_tenant_connection()
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ── Lazy imports (tenant extra may not be installed) ─────────────────────

try:
    import asyncpg
except ImportError:  # pragma: no cover
    asyncpg = None  # type: ignore[assignment]

try:
    import aiomysql
except ImportError:  # pragma: no cover
    aiomysql = None  # type: ignore[assignment]


# ── Constants ─────────────────────────────────────────────────────────────

_CENTRAL_DB_URL_ENV: str = "HERMES_CENTRAL_DB_URL"

# Central DB pool configuration
_POOL_MIN_SIZE: int = 2
_POOL_MAX_SIZE: int = 10


# ── Exceptions ────────────────────────────────────────────────────────────


class TenantRouterError(Exception):
    """Base exception for tenant routing failures."""


class CentralDBUnavailableError(TenantRouterError):
    """Raised when the central RDS database is unreachable."""


class TenantConnectionError(TenantRouterError):
    """Raised when connecting to a tenant database fails."""


# ── Data Classes ──────────────────────────────────────────────────────────


@dataclass
class TenantContext:
    """Resolved tenant information, held in memory only for one request.

    The ``connection_string`` field carries decrypted credentials and MUST
    be wiped from memory after use.
    """

    tenant_id: str
    business_name: str
    connection_string: str  # plaintext — ephemeral, wiped after use
    semantic_mapping: dict = field(default_factory=dict)
    business_rules: str = ""
    db_type: str = "postgresql"


@dataclass
class TenantState:
    """Mutable state for one message-turn tenant lifecycle."""

    ctx: Optional[TenantContext] = None
    _connection: Any = None  # asyncpg.Connection or aiomysql.Connection
    _closed: bool = False


# ── Connection Protocol ───────────────────────────────────────────────────


class _TenantConnection:
    """Wrapper around an async database connection with a uniform interface.

    Abstracts over asyncpg (PostgreSQL) and aiomysql (MySQL) so the router
    can query tenant databases without knowing the engine type.
    """

    def __init__(self, conn: Any, db_type: str) -> None:
        self._conn = conn
        self.db_type = db_type

    async def fetch_all(self, query: str, *args: Any) -> List[Dict[str, Any]]:
        """Execute a query and return all rows as dicts."""
        if self.db_type == "postgresql":
            records = await self._conn.fetch(query, *args)
            return [dict(r) for r in records]
        elif self.db_type == "mysql":
            async with self._conn.cursor() as cur:
                await cur.execute(query, args)
                rows = await cur.fetchall()
                if rows:
                    col_names = [desc[0] for desc in cur.description]
                    return [dict(zip(col_names, row)) for row in rows]
                return []
        else:
            raise TenantConnectionError(
                f"Unsupported database type: {self.db_type}"
            )

    async def close(self) -> None:
        """Close the underlying connection."""
        try:
            await self._conn.close()
        except Exception:
            logger.debug("Error closing tenant connection", exc_info=True)


# ── Central DB Pool ───────────────────────────────────────────────────────


class CentralDBPool:
    """Singleton asyncpg connection pool for the central AWS RDS database.

    Used to resolve ``channel_user_id → tenant`` lookups. Initialized once
    at gateway startup and drained at shutdown.
    """

    _pool: Optional[asyncpg.Pool] = None
    _dsn: Optional[str] = None

    @classmethod
    async def initialize(cls, dsn: str) -> None:
        """Create the central DB pool. Idempotent — second call is a no-op."""
        if cls._pool is not None:
            return

        if asyncpg is None:
            raise CentralDBUnavailableError(
                "asyncpg is not installed. Install the 'tenant' extra: "
                "uv pip install hermes-agent[tenant]"
            )

        cls._dsn = dsn
        try:
            cls._pool = await asyncpg.create_pool(
                dsn=dsn,
                min_size=_POOL_MIN_SIZE,
                max_size=_POOL_MAX_SIZE,
            )
            logger.info(
                "Central DB pool initialized (min=%d, max=%d)",
                _POOL_MIN_SIZE,
                _POOL_MAX_SIZE,
            )
        except Exception as exc:
            logger.critical(
                "Failed to connect to central database at %s: %s",
                _redact_dsn(dsn),
                exc,
            )
            cls._pool = None
            raise CentralDBUnavailableError(
                f"Central database unreachable: {exc}"
            ) from exc

    @classmethod
    async def close(cls) -> None:
        """Drain and close the central DB pool. Idempotent."""
        if cls._pool is not None:
            await cls._pool.close()
            cls._pool = None
            logger.info("Central DB pool closed")

    @classmethod
    def is_initialized(cls) -> bool:
        """Return True if the pool is ready for queries."""
        return cls._pool is not None

    @classmethod
    async def fetchrow(
        cls, query: str, *args: Any
    ) -> Optional[asyncpg.Record]:
        """Execute a single-row query and return the result or None."""
        if cls._pool is None:
            raise CentralDBUnavailableError("Central DB pool not initialized")
        async with cls._pool.acquire() as conn:
            return await conn.fetchrow(query, *args)


# ── Tenant Router ─────────────────────────────────────────────────────────


class TenantRouter:
    """Per-message tenant resolver with connection lifecycle.

    Resolves a ``channel_user_id`` to a tenant context by querying the
    central RDS database, decrypts the connection string, opens a
    temporary connection to the tenant database, and provides methods for
    querying tenant chat history and building tenant-aware prompt blocks.
    """

    def __init__(self) -> None:
        self._tenant_conn: Optional[_TenantConnection] = None
        self._tenant_ctx: Optional[TenantContext] = None

    # ── Tenant Resolution ─────────────────────────────────────────────

    async def resolve_tenant(
        self, channel_user_id: str, platform: str
    ) -> Optional[TenantContext]:
        """Resolve a channel user to a tenant context.

        Queries the central RDS database joining ``channel_links``,
        ``tenants``, and ``connections`` tables.

        Returns None when no link exists for the given channel user
        (graceful fallback to single-user mode).
        """
        if not channel_user_id:
            logger.debug("resolve_tenant: empty channel_user_id, skipping")
            return None

        if not CentralDBPool.is_initialized():
            logger.error("Central DB pool not initialized")
            return None

        query = """
            SELECT
                t.tenant_id,
                t.business_name,
                t.semantic_mapping,
                t.business_rules,
                c.encrypted_connection_string,
                c.db_engine
            FROM channel_links cl
            JOIN tenants t ON cl.tenant_id = t.tenant_id
            JOIN connections c ON t.tenant_id = c.tenant_id
            WHERE cl.channel_user_id = $1
              AND cl.platform = $2
            LIMIT 1
        """

        try:
            row = await CentralDBPool.fetchrow(query, channel_user_id, platform)
        except Exception as exc:
            logger.error(
                "Central DB query failed for channel_user_id=%s platform=%s: %s",
                channel_user_id,
                platform,
                exc,
            )
            return None

        if row is None:
            logger.info(
                "No tenant link found for channel_user_id=%s platform=%s",
                channel_user_id,
                platform,
            )
            return None

        # Decrypt the connection string
        encrypted_conn = row.get("encrypted_connection_string") or ""
        if not encrypted_conn:
            logger.error(
                "Empty connection_string for channel_user_id=%s tenant=%s",
                channel_user_id,
                row.get("tenant_id"),
            )
            return None

        try:
            from hermes_crypto import decrypt_connection_string

            plaintext = decrypt_connection_string(encrypted_conn)
        except Exception as exc:
            logger.error(
                "Failed to decrypt connection_string for channel_user_id=%s: %s",
                channel_user_id,
                exc,
            )
            return None

        # Parse semantic_mapping from JSONB
        semantic_mapping_raw = row.get("semantic_mapping")
        if isinstance(semantic_mapping_raw, str):
            try:
                semantic_mapping = json.loads(semantic_mapping_raw)
            except json.JSONDecodeError:
                logger.warning(
                    "Invalid JSON in semantic_mapping for tenant=%s",
                    row.get("tenant_id"),
                )
                semantic_mapping = {}
        elif isinstance(semantic_mapping_raw, dict):
            semantic_mapping = semantic_mapping_raw
        else:
            semantic_mapping = {}

        ctx = TenantContext(
            tenant_id=str(row["tenant_id"]),
            business_name=str(row.get("business_name") or ""),
            connection_string=plaintext,
            semantic_mapping=semantic_mapping,
            business_rules=str(row.get("business_rules") or ""),
            db_type=str(row.get("db_engine") or "postgresql"),
        )

        logger.info(
            "Tenant resolved: channel_user_id=%s → tenant=%s (%s)",
            channel_user_id,
            ctx.tenant_id,
            ctx.business_name,
        )
        return ctx

    # ── Tenant Connection ─────────────────────────────────────────────

    async def open_tenant_connection(self, ctx: TenantContext) -> None:
        """Open a temporary connection to the tenant database.

        Uses the appropriate async driver based on ``ctx.db_type``.
        """
        self._tenant_ctx = ctx

        if ctx.db_type == "postgresql":
            await self._open_postgres_connection(ctx)
        elif ctx.db_type in ("mysql", "mariadb"):
            await self._open_mysql_connection(ctx)
        else:
            logger.warning(
                "Unsupported tenant db_type=%s for tenant=%s — "
                "falling back to PostgreSQL",
                ctx.db_type,
                ctx.tenant_id,
            )
            await self._open_postgres_connection(ctx)

    async def _open_postgres_connection(self, ctx: TenantContext) -> None:
        """Open an asyncpg connection to a PostgreSQL tenant database."""
        if asyncpg is None:
            raise TenantConnectionError(
                "asyncpg is not installed. Install the 'tenant' extra."
            )
        try:
            conn = await asyncpg.connect(ctx.connection_string)
            self._tenant_conn = _TenantConnection(conn, "postgresql")
            logger.debug(
                "Opened PostgreSQL connection to tenant=%s", ctx.tenant_id
            )
        except Exception as exc:
            logger.error(
                "Failed to connect to tenant DB for tenant=%s: %s",
                ctx.tenant_id,
                exc,
            )
            raise TenantConnectionError(
                f"Tenant database unreachable: {exc}"
            ) from exc

    async def _open_mysql_connection(self, ctx: TenantContext) -> None:
        """Open an aiomysql connection to a MySQL tenant database."""
        if aiomysql is None:
            raise TenantConnectionError(
                "aiomysql is not installed. Install the 'tenant' extra."
            )
        try:
            parsed = urlparse(ctx.connection_string)
            conn = await aiomysql.connect(
                host=parsed.hostname or "localhost",
                port=parsed.port or 3306,
                user=parsed.username or "",
                password=parsed.password or "",
                db=(parsed.path or "/").lstrip("/") or "",
                charset="utf8mb4",
                autocommit=True,
            )
            self._tenant_conn = _TenantConnection(conn, "mysql")
            logger.debug(
                "Opened MySQL connection to tenant=%s", ctx.tenant_id
            )
        except Exception as exc:
            logger.error(
                "Failed to connect to tenant MySQL DB for tenant=%s: %s",
                ctx.tenant_id,
                exc,
            )
            raise TenantConnectionError(
                f"Tenant MySQL database unreachable: {exc}"
            ) from exc

    async def close_tenant_connection(self) -> None:
        """Close the tenant database connection and wipe credentials.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._tenant_conn is not None:
            await self._tenant_conn.close()
            self._tenant_conn = None

        if self._tenant_ctx is not None:
            # Wipe the decrypted connection string from memory
            self._tenant_ctx.connection_string = ""
            self._tenant_ctx = None

    # ── Chat History ──────────────────────────────────────────────────

    async def query_chat_history(
        self,
        ctx: TenantContext,
        channel_user_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Query chat history from the tenant database.

        Returns rows from the ``chat_history`` table filtered by
        ``tenant_id`` and ``channel_user_id``.
        """
        if self._tenant_conn is None:
            logger.debug("No tenant connection open; skipping chat history")
            return []

        query = """
            SELECT role, content, created_at
            FROM chat_history
            WHERE tenant_id = $1 AND channel_user_id = $2
            ORDER BY created_at DESC
            LIMIT $3
        """
        try:
            rows = await self._tenant_conn.fetch_all(
                query, ctx.tenant_id, channel_user_id, limit
            )
            logger.debug(
                "Retrieved %d chat history rows for tenant=%s user=%s",
                len(rows),
                ctx.tenant_id,
                channel_user_id,
            )
            return rows
        except Exception as exc:
            logger.error(
                "Failed to query chat_history for tenant=%s: %s",
                ctx.tenant_id,
                exc,
            )
            return []

    # ── System Prompt Block ───────────────────────────────────────────

    @staticmethod
    def build_tenant_system_prompt_block(ctx: TenantContext) -> str:
        """Build a Markdown block of tenant context for the system prompt.

        Injected into the ``context_prompt`` before it flows into the
        agent's system prompt.
        """
        parts: List[str] = []
        parts.append("## Tenant Configuration")
        parts.append("")
        parts.append(f"**Business:** {ctx.business_name}")

        if ctx.business_rules:
            parts.append("")
            parts.append("### Business Rules")
            parts.append("")
            parts.append(ctx.business_rules)

        if ctx.semantic_mapping:
            parts.append("")
            parts.append("### Semantic Mapping")
            parts.append("")
            try:
                mapping_json = json.dumps(
                    ctx.semantic_mapping, indent=2, ensure_ascii=False
                )
            except (TypeError, ValueError):
                mapping_json = str(ctx.semantic_mapping)
            parts.append("```json")
            parts.append(mapping_json)
            parts.append("```")

        return "\n".join(parts)


# ── Helpers ───────────────────────────────────────────────────────────────


def _convert_tenant_history_to_conversation(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert tenant chat_history rows to the agent's conversation format.

    Maps ``role`` + ``content`` columns to the standard message dict
    expected by ``AIAgent.run_conversation()``.
    """
    conversation: List[Dict[str, Any]] = []
    for row in reversed(rows):  # oldest first
        role = row.get("role", "user")
        content = row.get("content", "")
        # Only include roles the agent engine understands
        if role in ("user", "assistant", "system"):
            conversation.append({"role": role, "content": content})
    return conversation


def _redact_dsn(dsn: str) -> str:
    """Return a DSN safe for logging (password replaced)."""
    try:
        parsed = urlparse(dsn)
        if parsed.password:
            safe = parsed._replace(
                netloc=f"{parsed.username}:***@{parsed.hostname}:{parsed.port or ''}"
            )
            # Remove trailing colon if no port
            result = safe.geturl()
            if ":@***@:" in result:
                result = result.replace(":@***@:", "@***@")
            return result
        return dsn
    except Exception:
        return dsn


# ── Module-Level Lifecycle ────────────────────────────────────────────────


def is_tenant_mode_enabled() -> bool:
    """Return True if the tenant routing infrastructure should activate."""
    return bool(os.environ.get(_CENTRAL_DB_URL_ENV))


async def initialize_tenant_system() -> Optional[TenantRouter]:
    """Initialize the central DB pool and return a TenantRouter.

    Called once during gateway startup. Returns None when tenant mode is
    disabled (``HERMES_CENTRAL_DB_URL`` unset) or the central DB is
    unreachable.
    """
    if not is_tenant_mode_enabled():
        logger.debug(
            "%s not set — tenant routing disabled", _CENTRAL_DB_URL_ENV
        )
        return None

    dsn = os.environ[_CENTRAL_DB_URL_ENV]
    try:
        await CentralDBPool.initialize(dsn)
        return TenantRouter()
    except CentralDBUnavailableError:
        logger.critical(
            "Tenant routing unavailable — gateway will run in single-user mode"
        )
        return None


async def shutdown_tenant_system() -> None:
    """Close the central DB pool. Called during gateway shutdown."""
    await CentralDBPool.close()


__all__ = [
    "TenantContext",
    "TenantState",
    "TenantRouter",
    "TenantRouterError",
    "CentralDBUnavailableError",
    "TenantConnectionError",
    "CentralDBPool",
    "_convert_tenant_history_to_conversation",
    "is_tenant_mode_enabled",
    "initialize_tenant_system",
    "shutdown_tenant_system",
]
