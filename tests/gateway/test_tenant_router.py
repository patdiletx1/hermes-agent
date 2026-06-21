"""Tests for gateway/tenant_router.py — Dynamic Multi-Tenant Connection Router."""

from __future__ import annotations

import json
import os
from unittest import mock

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_tenant_env():
    """Ensure HERMES_CENTRAL_DB_URL is unset by default."""
    old = os.environ.pop("HERMES_CENTRAL_DB_URL", None)
    yield
    if old is not None:
        os.environ["HERMES_CENTRAL_DB_URL"] = old


@pytest.fixture
def central_db_url() -> str:
    """Set a test central DB URL and return it."""
    url = "postgresql://user:pass@test-host:5432/hermes_central"
    os.environ["HERMES_CENTRAL_DB_URL"] = url
    return url


@pytest.fixture
def mock_asyncpg_pool():
    """Mock asyncpg.create_pool to return a fake pool."""
    with mock.patch("gateway.tenant_router.asyncpg") as m:
        # Pool is an AsyncMock (for async close(), etc.) but acquire()
        # is sync in asyncpg — returns a context manager directly.
        fake_pool = mock.AsyncMock()

        fake_conn = mock.AsyncMock()
        fake_conn.fetchrow = mock.AsyncMock()

        # async with pool.acquire() as conn:
        _acquire_cm = mock.AsyncMock()
        _acquire_cm.__aenter__.return_value = fake_conn

        # acquire() must return synchronously, not a coroutine
        fake_pool.acquire = mock.MagicMock(return_value=_acquire_cm)

        m.create_pool = mock.AsyncMock(return_value=fake_pool)
        m.connect = mock.AsyncMock()
        yield m


@pytest.fixture
def tenant_router(mock_asyncpg_pool, central_db_url):
    """Create an initialized TenantRouter with mocked central DB pool."""
    from gateway.tenant_router import CentralDBPool, TenantRouter

    # Reset singleton state
    CentralDBPool._pool = None
    return TenantRouter()


# ── TenantContext ─────────────────────────────────────────────────────────


def test_tenant_context_defaults():
    """TenantContext has correct defaults."""
    from gateway.tenant_router import TenantContext

    ctx = TenantContext(
        tenant_id="t1",
        business_name="Test Corp",
        connection_string="postgresql://a:b@h/d",
    )
    assert ctx.tenant_id == "t1"
    assert ctx.business_name == "Test Corp"
    assert ctx.semantic_mapping == {}
    assert ctx.business_rules == ""
    assert ctx.db_type == "postgresql"


# ── is_tenant_mode_enabled ────────────────────────────────────────────────


def test_tenant_mode_disabled_when_env_unset():
    """Without HERMES_CENTRAL_DB_URL, tenant mode is disabled."""
    from gateway.tenant_router import is_tenant_mode_enabled

    assert os.environ.get("HERMES_CENTRAL_DB_URL") is None
    assert is_tenant_mode_enabled() is False


def test_tenant_mode_enabled_when_env_set(central_db_url):
    """With HERMES_CENTRAL_DB_URL set, tenant mode is enabled."""
    from gateway.tenant_router import is_tenant_mode_enabled

    assert is_tenant_mode_enabled() is True


# ── Initialize / Shutdown ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_initialize_returns_none_when_disabled():
    """When the env var is unset, initialize returns None."""
    from gateway.tenant_router import initialize_tenant_system

    result = await initialize_tenant_system()
    assert result is None


@pytest.mark.asyncio
async def test_initialize_creates_router_when_enabled(mock_asyncpg_pool, central_db_url):
    """When HERMES_CENTRAL_DB_URL is set, initialize returns a TenantRouter."""
    from gateway.tenant_router import CentralDBPool, initialize_tenant_system

    CentralDBPool._pool = None
    result = await initialize_tenant_system()
    assert result is not None
    mock_asyncpg_pool.create_pool.assert_awaited_once()


@pytest.mark.asyncio
async def test_initialize_is_idempotent(mock_asyncpg_pool, central_db_url):
    """Second call to initialize is a no-op."""
    from gateway.tenant_router import CentralDBPool, initialize_tenant_system

    CentralDBPool._pool = None
    r1 = await initialize_tenant_system()
    r2 = await initialize_tenant_system()
    # create_pool called exactly once
    assert mock_asyncpg_pool.create_pool.await_count == 1


@pytest.mark.asyncio
async def test_shutdown_closes_pool(mock_asyncpg_pool, central_db_url):
    """shutdown_tenant_system closes the pool."""
    from gateway.tenant_router import CentralDBPool, shutdown_tenant_system

    CentralDBPool._pool = None
    await CentralDBPool.initialize(central_db_url)
    await shutdown_tenant_system()
    fake_pool = mock_asyncpg_pool.create_pool.return_value
    fake_pool.close.assert_awaited_once()


# ── Tenant Resolution ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resolve_tenant_not_found(tenant_router, mock_asyncpg_pool):
    """When no row is returned, resolve_tenant returns None."""
    # Get the fake connection through the async context manager
    fake_pool = mock_asyncpg_pool.create_pool.return_value
    fake_conn = fake_pool.acquire.return_value.__aenter__.return_value
    fake_conn.fetchrow.return_value = None

    result = await tenant_router.resolve_tenant("unknown_user", "telegram")
    assert result is None


@pytest.mark.asyncio
async def test_resolve_tenant_found(tenant_router, mock_asyncpg_pool):
    """When a row is returned, resolve_tenant builds a TenantContext."""
    from gateway.tenant_router import CentralDBPool

    # Initialize pool first
    await CentralDBPool.initialize("postgresql://u:p@h/db")

    fake_pool = mock_asyncpg_pool.create_pool.return_value
    fake_conn = fake_pool.acquire.return_value.__aenter__.return_value
    fake_conn.fetchrow.return_value = {
        "tenant_id": "tenant-001",
        "business_name": "Acme Corp",
        "semantic_mapping": json.dumps({"status": {"1": "active"}}),
        "business_rules": "Always use formal language.",
        "encrypted_connection_string": "encrypted_str",
        "db_engine": "postgresql",
    }

    with mock.patch(
        "hermes_crypto.decrypt_connection_string",
        return_value="postgresql://acme:secret@tenant-db:5432/acme_prod",
    ):
        result = await tenant_router.resolve_tenant("user_42", "telegram")

    assert result is not None
    assert result.tenant_id == "tenant-001"
    assert result.business_name == "Acme Corp"
    assert result.business_rules == "Always use formal language."
    assert result.db_type == "postgresql"
    assert result.connection_string == "postgresql://acme:secret@tenant-db:5432/acme_prod"
    assert result.semantic_mapping == {"status": {"1": "active"}}


@pytest.mark.asyncio
async def test_resolve_tenant_empty_channel_user_id(tenant_router):
    """Empty channel_user_id returns None early."""
    result = await tenant_router.resolve_tenant("", "telegram")
    assert result is None


@pytest.mark.asyncio
async def test_resolve_tenant_decrypt_fails(tenant_router, mock_asyncpg_pool):
    """When decrypt raises, resolve_tenant returns None gracefully."""
    from gateway.tenant_router import CentralDBPool

    await CentralDBPool.initialize("postgresql://u:p@h/db")

    fake_pool = mock_asyncpg_pool.create_pool.return_value
    fake_conn = fake_pool.acquire.return_value.__aenter__.return_value
    fake_conn.fetchrow.return_value = {
        "tenant_id": "t1",
        "business_name": "Bad Corp",
        "semantic_mapping": "{}",
        "business_rules": "",
        "connection_string": "bad_encrypted",
        "db_type": "postgresql",
    }

    with mock.patch(
        "hermes_crypto.decrypt_connection_string",
        side_effect=Exception("decryption failed"),
    ):
        result = await tenant_router.resolve_tenant("user_42", "telegram")

    assert result is None


@pytest.mark.asyncio
async def test_resolve_tenant_pool_not_initialized():
    """When pool is not initialized, resolve returns None with an error log."""
    from gateway.tenant_router import CentralDBPool, TenantRouter

    CentralDBPool._pool = None
    router = TenantRouter()
    result = await router.resolve_tenant("user_42", "telegram")
    assert result is None


# ── Tenant Connection Lifecycle ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_open_and_close_postgres_connection(tenant_router, mock_asyncpg_pool):
    """Open a PostgreSQL connection, then close it and wipe credentials."""
    from gateway.tenant_router import TenantContext

    ctx = TenantContext(
        tenant_id="t1",
        business_name="Test",
        connection_string="postgresql://u:p@localhost:5432/db",
    )

    await tenant_router.open_tenant_connection(ctx)
    assert tenant_router._tenant_conn is not None
    mock_asyncpg_pool.connect.assert_awaited_once()

    await tenant_router.close_tenant_connection()
    assert tenant_router._tenant_conn is None
    assert tenant_router._tenant_ctx is None
    assert ctx.connection_string == ""  # wiped


@pytest.mark.asyncio
async def test_close_connection_is_idempotent(tenant_router):
    """Calling close twice does not raise."""
    await tenant_router.close_tenant_connection()
    await tenant_router.close_tenant_connection()  # no-op, no error


# ── Build System Prompt Block ─────────────────────────────────────────────


def test_build_tenant_system_prompt_block():
    """Builds correct markdown block with business info."""
    from gateway.tenant_router import TenantContext, TenantRouter

    ctx = TenantContext(
        tenant_id="t1",
        business_name="Acme Corp",
        connection_string="unused",
        semantic_mapping={"priorities": {"1": "low", "2": "high"}},
        business_rules="Keep responses under 500 chars.",
    )

    block = TenantRouter.build_tenant_system_prompt_block(ctx)
    assert "## Tenant Configuration" in block
    assert "Acme Corp" in block
    assert "Keep responses under 500 chars." in block
    assert "priorities" in block
    assert '"1": "low"' in block


def test_build_tenant_system_prompt_block_no_rules():
    """When no business_rules, the rules section is omitted."""
    from gateway.tenant_router import TenantContext, TenantRouter

    ctx = TenantContext(
        tenant_id="t1",
        business_name="Minimal",
        connection_string="unused",
    )

    block = TenantRouter.build_tenant_system_prompt_block(ctx)
    assert "## Tenant Configuration" in block
    assert "Minimal" in block
    assert "Business Rules" not in block
    assert "Semantic Mapping" not in block


# ── Chat History Conversion ───────────────────────────────────────────────


def test_convert_tenant_history_empty():
    """Empty list returns empty list."""
    from gateway.tenant_router import _convert_tenant_history_to_conversation

    result = _convert_tenant_history_to_conversation([])
    assert result == []


def test_convert_tenant_history_reverses_order():
    """Rows are returned oldest-first (reversed from DESC order)."""
    from gateway.tenant_router import _convert_tenant_history_to_conversation

    rows = [
        {"role": "assistant", "content": "reply", "created_at": "2026-06-20T12:00:00Z"},
        {"role": "user", "content": "question", "created_at": "2026-06-20T11:00:00Z"},
    ]
    result = _convert_tenant_history_to_conversation(rows)
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "question"
    assert result[1]["role"] == "assistant"
    assert result[1]["content"] == "reply"


def test_convert_tenant_history_filters_unknown_roles():
    """Roles other than user/assistant/system are filtered out."""
    from gateway.tenant_router import _convert_tenant_history_to_conversation

    rows = [
        {"role": "tool", "content": "tool output"},
        {"role": "user", "content": "hello"},
    ]
    result = _convert_tenant_history_to_conversation(rows)
    assert len(result) == 1
    assert result[0]["role"] == "user"


# ── SessionSource channel_user_id ─────────────────────────────────────────


def test_session_source_channel_user_id_default():
    """channel_user_id defaults to None."""
    from gateway.session import Platform, SessionSource

    s = SessionSource(platform=Platform.TELEGRAM, chat_id="123")
    assert s.channel_user_id is None


def test_session_source_channel_user_id_roundtrip():
    """channel_user_id survives to_dict/from_dict roundtrip."""
    from gateway.session import Platform, SessionSource

    s = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        channel_user_id="user_abc",
    )
    d = s.to_dict()
    assert d["channel_user_id"] == "user_abc"

    s2 = SessionSource.from_dict(d)
    assert s2.channel_user_id == "user_abc"


def test_session_source_channel_user_id_none_omitted():
    """When None, channel_user_id is omitted from to_dict."""
    from gateway.session import Platform, SessionSource

    s = SessionSource(platform=Platform.TELEGRAM, chat_id="123")
    d = s.to_dict()
    assert "channel_user_id" not in d


# ── _redact_dsn ───────────────────────────────────────────────────────────


def test_redact_dsn_with_password():
    """Password is replaced with ***."""
    from gateway.tenant_router import _redact_dsn

    dsn = "postgresql://admin:s3cret@db.example.com:5432/mydb"
    redacted = _redact_dsn(dsn)
    assert "s3cret" not in redacted
    assert "***" in redacted
    assert "admin" in redacted
    assert "db.example.com" in redacted


def test_redact_dsn_without_password():
    """DSN without password is returned as-is."""
    from gateway.tenant_router import _redact_dsn

    dsn = "postgresql://localhost:5432/mydb"
    assert _redact_dsn(dsn) == dsn


# ── TenantState ───────────────────────────────────────────────────────────


def test_tenant_state_defaults():
    """TenantState is initially empty."""
    from gateway.tenant_router import TenantState

    ts = TenantState()
    assert ts.ctx is None
    assert ts._closed is False


# ── Error hierarchy ───────────────────────────────────────────────────────


def test_tenant_errors_are_exceptions():
    """All tenant errors subclass Exception."""
    from gateway.tenant_router import (
        CentralDBUnavailableError,
        TenantConnectionError,
        TenantRouterError,
    )

    assert issubclass(TenantRouterError, Exception)
    assert issubclass(CentralDBUnavailableError, TenantRouterError)
    assert issubclass(TenantConnectionError, TenantRouterError)
