"""Tests for hermes_cli/tenant_onboard — Semantic Tenant Onboarding."""

from __future__ import annotations

import json
import os
from unittest import mock

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_central_db_env():
    """Ensure HERMES_CENTRAL_DB_URL is unset by default."""
    old = os.environ.pop("HERMES_CENTRAL_DB_URL", None)
    yield
    if old is not None:
        os.environ["HERMES_CENTRAL_DB_URL"] = old


@pytest.fixture
def central_db_url() -> str:
    url = "postgresql://saas:pass@central-db:5432/hermes_central"
    os.environ["HERMES_CENTRAL_DB_URL"] = url
    return url


@pytest.fixture
def sample_schema() -> dict:
    """Return a sample SchemaInfo as a dict for prompt-building tests."""
    from hermes_cli.tenant_onboard import SchemaInfo

    return SchemaInfo(
        db_type="postgresql",
        tables=[
            {"table_name": "users"},
            {"table_name": "orders"},
        ],
        columns=[
            {
                "table_name": "users",
                "column_name": "id",
                "data_type": "integer",
                "is_nullable": "NO",
            },
            {
                "table_name": "users",
                "column_name": "email",
                "data_type": "varchar",
                "is_nullable": "YES",
            },
            {
                "table_name": "orders",
                "column_name": "id",
                "data_type": "integer",
                "is_nullable": "NO",
            },
            {
                "table_name": "orders",
                "column_name": "user_id",
                "data_type": "integer",
                "is_nullable": "YES",
            },
        ],
        foreign_keys=[
            {
                "table_name": "orders",
                "column_name": "user_id",
                "foreign_table_name": "users",
                "foreign_column_name": "id",
            },
        ],
    )


@pytest.fixture
def valid_llm_mapping() -> dict:
    """Return a valid semantic mapping dict."""
    return {
        "description": "E-commerce database with users and orders",
        "tables": [
            {
                "table_name": "users",
                "business_name": "Customers",
                "business_description": "Registered customer accounts",
            },
            {
                "table_name": "orders",
                "business_name": "Purchase Orders",
                "business_description": "Customer purchase transactions",
            },
        ],
        "foreign_keys": [
            {
                "table_name": "orders",
                "column_name": "user_id",
                "foreign_table_name": "users",
                "business_description": "Links order to customer",
            },
        ],
    }


# ── SchemaInfo ────────────────────────────────────────────────────────────


def test_schema_info_is_empty():
    """SchemaInfo with no tables reports is_empty=True."""
    from hermes_cli.tenant_onboard import SchemaInfo

    schema = SchemaInfo(db_type="postgresql")
    assert schema.is_empty is True


def test_schema_info_table_names():
    """SchemaInfo.table_names returns a list of table name strings."""
    from hermes_cli.tenant_onboard import SchemaInfo

    schema = SchemaInfo(
        db_type="postgresql",
        tables=[{"table_name": "a"}, {"table_name": "b"}],
    )
    assert schema.table_names == ["a", "b"]


# ── SchemaInspector ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_schema_inspector_extract_postgresql():
    """SchemaInspector.extract() returns expected SchemaInfo for PostgreSQL."""
    from hermes_cli.tenant_onboard import SchemaInspector

    fake_conn = mock.AsyncMock()

    # Mock fetch_all to return different results per call
    call_results = [
        [{"table_name": "users"}, {"table_name": "orders"}],
        [
            {
                "table_name": "users",
                "column_name": "id",
                "data_type": "integer",
                "is_nullable": "NO",
            },
            {
                "table_name": "users",
                "column_name": "email",
                "data_type": "varchar",
                "is_nullable": "YES",
            },
            {
                "table_name": "orders",
                "column_name": "id",
                "data_type": "integer",
                "is_nullable": "NO",
            },
        ],
        [
            {
                "table_name": "orders",
                "column_name": "user_id",
                "foreign_table_name": "users",
                "foreign_column_name": "id",
            },
        ],
    ]
    fake_conn.fetch_all = mock.AsyncMock(side_effect=call_results)

    inspector = SchemaInspector("postgresql://u:p@h/db", "postgresql")
    inspector._conn = fake_conn

    schema = await inspector.extract()

    assert schema.db_type == "postgresql"
    assert len(schema.tables) == 2
    assert schema.tables[0]["table_name"] == "users"
    assert len(schema.columns) == 3
    assert len(schema.foreign_keys) == 1


@pytest.mark.asyncio
async def test_schema_inspector_empty_database():
    """SchemaInspector returns empty SchemaInfo when DB has no tables."""
    from hermes_cli.tenant_onboard import SchemaInspector

    fake_conn = mock.AsyncMock()
    fake_conn.fetch_all = mock.AsyncMock(side_effect=[[], [], []])

    inspector = SchemaInspector("postgresql://u:p@h/db", "postgresql")
    inspector._conn = fake_conn

    schema = await inspector.extract()
    assert schema.is_empty is True
    assert len(schema.tables) == 0


@pytest.mark.asyncio
async def test_schema_inspector_no_foreign_keys():
    """Schema with no FKs returns empty foreign_keys list."""
    from hermes_cli.tenant_onboard import SchemaInspector

    fake_conn = mock.AsyncMock()
    fake_conn.fetch_all = mock.AsyncMock(
        side_effect=[
            [{"table_name": "logs"}],
            [
                {
                    "table_name": "logs",
                    "column_name": "msg",
                    "data_type": "text",
                    "is_nullable": "YES",
                },
            ],
            [],
        ]
    )

    inspector = SchemaInspector("postgresql://u:p@h/db", "postgresql")
    inspector._conn = fake_conn

    schema = await inspector.extract()
    assert len(schema.foreign_keys) == 0
    assert len(schema.tables) == 1


@pytest.mark.asyncio
async def test_schema_inspector_not_connected():
    """extract() raises RuntimeError if connect() was not called."""
    from hermes_cli.tenant_onboard import SchemaInspector

    inspector = SchemaInspector("postgresql://u:p@h/db")
    with pytest.raises(RuntimeError, match="not connected"):
        await inspector.extract()


# ── LLM Prompt Builder ────────────────────────────────────────────────────


def test_build_llm_prompt_includes_schema(sample_schema):
    """build_llm_prompt() produces messages containing table/column info."""
    from hermes_cli.tenant_onboard import build_llm_prompt

    messages = build_llm_prompt(sample_schema)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    # Check schema content in user message
    content = messages[1]["content"]
    assert "users" in content
    assert "orders" in content
    assert "email" in content
    assert "varchar" in content
    assert "user_id" in content
    assert "orders(user_id) → users(id)" not in content  # FK format differs
    # Check for FK mention
    assert "→" in content


def test_build_llm_prompt_empty_schema():
    """Empty schema produces a single system message indicating no tables."""
    from hermes_cli.tenant_onboard import SchemaInfo, build_llm_prompt

    schema = SchemaInfo(db_type="postgresql")
    messages = build_llm_prompt(schema)
    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    assert "no tables" in messages[0]["content"].lower()


def test_format_schema_for_prompt(sample_schema):
    """_format_schema_for_prompt produces readable text."""
    from hermes_cli.tenant_onboard import _format_schema_for_prompt

    text = _format_schema_for_prompt(sample_schema)
    assert "Table: users" in text
    assert "Table: orders" in text
    assert "id  integer  NOT NULL" in text
    assert "email  varchar  NULL" in text
    assert "user_id → users(id)" in text


# ── Semantic Mapping Validation ───────────────────────────────────────────


def test_validate_semantic_mapping_valid(valid_llm_mapping):
    """A complete mapping passes validation without exception."""
    from hermes_cli.tenant_onboard import validate_semantic_mapping

    validate_semantic_mapping(valid_llm_mapping)  # should not raise


def test_validate_semantic_mapping_not_dict():
    """Non-dict input raises LLMAnalysisError."""
    from hermes_cli.tenant_onboard import LLMAnalysisError, validate_semantic_mapping

    with pytest.raises(LLMAnalysisError, match="not a JSON object"):
        validate_semantic_mapping(["not", "a", "dict"])


def test_validate_semantic_mapping_missing_description():
    """Missing 'description' raises LLMAnalysisError."""
    from hermes_cli.tenant_onboard import LLMAnalysisError, validate_semantic_mapping

    with pytest.raises(LLMAnalysisError, match="missing 'description'"):
        validate_semantic_mapping({"tables": []})


def test_validate_semantic_mapping_missing_tables():
    """Missing 'tables' key raises LLMAnalysisError."""
    from hermes_cli.tenant_onboard import LLMAnalysisError, validate_semantic_mapping

    with pytest.raises(LLMAnalysisError, match="missing 'tables'"):
        validate_semantic_mapping({"description": "test"})


def test_validate_semantic_mapping_tables_not_list():
    """'tables' must be a list."""
    from hermes_cli.tenant_onboard import LLMAnalysisError, validate_semantic_mapping

    with pytest.raises(LLMAnalysisError, match="'tables' is not a list"):
        validate_semantic_mapping({"description": "test", "tables": "not_a_list"})


def test_validate_semantic_mapping_table_missing_business_name():
    """Table entry without business_name raises error."""
    from hermes_cli.tenant_onboard import LLMAnalysisError, validate_semantic_mapping

    with pytest.raises(LLMAnalysisError, match="missing 'business_name'"):
        validate_semantic_mapping({
            "description": "test",
            "tables": [
                {"table_name": "users", "business_description": "Users table"}
            ],
        })


def test_validate_semantic_mapping_foreign_keys_bad_type():
    """'foreign_keys' must be a list if present."""
    from hermes_cli.tenant_onboard import LLMAnalysisError, validate_semantic_mapping

    with pytest.raises(LLMAnalysisError, match="'foreign_keys' must be a list"):
        validate_semantic_mapping({
            "description": "test",
            "tables": [
                {"table_name": "users", "business_name": "Users", "business_description": "x"}
            ],
            "foreign_keys": "not_a_list",
        })


# ── Tenant Fetch from Central DB ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_tenant_from_central_not_found(central_db_url):
    """TenantNotFoundError when no row matches tenant_id."""
    from hermes_cli.tenant_onboard import TenantNotFoundError, _fetch_tenant_from_central

    with mock.patch("hermes_cli.tenant_onboard.asyncpg") as m:
        fake_conn = mock.AsyncMock()
        fake_conn.fetchrow = mock.AsyncMock(return_value=None)
        m.connect = mock.AsyncMock(return_value=fake_conn)

        with pytest.raises(TenantNotFoundError, match="No connection found"):
            await _fetch_tenant_from_central(central_db_url, "nonexistent-tenant")


@pytest.mark.asyncio
async def test_fetch_tenant_from_central_decrypt_fails(central_db_url):
    """Decryption failure surfaces as TenantOnboardError."""
    from hermes_cli.tenant_onboard import TenantOnboardError, _fetch_tenant_from_central

    with mock.patch("hermes_cli.tenant_onboard.asyncpg") as m:
        fake_conn = mock.AsyncMock()
        fake_conn.fetchrow = mock.AsyncMock(return_value={
            "encrypted_connection_string": "bad_encrypted_data",
            "db_engine": "postgresql",
        })
        m.connect = mock.AsyncMock(return_value=fake_conn)

        with mock.patch(
            "hermes_crypto.decrypt_connection_string",
            side_effect=Exception("decryption failed"),
        ):
            with pytest.raises(TenantOnboardError, match="Failed to decrypt"):
                await _fetch_tenant_from_central(central_db_url, "tenant-1")


@pytest.mark.asyncio
async def test_fetch_tenant_from_central_success(central_db_url):
    """Successful fetch returns TenantDbInfo with decrypted connection."""
    from hermes_cli.tenant_onboard import TenantDbInfo, _fetch_tenant_from_central

    with mock.patch("hermes_cli.tenant_onboard.asyncpg") as m:
        fake_conn = mock.AsyncMock()
        fake_conn.fetchrow = mock.AsyncMock(return_value={
            "encrypted_connection_string": "encrypted_val",
            "db_engine": "postgresql",
        })
        m.connect = mock.AsyncMock(return_value=fake_conn)

        with mock.patch(
            "hermes_crypto.decrypt_connection_string",
            return_value="postgresql://u:p@tenant-db:5432/db",
        ):
            result = await _fetch_tenant_from_central(central_db_url, "tenant-1")

    assert isinstance(result, TenantDbInfo)
    assert result.tenant_id == "tenant-1"
    assert result.connection_string == "postgresql://u:p@tenant-db:5432/db"
    assert result.db_type == "postgresql"


# ── Run Orchestrator ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_tenant_onboard_missing_env():
    """ConfigurationError when HERMES_CENTRAL_DB_URL is not set."""
    from hermes_cli.tenant_onboard import ConfigurationError, run_tenant_onboard

    assert os.environ.get("HERMES_CENTRAL_DB_URL") is None
    with pytest.raises(ConfigurationError, match="HERMES_CENTRAL_DB_URL"):
        await run_tenant_onboard("tenant-1")


@pytest.mark.asyncio
async def test_run_tenant_onboard_dry_run(central_db_url, valid_llm_mapping):
    """--dry-run inspects schema but does NOT call LLM or persist."""
    from hermes_cli.tenant_onboard import OnboardResult, run_tenant_onboard

    # Mock central DB → fetch tenant
    # Mock tenant DB → SchemaInspector
    with mock.patch("hermes_cli.tenant_onboard.asyncpg") as m:
        fake_conn = mock.AsyncMock()
        fake_conn.fetchrow = mock.AsyncMock(return_value={
            "connection_string": "enc_val",
            "db_type": "postgresql",
        })
        m.connect = mock.AsyncMock(return_value=fake_conn)

        with mock.patch(
            "hermes_crypto.decrypt_connection_string",
            return_value="postgresql://u:p@tenant:5432/db",
        ):
            with mock.patch(
                "hermes_cli.tenant_onboard.SchemaInspector.connect",
                new_callable=mock.AsyncMock,
            ):
                with mock.patch(
                    "hermes_cli.tenant_onboard.SchemaInspector.extract",
                    new_callable=mock.AsyncMock,
                ) as mock_extract:
                    from hermes_cli.tenant_onboard import SchemaInfo

                    mock_extract.return_value = SchemaInfo(
                        db_type="postgresql",
                        tables=[
                            {"table_name": "users"},
                            {"table_name": "orders"},
                        ],
                        columns=[
                            {"table_name": "users", "column_name": "id", "data_type": "integer", "is_nullable": "NO"},
                            {"table_name": "orders", "column_name": "id", "data_type": "integer", "is_nullable": "NO"},
                        ],
                        foreign_keys=[],
                    )

                    result = await run_tenant_onboard(
                        "tenant-1", dry_run=True,
                    )

    assert isinstance(result, OnboardResult)
    assert result.table_count == 2
    assert result.column_count == 2
    assert result.inserted is False
    assert result.semantic_mapping is None


@pytest.mark.asyncio
async def test_run_tenant_onboard_full_flow(central_db_url, valid_llm_mapping):
    """End-to-end flow with mocked central DB, tenant DB, and LLM."""
    from hermes_cli.tenant_onboard import OnboardResult, run_tenant_onboard

    with mock.patch("hermes_cli.tenant_onboard.asyncpg") as m:
        # Central DB mock
        fake_central = mock.AsyncMock()
        fake_central.fetchrow = mock.AsyncMock(return_value={
            "connection_string": "enc_val",
            "db_type": "postgresql",
        })
        # Second connect for persistence
        fake_persist = mock.AsyncMock()
        fake_persist.execute = mock.AsyncMock()

        m.connect = mock.AsyncMock(side_effect=[fake_central, fake_persist])

        with mock.patch(
            "hermes_crypto.decrypt_connection_string",
            return_value="postgresql://u:p@tenant:5432/db",
        ):
            with mock.patch(
                "hermes_cli.tenant_onboard.SchemaInspector.connect",
                new_callable=mock.AsyncMock,
            ):
                with mock.patch(
                    "hermes_cli.tenant_onboard.SchemaInspector.extract",
                    new_callable=mock.AsyncMock,
                ) as mock_extract:
                    from hermes_cli.tenant_onboard import SchemaInfo

                    mock_extract.return_value = SchemaInfo(
                        db_type="postgresql",
                        tables=[{"table_name": "users"}],
                        columns=[{"table_name": "users", "column_name": "id", "data_type": "integer", "is_nullable": "NO"}],
                        foreign_keys=[],
                    )

                    with mock.patch(
                        "agent.auxiliary_client.call_llm",
                    ) as mock_llm:
                        fake_response = mock.MagicMock()
                        fake_response.choices = [
                            mock.MagicMock()
                        ]
                        fake_response.choices[0].message.content = json.dumps(
                            valid_llm_mapping
                        )
                        mock_llm.return_value = fake_response

                        result = await run_tenant_onboard("tenant-1")

    assert isinstance(result, OnboardResult)
    assert result.table_count == 1
    assert result.inserted is True
    assert result.semantic_mapping is not None
    assert result.semantic_mapping["description"] == valid_llm_mapping["description"]
    mock_llm.assert_called_once()
    fake_persist.execute.assert_awaited_once()


# ── Persistence ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_persist_semantic_mapping(central_db_url, valid_llm_mapping):
    """persist_semantic_mapping executes INSERT and returns True."""
    from hermes_cli.tenant_onboard import persist_semantic_mapping

    with mock.patch("hermes_cli.tenant_onboard.asyncpg") as m:
        fake_conn = mock.AsyncMock()
        fake_conn.execute = mock.AsyncMock()
        m.connect = mock.AsyncMock(return_value=fake_conn)

        result = await persist_semantic_mapping(
            central_db_url, "tenant-1", valid_llm_mapping
        )

    assert result is True
    fake_conn.execute.assert_awaited_once()


# ── Exception Hierarchy ───────────────────────────────────────────────────


def test_exception_hierarchy():
    """All onboarding errors subclass TenantOnboardError."""
    from hermes_cli.tenant_onboard import (
        ConfigurationError,
        EmptySchemaError,
        LLMAnalysisError,
        PersistenceError,
        TenantNotFoundError,
        TenantOnboardError,
    )

    assert issubclass(ConfigurationError, TenantOnboardError)
    assert issubclass(TenantNotFoundError, TenantOnboardError)
    assert issubclass(EmptySchemaError, TenantOnboardError)
    assert issubclass(LLMAnalysisError, TenantOnboardError)
    assert issubclass(PersistenceError, TenantOnboardError)
    assert issubclass(TenantOnboardError, Exception)
