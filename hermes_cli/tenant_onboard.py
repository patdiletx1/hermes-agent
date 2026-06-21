"""Semantic Tenant Onboarding — schema extraction + LLM analysis + persistence.

Provides the ``hermes tenant-onboard`` command workflow::

    hermes tenant-onboard --tenant-id <UUID>
    hermes tenant-onboard --tenant-id <UUID> --dry-run
    hermes tenant-onboard --tenant-id <UUID> --llm-model google/gemini-2.5-flash

Uses ``call_llm()`` from ``agent/auxiliary_client`` for one-shot structured
completions and ``asyncpg`` for database connectivity.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Lazy imports ─────────────────────────────────────────────────────────

try:
    import asyncpg
except ImportError:  # pragma: no cover
    asyncpg = None  # type: ignore[assignment]

try:
    import aiomysql
except ImportError:  # pragma: no cover
    aiomysql = None  # type: ignore[assignment]


# ── SQL query constants ──────────────────────────────────────────────────

PG_TABLES_QUERY = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
      AND table_type = 'BASE TABLE'
    ORDER BY table_name
"""

PG_COLUMNS_QUERY = """
    SELECT table_name, column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_schema = 'public'
    ORDER BY table_name, ordinal_position
"""

PG_FOREIGN_KEYS_QUERY = """
    SELECT
        tc.table_name,
        kcu.column_name,
        ccu.table_name AS foreign_table_name,
        ccu.column_name AS foreign_column_name
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu
        ON tc.constraint_name = kcu.constraint_name
        AND tc.table_schema = kcu.table_schema
    JOIN information_schema.constraint_column_usage ccu
        ON ccu.constraint_name = tc.constraint_name
    WHERE tc.constraint_type = 'FOREIGN KEY'
      AND tc.table_schema = 'public'
    ORDER BY tc.table_name, kcu.ordinal_position
"""

# MySQL variants — same information_schema, different schema filter
MYSQL_TABLES_QUERY = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = DATABASE()
      AND table_type = 'BASE TABLE'
    ORDER BY table_name
"""

MYSQL_COLUMNS_QUERY = """
    SELECT table_name, column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_schema = DATABASE()
    ORDER BY table_name, ordinal_position
"""

MYSQL_FOREIGN_KEYS_QUERY = """
    SELECT
        tc.table_name,
        kcu.column_name,
        ccu.table_name AS foreign_table_name,
        ccu.column_name AS foreign_column_name
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu
        ON tc.constraint_name = kcu.constraint_name
        AND tc.table_schema = kcu.table_schema
    JOIN information_schema.constraint_column_usage ccu
        ON ccu.constraint_name = tc.constraint_name
    WHERE tc.constraint_type = 'FOREIGN KEY'
      AND tc.table_schema = DATABASE()
    ORDER BY tc.table_name, kcu.ordinal_position
"""


# ── Exception hierarchy ──────────────────────────────────────────────────


class TenantOnboardError(Exception):
    """Base exception for tenant onboarding failures."""


class ConfigurationError(TenantOnboardError):
    """Missing or invalid configuration (e.g., env vars not set)."""


class TenantNotFoundError(TenantOnboardError):
    """Tenant ID not found in the central database."""


class EmptySchemaError(TenantOnboardError):
    """Tenant database has no tables to analyze."""


class LLMAnalysisError(TenantOnboardError):
    """LLM returned invalid or unparseable JSON."""


class PersistenceError(TenantOnboardError):
    """Failed to persist the semantic mapping to the central database."""


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class TenantDbInfo:
    """Resolved tenant connection details (ephemeral)."""

    tenant_id: str
    connection_string: str  # decrypted — wiped after use
    db_type: str = "postgresql"


@dataclass
class SchemaInfo:
    """Normalised schema metadata extracted from INFORMATION_SCHEMA."""

    db_type: str
    tables: list = field(default_factory=list)
    columns: list = field(default_factory=list)
    foreign_keys: list = field(default_factory=list)

    @property
    def table_names(self) -> List[str]:
        return [t["table_name"] for t in self.tables]

    @property
    def is_empty(self) -> bool:
        return len(self.tables) == 0


@dataclass
class OnboardResult:
    """Result returned to the CLI handler for display."""

    tenant_id: str
    table_count: int
    column_count: int
    semantic_mapping: Optional[dict] = None
    inserted: bool = False
    error: Optional[str] = None


# ── Connection wrapper ───────────────────────────────────────────────────


class _TenantConnection:
    """Lightweight async connection wrapper for INFORMATION_SCHEMA queries.

    Abstracts over asyncpg (PostgreSQL) and aiomysql (MySQL).
    Mirrors the pattern in ``gateway/tenant_router._TenantConnection``.
    """

    def __init__(self, conn: Any, db_type: str) -> None:
        self._conn = conn
        self.db_type = db_type

    async def fetch_all(self, query: str, *args: Any) -> list:
        """Execute *query* and return all rows as dicts."""
        if self.db_type == "postgresql":
            if asyncpg is None:
                raise TenantOnboardError("asyncpg is not installed")
            records = await self._conn.fetch(query, *args)
            return [dict(r) for r in records]
        elif self.db_type in ("mysql", "mariadb"):
            if aiomysql is None:
                raise TenantOnboardError("aiomysql is not installed")
            cursor = await self._conn.cursor()
            try:
                await cursor.execute(query, args)
                rows = await cursor.fetchall()
                if rows:
                    col_names = [desc[0] for desc in cursor.description]
                    return [dict(zip(col_names, row)) for row in rows]
                return []
            finally:
                await cursor.close()
        else:
            raise TenantOnboardError(f"Unsupported db_type: {self.db_type}")

    async def close(self) -> None:
        """Close the underlying connection."""
        try:
            await self._conn.close()
        except Exception:
            logger.debug("Error closing tenant connection", exc_info=True)


# ── Schema Inspector ─────────────────────────────────────────────────────


class SchemaInspector:
    """Connect to a tenant database and extract its schema metadata.

    Uses ``_TenantConnection`` to abstract over PostgreSQL and MySQL,
    running INFORMATION_SCHEMA queries to discover tables, columns,
    and foreign key relationships.
    """

    def __init__(self, conn_string: str, db_type: str = "postgresql") -> None:
        self._conn_string = conn_string
        self._db_type = db_type
        self._conn: Optional[_TenantConnection] = None

    # ── Connection lifecycle ──────────────────────────────────────────

    async def connect(self) -> None:
        """Open a connection to the tenant database."""
        if self._db_type == "postgresql":
            await self._connect_postgres()
        elif self._db_type in ("mysql", "mariadb"):
            await self._connect_mysql()
        else:
            raise TenantOnboardError(f"Unsupported db_type: {self._db_type}")

    async def _connect_postgres(self) -> None:
        if asyncpg is None:
            raise TenantOnboardError(
                "asyncpg is not installed. Install the 'tenant' extra: "
                "uv pip install hermes-agent[tenant]"
            )
        raw_conn = await asyncpg.connect(self._conn_string)
        self._conn = _TenantConnection(raw_conn, "postgresql")

    async def _connect_mysql(self) -> None:
        if aiomysql is None:
            raise TenantOnboardError(
                "aiomysql is not installed. Install it to connect to MySQL "
                "tenant databases."
            )
        from urllib.parse import urlparse

        parsed = urlparse(self._conn_string)
        raw_conn = await aiomysql.connect(
            host=parsed.hostname or "localhost",
            port=parsed.port or 3306,
            user=parsed.username or "",
            password=parsed.password or "",
            db=(parsed.path or "/").lstrip("/") or "",
            charset="utf8mb4",
            autocommit=True,
        )
        self._conn = _TenantConnection(raw_conn, "mysql")

    async def close(self) -> None:
        """Close the tenant database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    # ── Schema extraction ─────────────────────────────────────────────

    async def extract(self) -> SchemaInfo:
        """Run INFORMATION_SCHEMA queries and return schema metadata.

        Raises:
            RuntimeError: if ``connect()`` was not called first.
        """
        if self._conn is None:
            raise RuntimeError(
                "SchemaInspector not connected — call connect() first"
            )

        tables = await self._fetch_tables()
        columns = await self._fetch_columns()
        foreign_keys = await self._fetch_foreign_keys()

        return SchemaInfo(
            db_type=self._db_type,
            tables=tables,
            columns=columns,
            foreign_keys=foreign_keys,
        )

    async def _fetch_tables(self) -> list:
        query = (
            PG_TABLES_QUERY
            if self._db_type == "postgresql"
            else MYSQL_TABLES_QUERY
        )
        return await self._conn.fetch_all(query)

    async def _fetch_columns(self) -> list:
        query = (
            PG_COLUMNS_QUERY
            if self._db_type == "postgresql"
            else MYSQL_COLUMNS_QUERY
        )
        return await self._conn.fetch_all(query)

    async def _fetch_foreign_keys(self) -> list:
        query = (
            PG_FOREIGN_KEYS_QUERY
            if self._db_type == "postgresql"
            else MYSQL_FOREIGN_KEYS_QUERY
        )
        return await self._conn.fetch_all(query)


# ── Central DB helpers ───────────────────────────────────────────────────


async def _fetch_tenant_from_central(
    central_dsn: str, tenant_id: str
) -> TenantDbInfo:
    """Query the central RDS for a tenant's encrypted connection string.

    Returns a ``TenantDbInfo`` with the *decrypted* connection string.
    """
    if asyncpg is None:
        raise TenantOnboardError(
            "asyncpg is not installed. Install the 'tenant' extra."
        )

    conn = await asyncpg.connect(dsn=central_dsn)
    try:
        row = await conn.fetchrow(
            "SELECT encrypted_connection_string, db_engine "
            "FROM public.connections "
            "WHERE tenant_id = $1 "
            "LIMIT 1",
            tenant_id,
        )
    except Exception as exc:
        await conn.close()
        raise TenantOnboardError(
            f"Failed to query central database: {exc}"
        ) from exc

    if row is None:
        await conn.close()
        raise TenantNotFoundError(
            f"No connection found for tenant_id={tenant_id}"
        )

    encrypted = row["encrypted_connection_string"]
    if not encrypted:
        await conn.close()
        raise TenantNotFoundError(
            f"Empty encrypted_connection_string for tenant_id={tenant_id}"
        )

    db_type = str(row.get("db_engine") or "postgresql")

    try:
        from hermes_crypto import decrypt_connection_string

        plaintext = decrypt_connection_string(encrypted)
    except Exception as exc:
        await conn.close()
        raise TenantOnboardError(
            f"Failed to decrypt connection string for tenant {tenant_id}: {exc}"
        ) from exc

    await conn.close()
    return TenantDbInfo(
        tenant_id=tenant_id,
        connection_string=plaintext,
        db_type=db_type,
    )


# ── LLM analysis ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a database semantic analyst. Your task is to analyze a raw database \
schema and infer the business purpose of each table, column, and foreign key \
relationship.

Guidelines:
- Map technical names (e.g. "Fct_Vts_2026", "tbl_cust_orders") to business \
concepts ("Sales Facts 2026", "Customer Orders").
- Identify fact tables (transactions, sales, events) vs dimension tables \
(catalogs, lookup tables, master data).
- For each column in a table, provide a business-friendly name and a brief \
description of what it represents.
- For foreign keys, describe the business relationship between the two tables.
- Use the language of the schema (if column names are in Spanish, use Spanish \
business names; if English, use English).

Output ONLY valid JSON. No markdown, no code fences, no explanation — just the \
JSON object."""

_OUTPUT_SCHEMA_DESCRIPTION = """\
{
  "description": "<one-sentence summary of the database's business purpose>",
  "tables": [
    {
      "table_name": "<original name>",
      "business_name": "<human-readable business name>",
      "business_description": "<1-2 sentences explaining what this table stores>",
      "columns": [
        {
          "column_name": "<original name>",
          "business_name": "<human-readable business name>",
          "business_description": "<brief description of what this column represents>"
        }
      ]
    }
  ],
  "foreign_keys": [
    {
      "table_name": "<original name>",
      "column_name": "<original FK column>",
      "foreign_table_name": "<referenced table>",
      "business_description": "<what the relationship means in business terms>"
    }
  ]
}"""


def _format_schema_for_prompt(schema: SchemaInfo) -> str:
    """Render a SchemaInfo as a human-readable text block for the LLM."""
    lines: List[str] = []
    lines.append(f"Database type: {schema.db_type}")
    lines.append(f"Tables: {len(schema.tables)}")
    lines.append(f"Columns: {len(schema.columns)}")
    lines.append(f"Foreign keys: {len(schema.foreign_keys)}")
    lines.append("")
    lines.append("─" * 72)
    lines.append("")

    for table in schema.tables:
        tname = table["table_name"]
        lines.append(f"Table: {tname}")

        # Columns for this table
        tbl_cols = [
            c for c in schema.columns if c["table_name"] == tname
        ]
        for col in tbl_cols:
            nullable = "NULL" if col.get("is_nullable") == "YES" else "NOT NULL"
            lines.append(
                f"  {col['column_name']}  {col['data_type']}  {nullable}"
            )

        # Foreign keys for this table
        tbl_fks = [
            fk for fk in schema.foreign_keys if fk["table_name"] == tname
        ]
        for fk in tbl_fks:
            lines.append(
                f"  FK: {fk['column_name']} → "
                f"{fk['foreign_table_name']}({fk['foreign_column_name']})"
            )
        lines.append("")

    return "\n".join(lines)


def build_llm_prompt(schema: SchemaInfo) -> list:
    """Build the messages list for the LLM semantic analysis call.

    Returns a list of OpenAI-format message dicts with system and user
    roles.
    """
    if schema.is_empty:
        return [
            {
                "role": "system",
                "content": "The database has no tables to analyze.",
            }
        ]

    system_content = _SYSTEM_PROMPT + "\n\nExpected JSON structure:\n" + _OUTPUT_SCHEMA_DESCRIPTION

    schema_text = _format_schema_for_prompt(schema)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Analyze this database schema:\n\n{schema_text}"},
    ]


async def analyze_schema_with_llm(
    schema: SchemaInfo,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """Send the schema to an LLM and parse the JSON response.

    Uses ``call_llm()`` from ``agent.auxiliary_client`` with JSON mode.
    Retries once if the first response is not valid JSON.

    Args:
        schema: The extracted schema metadata.
        provider: Optional LLM provider override (e.g. ``"openrouter"``).
        model: Optional LLM model override
            (e.g. ``"google/gemini-2.5-flash"``).

    Returns:
        The parsed semantic mapping dict.

    Raises:
        LLMAnalysisError: If the LLM fails to return valid JSON after retries.
    """
    from agent.auxiliary_client import call_llm

    messages = build_llm_prompt(schema)
    if schema.is_empty:
        return {"description": "Empty database — no tables found", "tables": [], "foreign_keys": []}

    for attempt in range(2):
        try:
            response = call_llm(
                task="tenant_onboard",
                provider=provider,
                model=model or "google/gemini-2.5-flash",
                messages=messages,
                temperature=0.1,
                max_tokens=4096,
                extra_body={"response_format": {"type": "json_object"}},
            )
            raw = response.choices[0].message.content
        except Exception as exc:
            logger.warning(
                "LLM call failed (attempt %d/2): %s", attempt + 1, exc
            )
            if attempt == 0:
                continue
            raise LLMAnalysisError(
                f"LLM call failed after 2 attempts: {exc}"
            ) from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "LLM response was not valid JSON (attempt %d/2): %s",
                attempt + 1,
                exc,
            )
            if attempt == 0:
                # Add correction feedback and retry
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not valid JSON. "
                            "Please output ONLY valid JSON — no markdown "
                            "code fences, no surrounding text."
                        ),
                    }
                )
                continue
            raise LLMAnalysisError(
                f"LLM returned invalid JSON after 2 attempts: {raw[:200]}"
            )

        # Success — validate and return
        validate_semantic_mapping(parsed)
        return parsed

    # Should not be reached — both attempts exhausted
    raise LLMAnalysisError(
        "LLM analysis failed after exhausting all retries"
    )


def validate_semantic_mapping(mapping: Any) -> None:
    """Validate that the LLM output has the required structure.

    Raises:
        LLMAnalysisError: If the structure is invalid.
    """
    if not isinstance(mapping, dict):
        raise LLMAnalysisError(
            f"LLM output is not a JSON object: {type(mapping).__name__}"
        )

    if "description" not in mapping:
        raise LLMAnalysisError("LLM output missing 'description' field")

    if not isinstance(mapping["description"], str):
        raise LLMAnalysisError("LLM output 'description' must be a string")

    if "tables" not in mapping:
        raise LLMAnalysisError("LLM output missing 'tables' array")

    if not isinstance(mapping["tables"], list):
        raise LLMAnalysisError("LLM output 'tables' is not a list")

    for i, table in enumerate(mapping["tables"]):
        if not isinstance(table, dict):
            raise LLMAnalysisError(
                f"LLM output 'tables[{i}]' is not an object: {type(table).__name__}"
            )
        if "table_name" not in table:
            raise LLMAnalysisError(
                f"LLM output 'tables[{i}]' missing 'table_name'"
            )
        if "business_name" not in table:
            raise LLMAnalysisError(
                f"LLM output table '{table.get('table_name', '?')}' "
                f"missing 'business_name'"
            )
        if "business_description" not in table:
            raise LLMAnalysisError(
                f"LLM output table '{table.get('table_name', '?')}' "
                f"missing 'business_description'"
            )

    # Optional: validate foreign_keys if present
    if "foreign_keys" in mapping:
        if not isinstance(mapping["foreign_keys"], list):
            raise LLMAnalysisError("LLM output 'foreign_keys' must be a list")
        for i, fk in enumerate(mapping["foreign_keys"]):
            if not isinstance(fk, dict):
                raise LLMAnalysisError(
                    f"LLM output 'foreign_keys[{i}]' is not an object"
                )
            if "table_name" not in fk:
                raise LLMAnalysisError(
                    f"LLM output 'foreign_keys[{i}]' missing 'table_name'"
                )
            if "column_name" not in fk:
                raise LLMAnalysisError(
                    f"LLM output 'foreign_keys[{i}]' missing 'column_name'"
                )
            if "foreign_table_name" not in fk:
                raise LLMAnalysisError(
                    f"LLM output 'foreign_keys[{i}]' missing 'foreign_table_name'"
                )


# ── Persistence ──────────────────────────────────────────────────────────


async def persist_semantic_mapping(
    central_dsn: str, tenant_id: str, mapping: dict
) -> bool:
    """Insert or update the semantic mapping in the central database.

    Checks for an existing active row for the tenant and updates it if
    found, otherwise inserts a new row. Safe to re-run — replaces the
    previous mapping.
    """
    if asyncpg is None:
        raise TenantOnboardError("asyncpg is not installed")

    conn = await asyncpg.connect(dsn=central_dsn)
    try:
        # Check if an active mapping already exists for this tenant
        existing = await conn.fetchval(
            "SELECT id FROM public.semantic_mappings "
            "WHERE tenant_id = $1 AND is_active = TRUE "
            "LIMIT 1",
            tenant_id,
        )
        if existing:
            await conn.execute(
                "UPDATE public.semantic_mappings "
                "SET mapping_json = $1::jsonb, "
                "    version = version + 1 "
                "WHERE id = $2",
                json.dumps(mapping, ensure_ascii=False),
                existing,
            )
        else:
            await conn.execute(
                "INSERT INTO public.semantic_mappings "
                "(tenant_id, mapping_json, is_active, version, created_at) "
                "VALUES ($1, $2::jsonb, TRUE, 1, NOW())",
                tenant_id,
                json.dumps(mapping, ensure_ascii=False),
            )
        return True
    except Exception as exc:
        raise PersistenceError(
            f"Failed to persist semantic mapping: {exc}"
        ) from exc
    finally:
        await conn.close()


# ── Orchestrator ─────────────────────────────────────────────────────────


async def run_tenant_onboard(
    tenant_id: str,
    *,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    dry_run: bool = False,
) -> OnboardResult:
    """Orchestrate the full tenant onboarding workflow.

    Progress is printed to stdout as each phase completes.
    """
    # ── 0. Check configuration ──────────────────────────────────────
    central_dsn = os.environ.get("HERMES_CENTRAL_DB_URL")
    if not central_dsn:
        raise ConfigurationError(
            "HERMES_CENTRAL_DB_URL environment variable is not set. "
            "Tenant onboarding requires a connection to the central RDS "
            "database where connection strings and semantic mappings are "
            "stored."
        )

    # ── 1. Fetch tenant connection from central DB ──────────────────
    _print_step(1, 5, f"Fetching tenant {tenant_id} from central database...")
    tenant_info = await _fetch_tenant_from_central(central_dsn, tenant_id)
    print(f"      Found: db_type={tenant_info.db_type}")

    # ── 2. Inspect tenant schema ────────────────────────────────────
    _print_step(2, 5, "Connecting to tenant database and inspecting schema...")
    inspector = SchemaInspector(tenant_info.connection_string, tenant_info.db_type)
    try:
        await inspector.connect()
        schema = await inspector.extract()
    finally:
        await inspector.close()
        # Wipe decrypted connection string from memory
        tenant_info.connection_string = ""

    if schema.is_empty:
        raise EmptySchemaError(
            f"Tenant {tenant_id} database has no tables in public schema. "
            "Nothing to analyze."
        )

    table_count = len(schema.tables)
    col_count = len(schema.columns)
    fk_count = len(schema.foreign_keys)
    print(
        f"      Found {table_count} tables, {col_count} columns, "
        f"{fk_count} foreign keys"
    )

    # ── 3. Dry-run exit ─────────────────────────────────────────────
    if dry_run:
        _print_step(3, 3, "DRY RUN — skipping LLM analysis and persistence")
        print("      Schema inspection complete.")
        for tname in schema.table_names:
            col_n = sum(
                1 for c in schema.columns if c["table_name"] == tname
            )
            fk_n = sum(
                1 for f in schema.foreign_keys if f["table_name"] == tname
            )
            print(f"      • {tname}: {col_n} columns, {fk_n} FKs")
        return OnboardResult(
            tenant_id=tenant_id,
            table_count=table_count,
            column_count=col_count,
            semantic_mapping=None,
            inserted=False,
        )

    # ── 4. LLM analysis ─────────────────────────────────────────────
    provider_label = llm_provider or "auto"
    model_label = llm_model or "google/gemini-2.5-flash"
    _print_step(3, 5, f"Analyzing schema with LLM ({provider_label}/{model_label})...")
    mapping = await analyze_schema_with_llm(
        schema, provider=llm_provider, model=llm_model
    )
    annotated = len(mapping.get("tables", []))
    print(f"      Analysis complete: {annotated} tables annotated")

    # ── 5. Persist ──────────────────────────────────────────────────
    _print_step(4, 5, "Persisting semantic mapping to central database...")
    inserted = await persist_semantic_mapping(central_dsn, tenant_id, mapping)
    print(
        f"      {'Inserted' if inserted else 'Updated'} "
        f"semantic_mappings record for tenant {tenant_id}"
    )

    # ── 6. Done ─────────────────────────────────────────────────────
    _print_step(5, 5, "Done.")
    return OnboardResult(
        tenant_id=tenant_id,
        table_count=table_count,
        column_count=col_count,
        semantic_mapping=mapping,
        inserted=inserted,
    )


# ── Helpers ───────────────────────────────────────────────────────────────


def _print_step(current: int, total: int, message: str) -> None:
    """Print a progress step to stdout."""
    print(f"[{current}/{total}] {message}")


__all__ = [
    "TenantOnboardError",
    "ConfigurationError",
    "TenantNotFoundError",
    "EmptySchemaError",
    "LLMAnalysisError",
    "PersistenceError",
    "TenantDbInfo",
    "SchemaInfo",
    "OnboardResult",
    "SchemaInspector",
    "build_llm_prompt",
    "validate_semantic_mapping",
    "analyze_schema_with_llm",
    "persist_semantic_mapping",
    "run_tenant_onboard",
]
