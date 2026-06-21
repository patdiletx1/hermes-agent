"""``hermes tenant-onboard`` subcommand — auto-generate semantic mapping.

Uses LLM-powered analysis to infer business concepts from a tenant's
raw database schema.
"""

from __future__ import annotations

from typing import Callable


def build_tenant_onboard_parser(
    subparsers, *, cmd_tenant_onboard: Callable
) -> None:
    """Attach the ``tenant-onboard`` subcommand to *subparsers*."""
    parser = subparsers.add_parser(
        "tenant-onboard",
        help="Auto-generate semantic mapping for a tenant database",
        description=(
            "Connect to a tenant database via the central RDS registry, "
            "inspect its schema (tables, columns, foreign keys), send the "
            "raw schema to an LLM for business-name inference, and persist "
            "the resulting semantic mapping to the central database.\n\n"
            "Use --dry-run to inspect the schema without calling the LLM "
            "or writing to the database."
        ),
    )
    parser.add_argument(
        "--tenant-id",
        required=True,
        help="UUID of the tenant to onboard (must exist in public.connections)",
    )
    parser.add_argument(
        "--llm-provider",
        default=None,
        help=(
            "LLM provider override (e.g. openrouter, anthropic, gemini). "
            "Default: auto-detect from config or environment."
        ),
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help=(
            "LLM model override (e.g. google/gemini-2.5-flash, "
            "claude-haiku-4-5-20251001). Default: google/gemini-2.5-flash."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Inspect the schema and print a summary without calling the "
            "LLM or writing to the database."
        ),
    )
    parser.set_defaults(func=cmd_tenant_onboard)
