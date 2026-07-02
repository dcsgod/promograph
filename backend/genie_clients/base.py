"""Genie client interface. The mock and Databricks backends return the SAME
`GenieResult` shape, so swapping is a one-line env change (TPO_GENIE_BACKEND).
"""
from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field


class GenieResult(BaseModel):
    space: str                                  # logical space: sales | promo | inventory
    question: str
    sql: str = ""                               # SQL Genie generated (or the mock ran)
    columns: list[str] = Field(default_factory=list)
    rows: list[dict] = Field(default_factory=list)
    text: str = ""                              # short natural-language summary

    def preview(self, n: int = 8) -> dict:
        """Compact form for feeding back to the LLM (caps row count)."""
        return {
            "space": self.space,
            "sql": self.sql,
            "columns": self.columns,
            "rows": self.rows[:n],
            "row_count": len(self.rows),
            "text": self.text,
        }


class GenieClient(Protocol):
    def ask(self, question: str, space: str = "sales") -> GenieResult: ...


def get_genie_client() -> GenieClient:
    from backend.config import settings

    if settings.tpo_genie_backend == "databricks":
        from backend.genie_clients.databricks import DatabricksGenieClient

        return DatabricksGenieClient()
    from backend.genie_clients.mock import MockGenieClient

    return MockGenieClient()
