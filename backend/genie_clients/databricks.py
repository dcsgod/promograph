"""Real Databricks Genie Conversation API client.

Flow (per Databricks docs):
  1. POST /api/2.0/genie/spaces/{space_id}/start-conversation  {content}
  2. poll GET .../conversations/{cid}/messages/{mid} until status COMPLETED
  3. GET .../messages/{mid}/query-result/{attachment_id}  -> tabular data

Auth: Bearer PAT. Free tier ~5 questions/min/workspace, so we keep calls
sparse and use exponential backoff while polling.

Selected when TPO_GENIE_BACKEND=databricks. Requires DATABRICKS_HOST,
DATABRICKS_TOKEN and the GENIE_SPACE_* ids in .env.
"""
from __future__ import annotations

import time

import httpx

from backend.config import settings
from backend.genie_clients.base import GenieResult

POLL_INTERVAL_START = 1.0
POLL_INTERVAL_MAX = 8.0
POLL_TIMEOUT = 120.0
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED"}


class DatabricksGenieClient:
    def __init__(self) -> None:
        if not (settings.databricks_host and settings.databricks_token):
            raise RuntimeError("DATABRICKS_HOST / DATABRICKS_TOKEN required for Genie.")
        self.host = settings.databricks_host.rstrip("/")
        self.headers = {"Authorization": f"Bearer {settings.databricks_token}"}
        self.spaces = {
            "sales": settings.genie_space_sales,
            "promo": settings.genie_space_promo,
            "inventory": settings.genie_space_inventory,
        }
        self.client = httpx.Client(base_url=self.host, headers=self.headers, timeout=30.0)

    def _space_id(self, space: str) -> str:
        sid = self.spaces.get(space) or settings.genie_space_sales
        if not sid:
            raise RuntimeError(f"No Genie space id configured for '{space}'.")
        return sid

    def ask(self, question: str, space: str = "sales") -> GenieResult:
        sid = self._space_id(space)
        start = self.client.post(
            f"/api/2.0/genie/spaces/{sid}/start-conversation",
            json={"content": question},
        )
        start.raise_for_status()
        body = start.json()
        cid = body["conversation"]["id"]
        mid = body["message"]["id"]

        msg = self._poll(sid, cid, mid)
        return self._to_result(sid, cid, mid, space, question, msg)

    def _poll(self, sid: str, cid: str, mid: str) -> dict:
        deadline = time.monotonic() + POLL_TIMEOUT
        interval = POLL_INTERVAL_START
        while time.monotonic() < deadline:
            r = self.client.get(
                f"/api/2.0/genie/spaces/{sid}/conversations/{cid}/messages/{mid}"
            )
            r.raise_for_status()
            msg = r.json()
            if msg.get("status") in TERMINAL:
                return msg
            time.sleep(interval)
            interval = min(POLL_INTERVAL_MAX, interval * 1.6)
        raise TimeoutError("Genie message did not complete in time.")

    def _to_result(self, sid, cid, mid, space, question, msg) -> GenieResult:
        if msg.get("status") != "COMPLETED":
            return GenieResult(space=space, question=question,
                               text=f"Genie returned status {msg.get('status')}.")
        attachments = msg.get("attachments") or []
        sql, text, attachment_id = "", "", None
        for att in attachments:
            if "text" in att and att["text"]:
                text = att["text"].get("content", "") if isinstance(att["text"], dict) else str(att["text"])
            if "query" in att and att["query"]:
                sql = att["query"].get("query", "") or att["query"].get("description", "")
                attachment_id = att.get("attachment_id")

        columns: list[str] = []
        rows: list[dict] = []
        if attachment_id:
            qr = self.client.get(
                f"/api/2.0/genie/spaces/{sid}/conversations/{cid}/messages/{mid}"
                f"/query-result/{attachment_id}"
            )
            if qr.status_code == 200:
                columns, rows = self._parse_query_result(qr.json())
        return GenieResult(space=space, question=question, sql=sql,
                           columns=columns, rows=rows, text=text)

    @staticmethod
    def _parse_query_result(payload: dict) -> tuple[list[str], list[dict]]:
        stmt = (payload.get("statement_response") or payload.get("query_result") or {})
        schema = (stmt.get("manifest", {}).get("schema", {}).get("columns", []))
        columns = [c.get("name", f"c{i}") for i, c in enumerate(schema)]
        data = stmt.get("result", {}).get("data_array", []) or []
        rows = [dict(zip(columns, r)) for r in data]
        return columns, rows

    def close(self) -> None:
        self.client.close()
