"""Neo4j driver singleton + schema bootstrap."""
from __future__ import annotations

from pathlib import Path

from neo4j import Driver, GraphDatabase

from backend.config import settings

_driver: Driver | None = None


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
    return _driver


def close_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def apply_schema() -> None:
    """Run the constraint/index statements in schema.cypher."""
    schema = (Path(__file__).parent / "schema.cypher").read_text(encoding="utf-8")
    stmts = [s.strip() for s in schema.split(";") if s.strip() and not s.strip().startswith("//")]
    with get_driver().session() as session:
        for stmt in stmts:
            session.run(stmt)
