"""Thin DuckDB helpers shared across the codebase (generator, mock Genie, ETL)."""
from __future__ import annotations

import duckdb
import pandas as pd

from backend.config import settings


def connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Open the project DuckDB file. Use read_only=True for query paths."""
    path = settings.duckdb_path
    if read_only and not path.exists():
        raise FileNotFoundError(
            f"DuckDB not found at {path}. Run `python -m backend.data_gen.generate` first."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def write_table(con: duckdb.DuckDBPyConnection, name: str, df: pd.DataFrame) -> None:
    """Replace a table with the contents of a DataFrame."""
    con.register("_df_tmp", df)
    con.execute(f"DROP TABLE IF EXISTS {name}")
    con.execute(f"CREATE TABLE {name} AS SELECT * FROM _df_tmp")
    con.unregister("_df_tmp")


def query(sql: str, params: list | None = None) -> pd.DataFrame:
    """Run a read-only query and return a DataFrame."""
    con = connect(read_only=True)
    try:
        return con.execute(sql, params or []).df()
    finally:
        con.close()
