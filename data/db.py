"""Read-only SQL access; parameters prevent path/string interpolation into SQL."""
import duckdb


def query_parquet(path, start, end):
    with duckdb.connect() as connection:
        return connection.execute(
            "SELECT * FROM read_parquet(?) WHERE timestamp >= ? AND timestamp < ? ORDER BY timestamp",
            [str(path), int(start), int(end)],
        ).fetchdf()
