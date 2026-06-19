"""Database connection for the GizmoSQL / Arrow Flight SQL poetry corpus."""

import os

import pandas as pd
import adbc_driver_flightsql.dbapi as flight


def connect():
    """Open a connection to the GizmoSQL server.

    All connection parameters must be provided via environment variables
    (see .env.example):

      GIZMOSQL_DSN       — server URI  (e.g. grpc+tls://hostname:443)
      GIZMOSQL_USER      — username
      GIZMOSQL_PASSWORD  — password
      GIZMOSQL_DATABASE  — database name
    """
    for var in ('GIZMOSQL_DSN', 'GIZMOSQL_USER', 'GIZMOSQL_PASSWORD', 'GIZMOSQL_DATABASE'):
        if not os.getenv(var):
            raise EnvironmentError(
                f'{var} is not set. Copy .env.example to .env and fill in your credentials.'
            )

    return flight.connect(
        os.environ['GIZMOSQL_DSN'],
        db_kwargs={
            'username': os.environ['GIZMOSQL_USER'],
            'password': os.environ['GIZMOSQL_PASSWORD'],
            'adbc.flight.sql.rpc.call_header.database': os.environ['GIZMOSQL_DATABASE'],
        },
    )


def query(sql: str) -> pd.DataFrame:
    """Run *sql* and return the result as a DataFrame."""
    with connect() as conn:
        return pd.read_sql(sql, conn)
