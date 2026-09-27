"""Helpers shared by the pandera contracts."""

import pandas as pd
import pandera.pandas as pa


def schema_dtypes(schema: pa.DataFrameSchema) -> dict[str, str]:
    """Pandas dtypes of a schema's columns (strings as the default ``str`` dtype).

    Cast to these before validating: an empty frame's columns are ``object`` otherwise.
    """
    return {
        name: "str" if str(column.dtype).startswith("string") else str(column.dtype)
        for name, column in schema.columns.items()
    }


def validated(frame: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    """Cast ``frame`` to the schema's dtypes, then validate it."""
    return schema.validate(frame.astype(schema_dtypes(schema)))
