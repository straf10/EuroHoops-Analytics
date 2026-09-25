"""Helpers shared by the pandera contracts."""

import pandera.pandas as pa


def schema_dtypes(schema: pa.DataFrameSchema) -> dict[str, str]:
    """Pandas dtypes of a schema's columns (strings as the default ``str`` dtype).

    Cast to these before validating: an empty frame's columns are ``object`` otherwise.
    """
    return {
        name: "str" if str(column.dtype).startswith("string") else str(column.dtype)
        for name, column in schema.columns.items()
    }
