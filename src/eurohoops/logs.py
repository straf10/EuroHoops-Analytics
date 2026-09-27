"""The public append-only CSV logs (predictions, odds, odds calls): rows are only ever appended."""

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def append_rows(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """Append ``rows`` (a new file gets the header first, even with no rows)."""
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
