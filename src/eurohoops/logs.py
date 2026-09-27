"""The public append-only CSV logs (predictions, odds, odds calls): rows are only ever appended."""

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def append_rows(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """Append ``rows`` (a new file gets the header first, even with no rows).

    A file whose last row has no newline (a run killed mid-write, a hand edit) is refused: the
    next row would be glued onto it. Nothing is written and ValueError is raised.
    """
    if path.exists() and path.stat().st_size:
        with path.open("rb") as fh:
            fh.seek(-1, 2)
            if fh.read() != b"\n":
                raise ValueError(f"{path} does not end with a newline; fix it by hand first")
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
