"""The public append-only CSV logs (predictions, odds, odds calls): rows are only ever appended."""

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"  # UTC stamps in the logs, the reports and the site data


def require_terminated(path: Path) -> None:
    """Raise ValueError if ``path`` exists and its last row has no newline.

    A run killed mid-write or a hand edit can leave it so; the next row would be glued onto it.
    """
    if not path.exists() or not path.stat().st_size:
        return
    with path.open("rb") as fh:
        fh.seek(-1, 2)
        if fh.read() != b"\n":
            raise ValueError(f"{path} does not end with a newline; fix it by hand first")


def append_rows(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """Append ``rows`` (a new file gets the header first, even with no rows).

    Refuses a file whose last row is unterminated (``require_terminated``): nothing is written.
    """
    require_terminated(path)
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
