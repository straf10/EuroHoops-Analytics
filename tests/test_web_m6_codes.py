"""The site's copy of the display codes (web/src/lib/m6_codes.json, for the API's source codes)
must equal publish.DISPLAY_CODES."""

import json
from pathlib import Path

from eurohoops.publish import DISPLAY_CODES

COPY = Path(__file__).parents[1] / "web" / "src" / "lib" / "m6_codes.json"


def test_site_display_codes_match_publish() -> None:
    assert json.loads(COPY.read_text(encoding="utf-8")) == DISPLAY_CODES
