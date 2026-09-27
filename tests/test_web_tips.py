"""web/src/lib/tip.ts is the only way to build a tooltip; these source checks keep it usable.

There is no TypeScript check in CI (esbuild strips types), so a local that shadows the ``tip``
tag in its own initializer, ``const tip = ... tip`...` ...``, only fails in the browser, when a
row with a floor note is drawn (ReferenceError: Cannot access 'tip' before initialization)."""

import re

from tests.conftest import REPO

WEB = REPO / "web" / "src"
IMPORTS_TIP = re.compile(r'import \{[^}]*\btip\b[^}]*\} from "[./]+(?:lib/)?tip"')
SHADOW = re.compile(r"\b(?:const|let|var)\s+tip\b")


def test_no_file_that_imports_tip_declares_its_own_tip() -> None:
    offenders = [
        f"{path.relative_to(WEB)}:{text[: m.start()].count(chr(10)) + 1}"
        for path in sorted(WEB.rglob("*"))
        if path.suffix in {".ts", ".astro"}
        and IMPORTS_TIP.search(text := path.read_text(encoding="utf-8"))
        for m in SHADOW.finditer(text)
    ]
    assert offenders == []
