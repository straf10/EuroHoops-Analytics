"""Weeks 16-18 L11 (I): screenshots of the player page of a projected fixture player and the
Scouting page at 1440 and 390 px, light and dark, plus three checks on every page: no horizontal
overflow, no console error, and every displayed model number ([data-num]) sits in a [data-figure]
that shows its interval or sample ([data-interval] or [data-sample]).
Usage: screenshots_m6_players.py <built site dir> <out dir>.
Files are m6_<page>_<width>_<scheme>.png.
"""

import functools
import http.server
import shutil
import sys
import tempfile
import threading
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

PAGES = {
    "player": "players/levi-randolph/",  # projection, impact, shot-making and comparables
    "player_sparse": "players/saben-lee/",  # a projection with no history; nothing else published
    "scouting": "scouting/",
}
# Controls whose every option shows other numbers: each option is clicked and checked in turn.
CHOICES = {"scouting": ["dim", "league"]}

LABELLED = """() => {
  const shown = (el) => {
    const r = el.getBoundingClientRect();
    const s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== "hidden";
  };
  const bad = [];
  let count = 0;
  for (const num of document.querySelectorAll("[data-num]")) {
    if (!shown(num)) continue;
    count++;
    const figure = num.closest("[data-figure]");
    const text = num.textContent.trim();
    if (!figure) bad.push(`${text}: not in a [data-figure]`);
    else if (
      ![...figure.querySelectorAll("[data-interval], [data-sample]")].some(
        (l) => shown(l) && l.textContent.trim(),
      )
    )
      bad.push(`${text}: its figure shows no interval or sample`);
  }
  return { count, bad };
}"""


def unlabelled(page: Page, name: str) -> tuple[int, list[str]]:
    """Count the shown [data-num] and list those without a label, for every control option."""
    seen, bad = 0, []
    options: list[tuple[str, str]] = [("", "")]
    for control in CHOICES.get(name, []):
        values = page.locator(f'input[name="{control}"]').evaluate_all(
            "els => els.map(e => e.value)"
        )
        options += [(control, v) for v in values]
    for control, value in options:
        if control:
            page.locator(f'input[name="{control}"][value="{value}"]').evaluate("e => e.click()")
        result = page.evaluate(LABELLED)
        seen += result["count"]
        bad += [f"{control}={value} {line}" if control else line for line in result["bad"]]
    return seen, bad


def main() -> int:
    site, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp())
    shutil.copytree(site, root / "EuroHoops-Analytics")  # the site's base path

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

    handler = functools.partial(Quiet, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/EuroHoops-Analytics/"
    ok = True
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge")
        for name, path in PAGES.items():
            for scheme in ("light", "dark"):
                for width in (1440, 390):
                    page = browser.new_page(
                        viewport={"width": width, "height": 900},
                        color_scheme=scheme,  # type: ignore[arg-type]
                    )
                    errors: list[str] = []
                    page.on("console", lambda m, e=errors: m.type == "error" and e.append(m.text))
                    page.on("pageerror", lambda x, e=errors: e.append(str(x)))
                    page.goto(url + path)
                    page.wait_for_load_state("networkidle")
                    overflow = page.evaluate(
                        "document.documentElement.scrollWidth"
                        " - document.documentElement.clientWidth"
                    )
                    page.screenshot(
                        path=str(out / f"m6_{name}_{width}_{scheme}.png"), full_page=True
                    )
                    numbers, bad = unlabelled(page, name)
                    # the fixture publishes numbers on every page here: zero would be a vacuous pass
                    problems = overflow != 0 or bool(bad) or bool(errors) or numbers == 0
                    print(
                        f"{name} {scheme} {width}: overflow {overflow} px; "
                        f"{numbers} model numbers, "
                        f"{len(bad)} unlabelled; {len(errors)} console errors"
                        + ("" if not problems else f" FAIL {bad[:3]} {errors[:3]}")
                    )
                    ok &= not problems
                    page.close()
        browser.close()
    server.shutdown()
    shutil.rmtree(root, ignore_errors=True)
    return 0 if ok else 1


sys.exit(main())
