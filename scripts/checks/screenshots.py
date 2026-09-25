"""Checklist item 20: screenshots of the EuroLeague scorecard (with the M1 row) at 1440 and 390 px,
light and dark, plus the horizontal overflow. Usage: screenshots.py <built site dir> <out dir>."""

import functools
import http.server
import shutil
import sys
import tempfile
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

site, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
root = Path(tempfile.mkdtemp())
shutil.copytree(site, root / "EuroHoops-Analytics")  # the site's base path
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{server.server_address[1]}/EuroHoops-Analytics/"
scorecard = '.scorecard[data-comp="euroleague"]'
ok = True
with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge")
    for scheme in ("light", "dark"):
        for width in (1440, 390):
            page = browser.new_page(
                viewport={"width": width, "height": 900},
                color_scheme=scheme,  # type: ignore[arg-type]
            )
            page.goto(url)
            page.wait_for_load_state("networkidle")
            overflow = page.evaluate(
                "document.documentElement.scrollWidth - document.documentElement.clientWidth"
            )
            row = page.locator(f"{scorecard} tr.m1")
            visible = row.count() == 1 and row.is_visible()
            page.locator(scorecard).screenshot(path=str(out / f"scorecard_m1_{width}_{scheme}.png"))
            print(f"{scheme} {width}: horizontal overflow {overflow} px; M1 row visible: {visible}")
            ok &= overflow == 0 and visible
            page.close()
    browser.close()
server.shutdown()
shutil.rmtree(root, ignore_errors=True)
sys.exit(0 if ok else 1)
