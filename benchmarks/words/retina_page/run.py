"""R6: the retina, the brain and the map in a web page (cadence#330, RETINA_PROTOCOL.md).

    python benchmarks/words/retina_page/run.py [port]

Builds the wheel of this checkout, writes a page beside it (``index.html``: Pyodide from
jsDelivr, NumPy, the wheel, the chamber's ``world.py``, ``words.py`` and ``retina.py``, and
``bench.py``), serves it, opens it in headless Chromium (Playwright) and prints the timings the
page measured as JSON: each moment's parts in milliseconds (``render``: drawing the eyes' images;
``see``: finding and recognising, with the lessons; ``map``: the map's moment; ``moment``: the
brain's, ``see`` and ``map``) and each question's.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PYODIDE = "314.0.7"


def main() -> int:
    from playwright.sync_api import sync_playwright

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8930
    with tempfile.TemporaryDirectory() as tmp:
        site = Path(tmp) / "site"
        (site / "chamber").mkdir(parents=True)
        subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", str(site),
                        str(ROOT)], check=True, capture_output=True)
        wheel = next(site.glob("cadence_net-*.whl")).name
        for name in ("index.html", "bench.py"):
            shutil.copy(HERE / name, site / name)
        chamber = ["world.py", "words.py", "retina.py"]
        for name in chamber:
            shutil.copy(HERE.parent / name, site / "chamber" / name)
        (site / "manifest.json").write_text(json.dumps(
            {"pyodide": PYODIDE, "wheel": wheel, "chamber": chamber}))
        server = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "-d",
                                   str(site)], stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
        time.sleep(1.0)
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(f"http://localhost:{port}/index.html")
                page.wait_for_function("document.title === 'done'", timeout=1_800_000)
                result = json.loads(page.inner_text("#result"))
                result["log"] = page.inner_text("#log")
                result["browser"] = browser.version
                browser.close()
        finally:
            server.terminate()
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
