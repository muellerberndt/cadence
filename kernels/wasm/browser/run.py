"""Run a script of this folder under Pyodide in headless Chromium, with the browser kernel.

    python kernels/wasm/browser/run.py shadow.py [args...]
    python kernels/wasm/browser/run.py --workers 6 many.py wasm 8 10 3 {k}

Bundles the library's source (``src/cadence``), the kernel (``kernels/wasm/cadence_kernel.wasm``,
built by ``kernels/wasm/build.py``) and this folder's scripts, serves them on a local port, runs
the script on the page (or in ``--workers`` Pyodide workers, each with its number for ``{k}``)
and prints its output. Needs Playwright with Chromium (``pip install playwright``, ``playwright
install chromium``). Pyodide comes from its CDN at the pinned version, or from a local copy
(``--pyodide DIR``, a folder holding ``pyodide.mjs``, its assets and the NumPy wheel). The exit
status is the script's.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import io
import os
import sys
import tarfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PYODIDE = "314.0.7"


def bundle(kernel: Path) -> bytes:
    def clean(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        return None if "__pycache__" in info.name or info.name.endswith(".pyc") else info

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        tar.add(ROOT / "src" / "cadence", arcname="src/cadence", filter=clean)
        tar.add(kernel, arcname="cadence_kernel.wasm")
        for path in sorted(HERE.glob("*.py")):
            tar.add(path, arcname=f"browser/{path.name}")
    return buffer.getvalue()


def serve(files: dict[str, bytes], pyodide: Path | None) -> http.server.ThreadingHTTPServer:
    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if path in files:
                body = files[path]
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path.startswith("/pyodide/") and pyodide is not None:
                self.path = path[len("/pyodide"):]
                self.directory = str(pyodide)
            super().do_GET()

        def end_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Handler, directory=str(HERE)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def say(line: str) -> None:
    """A line of output, blocking until a slow pipe takes it (Playwright leaves the descriptor
    non-blocking)."""
    os.set_blocking(sys.stdout.fileno(), True)
    print(line, flush=True)


def main() -> int:
    from playwright.sync_api import sync_playwright

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--pyodide", type=Path, default=None)
    parser.add_argument("--kernel", type=Path,
                        default=ROOT / "kernels" / "wasm" / "cadence_kernel.wasm")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("script")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    if not options.kernel.exists():
        raise SystemExit(f"{options.kernel} is missing: run kernels/wasm/build.py first")
    server = serve({"/bundle.tar": bundle(options.kernel)}, options.pyodide)
    base = "/pyodide/" if options.pyodide else f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE}/full/"
    query = [("script", f"browser/{options.script}"), ("pyodide", base)]
    query += [("arg", a) for a in options.args]
    if options.workers:
        query.append(("workers", str(options.workers)))
    from urllib.parse import urlencode

    url = f"http://127.0.0.1:{server.server_address[1]}/page.html?{urlencode(query)}"
    status = "error"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        printed, began = 0, time.time()
        while time.time() - began < options.timeout:
            lines = page.evaluate("window.__out || []")
            for line in lines[printed:]:
                say(line)
            printed = len(lines)
            done = page.evaluate("window.__done || null")
            if done:
                for line in page.evaluate("window.__out || []")[printed:]:
                    say(line)
                status = done
                break
            time.sleep(0.25)
        say(f"[{status} in {time.time() - began:.1f}s, Chromium {browser.version}, "
            f"Pyodide {PYODIDE}]")
        browser.close()
    server.shutdown()
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
