"""Build the browser kernel: ``cadence_kernel.wasm``, the compiled kernels of ``cadence.fused``
for the library under Pyodide (``cadence/_wasm.py`` loads it).

    python kernels/wasm/build.py [--out PATH]

Needs Rust through ``rustup``, which installs the compiler and the ``wasm32-unknown-unknown``
target that ``rust-toolchain.toml`` names; ``Cargo.lock`` pins the one dependency. Cargo builds
the module with SIMD128 against an imported memory; this script then makes the module's stack
pointer an import too, so the kernels run on the host's stack inside Pyodide's own memory, and
refuses a module that holds data of its own (it would be written into the host's memory at a
fixed address when the module starts). The module holds no path and no name, so the same
source and compiler give the same bytes on any machine; the script prints their SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = "wasm32-unknown-unknown"
FLAGS = [
    "-C", "target-feature=+simd128",
    "-C", "link-arg=--import-memory",
    "-C", "link-arg=-zstack-size=0",
]


def _uleb(data: bytes, i: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[i]
        i += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return value, i


def _leb(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def sections(module: bytes) -> list[tuple[int, bytes]]:
    """The module's sections as ``(id, body)``."""
    if module[:4] != b"\0asm":
        raise ValueError("not a WebAssembly module")
    out, i = [], 8
    while i < len(module):
        size, j = _uleb(module, i + 1)
        out.append((module[i], module[j:j + size]))
        i = j + size
    return out


def import_stack_pointer(module: bytes) -> bytes:
    """The module's first global, the stack pointer the linker defines, as the import
    ``env.__stack_pointer``. Imported globals come first in the index space, so every reference
    to global 0 then reads the host's stack pointer; the other globals keep their indices."""
    parts = sections(module)
    if any(sid == 11 and _uleb(body, 0)[0] for sid, body in parts):
        raise SystemExit("the module holds data of its own; it would be written into the host")
    out = bytearray(module[:8])
    for sid, body in parts:
        if sid == 2:
            count, i = _uleb(body, 0)
            entry = _leb(3) + b"env" + _leb(15) + b"__stack_pointer" + bytes([3, 0x7F, 1])
            body = _leb(count + 1) + body[i:] + entry
        elif sid == 6:
            count, i = _uleb(body, 0)
            if body[i:i + 2] != bytes([0x7F, 1]):
                raise SystemExit("the module's first global is not the stack pointer")
            j = i + 2
            while body[j] != 0x0B:
                j += 1
            body = _leb(count - 1) + body[j + 1:]
        elif sid == 0:  # custom sections (names, producers) are left out
            continue
        out += bytes([sid]) + _leb(len(body)) + body
    return bytes(out)


def build(out: Path) -> Path:
    env = dict(os.environ)
    env["RUSTFLAGS"] = " ".join(FLAGS)
    env["CARGO_TARGET_DIR"] = str(HERE / "target")
    subprocess.run(  # in this folder, where rustup reads rust-toolchain.toml
        ["cargo", "build", "--release", "--locked", "--target", TARGET, "--quiet"],
        check=True, env=env, cwd=HERE,
    )
    built = HERE / "target" / TARGET / "release" / "cadence_kernel.wasm"
    module = import_stack_pointer(built.read_bytes())
    out.write_bytes(module)
    digest = hashlib.sha256(module).hexdigest()
    print(f"{out} {len(module):,} bytes sha256 {digest}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "cadence_kernel.wasm")
    args = parser.parse_args()
    build(args.out)


if __name__ == "__main__":
    sys.exit(main())
