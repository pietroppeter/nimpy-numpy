"""Benchmark tests/vander.nim against numpy.vander and pure Python.

Builds the Nim module twice with nimlang (default -d:release, and -d:danger),
times its two implementations (vander, indexing element by element, and
vander_fast, using the contiguous fast path) and prints a Markdown table.

    uv run python benchmark.py
    uv run python benchmark.py --sizes 10 100 1000 --python-max 1000
"""

import argparse
import importlib.machinery
import importlib.util
import platform
import shutil
import subprocess
import sys
import tempfile
import timeit
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def build(out_dir: Path, *nim_args: str):
    """Compile tests/vander.nim into out_dir and return the module."""
    out_dir.mkdir()
    cmd = [sys.executable, "-m", "nimlang", "build-ext", str(HERE / "tests" / "vander.nim"), "-o", str(out_dir)]
    cmd += [f"--nim-arg={a}" for a in nim_args]
    subprocess.run(cmd, check=True, capture_output=True)
    # Only the extension module itself (Windows also leaves .lib/.exp files next to the .pyd).
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    (path,) = [p for p in out_dir.iterdir() if p.name.startswith("vander.") and p.name.endswith(suffixes)]
    spec = importlib.util.spec_from_file_location("vander", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def python_vander(x):
    n = len(x)
    return [[xi ** (n - 1 - j) for j in range(n)] for xi in x]


def best_time(f, arg, repeat):
    """Best time per call, in seconds."""
    number, _ = timeit.Timer(lambda: f(arg)).autorange()
    return min(timeit.repeat(lambda: f(arg), number=number, repeat=repeat)) / number


def fmt(seconds):
    if seconds is None:
        return "not run"
    for unit, scale in (("s", 1), ("ms", 1e-3), ("µs", 1e-6)):
        if seconds >= scale:
            return f"{seconds / scale:.3g} {unit}"
    return f"{seconds / 1e-9:.3g} ns"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 100, 1000, 3000])
    parser.add_argument("--python-max", type=int, default=1000, help="largest size timed in pure Python")
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    subprocess.run([sys.executable, "-m", "nimlang", "sync"], check=True, cwd=HERE, capture_output=True)
    # Not a TemporaryDirectory: on Windows the loaded .pyd files can't be deleted until exit.
    tmp = tempfile.mkdtemp(prefix="nimpy-numpy-bench-")
    try:
        release = build(Path(tmp) / "release")
        danger = build(Path(tmp) / "danger", "-d:danger")
        columns = {
            "np.vander": np.vander,
            "vander": release.vander,
            "vander_fast": release.vander_fast,
            "vander -d:danger": danger.vander,
            "vander_fast -d:danger": danger.vander_fast,
        }

        x = np.random.default_rng(0).random(5)
        for f in columns.values():
            np.testing.assert_allclose(f(x), np.vander(x))

        print(f"vander(x) for x of n random float64, best of {args.repeat}, time per call")
        print(f"({platform.system()} {platform.machine()}, Python {platform.python_version()}, numpy {np.__version__})")
        print()
        print("| n | " + " | ".join(columns) + " | pure Python |")
        print("|---:|" + "---:|" * (len(columns) + 1))
        for n in args.sizes:
            x = np.random.default_rng(n).random(n)
            times = [best_time(f, x, args.repeat) for f in columns.values()]
            t_py = best_time(python_vander, x.tolist(), args.repeat) if n <= args.python_max else None
            t_np = times[0]

            def cell(t):
                return fmt(t) if t is None else f"{fmt(t)} ({t / t_np:.2g}x)"

            print(f"| {n} | {fmt(t_np)} | " + " | ".join(cell(t) for t in times[1:] + [t_py]) + " |")
        print()
        print("In parentheses: time relative to np.vander (lower is faster).")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
