# nimpy-numpy

numpy arrays for Python-Nim interoperability using [nimpy](https://github.com/yglukhov/nimpy): write Nim procs that take and
return numpy arrays, and call them from Python.

## Example

We want to write a nim version for computing a [Vandermonde matrix](https://en.wikipedia.org/wiki/Vandermonde_matrix). We will need a Nim file, `vander.nim`, compiled into a Python extension module and imported from Python.

With Python package [nimlang](https://github.com/pietroppeter/uv-add-nimlang) the whole round trip is:

```sh
uv init vander-demo && cd vander-demo
uv add nimlang numpy
uv run nimlang add nimpy https://github.com/pietroppeter/nimpy-numpy   # Nim deps
```

Write `vander.nim`:

```nim
import nimpy, nimpy_numpy

proc vander(x: NumpyArray[float64], n: int = -1): NumpyArray[float64] {.exportpy.} =
  ## Vandermonde matrix, as numpy.vander(x, N)
  let m = x.len
  let n = if n < 0: m else: n
  result = newNumpyArray[float64](m, n)
  for i in 0 ..< m:
    var p = 1.0
    for j in countdown(n - 1, 0):
      result[i, j] = p
      p *= x[i]
```

Compile it into the Python extension module `vander` (`vander.cpython-*.so`, or `.pyd` on
Windows), written next to it:

```sh
uv run nimlang build-ext vander.nim
```

and import it from Python (`uv run python`):

```python
>>> import numpy as np
>>> from vander import vander
>>> vander(np.array([1.0, 2.0, 3.0]))
array([[1., 1., 1.],
       [4., 2., 1.],
       [9., 3., 1.]])
```

## How it works

- `NumpyArray[T]` is a view on the array's memory through the Python buffer protocol. Nothing
  is copied, and strided arrays (`x[::2]`, `x.T`) work as they are.
- The element type is checked: passing an `int64` array where `NumpyArray[float64]` is expected
  raises an error in Python instead of reading garbage.
- `newNumpyArray[T](shape)` allocates with `numpy.empty`, so numpy owns the memory and returning
  the array to Python is safe.
- Views given to a proc as parameters are read-only; use `asNumpyArray[T](obj, writable = true)`
  to modify an array in place.
- Indices are bounds-checked unless compiled with `-d:danger`. How the `vander` above compares
  with `numpy.vander` depends on the machine: see [Benchmark](#benchmark).

Supported element types: `float32`, `float64`, signed and unsigned integers of 8 to 64 bits,
and `bool`.

## Fast path for contiguous arrays

`x[i]` and `result[i, j]` work for any array, strided or not, but each access recomputes the
element's address from the shape and strides. Most arrays are *C-contiguous*: their elements sit
row after row in one buffer with no gaps (every array from `newNumpyArray`, and any plain
`np.array`). For those, `toOpenArray` gives the elements as one flat `openArray[T]`, where
element `[i, j]` of an `m×n` array is at `i * n + j`. That lets the inner loop be a plain Nim
proc that knows nothing about numpy:

```nim
proc fillVander(v: var openArray[float64], x: openArray[float64], n: int, increasing: bool) =
  let (first, step) = if increasing: (0, 1) else: (n - 1, -1)
  for i, xi in x:
    var p = 1.0
    var k = i * n + first
    for j in 0 ..< n:
      v[k] = p
      p *= xi
      k += step

proc vander_fast(x: NumpyArray[float64], n: int = -1, increasing: bool = false): NumpyArray[float64] {.exportpy.} =
  let x = x.asContiguous    # a copy only if x is strided, e.g. x[::2]
  let n = if n < 0: x.len else: n
  result = newNumpyArray[float64](x.len, n)   # new arrays are always C-contiguous
  fillVander(result.toOpenArray, x.toOpenArray, n, increasing)
```

Accesses to the openArray are still bounds-checked (one comparison instead of the full index
computation), so a bug raises an error in Python rather than crashing it. Both versions are in
[tests/vander.nim](tests/vander.nim).

## API

| | |
|---|---|
| `NumpyArray[T]` | view with `shape`, `strides` (bytes) and `writable` |
| `asNumpyArray[T](obj, writable = false)` | view on a numpy array given as `PyObject` |
| `newNumpyArray[T](shape)` | new uninitialized numpy array |
| `a[i, j, ...]`, `a[i, j, ...] = v` | element access, one index per dimension |
| `a.toOpenArray` | the elements as one flat, row-major `openArray[T]` (C-contiguous only) |
| `a.asContiguous` | `a` itself if C-contiguous, else a contiguous copy |
| `a.unsafeData` | raw `ptr UncheckedArray[T]` to the elements (C-contiguous only, never checked) |
| `ndim`, `len`, `size`, `isCContiguous` | as in numpy |
| `toPyObject(a)` | the underlying numpy array |
| `dtypeName(T)` | numpy dtype name for `T`, e.g. `"float64"` |

## Install

```sh
nimble install https://github.com/pietroppeter/nimpy-numpy
```

or, in a Python project that builds its Nim extensions with nimlang,
`uv run nimlang add https://github.com/pietroppeter/nimpy-numpy` (as in the example above).

Requires Nim 2.0 or later and nimpy 0.2.1 or later.

## Benchmark

`uv run python benchmark.py` builds `tests/vander.nim` with `-d:release` (the default) and
`-d:danger`, times `vander` and `vander_fast` against `numpy.vander` and a pure Python version,
and prints a Markdown table (not run in CI).

Results vary a lot between machines (and between runs for large arrays), so run it on yours
rather than reading too much into these examples. In parentheses, time relative to `np.vander`.

Linux x86_64 (cloud VM), Python 3.13, numpy 2.5:

| n | np.vander | vander | vander_fast | vander -d:danger | vander_fast -d:danger | pure Python |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 3.55 µs | 2.16 µs (0.61x) | 1.77 µs (0.5x) | 1.67 µs (0.47x) | 1.87 µs (0.53x) | 6.62 µs (1.9x) |
| 100 | 38 µs | 54.1 µs (1.4x) | 13.8 µs (0.36x) | 18.8 µs (0.49x) | 11.1 µs (0.29x) | 647 µs (17x) |
| 1000 | 4.13 ms | 5.41 ms (1.3x) | 1.96 ms (0.47x) | 2.24 ms (0.54x) | 1.89 ms (0.46x) | 103 ms (25x) |
| 3000 | 100 ms | 107 ms (1.1x) | 71.2 ms (0.71x) | 84.3 ms (0.84x) | 72.7 ms (0.73x) | not run |

macOS arm64 (Apple silicon), Python 3.13, numpy 2.5:

| n | np.vander | vander | vander_fast | vander -d:danger | vander_fast -d:danger | pure Python |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1.57 µs | 1.04 µs (0.66x) | 778 ns (0.5x) | 737 ns (0.47x) | 724 ns (0.46x) | 3.87 µs (2.5x) |
| 100 | 11.1 µs | 37.6 µs (3.4x) | 8.26 µs (0.74x) | 10.5 µs (0.94x) | 6.22 µs (0.56x) | 403 µs (36x) |
| 1000 | 1.29 ms | 3.95 ms (3x) | 1.19 ms (0.92x) | 1.31 ms (1x) | 1.17 ms (0.91x) | 55.8 ms (43x) |
| 3000 | 12.9 ms | 35.8 ms (2.8x) | 11 ms (0.86x) | 12.1 ms (0.94x) | 10.9 ms (0.85x) | not run |

On both machines, with the default build, `vander_fast` is about 3x faster than `vander` for 100
to 1000 points and faster than `np.vander` (slightly on the Mac, about 2x on the Linux VM);
`-d:danger` adds little on top of it.

## Development

```sh
uv run pytest tests      # compiles tests/*.nim with nimlang, then checks them against numpy
```

## Prior art

- nimpy's own [`raw_buffers`](https://github.com/yglukhov/nimpy/blob/master/nimpy/raw_buffers.nim)
  module and [numpy notes](https://github.com/yglukhov/nimpy/blob/master/docs/numpy.md), which
  this package builds on.
- [scinim](https://github.com/SciNim/scinim)'s `numpyarrays` module, which pairs numpy arrays with
  Arraymancer tensors. nimpy-numpy depends only on nimpy.

See [ROADMAP.md](ROADMAP.md) for what comes next, including Arraymancer support.
