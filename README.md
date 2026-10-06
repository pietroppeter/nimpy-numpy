# nimpy-numpy

numpy arrays for Python-Nim interoperability using [nimpy](https://github.com/yglukhov/nimpy): write Nim procs that take and
return numpy arrays, and call them from Python.

> AI disclosure: this project is mostly vibed. Currently [level 7](https://www.visidata.org/blog/2026/ai/#level-6%3A-bots-coded%2C-human-understands-mostly) on visidata AI scale: Human specced, bots coded.

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
- Indices are bounds-checked unless compiled with `-d:danger`, which makes the `vander` above
  noticeably faster. How it compares with `numpy.vander` depends on the machine: see
  [Benchmark](#benchmark).

Supported element types: `float32`, `float64`, signed and unsigned integers of 8 to 64 bits,
and `bool`.

## API

| | |
|---|---|
| `NumpyArray[T]` | view with `shape`, `strides` (bytes) and `writable` |
| `asNumpyArray[T](obj, writable = false)` | view on a numpy array given as `PyObject` |
| `newNumpyArray[T](shape)` | new uninitialized numpy array |
| `a[i, j, ...]`, `a[i, j, ...] = v` | element access, one index per dimension |
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

`uv run python benchmark.py` builds `tests/vander.nim` with `-d:release` and `-d:danger`, times
both against `numpy.vander` and a pure Python version, and prints a Markdown table (not run in CI):

Results vary a lot between machines (and between runs for large arrays), so run it on yours
rather than reading too much into these two examples. In parentheses, time relative to `np.vander`.

Linux x86_64 (cloud VM), Python 3.13, numpy 2.5:

| n | np.vander | Nim -d:release | Nim -d:danger | pure Python |
|---:|---:|---:|---:|---:|
| 10 | 3.29 µs | 2.36 µs (0.72x) | 1.89 µs (0.58x) | 6.6 µs (2x) |
| 100 | 48 µs | 53.5 µs (1.1x) | 17.7 µs (0.37x) | 630 µs (13x) |
| 1000 | 3.9 ms | 5.05 ms (1.3x) | 2.09 ms (0.54x) | 106 ms (27x) |
| 3000 | 97.5 ms | 102 ms (1x) | 72.9 ms (0.75x) | not run |

macOS arm64 (Apple silicon), Python 3.13, numpy 2.5:

| n | np.vander | Nim -d:release | Nim -d:danger | pure Python |
|---:|---:|---:|---:|---:|
| 10 | 1.58 µs | 1.05 µs (0.66x) | 761 ns (0.48x) | 3.91 µs (2.5x) |
| 100 | 11.3 µs | 37.7 µs (3.3x) | 10.6 µs (0.94x) | 402 µs (35x) |
| 1000 | 1.29 ms | 3.92 ms (3x) | 1.3 ms (1x) | 55.2 ms (43x) |
| 3000 | 12.9 ms | 35.7 ms (2.8x) | 12.2 ms (0.95x) | not run |

On the Mac, `-d:danger` is on par with numpy and `-d:release` up to about 3x slower; at
n = 3000, numpy took 30.6 ms in one run and 12.9 ms in the next.

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
