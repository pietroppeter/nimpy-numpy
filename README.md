# nimpy-numpy

numpy arrays for [nimpy](https://github.com/yglukhov/nimpy): write Nim procs that take and
return numpy arrays, and call them from Python.

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

```python
>>> import numpy as np, vander
>>> vander.vander(np.array([1.0, 2.0, 3.0]))
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
- Indices are bounds-checked unless compiled with `-d:danger`. With `-d:danger` the `vander`
  above runs about as fast as `numpy.vander`.

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

or, in a Python project that builds its Nim extension with
[nimlang](https://github.com/pietroppeter/uv-add-nimlang), add it to `[tool.nimlang]` in
`pyproject.toml`.

Requires Nim 2.0 or later and nimpy 0.2.1 or later.

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
