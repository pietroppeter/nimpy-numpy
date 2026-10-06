import nimpy, nimpy_numpy

proc vander(x: NumpyArray[float64], n: int = -1, increasing: bool = false): NumpyArray[float64] {.exportpy.} =
  ## Vandermonde matrix, as ``numpy.vander(x, N, increasing)``.
  ## Standard version: indexes the arrays element by element.
  if x.ndim != 1:
    raise newException(ValueError, "x must be a one-dimensional array")
  let m = x.len
  let n = if n < 0: m else: n
  result = newNumpyArray[float64](m, n)
  for i in 0 ..< m:
    var p = 1.0
    for j in 0 ..< n:
      result[i, if increasing: j else: n - 1 - j] = p
      p *= x[i]

proc fillVander(v: var openArray[float64], x: openArray[float64], n: int, increasing: bool) =
  ## Plain Nim kernel: writes the ``x.len × n`` Vandermonde matrix into `v`,
  ## row after row. Knows nothing about numpy.
  let (first, step) = if increasing: (0, 1) else: (n - 1, -1)
  for i, xi in x:
    var p = 1.0
    var k = i * n + first
    for j in 0 ..< n:
      v[k] = p
      p *= xi
      k += step

proc vander_fast(x: NumpyArray[float64], n: int = -1, increasing: bool = false): NumpyArray[float64] {.exportpy.} =
  ## Same result as `vander`, using the contiguous fast path: the arrays are
  ## passed to a plain Nim kernel as flat openArrays.
  if x.ndim != 1:
    raise newException(ValueError, "x must be a one-dimensional array")
  let x = x.asContiguous    # a copy only if x is strided, e.g. x[::2]
  let n = if n < 0: x.len else: n
  result = newNumpyArray[float64](x.len, n)   # new arrays are always C-contiguous
  fillVander(result.toOpenArray, x.toOpenArray, n, increasing)
