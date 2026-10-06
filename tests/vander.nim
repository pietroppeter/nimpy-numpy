import nimpy, nimpy_numpy

proc vander(x: NumpyArray[float64], n: int = -1, increasing: bool = false): NumpyArray[float64] {.exportpy.} =
  ## Vandermonde matrix, as ``numpy.vander(x, N, increasing)``.
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
