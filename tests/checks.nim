## Exported procs exercising the rest of the nimpy_numpy API.
import nimpy, nimpy_numpy

proc info_f32(x: NumpyArray[float32]): (seq[int], seq[int], int, bool) {.exportpy.} =
  (x.shape, x.strides, x.size, x.isCContiguous)

proc sum_i32(x: NumpyArray[int32]): int {.exportpy.} =
  for i in 0 ..< x.len: result += x[i].int

proc sum_u8(x: NumpyArray[uint8]): int {.exportpy.} =
  for i in 0 ..< x.len: result += x[i].int

proc count_true(x: NumpyArray[bool]): int {.exportpy.} =
  for i in 0 ..< x.len:
    if x[i]: inc result

proc scale_inplace(o: PyObject, k: float64) {.exportpy.} =
  var x = asNumpyArray[float64](o, writable = true)
  for i in 0 ..< x.len: x[i] = k * x[i]

proc write_readonly(x: NumpyArray[float64]) {.exportpy.} =
  var x = x
  x[0] = 1.0

proc get3(x: NumpyArray[int64], i, j, k: int): int64 {.exportpy.} =
  x[i, j, k]

proc ints(n: int): NumpyArray[int] {.exportpy.} =
  result = newNumpyArray[int](n)
  for i in 0 ..< n: result[i] = i * i
