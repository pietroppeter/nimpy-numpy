## numpy arrays for nimpy.
##
## `NumpyArray[T]` is a typed view on a numpy array's memory, obtained through
## the Python buffer protocol: no copy is made, and strided arrays (slices such
## as ``x[::2]`` or transposes) are supported. Procs exported with
## `{.exportpy.}` can take and return `NumpyArray[T]` directly.
##
## .. code-block:: nim
##   import nimpy, nimpy_numpy
##
##   proc double(x: NumpyArray[float64]): NumpyArray[float64] {.exportpy.} =
##     result = newNumpyArray[float64](x.shape)
##     for i in 0 ..< x.len:
##       result[i] = 2 * x[i]

import nimpy
import nimpy/[raw_buffers, py_types]

type
  BufferHolder = object
    raw: RawPyBuffer
    acquired: bool

  NumpyArray*[T] = object
    ## A view on the memory of a numpy array with elements of type `T`.
    ## Copies of a `NumpyArray` share the view; the buffer is released when
    ## the last copy is destroyed.
    obj: PyObject              # keeps the numpy array alive
    holder: ref BufferHolder   # releases the buffer on destroy
    data: ptr UncheckedArray[byte]
    shape*: seq[int]
    strides*: seq[int]         ## in bytes, as in numpy
    writable*: bool            ## whether `[]=` is allowed; see `asNumpyArray`

  NumpyElement* = SomeNumber | bool
    ## Element types supported by `NumpyArray`.

proc `=destroy`(h: BufferHolder) {.raises: [].} =
  if h.acquired:
    var raw = h.raw
    try:
      raw.release()
    except Exception:
      discard

proc formatKinds(T: typedesc): set[char] =
  ## Buffer protocol format characters (struct module syntax) compatible with
  ## `T`; the item size is checked separately.
  when T is bool: {'?'}
  elif T is SomeFloat: {'e', 'f', 'd', 'g'}
  elif T is SomeSignedInt: {'b', 'h', 'i', 'l', 'q', 'n'}
  elif T is SomeUnsignedInt: {'B', 'H', 'I', 'L', 'Q', 'N'}
  else: {.error: "nimpy_numpy: unsupported element type " & $T.}

proc dtypeName*(T: typedesc[NumpyElement]): string =
  ## The numpy dtype name matching `T`, e.g. ``"float64"`` for `float`.
  when T is bool: "bool"
  elif T is SomeFloat: "float" & $(8 * sizeof(T))
  elif T is SomeSignedInt: "int" & $(8 * sizeof(T))
  else: "uint" & $(8 * sizeof(T))

proc asNumpyArray*[T: NumpyElement](o: PyObject, writable = false): NumpyArray[T] =
  ## A view on the numpy array (or any object supporting the buffer protocol)
  ## `o`. The view is read-only unless `writable` is true; procs that take a
  ## `NumpyArray` parameter get a read-only view. Raises `ValueError` if the
  ## element type does not match `T`, and a Python `BufferError` if
  ## `writable` is requested on a read-only array.
  var flags = PyBUF_RECORDS_RO
  if writable: flags = flags or PyBUF_WRITABLE
  result.obj = o
  result.holder = new BufferHolder
  o.getBuffer(result.holder.raw, flags.cint)
  result.holder.acquired = true
  let raw = result.holder.raw
  let fmt = if raw.format.isNil: "B" else: $cast[cstring](raw.format)
  let nativeOrder = fmt.len == 1 or (fmt.len == 2 and (fmt[0] == '@' or
    fmt[0] == '=' or (fmt[0] == '<') == (cpuEndian == littleEndian)))
  if not nativeOrder or fmt[^1] notin formatKinds(T) or raw.itemsize != sizeof(T):
    raise newException(ValueError, "nimpy_numpy: expected an array of " &
      dtypeName(T) & ", got buffer format '" & fmt & "' with item size " &
      $raw.itemsize)
  result.data = cast[ptr UncheckedArray[byte]](raw.buf)
  result.writable = writable
  let shape = cast[ptr UncheckedArray[Py_ssize_t]](raw.shape)
  let strides = cast[ptr UncheckedArray[Py_ssize_t]](raw.strides)
  for i in 0 ..< raw.ndim.int:
    result.shape.add shape[i].int
    result.strides.add strides[i].int

proc newNumpyArray*[T: NumpyElement](shape: varargs[int]): NumpyArray[T] =
  ## A new, uninitialized numpy array (``numpy.empty``). Its memory is owned
  ## by numpy, so it is safe to return it to Python.
  let np = pyImport("numpy")
  asNumpyArray[T](np.empty(@shape, dtypeName(T)), writable = true)

proc toPyObject*[T](a: NumpyArray[T]): PyObject {.inline.} =
  ## The numpy array this view is on.
  a.obj

proc ndim*[T](a: NumpyArray[T]): int {.inline.} = a.shape.len

proc len*[T](a: NumpyArray[T]): int {.inline.} =
  ## Length of the first dimension, as Python's ``len``.
  if a.shape.len == 0: 0 else: a.shape[0]

proc size*[T](a: NumpyArray[T]): int =
  ## Total number of elements.
  result = 1
  for s in a.shape: result *= s

proc isCContiguous*[T](a: NumpyArray[T]): bool =
  var expected = sizeof(T)
  for i in countdown(a.ndim - 1, 0):
    if a.shape[i] != 1 and a.strides[i] != expected: return false
    expected *= a.shape[i]
  true

proc indexError(a: NumpyArray, idx: openArray[int]) {.noinline, noreturn.} =
  if idx.len != a.ndim:
    raise newException(IndexDefect, "nimpy_numpy: " & $idx.len &
      " indices for an array of dimension " & $a.ndim)
  for i, k in idx:
    if k < 0 or k >= a.shape[i]:
      raise newException(IndexDefect, "nimpy_numpy: index " & $k &
        " out of bounds for axis " & $i & " with size " & $a.shape[i])
  raise newException(IndexDefect, "nimpy_numpy: invalid index")

template elementAt[T](a: NumpyArray[T], idx: openArray[int]): ptr T =
  var offset = 0
  when compileOption("boundChecks"):
    if idx.len != a.ndim: indexError(a, idx)
  for i, k in idx:
    when compileOption("boundChecks"):
      if k < 0 or k >= a.shape[i]: indexError(a, idx)
    offset += k * a.strides[i]
  cast[ptr T](addr a.data[offset])

proc `[]`*[T](a: NumpyArray[T], idx: varargs[int]): T {.inline.} =
  ## The element at `idx`, one index per dimension. Indices are checked
  ## unless bound checks are off (``-d:danger``).
  a.elementAt(idx)[]

proc readOnlyError() {.noinline, noreturn.} =
  raise newException(ValueError, "nimpy_numpy: array is read-only")

proc `[]=`*[T](a: var NumpyArray[T], idx: varargs[int], value: T) {.inline.} =
  if not a.writable: readOnlyError()
  a.elementAt(idx)[] = value

# nimpy conversion hooks, so that exported procs take and return NumpyArray[T].

proc pyValueToNim*[T: NumpyElement](v: PPyObject, o: var NumpyArray[T]) =
  var obj: PyObject
  pyValueToNim(v, obj)
  o = asNumpyArray[T](obj)

proc nimValueToPy*[T](a: NumpyArray[T]): PPyObject =
  nimValueToPy(a.obj)
