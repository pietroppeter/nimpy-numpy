# Roadmap

- Fast path for C-contiguous arrays: `toOpenArray` and flat iteration without per-element
  stride arithmetic.
- More element types: complex, float16; 0-d arrays and scalars.
- Arraymancer: a zero-copy `Tensor` view of an input array, and returning tensors as numpy arrays
  without a copy (see [scinim#8](https://github.com/SciNim/scinim/issues/8)).
- Nim-owned arrays (idea, not a priority): split the strided indexing from storage, so the same
  code runs on `NumpyArray[T]` (Python buffer) and on a new `NdArray[T]` that owns a `seq[T]` and
  needs neither Python nor numpy, for pure Nim projects. `toNumpy` would hand one to Python,
  first by copying, later without a copy (keep the Nim memory alive with a capsule as the numpy
  array's `base`). Overlaps with Arraymancer's `Tensor`, which may cover this use case instead.
- Publish to the nimble package directory.
