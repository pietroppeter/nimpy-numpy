# Roadmap

- Fast path for C-contiguous arrays: `toOpenArray` and flat iteration without per-element
  stride arithmetic.
- More element types: complex, float16; 0-d arrays and scalars.
- Arraymancer: a zero-copy `Tensor` view of an input array, and returning tensors as numpy arrays
  without a copy (see [scinim#8](https://github.com/SciNim/scinim/issues/8)).
- Publish to the nimble package directory.
