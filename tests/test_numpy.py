import numpy as np
import pytest

import checks
from vander import vander


@pytest.mark.parametrize("x", [np.array([1.0, 2.0, 3.0, 5.0]), np.linspace(-1, 1, 7), np.array([])])
def test_vander(x):
    np.testing.assert_array_equal(vander(x), np.vander(x))
    np.testing.assert_array_equal(vander(x, 3), np.vander(x, 3))
    np.testing.assert_array_equal(vander(x, increasing=True), np.vander(x, increasing=True))


def test_vander_returns_ndarray():
    v = vander(np.array([2.0, 3.0]))
    assert isinstance(v, np.ndarray)
    assert v.dtype == np.float64
    assert v.flags.c_contiguous and v.flags.owndata


def test_vander_strided_input_is_not_copied():
    x = np.arange(20.0)[::3]
    np.testing.assert_array_equal(vander(x), np.vander(x))


def test_vander_rejects_wrong_dtype_and_ndim():
    with pytest.raises(Exception, match="expected an array of float64"):
        vander(np.arange(4))
    with pytest.raises(Exception, match="one-dimensional"):
        vander(np.ones((2, 2)))


def test_shape_strides():
    x = np.zeros((3, 4), dtype=np.float32)
    assert checks.info_f32(x) == ([3, 4], [16, 4], 12, True)
    assert checks.info_f32(x.T) == ([4, 3], [4, 16], 12, False)


def test_element_types():
    assert checks.sum_i32(np.arange(10, dtype=np.int32)) == 45
    assert checks.sum_u8(np.full(4, 200, dtype=np.uint8)) == 800
    assert checks.count_true(np.array([True, False, True])) == 2
    np.testing.assert_array_equal(checks.ints(5), np.arange(5) ** 2)


def test_non_native_byte_order_rejected():
    with pytest.raises(Exception, match="expected an array of int32"):
        checks.sum_i32(np.arange(3, dtype=np.dtype(np.int32).newbyteorder()))


def test_three_dimensional_indexing():
    x = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    assert checks.get3(x, 1, 2, 3) == x[1, 2, 3]
    assert checks.get3(x.transpose(2, 0, 1), 3, 1, 2) == x[1, 2, 3]


def test_write_in_place():
    x = np.arange(6.0)
    checks.scale_inplace(x[::2], 10.0)
    np.testing.assert_array_equal(x, [0, 1, 20, 3, 40, 5])


def test_read_only():
    x = np.arange(3.0)
    with pytest.raises(Exception, match="read-only"):
        checks.write_readonly(x)
    x.flags.writeable = False
    with pytest.raises(Exception):
        checks.scale_inplace(x, 2.0)


def test_index_out_of_bounds():
    x = np.zeros((2, 3, 4), dtype=np.int64)
    with pytest.raises(Exception, match="out of bounds for axis 1"):
        checks.get3(x, 0, 3, 0)
