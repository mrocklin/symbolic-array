from dask.array.utils import assert_eq
import numpy as np
from symbolic_array import Leaf, compute
import pytest
import dask.array as da


@pytest.mark.parametrize('func', [
    lambda x: (x + 1)[::5, :3].sum(axis=0),
    lambda x: x.max(axis=1),
    lambda x: np.sin(x) ** 2 + np.cos(x) ** 2,
])
def test_basic(func):
    x = Leaf(shape=(10, 10), dtype='float32', name='x')

    y = func(x)
    repr(y)
    assert isinstance(y.meta, da.Array)

    data = np.arange(100).reshape((10, 10))

    expected = func(data)
    [result] = compute([y], {x: data})

    assert_eq(result, expected)
