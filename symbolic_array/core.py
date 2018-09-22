import uuid
import operator
import numpy as np
from dask.utils import funcname
import dask.array as da
import dask.local


class Array:
    """ Symbolic array superclass """
    def __init__(self, children):
        self.children = children

    @property
    def child(self):
        [child] = self.children
        return child

    def __getitem__(self, index):
        return GetItem(self, index)

    def __add__(self, other):
        return Call([self, other], operator.add)

    def __mul__(self, other):
        return Call([self, other], operator.mul)

    def __pow__(self, other):
        return Call([self, other], operator.pow)

    def max(self, axis=None, keepdims=False):
        return Call([self], np.max, axis=axis, keepdims=keepdims)

    def sum(self, axis=None, keepdims=False):
        return Call([self], np.sum, axis=axis, keepdims=keepdims)

    @property
    def meta(self):
        children = [child.meta if isinstance(child, Array) else child for child in self.children]
        return self.execute(*children)

    @property
    def shape(self):
        return self.meta.shape

    @property
    def dtype(self):
        return self.meta.dtype

    def __array_ufunc__(self, numpy_ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            return Call(inputs, numpy_ufunc, **kwargs)
        else:
            return NotImplemented


class Leaf(Array):
    """ Leaf of the symbolic expression """
    def __init__(self, shape, dtype, name):
        self._meta = da.zeros(shape=shape, dtype=dtype, chunks=-1, name=name)
        super(Leaf, self).__init__([])

    @property
    def meta(self):
        return self._meta

    def __repr__(self):
        return self._meta.name


class GetItem(Array):
    def __init__(self, child, index):
        self.index = index
        super(GetItem, self).__init__([child])

    def execute(self, *children):
        [child] = children
        return child[self.index]

    def __repr__(self):
        return '%s[%s]' % (self.child, ', '.join(map(repr, self.index)))


class Call(Array):
    def __init__(self, children, func, **kwargs):
        self.func = func
        self.kwargs = kwargs
        super(Call, self).__init__(children)

    def execute(self, *children):
        return self.func(*children, **self.kwargs)

    def __repr__(self):
        return '%s(%s)' % (funcname(self.func), ', '.join(map(repr, self.children))
                + (', ' + ', '.join('%s=%s' % (k, v) for k, v in self.kwargs.items())
                    if self.kwargs else ''))


class TOP(Array):
    def __init__(self, output, output_indices, dsk, indices, concatenate,
                 new_axes):
        self.output = output
        self.output_indices = output_indices


def compute(outputs, inputs):
    """ Compute symblic arrays with a fixed set of outputs

    Parameters
    ----------
    outputs: List[Array]
    inputs: List[array-like]

    Examples
    --------
    >>> x = Leaf(shape=(10,), dtype='float64', name='x')
    >>> y = x.sum()


    >>> data = np.ones(10)
    >>> [result] = compute([y], {x: data})
    >>> result
    10
    """
    dsk = dict(inputs)
    stack = list(outputs)
    seen = set()
    while stack:
        k = stack.pop()
        seen.add(k)
        if isinstance(k, Array):
            if k not in dsk:
                dsk[k] = (k.execute,) + tuple(k.children)

            for child in k.children:
                if child not in seen:
                    stack.append(child)

    return dask.local.get_sync(dsk, outputs)
