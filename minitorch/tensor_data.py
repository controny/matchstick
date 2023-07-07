from __future__ import annotations
import random
from .operators import prod
from numpy import array, float64, ndarray
import numba
from typing import Tuple
from numpy.typing import ArrayLike
import os

MAX_DIMS = 32


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass


def index_to_position(index: ArrayLike, strides: ArrayLike) -> int:
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index (array-like): index tuple of ints
        strides (array-like): tensor strides

    Returns:
        int : position in storage
    """
    pos = 0
    for i in range(len(strides)):
        pos += index[i] * strides[i]
    return pos


def to_index(ordinal: int, shape: Tuple[int], out_index: ArrayLike) -> None:
    """
    Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.
    Similar to the algorithm of converting decimal to binary.

    Args:
        ordinal (int): ordinal position to convert.
        shape (tuple): tensor shape.
        out_index (array): the index corresponding to position.

    Returns:
      None : Fills in `out_index`.

    """
    for i in range(len(shape) - 1, -1 ,-1):
        dim = shape[i]
        cur_idx = ordinal % dim
        out_index[i] = cur_idx
        ordinal = int(ordinal / dim)


def broadcast_index(big_index, big_shape, shape, out_index):
    """
    Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Example 1:
        5 x 3 <- 5 x 1
    Example 2:
        2 x 5 x 1 <- 1 x 5 x 1 <- 5 x 1

    Args:
        big_index (array-like): multidimensional index of bigger tensor
        big_shape (array-like): tensor shape of bigger tensor
        shape (array-like): tensor shape of smaller tensor
        out_index (array-like): multidimensional index of smaller tensor

    Returns:
        None : Fills in `out_index`.
    """
    big_num_dim = len(big_shape)
    small_num_dim = len(shape)
    num_dim_diff = big_num_dim - small_num_dim
    # remove extra dimensions for bigger shape
    big_shape = big_shape[num_dim_diff:]
    big_index = big_index[num_dim_diff:]
    for i in range(small_num_dim):
        if big_shape[i] == shape[i]:
            out_index[i] = big_index[i]
        else:
            assert shape[i] == 1, 'Only dimension of size 1 can be broadcast'
            out_index[i] = 0


def shape_broadcast(shape1: Tuple, shape2: Tuple):
    """
    Broadcast two shapes to create a new union shape.

    Args:
        shape1 (tuple) : first shape
        shape2 (tuple) : second shape

    Returns:
        tuple : broadcasted shape

    Raises:
        IndexingError : if cannot broadcast
    """
    # add extra dimension of 1
    len1 = len(shape1)
    len2 = len(shape2)
    if len1 < len2:
        shape1 = tuple([1] * (len2 - len1) + list(shape1))
    else:
        shape2 = tuple([1] * (len1 - len2) + list(shape2))
    out_shape = []
    for dim1, dim2 in zip(shape1, shape2):
        if dim1 == dim2:
            out_dim = dim1
        elif dim1 == 1:
            out_dim = dim2
        elif dim2 == 1:
            out_dim = dim1
        else:
            raise IndexingError(f'Fail to broadcast shape {shape1} and {shape2}')
        out_shape.append(out_dim)
    return tuple(out_shape)


def strides_from_shape(shape):
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    def __init__(self, storage, shape, strides=None):
        if isinstance(storage, ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self):  # pragma: no cover
        if os.environ.get('NUMBA_ENABLE_CUDASIM') == '1' or not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self):
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a, shape_b):
        return shape_broadcast(shape_a, shape_b)

    def index(self, index):
        if isinstance(index, int):
            index = array([index])
        if isinstance(index, tuple):
            index = array(index)

        # Check for errors
        if index.shape[0] != len(self.shape):
            raise IndexingError(f"Index {index} must be size of {self.shape}.")
        for i, ind in enumerate(index):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {index} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {index} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self):
        lshape = array(self.shape)
        out_index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self):
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key):
        return self._storage[self.index(key)]

    def set(self, key, val):
        self._storage[self.index(key)] = val

    def tuple(self):
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """
        Permute the dimensions of the tensor.

        Args:
            order (list): a permutation of the dimensions

        Returns:
            :class:`TensorData`: a new TensorData with the same storage and a new dimension order.
        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"
        new_shape = list()
        new_strides = list()
        for idx in order:
            new_shape.append(self._shape[idx])
            new_strides.append(self._strides[idx])
        return TensorData(self._storage, tuple(new_shape), tuple(new_strides))

    def to_string(self):
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
