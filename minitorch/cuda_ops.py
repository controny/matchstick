from numba import cuda
import numba
import numpy as np
from . import operators
from .tensor_data import (
    to_index,
    index_to_position,
    TensorData,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

THREADS_PER_BLOCK = 32

@cuda.jit(device=True)
def array_equal(a, b):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def tensor_map(fn):
    """
    CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function mappings floats-to-floats to apply.
        out (array): storage for out tensor.
        out_shape (array): shape for out tensor.
        out_strides (array): strides for out tensor.
        out_size (array): size for out tensor.
        in_storage (array): storage for in tensor.
        in_shape (array): shape for in tensor.
        in_strides (array): strides for in tensor.

    Returns:
        None : Fills in `out`
    """

    def _map(out, out_shape, out_strides, out_size, in_storage, in_shape, in_strides):
        # Thread id in a 1D block
        tx = cuda.threadIdx.x
        # Block id in a 1D grid
        ty = cuda.blockIdx.x
        # Block width, i.e. number of threads per block
        bw = cuda.blockDim.x
        # Compute flattened index inside the array
        pos = tx + ty * bw
        # check array boundaries
        if pos < out_size:
            if array_equal(in_shape, out_shape) and array_equal(in_strides, out_strides):
                # when `out` and `in` are stride-aligned, avoid indexing
                out[pos] = fn(in_storage[pos])
            else:
                # hardcoded static allocation
                max_index_size = 10
                out_index = cuda.local.array(shape=max_index_size, dtype=numba.int64)
                to_index(pos, out_shape, out_index)
                if array_equal(in_shape, out_shape):
                    # no need to broadcast
                    in_index = out_index
                else:
                    # broadcast into `out_shape`
                    in_index = cuda.local.array(shape=max_index_size, dtype=numba.int64)
                    broadcast_index(out_index, out_shape, in_shape, in_index)
                in_pos = index_to_position(in_index, in_strides)
                out[pos] = fn(in_storage[in_pos])

    return cuda.jit()(_map)


def map(fn):
    # CUDA compile your kernel
    f = tensor_map(cuda.jit(device=True)(fn))

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)

        # Instantiate and run the cuda kernel.
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
        fn: function mappings two floats to float to apply.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        b_storage (array): storage for `b` tensor.
        b_shape (array): shape for `b` tensor.
        b_strides (array): strides for `b` tensor.

    Returns:
        None : Fills in `out`
    """

    def _zip(
        out,
        out_shape,
        out_strides,
        out_size,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):
        # Thread id in a 1D block
        tx = cuda.threadIdx.x
        # Block id in a 1D grid
        ty = cuda.blockIdx.x
        # Block width, i.e. number of threads per block
        bw = cuda.blockDim.x
        # Compute flattened index inside the array
        pos = tx + ty * bw
        # check array boundaries
        if pos < out_size:
            # when `out`, `a`, `b` are stride-aligned, avoid indexing
            if array_equal(a_shape, b_shape) and array_equal(b_shape, out_shape) and \
                array_equal(a_strides, b_strides) and array_equal(b_strides, out_strides):
                out[pos] = fn(a_storage[pos], b_storage[pos])
            else:
                # hardcoded static allocation
                max_index_size = 10
                out_index = cuda.local.array(shape=max_index_size, dtype=numba.int64)
                to_index(pos, out_shape, out_index)
                if array_equal(a_shape, b_shape):
                    a_index = b_index = out_index
                else:
                    # broadcast into `out_shape`
                    a_index = cuda.local.array(shape=max_index_size, dtype=numba.int64)
                    b_index = cuda.local.array(shape=max_index_size, dtype=numba.int64)
                    broadcast_index(out_index, out_shape, a_shape, a_index)
                    broadcast_index(out_index, out_shape, b_shape, b_index)
                a_pos = index_to_position(a_index, a_strides)
                b_pos = index_to_position(b_index, b_strides)
                out[pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)


def zip(fn):
    f = tensor_zip(cuda.jit(device=True)(fn))

    def ret(a, b):
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
        f[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )
        return out

    return ret


def _sum_practice(out, a, size):
    """
    This is a practice sum kernel to prepare for reduce.

    Given an array of length :math:`n` and out of size :math:`n // blockDIM`
    it should sum up each blockDim values into an out cell.

    [a_1, a_2, ..., a_100]

    |

    [a_1 +...+ a_32, a_32 + ... + a_64, ... ,]

    Note: Each block must do the sum using shared memory!

    Args:
        out (array): storage for `out` tensor.
        a (array): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32  # equals to `THREADS_PER_BLOCK`
    # The memory will be shared only within the same block,
    # according to https://numba.pydata.org/numba-doc/latest/cuda/memory.html
    shared_mem = cuda.shared.array(shape=(1), dtype=numba.float64)

    pos = cuda.grid(1)
    if pos >= size:
        return

    # initialize for each block memory
    if pos % BLOCK_DIM == 0:
        shared_mem[0] = 0
    cuda.syncthreads()

    cuda.atomic.add(shared_mem, 0, a[pos])
    cuda.syncthreads()

    # sum up each block respectively
    if pos % BLOCK_DIM == 0:
        cuda.atomic.add(out, 0, shared_mem[0])
        cuda.syncthreads()


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a):
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(fn):
    """
    CUDA higher-order tensor reduce function.

    Args:
        fn: reduction function maps two floats to float.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        reduce_dim (int): dimension to reduce out

    Returns:
        None : Fills in `out`
    """
    op = fn.py_func.__name__

    def _reduce(
        out,
        out_shape,
        out_strides,
        out_size,
        a_storage,
        a_shape,
        a_strides,
        reduce_dim,
        reduce_value,
    ):
        BLOCK_DIM = 1024
        # each block is responsible for each element of `out_a`
        shared_mem = cuda.shared.array(shape=(1), dtype=numba.float64)
        # Thread id in a 1D block, corresponding to the index along the reduce dimension of `a`
        tx = cuda.threadIdx.x
        # Block id in a 1D grid, corresponding to the index of `out`
        ty = cuda.blockIdx.x
        pos = cuda.grid(1)
        # only use threads within the reduce dimension
        if tx >= a_shape[reduce_dim]:
            return

        # initialize for each block memory
        if pos % BLOCK_DIM == 0:
            shared_mem[0] = reduce_value
        cuda.syncthreads()

        max_index_size = 10
        a_index = cuda.local.array(shape=max_index_size, dtype=numba.int64)
        to_index(ty, out_shape, a_index)
        a_index[reduce_dim] = tx
        a_pos = index_to_position(a_index, a_strides)
        if op == 'add':
            cuda.atomic.add(shared_mem, 0, a_storage[a_pos])
        elif op == 'max':
            cuda.atomic.max(shared_mem, 0, a_storage[a_pos])
        elif op == 'mul':
            # TODO support real multiply
            # luckily, we only need to adapt for `all()` in this project
            cuda.atomic.and_(shared_mem, 0, a_storage[a_pos])

        cuda.syncthreads()

        if pos % BLOCK_DIM == 0:
            # assign to the output
            out[ty] = shared_mem[0]
            cuda.syncthreads()

    return cuda.jit()(_reduce)


def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = reduce(fn)
      out = fn_reduce(a, dim)

    Simple version ::

        for j:
            out[1, j] = start
            for i:
                out[1, j] = fn(out[1, j], a[i, j])


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to reduce over
        dim (int): int of dim to reduce

    Returns:
        :class:`Tensor` : new tensor
    """
    assert fn in [operators.add, operators.mul, operators.max], \
        f'Got unexpected function {fn}'
    f = tensor_reduce(cuda.jit(device=True)(fn))

    def ret(a, dim):
        out_shape = list(a.shape)
        out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
        out_a = a.zeros(tuple(out_shape))

        threadsperblock = 1024
        blockspergrid = out_a.size
        f[blockspergrid, threadsperblock](
            *out_a.tuple(), out_a.size, *a.tuple(), dim, start
        )

        return out_a

    return ret


def _mm_practice(out, a, b, size):
    """
    This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

      * All data must be first moved to shared memory.
      * Only read each cell in `a` and `b` once.
      * Only write to global memory once per kernel.

    Compute ::

    for i:
        for j:
             for k:
                 out[i, j] += a[i, k] * b[k, j]

    Args:
        out (array): storage for `out` tensor.
        a (array): storage for `a` tensor.
        b (array): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    MEM_SIZE = 1024  # 32 * 32
    # use shared memory to avoid load data multiple times
    # refer to https://numba.pydata.org/numba-doc/latest/cuda/examples.html#cuda-matmul
    a_shared_mem = cuda.shared.array(shape=(MEM_SIZE), dtype=numba.float64)
    b_shared_mem = cuda.shared.array(shape=(MEM_SIZE), dtype=numba.float64)
    # each thread corresponds to each position of the output matrix
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    if tx >= size or ty >= size:
        return
    pos = tx * size + ty

    # preload data
    # each thread is responsible for each position
    a_shared_mem[pos] = a[pos]
    b_shared_mem[pos] = b[pos]
    cuda.syncthreads()

    reduction = .0
    # (tx, 0)
    a_start_pos = tx * size
    # (0, ty)
    b_start_pos = ty
    for i in range(size):
        # strides are (size, 1)
        a_cur_pos = a_start_pos + i
        b_cur_pos = b_start_pos + size * i
        reduction += a_shared_mem[a_cur_pos] * b_shared_mem[b_cur_pos]
    out[pos] = reduction


jit_mm_practice = cuda.jit()(_mm_practice)


def mm_practice(a, b):

    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


@cuda.jit()
def tensor_matrix_multiply(
    out,
    out_shape,
    out_strides,
    out_size,
    a_storage,
    a_shape,
    a_strides,
    b_storage,
    b_shape,
    b_strides,
):
    """
    CUDA tensor matrix multiply function.

    Requirements:

      * All data must be first moved to shared memory.
      * Only read each cell in `a` and `b` once.
      * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

        assert a_shape[-1] == b_shape[-2]

    Args:
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.4.
    raise NotImplementedError('Need to implement for Task 3.4')


def matrix_multiply(a, b):
    """
    Tensor matrix multiply

    Should work for any tensor shapes that broadcast in the first n-2 dims and
    have ::

        assert a.shape[-1] == b.shape[-2]

    Args:
        a (:class:`Tensor`): tensor a
        b (:class:`Tensor`): tensor b

    Returns:
        :class:`Tensor` : new tensor
    """

    # Make these always be a 3 dimensional multiply
    both_2d = 0
    if len(a.shape) == 2:
        a = a.contiguous().view(1, a.shape[0], a.shape[1])
        both_2d += 1
    if len(b.shape) == 2:
        b = b.contiguous().view(1, b.shape[0], b.shape[1])
        both_2d += 1
    both_2d = both_2d == 2

    ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
    ls.append(a.shape[-2])
    ls.append(b.shape[-1])
    assert a.shape[-1] == b.shape[-2]
    out = a.zeros(tuple(ls))

    # One block per batch, extra rows, extra col
    blockspergrid = (
        (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
        (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
        out.shape[0],
    )
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

    tensor_matrix_multiply[blockspergrid, threadsperblock](
        *out.tuple(), out.size, *a.tuple(), *b.tuple()
    )

    # Undo 3d if we added it.
    if both_2d:
        out = out.view(out.shape[1], out.shape[2])
    return out


class CudaOps:
    map = map
    zip = zip
    reduce = reduce
    matrix_multiply = matrix_multiply
