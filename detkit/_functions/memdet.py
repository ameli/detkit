# SPDX-FileCopyrightText: Copyright 2022, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =======
# Imports
# =======

import os
import numpy
import scipy
import tempfile
from .memory import Memory
from ._parallel_io import load, store
from multiprocessing import shared_memory
import inspect
from .._cy_linear_algebra import fill_triangle
from .._openmp import get_avail_num_threads
import time
import shutil
import zarr
import dask
from ..__version__ import __version__
from ._utilities import get_processor_name
import tensorstore

__all__ = ['memdet']


# ============
# get dir size
# ============

def _get_dir_size(path):
    """
    get the size of a director.
    """

    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)

    return total_size


# ==================
# get scratch prefix
# ==================

def _get_scratch_prefix():
    """
    Prefix for filename of scratch space. The prefix is the combination of
    package name and function name.
    """

    # Get the name of caller function
    stack = inspect.stack()
    caller_frame = stack[1]
    caller_function_name = caller_frame.function

    # Get the name of package
    frame = inspect.currentframe()
    module_name = frame.f_globals['__name__']
    package_name = module_name.split('.')[0]

    # scratch space filename prefix
    prefix = '.' + package_name + '-' + caller_function_name + '-'

    return prefix


# =========
# get array
# =========

def _get_array(shared_mem, shape, dtype, order):
    """
    Get numpy array from shared memory buffer.
    """

    if len(shape) != 2:
        raise ValueError('"shape" should have length of two.')

    if isinstance(shared_mem, shared_memory.SharedMemory):
        # This is shared memory. Return its buffer.
        return numpy.ndarray(shape=shape, dtype=dtype, order=order,
                             buffer=shared_mem.buf)

    else:
        # This is already numpy array. Return itself.
        return shared_mem


# ====================
# pivot to permutation
# ====================

def _pivot_to_permutation(piv):
    """
    Convert pivot of indices to permutation of indices.
    """

    perm = numpy.arange(len(piv))
    for i in range(len(piv)):
        perm[i], perm[piv[i]] = perm[piv[i]], perm[i]

    return perm


# ==================
# permutation parity
# ==================

def _permutation_parity(p_inv):
    """
    Compute the parity of a permutation represented by the pivot array `piv`.

    Parameters
    ----------

    piv (array_like): The pivot array returned by `scipy.linalg.lu_factor`.

    Returns
    -------
    int: The parity of the permutation (+1 or -1).
    """

    n = len(p_inv)
    visited = numpy.zeros(n, dtype=bool)
    parity = 1

    for i in range(n):
        if not visited[i]:
            j = i
            while not visited[j]:
                visited[j] = True
                j = p_inv[j]
                if j != i:
                    parity = -parity

    return parity


# =============
# permute array
# =============

def _permute_array(array, perm_inv, shape, dtype, order):
    """
    Permutes rows of 2D array.

    This function overwrites the input array. Note that this function creates
    new memory, hence, is not memory efficient.
    """

    # Get buffer from shared memory
    array_ = _get_array(array, shape, dtype, order)
    array_copy = numpy.copy(array_, order=order)
    array_[perm_inv, :] = array_copy[:, :]


# =====
# shift
# =====

def _shift(perm, shift):
    """
    Shifts a slice or permutation array.
    """

    if isinstance(perm, numpy.ndarray):
        shifted_perm = perm + shift
    elif isinstance(perm, slice):
        start = perm.start + shift
        stop = perm.stop + shift
        step = perm.step
        shifted_perm = slice(start, stop, step)
    else:
        raise ValueError('"perm" type is not recognized.')

    return shifted_perm


# =========
# lu factor
# =========

def _lu_factor(A, shape, dtype, order, overwrite, verbose=False):
    """
    Performs LU factorization of an input matrix.
    """

    if verbose:
        print('lu decompos ... ', end='', flush=True)

    # Get buffer from shared memory
    A_ = _get_array(A, shape, dtype, order)

    lu, piv = scipy.linalg.lu_factor(A_, overwrite_a=overwrite,
                                     check_finite=False)

    if verbose:
        print('done', flush=True)

    return lu, piv


# ================
# solve triangular
# ================

def _solve_triangular(lu, B, shape, dtype, order, trans, lower, unit_diagonal,
                      overwrite, verbose=False):
    """
    Solve triangular system of equations.
    """

    if verbose:
        if lower:
            print('solve lotri ... ', end='', flush=True)
        else:
            print('solve uptri ... ', end='', flush=True)

    # Get buffer from shared memory
    B_ = _get_array(B, shape, dtype, order)

    x = scipy.linalg.solve_triangular(lu, B_, trans=trans, lower=lower,
                                      unit_diagonal=unit_diagonal,
                                      check_finite=False,
                                      overwrite_b=overwrite)

    if verbose:
        print('done', flush=True)

    return x


# ================
# schur complement
# ================

def _schur_complement(L_t, U, S, shape, dtype, order, verbose=False):
    """
    Computes in-place Schur complement without allocating any intermediate
    memory. This method is parallel.

    For this function to not allocate any new memory, all matrices, L, U,
    and S should be in Fortran ordering.
    """

    if verbose:
        print('schur compl ... ', end='', flush=True)

    alpha = -1
    beta = 1
    trans_a = 1
    trans_b = 0
    overwrite_c = 1

    # Get buffer from shared memory
    S_ = _get_array(S, shape, dtype, order)

    # Check all matrices have Fortran ordering
    if not L_t.flags['F_CONTIGUOUS']:
        raise TypeError('Matrix "L" should have column-ordering.')
    if not U.flags['F_CONTIGUOUS']:
        raise TypeError('Matrix "U" should have column-ordering.')
    if not S_.flags['F_CONTIGUOUS']:
        raise TypeError('Matrix "S" should have column-ordering.')

    if numpy.dtype(dtype) == numpy.float64:
        scipy.linalg.blas.dgemm(alpha, L_t, U, beta, S_, trans_a, trans_b,
                                overwrite_c)
    elif numpy.dtype(dtype) == numpy.float32:
        scipy.linalg.blas.sgemm(alpha, L_t, U, beta, S_, trans_a, trans_b,
                                overwrite_c)
    else:
        raise TypeError('dtype should be float64 or float32.')

    if verbose:
        print('done', flush=True)


# ======
# memdet
# ======

def memdet(
        A,
        num_blocks=1,
        assume='gen',
        triangle=None,
        overwrite=False,
        mixed_precision='float64',
        parallel_io=None,
        io_chunk=5000,
        scratch_dir=None,
        return_info=False,
        verbose=False):
    """
    Compute log-determinant on contained memory.

    Parameters
    ----------

    A : numpy.ndarray or numpy.memmap or zarr
        Square non-singular matrix.

    num_blocks : int, default=1
        Number of blocks

    triangle : ``'l'``, ``'u'``, or None, default=None
        Indicates the  matrix symmetric, but only half triangle part of the
        matrix is given. ``'l'`` assumes the lower-triangle part of the
        matrix is given, and ``'u'`` assumes the upper-triangle part of the
        matrix is given. `None` indicates all the matrix is given.

    assume : str {``'gen'``, ``'sym'``, ``'spd'``}, default=``'gen'``
        Assumption on the matrix `A`. Matrix is assumed to be generic if
        ``'gen'``, symmetric if ``'sym'``, and symmetric positive-definite if
        ``'psd'``.

    parallel_io : str {'mp', 'dask'} or None, default=None
        Parallel load and store from memory to scratchpad.

    overwrite : boolean, default=True
        Overwrites intermediate computations. May increase performance and
        memory consumption.

    verbose : bool, default=False
        Prints verbose output during computation.

    Returns
    -------

    ld : float
        Log-determinant
    sign : int
        Sign of determinant

    Raises
    ------

    See also
    --------

    detkit.logdet
    detkit.loggdet
    detkit.logpdet

    Notes
    -----

    for dask, make sure to use if-clause protection
    https://pytorch.org/docs/stable/notes/windows.html
    #multiprocessing-error-without-if-clause-protection

    Examples
    --------

    .. code-block:: python
        :emphasize-lines: 9

        >>> # Open a memmmap matrix
        >>> import numpy
        >>> n = 10000
        >>> A = numpy.memmap('matrix.npy', shape=(n, n), mode='r',
        ...                  dtype=numpy.float32, order='C')

        >>> # Compute log-determinant
        >>> from detkit import  memdet
        >>> ld = memdet(A, mem=64)
    """

    n = A.shape[0]
    if mixed_precision is not None:
        dtype = mixed_precision
    else:
        dtype = A.dtype
    order = 'F'

    temp_file = None
    temp_dir = None
    scratch_file = ''
    scratch_nbytes = 0
    num_scratch_blocks = 0

    # Keep time of load and store
    io = {
        'load_wall_time': 0,
        'load_proc_time': 0,
        'store_wall_time': 0,
        'store_proc_time': 0,
        'num_block_loads': 0,
        'num_block_stores': 0,
    }

    # Block size
    m = (n + num_blocks - 1) // num_blocks

    if verbose:
        print(f'matrix size: {n}', flush=True)
        print(f'num blocks: {num_blocks}', flush=True)
        print(f'block size: {m}', flush=True)
        print('dtype: %s' % str(dtype), flush=True)

    # Initialize time and set memory counter
    mem = Memory(chunk=n**2, dtype=dtype, unit='MB')
    mem.set()
    init_wall_time = time.time()
    init_proc_time = time.process_time()

    # Start a Dask client to use multiple threads
    if (parallel_io == 'dask') and (num_blocks > 2):
        client = dask.distributed.Client()
        # lock = dask.utils.SerializableLock()
        lock = False

    block_nbytes = numpy.dtype(dtype).itemsize * (m**2)
    if parallel_io == 'mp':
        A11 = shared_memory.SharedMemory(create=True, size=block_nbytes)
    else:
        A11 = numpy.empty((m, m), dtype=dtype, order=order)

    if verbose:
        print('Allocated A11, %d bytes' % block_nbytes, flush=True)

    # Context for tensorstore
    if parallel_io == 'ts':
        ts_context = tensorstore.Context({
            'cache_pool': {
                'total_bytes_limit': 1_000_000_000_000,
            },
            'data_copy_concurrency': {
                'limit': get_avail_num_threads(),
            }
        })

    # Create dask for input data
    if parallel_io == 'dask':
        if isinstance(A, zarr.core.Array):
            dask_A = dask.array.from_zarr(A, chunks=(io_chunk, io_chunk))
        else:
            dask_A = dask.array.from_array(A, chunks=(io_chunk, io_chunk))
    elif parallel_io == 'ts':

        if isinstance(A, zarr.core.Array):
            spec_1 = {
                'driver': 'zarr',
                'kvstore': {
                    'driver': 'file',
                    'path': A.store.path,
                }
            }

            ts_A = tensorstore.open(spec_1, context=ts_context).result()
        else:
            raise RuntimeError('The "ts" parallel io can be used only for ' +
                               'zarr arrays.')

    if num_blocks > 1:

        if parallel_io == 'mp':
            A12 = shared_memory.SharedMemory(create=True, size=block_nbytes)
            A21_t = shared_memory.SharedMemory(create=True, size=block_nbytes)
        else:
            A12 = numpy.empty((m, m), dtype=dtype, order=order)
            A21_t = numpy.empty((m, m), dtype=dtype, order=order)

        if verbose:
            print('Allocated A12, %d bytes' % block_nbytes, flush=True)
            print('Allocated A21, %d bytes' % block_nbytes, flush=True)

        if num_blocks > 2:

            num_scratch_blocks = num_blocks * (num_blocks - 1) - 1

            if parallel_io == 'mp':
                A22 = shared_memory.SharedMemory(create=True,
                                                 size=block_nbytes)
            else:
                A22 = numpy.empty((m, m), dtype=dtype, order=order)

            if verbose:
                print('Allocated A22, %d bytes' % block_nbytes, flush=True)

            # Scratch space to hold temporary intermediate blocks
            if parallel_io == 'mp':

                # Temporary file as scratch space
                temp_file = tempfile.NamedTemporaryFile(
                        prefix=_get_scratch_prefix(), suffix='.npy',
                        delete=True, dir=scratch_dir)
                scratch_file = temp_file.name

                scratch = numpy.memmap(temp_file.name, dtype=dtype, mode='w+',
                                       shape=(n, n-m), order=order)

            else:
                # Temporary directory as scratch space
                temp_dir = tempfile.mkdtemp(prefix=_get_scratch_prefix(),
                                            suffix='.zarr', dir=scratch_dir)

                scratch_file = temp_dir

                scratch = zarr.open(temp_dir, mode='w', shape=(n, n-m),
                                    dtype=dtype, order=order)

                if parallel_io == 'dask':
                    dask_scratch = dask.array.from_zarr(
                            scratch, chunks=(io_chunk, io_chunk))
                elif parallel_io == 'ts':

                    spec_2 = {
                        'driver': 'zarr',
                        'kvstore': {
                            'driver': 'file',
                            'path': scratch.store.path,
                        }
                    }

                    # Open the Zarr array using tensorstore
                    ts_scratch = tensorstore.open(
                            spec_2, context=ts_context).result()

            if verbose:
                print('created scratch space: %s.' % scratch_file, flush=True)

            # Cache table flagging which block is moved to scratch space. False
            # means the block is not yet on scratch space, True means it is
            # cached in the scratch space
            cached = numpy.zeros((num_blocks, num_blocks), dtype=bool)

    alloc_mem = mem.now()
    alloc_mem_peak = mem.peak()

    # ----------
    # load block
    # ----------

    def _load_block(array, i, j, trans=False, perm=None):
        """
        If triangle is 'l' or 'u', it replicates the other half of the
        triangle only if reading the original data from the input matrix. But
        when loading from the scratch space, it does not replicate the other
        half. This is because when data are stored to scratch space, all matrix
        is stored, not just a half triangle of it. Hence its loading should be
        full.

        perm_inv is the inverse permutation of perm. The operation of
        A[:, :] = B[perm, :] is equivalent to A[perm_inv, :] = B[:, :].
        However, the latter creates additional memory of the same size as the
        original matrix, and is much slower. This is because numpy copies a
        slice and then permutes it. However, the first operation with perm_inv
        does not create any additional memory and is very fast.
        """

        io['num_block_loads'] += 1

        # Initialize load times
        init_load_wall_time = time.time()
        init_load_proc_time = time.process_time()

        if verbose:
            print('loading blk ... ', end='', flush=True)

        if (num_blocks > 2) and (bool(cached[i, j]) is True):
            read_from_scratch = True
        else:
            read_from_scratch = False

        if ((not read_from_scratch) and
            (((triangle == 'l') and (i < j)) or
             ((triangle == 'u') and (i > j)))):
            i_ = j
            j_ = i
            trans = numpy.logical_not(trans)
        else:
            i_ = i
            j_ = j

        i1 = m*i_
        if i_ == num_blocks-1:
            i2 = n
        else:
            i2 = m*(i_+1)

        j1 = m*j_
        if j_ == num_blocks-1:
            j2 = n
        else:
            j2 = m*(j_+1)

        # Permutation of rows
        if perm is not None:

            # Row orders with permutation. perm_inv is the inverse of
            # permutation to be applied on the target array, rather than the
            # source array, while perm is applied to the source array. In the
            # case of transpose, perm is applied to the columns of the source
            # array, which is equivalent of taking the transpose first, then
            # apply perm on the rows.
            perm = numpy.array(perm)
            perm_inv = numpy.argsort(perm)

            if perm.ndim != 1:
                raise ValueError('"perm" should be a 1D array.')
            elif (((trans is False) and (perm.size != i2-i1)) or
                  (trans is True) and (perm.size != j2-j1)):
                raise ValueError('"perm" size does not match the slice size')

            # When using perm on the source array, the indices should be
            # shifted to start from the beginning of the block, but this is not
            # necessary for perm_inv on target array.
            if trans:
                perm = _shift(perm, j1)
            else:
                perm = _shift(perm, i1)

        else:
            # Rows order with no permutation.
            # Note: do not use numpy.arange(0, i2-i1) as this is much slower
            # and takes much more memory than slice(0, i2-i1)
            if trans:
                perm = slice(j1, j2)
                perm_inv = slice(0, j2-j1)
            else:
                perm = slice(i1, i2)
                perm_inv = slice(0, i2-i1)

        if read_from_scratch:

            if parallel_io == 'mp':
                # Read using multiprocessing
                load(scratch, (i1, i2), (j1-m, j2-m), array, (m, m), order,
                     trans, perm_inv, num_proc=None)
            else:
                # Get buffer from shared memory
                array_ = _get_array(array, (m, m), dtype, order)

                if parallel_io == 'dask':
                    # Read using dask
                    if trans:
                        with dask.config.set(scheduler='threads'):
                            dask.array.store(
                                    dask_scratch[i1:i2, (j1-m):(j2-m)].T,
                                    array_, lock=lock)
                    else:
                        with dask.config.set(scheduler='threads'):
                            dask.array.store(
                                    dask_scratch[i1:i2, (j1-m):(j2-m)],
                                    array_, lock=lock)

                    # Dask cannot do permutation within store function. Do it
                    # here manually
                    if isinstance(perm, numpy.ndarray):
                        _permute_array(array_, perm_inv, (m, m), dtype, order)

                elif parallel_io == 'ts':
                    # Read using tensorstore
                    # For ts mode, when source is 'C' order and target is 'F'
                    # order, using perm on source array is faster than
                    # using perm_inv on target array. But, if source and target
                    # have the same ordering, either perm or perm_inv have the
                    # same performance. Here, array_ and scratch are both 'F'
                    # ordering, so using either perm and perm_inv are fine.
                    if trans:
                        # Using perm in columns of source when transposing.
                        array_[:, :] = \
                            ts_scratch[i1:i2, _shift(perm, -m)].T.read(
                                order=order).result()
                    else:
                        array_[:, :] = ts_scratch[perm, (j1-m):(j2-m)].read(
                                order=order).result()

                else:
                    # Read using numpy. Here, using perm_inv on target array is
                    # faster.
                    if trans:
                        array_[perm_inv, :] = scratch[i1:i2, (j1-m):(j2-m)].T
                    else:
                        array_[perm_inv, :] = scratch[i1:i2, (j1-m):(j2-m)]

        else:
            # Reading from input array A (not from scratch)
            if (parallel_io == 'mp') and isinstance(A, numpy.memmap):
                # Read using multiprocessing
                load(A, (i1, i2), (j1, j2), array, (m, m), order, trans,
                     perm_inv, num_proc=None)
            else:

                # Get buffer from shared memory
                array_ = _get_array(array, (m, m), dtype, order)

                if parallel_io == 'dask':
                    # Read using dask
                    if trans:
                        with dask.config.set(scheduler='threads'):
                            dask.array.store(dask_A[i1:i2, j1:j2].T, array_,
                                             lock=lock)
                    else:
                        with dask.config.set(scheduler='threads'):
                            dask.array.store(dask_A[i1:i2, j1:j2], array_,
                                             lock=lock)

                    # Dask cannot do permutation within store function. Do it
                    # here manually
                    if isinstance(perm, numpy.ndarray):
                        _permute_array(array_, perm_inv, (m, m), dtype, order)

                elif parallel_io == 'ts':
                    # Read using tensorstore
                    # For ts mode, when source is 'C' order and target is 'F'
                    # order, using perm on source array is faster than
                    # using perm_inv on target array. But, if source and target
                    # have the same ordering, either perm or perm_inv have the
                    # same performance. Here, array_ is 'F' ordering while
                    # ts_A is 'C' ordering, so using perm is preferred.
                    if trans:
                        # Using perm in columns of source when transposing.
                        array_[:, :] = \
                            ts_A[i1:i2, perm].T.read(order=order).result()
                    else:
                        array_[:, :] = ts_A[perm, j1:j2].read(
                                order=order).result()

                else:
                    # Read using numpy. Here, using perm_inv on target array is
                    # faster.
                    if trans:
                        array_[perm_inv, :] = A[i1:i2, j1:j2].T
                    else:
                        array_[perm_inv, :] = A[i1:i2, j1:j2]

        # Fill the other half of diagonal blocks (if input data is triangle)
        if (i == j) and (triangle is not None) and (not read_from_scratch):

            # Get buffer from shared memory
            array_ = _get_array(array, (m, m), dtype, order)

            if (triangle == 'l'):
                lower = True
            else:
                lower = False

            fill_triangle(array_, lower)

        # load times
        io['load_wall_time'] += time.time() - init_load_wall_time
        io['load_proc_time'] += time.process_time() - init_load_proc_time

        if verbose:
            print('done', flush=True)

    # -----------
    # store block
    # -----------

    def _store_block(array, i, j, flush=True):
        """
        Store array to scratch space.
        """

        io['num_block_stores'] += 1

        # Initialize store times
        init_store_wall_time = time.time()
        init_store_proc_time = time.process_time()

        if verbose:
            print('storing blk ... ', end='', flush=True)

        i1 = m*i
        if i == num_blocks-1:
            i2 = n
        else:
            i2 = m*(i+1)

        j1 = m*j
        if j == num_blocks-1:
            j2 = n
        else:
            j2 = m*(j+1)

        if parallel_io == 'mp':
            # Write in parallel
            trans = False
            store(scratch, (i1, i2), (j1-m, j2-m), array, (m, m), order, trans,
                  num_proc=None)
        else:
            # Get buffer from shared memory
            array_ = _get_array(array, (m, m), dtype, order)

            if parallel_io == 'dask':
                with dask.config.set(scheduler='threads'):
                    dask_array = dask.array.from_array(
                            array_, chunks=(io_chunk, io_chunk))
                    scratch[i1:i2, (j1-m):(j2-m)] = dask_array.compute()
            elif parallel_io == 'ts':
                ts_scratch[i1:i2, (j1-m):(j2-m)].write(array_).result()
            else:
                scratch[i1:i2, (j1-m):(j2-m)] = array_

        # Cache table to flag the block is now written to scratch space, so
        # next time, in order to access the block, scratch space should be
        # read, rather than the input matrix.
        cached[i, j] = True

        if flush and isinstance(scratch, numpy.memmap):
            scratch.flush()

        # store times
        io['store_wall_time'] += time.time() - init_store_wall_time
        io['store_proc_time'] += time.process_time() - init_store_proc_time

        if verbose:
            print('done', flush=True)

    # ------

    try:

        # Output, this will accumulate logdet of each diagonal block
        ld = 0
        sign = 1
        diag = []
        counter = 0
        total_count = (num_blocks-1) * num_blocks * (2*num_blocks+-1) // 6

        # Diagonal iterations
        for k in range(num_blocks):

            if k == 0:
                _load_block(A11, k, k)

            lu_11, piv = _lu_factor(A11, (m, m), dtype, order, overwrite,
                                    verbose=verbose)

            # log-determinant
            diag_lu_11 = numpy.diag(lu_11)
            ld += numpy.sum(numpy.log(numpy.abs(diag_lu_11)))

            # Sign of determinant
            perm = _pivot_to_permutation(piv)
            parity = _permutation_parity(perm)
            sign *= numpy.prod(numpy.sign(diag_lu_11)) * parity

            # Save diagonals
            diag.append(numpy.copy(diag_lu_11))

            # Row iterations
            for i in range(num_blocks-1, k, -1):

                _load_block(A21_t, i, k, trans=True)

                # Solve upper-triangular system
                l_21_t = _solve_triangular(lu_11, A21_t, (m, m), dtype, order,
                                           trans='T', lower=False,
                                           unit_diagonal=False,
                                           overwrite=overwrite,
                                           verbose=verbose)

                if (i - k) % 2 == 0:
                    # Start space-filling curve in a forward direction in the
                    # last row
                    j_start = k+1
                    j_end = num_blocks
                    j_step = +1
                else:
                    # start space-filling curve in a backward direction in the
                    # last row
                    j_start = num_blocks-1
                    j_end = k
                    j_step = -1

                # Column iterations
                for j in range(j_start, j_end, j_step):

                    # When the space-filling curve changes direction, do not
                    # read new A12, rather use the previous matrix already
                    # loaded to memory
                    if ((i == num_blocks-1) or (j != j_start) or
                            (overwrite is False)):
                        # _load_block(A12, k, j)
                        if i == num_blocks-1:
                            _load_block(A12, k, j, perm=perm)
                        else:
                            _load_block(A12, k, j, perm=None)

                    if i == num_blocks-1:

                        # Permute A12
                        # _permute_array(A12, perm, (m, m), dtype, order)

                        # Solve lower-triangular system
                        u_12 = _solve_triangular(
                                lu_11, A12, (m, m), dtype, order, trans='N',
                                lower=True, unit_diagonal=True,
                                overwrite=overwrite,
                                verbose=verbose)

                        # Check u_12 is actually overwritten to A12
                        if overwrite:
                            A_12_array = _get_array(A12, (m, m), dtype, order)
                            if not numpy.may_share_memory(u_12, A_12_array):
                                raise RuntimeError(
                                    '"A12" is not overwritten to "u_12".')

                        if num_blocks > 2:
                            # Store u_12, which is the same as A12 since u_12
                            # is overwritten to A_12. For this to always be the
                            # case, make sure overwrite is set to True in
                            # computing u_12.
                            if overwrite is True:
                                _store_block(A12, k, j)
                            else:
                                _store_block(u_12, k, j)
                    else:
                        u_12 = _get_array(A12, (m, m), dtype, order)

                    # Compute Schur complement
                    if (i == k+1) and (j == k+1):
                        _load_block(A11, i, j)
                        _schur_complement(l_21_t, u_12, A11, (m, m), dtype,
                                          order, verbose=verbose)
                    else:
                        _load_block(A22, i, j)
                        _schur_complement(l_21_t, u_12, A22, (m, m), dtype,
                                          order, verbose=verbose)
                        _store_block(A22, i, j)

                    counter += 1
                    if verbose:
                        print(f'progress: {counter:>3d}/{total_count:>3d}, ',
                              end='', flush=True)
                        print(f'diag: {k+1:>2d}, ', end='', flush=True)
                        print(f'row: {i+1:>2d}, ', end='', flush=True)
                        print(f'col: {j+1:>2d}', flush=True)

        # concatenate diagonals of blocks of U
        diag = numpy.concatenate(diag)

        # record time
        tot_wall_time = time.time() - init_wall_time
        tot_proc_time = time.process_time() - init_proc_time

    except Exception as e:
        print('failed')
        raise e

    finally:

        if temp_file is not None:
            scratch_nbytes = os.path.getsize(scratch_file)
            temp_file.close()
        elif temp_dir is not None:
            scratch_nbytes = _get_dir_size(temp_dir)
            shutil.rmtree(temp_dir)

            if verbose:
                print('removed scratch space: %s.' % scratch_file)

        # Free memory
        if ('A11' in locals()) and isinstance(A11, shared_memory.SharedMemory):
            A11.close()
            A11.unlink

        if ('A12' in locals()) and isinstance(A12, shared_memory.SharedMemory):
            A12.close()
            A12.unlink

        if ('A21_t' in locals()) and \
                isinstance(A21_t, shared_memory.SharedMemory):
            A21_t.close()
            A21_t.unlink

        if ('A22' in locals()) and isinstance(A22, shared_memory.SharedMemory):
            A22.close()
            A22.unlink

        # Record total memory consumption since start
        total_mem = mem.now()
        total_mem_peak = mem.peak()

    # Shut down Dask client
    if (parallel_io == 'dask') and (num_blocks > 2):
        client.close()

    if return_info:
        info = {
            'matrix': {
                'dtype': str(A.dtype),
                'matrix_shape': (n, n),
                'triangle': triangle,
                'assume': assume,
            },
            'process': {
                'processor': get_processor_name(),
                'num_proc': get_avail_num_threads(),
                'tot_wall_time': tot_wall_time,
                'tot_proc_time': tot_proc_time,
                'load_wall_time': io['load_wall_time'],
                'load_proc_time': io['load_proc_time'],
                'store_wall_time': io['store_wall_time'],
                'store_proc_time': io['store_proc_time'],
            },
            'block': {
                'block_nbytes': block_nbytes,
                'block_shape': (m, m),
                'matrix_blocks': (num_blocks, num_blocks),
            },
            'scratch': {
                'num_scratch_blocks': num_scratch_blocks,
                'scratch_file': scratch_file,
                'scratch_nbytes': scratch_nbytes,
                'num_block_loads': io['num_block_loads'],
                'num_block_stores': io['num_block_stores'],
            },
            'memory': {
                'alloc_mem': alloc_mem,
                'alloc_mem_peak': alloc_mem_peak,
                'total_mem': total_mem,
                'total_mem_peak': total_mem_peak,
                'mem_unit': '%d bytes' % block_nbytes,
            },
            'solver': {
                'version': __version__,
                'method': 'lu',
                'dtype': str(dtype),
                'order': order,
            }
        }

        return ld, sign, diag, info

    else:

        return ld, sign, diag
