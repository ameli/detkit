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

import numpy
from ._utilities import get_data_type_name
from .sy_loggdet import sy_loggdet
from .._definitions.types cimport LongIndexType, FlagType
from .._c_linear_algebra.c_matrix_functions cimport cMatrixFunctions

# To avoid cython's bug that does not recognizes "long double" in template []
ctypedef long double long_double

__all__ = ['loggdet']


# =======
# loggdet
# =======

def loggdet(
        A,
        X,
        method='legacy',
        sym_pos=False,
        X_orth=False,
        flops=False,
        use_scipy=False):
    """
    Computes the `loggdet` of a matrix.

    The `loggdet` function is defined by

    .. math::

        \\mathrm{loggdet}(\\mathbf{A}, \\mathbf{X}) :=
        \\log_{e} |\\mathrm{gdet}(\\mathbf{A}, \\mathbf{X})|,

    where

    .. math::

        \\mathrm{gdet}(\\mathbf{A}, \\mathbf{X}) := \\det(\\mathbf{A})
        \\det(\\mathbf{X}^{\\intercal} \\mathbf{A}^{-1} \\mathbf{X}),

    and

    * :math:`\\mathbf{A}` is an :math:`n \\times n` invertible matrix.
    * :math:`\\mathbf{X}` is an :math:`n \\times m` full column-rank matrix.

    Parameters
    ----------
    A : (n, n) array_like
        Invertible matrix. The matrix type can be `float32`, `float64`, or
        `float128`. If a matrix of the type `int32` or `int64` is given, the
        type is cast to `float64`.

    X : (n, m) array_like
        Rectangular matrix with full column-rank.

    method : {'legacy', 'proj'}, default='legacy'
        Method of computing, and can be one of `legacy` or `proj`.

        * `'legacy'`: Computes `loggdet` directly by the equation given in
          the above.
        * `'proj'`: Computes 'loggdet' using Bott-Duffin inverse (See [1]_).

    sym_pos : bool, default=False
        If `True`, the matrix `A` is assumed to be symmetric and
        positive-definite (SPD). The computation can be twice as fast as when
        the matrix is not SPD. This function does not verify whether `A` is
        symmetric or positive-definite. This option is applicable when
        ``method='legacy'``.

    X_orth : bool, default=False
        If `True`, the matrix `X` is assumed to have orthogonal columns. The
        computation in this case is faster. This function will not verify
        whether `X` is orthogonal. This option is only applicable when
        ``method='proj'``.

    flops : bool, default=False
        If `True`, the count of the retired hardware instructions is returned
        using `perf` tool. This functionality is only available on Linux
        operating system and only newer models of Intel CPUs. This option is
        only relevant when ``use_scipy=False``.

    use_scipy : bool, default=False
        If `True`, it uses scipy functions which are the wrappers around
        Fortran routines in BLAS and LAPACK. If `False`, it uses a C++
        library developed in this package.

    Returns
    -------
        loggdet : float
            `loggdet` of `A`.

        sign : int
            Sign of `gdet` function and can be ``+1`` for positive or ``-1``
            for negative determinant.

        flops : int
            (returned if ``flops=True`` and ``use_scipy=False``)

            Count of the retired hardware instructions of the processor during
            the runtime of this function. If this feature is not supported on
            the operating system or the processor, `-1` is returned.

    Raises
    ------
        RuntimeError
            Error raised when:

            * ``sym_pos=True`` and matrix `A` is not symmetric
              positive-definite.
            * ``method='legacy'`` and matrix `A` is degenerate.

        ValueError
            Error raised when

            * ``method='legacy'`` and ``X_orth=True``.
            * ``method='proj'`` and ``sym_pos=True``.

    Warnings
    --------

        When ``method='proj'`` and `A` is singular, no error is raised and an
        incorrect value is returned.

    Notes
    -----
        When the method is `legacy`, the `loggdet` function is computed using
        the equation given in the above. However, when the method is set to
        'proj', an alternative formulation is used. Note that:

        * `legacy` method is independent of whether `X` is orthogonal or not,
          thus, cannot take advantage of an orthogonal `X`.
        * `proj` method is independent of whether `A` is symmetric
          positive-definite or not, thus, cannot take advantage of an SPD
          matrix `A`.

        The inner computation of the function `loggdet` applies the following
        algorithms:

        * When ``sym_pos=False``, the `logdet` function is computed using the
          *PLU decomposition* of `A`.
        * When ``sym_pos=True``, the `logdet` function is computed using the
          Cholesky decomposition of `A`.

        This function is not parallelized and does not accept sparse matrices.

        The `loggdet` function is used in the likelihood function of the
        Gaussian process regression (see [2]_).

    See Also
    --------
        logdet : Computes the `logdet` of a matrix.
        logpdet : Log of pseudo-determinant of the precision matrix in Gaussian
            process regression.

    References
    ----------

    .. [1] Ameli, S., Shadden, S. C. (2022) A Singular Woodbury and
           Pseudo-Determinant Matrix Identities and Application to Gaussian
           Process Regression.

    .. [2] Ameli, S., Shadden, S. C. (2022) Noise Estimation in Gaussian
           Process Regression.

    Examples
    --------
    .. code-block:: python

        >>> import numpy
        >>> from detkit import loggdet, orthogonalize

        >>> # Generate a random matrix
        >>> n, m = 1000, 500
        >>> rng = numpy.random.RandomState(0)
        >>> A = rng.rand(n, n)
        >>> X = rng.rand(n, m)

        >>> # Compute loggdet of matrix
        >>> loggdet(A, X)
        (1710.9576831500378, -1)

        >>> # Compute loggdet of a symmetric and positive-definite matrix
        >>> B = A.T @ A
        >>> loggdet(B, X, sym_pos=True)
        (3421.9153663693114, 1)

        >>> # Compute loggdet when X is orthogonal
        >>> orthogonalize(X)
        >>> loggdet(B, X, sym_pos=True, X_orth=True)
        (3421.9153663693114, 1)

        >>> # Compute loggdet of a singular matrix
        >>> A[:, 0] = 0
        >>> loggdet(A, X)
        RuntimeError: LUP factorization failed since matrix "A" is degenerate.

    The count of the hardware instructions is only supported on the Linux
    operating system and on recent Intel processors.

    .. code-block:: python

        >>> # Count the processor instructions
        >>> loggdet(B, X, sym_pos=True, X_orth=True, flops=True)
        (3421.9153663693114, 1)
    """

    # Using scipy
    if use_scipy:
        return sy_loggdet(A, X, method=method, sym_pos=sym_pos, X_orth=X_orth)

    # Check method
    if method not in ['legacy', 'proj']:
        raise ValueError('"method" should be either "legacy" or "proj".')
    elif method == 'legacy':
        method = 0
    elif method == 'proj':
        method = 1

    # X_orth only applicable to proj method
    if X_orth and method == 'legacy':
        raise ValueError('"X_orth=True" can only used in "proj" method.')
    else:
        X_orth = int(X_orth)

    # sym_pos is only applicable to legacy method
    if sym_pos and method == 'proj':
        raise ValueError('"sym_pos=True" can only used in "legacy" method.')
    else:
        sym_pos = int(sym_pos)

    # flops determines whether to compute flops of the algorithm
    flops = int(flops)

    data_type_name = get_data_type_name(A)
    loggdet_, sign, flops_ = pyc_loggdet(A, X, A.shape[0], X.shape[1],
                                         data_type_name, sym_pos, method,
                                         X_orth, flops)

    if flops != 0:
        return loggdet_, sign, flops_
    else:
        return loggdet_, sign


# ===========
# pyc loggdet
# ===========

cpdef pyc_loggdet(
        A,
        X,
        num_rows,
        num_columns,
        data_type_name,
        sym_pos,
        method,
        X_orth,
        flops):
    """
    """

    sign = numpy.array([0], dtype=numpy.int32)
    cdef FlagType[:] mv_sign = sign
    cdef FlagType* c_sign = &mv_sign[0]
    cdef long long flops_ = flops

    if data_type_name == b'float32':
        loggdet_ = pyc_loggdet_float(A, X, num_rows, num_columns, sym_pos,
                                     method, X_orth, c_sign, flops_)
    elif data_type_name == b'float64':
        loggdet_ = pyc_loggdet_double(A, X, num_rows, num_columns, sym_pos,
                                      method, X_orth, c_sign, flops_)
    elif data_type_name == b'float128':
        loggdet_ = pyc_loggdet_long_double(A, X, num_rows, num_columns,
                                           method, X_orth, sym_pos, c_sign,
                                           flops_)
    else:
        raise TypeError('Data type should be "float32", "float64", or ' +
                        '"float128".')

    if (sign[0] == -3):
        message = 'Cholesky decomposition failed since matrix "A" is not ' + \
                  'symmetric positive-definite.'
        if sym_pos:
            message += ' Set "sym_pos" to False.'
        raise RuntimeError(message)
    elif (sign[0] == -4):
        raise RuntimeError('LUP factorization failed since matrix "A" is ' +
                           'degenerate.')
    elif sign[0] == -2:
        raise RuntimeError('Matrix "A" is degenerate.')
    elif sign[0] == 2:
        raise RuntimeError('Matrix "A" is degenerate.')

    return loggdet_, sign[0], flops_


# =================
# pyc loggdet float
# =================

cdef float pyc_loggdet_float(
        float[:, ::1] A,
        float[:, ::1] X,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        const FlagType sym_pos,
        const FlagType method,
        const FlagType X_orth,
        FlagType* sign,
        long long& flops) except *:
    """
    """

    # Get c-pointer from memoryviews
    cdef float* c_A = &A[0, 0]
    cdef float* c_X = &X[0, 0]

    # Compute loggdet
    cdef float loggdet_ = cMatrixFunctions[float].loggdet(
            c_A, c_X, num_rows, num_columns, sym_pos, method, X_orth, sign[0],
            flops)

    return loggdet_


# ==================
# pyc loggdet double
# ==================

cdef double pyc_loggdet_double(
        double[:, ::1] A,
        double[:, ::1] X,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        const FlagType sym_pos,
        const FlagType method,
        const FlagType X_orth,
        FlagType* sign,
        long long& flops) except *:
    """
    """

    # Get c-pointer from memoryviews
    cdef double* c_A = &A[0, 0]
    cdef double* c_X = &X[0, 0]

    # Compute loggdet
    cdef double loggdet_ = cMatrixFunctions[double].loggdet(
            c_A, c_X, num_rows, num_columns, sym_pos, method, X_orth, sign[0],
            flops)

    return loggdet_


# =======================
# pyc loggdet long double
# =======================

cdef long double pyc_loggdet_long_double(
        long double[:, ::1] A,
        long double[:, ::1] X,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        const FlagType sym_pos,
        const FlagType method,
        const FlagType X_orth,
        FlagType* sign,
        long long& flops) except *:
    """
    """

    # Get c-pointer from memoryviews
    cdef long double* c_A = &A[0, 0]
    cdef long double* c_X = &X[0, 0]

    # Compute loggdet
    cdef long double loggdet_ = cMatrixFunctions[long_double].loggdet(
            c_A, c_X, num_rows, num_columns, sym_pos, method, X_orth, sign[0],
            flops)

    return loggdet_
