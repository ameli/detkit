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
from .sy_logpdet import sy_logpdet
from .._definitions.types cimport LongIndexType, FlagType
from .._c_linear_algebra.c_matrix_functions cimport cMatrixFunctions

# To avoid cython's bug that does not recognizes "long double" in template []
ctypedef long double long_double

__all__ = ['logpdet']


# =======
# logpdet
# =======

def logpdet(
        A,
        X,
        Xp=None,
        method='legacy',
        sym_pos=False,
        X_orth=False,
        flops=False,
        use_scipy=False):
    """
    Compute the `logpdet` of a matrix.

    The `logpdet` function is defined by

    .. math::

        \\mathrm{logpdet}(\\mathbf{A}, \\mathbf{X}) :=
        \\log_{e} |\\mathrm{pdet}(\\mathbf{A}, \\mathbf{X})|,

    where

    .. math::

        \\mathrm{pdet}(\\mathbf{A}, \\mathbf{X}) :=
        \\det(\\mathbf{X}^{\\intercal} \\mathbf{X}) \\det(\\mathbf{A})
        \\det(\\mathbf{X}^{\\intercal} \\mathbf{A}^{-1} \\mathbf{X}),

    and

    * :math:`\\mathbf{A}` is an :math:`n \\times n` invertible matrix.
    * :math:`\\mathbf{X}` is an :math:`n \\times m` full column-rank matrix.

    The value of `logpdet` is independent of whether :math:`\\mathbf{X}` is
    orthogonal or not.

    Parameters
    ----------
        A : (n, n) array_like
            Invertible matrix. The matrix type can be `float32`, `float64`, or
            `float128`. If a matrix of the type `int32` or `int64` is given,
            the type is cast to `float64`.

        X : (n, m) array_like
            Rectangular matrix with full column-rank.

        Xp : (n, n-m) array_like
            Rectangular matrix with full column-rank. `Xp` is the orthonormal
            complement of `X`. If `None`, this matrix will be generated. Only
            used if `method` is `comp`.

        method : {'legacy', 'proj', 'comp'}, default='legacy'
            Method of computing, and can be one of `legacy` or `proj`.

            * `'legacy'`: Computes `logpdet` directly by the equation given in
              the above.
            * `'proj'`: Computes `logpdet` using Bott-Duffin inverse
              (See [1]_).
            * `'comp'`: Computes `loggdet` using compression matrix
              (See [1]_).

        sym_pos : bool, default=False
            If `True`, the matrix `A` is assumed to be symmetric and
            positive-definite (SPD). The computation can be twice as fast as
            when the matrix is not SPD. This function does not verify whether
            `A` is symmetric or positive-definite. This option is applicable
            when ``method='legacy'`` or ``method='comp'``.

        X_orth : bool, default=False
            If `True`, the matrix `X` is assumed to have orthogonal columns.
            The computation in this case is faster. This function will not
            verify whether `X` is orthogonal. This option is only applicable
            when ``method='proj'`` or ``method='comp'``.

        flops : bool, default=False
            If `True`, the count of the retired hardware instructions is
            returned using `perf` tool. This functionality is only available on
            Linux operating system and only newer models of Intel CPUs. This
            option is only relevant when ``use_scipy=False``.

        use_scipy : bool, default=False
            If `True`, it uses scipy functions which are the wrappers around
            Fortran routines in BLAS and LAPACK. If `False`, it uses a C++
            library developed in this package.

    Returns
    -------
        logpdet : float
            `logpdet` of `A`.

        sign : int
            Sign of `gdet` function and can be ``+1`` for positive or ``-1``
            for negative determinant.

        flops : int
            (returned if ``flops=True`` and ``use_scipy=False``)

            Count of the retired hardware instructions of the processor during
            the runtime of this function.

    Raises
    ------
        RuntimeError
            Error raised when:

            * ``sym_pos=True`` and matrix `A` is not symmetric
              positive-definite.
            * ``method='legacy'`` and matrix `A` is degenerate.
            * ``flops=True`` and either `Perf` tool is not installed (Linux
              only), or the user permission for the `Perf` tool is not set, or
              the performance counter is not supported on the user's CPU.

        ValueError
            Error raised when

            * ``method='proj'`` and ``sym_pos=True``.

    Warnings
    --------

        When ``method='proj'`` and `A` is singular, no error is raised and an
        incorrect value is returned.

    Notes
    -----

        When the method is `legacy`, the `logpdet` function is computed using
        the equation given in the above. However, when the method is set to
        'proj' or 'comp', an alternative formulation is used. Note that:

        * `proj` method is independent of whether `A` is symmetric
          positive-definite or not, thus, cannot take advantage of an SPD
          matrix `A`.
        * The *value* of `logpdet` function is independent of whether `X` is
          orthogonal or not. However, when `X` is orthogonal, the option
          ``X_orth=True`` can enhance the performance of both `proj` and
          `legacy` method.

        The inner computation of the function `logpdet` applies the following
        algorithms:

        * When ``sym_pos=False``, the `logdet` function is computed using the
          *PLU decomposition* of `A`.
        * When ``sym_pos=True``, the `logdet` function is computed using the
          Cholesky decomposition of `A`.

        The `logpdet` function is used in the likelihood function of the
        Gaussian process regression (see [2]_).

        If the compile-time variable ``USE_OPENMP`` is set to ``1``, this
        function is parallelized. The compile-time default is ``0``. To see
        a list of compile-time definitions, see :func:`detkit.get_config`
        function.

        **Counting Flops:**

        When ``flops`` is set to `True`, make sure
        `perf tool <https://perf.wiki.kernel.org/>`__ is installed
        (Linux only). On Ubuntu, install `perf` by

        .. prompt:: bash

            sudo apt-get install linux-tools-common linux-tools-generic \\
                    linux-tools-`uname -r`

        .. prompt:: bash

            sudo sh -c 'echo -1 >/proc/sys/kernel/perf_event_paranoid'

        Test if the `perf` tool works by

        .. prompt::

            perf stat -e instructions:u dd if=/dev/zero of=/dev/null count=1000

        Alternatively, you can test the ``perf`` tool directly with
        :func:`detkit.check_perf_support`:

        .. code-block:: python

            >>> import detkit
            >>> detkit.check_perf_support()

        If the ``perf`` tool is installed and configured properly, the output
        of either of the above commands should be like:

        .. code-block::

            {
                'kernel_version': '6.8.0-51-generic',
                'perf_event_paranoid': 1,
                'perf_installed': True,
                'perf_working': True
            }

    See Also
    --------

    logdet
    loggdet
    memdet

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
        >>> from detkit import logpdet, orthogonalize

        >>> # Generate a random matrix
        >>> n, m = 1000, 500
        >>> rng = numpy.random.RandomState(0)
        >>> A = rng.rand(n, n)
        >>> X = rng.rand(n, m)

        >>> # Compute logpdet of matrix
        >>> logpdet(A, X)
        (-680.9183141420649, -1)

        >>> # Compute logpdet of a symmetric and positive-definite matrix
        >>> B = A.T @ A
        >>> logpdet(B, X, sym_pos=True)
        (-2059.6208046883685, 1)

        >>> # Compute logpdet when X is orthogonal
        >>> orthogonalize(X)
        >>> logpdet(B, X, sym_pos=True, X_orth=True)
        (-2059.6208046883685, 1)

        >>> # Compute logpdet of a singular matrix
        >>> A[:, 0] = 0
        >>> logpdet(A, X)
        RuntimeError: LUP factorization failed since matrix "A" is degenerate.

    The count of the hardware instructions is only supported on the Linux
    operating system and on recent Intel processors.

    .. code-block:: python

        >>> # Count the processor instructions
        >>> logpdet(B, X, sym_pos=True, X_orth=True, flops=True)
        (-2059.6208046883685, 1, 8520537034)
    """

    # Using scipy
    if use_scipy:
        return sy_logpdet(A, X, Xp, method=method, sym_pos=sym_pos,
                          X_orth=X_orth)

    # Check method
    if method not in ['legacy', 'proj', 'comp']:
        raise ValueError('"method" should be either "legacy", "proj", or ' +
                         '"comp".')
    elif method == 'legacy':
        method = 0
    elif method == 'proj':
        method = 1
    elif method == 'comp':
        method = 2

    # X_orth only applicable to proj or comp method
    if X_orth and method == 'legacy':
        raise ValueError('"X_orth=True" can only used in "proj" or "comp" ' +
                         ' methods.')
    else:
        X_orth = int(X_orth)

    # sym_pos is only applicable to legacy or comp method
    if sym_pos and method == 'proj':
        raise ValueError('"sym_pos=True" can only used in "legacy" or ' +
                         '"comp" methods.')
    else:
        sym_pos = int(sym_pos)

    # flops determines whether to compute flops of the algorithm
    flops = int(flops)

    # Use Xp if not None
    if Xp is None:
        use_Xp = 0
    else:
        use_Xp = 1

        if method != 2:
            raise ValueError('"Xp" can be used only if "method" is "comp".')

    data_type_name = get_data_type_name(A)
    logpdet_, sign, flops_ = pyc_logpdet(A, X, Xp, use_Xp, A.shape[0],
                                         X.shape[1], data_type_name, sym_pos,
                                         method, X_orth, flops)
    
    # Check is perf tool is unable to count performance
    if (flops == 1) and (use_scipy == 0) and (flops_ < 0):
        raise RuntimeError('Cannot count flops. Make sure "perf" tool is ' +
                           'installed (Linux only) and user permission is ' +
                           'set. See documentation for this function for ' +
                           'details. Alternatively, set "flops" option to ' +
                           'False.')

    if flops != 0:
        return logpdet_, sign, flops_
    else:
        return logpdet_, sign


# ===========
# pyc logpdet
# ===========

cpdef pyc_logpdet(
        A,
        X,
        Xp,
        use_Xp,
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
        logpdet_ = pyc_logpdet_float(A, X, Xp, use_Xp, num_rows, num_columns,
                                     sym_pos, method, X_orth, c_sign, flops_)
    elif data_type_name == b'float64':
        logpdet_ = pyc_logpdet_double(A, X, Xp, use_Xp, num_rows, num_columns,
                                      sym_pos, method, X_orth, c_sign, flops_)
    elif data_type_name == b'float128':
        logpdet_ = pyc_logpdet_long_double(A, X, Xp, use_Xp, num_rows,
                                           num_columns, method, X_orth,
                                           sym_pos, c_sign, flops_)
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

    return logpdet_, sign[0], flops_


# =================
# pyc logpdet float
# =================

cdef float pyc_logpdet_float(
        float[:, ::1] A,
        float[:, ::1] X,
        float[:, ::1] Xp,
        const FlagType use_Xp,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        const FlagType sym_pos,
        const FlagType method,
        const FlagType X_orth,
        FlagType* sign,
        long long& flops) noexcept nogil:
    """
    """

    # Get c-pointer from memoryviews
    cdef float* c_A = &A[0, 0]
    cdef float* c_X = &X[0, 0]
    cdef float* c_Xp

    if use_Xp == 0:
        c_Xp = NULL
    else:
        c_Xp = &Xp[0, 0]

    # Compute logpdet
    cdef float logpdet_
    with nogil:
        logpdet_ = cMatrixFunctions[float].logpdet(
            c_A, c_X, c_Xp, use_Xp, num_rows, num_columns, sym_pos, method,
            X_orth, sign[0], flops)

    return logpdet_


# ==================
# pyc logpdet double
# ==================

cdef double pyc_logpdet_double(
        double[:, ::1] A,
        double[:, ::1] X,
        double[:, ::1] Xp,
        const FlagType use_Xp,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        const FlagType sym_pos,
        const FlagType method,
        const FlagType X_orth,
        FlagType* sign,
        long long& flops) noexcept nogil:
    """
    """

    # Get c-pointer from memoryviews
    cdef double* c_A = &A[0, 0]
    cdef double* c_X = &X[0, 0]
    cdef double* c_Xp

    if use_Xp == 0:
        c_Xp = NULL
    else:
        c_Xp = &Xp[0, 0]

    # Compute logpdet
    cdef double logpdet_
    with nogil:
        logpdet_ = cMatrixFunctions[double].logpdet(
            c_A, c_X, c_Xp, use_Xp, num_rows, num_columns, sym_pos, method,
            X_orth, sign[0], flops)

    return logpdet_


# =======================
# pyc logpdet long double
# =======================

cdef long double pyc_logpdet_long_double(
        long double[:, ::1] A,
        long double[:, ::1] X,
        long double[:, ::1] Xp,
        const FlagType use_Xp,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        const FlagType sym_pos,
        const FlagType method,
        const FlagType X_orth,
        FlagType* sign,
        long long& flops) noexcept nogil:
    """
    """

    # Get c-pointer from memoryviews
    cdef long double* c_A = &A[0, 0]
    cdef long double* c_X = &X[0, 0]
    cdef long double* c_Xp

    if use_Xp == 0:
        c_Xp = NULL
    else:
        c_Xp = &Xp[0, 0]

    # Compute logpdet
    cdef long double logpdet_
    with nogil:
        logpdet_ = cMatrixFunctions[long_double].logpdet(
            c_A, c_X, c_Xp, use_Xp, num_rows, num_columns, sym_pos, method,
            X_orth, sign[0], flops)

    return logpdet_
