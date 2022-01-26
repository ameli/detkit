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
from .._functions._utilities import get_data_type_name
from .._definitions.types cimport LongIndexType, FlagType
from .._device cimport MatMatBenchmark

# To avoid cython's bug that does not recognizes "long double" in template []
ctypedef long double long_double

__all__ = ['get_instructions_per_task']


# =========================
# get instructions per task
# =========================

cpdef get_instructions_per_task(dtype=numpy.float64):
    """
    Finds how many hardware instructions are used on the current device to
    compute a single flop task.
    """

    n = (1.0 / numpy.linspace(1.0/30.0, 1.0/500.0, 10) + 0.5).astype(int)
    inst_per_task = numpy.zeros((n.size, ))

    dummy_var = numpy.zeros((1,1), dtype=dtype)

    for i in range(n.size):

        # Flops for matrix-matrix multiplication
        matmat_flops = n[i]**3
        inst_per_task[i] = _get_instructions(dummy_var, n[i]) / matmat_flops

        # Negative means the perf_tool is not installed on the linux machine.
        if inst_per_task[i] < 0:
            return numpy.nan

    # Find inst_per_task when n tends to infinity using an exponential model
    # inst_per_task = a/n + b
    coeff = numpy.polyfit(1.0/n, inst_per_task, deg=1)

    # In the limit n=infinity, b is the number of inst_per_task
    inst_per_task_limit = coeff[1]

    return inst_per_task_limit


# ================
# get instructions
# ================

cpdef _get_instructions(
        dummy_var,
        n):
    """
    Finds how many hardware instructions are used on the current device to
    compute a single flop task.
    """

    data_type_name = get_data_type_name(dummy_var)
    cdef long long inst_per_flop = -1;

    if data_type_name == b'float32':
        inst_per_task = get_instructions_float(dummy_var, n)
    elif data_type_name == b'float64':
        inst_per_task = get_instructions_double(dummy_var, n)
    elif data_type_name == b'float128':
        inst_per_task = get_instructions_long_double(dummy_var, n)
    else:
        raise TypeError('Data type should be "float32", "float64", or ' +
                        '"float128".')

    return inst_per_task


# ======================
# get instructions float
# ======================

cpdef get_instructions_float(
        float[:, ::1] dummy_var,
        int n):
    """
    Specialized for float type.
    """

    # Get c-pointer of a 2D array
    cdef float* c_dummy_var = &dummy_var[0, 0]

    inst_per_task = MatMatBenchmark[float].matmat_benchmark(c_dummy_var, n)

    return inst_per_task


# =======================
# get instructions double
# =======================

cpdef get_instructions_double(
        double[:, ::1] dummy_var,
        int n):
    """
    Specialized for double type.
    """

    # Get c-pointer of a 2D array
    cdef double* c_dummy_var = &dummy_var[0, 0]

    inst_per_task = MatMatBenchmark[double].matmat_benchmark(c_dummy_var, n)

    return inst_per_task


# ============================
# get instructions long double
# ============================

cpdef get_instructions_long_double(
        long double[:, ::1] dummy_var,
        int n):
    """
    Specialized for long double type.
    """

    # Get c-pointer of a 2D array
    cdef long double* c_dummy_var = &dummy_var[0, 0]

    inst_per_task = MatMatBenchmark[long_double].matmat_benchmark(
            c_dummy_var, n)

    return inst_per_task
