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

from .._definitions.types cimport LongIndexType


# =======
# Externs
# =======

cdef extern from "c_matrix_decompositions.h":

    cdef cppclass cMatrixDecompositions[DataType]:

        @staticmethod
        void lu(
                DataType* A,
                const LongIndexType num_rows,
                DataType* L) nogil

        @staticmethod
        void cholesky(
                DataType* A,
                const LongIndexType num_rows,
                DataType* L) nogil

        @staticmethod
        void gram_schmidt(
                DataType *A,
                const LongIndexType num_rows,
                const LongIndexType num_columns) nogil
