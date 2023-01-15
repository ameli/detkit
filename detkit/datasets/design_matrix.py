# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

import numpy


def design_matrix(
        size=2**9,
        num_basis=2**8,
        orthonormalize=False):
    """
    Generate design matrix.
    """

    X = numpy.zeros((size, num_basis), dtype=float)

    
    for j in range(num_basis)

    return X
