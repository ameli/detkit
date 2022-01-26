#! /usr/bin/env python

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

import sys
import numpy
from detkit import get_instructions_per_task


# ======================
# test get inst per task
# ======================

def test_get_inst_per_task(argv):
    """
    """

    inst_per_task = get_instructions_per_task()
    print('instructions per task: %0.3f' % inst_per_task)


# ===========
# Script main
# ===========

if __name__ == "__main__":
    test_get_inst_per_task(sys.argv)
