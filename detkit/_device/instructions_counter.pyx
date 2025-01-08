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

from libc.stdint cimport int64_t

cdef extern from "c_instructions_counter.h":
    cdef cppclass cInstructionsCounter:
        cInstructionsCounter()
        void set_simd_factor(double simd_factor)
        void start()
        void stop()
        void reset()
        int64_t get_count()
        int64_t get_flops()

__all__ = ['InstructionCounter']


# ====================
# Instructions Counter
# ====================

cdef class InstructionsCounter:
    """
    Wrapper for Linux's Perf tool.
    """

    cdef cInstructionsCounter* c_instructions_counter

    # ====
    # cinit
    # ====

    def __cinit__(self, simd_factor=1.0):
        """
        Initialize the C++ InstructionsCounter
        """

        self.c_instructions_counter = new cInstructionsCounter()
        self.c_instructions_counter.set_simd_factor(1.0)

    # =======
    # dealloc
    # =======

    def __dealloc__(self):
        """
        Clean up the C++ object.
        """

        del self.c_instructions_counter

    # =====
    # start
    # =====

    def start(self):
        """
        Start counting instructions.
        """

        self.c_instructions_counter.start()

    # ====
    # stop
    # ====

    def stop(self):
        """
        Stop counting instructions.
        """

        self.c_instructions_counter.stop()

    # =====
    # reset
    # =====

    def reset(self):
        """
        Reset counts.
        """

        self.c_instructions_counter.reset()

    # =========
    # get count
    # =========

    def get_count(self):
        """
        Get the instruction count.
        
        Returns
        -------
    
        count : int
            Number of hardware instructions counted.
        """

        return self.c_instructions_counter.get_count()

    # ===============
    # set simd factor
    # ===============

    def set_simd_factor(self, simd_factor: float):
        """
        Set the SIMD adjustment factor for floating-point operations.

        Parameters
        ----------

        factor : float
            Number of FLOPs per hardware instruction.
        """

        self.c_instructions_counter.set_simd_factor(simd_factor)

    # =========
    # get flops
    # =========

    def get_flops(self) -> float:
        """
        Get the adjusted floating-point operation count (FLOPs).

        Returns
        -------

        flops : float
            Number of FLOPs counted.
        """

        return self.c_instructions_counter.get_flops()
