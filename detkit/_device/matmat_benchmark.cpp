/*
 *  SPDX-FileCopyrightText: Copyright 2022, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


// =======
// Headers
// =======

#include "./matmat_benchmark.h"
#include <cstdlib>  // rand, RAND_MAX
#include "../_definitions/definitions.h"  // COUNT_PERF
#include "./instructions_counter.h"  // InstructionsCounter


// ================
// matmat benchmark
// ================

/// \brief  Computes the number of hardware instructions for a matrix-martrix
///         multiplication. This number is used to find on a certain device,
///         how many insrtrucitons is needed to compute a single task in
///         mat-mat multiplication, that is, a multiplication of two scalars
///         and one addition.

template <typename DataType>
long long MatMatBenchmark<DataType>::matmat_benchmark(
        DataType* dummy_var,
        LongIndexType n)
{
    long long hw_instructions = -1;

    // Mark unused variables to avoid compiler warnings
    // (-Wno-unused-parameter)
    (void) dummy_var;

    #if COUNT_PERF

        DataType* A = new DataType[n*n];
        DataType* B = new DataType[n*n];
        DataType* C = new DataType[n*n];

        for (int i=0; i < n; ++i)
        {
            for (int j=0; j < n; ++j)
            {
                A[i*n + j] = rand() / RAND_MAX;
                B[i*n + j] = rand() / RAND_MAX;
                C[i*n + j] = 0.0;
            }
        }

        // Measure flops
        InstructionsCounter instructions_counter = InstructionsCounter();
        instructions_counter.start();

        DataType sum;

        for (LongIndexType i=0; i < n; ++i)
        {
            for (LongIndexType j=0; j < n; ++j)
            {
                sum = 0.0;

                for (LongIndexType k=0; k < n; ++k)
                {
                    sum += A[i*n+k] * B[k*n+j];
                }

                C[i*n + j] = sum;
            }
        }

        instructions_counter.stop();
        hw_instructions = instructions_counter.get_count();

    #endif

    return hw_instructions;
}


// ===============================
// Explicit template instantiation
// ===============================

template class MatMatBenchmark<float>;
template class MatMatBenchmark<double>;
template class MatMatBenchmark<long double>;
