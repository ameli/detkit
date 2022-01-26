/*
 *  SPDX-FileCopyrightText: Copyright 2022, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _DEVICE_MATMAT_BENCHMARK_H_
#define _DEVICE_MATMAT_BENCHMARK_H_

// =======
// Headers
// =======

#include <linux/perf_event.h>
#include "../_definitions/types.h"  // IndexType


// ================
// MatMat Benchmark
// ================

template <typename DataType>
class MatMatBenchmark
{
    public:

    static long long matmat_benchmark(
            DataType* dummy_var,
            IndexType n);
};

#endif  // _DEVICE_MATMAT_BENCHMARK_H_
