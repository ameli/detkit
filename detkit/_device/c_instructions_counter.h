/*
 *  SPDX-FileCopyrightText: Copyright 2022, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _DEVICE_INSTRUCTIONS_COUNTER_H_
#define _DEVICE_INSTRUCTIONS_COUNTER_H_

// =======
// Headers
// =======

#if __linux__
    #include <linux/perf_event.h>
#endif


// ======================
// c Instructions Counter
// ======================

class cInstructionsCounter
{
    public:
        cInstructionsCounter();
        ~cInstructionsCounter();
        void set_simd_factor(double factor);
        void start();
        void stop();
        void reset();
        long long get_count();
        long long get_flops();
        
    // Member data
    private:
        #if __linux__
        struct perf_event_attr pe;
        #endif
        int fd;
        long long count;
        double simd_factor;
};

#endif  // _DEVICE_INSTRUCTIONS_COUNTER_H_
