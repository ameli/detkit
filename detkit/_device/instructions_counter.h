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

#include <linux/perf_event.h>


// ====================
// Instructions Counter
// ====================

class InstructionsCounter
{
    public:
        InstructionsCounter();
        ~InstructionsCounter();
        void start();
        void stop();
        long long get_count();
        
    // Member data
    struct perf_event_attr pe;
    int fd;
    long long count;
};

#endif  // _DEVICE_INSTRUCTIONS_COUNTER_H_
