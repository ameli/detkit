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

#include "./instructions_counter.h"
#include <asm/unistd.h>
#include <iostream>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <inttypes.h>
#include <sys/types.h>


// ===============
// perf event open
// ===============

static long perf_event_open(
        struct perf_event_attr* hw_event,
        pid_t pid,
        int cpu,
        int group_fd,
        unsigned long flags)
{
    int ret;
    ret = syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
    return ret;
}


// ===========
// Constructor
// ===========

InstructionsCounter::InstructionsCounter():
    count(0),
    fd(-1)
{
    memset(&this->pe, 0, sizeof(struct perf_event_attr));
    this->pe.type = PERF_TYPE_HARDWARE;
    this->pe.size = sizeof(struct perf_event_attr);
    this->pe.config = PERF_COUNT_HW_INSTRUCTIONS;
    this->pe.disabled = 1;
    this->pe.exclude_kernel = 1;
    this->pe.exclude_hv = 1;  // Don't count hypervisor events.

    this->fd = perf_event_open(&this->pe, 0, -1, -1, 0);
    if (this->fd == -1)
    {
        // Error, cannot open th leader.
        this->count = -1;
    }
}


// ==========
// Destructor
// ==========

InstructionsCounter::~InstructionsCounter()
{
    if (this->fd != -1)
    {
        close(this->fd);
    }
}


// =====
// Start
// =====

void InstructionsCounter::start()
{
    if (this->fd != -1)
    {
        ioctl(this->fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(this->fd, PERF_EVENT_IOC_ENABLE, 0);
    }
}


// ====
// Stop
// ====

void InstructionsCounter::stop()
{
    if (this->fd != -1)
    {
        ioctl(this->fd, PERF_EVENT_IOC_DISABLE, 0);
        read(this->fd, &this->count, sizeof(long long));
    }
}


// =========
// get count
// =========

long long InstructionsCounter::get_count()
{
    return this->count;
}
