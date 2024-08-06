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

from detkit import memdet, get_config, get_processor_name
from imate import correlation_matrix
import multiprocessing
import numpy
import pickle
import zarr
import sys
import os
from os.path import join
import shutil
import getopt


# ===============
# parse arguments
# ===============

def parse_arguments(argv):
    """
    Parses the argument of the executable.
    """

    # -----------
    # print usage
    # -----------

    def print_usage(exec_name):
        usage_string = "Usage: " + exec_name + " <arguments>"
        options_string = """
The following arguments are required:

    -i --min-size=[int]     Minimum size of the matrix in log2. The size of
                            matrix is 2 to the power of this number.
    -j --max-size=[int]     Maximum size of the matrix in log2. The size of
                            matrix is 2 to the power of this number.
    -b --max-blocks=[int]   Maximum number of blocks.

The following arguments are optional:

    -r --repeat=[int]       Number of times the numerical experiment is
                            repeated. Default is 5.
    -v --verbose            Prints verbose output. Default is False.
    -h --help               Prints the help message.

Examples:

    Compute loggdet, set the matrix size from 2**8=256 to 2**12, each with
    the number of blocks from 1 to 5, and each repeated 10 times:

    $ %s -i 8 -j 12 -b 5 -r 10 -v
        """ % (argv[0])

        print(usage_string)
        print(options_string)

    # -----------------

    # Initialize variables (defaults)
    arguments = {
        'min_size': None,
        'max_size': None,
        'max_blocks': None,
        'repeat': 5,
        'verbose': False,
    }

    # Get options
    try:
        opts, args = getopt.getopt(argv[1:], "i:j:b:r:vh",
                                   ["min_size=", "max-size=", "max-blocks=",
                                    "repeat=", "verbose", "help"])
    except getopt.GetoptError:
        print_usage(argv[0])
        sys.exit(2)

    # Assign options
    for opt, arg in opts:
        if opt in ('-i', '--min-size'):
            arguments['min_size'] = int(arg)
        elif opt in ('-j', '--max_size'):
            arguments['max_size'] = int(arg)
        elif opt in ('-b', '--max-blocks'):
            arguments['max_blocks'] = int(arg)
        elif opt in ('-r', '--repeat'):
            arguments['repeat'] = int(arg)
        elif opt in ('-v', '--verbose'):
            arguments['verbose'] = True
        elif opt in ('-h', '--help'):
            print_usage(argv[0])
            sys.exit()

    if arguments['min_size'] is None:
        print('ERROR: argument "-i" is required.')
        print('')
        print_usage(argv[0])
        sys.exit()
    if arguments['max_size'] is None:
        print('ERROR: argument "-j" is required.')
        print('')
        print_usage(argv[0])
        sys.exit()
    if arguments['max_blocks'] is None:
        print('ERROR: argument "-b" is required.')
        print('')
        print_usage(argv[0])
        sys.exit()

    return arguments


# ==========
# remove dir
# ==========

def remove_dir(dir_name, verbose=False):
    """
    Removes a directory
    """

    if os.path.exists(dir_name):
        try:
            shutil.rmtree(dir_name)
            if verbose:
                print('File %s is deleted.' % dir_name)
        except OSError:
            pass

    else:
        if verbose:
            print('Directory %s does not exists.' % dir_name)


# =========
# benchmark
# =========

def benchmark(argv):
    """
    Benchmarks for computing determinant using the conventional verses the
    presented methods.
    """

    # get use input arguments from commandline
    arguments = parse_arguments(argv)

    # detkit config
    detkit_config = get_config()

    # Devices dictionary
    devices = {
        'cpu_name': get_processor_name(),
        'num_all_cpu_threads': multiprocessing.cpu_count(),
    }

    # Unpack arguments
    min_size = arguments['min_size']
    max_size = arguments['max_size']
    max_blocks = arguments['max_blocks']
    num_repeats = arguments['repeat']
    verbose_benchmark = arguments['verbose']

    # Settings
    matrix_sizes = 2**numpy.arange(min_size, max_size+1)
    num_blocks = numpy.arange(1, max_blocks+1)
    assume = 'spd'
    verbose_memdet = False
    parallel_io = 'tensorstore'

    # Config dictionary
    config = {
        'matrix_sizes': matrix_sizes,
        'num_blocks': num_blocks,
        'num_repeats': num_repeats,
        'assume': assume,
        'parallel_io': parallel_io,
    }

    # Allocate outputs
    shape = (matrix_sizes.size, num_blocks.size, num_repeats)
    tot_wall_time = numpy.empty(shape, dtype=numpy.float64)
    tot_proc_time = numpy.empty(shape, dtype=numpy.float64)
    load_wall_time = numpy.empty(shape, dtype=numpy.float64)
    load_proc_time = numpy.empty(shape, dtype=numpy.float64)
    store_wall_time = numpy.empty(shape, dtype=numpy.float64)
    store_proc_time = numpy.empty(shape, dtype=numpy.float64)
    alloc_mem = numpy.empty(shape, dtype=int)
    alloc_mem_peak = numpy.empty(shape, dtype=int)
    total_mem = numpy.empty(shape, dtype=int)
    total_mem_peak = numpy.empty(shape, dtype=int)
    scratch_nbytes = numpy.empty(shape[:-1], dtype=int)
    matrix_blocks = numpy.empty(shape[:-1], dtype=int)
    block_size = numpy.empty(shape[:-1], dtype=int)
    num_block_loads = numpy.empty(shape[:-1], dtype=int)
    num_block_stores = numpy.empty(shape[:-1], dtype=int)
    # mem_unit = numpy.empty(shape[:-1], dtype=int)
    lds = numpy.empty(shape, dtype=numpy.float64)
    signs = numpy.empty(shape, dtype=int)

    # Compute log-determinant
    for i in range(matrix_sizes.size):

        if verbose_benchmark:
            print(f'size: 2**{int(0.5 + numpy.log2(matrix_sizes[i]))} ',
                  flush=True)

        # Create a symmetric positive-definite matrix
        A = correlation_matrix(size=matrix_sizes[i], dimension=1)

        # Store matrix as a zarr array on disk (optional)
        z_path = 'matrix-' + str(matrix_sizes[i]) + '.zarr'
        z = zarr.open(z_path, mode='w',
                      shape=(matrix_sizes[i], matrix_sizes[i]), dtype=A.dtype)

        z[:, :] = A

        for j in range(num_blocks.size):

            if verbose_benchmark:
                print(f'    blocks: {num_blocks[j]} ', end='', flush=True)

            for k in range(num_repeats):

                # Compute log-determinant
                ld, sign, _, info = memdet(
                        z, assume=assume,
                        num_blocks=num_blocks[j],
                        parallel_io=parallel_io,
                        verbose=verbose_memdet, return_info=True)

                # Store output
                tot_wall_time[i, j, k] = info['process']['tot_wall_time']
                tot_proc_time[i, j, k] = info['process']['tot_proc_time']
                load_wall_time[i, j, k] = info['process']['load_wall_time']
                load_proc_time[i, j, k] = info['process']['load_proc_time']
                store_wall_time[i, j, k] = info['process']['store_wall_time']
                store_proc_time[i, j, k] = info['process']['store_proc_time']
                alloc_mem[i, j, k] = info['memory']['alloc_mem']
                alloc_mem_peak[i, j, k] = info['memory']['alloc_mem_peak']
                total_mem[i, j, k] = info['memory']['total_mem']
                total_mem_peak[i, j, k] = info['memory']['total_mem_peak']
                lds[i, j, k] = ld
                signs[i, j, k] = sign

                if k == 0:
                    scratch_nbytes[i, j] = info['scratch']['scratch_nbytes']
                    matrix_blocks[i, j] = info['block']['matrix_blocks'][0]
                    block_size[i, j] = info['block']['block_shape'][0]
                    num_block_loads[i, j] = info['scratch']['num_block_loads']
                    num_block_stores[i, j] = \
                        info['scratch']['num_block_stores']
                    # mem_unit[i, j] = info['memory']['mem_unit']

                if verbose_benchmark:
                    print('.', end='', flush=True)

            if verbose_benchmark:
                print('', flush=True)

        remove_dir(z_path)

        if verbose_benchmark:
            print('', flush=True)

    # Dictionary of results
    results = {
            'tot_wall_time': tot_wall_time,
            'tot_proc_time': tot_proc_time,
            'load_wall_time': load_wall_time,
            'load_proc_time': load_proc_time,
            'store_wall_time': store_wall_time,
            'store_proc_time': store_proc_time,
            'scratch_nbytes': scratch_nbytes,
            'matrix_blocks': matrix_blocks,
            'block_size': block_size,
            'num_block_loads': num_block_loads,
            'num_block_stores': num_block_stores,
            'alloc_mem': alloc_mem,
            'alloc_mem_peak': alloc_mem_peak,
            'total_mem': total_mem,
            'total_mem_peak': total_mem_peak,
            # 'mem_unit': mem_unit,
            'lds': lds,
            'signs': signs,
    }

    # Store all results
    benchmark_results = {
        'detkit_config': detkit_config,
        'config': config,
        'devices': devices,
        'results': results
    }

    # Save results
    benchmark_dir = '.'
    pickle_dir = 'pickle_results'
    base_dir = join(benchmark_dir, pickle_dir)
    if not os.path.isdir(base_dir):
        base_dir = '.'
    output_filename = 'benchmark-memdet.pickle'
    output_full_filename = join(base_dir, output_filename)
    with open(output_full_filename, 'wb') as file:
        pickle.dump(benchmark_results, file,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print('Results saved to %s.' % output_full_filename)


# ===========
# Script main
# ===========

if __name__ == "__main__":
    benchmark(sys.argv)
