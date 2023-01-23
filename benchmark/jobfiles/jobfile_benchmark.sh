#!/bin/bash

#SBATCH --job-name=benchmark_speed
#SBATCH --mail-type=ALL                         # (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sameli@berkeley.edu
#SBATCH --partition=savio2
#SBATCH --account=fc_biome
#SBATCH --qos=savio_normal
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
###SBATCH --mem=64gb
#SBATCH --output=output.log

PYTHON_DIR=$HOME/programs/miniconda3
SCRIPTS_DIR=$(dirname $PWD)/scripts
LOG_DIR=$PWD

# Logarithm of base 2 of the matrix size. For example N=9 is the size 512.
N=9
FUNC="loggdet"
REPEAT=10
NUM_RATIOS=50
STREAM_OUTPUT="stream_output-${N}.txt"

$PYTHON_DIR/bin/python ${SCRIPTS_DIR}/benchmark.py -n $N -f $FUNC -r $REPEAT \
    -t $NUM_RATIOS -v > ${LOG_DIR}/${STREAM_OUTPUT}
