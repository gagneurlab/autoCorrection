#!/bin/bash
#
# Slurm parameters
#SBATCH --mem=16G
#SBATCH -c 4
# ------------------

DB=corrector
# path to the repository containing the script
THIS_DIR=$PWD

export PYTHONPATH=${THIS_DIR}

hyperopt-mongo-worker \
    --mongo=ouga03:1234/$DB \
    --poll-interval=1 \
    --reserve-timeout=3600
