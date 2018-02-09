#!/bin/bash -x
#MSUB -l nodes=1:ppn=4,pmem=16000mb
#MSUB -l walltime=03:00:00
#MSUB -N HPC_WITH_PYTHON

module load devel/python/3.6.0
module load mpi/openmpi/2.1-gnu-5.2

echo "Running on ${MOAB_PROCCOUNT} cores."
time mpirun -n ${MOAB_PROCCOUNT} $HOME/venvs/rl/bin/python3.exe $HOME/git/hpc/flow_main_mpi.py
