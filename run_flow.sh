#!/bin/bash -x
#MSUB -l nodes=1:ppn=4,pmem=16000mb
#MSUB -l walltime=10:00:00
#MSUB -N HPC_WITH_PYTHON
#MSUB -m abe
#MSUB -M stephanschraivogel@gmail.com

module load devel/python/3.6
module load mpi/openmpi/2.1-gnu-5.2

echo "Running on ${MOAB_PROCCOUNT} cores."
time mpirun -n ${MOAB_PROCCOUNT} $HOME/venvs/rl/bin/python3 $HOME/git/hpc/flow_main_mpi.py -num_procs ${MOAB_PROCCOUNT} -timesteps 100000
