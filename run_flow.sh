#!/bin/bash -x
#MSUB -l nodes=1:ppn=16,pmem=16000mb
#MSUB -l walltime=03:00:00
#MSUB -N HPC_WITH_PYTHON

module load devel/python/3.6.0
module load mpi/openmpi/2.1-gnu-5.2

cd $PBS_O_WORKDIR

echo "Running on ${MOAB_PROCCOUNT} cores."
mpirun -n ${MOAB_PROCCOUNT} python3 flow_main.py
 
#python3  "$HOME/git/hpc/flow_main.py"

