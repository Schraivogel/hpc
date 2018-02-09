
module load mpi/openmpi/2.1-gnu-5.2
# Use when loading OpenMPI in version 1.8.x
mpirun --bind-to core --map-by core -report-bindings flow_main.py
