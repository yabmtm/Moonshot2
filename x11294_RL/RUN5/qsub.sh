#!/bin/bash

#PBS -N test_Ee.pf
#PBS -o test_Ee.out
#PBS -q normal
#PBS -l nodes=1:ppn=28
#PBS -l walltime=48:00:00
#PBS

cd $PBS_O_WORKDIR
export OMP_NUM_THREADS=1

module load gromacs
module load mpi/openmpi

#gmx mdrun -v -deffnm npt
mpirun mdrun_mpi -maxh 47 -s prod.tpr
##mpirun mdrun_mpi -cpi state.cpt -maxh 47 -s prod.tpr

##python python.py
