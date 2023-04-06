#!/bin/bash

#SBATCH -J test_bridges
#SBATCH -o %x_%j.out                    # Name of stdout output file
#SBATCH -e %x_%j.err                    # Name of stderr error file
#SBATCH -p RM                           # Queue (partition) name
#SBATCH -N 4                            # Number of nodes requested
#SBATCH --ntasks-per-node=128           # Number of processes/tasks per node
#SBATCH -t 00:10:00                     # Run time (hh:mm:ss)
#SBATCH --mail-user=email@email.edu     # User to receive email notification
#SBATCH --mail-type=all                 # Send email at begin, end, or fail of job
#SBATCH --account=mcb200029p            # Account to charge resources used by this job

# Other commands must follow all #SBATCH directives...

# File to be executed
PROGNAME="flow_past_sphere_case.py"

# Print some details on launched job
date
echo Job name: $SLURM_JOB_NAME
echo Execution dir: $SLURM_SUBMIT_DIR
echo Number of processes: $SLURM_NTASKS

# Setup relevant module on expanse
module load anaconda3 gcc/10.2.0 openmpi/4.0.5-gcc10.2.0 phdf5/1.10.7-openmpi4.0.5-gcc10.2.0 fftw
source deactivate # deactivate any existing environment
source activate sopht-mpi-env
# Print loaded python (sanity check for correctly loaded environment)
which python

# Execute the program
mpiexec -n ${SLURM_NTASKS} python -u ${PROGNAME}
