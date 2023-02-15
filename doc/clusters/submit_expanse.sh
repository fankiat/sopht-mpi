#!/bin/bash

#SBATCH -J test_expanse
#SBATCH -o %x_%j.out                    # Name of stdout output file
#SBATCH -e %x_%j.err                    # Name of stderr error file
#SBATCH -p compute                      # Queue (partition) name
#SBATCH -N 4                            # Number of nodes requested
#SBATCH --ntasks-per-node=128           # Number of processes/tasks per node
#SBATCH --mem=249325M                   # Memory per compute node (set to expanse limit)
#SBATCH -t 00:10:00                     # Run time (hh:mm:ss)
#SBATCH --mail-user=email@email.edu     # User to receive email notification
#SBATCH --mail-type=all                 # Send email at begin, end, or fail of job
#SBATCH --account=TG-MCB190004          # Account to charge resources used by this job

# Other commands must follow all #SBATCH directives...

# File to be executed
PROGNAME="flow_past_sphere_case.py"

# Print some details on launched job
date
echo Job name: $SLURM_JOB_NAME
echo Execution dir: $SLURM_SUBMIT_DIR
echo Number of processes: $SLURM_NTASKS

# Setup relevant module on expanse
module reset
module load gcc openmpi fftw hdf5 anaconda3
source deactivate # deactivate any existing environment
source activate sopht-mpi-env
# Print loaded python (sanity check for correctly loaded environment)
which python

# Execute the program
mpiexec -n ${SLURM_NTASKS} python -u ${PROGNAME}
