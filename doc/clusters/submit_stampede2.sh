#!/bin/bash

#SBATCH -J test_stampede
#SBATCH -o %x_%j.out                    # Name of stdout output file
#SBATCH -e %x_%j.err                    # Name of stderr error file
#SBATCH -p compute                      # Queue (partition) name
#SBATCH -N 4                            # Number of nodes requested
#SBATCH --ntasks-per-node=64            # Number of processes/tasks per node
#SBATCH --export=ALL                    # Propagate all user's environment variables
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

# Setup relevant module on stampede2
module reset
module unload python2
# Other mpi libraries are loaded by default (intel mpi)
source deactivate # deactivate any existing environment
source activate sopht-mpi-env
# Print loaded python (sanity check for correctly loaded environment)
which python

# Execute the program
# Use ibrun instead of mpirun or mpiexec
# The number of processes will spawn automatically according to the resources requested
# Number of processes = number of nodes * number of tasks/processes per node
ibrun python -u ${PROGNAME}
