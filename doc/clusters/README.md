# Installation on supercomputing clusters
Installation on supercomputing clusters are mostly similar to how installation is done
your local machine, with a few pre-installation steps involving loading readily
available modules and setting relevant environment variables.
Once you have completed the pre-installation steps according to the cluster of your
choice, you should be ready to proceed with the installation steps as detailed in the
[main repository](https://github.com/fankiat/sopht-mpi).

## Expanse
1. Load the relevant modules and set environment variables as below.
```bash
module load gcc openmpi anaconda3 hdf5
export HDF5_DIR=$HDF5HOME
```
2. Proceed with usual installation steps as detailed in the [main repository](https://github.com/fankiat/sopht-mpi).


# Submitting jobs on cluster
Once you have setup the solver on your desired cluster, you can submit jobs using the
batch submission scripts provided here as reference.
