# Installation on supercomputing clusters
Installation on supercomputing clusters are mostly similar to how installation is done
your local machine, with a few pre-installation steps involving loading readily
available modules and setting relevant environment variables.
Once you have completed the pre-installation steps according to the cluster of your
choice, you should be ready to proceed with the installation steps as detailed in the
[main repository](https://github.com/fankiat/sopht-mpi).
*Please note that you should skip **step 3** of the installation process detailed there
since the non-python modules are loaded on the cluster environment after taking the
pre-installation steps below.*

## Expanse
1. Load the relevant modules and set environment variables as below.
```bash
module reset
module load gcc openmpi fftw anaconda3 hdf5
export HDF5_DIR=$HDF5HOME
export FFTW_DIR=$FFTWHOME
```
2. Create python virtual environment and proceed with usual installation steps as
detailed in the [main repository](https://github.com/fankiat/sopht-mpi) (skipping step
3).

## Bridges-2
1. Load the relevant modules and set environment variables as below.
```bash
module reset
module load anaconda3 gcc/10.2.0 openmpi/4.0.5-gcc10.2.0 phdf5/1.10.7-openmpi4.0.5-gcc10.2.0 fftw
```
2. Create python virtual environment and proceed with usual installation steps as
detailed in the [main repository](https://github.com/fankiat/sopht-mpi) (skipping step
3).

## Stampede2
1. Load the relevant modules and set environment variables as below. Here we unload the
old Python 2.7 version to remove older, unnecessary reference of `PYTHONPATH` to ensure
smooth installation of `sopht-mpi`. We also use the default modules on `Stampede2`,
which includes `Intel-MPI`.
```bash
module reset
module unload python2 # remove unnecessary python2
module load phdf5 fftw3
export HDF5_DIR=$TACC_HDF5_DIR
export FFTW_DIR=$TACC_FFTW3_DIR
```

2. Since `anaconda` is not an available module on `Stampede2`, we need to install
miniconda separately. Detailed steps on installation can be found in
[TACC wiki](https://wikis.utexas.edu/display/bioiteam/Linux+and+stampede2+Setup+--+GVA2021#Linuxandstampede2SetupGVA2021-MovingbeyondthepreinstalledcommandsonTACC)
and [Miniconda documentation](https://docs.conda.io/en/latest/miniconda.html) page.

3. Create python virtual environment and proceed with usual installation steps as
detailed in the [main repository](https://github.com/fankiat/sopht-mpi) (skipping step
3).

### **Note for jobs with large number of processes**
*Once the facility-specific installation steps are done, please update the cache
directory for `pystencils`.
1. Find out your `$SCRATCH` directory by doing `echo $SCRATCH`. (For `Bridges-2`, we use
the `$LOCAL`).
2. Edit the config file for `pystencils` located in `~/.config/pystencils/config.json`
and set the `"object_cache"` directory with `"your_scratch_directory/.cache/pystencils/objectcache/{pid}"`, where `your_scratch_directory` is replaced with the output of the
command above. This will allow the cache objects to be generated in the `$SCRATCH`
instead of `$HOME`. More documentation on setting the config file can be found
[here](https://github.com/mabau/pystencils/blob/master/pystencils/cpu/cpujit.py).*


# Submitting jobs on cluster
Once you have setup the solver on your desired cluster, you can submit jobs using the
`submit_*.sh` batch submission scripts provided as reference.
