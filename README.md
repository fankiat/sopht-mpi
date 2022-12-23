# sopht-mpi

Scalable One-stop Platform for Hydroelastic Things (SOPHT) MPI solver.

Python implementation of an elastohydrodynamic MPI solver, for resolving
flow-structure interaction of 3D mixed soft/rigid bodies in viscous flows.

## Installation

Below are steps of how to install `sopht-mpi`. We mainly use `poetry` to manage
the project, although most of the important commands will be provided in `Makefile`.

1. Clone!

First **create the fork repository and clone** to your local machine.

2. Virtual python workspace: `conda`.

We recommend using python version above 3.10.

```bash
conda create --name sopht-mpi-env
conda activate sopht-mpi-env
conda install python==3.10
```

3. Install non-python dependencies, that include `MPI`, `hdf5-mpi`
and `fftw`. For `Ubuntu` you can use:
```bash
make install_non_python_modules_on_ubuntu
```
And for Mac-OS one can use
```bash
make install_non_python_modules_on_macos
```
**NOTE**: For cluster, optimised versions of the above modules are already
installed, and as such this step can be skipped, and the internal modules
should be loaded directly.

4. Setup [`poetry`](https://python-poetry.org) and `dependencies`!

```bash
make poetry-download
make install
make pre-commit-install
```
