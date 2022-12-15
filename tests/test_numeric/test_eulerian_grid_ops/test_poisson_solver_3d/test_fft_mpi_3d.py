import numpy as np
import pytest
from sopht_mpi.utils import (
    MPIConstruct3D,
    MPIFieldCommunicator3D,
)
from sopht_mpi.numeric.eulerian_grid_ops.poisson_solver_3d import FFTMPI3D
from sopht.utils.precision import get_real_t, get_test_tol
from scipy.fft import rfftn


@pytest.mark.mpi(group="MPI_Poisson_solver_3d", min_size=4)
@pytest.mark.parametrize("ghost_size", [1, 2])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize(
    "rank_distribution",
    [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
)
@pytest.mark.parametrize("aspect_ratio", [(1, 1, 1), (1, 1.5, 2)])
def test_mpi_fft_3d(ghost_size, precision, rank_distribution, aspect_ratio):
    """
    Test parallel FFT on (slab / pencil) distributed 3d arrays
    """
    n_values = 8
    grid_size_z, grid_size_y, grid_size_x = (n_values * np.array(aspect_ratio)).astype(
        int
    )
    real_t = get_real_t(precision)
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct3D(
        grid_size_z=grid_size_z,
        grid_size_y=grid_size_y,
        grid_size_x=grid_size_x,
        real_t=real_t,
        rank_distribution=rank_distribution,
    )

    # Create parallel fft plan
    mpi_fft = FFTMPI3D(
        grid_size_z=mpi_construct.global_grid_size[0],
        grid_size_y=mpi_construct.global_grid_size[1],
        grid_size_x=mpi_construct.global_grid_size[2],
        mpi_construct=mpi_construct,
        real_t=real_t,
    )

    # Initialize communicator for scatter and gather
    mpi_field_comm = MPIFieldCommunicator3D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )
    gather_local_scalar_field = mpi_field_comm.gather_local_scalar_field
    scatter_global_scalar_field = mpi_field_comm.scatter_global_scalar_field
    local_field_inner_idx = mpi_field_comm.inner_idx

    # Generate solution and broadcast solution from rank 0 to all ranks
    if mpi_construct.rank == 0:
        ref_field = np.random.randn(grid_size_z, grid_size_y, grid_size_x).astype(
            real_t
        )
    else:
        ref_field = None
    ref_field = mpi_construct.grid.bcast(ref_field, root=0)

    # 1. Scatter initial local field from solution ref field
    local_field = np.zeros(mpi_construct.local_grid_size + 2 * ghost_size).astype(
        real_t
    )
    scatter_global_scalar_field(local_field, ref_field)

    # 2. Forward transform (fourier field)
    local_fourier_field = np.zeros_like(mpi_fft.fourier_field_buffer)
    mpi_fft.forward(
        field=local_field[local_field_inner_idx], fourier_field=local_fourier_field
    )

    # 3. Backward transform (inverse fourier field)
    local_inv_fourier_field = np.zeros_like(local_field)
    mpi_fft.backward(
        fourier_field=local_fourier_field,
        inv_fourier_field=local_inv_fourier_field[local_field_inner_idx],
    )

    # 4. Gather local fields
    # For gathering fourier field, we use features from distarray directly
    # (which depend on h5py) since the r2c transform causes the array to be
    # distributed in an uneven fashion, for which gather_array doesn't work
    fourier_field = local_fourier_field.get((slice(None),) * mpi_construct.grid_dim)
    inv_fourier_field = np.zeros_like(ref_field)
    gather_local_scalar_field(
        global_field=inv_fourier_field, local_field=local_inv_fourier_field
    )
    # 5. Assert correct
    if mpi_construct.rank == 0:
        # unpack fft axes from mpi4py-fft, and use for scipy axes
        solution_fft_axes = []
        for ax in mpi_fft.fft.axes:
            # It can either be a sequence of ints, or sequence of sequence of ints
            # i.e. (1, 2, 3) or ((1,), (2,), (3,))
            if isinstance(ax, (int, np.integer)):
                solution_fft_axes.append(ax)
            else:
                ax = list(ax)
                for a in ax:
                    solution_fft_axes.append(a)
        correct_fourier_field = rfftn(ref_field, axes=solution_fft_axes)
        np.testing.assert_allclose(
            ref_field,
            inv_fourier_field,
            atol=get_test_tol(precision),
        )
        np.testing.assert_allclose(
            fourier_field,
            correct_fourier_field,
            atol=get_test_tol(precision),
        )
