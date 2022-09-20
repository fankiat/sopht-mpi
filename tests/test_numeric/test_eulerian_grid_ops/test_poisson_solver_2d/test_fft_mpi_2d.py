import numpy as np
import pytest
from sopht_mpi.utils import (
    MPIConstruct2D,
    MPIFieldCommunicator2D,
)
from sopht_mpi.numeric.eulerian_grid_ops.poisson_solver_2d import FFTMPI2D
from sopht.utils.precision import get_real_t, get_test_tol
from scipy.fft import rfftn


@pytest.mark.mpi(group="MPI_Poisson_solver_2d")
@pytest.mark.parametrize("ghost_size", [1, 2, 3])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rank_distribution", [(1, 0), (0, 1)])
@pytest.mark.parametrize("aspect_ratio", [(1, 1), (1, 2), (2, 1)])
def test_mpi_fft_slab(ghost_size, precision, rank_distribution, aspect_ratio):
    """
    Test parallel FFT on slab distributed along x and y
    """
    n_values = 32
    real_t = get_real_t(precision)
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=n_values * aspect_ratio[0],
        grid_size_x=n_values * aspect_ratio[1],
        real_t=real_t,
        rank_distribution=rank_distribution,
    )

    # Create parallel fft plan
    mpi_fft = FFTMPI2D(
        grid_size_y=mpi_construct.global_grid_size[0],
        grid_size_x=mpi_construct.global_grid_size[1],
        mpi_construct=mpi_construct,
        real_t=real_t,
    )

    # Initialize communicator for scatter and gather
    mpi_field_io_comm = MPIFieldCommunicator2D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )
    gather_local_field = mpi_field_io_comm.gather_local_field
    scatter_global_field = mpi_field_io_comm.scatter_global_field
    local_field_inner_idx = mpi_field_io_comm.inner_idx

    # Generate solution and broadcast solution from rank 0 to all ranks
    if mpi_construct.rank == 0:
        ref_field = np.random.randn(
            n_values * aspect_ratio[0], n_values * aspect_ratio[1]
        ).astype(real_t)
    else:
        ref_field = None
    ref_field = mpi_construct.grid.bcast(ref_field, root=0)

    # 1. Scatter initial local field from solution ref field
    local_field = np.zeros(mpi_construct.local_grid_size + 2 * ghost_size).astype(
        real_t
    )
    scatter_global_field(local_field, ref_field, mpi_construct)

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
    gather_local_field(
        global_field=inv_fourier_field,
        local_field=local_inv_fourier_field,
        mpi_construct=mpi_construct,
    )
    # # 5. Assert correct
    if mpi_construct.rank == 0:
        # flip FFT axes for ref solution, to match mpi4py-fft distribution
        solution_fft_axes = (1, 0) if rank_distribution == (1, 0) else None
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
