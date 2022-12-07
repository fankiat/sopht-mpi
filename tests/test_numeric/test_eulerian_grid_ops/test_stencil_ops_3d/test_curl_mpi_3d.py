import numpy as np
import pytest
from sopht.utils.precision import get_real_t, get_test_tol
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d import gen_curl_pyst_kernel_3d
from sopht_mpi.utils import (
    MPIConstruct3D,
    MPIGhostCommunicator3D,
    MPIFieldCommunicator3D,
)
from sopht_mpi.numeric.eulerian_grid_ops.stencil_ops_3d import (
    gen_curl_pyst_mpi_kernel_3d,
)


@pytest.mark.mpi(group="MPI_stencil_ops_3d", min_size=4)
@pytest.mark.parametrize("ghost_size", [1, 2])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize(
    "rank_distribution",
    [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
)
@pytest.mark.parametrize("aspect_ratio", [(1, 1, 1), (1, 1.5, 2)])
def test_mpi_curl_3d(ghost_size, precision, rank_distribution, aspect_ratio):
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

    # extra width needed for kernel computation
    mpi_ghost_exchange_communicator = MPIGhostCommunicator3D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )
    mpi_field_communicator = MPIFieldCommunicator3D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )
    gather_local_field = mpi_field_communicator.gather_local_field
    scatter_global_field = mpi_field_communicator.scatter_global_field

    # Allocate local field
    local_vector_field = np.zeros(
        (
            mpi_construct.grid_dim,
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
            mpi_construct.local_grid_size[2] + 2 * ghost_size,
        )
    ).astype(real_t)
    local_curl = np.zeros_like(local_vector_field)

    # Initialize and broadcast solution for comparison later
    if mpi_construct.rank == 0:
        ref_vector_field = np.random.rand(
            mpi_construct.grid_dim, grid_size_z, grid_size_y, grid_size_x
        ).astype(real_t)
        prefactor = real_t(0.1)
    else:
        ref_vector_field = None
        prefactor = None
    ref_vector_field = mpi_construct.grid.bcast(ref_vector_field, root=0)
    prefactor = mpi_construct.grid.bcast(prefactor, root=0)

    # scatter global field
    scatter_global_field(local_vector_field[0], ref_vector_field[0], mpi_construct)
    scatter_global_field(local_vector_field[1], ref_vector_field[1], mpi_construct)
    scatter_global_field(local_vector_field[2], ref_vector_field[2], mpi_construct)

    # compute the diffusion flux
    curl_pyst_mpi_kernel_3d = gen_curl_pyst_mpi_kernel_3d(
        real_t=real_t,
        mpi_construct=mpi_construct,
        ghost_exchange_communicator=mpi_ghost_exchange_communicator,
    )

    curl_pyst_mpi_kernel_3d(
        curl=local_curl,
        field=local_vector_field,
        prefactor=prefactor,
    )

    # gather back the diffusion flux globally
    global_curl = np.zeros(
        (mpi_construct.grid_dim, grid_size_z, grid_size_y, grid_size_x)
    ).astype(real_t)
    gather_local_field(global_curl[0], local_curl[0], mpi_construct)
    gather_local_field(global_curl[1], local_curl[1], mpi_construct)
    gather_local_field(global_curl[2], local_curl[2], mpi_construct)

    # assert correct
    if mpi_construct.rank == 0:
        curl_pyst_kernel_3d = gen_curl_pyst_kernel_3d(
            real_t=real_t,
        )
        ref_curl = np.zeros_like(global_curl)
        curl_pyst_kernel_3d(
            curl=ref_curl,
            field=ref_vector_field,
            prefactor=prefactor,
        )
        kernel_support = curl_pyst_mpi_kernel_3d.kernel_support
        # check kernel_support for the curl kernel
        assert kernel_support == 1, "Incorrect kernel support!"
        # check field correctness
        inner_idx = (
            slice(None),
            slice(kernel_support, -kernel_support),
            slice(kernel_support, -kernel_support),
        )
        np.testing.assert_allclose(
            ref_curl[inner_idx],
            global_curl[inner_idx],
            atol=get_test_tol(precision),
        )
