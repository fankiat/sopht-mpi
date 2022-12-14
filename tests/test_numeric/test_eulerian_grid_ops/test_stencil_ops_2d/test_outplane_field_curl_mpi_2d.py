import numpy as np
import pytest
from sopht.utils.precision import get_real_t, get_test_tol
from sopht.numeric.eulerian_grid_ops.stencil_ops_2d import (
    gen_outplane_field_curl_pyst_kernel_2d,
)
from sopht_mpi.utils import (
    MPIConstruct2D,
    MPIGhostCommunicator2D,
    MPIFieldCommunicator2D,
)
from sopht_mpi.numeric.eulerian_grid_ops.stencil_ops_2d import (
    gen_outplane_field_curl_pyst_mpi_kernel_2d,
)


@pytest.mark.mpi(group="MPI_stencil_ops_2d", min_size=4)
@pytest.mark.parametrize("ghost_size", [1, 2])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rank_distribution", [(1, 0), (0, 1)])
@pytest.mark.parametrize("aspect_ratio", [(1, 1), (1, 1.5)])
def test_mpi_outplane_field_curl_2d(
    ghost_size, precision, rank_distribution, aspect_ratio
):
    n_values = 8
    grid_size_y, grid_size_x = (n_values * np.array(aspect_ratio)).astype(int)
    real_t = get_real_t(precision)
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=grid_size_y,
        grid_size_x=grid_size_x,
        real_t=real_t,
        rank_distribution=rank_distribution,
    )

    # extra width needed for kernel computation
    mpi_ghost_exchange_communicator = MPIGhostCommunicator2D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )
    mpi_field_communicator = MPIFieldCommunicator2D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )
    gather_local_vector_field = mpi_field_communicator.gather_local_vector_field
    scatter_global_scalar_field = mpi_field_communicator.scatter_global_scalar_field

    # Allocate local field
    local_field = np.zeros(
        (
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
        )
    ).astype(real_t)
    local_curl = np.zeros(
        (
            mpi_construct.grid_dim,
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
        )
    ).astype(real_t)

    # Initialize and broadcast solution for comparison later
    if mpi_construct.rank == 0:
        ref_field = np.random.rand(grid_size_y, grid_size_x).astype(real_t)
        prefactor = real_t(0.1)
    else:
        ref_field = None
        prefactor = None
    prefactor = mpi_construct.grid.bcast(prefactor, root=0)

    # scatter global field
    scatter_global_scalar_field(local_field, ref_field)

    # compute the diffusion flux
    outplane_field_curl_pyst_mpi_kernel_2d = gen_outplane_field_curl_pyst_mpi_kernel_2d(
        real_t=real_t,
        mpi_construct=mpi_construct,
        ghost_exchange_communicator=mpi_ghost_exchange_communicator,
    )

    outplane_field_curl_pyst_mpi_kernel_2d(
        curl=local_curl,
        field=local_field,
        prefactor=prefactor,
    )

    # gather back the diffusion flux globally
    global_curl = np.zeros((mpi_construct.grid_dim, grid_size_y, grid_size_x)).astype(
        real_t
    )
    gather_local_vector_field(global_curl, local_curl)

    # assert correct
    if mpi_construct.rank == 0:
        outplane_field_curl_pyst_kernel_2d = gen_outplane_field_curl_pyst_kernel_2d(
            real_t=real_t,
        )
        ref_curl = np.zeros_like(global_curl)
        outplane_field_curl_pyst_kernel_2d(
            curl=ref_curl,
            field=ref_field,
            prefactor=prefactor,
        )
        kernel_support = outplane_field_curl_pyst_mpi_kernel_2d.kernel_support
        # check kernel_support for the diffusion kernel
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
