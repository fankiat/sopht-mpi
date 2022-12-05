import numpy as np
import pytest
from sopht.utils.precision import get_real_t, get_test_tol
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d import gen_divergence_pyst_kernel_3d
from sopht_mpi.utils import (
    MPIConstruct3D,
    MPIGhostCommunicator3D,
    MPIFieldCommunicator3D,
)
from sopht_mpi.numeric.eulerian_grid_ops.stencil_ops_3d import (
    gen_divergence_pyst_mpi_kernel_3d,
)


@pytest.mark.mpi(group="MPI_stencil_ops_3d", min_size=4)
@pytest.mark.parametrize("ghost_size", [1, 2, 3])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize(
    "rank_distribution",
    [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
)
@pytest.mark.parametrize(
    "aspect_ratio",
    [(1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 1, 1), (1, 2, 2), (2, 1, 2), (2, 2, 1)],
)
def test_mpi_divergence_3d(ghost_size, precision, rank_distribution, aspect_ratio):
    n_values = 8
    real_t = get_real_t(precision)
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct3D(
        grid_size_z=n_values * aspect_ratio[0],
        grid_size_y=n_values * aspect_ratio[1],
        grid_size_x=n_values * aspect_ratio[2],
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
    local_divergence = np.zeros(
        (
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
            mpi_construct.local_grid_size[2] + 2 * ghost_size,
        )
    ).astype(real_t)

    # Initialize and broadcast solution for comparison later
    if mpi_construct.rank == 0:
        ref_vector_field = np.random.rand(
            mpi_construct.grid_dim,
            n_values * aspect_ratio[0],
            n_values * aspect_ratio[1],
            n_values * aspect_ratio[2],
        ).astype(real_t)
        inv_dx = real_t(0.1)
    else:
        ref_vector_field = None
        inv_dx = None
    ref_vector_field = mpi_construct.grid.bcast(ref_vector_field, root=0)
    inv_dx = mpi_construct.grid.bcast(inv_dx, root=0)

    # scatter global field
    scatter_global_field(local_vector_field[0], ref_vector_field[0], mpi_construct)
    scatter_global_field(local_vector_field[1], ref_vector_field[1], mpi_construct)
    scatter_global_field(local_vector_field[2], ref_vector_field[2], mpi_construct)

    # compute the divergence
    divergence_pyst_mpi_kernel = gen_divergence_pyst_mpi_kernel_3d(
        real_t=real_t,
        mpi_construct=mpi_construct,
        ghost_exchange_communicator=mpi_ghost_exchange_communicator,
    )
    divergence_pyst_mpi_kernel(
        divergence=local_divergence,
        field=local_vector_field,
        inv_dx=inv_dx,
    )

    # gather back the divergence globally
    global_divergence = np.zeros_like(ref_vector_field[0])
    gather_local_field(global_divergence, local_divergence, mpi_construct)

    # assert correct
    if mpi_construct.rank == 0:
        divergence_pyst_kernel = gen_divergence_pyst_kernel_3d(real_t=real_t)
        ref_divergence = np.zeros_like(ref_vector_field[0])
        divergence_pyst_kernel(
            divergence=ref_divergence,
            field=ref_vector_field,
            inv_dx=inv_dx,
        )

        # check kernel_support for the divergence kernel
        assert (
            divergence_pyst_mpi_kernel.kernel_support == 1
        ), "Incorrect kernel support!"
        # check field correctness
        np.testing.assert_allclose(
            ref_divergence,
            global_divergence,
            atol=get_test_tol(precision),
        )
