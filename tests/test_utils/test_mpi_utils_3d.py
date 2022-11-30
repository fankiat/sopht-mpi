import numpy as np
import pytest
from sopht.utils.precision import get_real_t
from sopht_mpi.utils import (
    MPIConstruct3D,
    MPIGhostCommunicator3D,
    MPIFieldCommunicator3D,
)


@pytest.mark.mpi(group="MPI_utils", min_size=2)
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
@pytest.mark.parametrize("master_rank", [0, 1])
def test_mpi_field_gather_scatter(
    ghost_size, precision, rank_distribution, aspect_ratio, master_rank
):
    n_values = 32
    real_t = get_real_t(precision)
    mpi_construct = MPIConstruct3D(
        grid_size_z=n_values * aspect_ratio[0],
        grid_size_y=n_values * aspect_ratio[1],
        grid_size_x=n_values * aspect_ratio[2],
        real_t=real_t,
        rank_distribution=rank_distribution,
    )
    mpi_field_communicator = MPIFieldCommunicator3D(
        ghost_size=ghost_size, mpi_construct=mpi_construct, master_rank=master_rank
    )
    global_field = np.random.rand(
        mpi_construct.global_grid_size[0],
        mpi_construct.global_grid_size[1],
        mpi_construct.global_grid_size[2],
    ).astype(real_t)
    ref_global_field = global_field.copy()
    local_field = np.zeros(
        (
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
            mpi_construct.local_grid_size[2] + 2 * ghost_size,
        )
    ).astype(real_t)
    gather_local_field = mpi_field_communicator.gather_local_field
    scatter_global_field = mpi_field_communicator.scatter_global_field
    # scatter global field to other ranks
    scatter_global_field(local_field, ref_global_field, mpi_construct)
    # randomise global field after scatter
    global_field[...] = np.random.rand(
        mpi_construct.global_grid_size[0],
        mpi_construct.global_grid_size[1],
        mpi_construct.global_grid_size[2],
    ).astype(real_t)
    # reconstruct global field from local ranks
    gather_local_field(global_field, local_field, mpi_construct)
    if mpi_construct.rank == master_rank:
        np.testing.assert_allclose(ref_global_field, global_field)


@pytest.mark.mpi(group="MPI_utils", min_size=2)
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
def test_mpi_ghost_communication(
    ghost_size, precision, rank_distribution, aspect_ratio
):
    n_values = 32
    real_t = get_real_t(precision)
    mpi_construct = MPIConstruct3D(
        grid_size_z=n_values * aspect_ratio[0],
        grid_size_y=n_values * aspect_ratio[1],
        grid_size_x=n_values * aspect_ratio[2],
        periodic_flag=True,
        real_t=real_t,
        rank_distribution=rank_distribution,
    )
    # extra width needed for kernel computation
    mpi_ghost_exchange_communicator = MPIGhostCommunicator3D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )
    # Set internal field to manufactured values
    np.random.seed(0)
    local_field = np.random.rand(
        mpi_construct.local_grid_size[0] + 2 * ghost_size,
        mpi_construct.local_grid_size[1] + 2 * ghost_size,
        mpi_construct.local_grid_size[2] + 2 * ghost_size,
    ).astype(real_t)

    # ghost comm.
    mpi_ghost_exchange_communicator.exchange_init(local_field, mpi_construct)
    mpi_ghost_exchange_communicator.exchange_finalise()

    # check if comm. done rightly!
    # Along X: comm with previous block
    np.testing.assert_allclose(
        local_field[
            ghost_size:-ghost_size, ghost_size:-ghost_size, ghost_size : 2 * ghost_size
        ],
        local_field[
            ghost_size:-ghost_size,
            ghost_size:-ghost_size,
            -ghost_size : local_field.shape[2],
        ],
    )
    # Along X: comm with next block
    np.testing.assert_allclose(
        local_field[
            ghost_size:-ghost_size,
            ghost_size:-ghost_size,
            -2 * ghost_size : -ghost_size,
        ],
        local_field[ghost_size:-ghost_size, ghost_size:-ghost_size, 0:ghost_size],
    )
    # Along Y: comm with previous block
    np.testing.assert_allclose(
        local_field[
            ghost_size:-ghost_size, ghost_size : 2 * ghost_size, ghost_size:-ghost_size
        ],
        local_field[
            ghost_size:-ghost_size,
            -ghost_size : local_field.shape[1],
            ghost_size:-ghost_size,
        ],
    )
    # Along Y: comm with next block
    np.testing.assert_allclose(
        local_field[
            ghost_size:-ghost_size,
            -2 * ghost_size : -ghost_size,
            ghost_size:-ghost_size,
        ],
        local_field[ghost_size:-ghost_size, 0:ghost_size, ghost_size:-ghost_size],
    )
    # Along Z: comm with previous block
    np.testing.assert_allclose(
        local_field[
            ghost_size : 2 * ghost_size, ghost_size:-ghost_size, ghost_size:-ghost_size
        ],
        local_field[
            -ghost_size : local_field.shape[0],
            ghost_size:-ghost_size,
            ghost_size:-ghost_size,
        ],
    )
    # Along Z: comm with next block
    np.testing.assert_allclose(
        local_field[
            -2 * ghost_size : -ghost_size,
            ghost_size:-ghost_size,
            ghost_size:-ghost_size,
        ],
        local_field[0:ghost_size, ghost_size:-ghost_size, ghost_size:-ghost_size],
    )
