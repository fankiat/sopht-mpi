import numpy as np
import pytest
from sopht.utils.precision import get_real_t, get_test_tol
from sopht_mpi.utils import (
    MPIConstruct3D,
    MPIGhostCommunicator3D,
)


@pytest.mark.mpi(group="MPI_utils", min_size=2)
@pytest.mark.parametrize("ghost_size", [1, 2, 3])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize(
    "rank_distribution",
    [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
)
def test_mpi_ghost_blocking_communication(ghost_size, precision, rank_distribution):
    n_values = 8
    real_t = get_real_t(precision)
    mpi_construct = MPIConstruct3D(
        grid_size_z=n_values,
        grid_size_y=n_values,
        grid_size_x=2 * n_values,
        periodic_flag=True,
        real_t=real_t,
        rank_distribution=rank_distribution,
    )
    # extra width needed for kernel computation
    ghost_size = 1
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
    mpi_ghost_exchange_communicator.blocking_exchange(local_field, mpi_construct)

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


@pytest.mark.mpi(group="MPI_utils", min_size=2)
@pytest.mark.parametrize("ghost_size", [1, 2, 3])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize(
    "rank_distribution",
    [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
)
def test_mpi_ghost_non_blocking_communication(ghost_size, precision, rank_distribution):
    n_values = 8
    real_t = get_real_t(precision)
    mpi_construct = MPIConstruct3D(
        grid_size_z=n_values,
        grid_size_y=n_values,
        grid_size_x=2 * n_values,
        periodic_flag=True,
        real_t=real_t,
        rank_distribution=rank_distribution,
    )
    # extra width needed for kernel computation
    ghost_size = 1
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
    mpi_ghost_exchange_communicator.non_blocking_exchange_init(
        local_field, mpi_construct
    )
    mpi_ghost_exchange_communicator.non_blocking_exchange_finalise()

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
