import numpy as np
import pytest
from sopht.utils.precision import get_real_t, get_test_tol
from sopht_mpi.utils import (
    MPIConstruct2D,
    MPIGhostCommunicator2D,
    MPIFieldIOCommunicator2D,
)


@pytest.mark.mpi(group="MPI_utils")
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_mpi_field_io_gather_scatter(n_values, precision):
    real_t = get_real_t(precision)
    mpi_construct = MPIConstruct2D(
        grid_size_y=n_values, grid_size_x=n_values, real_t=real_t
    )
    offset = 1
    mpi_field_io_comm_with_offset_size_1 = MPIFieldIOCommunicator2D(
        field_offset=offset, mpi_construct=mpi_construct
    )
    global_field = np.random.rand(
        mpi_construct.global_grid_size[0], mpi_construct.global_grid_size[1]
    ).astype(real_t)
    ref_global_field = global_field.copy()
    local_field = np.zeros(
        (
            mpi_construct.local_grid_size[0] + 2 * offset,
            mpi_construct.local_grid_size[1] + 2 * offset,
        )
    ).astype(real_t)
    gather_local_field = mpi_field_io_comm_with_offset_size_1.gather_local_field
    scatter_global_field = mpi_field_io_comm_with_offset_size_1.scatter_global_field
    # scatter global field to other ranks
    scatter_global_field(local_field, ref_global_field, mpi_construct)
    # randomise global field after scatter
    global_field[...] = np.random.rand(
        mpi_construct.global_grid_size[0], mpi_construct.global_grid_size[1]
    ).astype(real_t)
    # reconstruct global field from local ranks
    gather_local_field(global_field, local_field, mpi_construct)
    if mpi_construct.rank == 0:
        np.testing.assert_allclose(
            ref_global_field, global_field, atol=get_test_tol(precision)
        )


@pytest.mark.mpi(group="MPI_utils")
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_mpi_ghost_blocking_communication(n_values, precision):
    real_t = get_real_t(precision)
    mpi_construct = MPIConstruct2D(
        grid_size_y=n_values, grid_size_x=n_values, periodic_flag=True, real_t=real_t
    )
    # extra width needed for kernel computation
    ghost_size = 1
    # extra width involved in the field storage (>= ghost_size)
    field_offset = 2 * ghost_size
    # variables to improve code readability
    field_offset_minus_ghost_size = field_offset - ghost_size
    field_offset_plus_ghost_size = field_offset + ghost_size
    mpi_ghost_exchange_communicator = MPIGhostCommunicator2D(
        ghost_size=ghost_size, field_offset=field_offset, mpi_construct=mpi_construct
    )
    # Set internal field to manufactured values
    np.random.seed(0)
    local_field = np.random.rand(
        mpi_construct.local_grid_size[0] + 2 * field_offset,
        mpi_construct.local_grid_size[1] + 2 * field_offset,
    ).astype(real_t)

    # ghost comm.
    mpi_ghost_exchange_communicator.blocking_exchange(local_field, mpi_construct)

    # check if comm. done rightly!
    np.testing.assert_allclose(
        local_field[
            field_offset:field_offset_plus_ghost_size, field_offset:-field_offset
        ],
        local_field[
            -field_offset : local_field.shape[0] - field_offset_minus_ghost_size,
            field_offset:-field_offset,
        ],
    )
    np.testing.assert_allclose(
        local_field[
            -field_offset_plus_ghost_size:-field_offset, field_offset:-field_offset
        ],
        local_field[
            field_offset_minus_ghost_size:field_offset, field_offset:-field_offset
        ],
    )
    np.testing.assert_allclose(
        local_field[
            field_offset:-field_offset, field_offset:field_offset_plus_ghost_size
        ],
        local_field[
            field_offset:-field_offset,
            -field_offset : local_field.shape[1] - field_offset_minus_ghost_size,
        ],
    )
    np.testing.assert_allclose(
        local_field[
            field_offset:-field_offset, -field_offset_plus_ghost_size:-field_offset
        ],
        local_field[
            field_offset:-field_offset, field_offset_minus_ghost_size:field_offset
        ],
    )


@pytest.mark.mpi(group="MPI_utils")
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_mpi_ghost_non_blocking_communication(n_values, precision):
    real_t = get_real_t(precision)
    mpi_construct = MPIConstruct2D(
        grid_size_y=n_values, grid_size_x=n_values, periodic_flag=True, real_t=real_t
    )
    # extra width needed for kernel computation
    ghost_size = 1
    # extra width involved in the field storage (>= ghost_size)
    field_offset = 2 * ghost_size
    # variables to improve code readability
    field_offset_minus_ghost_size = field_offset - ghost_size
    field_offset_plus_ghost_size = field_offset + ghost_size
    mpi_ghost_exchange_communicator = MPIGhostCommunicator2D(
        ghost_size=ghost_size, field_offset=field_offset, mpi_construct=mpi_construct
    )
    # Set internal field to manufactured values
    np.random.seed(0)
    local_field = np.random.rand(
        mpi_construct.local_grid_size[0] + 2 * field_offset,
        mpi_construct.local_grid_size[1] + 2 * field_offset,
    ).astype(real_t)

    # ghost comm.
    mpi_ghost_exchange_communicator.non_blocking_exchange_init(
        local_field, mpi_construct
    )
    mpi_ghost_exchange_communicator.non_blocking_exchange_finalise()

    # check if comm. done rightly!
    np.testing.assert_allclose(
        local_field[
            field_offset:field_offset_plus_ghost_size, field_offset:-field_offset
        ],
        local_field[
            -field_offset : local_field.shape[0] - field_offset_minus_ghost_size,
            field_offset:-field_offset,
        ],
    )
    np.testing.assert_allclose(
        local_field[
            -field_offset_plus_ghost_size:-field_offset, field_offset:-field_offset
        ],
        local_field[
            field_offset_minus_ghost_size:field_offset, field_offset:-field_offset
        ],
    )
    np.testing.assert_allclose(
        local_field[
            field_offset:-field_offset, field_offset:field_offset_plus_ghost_size
        ],
        local_field[
            field_offset:-field_offset,
            -field_offset : local_field.shape[1] - field_offset_minus_ghost_size,
        ],
    )
    np.testing.assert_allclose(
        local_field[
            field_offset:-field_offset, -field_offset_plus_ghost_size:-field_offset
        ],
        local_field[
            field_offset:-field_offset, field_offset_minus_ghost_size:field_offset
        ],
    )
