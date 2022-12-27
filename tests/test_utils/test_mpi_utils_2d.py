import numpy as np
import pytest
from sopht.utils.precision import get_real_t
from sopht.utils.field import VectorField
from sopht_mpi.utils import (
    MPIConstruct2D,
    MPIGhostCommunicator2D,
    MPIFieldCommunicator2D,
    MPILagrangianFieldCommunicator2D,
)
from mpi4py import MPI


@pytest.mark.mpi(group="MPI_utils", min_size=4)
@pytest.mark.parametrize("ghost_size", [1, 2, 3])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rank_distribution", [(1, 0), (0, 1)])
@pytest.mark.parametrize("aspect_ratio", [(1, 1), (1, 2), (2, 1)])
@pytest.mark.parametrize("master_rank", [0, 1])
def test_mpi_field_gather_scatter(
    ghost_size, precision, rank_distribution, aspect_ratio, master_rank
):
    n_values = 32
    real_t = get_real_t(precision)
    mpi_construct = MPIConstruct2D(
        grid_size_y=n_values * aspect_ratio[0],
        grid_size_x=n_values * aspect_ratio[1],
        real_t=real_t,
        rank_distribution=rank_distribution,
    )
    mpi_field_communicator = MPIFieldCommunicator2D(
        ghost_size=ghost_size, mpi_construct=mpi_construct, master_rank=master_rank
    )
    global_scalar_field = np.random.rand(
        mpi_construct.global_grid_size[0], mpi_construct.global_grid_size[1]
    ).astype(real_t)
    global_vector_field = np.random.rand(
        mpi_construct.grid_dim,
        mpi_construct.global_grid_size[0],
        mpi_construct.global_grid_size[1],
    ).astype(real_t)
    ref_global_scalar_field = global_scalar_field.copy()
    ref_global_vector_field = global_vector_field.copy()
    local_scalar_field = np.zeros(
        (
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
        )
    ).astype(real_t)
    local_vector_field = np.zeros(
        (
            mpi_construct.grid_dim,
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
        )
    ).astype(real_t)
    gather_local_scalar_field = mpi_field_communicator.gather_local_scalar_field
    scatter_global_scalar_field = mpi_field_communicator.scatter_global_scalar_field
    gather_local_vector_field = mpi_field_communicator.gather_local_vector_field
    scatter_global_vector_field = mpi_field_communicator.scatter_global_vector_field
    # scatter global field to other ranks
    scatter_global_scalar_field(local_scalar_field, ref_global_scalar_field)
    scatter_global_vector_field(local_vector_field, ref_global_vector_field)
    # randomise global field after scatter
    global_scalar_field[...] = np.random.rand(
        mpi_construct.global_grid_size[0], mpi_construct.global_grid_size[1]
    ).astype(real_t)
    global_vector_field[...] = np.random.rand(
        mpi_construct.grid_dim,
        mpi_construct.global_grid_size[0],
        mpi_construct.global_grid_size[1],
    ).astype(real_t)
    # reconstruct global field from local ranks
    gather_local_scalar_field(global_scalar_field, local_scalar_field)
    gather_local_vector_field(global_vector_field, local_vector_field)
    if mpi_construct.rank == master_rank:
        np.testing.assert_allclose(ref_global_scalar_field, global_scalar_field)
        np.testing.assert_allclose(ref_global_vector_field, global_vector_field)


@pytest.mark.mpi(group="MPI_utils", min_size=4)
@pytest.mark.parametrize("ghost_size", [1, 2, 3])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rank_distribution", [(1, 0), (0, 1)])
@pytest.mark.parametrize("aspect_ratio", [(1, 1), (1, 2), (2, 1)])
def test_mpi_ghost_communication(
    ghost_size, precision, rank_distribution, aspect_ratio
):
    n_values = 32
    real_t = get_real_t(precision)
    mpi_construct = MPIConstruct2D(
        grid_size_y=n_values * aspect_ratio[0],
        grid_size_x=n_values * aspect_ratio[1],
        periodic_flag=True,
        real_t=real_t,
        rank_distribution=rank_distribution,
    )
    # extra width needed for kernel computation
    mpi_ghost_exchange_communicator = MPIGhostCommunicator2D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )
    # Set internal field to manufactured values
    np.random.seed(0)
    local_scalar_field = np.random.rand(
        mpi_construct.local_grid_size[0] + 2 * ghost_size,
        mpi_construct.local_grid_size[1] + 2 * ghost_size,
    ).astype(real_t)
    local_vector_field = np.random.rand(
        mpi_construct.grid_dim,
        mpi_construct.local_grid_size[0] + 2 * ghost_size,
        mpi_construct.local_grid_size[1] + 2 * ghost_size,
    ).astype(real_t)

    # ghost comm.
    mpi_ghost_exchange_communicator.exchange_scalar_field_init(local_scalar_field)
    mpi_ghost_exchange_communicator.exchange_vector_field_init(local_vector_field)
    mpi_ghost_exchange_communicator.exchange_finalise()

    # check if comm. done rightly!
    # Test scalar field
    np.testing.assert_allclose(
        local_scalar_field[ghost_size : 2 * ghost_size, ghost_size:-ghost_size],
        local_scalar_field[
            -ghost_size : local_scalar_field.shape[0], ghost_size:-ghost_size
        ],
    )
    np.testing.assert_allclose(
        local_scalar_field[-2 * ghost_size : -ghost_size, ghost_size:-ghost_size],
        local_scalar_field[0:ghost_size, ghost_size:-ghost_size],
    )
    np.testing.assert_allclose(
        local_scalar_field[ghost_size:-ghost_size, ghost_size : 2 * ghost_size],
        local_scalar_field[
            ghost_size:-ghost_size, -ghost_size : local_scalar_field.shape[1]
        ],
    )
    np.testing.assert_allclose(
        local_scalar_field[ghost_size:-ghost_size, -2 * ghost_size : -ghost_size],
        local_scalar_field[ghost_size:-ghost_size, 0:ghost_size],
    )
    # Test vector field
    np.testing.assert_allclose(
        local_vector_field[:, ghost_size : 2 * ghost_size, ghost_size:-ghost_size],
        local_vector_field[
            :, -ghost_size : local_vector_field.shape[1], ghost_size:-ghost_size
        ],
    )
    np.testing.assert_allclose(
        local_vector_field[:, -2 * ghost_size : -ghost_size, ghost_size:-ghost_size],
        local_vector_field[:, 0:ghost_size, ghost_size:-ghost_size],
    )
    np.testing.assert_allclose(
        local_vector_field[:, ghost_size:-ghost_size, ghost_size : 2 * ghost_size],
        local_vector_field[
            :, ghost_size:-ghost_size, -ghost_size : local_vector_field.shape[2]
        ],
    )
    np.testing.assert_allclose(
        local_vector_field[:, ghost_size:-ghost_size, -2 * ghost_size : -ghost_size],
        local_vector_field[:, ghost_size:-ghost_size, 0:ghost_size],
    )


@pytest.mark.mpi(group="MPI_utils", min_size=4)
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rank_distribution", [(1, 0), (0, 1)])
@pytest.mark.parametrize("aspect_ratio", [(1, 1), (1, 2), (2, 1)])
def test_mpi_lagrangian_field_map(precision, rank_distribution, aspect_ratio):
    # Eulerian grid stuff
    n_values = 32
    real_t = get_real_t(precision)
    grid_size_y = n_values * aspect_ratio[0]
    grid_size_x = n_values * aspect_ratio[1]
    mpi_construct = MPIConstruct2D(
        grid_size_y=grid_size_y,
        grid_size_x=grid_size_x,
        real_t=real_t,
        rank_distribution=rank_distribution,
    )
    x_range = real_t(1.0)
    y_range = x_range * grid_size_y / grid_size_x
    eul_grid_dx = real_t(x_range / grid_size_x)
    eul_grid_coord_shift = real_t(eul_grid_dx / 2.0)

    # Lagrangian grid stuff
    # Define a master rank that "owns" the lagrangian grid
    # It serves as the central process where points are gathered to / scattered from
    master_rank = 0
    mpi_lagrangian_field_communicator = MPILagrangianFieldCommunicator2D(
        eul_grid_dx=eul_grid_dx,
        eul_grid_coord_shift=eul_grid_coord_shift,
        mpi_construct=mpi_construct,
        master_rank=master_rank,
        real_t=real_t,
    )

    # Initialize fields for testing
    # Sample a diagonal line across domain
    global_lagrangian_positions = np.tile(
        np.arange(
            2 * eul_grid_coord_shift,
            x_range - 2 * eul_grid_coord_shift,
            eul_grid_dx / 2.0,
        ),
        [mpi_lagrangian_field_communicator.grid_dim, 1],
    ).astype(real_t)
    # rescale to spread them across the scaled eulerian domain
    global_lagrangian_positions[VectorField.x_axis_idx(), :] *= x_range
    global_lagrangian_positions[VectorField.y_axis_idx(), :] *= y_range

    # Map the lagrangian nodes to respective ranks
    mpi_lagrangian_field_communicator.map_lagrangian_nodes_based_on_position(
        global_lag_positions=global_lagrangian_positions
    )
    # Get locally mapped lagrangian grids
    rank_address = mpi_lagrangian_field_communicator.rank_address
    idx = np.where(rank_address == mpi_construct.rank)[0]
    local_lagrangian_positions = global_lagrangian_positions[:, idx]

    # Define local bounds
    local_grid_size = mpi_construct.local_grid_size
    substart_idx = mpi_construct.grid.coords * local_grid_size
    subend_idx = substart_idx + local_grid_size
    substart_y, substart_x = substart_idx * eul_grid_dx + eul_grid_coord_shift
    subend_y, subend_x = subend_idx * eul_grid_dx + eul_grid_coord_shift

    # Check locally mapped lagrangian nodes are within local eulerian grid bounds
    assert np.all(
        local_lagrangian_positions[VectorField.x_axis_idx(), :] >= substart_x
    ) & np.all(local_lagrangian_positions[VectorField.x_axis_idx(), :] < subend_x)
    assert np.all(
        local_lagrangian_positions[VectorField.y_axis_idx(), :] >= substart_y
    ) & np.all(local_lagrangian_positions[VectorField.y_axis_idx(), :] < subend_y)

    # Ensure that we taken all lagrangian points into account
    local_num_lag_nodes = local_lagrangian_positions.shape[1]
    assert local_num_lag_nodes == mpi_lagrangian_field_communicator.local_num_lag_nodes
    global_num_lag_nodes = mpi_construct.grid.allreduce(local_num_lag_nodes, op=MPI.SUM)
    assert global_num_lag_nodes == global_lagrangian_positions.shape[1]


@pytest.mark.mpi(group="MPI_utils", min_size=4)
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rank_distribution", [(1, 0), (0, 1)])
@pytest.mark.parametrize("aspect_ratio", [(1, 1), (1, 2), (2, 1)])
def test_mpi_lagrangian_field_gather_scatter(
    precision, rank_distribution, aspect_ratio
):
    # Eulerian grid stuff
    n_values = 32
    real_t = get_real_t(precision)
    grid_size_y = n_values * aspect_ratio[0]
    grid_size_x = n_values * aspect_ratio[1]
    mpi_construct = MPIConstruct2D(
        grid_size_y=grid_size_y,
        grid_size_x=grid_size_x,
        real_t=real_t,
        rank_distribution=rank_distribution,
    )
    x_range = real_t(1.0)
    y_range = x_range * grid_size_y / grid_size_x
    eul_grid_dx = real_t(x_range / grid_size_x)
    eul_grid_coord_shift = real_t(eul_grid_dx / 2.0)

    # Lagrangian grid stuff
    # Define a master rank that "owns" the lagrangian grid
    # It serves as the central process where points are gathered to / scattered from
    master_rank = 0
    mpi_lagrangian_field_communicator = MPILagrangianFieldCommunicator2D(
        eul_grid_dx=eul_grid_dx,
        eul_grid_coord_shift=eul_grid_coord_shift,
        mpi_construct=mpi_construct,
        master_rank=master_rank,
        real_t=real_t,
    )

    # Initialize fields for testing
    global_num_lag_nodes = 100
    # generate random lagrangian nodes within eulerian bounds
    global_lagrangian_positions = np.zeros(
        (mpi_lagrangian_field_communicator.grid_dim, global_num_lag_nodes)
    ).astype(real_t)
    global_lagrangian_positions[VectorField.x_axis_idx(), :] = np.random.uniform(
        2 * eul_grid_coord_shift,
        x_range - 2 * eul_grid_coord_shift,
        global_num_lag_nodes,
    )
    global_lagrangian_positions[VectorField.y_axis_idx(), :] = np.random.uniform(
        2 * eul_grid_coord_shift,
        y_range - 2 * eul_grid_coord_shift,
        global_num_lag_nodes,
    )
    # generate random velocities at these lagrangian nodes
    global_lagrangian_velocities = np.random.rand(
        mpi_lagrangian_field_communicator.grid_dim, global_num_lag_nodes
    ).astype(real_t)
    ref_global_lagrangian_positions = global_lagrangian_positions.copy()
    ref_global_lagrangian_velocities = global_lagrangian_velocities.copy()

    # Map the lagrangian nodes to respective ranks
    mpi_lagrangian_field_communicator.map_lagrangian_nodes_based_on_position(
        global_lag_positions=global_lagrangian_positions
    )

    local_lagrangian_positions = np.zeros(
        (
            mpi_lagrangian_field_communicator.grid_dim,
            mpi_lagrangian_field_communicator.local_num_lag_nodes,
        )
    ).astype(real_t)
    local_lagrangian_velocities = np.zeros_like(local_lagrangian_positions)

    # scatter global field to other ranks
    mpi_lagrangian_field_communicator.scatter_global_field(
        local_lag_field=local_lagrangian_positions,
        global_lag_field=ref_global_lagrangian_positions,
    )
    mpi_lagrangian_field_communicator.scatter_global_field(
        local_lag_field=local_lagrangian_velocities,
        global_lag_field=ref_global_lagrangian_velocities,
    )

    # randomise global fields after scatter
    global_lagrangian_positions[...] = np.random.rand(
        mpi_lagrangian_field_communicator.grid_dim, global_num_lag_nodes
    ).astype(real_t)
    global_lagrangian_velocities[...] = np.random.rand(
        mpi_lagrangian_field_communicator.grid_dim, global_num_lag_nodes
    ).astype(real_t)
    # reconstruct global field from local ranks
    mpi_lagrangian_field_communicator.gather_local_field(
        global_lag_field=global_lagrangian_positions,
        local_lag_field=local_lagrangian_positions,
    )
    mpi_lagrangian_field_communicator.gather_local_field(
        global_lag_field=global_lagrangian_velocities,
        local_lag_field=local_lagrangian_velocities,
    )
    if mpi_construct.rank == mpi_lagrangian_field_communicator.master_rank:
        np.testing.assert_allclose(
            ref_global_lagrangian_positions, global_lagrangian_positions
        )
        np.testing.assert_allclose(
            ref_global_lagrangian_velocities, global_lagrangian_velocities
        )
