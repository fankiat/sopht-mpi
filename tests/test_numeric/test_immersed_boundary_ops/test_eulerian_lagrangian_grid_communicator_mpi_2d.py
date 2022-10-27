import numpy as np
import pytest
from sopht_mpi.numeric.immersed_boundary_ops import (
    EulerianLagrangianGridCommunicatorMPI2D,
)
from sopht.numeric.immersed_boundary_ops import EulerianLagrangianGridCommunicator2D
from sopht_mpi.utils import (
    MPIConstruct2D,
    MPILagrangianFieldCommunicator2D,
)
from sopht.utils.precision import get_real_t, get_test_tol


@pytest.mark.mpi(group="MPI_immersed_boundary_ops_2d", min_size=4)
@pytest.mark.parametrize("ghost_size", [1, 2, 3])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rank_distribution", [(1, 0), (0, 1)])
@pytest.mark.parametrize("aspect_ratio", [(1, 1), (1, 2), (2, 1)])
def test_mpi_local_eulerian_grid_support_of_lagrangian_grid_2d(
    ghost_size, precision, rank_distribution, aspect_ratio
):
    n_values = 32
    real_t = get_real_t(precision)
    grid_dim = 2
    eul_grid_size_y = n_values * aspect_ratio[0]
    eul_grid_size_x = n_values * aspect_ratio[1]
    eul_grid_size = eul_grid_size_x
    eul_domain_size = real_t(1.0)
    eul_grid_dx = real_t(eul_domain_size / eul_grid_size)
    eul_grid_coord_shift = real_t(0.5 * eul_grid_dx)
    interp_kernel_width = 2
    num_lag_nodes = 3

    # 1. Generate reference solution from sopht-backend
    eul_lag_communicator = EulerianLagrangianGridCommunicator2D(
        dx=eul_grid_dx,
        eul_grid_coord_shift=eul_grid_coord_shift,
        num_lag_nodes=num_lag_nodes,
        interp_kernel_width=interp_kernel_width,
        real_t=real_t,
    )
    # init lag. grid around center of the domain
    # this allows testing for the case where rank on first and last quarter (when there
    # are 4 processes) does not contain any lagrangian nodes AND when the point lies at
    # the boundary of MPI domain decomposition
    ref_nearest_eul_grid_index_to_lag_grid = np.empty((grid_dim, num_lag_nodes)).astype(
        int
    )
    ref_nearest_eul_grid_index_to_lag_grid[0] = np.arange(
        eul_grid_size_x // 2 - 1, eul_grid_size_x // 2 - 1 + num_lag_nodes
    ).astype(int)
    ref_nearest_eul_grid_index_to_lag_grid[1] = np.arange(
        eul_grid_size_y // 2 - 1, eul_grid_size_y // 2 - 1 + num_lag_nodes
    ).astype(int)
    ref_lag_positions = (
        ref_nearest_eul_grid_index_to_lag_grid * eul_grid_dx + eul_grid_coord_shift
    ).astype(real_t)

    # find interpolation zone support for the lag. grid
    ref_local_eul_grid_support_of_lag_grid = np.zeros(
        (grid_dim, 2 * interp_kernel_width, 2 * interp_kernel_width, num_lag_nodes)
    ).astype(real_t)
    eul_lag_communicator.local_eulerian_grid_support_of_lagrangian_grid_kernel(
        ref_local_eul_grid_support_of_lag_grid,
        ref_nearest_eul_grid_index_to_lag_grid,
        ref_lag_positions,
    )

    # 2. Initialize MPI related stuff
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=eul_grid_size_y,
        grid_size_x=eul_grid_size_x,
        real_t=real_t,
        rank_distribution=rank_distribution,
    )

    # Lagrangian grid inter-rank MPI communicator
    master_rank = 0
    mpi_lagrangian_field_communicator = MPILagrangianFieldCommunicator2D(
        eul_grid_dx=eul_grid_dx,
        eul_grid_shift=eul_grid_coord_shift,
        mpi_construct=mpi_construct,
        master_rank=master_rank,
        real_t=real_t,
    )
    # Map lagrangian nodes to ranks
    mpi_lagrangian_field_communicator.map_lagrangian_nodes_based_on_position(
        ref_lag_positions
    )
    rank_address = mpi_lagrangian_field_communicator.rank_address

    # Initialize Eulerian-Lagrangian grid numerics communicator based on mpi local eul
    # grid coord shift
    mpi_eul_lag_communicator = EulerianLagrangianGridCommunicatorMPI2D(
        dx=eul_grid_dx,
        eul_grid_coord_shift=eul_grid_coord_shift,
        interp_kernel_width=interp_kernel_width,
        real_t=real_t,
        mpi_construct=mpi_construct,
        ghost_size=ghost_size,
    )

    # 3. Compute solution using MPI implementation
    # since all the ranks have the same reference solution, we can just mask them out
    # the "mpi rank local" data after mapping the lag nodes to their respective ranks
    mask = np.where(rank_address == mpi_lagrangian_field_communicator.rank)[0]
    mpi_local_lag_positions = ref_lag_positions[..., mask]
    mpi_local_nearest_eul_grid_index_to_lag_grid = np.zeros(
        (grid_dim, mpi_lagrangian_field_communicator.local_num_lag_nodes)
    ).astype(int)
    mpi_local_local_eul_grid_support_of_lag_grid = np.zeros(
        (
            grid_dim,
            2 * interp_kernel_width,
            2 * interp_kernel_width,
            mpi_lagrangian_field_communicator.local_num_lag_nodes,
        )
    ).astype(real_t)

    mpi_eul_lag_communicator.local_eulerian_grid_support_of_lagrangian_grid_kernel(
        mpi_local_local_eul_grid_support_of_lag_grid,
        mpi_local_nearest_eul_grid_index_to_lag_grid,
        mpi_local_lag_positions,
    )

    # We need to shift the indices so that its based on the ref index frame and not the
    # rank local index frame
    mpi_substart_idx = mpi_eul_lag_communicator.mpi_substart_idx
    mpi_local_nearest_eul_grid_index_to_lag_grid = (
        mpi_local_nearest_eul_grid_index_to_lag_grid
        - ghost_size
        + mpi_substart_idx.reshape(grid_dim, 1)
    ).astype(int)

    # 4. Test and compare values
    np.testing.assert_allclose(
        ref_nearest_eul_grid_index_to_lag_grid[..., mask],
        mpi_local_nearest_eul_grid_index_to_lag_grid,
        atol=get_test_tol(precision),
    )
    np.testing.assert_allclose(
        ref_local_eul_grid_support_of_lag_grid[..., mask],
        mpi_local_local_eul_grid_support_of_lag_grid,
        atol=get_test_tol(precision),
    )


@pytest.mark.mpi(group="MPI_immersed_boundary_ops_2d", min_size=4)
@pytest.mark.parametrize("ghost_size", [pytest.param(1, marks=pytest.mark.xfail), 2, 3])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rank_distribution", [(1, 0), (0, 1)])
@pytest.mark.parametrize("aspect_ratio", [(1, 1), (1, 2), (2, 1)])
def test_mpi_eulerian_to_lagrangian_grid_interpolation_kernel_2d(
    ghost_size, precision, rank_distribution, aspect_ratio
):
    n_values = 32
    real_t = get_real_t(precision)
    grid_dim = 2
    eul_grid_size_y = n_values * aspect_ratio[0]
    eul_grid_size_x = n_values * aspect_ratio[1]
    eul_grid_size = eul_grid_size_x
    eul_domain_size = real_t(1.0)
    eul_grid_dx = real_t(eul_domain_size / eul_grid_size)
    eul_grid_coord_shift = real_t(0.5 * eul_grid_dx)
    interp_kernel_width = 2
    num_lag_nodes = 3

    # 1. Generate reference solution from sopht-backend
    eul_lag_communicator = EulerianLagrangianGridCommunicator2D(
        dx=eul_grid_dx,
        eul_grid_coord_shift=eul_grid_coord_shift,
        num_lag_nodes=num_lag_nodes,
        interp_kernel_width=interp_kernel_width,
        real_t=real_t,
    )
    eulerian_to_lagrangian_grid_interpolation_kernel = (
        eul_lag_communicator.eulerian_to_lagrangian_grid_interpolation_kernel
    )
    # init lag. grid around center of the domain
    # this allows testing for the case where rank on first and last quarter (when there
    # are 4 processes) does not contain any lagrangian nodes AND when the point lies at
    # the boundary of MPI domain decomposition
    ref_nearest_eul_grid_index_to_lag_grid = np.empty((grid_dim, num_lag_nodes)).astype(
        int
    )
    ref_nearest_eul_grid_index_to_lag_grid[0] = np.arange(
        eul_grid_size_x // 2 - 1, eul_grid_size_x // 2 - 1 + num_lag_nodes
    ).astype(int)
    ref_nearest_eul_grid_index_to_lag_grid[1] = np.arange(
        eul_grid_size_y // 2 - 1, eul_grid_size_y // 2 - 1 + num_lag_nodes
    ).astype(int)
    ref_lag_positions = (
        ref_nearest_eul_grid_index_to_lag_grid * eul_grid_dx + eul_grid_coord_shift
    ).astype(real_t)
    # set interp weight as a series of 0 to 4 * interp_kernel_width ** 2 - 1
    ref_interp_weights = np.arange(0, 4 * interp_kernel_width**2).reshape(
        2 * interp_kernel_width, 2 * interp_kernel_width, 1
    )
    ref_interp_weights = np.tile(ref_interp_weights, reps=(1, 1, num_lag_nodes)).astype(
        real_t
    )
    ref_lag_grid_field = np.zeros((num_lag_nodes), dtype=real_t)
    ref_eul_grid_field = np.ones((eul_grid_size_y, eul_grid_size_x), dtype=real_t)
    eulerian_to_lagrangian_grid_interpolation_kernel(
        ref_lag_grid_field,
        ref_eul_grid_field,
        ref_interp_weights,
        ref_nearest_eul_grid_index_to_lag_grid,
    )

    # 2. Initialize MPI related stuff
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=eul_grid_size_y,
        grid_size_x=eul_grid_size_x,
        real_t=real_t,
        rank_distribution=rank_distribution,
    )

    # Lagrangian grid inter-rank MPI communicator
    master_rank = 0
    mpi_lagrangian_field_communicator = MPILagrangianFieldCommunicator2D(
        eul_grid_dx=eul_grid_dx,
        eul_grid_shift=eul_grid_coord_shift,
        mpi_construct=mpi_construct,
        master_rank=master_rank,
        real_t=real_t,
    )
    # Map lagrangian nodes to ranks
    mpi_lagrangian_field_communicator.map_lagrangian_nodes_based_on_position(
        ref_lag_positions
    )
    rank_address = mpi_lagrangian_field_communicator.rank_address

    # Initialize Eulerian-Lagrangian grid numerics communicator based on mpi local eul
    # grid coord shift
    mpi_eul_lag_communicator = EulerianLagrangianGridCommunicatorMPI2D(
        dx=eul_grid_dx,
        eul_grid_coord_shift=eul_grid_coord_shift,
        interp_kernel_width=interp_kernel_width,
        real_t=real_t,
        mpi_construct=mpi_construct,
        ghost_size=ghost_size,
    )

    # 3. Compute solution using MPI implementation
    # since all the ranks have the same reference solution, we can just mask them out
    # the "mpi rank local" data after mapping the lag nodes to their respective ranks
    mask = np.where(rank_address == mpi_lagrangian_field_communicator.rank)[0]
    mpi_local_eul_grid_field = np.ones(
        mpi_construct.local_grid_size + 2 * ghost_size, dtype=real_t
    )
    mpi_local_interp_weights = ref_interp_weights[..., mask]
    # indices needs to be offset with mpi local reference index
    mpi_substart_idx = mpi_eul_lag_communicator.mpi_substart_idx
    mpi_local_nearest_eul_grid_index_to_lag_grid = (
        ref_nearest_eul_grid_index_to_lag_grid[..., mask]
        + ghost_size
        - mpi_substart_idx.reshape(grid_dim, 1)
    )
    mpi_local_lag_grid_field = np.zeros(
        (mpi_lagrangian_field_communicator.local_num_lag_nodes), dtype=real_t
    )
    mpi_eul_lag_communicator.eulerian_to_lagrangian_grid_interpolation_kernel(
        mpi_local_lag_grid_field,
        mpi_local_eul_grid_field,
        mpi_local_interp_weights,
        mpi_local_nearest_eul_grid_index_to_lag_grid,
    )

    np.testing.assert_allclose(
        ref_lag_grid_field[..., mask],
        mpi_local_lag_grid_field,
        atol=get_test_tol(precision),
    )


@pytest.mark.mpi(group="MPI_immersed_boundary_ops_2d", min_size=4)
@pytest.mark.parametrize("ghost_size", [pytest.param(1, marks=pytest.mark.xfail), 2, 3])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rank_distribution", [(1, 0), (0, 1)])
@pytest.mark.parametrize("aspect_ratio", [(1, 1), (1, 2), (2, 1)])
def test_mpi_vector_field_eul_to_lag_grid_interpolation_kernel_2d(
    ghost_size, precision, rank_distribution, aspect_ratio
):
    n_values = 32
    real_t = get_real_t(precision)
    grid_dim = 2
    n_components = grid_dim
    eul_grid_size_y = n_values * aspect_ratio[0]
    eul_grid_size_x = n_values * aspect_ratio[1]
    eul_grid_size = eul_grid_size_x
    eul_domain_size = real_t(1.0)
    eul_grid_dx = real_t(eul_domain_size / eul_grid_size)
    eul_grid_coord_shift = real_t(0.5 * eul_grid_dx)
    interp_kernel_width = 2
    num_lag_nodes = 3

    # 1. Generate reference solution from sopht-backend
    eul_lag_communicator = EulerianLagrangianGridCommunicator2D(
        dx=eul_grid_dx,
        eul_grid_coord_shift=eul_grid_coord_shift,
        num_lag_nodes=num_lag_nodes,
        interp_kernel_width=interp_kernel_width,
        n_components=2,
        real_t=real_t,
    )
    eulerian_to_lagrangian_grid_interpolation_kernel = (
        eul_lag_communicator.eulerian_to_lagrangian_grid_interpolation_kernel
    )
    # init lag. grid around center of the domain
    # this allows testing for the case where rank on first and last quarter (when there
    # are 4 processes) does not contain any lagrangian nodes AND when the point lies at
    # the boundary of MPI domain decomposition
    ref_nearest_eul_grid_index_to_lag_grid = np.empty((grid_dim, num_lag_nodes)).astype(
        int
    )
    ref_nearest_eul_grid_index_to_lag_grid[0] = np.arange(
        eul_grid_size_x // 2 - 1, eul_grid_size_x // 2 - 1 + num_lag_nodes
    ).astype(int)
    ref_nearest_eul_grid_index_to_lag_grid[1] = np.arange(
        eul_grid_size_y // 2 - 1, eul_grid_size_y // 2 - 1 + num_lag_nodes
    ).astype(int)
    ref_lag_positions = (
        ref_nearest_eul_grid_index_to_lag_grid * eul_grid_dx + eul_grid_coord_shift
    ).astype(real_t)
    # set interp weight as a series of 0 to 4 * interp_kernel_width ** 2 - 1
    ref_interp_weights = np.arange(0, 4 * interp_kernel_width**2).reshape(
        2 * interp_kernel_width, 2 * interp_kernel_width, 1
    )
    ref_interp_weights = np.tile(ref_interp_weights, reps=(1, 1, num_lag_nodes)).astype(
        real_t
    )
    ref_lag_grid_field = np.zeros((n_components, num_lag_nodes), dtype=real_t)
    ref_eul_grid_field = np.ones(
        (n_components, eul_grid_size_y, eul_grid_size_x), dtype=real_t
    )
    eulerian_to_lagrangian_grid_interpolation_kernel(
        ref_lag_grid_field,
        ref_eul_grid_field,
        ref_interp_weights,
        ref_nearest_eul_grid_index_to_lag_grid,
    )

    # 2. Initialize MPI related stuff
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=eul_grid_size_y,
        grid_size_x=eul_grid_size_x,
        real_t=real_t,
        rank_distribution=rank_distribution,
    )

    # Lagrangian grid inter-rank MPI communicator
    master_rank = 0
    mpi_lagrangian_field_communicator = MPILagrangianFieldCommunicator2D(
        eul_grid_dx=eul_grid_dx,
        eul_grid_shift=eul_grid_coord_shift,
        mpi_construct=mpi_construct,
        master_rank=master_rank,
        real_t=real_t,
    )
    # Map lagrangian nodes to ranks
    mpi_lagrangian_field_communicator.map_lagrangian_nodes_based_on_position(
        ref_lag_positions
    )
    rank_address = mpi_lagrangian_field_communicator.rank_address

    # Initialize Eulerian-Lagrangian grid numerics communicator based on mpi local eul
    # grid coord shift
    mpi_eul_lag_communicator = EulerianLagrangianGridCommunicatorMPI2D(
        dx=eul_grid_dx,
        eul_grid_coord_shift=eul_grid_coord_shift,
        interp_kernel_width=interp_kernel_width,
        n_components=2,
        real_t=real_t,
        mpi_construct=mpi_construct,
        ghost_size=ghost_size,
    )

    # 3. Compute solution using MPI implementation
    # since all the ranks have the same reference solution, we can just mask them out
    # the "mpi rank local" data after mapping the lag nodes to their respective ranks
    mask = np.where(rank_address == mpi_lagrangian_field_communicator.rank)[0]
    mpi_local_eul_grid_field = np.ones(
        (
            n_components,
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
        ),
        dtype=real_t,
    )
    mpi_local_interp_weights = ref_interp_weights[..., mask]
    # indices needs to be offset with mpi local reference index
    mpi_substart_idx = mpi_eul_lag_communicator.mpi_substart_idx
    mpi_local_nearest_eul_grid_index_to_lag_grid = (
        ref_nearest_eul_grid_index_to_lag_grid[..., mask]
        + ghost_size
        - mpi_substart_idx.reshape(grid_dim, 1)
    )
    mpi_local_lag_grid_field = np.zeros(
        (n_components, mpi_lagrangian_field_communicator.local_num_lag_nodes),
        dtype=real_t,
    )
    mpi_eul_lag_communicator.eulerian_to_lagrangian_grid_interpolation_kernel(
        mpi_local_lag_grid_field,
        mpi_local_eul_grid_field,
        mpi_local_interp_weights,
        mpi_local_nearest_eul_grid_index_to_lag_grid,
    )

    np.testing.assert_allclose(
        ref_lag_grid_field[..., mask],
        mpi_local_lag_grid_field,
        atol=get_test_tol(precision),
    )


@pytest.mark.mpi(group="MPI_immersed_boundary_ops_2d", min_size=4)
@pytest.mark.parametrize("ghost_size", [pytest.param(1, marks=pytest.mark.xfail), 2, 3])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rank_distribution", [(1, 0), (0, 1)])
@pytest.mark.parametrize("aspect_ratio", [(1, 1), (1, 2), (2, 1)])
def test_mpi_lagrangian_to_eulerian_grid_interpolation_kernel_2d(
    ghost_size,
    precision,
    rank_distribution,
    aspect_ratio,
):
    n_values = 32
    real_t = get_real_t(precision)
    grid_dim = 2
    eul_grid_size_y = n_values * aspect_ratio[0]
    eul_grid_size_x = n_values * aspect_ratio[1]
    eul_grid_size = eul_grid_size_x
    eul_domain_size = real_t(1.0)
    eul_grid_dx = real_t(eul_domain_size / eul_grid_size)
    eul_grid_coord_shift = real_t(0.5 * eul_grid_dx)
    interp_kernel_width = 2
    num_lag_nodes = 3

    # 1. Generate reference solution from sopht-backend
    eul_lag_communicator = EulerianLagrangianGridCommunicator2D(
        dx=eul_grid_dx,
        eul_grid_coord_shift=eul_grid_coord_shift,
        num_lag_nodes=num_lag_nodes,
        interp_kernel_width=interp_kernel_width,
        real_t=real_t,
    )
    lagrangian_to_eulerian_grid_interpolation_kernel = (
        eul_lag_communicator.lagrangian_to_eulerian_grid_interpolation_kernel
    )
    # init lag. grid around center of the domain
    # this allows testing for the case where rank on first and last quarter (when there
    # are 4 processes) does not contain any lagrangian nodes AND when the points lie
    # close to the boundary of MPI domain decomposition
    ref_nearest_eul_grid_index_to_lag_grid = np.empty((grid_dim, num_lag_nodes)).astype(
        int
    )
    ref_nearest_eul_grid_index_to_lag_grid[0] = np.arange(
        eul_grid_size_x // 2 - 1, eul_grid_size_x // 2 - 1 + num_lag_nodes
    ).astype(int)
    ref_nearest_eul_grid_index_to_lag_grid[1] = np.arange(
        eul_grid_size_y // 2 - 1, eul_grid_size_y // 2 - 1 + num_lag_nodes
    ).astype(int)
    # need ref lag positions to get rank address later
    ref_lag_positions = (
        ref_nearest_eul_grid_index_to_lag_grid * eul_grid_dx + eul_grid_coord_shift
    ).astype(real_t)
    # init interp weights as all ones, essentially this should lead to
    # interpolation spreading ones onto the Eulerian grid
    ref_interp_weights = np.ones(
        (2 * interp_kernel_width, 2 * interp_kernel_width, num_lag_nodes),
        dtype=real_t,
    )
    prefactor_lag_field = 2
    ref_lag_grid_field = prefactor_lag_field * np.ones((num_lag_nodes), dtype=real_t)
    ref_eul_grid_field = np.zeros((eul_grid_size_y, eul_grid_size_x), dtype=real_t)
    lagrangian_to_eulerian_grid_interpolation_kernel(
        ref_eul_grid_field,
        ref_lag_grid_field,
        ref_interp_weights,
        ref_nearest_eul_grid_index_to_lag_grid,
    )

    # 2. Initialize MPI related stuff
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=eul_grid_size_y,
        grid_size_x=eul_grid_size_x,
        real_t=real_t,
        rank_distribution=rank_distribution,
    )

    # Lagrangian grid inter-rank MPI communicator
    master_rank = 0
    mpi_lagrangian_field_communicator = MPILagrangianFieldCommunicator2D(
        eul_grid_dx=eul_grid_dx,
        eul_grid_shift=eul_grid_coord_shift,
        mpi_construct=mpi_construct,
        master_rank=master_rank,
        real_t=real_t,
    )
    # Map lagrangian nodes to ranks
    mpi_lagrangian_field_communicator.map_lagrangian_nodes_based_on_position(
        ref_lag_positions
    )
    rank_address = mpi_lagrangian_field_communicator.rank_address

    # Initialize Eulerian-Lagrangian grid numerics communicator based on mpi local eul
    # grid coord shift
    mpi_eul_lag_communicator = EulerianLagrangianGridCommunicatorMPI2D(
        dx=eul_grid_dx,
        eul_grid_coord_shift=eul_grid_coord_shift,
        interp_kernel_width=interp_kernel_width,
        real_t=real_t,
        mpi_construct=mpi_construct,
        ghost_size=ghost_size,
    )

    # 3. Compute solution using MPI implementation
    # since all the ranks have the same reference solution, we can just mask them out
    # the "mpi rank local" data after mapping the lag nodes to their respective ranks
    mask = np.where(rank_address == mpi_lagrangian_field_communicator.rank)[0]
    mpi_local_eul_grid_field = np.zeros(
        mpi_construct.local_grid_size + 2 * ghost_size, dtype=real_t
    )
    mpi_local_lag_grid_field = ref_lag_grid_field[..., mask]
    mpi_local_interp_weights = ref_interp_weights[..., mask]
    # indices needs to be offset with mpi local reference index
    mpi_substart_idx = mpi_eul_lag_communicator.mpi_substart_idx
    mpi_local_nearest_eul_grid_index_to_lag_grid = (
        ref_nearest_eul_grid_index_to_lag_grid[..., mask]
        + ghost_size
        - mpi_substart_idx.reshape(grid_dim, 1)
    )

    mpi_eul_lag_communicator.lagrangian_to_eulerian_grid_interpolation_kernel(
        mpi_local_eul_grid_field,
        mpi_local_lag_grid_field,
        mpi_local_interp_weights,
        mpi_local_nearest_eul_grid_index_to_lag_grid,
    )

    # Get corresponding local chunk of ref solution
    mpi_local_sol_idx = (
        slice(
            mpi_substart_idx[1], mpi_substart_idx[1] + mpi_construct.local_grid_size[0]
        ),
        slice(
            mpi_substart_idx[0], mpi_substart_idx[0] + mpi_construct.local_grid_size[1]
        ),
    )

    np.testing.assert_allclose(
        ref_eul_grid_field[mpi_local_sol_idx],
        mpi_local_eul_grid_field[ghost_size:-ghost_size, ghost_size:-ghost_size],
        atol=get_test_tol(precision),
    )


@pytest.mark.mpi(group="MPI_immersed_boundary_ops_2d", min_size=4)
@pytest.mark.parametrize("ghost_size", [pytest.param(1, marks=pytest.mark.xfail), 2, 3])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rank_distribution", [(1, 0), (0, 1)])
@pytest.mark.parametrize("aspect_ratio", [(1, 1), (1, 2), (2, 1)])
def test_mpi_vector_field_lag_to_eul_grid_interpolation_kernel_2d(
    ghost_size,
    precision,
    rank_distribution,
    aspect_ratio,
):
    n_values = 32
    real_t = get_real_t(precision)
    grid_dim = 2
    n_components = grid_dim
    eul_grid_size_y = n_values * aspect_ratio[0]
    eul_grid_size_x = n_values * aspect_ratio[1]
    eul_grid_size = eul_grid_size_x
    eul_domain_size = real_t(1.0)
    eul_grid_dx = real_t(eul_domain_size / eul_grid_size)
    eul_grid_coord_shift = real_t(0.5 * eul_grid_dx)
    interp_kernel_width = 2
    num_lag_nodes = 3

    # 1. Generate reference solution from sopht-backend
    eul_lag_communicator = EulerianLagrangianGridCommunicator2D(
        dx=eul_grid_dx,
        eul_grid_coord_shift=eul_grid_coord_shift,
        num_lag_nodes=num_lag_nodes,
        interp_kernel_width=interp_kernel_width,
        real_t=real_t,
        n_components=n_components,
    )
    lagrangian_to_eulerian_grid_interpolation_kernel = (
        eul_lag_communicator.lagrangian_to_eulerian_grid_interpolation_kernel
    )
    # init lag. grid around center of the domain
    # this allows testing for the case where rank on first and last quarter (when there
    # are 4 processes) does not contain any lagrangian nodes AND when the point lies at
    # the boundary of MPI domain decomposition
    ref_nearest_eul_grid_index_to_lag_grid = np.empty((grid_dim, num_lag_nodes)).astype(
        int
    )
    ref_nearest_eul_grid_index_to_lag_grid[0] = np.arange(
        eul_grid_size_x // 2 - 1, eul_grid_size_x // 2 - 1 + num_lag_nodes
    ).astype(int)
    ref_nearest_eul_grid_index_to_lag_grid[1] = np.arange(
        eul_grid_size_y // 2 - 1, eul_grid_size_y // 2 - 1 + num_lag_nodes
    ).astype(int)
    # need ref lag positions to get rank address later
    ref_lag_positions = (
        ref_nearest_eul_grid_index_to_lag_grid * eul_grid_dx + eul_grid_coord_shift
    ).astype(real_t)
    # init interp weights as all ones, essentially this should lead to
    # interpolation spreading ones onto the Eulerian grid
    ref_interp_weights = np.ones(
        (2 * interp_kernel_width, 2 * interp_kernel_width, num_lag_nodes),
        dtype=real_t,
    )
    prefactor_lag_field_y = 2
    prefactor_lag_field_x = 3
    ref_lag_grid_field = np.ones((n_components, num_lag_nodes), dtype=real_t)
    ref_lag_grid_field[0] *= prefactor_lag_field_y
    ref_lag_grid_field[1] *= prefactor_lag_field_x
    ref_eul_grid_field = np.zeros(
        (n_components, eul_grid_size_y, eul_grid_size_x), dtype=real_t
    )
    lagrangian_to_eulerian_grid_interpolation_kernel(
        ref_eul_grid_field,
        ref_lag_grid_field,
        ref_interp_weights,
        ref_nearest_eul_grid_index_to_lag_grid,
    )

    # 2. Initialize MPI related stuff
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=eul_grid_size_y,
        grid_size_x=eul_grid_size_x,
        real_t=real_t,
        rank_distribution=rank_distribution,
    )

    # Lagrangian grid inter-rank MPI communicator
    master_rank = 0
    mpi_lagrangian_field_communicator = MPILagrangianFieldCommunicator2D(
        eul_grid_dx=eul_grid_dx,
        eul_grid_shift=eul_grid_coord_shift,
        mpi_construct=mpi_construct,
        master_rank=master_rank,
        real_t=real_t,
    )
    # Map lagrangian nodes to ranks
    mpi_lagrangian_field_communicator.map_lagrangian_nodes_based_on_position(
        ref_lag_positions
    )
    rank_address = mpi_lagrangian_field_communicator.rank_address

    # Initialize Eulerian-Lagrangian grid numerics communicator based on mpi local eul
    # grid coord shift
    mpi_eul_lag_communicator = EulerianLagrangianGridCommunicatorMPI2D(
        dx=eul_grid_dx,
        eul_grid_coord_shift=eul_grid_coord_shift,
        interp_kernel_width=interp_kernel_width,
        real_t=real_t,
        mpi_construct=mpi_construct,
        ghost_size=ghost_size,
        n_components=n_components,
    )

    # 3. Compute solution using MPI implementation
    # since all the ranks have the same reference solution, we can just mask them out
    # the "mpi rank local" data after mapping the lag nodes to their respective ranks
    mask = np.where(rank_address == mpi_lagrangian_field_communicator.rank)[0]
    mpi_local_eul_grid_field = np.zeros(
        (
            2,
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
        ),
        dtype=real_t,
    )
    mpi_local_lag_grid_field = ref_lag_grid_field[..., mask]
    mpi_local_interp_weights = ref_interp_weights[..., mask]
    # indices needs to be offset with mpi local reference index
    mpi_substart_idx = mpi_eul_lag_communicator.mpi_substart_idx
    mpi_local_nearest_eul_grid_index_to_lag_grid = (
        ref_nearest_eul_grid_index_to_lag_grid[..., mask]
        + ghost_size
        - mpi_substart_idx.reshape(grid_dim, 1)
    )

    mpi_eul_lag_communicator.lagrangian_to_eulerian_grid_interpolation_kernel(
        mpi_local_eul_grid_field,
        mpi_local_lag_grid_field,
        mpi_local_interp_weights,
        mpi_local_nearest_eul_grid_index_to_lag_grid,
    )

    # Get corresponding local chunk of ref solution
    mpi_local_sol_idx = (
        slice(None, None),
        slice(
            mpi_substart_idx[1],
            mpi_substart_idx[1] + mpi_construct.local_grid_size[0],
        ),
        slice(
            mpi_substart_idx[0],
            mpi_substart_idx[0] + mpi_construct.local_grid_size[1],
        ),
    )

    np.testing.assert_allclose(
        ref_eul_grid_field[mpi_local_sol_idx],
        mpi_local_eul_grid_field[:, ghost_size:-ghost_size, ghost_size:-ghost_size],
        atol=get_test_tol(precision),
    )


@pytest.mark.mpi(group="MPI_immersed_boundary_ops_2d", min_size=4)
@pytest.mark.parametrize("ghost_size", [1, 2, 3])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rank_distribution", [(1, 0), (0, 1)])
@pytest.mark.parametrize("aspect_ratio", [(1, 1), (1, 2), (2, 1)])
@pytest.mark.parametrize("interp_kernel_type", ["cosine", "peskin"])
def test_mpi_interpolation_weights_kernel_on_nodes_2d(
    ghost_size,
    precision,
    rank_distribution,
    aspect_ratio,
    interp_kernel_type,
):
    n_values = 32
    real_t = get_real_t(precision)
    grid_dim = 2
    eul_grid_size_y = n_values * aspect_ratio[0]
    eul_grid_size_x = n_values * aspect_ratio[1]
    eul_grid_size = eul_grid_size_x
    eul_domain_size = real_t(1.0)
    eul_grid_dx = real_t(eul_domain_size / eul_grid_size)
    eul_grid_coord_shift = real_t(0.5 * eul_grid_dx)
    interp_kernel_width = 2
    num_lag_nodes = 3

    # 1. Generate reference solution from sopht-backend
    eul_lag_communicator = EulerianLagrangianGridCommunicator2D(
        dx=eul_grid_dx,
        eul_grid_coord_shift=eul_grid_coord_shift,
        num_lag_nodes=num_lag_nodes,
        interp_kernel_width=interp_kernel_width,
        real_t=real_t,
        interp_kernel_type=interp_kernel_type,
    )
    interpolation_weights_kernel = eul_lag_communicator.interpolation_weights_kernel
    # init lag. grid around center of the domain
    # this allows testing for the case where rank on first and last quarter (when there
    # are 4 processes) does not contain any lagrangian nodes AND when the point lies at
    # the boundary of MPI domain decomposition
    ref_nearest_eul_grid_index_to_lag_grid = np.empty((grid_dim, num_lag_nodes)).astype(
        int
    )
    ref_nearest_eul_grid_index_to_lag_grid[0] = np.arange(
        eul_grid_size_x // 2 - 1, eul_grid_size_x // 2 - 1 + num_lag_nodes
    ).astype(int)
    ref_nearest_eul_grid_index_to_lag_grid[1] = np.arange(
        eul_grid_size_y // 2 - 1, eul_grid_size_y // 2 - 1 + num_lag_nodes
    ).astype(int)
    ref_lag_positions = (
        ref_nearest_eul_grid_index_to_lag_grid * eul_grid_dx + eul_grid_coord_shift
    ).astype(real_t)
    # find interpolation zone support for the lag. grid
    ref_local_eul_grid_support_of_lag_grid = np.zeros(
        (grid_dim, 2 * interp_kernel_width, 2 * interp_kernel_width, num_lag_nodes)
    ).astype(real_t)
    eul_lag_communicator.local_eulerian_grid_support_of_lagrangian_grid_kernel(
        ref_local_eul_grid_support_of_lag_grid,
        ref_nearest_eul_grid_index_to_lag_grid,
        ref_lag_positions,
    )
    # compute reference interpolation weights
    ref_interp_weights = np.zeros(
        (2 * interp_kernel_width, 2 * interp_kernel_width, num_lag_nodes),
        dtype=real_t,
    )
    interpolation_weights_kernel(
        ref_interp_weights, ref_local_eul_grid_support_of_lag_grid
    )

    # 2. Initialize MPI related stuff
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=eul_grid_size_y,
        grid_size_x=eul_grid_size_x,
        real_t=real_t,
        rank_distribution=rank_distribution,
    )

    # Lagrangian grid inter-rank MPI communicator
    master_rank = 0
    mpi_lagrangian_field_communicator = MPILagrangianFieldCommunicator2D(
        eul_grid_dx=eul_grid_dx,
        eul_grid_shift=eul_grid_coord_shift,
        mpi_construct=mpi_construct,
        master_rank=master_rank,
        real_t=real_t,
    )
    # Map lagrangian nodes to ranks
    mpi_lagrangian_field_communicator.map_lagrangian_nodes_based_on_position(
        ref_lag_positions
    )
    rank_address = mpi_lagrangian_field_communicator.rank_address

    # Initialize Eulerian-Lagrangian grid numerics communicator based on mpi local eul
    # grid coord shift
    mpi_eul_lag_communicator = EulerianLagrangianGridCommunicatorMPI2D(
        dx=eul_grid_dx,
        eul_grid_coord_shift=eul_grid_coord_shift,
        interp_kernel_width=interp_kernel_width,
        real_t=real_t,
        mpi_construct=mpi_construct,
        ghost_size=ghost_size,
        interp_kernel_type=interp_kernel_type,
    )

    # 3. Compute solution using MPI implementation
    # For the reference local eul grid support, we compute using the previously tested
    # `local_eulerian_grid_support_of_lagrangian_grid_kernel(...)` kernel
    mask = np.where(rank_address == mpi_lagrangian_field_communicator.rank)[0]
    mpi_local_lag_positions = ref_lag_positions[..., mask]
    mpi_local_nearest_eul_grid_index_to_lag_grid = np.zeros(
        (grid_dim, mpi_lagrangian_field_communicator.local_num_lag_nodes)
    ).astype(int)
    mpi_local_local_eul_grid_support_of_lag_grid = np.zeros(
        (
            grid_dim,
            2 * interp_kernel_width,
            2 * interp_kernel_width,
            mpi_lagrangian_field_communicator.local_num_lag_nodes,
        )
    ).astype(real_t)
    mpi_eul_lag_communicator.local_eulerian_grid_support_of_lagrangian_grid_kernel(
        mpi_local_local_eul_grid_support_of_lag_grid,
        mpi_local_nearest_eul_grid_index_to_lag_grid,
        mpi_local_lag_positions,
    )
    mpi_local_interp_weights = np.zeros(
        (
            2 * interp_kernel_width,
            2 * interp_kernel_width,
            mpi_lagrangian_field_communicator.local_num_lag_nodes,
        )
    ).astype(real_t)

    # Test interpolation weights based on local eul grid support of lag grid
    mpi_eul_lag_communicator.interpolation_weights_kernel(
        interp_weights=mpi_local_interp_weights,
        local_eul_grid_support_of_lag_grid=mpi_local_local_eul_grid_support_of_lag_grid,
    )
    np.testing.assert_allclose(
        ref_interp_weights[..., mask],
        mpi_local_interp_weights,
        atol=get_test_tol(precision),
    )
