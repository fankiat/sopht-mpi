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
import os


def clear_numba_cache(mpi_construct):
    mpi_construct.grid.barrier()
    if mpi_construct.rank == 0:
        os.system("rm -rf __pycache__")


class MockEulLagGridCommSolution:
    """
    Mock solution test class based on sopht-backend for 2D Eulerian-Lagrangian grid
    communicator
    """

    def __init__(
        self,
        n_values,
        aspect_ratio,
        precision,
        interp_kernel_type="cosine",
        n_components=1,
    ):
        self.real_t = get_real_t(precision)
        self.grid_dim = 2
        self.eul_grid_size_y = n_values * aspect_ratio[0]
        self.eul_grid_size_x = n_values * aspect_ratio[1]
        self.eul_grid_size = self.eul_grid_size_x
        self.eul_domain_size = self.real_t(1.0)
        self.eul_grid_dx = self.real_t(self.eul_domain_size / self.eul_grid_size)
        self.eul_grid_coord_shift = self.real_t(0.5 * self.eul_grid_dx)
        self.interp_kernel_width = 2
        self.num_lag_nodes = 3
        self.interp_kernel_type = interp_kernel_type
        self.n_components = n_components

        self.generate_reference_solution()

    def generate_reference_solution(self):
        """Generate reference solution from sopht-backend"""
        self.eul_lag_communicator = EulerianLagrangianGridCommunicator2D(
            dx=self.eul_grid_dx,
            eul_grid_coord_shift=self.eul_grid_coord_shift,
            num_lag_nodes=self.num_lag_nodes,
            interp_kernel_width=self.interp_kernel_width,
            n_components=self.n_components,
            real_t=self.real_t,
        )
        # init lag. grid around center of the domain
        # this allows testing for the case where rank on first and last quarter (when there
        # are 4 processes) does not contain any lagrangian nodes AND when the point lies at
        # the boundary of MPI domain decomposition
        self.nearest_eul_grid_index_to_lag_grid = np.empty(
            (self.grid_dim, self.num_lag_nodes)
        ).astype(int)
        self.nearest_eul_grid_index_to_lag_grid[0] = np.arange(
            self.eul_grid_size_x // 2 - 1,
            self.eul_grid_size_x // 2 - 1 + self.num_lag_nodes,
        ).astype(int)
        self.nearest_eul_grid_index_to_lag_grid[1] = np.arange(
            self.eul_grid_size_y // 2 - 1,
            self.eul_grid_size_y // 2 - 1 + self.num_lag_nodes,
        ).astype(int)
        self.lag_positions = (
            self.nearest_eul_grid_index_to_lag_grid * self.eul_grid_dx
            + self.eul_grid_coord_shift
        ).astype(self.real_t)

        # 1. Compute reference interpolation zone support for the lag. grid
        self.local_eul_grid_support_of_lag_grid = np.zeros(
            (
                self.grid_dim,
                2 * self.interp_kernel_width,
                2 * self.interp_kernel_width,
                self.num_lag_nodes,
            )
        ).astype(self.real_t)
        self.eul_lag_communicator.local_eulerian_grid_support_of_lagrangian_grid_kernel(
            self.local_eul_grid_support_of_lag_grid,
            self.nearest_eul_grid_index_to_lag_grid,
            self.lag_positions,
        )

        # 2. Compute reference interpolation weights
        self.interp_weights = np.zeros(
            (
                2 * self.interp_kernel_width,
                2 * self.interp_kernel_width,
                self.num_lag_nodes,
            ),
            dtype=self.real_t,
        )
        self.mock_local_eul_grid_support_of_lag_grid = (
            self.local_eul_grid_support_of_lag_grid.copy()
        )
        self.eul_lag_communicator.interpolation_weights_kernel(
            self.interp_weights, self.mock_local_eul_grid_support_of_lag_grid
        )

        # 3. Compute reference lagrangian grid interpolated from mock eulerian grid
        if self.n_components == 1:
            self.lag_grid_field = np.zeros((self.num_lag_nodes), dtype=self.real_t)
            self.mock_eul_grid_field_prefactor = self.real_t(2.0)
            self.mock_eul_grid_field = self.mock_eul_grid_field_prefactor * np.ones(
                (self.eul_grid_size_y, self.eul_grid_size_x), dtype=self.real_t
            )
        else:
            self.lag_grid_field = np.zeros(
                (self.n_components, self.num_lag_nodes), dtype=self.real_t
            )
            self.mock_eul_grid_field_prefactor_x = self.real_t(2.0)
            self.mock_eul_grid_field_prefactor_y = self.real_t(2.0)
            self.mock_eul_grid_field = np.ones(
                (self.n_components, self.eul_grid_size_y, self.eul_grid_size_x),
                dtype=self.real_t,
            )
            self.mock_eul_grid_field[0] *= self.mock_eul_grid_field_prefactor_y
            self.mock_eul_grid_field[1] *= self.mock_eul_grid_field_prefactor_x
        self.eul_lag_communicator.eulerian_to_lagrangian_grid_interpolation_kernel(
            self.lag_grid_field,
            self.mock_eul_grid_field,
            self.interp_weights,
            self.nearest_eul_grid_index_to_lag_grid,
        )

        # 4. Compute reference eulerian grid interpolated from mock lagrangian grid
        if self.n_components == 1:
            prefactor_lag_field = 2
            self.mock_lag_grid_field = prefactor_lag_field * np.ones(
                (self.num_lag_nodes), dtype=self.real_t
            )
            self.eul_grid_field = np.zeros(
                (self.eul_grid_size_y, self.eul_grid_size_x), dtype=self.real_t
            )
        else:
            prefactor_lag_field_y = 2
            prefactor_lag_field_x = 3
            self.mock_lag_grid_field = np.ones(
                (self.n_components, self.num_lag_nodes), dtype=self.real_t
            )
            self.mock_lag_grid_field[0] *= prefactor_lag_field_y
            self.mock_lag_grid_field[1] *= prefactor_lag_field_x
            self.eul_grid_field = np.zeros(
                (self.n_components, self.eul_grid_size_y, self.eul_grid_size_x),
                dtype=self.real_t,
            )

        self.eul_lag_communicator.lagrangian_to_eulerian_grid_interpolation_kernel(
            self.eul_grid_field,
            self.mock_lag_grid_field,
            self.interp_weights,
            self.nearest_eul_grid_index_to_lag_grid,
        )


@pytest.mark.mpi(group="MPI_immersed_boundary_ops_2d", min_size=4)
@pytest.mark.parametrize("ghost_size", [1, 2, 3])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rank_distribution", [(1, 0), (0, 1)])
@pytest.mark.parametrize("aspect_ratio", [(1, 1), (1, 2), (2, 1)])
def test_mpi_local_eulerian_grid_support_of_lagrangian_grid_2d(
    ghost_size, precision, rank_distribution, aspect_ratio
):
    # 1. Generate reference solution (the solution is in the global domain, and each of
    # the ranks has the same reference copy)
    mock_soln = MockEulLagGridCommSolution(
        n_values=32, aspect_ratio=aspect_ratio, precision=precision
    )

    # 2. Initialize MPI related stuff
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=mock_soln.eul_grid_size_y,
        grid_size_x=mock_soln.eul_grid_size_x,
        real_t=mock_soln.real_t,
        rank_distribution=rank_distribution,
    )
    # Clear any exising cache from previous numba tests
    clear_numba_cache(mpi_construct)

    # Lagrangian grid inter-rank MPI communicator
    master_rank = 0
    mpi_lagrangian_field_communicator = MPILagrangianFieldCommunicator2D(
        eul_grid_dx=mock_soln.eul_grid_dx,
        eul_grid_coord_shift=mock_soln.eul_grid_coord_shift,
        mpi_construct=mpi_construct,
        master_rank=master_rank,
        real_t=mock_soln.real_t,
    )
    # Map lagrangian nodes to ranks
    mpi_lagrangian_field_communicator.map_lagrangian_nodes_based_on_position(
        mock_soln.lag_positions
    )
    rank_address = mpi_lagrangian_field_communicator.rank_address

    # Initialize Eulerian-Lagrangian grid numerics communicator based on mpi local eul
    # grid coord shift
    mpi_eul_lag_communicator = EulerianLagrangianGridCommunicatorMPI2D(
        dx=mock_soln.eul_grid_dx,
        eul_grid_coord_shift=mock_soln.eul_grid_coord_shift,
        interp_kernel_width=mock_soln.interp_kernel_width,
        real_t=mock_soln.real_t,
        mpi_construct=mpi_construct,
        ghost_size=ghost_size,
    )

    # 3. Compute solution using MPI implementation
    # since all the ranks have the same reference solution, we can just mask them out
    # the "mpi rank local" data after mapping the lag nodes to their respective ranks
    mask = np.where(rank_address == mpi_lagrangian_field_communicator.rank)[0]
    mpi_local_lag_positions = mock_soln.lag_positions[..., mask]
    mpi_local_nearest_eul_grid_index_to_lag_grid = np.zeros(
        (mock_soln.grid_dim, mpi_lagrangian_field_communicator.local_num_lag_nodes)
    ).astype(int)
    mpi_local_local_eul_grid_support_of_lag_grid = np.zeros(
        (
            mock_soln.grid_dim,
            2 * mock_soln.interp_kernel_width,
            2 * mock_soln.interp_kernel_width,
            mpi_lagrangian_field_communicator.local_num_lag_nodes,
        )
    ).astype(mock_soln.real_t)

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
        + mpi_substart_idx.reshape(mock_soln.grid_dim, 1)
    ).astype(int)

    # 4. Test and compare values
    np.testing.assert_allclose(
        mock_soln.nearest_eul_grid_index_to_lag_grid[..., mask],
        mpi_local_nearest_eul_grid_index_to_lag_grid,
        atol=get_test_tol(precision),
    )
    np.testing.assert_allclose(
        mock_soln.local_eul_grid_support_of_lag_grid[..., mask],
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
    # 1. Generate reference solution (the solution is in the global domain, and each of
    # the ranks has the same reference copy)
    mock_soln = MockEulLagGridCommSolution(
        n_values=32, aspect_ratio=aspect_ratio, precision=precision
    )

    # 2. Initialize MPI related stuff
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=mock_soln.eul_grid_size_y,
        grid_size_x=mock_soln.eul_grid_size_x,
        real_t=mock_soln.real_t,
        rank_distribution=rank_distribution,
    )
    # Clear any exising cache from previous numba tests
    clear_numba_cache(mpi_construct)

    # Lagrangian grid inter-rank MPI communicator
    master_rank = 0
    mpi_lagrangian_field_communicator = MPILagrangianFieldCommunicator2D(
        eul_grid_dx=mock_soln.eul_grid_dx,
        eul_grid_coord_shift=mock_soln.eul_grid_coord_shift,
        mpi_construct=mpi_construct,
        master_rank=master_rank,
        real_t=mock_soln.real_t,
    )
    # Map lagrangian nodes to ranks
    mpi_lagrangian_field_communicator.map_lagrangian_nodes_based_on_position(
        mock_soln.lag_positions
    )
    rank_address = mpi_lagrangian_field_communicator.rank_address

    # Initialize Eulerian-Lagrangian grid numerics communicator based on mpi local eul
    # grid coord shift
    mpi_eul_lag_communicator = EulerianLagrangianGridCommunicatorMPI2D(
        dx=mock_soln.eul_grid_dx,
        eul_grid_coord_shift=mock_soln.eul_grid_coord_shift,
        interp_kernel_width=mock_soln.interp_kernel_width,
        real_t=mock_soln.real_t,
        mpi_construct=mpi_construct,
        ghost_size=ghost_size,
    )

    # 3. Compute solution using MPI implementation
    # since all the ranks have the same reference solution, we can just mask them out
    # the "mpi rank local" data after mapping the lag nodes to their respective ranks
    mask = np.where(rank_address == mpi_lagrangian_field_communicator.rank)[0]
    mpi_local_eul_grid_field = np.ones(
        mpi_construct.local_grid_size + 2 * ghost_size, dtype=mock_soln.real_t
    )
    mpi_local_eul_grid_field *= mock_soln.mock_eul_grid_field_prefactor
    mpi_local_interp_weights = mock_soln.interp_weights[..., mask]
    # indices needs to be offset with mpi local reference index
    mpi_substart_idx = mpi_eul_lag_communicator.mpi_substart_idx
    mpi_local_nearest_eul_grid_index_to_lag_grid = (
        mock_soln.nearest_eul_grid_index_to_lag_grid[..., mask]
        + ghost_size
        - mpi_substart_idx.reshape(mock_soln.grid_dim, 1)
    )
    mpi_local_lag_grid_field = np.zeros(
        (mpi_lagrangian_field_communicator.local_num_lag_nodes), dtype=mock_soln.real_t
    )
    mpi_eul_lag_communicator.eulerian_to_lagrangian_grid_interpolation_kernel(
        mpi_local_lag_grid_field,
        mpi_local_eul_grid_field,
        mpi_local_interp_weights,
        mpi_local_nearest_eul_grid_index_to_lag_grid,
    )

    np.testing.assert_allclose(
        mock_soln.lag_grid_field[..., mask],
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
    # 1. Generate reference solution (the solution is in the global domain, and each of
    # the ranks has the same reference copy)
    mock_soln = MockEulLagGridCommSolution(
        n_values=32, aspect_ratio=aspect_ratio, precision=precision, n_components=2
    )

    # 2. Initialize MPI related stuff
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=mock_soln.eul_grid_size_y,
        grid_size_x=mock_soln.eul_grid_size_x,
        real_t=mock_soln.real_t,
        rank_distribution=rank_distribution,
    )
    # Clear any exising cache from previous numba tests
    clear_numba_cache(mpi_construct)

    # Lagrangian grid inter-rank MPI communicator
    master_rank = 0
    mpi_lagrangian_field_communicator = MPILagrangianFieldCommunicator2D(
        eul_grid_dx=mock_soln.eul_grid_dx,
        eul_grid_coord_shift=mock_soln.eul_grid_coord_shift,
        mpi_construct=mpi_construct,
        master_rank=master_rank,
        real_t=mock_soln.real_t,
    )
    # Map lagrangian nodes to ranks
    mpi_lagrangian_field_communicator.map_lagrangian_nodes_based_on_position(
        mock_soln.lag_positions
    )
    rank_address = mpi_lagrangian_field_communicator.rank_address

    # Initialize Eulerian-Lagrangian grid numerics communicator based on mpi local eul
    # grid coord shift
    mpi_eul_lag_communicator = EulerianLagrangianGridCommunicatorMPI2D(
        dx=mock_soln.eul_grid_dx,
        eul_grid_coord_shift=mock_soln.eul_grid_coord_shift,
        interp_kernel_width=mock_soln.interp_kernel_width,
        n_components=2,
        real_t=mock_soln.real_t,
        mpi_construct=mpi_construct,
        ghost_size=ghost_size,
    )

    # 3. Compute solution using MPI implementation
    # since all the ranks have the same reference solution, we can just mask them out
    # the "mpi rank local" data after mapping the lag nodes to their respective ranks
    mask = np.where(rank_address == mpi_lagrangian_field_communicator.rank)[0]
    mpi_local_eul_grid_field = np.ones(
        (
            mock_soln.n_components,
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
        ),
        dtype=mock_soln.real_t,
    )
    mpi_local_eul_grid_field[0] *= mock_soln.mock_eul_grid_field_prefactor_y
    mpi_local_eul_grid_field[1] *= mock_soln.mock_eul_grid_field_prefactor_x
    mpi_local_interp_weights = mock_soln.interp_weights[..., mask]
    # indices needs to be offset with mpi local reference index
    mpi_substart_idx = mpi_eul_lag_communicator.mpi_substart_idx
    mpi_local_nearest_eul_grid_index_to_lag_grid = (
        mock_soln.nearest_eul_grid_index_to_lag_grid[..., mask]
        + ghost_size
        - mpi_substart_idx.reshape(mock_soln.grid_dim, 1)
    )
    mpi_local_lag_grid_field = np.zeros(
        (mock_soln.n_components, mpi_lagrangian_field_communicator.local_num_lag_nodes),
        dtype=mock_soln.real_t,
    )
    mpi_eul_lag_communicator.eulerian_to_lagrangian_grid_interpolation_kernel(
        mpi_local_lag_grid_field,
        mpi_local_eul_grid_field,
        mpi_local_interp_weights,
        mpi_local_nearest_eul_grid_index_to_lag_grid,
    )

    np.testing.assert_allclose(
        mock_soln.lag_grid_field[..., mask],
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
    # 1. Generate reference solution (the solution is in the global domain, and each of
    # the ranks has the same reference copy)
    mock_soln = MockEulLagGridCommSolution(
        n_values=32, aspect_ratio=aspect_ratio, precision=precision
    )

    # 2. Initialize MPI related stuff
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=mock_soln.eul_grid_size_y,
        grid_size_x=mock_soln.eul_grid_size_x,
        real_t=mock_soln.real_t,
        rank_distribution=rank_distribution,
    )
    # Clear any exising cache from previous numba tests
    clear_numba_cache(mpi_construct)

    # Lagrangian grid inter-rank MPI communicator
    master_rank = 0
    mpi_lagrangian_field_communicator = MPILagrangianFieldCommunicator2D(
        eul_grid_dx=mock_soln.eul_grid_dx,
        eul_grid_coord_shift=mock_soln.eul_grid_coord_shift,
        mpi_construct=mpi_construct,
        master_rank=master_rank,
        real_t=mock_soln.real_t,
    )
    # Map lagrangian nodes to ranks
    mpi_lagrangian_field_communicator.map_lagrangian_nodes_based_on_position(
        mock_soln.lag_positions
    )
    rank_address = mpi_lagrangian_field_communicator.rank_address

    # Initialize Eulerian-Lagrangian grid numerics communicator based on mpi local eul
    # grid coord shift
    mpi_eul_lag_communicator = EulerianLagrangianGridCommunicatorMPI2D(
        dx=mock_soln.eul_grid_dx,
        eul_grid_coord_shift=mock_soln.eul_grid_coord_shift,
        interp_kernel_width=mock_soln.interp_kernel_width,
        real_t=mock_soln.real_t,
        mpi_construct=mpi_construct,
        ghost_size=ghost_size,
    )

    # 3. Compute solution using MPI implementation
    # since all the ranks have the same reference solution, we can just mask them out
    # the "mpi rank local" data after mapping the lag nodes to their respective ranks
    mask = np.where(rank_address == mpi_lagrangian_field_communicator.rank)[0]
    mpi_local_eul_grid_field = np.zeros(
        mpi_construct.local_grid_size + 2 * ghost_size, dtype=mock_soln.real_t
    )
    mpi_local_lag_grid_field = mock_soln.mock_lag_grid_field[..., mask]
    mpi_local_interp_weights = mock_soln.interp_weights[..., mask]
    # indices needs to be offset with mpi local reference index
    mpi_substart_idx = mpi_eul_lag_communicator.mpi_substart_idx
    mpi_local_nearest_eul_grid_index_to_lag_grid = (
        mock_soln.nearest_eul_grid_index_to_lag_grid[..., mask]
        + ghost_size
        - mpi_substart_idx.reshape(mock_soln.grid_dim, 1)
    )

    mpi_eul_lag_communicator.lagrangian_to_eulerian_grid_interpolation_kernel(
        mpi_local_eul_grid_field,
        mpi_local_lag_grid_field,
        mpi_local_interp_weights,
        mpi_local_nearest_eul_grid_index_to_lag_grid,
    )

    # Get corresponding local chunk of ref solution
    local_grid_size_y, local_grid_size_x = mpi_construct.local_grid_size
    mpi_local_sol_idx = (
        slice(mpi_substart_idx[1], mpi_substart_idx[1] + local_grid_size_y),
        slice(mpi_substart_idx[0], mpi_substart_idx[0] + local_grid_size_x),
    )

    np.testing.assert_allclose(
        mock_soln.eul_grid_field[mpi_local_sol_idx],
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
    # 1. Generate reference solution (the solution is in the global domain, and each of
    # the ranks has the same reference copy)
    mock_soln = MockEulLagGridCommSolution(
        n_values=32, aspect_ratio=aspect_ratio, precision=precision, n_components=2
    )

    # 2. Initialize MPI related stuff
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=mock_soln.eul_grid_size_y,
        grid_size_x=mock_soln.eul_grid_size_x,
        real_t=mock_soln.real_t,
        rank_distribution=rank_distribution,
    )
    # Clear any exising cache from previous numba tests
    clear_numba_cache(mpi_construct)

    # Lagrangian grid inter-rank MPI communicator
    master_rank = 0
    mpi_lagrangian_field_communicator = MPILagrangianFieldCommunicator2D(
        eul_grid_dx=mock_soln.eul_grid_dx,
        eul_grid_coord_shift=mock_soln.eul_grid_coord_shift,
        mpi_construct=mpi_construct,
        master_rank=master_rank,
        real_t=mock_soln.real_t,
    )
    # Map lagrangian nodes to ranks
    mpi_lagrangian_field_communicator.map_lagrangian_nodes_based_on_position(
        mock_soln.lag_positions
    )
    rank_address = mpi_lagrangian_field_communicator.rank_address

    # Initialize Eulerian-Lagrangian grid numerics communicator based on mpi local eul
    # grid coord shift
    mpi_eul_lag_communicator = EulerianLagrangianGridCommunicatorMPI2D(
        dx=mock_soln.eul_grid_dx,
        eul_grid_coord_shift=mock_soln.eul_grid_coord_shift,
        interp_kernel_width=mock_soln.interp_kernel_width,
        real_t=mock_soln.real_t,
        mpi_construct=mpi_construct,
        ghost_size=ghost_size,
        n_components=mock_soln.n_components,
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
        dtype=mock_soln.real_t,
    )
    mpi_local_lag_grid_field = mock_soln.mock_lag_grid_field[..., mask]
    mpi_local_interp_weights = mock_soln.interp_weights[..., mask]
    # indices needs to be offset with mpi local reference index
    mpi_substart_idx = mpi_eul_lag_communicator.mpi_substart_idx
    mpi_local_nearest_eul_grid_index_to_lag_grid = (
        mock_soln.nearest_eul_grid_index_to_lag_grid[..., mask]
        + ghost_size
        - mpi_substart_idx.reshape(mock_soln.grid_dim, 1)
    )

    mpi_eul_lag_communicator.lagrangian_to_eulerian_grid_interpolation_kernel(
        mpi_local_eul_grid_field,
        mpi_local_lag_grid_field,
        mpi_local_interp_weights,
        mpi_local_nearest_eul_grid_index_to_lag_grid,
    )

    # Get corresponding local chunk of ref solution
    local_grid_size_y, local_grid_size_x = mpi_construct.local_grid_size
    mpi_local_sol_idx = (
        slice(None, None),
        slice(
            mpi_substart_idx[1],
            mpi_substart_idx[1] + local_grid_size_y,
        ),
        slice(
            mpi_substart_idx[0],
            mpi_substart_idx[0] + local_grid_size_x,
        ),
    )

    np.testing.assert_allclose(
        mock_soln.eul_grid_field[mpi_local_sol_idx],
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
    # 1. Generate reference solution (the solution is in the global domain, and each of
    # the ranks has the same reference copy)
    mock_soln = MockEulLagGridCommSolution(
        n_values=32,
        aspect_ratio=aspect_ratio,
        precision=precision,
        interp_kernel_type=interp_kernel_type,
    )

    # 2. Initialize MPI related stuff
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=mock_soln.eul_grid_size_y,
        grid_size_x=mock_soln.eul_grid_size_x,
        real_t=mock_soln.real_t,
        rank_distribution=rank_distribution,
    )
    # Clear any exising cache from previous numba tests
    clear_numba_cache(mpi_construct)

    # Lagrangian grid inter-rank MPI communicator
    master_rank = 0
    mpi_lagrangian_field_communicator = MPILagrangianFieldCommunicator2D(
        eul_grid_dx=mock_soln.eul_grid_dx,
        eul_grid_coord_shift=mock_soln.eul_grid_coord_shift,
        mpi_construct=mpi_construct,
        master_rank=master_rank,
        real_t=mock_soln.real_t,
    )
    # Map lagrangian nodes to ranks
    mpi_lagrangian_field_communicator.map_lagrangian_nodes_based_on_position(
        mock_soln.lag_positions
    )
    rank_address = mpi_lagrangian_field_communicator.rank_address

    # Initialize Eulerian-Lagrangian grid numerics communicator based on mpi local eul
    # grid coord shift
    mpi_eul_lag_communicator = EulerianLagrangianGridCommunicatorMPI2D(
        dx=mock_soln.eul_grid_dx,
        eul_grid_coord_shift=mock_soln.eul_grid_coord_shift,
        interp_kernel_width=mock_soln.interp_kernel_width,
        real_t=mock_soln.real_t,
        mpi_construct=mpi_construct,
        ghost_size=ghost_size,
        interp_kernel_type=interp_kernel_type,
    )

    # 3. Compute solution using MPI implementation
    # For the reference local eul grid support, we compute using the previously tested
    # `local_eulerian_grid_support_of_lagrangian_grid_kernel(...)` kernel
    mask = np.where(rank_address == mpi_lagrangian_field_communicator.rank)[0]
    mpi_local_local_eul_grid_support_of_lag_grid = (
        mock_soln.local_eul_grid_support_of_lag_grid[..., mask].copy()
    )
    mpi_local_interp_weights = np.zeros(
        (
            2 * mock_soln.interp_kernel_width,
            2 * mock_soln.interp_kernel_width,
            mpi_lagrangian_field_communicator.local_num_lag_nodes,
        )
    ).astype(mock_soln.real_t)

    # Test interpolation weights based on local eul grid support of lag grid
    mpi_eul_lag_communicator.interpolation_weights_kernel(
        interp_weights=mpi_local_interp_weights,
        local_eul_grid_support_of_lag_grid=mpi_local_local_eul_grid_support_of_lag_grid,
    )
    np.testing.assert_allclose(
        mock_soln.interp_weights[..., mask],
        mpi_local_interp_weights,
        atol=get_test_tol(precision),
    )
