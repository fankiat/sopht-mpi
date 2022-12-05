# TODO: Add tests for with grid dim parameter
import numpy as np
import pytest
from sopht.numeric.immersed_boundary_ops import VirtualBoundaryForcing
from sopht_mpi.numeric.immersed_boundary_ops import VirtualBoundaryForcingMPI
from sopht_mpi.utils import (
    MPIConstruct2D,
    MPILagrangianFieldCommunicator2D,
    MPIGhostCommunicator2D,
    MPIFieldCommunicator2D,
)
from sopht.utils.precision import get_real_t, get_test_tol
from mpi4py import MPI


class ReferenceVirtualBoundaryForcing(VirtualBoundaryForcing):
    """Mock solution test class for virtual boundary forcing."""

    def __init__(
        self,
        n_values,
        aspect_ratio,
        grid_dim=2,
        precision="single",
        enable_eul_grid_forcing_reset=True,
    ):
        """Class initialiser."""
        real_t = get_real_t(precision)
        self.test_tol = get_test_tol(precision)
        self.real_t = real_t
        self.virtual_boundary_stiffness_coeff = real_t(1e3)
        self.virtual_boundary_damping_coeff = real_t(1e1)
        self.grid_size_y = n_values * aspect_ratio[0]
        self.grid_size_x = n_values * aspect_ratio[1]
        self.domain_size = self.real_t(1.0)
        self.dx = self.real_t(self.domain_size / self.grid_size_x)
        self.eul_grid_coord_shift = real_t(self.dx / 2)
        self.num_lag_nodes = 3
        self.interp_kernel_width = 2
        self.grid_dim = grid_dim
        self.time = 0.0

        super().__init__(
            virtual_boundary_stiffness_coeff=self.virtual_boundary_stiffness_coeff,
            virtual_boundary_damping_coeff=self.virtual_boundary_damping_coeff,
            grid_dim=self.grid_dim,
            dx=self.dx,
            num_lag_nodes=self.num_lag_nodes,
            real_t=self.real_t,
            start_time=self.time,
            enable_eul_grid_forcing_reset=enable_eul_grid_forcing_reset,
        )

        self.init_reference_lagrangian_nodes()

    def init_reference_lagrangian_nodes(self):
        # init lag. grid around center of the domain
        self.nearest_eul_grid_index_to_lag_grid[0] = np.arange(
            self.grid_size_x // 2 - 1,
            self.grid_size_x // 2 - 1 + self.num_lag_nodes,
        ).astype(int)
        self.nearest_eul_grid_index_to_lag_grid[1] = np.arange(
            self.grid_size_y // 2 - 1,
            self.grid_size_y // 2 - 1 + self.num_lag_nodes,
        ).astype(int)
        self.lag_positions = (
            self.nearest_eul_grid_index_to_lag_grid * self.dx
            + self.eul_grid_coord_shift
        ).astype(self.real_t)

    def check_lag_grid_interaction_solution(self, virtual_boundary_forcing, mask):
        """Check solution for lag grid forcing in the interaction step."""
        # Adjust nearest eul grid index to mpi global index frame
        mpi_substart_idx = (
            virtual_boundary_forcing.eul_lag_grid_communicator.mpi_substart_idx
        )
        ghost_size = virtual_boundary_forcing.ghost_size
        mpi_global_nearest_eul_grid_index_to_lag_grid = (
            virtual_boundary_forcing.local_nearest_eul_grid_index_to_lag_grid
            - ghost_size
            + mpi_substart_idx.reshape(virtual_boundary_forcing.grid_dim, 1)
        )
        # Check nearest eul grid index
        local_allclose_local_nearest_eul_grid_index_to_lag_grid = np.allclose(
            mpi_global_nearest_eul_grid_index_to_lag_grid,
            self.nearest_eul_grid_index_to_lag_grid[..., mask],
            atol=self.test_tol,
        )
        # Check lag grid flow velocity field
        local_allclose_lag_grid_flow_velocity_field = np.allclose(
            virtual_boundary_forcing.local_lag_grid_flow_velocity_field,
            self.lag_grid_flow_velocity_field[..., mask],
            atol=self.test_tol,
        )
        # Check lag grid velocity mismatch field
        local_allclose_lag_grid_velocity_mismatch_field = np.allclose(
            virtual_boundary_forcing.local_lag_grid_velocity_mismatch_field,
            self.lag_grid_velocity_mismatch_field[..., mask],
            atol=self.test_tol,
        )
        # Check lag grid forcing field
        local_allclose_lag_grid_forcing_field = np.allclose(
            virtual_boundary_forcing.local_lag_grid_forcing_field,
            self.lag_grid_forcing_field[..., mask],
            atol=self.test_tol,
        )

        return (
            local_allclose_local_nearest_eul_grid_index_to_lag_grid,
            local_allclose_lag_grid_flow_velocity_field,
            local_allclose_lag_grid_velocity_mismatch_field,
            local_allclose_lag_grid_forcing_field,
        )


@pytest.mark.mpi(group="MPI_immersed_boundary_ops_2d", min_size=4)
@pytest.mark.parametrize("grid_dim", [2])
@pytest.mark.parametrize("ghost_size", [2, 3])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rank_distribution", [(1, 0), (0, 1)])
@pytest.mark.parametrize("aspect_ratio", [(1, 1), (1, 2), (2, 1)])
def test_mpi_virtual_boundary_forcing_init(
    grid_dim, ghost_size, precision, rank_distribution, aspect_ratio
):
    n_values = 32
    # 1. Generate reference solution
    ref_virtual_boundary_forcing = ReferenceVirtualBoundaryForcing(
        n_values=n_values,
        aspect_ratio=aspect_ratio,
        grid_dim=grid_dim,
        precision=precision,
    )

    # 2. Initialize MPI related stuff to distribute lag nodes to corresponding ranks
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=ref_virtual_boundary_forcing.grid_size_y,
        grid_size_x=ref_virtual_boundary_forcing.grid_size_x,
        real_t=ref_virtual_boundary_forcing.real_t,
        rank_distribution=rank_distribution,
    )
    # Lagrangian grid inter-rank MPI communicator
    master_rank = 0
    mpi_lagrangian_field_communicator = MPILagrangianFieldCommunicator2D(
        eul_grid_dx=ref_virtual_boundary_forcing.dx,
        eul_grid_coord_shift=ref_virtual_boundary_forcing.eul_grid_coord_shift,
        mpi_construct=mpi_construct,
        master_rank=master_rank,
        real_t=ref_virtual_boundary_forcing.real_t,
    )
    # Map lagrangian nodes to ranks
    mpi_lagrangian_field_communicator.map_lagrangian_nodes_based_on_position(
        ref_virtual_boundary_forcing.lag_positions
    )
    rank_address = mpi_lagrangian_field_communicator.rank_address
    mask = np.where(rank_address == mpi_lagrangian_field_communicator.rank)[0]
    mpi_local_num_lag_nodes = mpi_lagrangian_field_communicator.local_num_lag_nodes
    # Initialize MPI virtual boundary forcing with global reference lag positions
    if mpi_construct.rank == master_rank:
        global_ref_lag_grid_position_field = ref_virtual_boundary_forcing.lag_positions
    else:
        global_ref_lag_grid_position_field = None
    mpi_virtual_boundary_forcing = VirtualBoundaryForcingMPI(
        mpi_construct=mpi_construct,
        ghost_size=ghost_size,
        virtual_boundary_stiffness_coeff=ref_virtual_boundary_forcing.virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff=ref_virtual_boundary_forcing.virtual_boundary_damping_coeff,
        grid_dim=ref_virtual_boundary_forcing.grid_dim,
        dx=ref_virtual_boundary_forcing.dx,
        real_t=ref_virtual_boundary_forcing.real_t,
        start_time=ref_virtual_boundary_forcing.time,
        global_lag_grid_position_field=global_ref_lag_grid_position_field,
        moving_boundary=True,
    )

    # 3. Test initialized variables and field buffers
    assert mpi_local_num_lag_nodes == mpi_virtual_boundary_forcing.local_num_lag_nodes
    assert mpi_virtual_boundary_forcing.time == ref_virtual_boundary_forcing.time
    assert (
        mpi_virtual_boundary_forcing.virtual_boundary_stiffness_coeff
        == ref_virtual_boundary_forcing.virtual_boundary_stiffness_coeff
    )
    assert (
        mpi_virtual_boundary_forcing.virtual_boundary_damping_coeff
        == ref_virtual_boundary_forcing.virtual_boundary_damping_coeff
    )
    assert (
        mpi_virtual_boundary_forcing.local_nearest_eul_grid_index_to_lag_grid.dtype
        == int
    )
    assert (
        mpi_virtual_boundary_forcing.local_nearest_eul_grid_index_to_lag_grid.shape
        == ref_virtual_boundary_forcing.nearest_eul_grid_index_to_lag_grid[
            ..., mask
        ].shape
    )
    assert (
        mpi_virtual_boundary_forcing.local_local_eul_grid_support_of_lag_grid.shape
        == ref_virtual_boundary_forcing.local_eul_grid_support_of_lag_grid[
            ..., mask
        ].shape
    )
    assert (
        mpi_virtual_boundary_forcing.local_interp_weights.shape
        == ref_virtual_boundary_forcing.interp_weights[..., mask].shape
    )
    assert (
        mpi_virtual_boundary_forcing.local_lag_grid_flow_velocity_field.shape
        == ref_virtual_boundary_forcing.lag_grid_flow_velocity_field[..., mask].shape
    )
    assert (
        mpi_virtual_boundary_forcing.local_lag_grid_velocity_mismatch_field.shape
        == ref_virtual_boundary_forcing.lag_grid_velocity_mismatch_field[
            ..., mask
        ].shape
    )
    assert (
        mpi_virtual_boundary_forcing.local_lag_grid_position_mismatch_field.shape
        == ref_virtual_boundary_forcing.lag_grid_position_mismatch_field[
            ..., mask
        ].shape
    )


@pytest.mark.mpi(group="MPI_immersed_boundary_ops_2d", min_size=4)
@pytest.mark.parametrize("grid_dim", [2])
@pytest.mark.parametrize("ghost_size", [2, 3])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rank_distribution", [(1, 0), (0, 1)])
@pytest.mark.parametrize("aspect_ratio", [(1, 1), (1, 2), (2, 1)])
def test_mpi_compute_lag_grid_velocity_mismatch_field(
    grid_dim, ghost_size, precision, rank_distribution, aspect_ratio
):
    n_values = 32
    # 1. Generate reference solution
    ref_virtual_boundary_forcing = ReferenceVirtualBoundaryForcing(
        n_values=n_values,
        aspect_ratio=aspect_ratio,
        grid_dim=grid_dim,
        precision=precision,
    )

    # 2. Initialize MPI related stuff to distribute lag nodes to corresponding ranks
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=ref_virtual_boundary_forcing.grid_size_y,
        grid_size_x=ref_virtual_boundary_forcing.grid_size_x,
        real_t=ref_virtual_boundary_forcing.real_t,
        rank_distribution=rank_distribution,
    )
    # Lagrangian grid inter-rank MPI communicator
    master_rank = 0
    mpi_lagrangian_field_communicator = MPILagrangianFieldCommunicator2D(
        eul_grid_dx=ref_virtual_boundary_forcing.dx,
        eul_grid_coord_shift=ref_virtual_boundary_forcing.eul_grid_coord_shift,
        mpi_construct=mpi_construct,
        master_rank=master_rank,
        real_t=ref_virtual_boundary_forcing.real_t,
    )
    # Map lagrangian nodes to ranks
    mpi_lagrangian_field_communicator.map_lagrangian_nodes_based_on_position(
        ref_virtual_boundary_forcing.lag_positions
    )
    rank_address = mpi_lagrangian_field_communicator.rank_address
    mask = np.where(rank_address == mpi_lagrangian_field_communicator.rank)[0]
    # Initialize MPI virtual boundary forcing with global reference lag positions
    if mpi_construct.rank == master_rank:
        global_ref_lag_grid_position_field = ref_virtual_boundary_forcing.lag_positions
    else:
        global_ref_lag_grid_position_field = None
    mpi_virtual_boundary_forcing = VirtualBoundaryForcingMPI(
        mpi_construct=mpi_construct,
        ghost_size=ghost_size,
        virtual_boundary_stiffness_coeff=ref_virtual_boundary_forcing.virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff=ref_virtual_boundary_forcing.virtual_boundary_damping_coeff,
        grid_dim=ref_virtual_boundary_forcing.grid_dim,
        dx=ref_virtual_boundary_forcing.dx,
        real_t=ref_virtual_boundary_forcing.real_t,
        start_time=ref_virtual_boundary_forcing.time,
        global_lag_grid_position_field=global_ref_lag_grid_position_field,
    )

    # 3. Generate reference fields and test against reference solutions
    # Initialize and broadcast solution for comparison later
    if mpi_construct.rank == 0:
        ref_lag_grid_velocity_field = np.random.rand(
            grid_dim, ref_virtual_boundary_forcing.num_lag_nodes
        ).astype(ref_virtual_boundary_forcing.real_t)
        ref_lag_grid_flow_velocity_field = np.random.rand(
            grid_dim, ref_virtual_boundary_forcing.num_lag_nodes
        ).astype(ref_virtual_boundary_forcing.real_t)
    else:
        ref_lag_grid_velocity_field = None
        ref_lag_grid_flow_velocity_field = None
    ref_lag_grid_velocity_field = mpi_construct.grid.bcast(
        ref_lag_grid_velocity_field, root=0
    )
    ref_lag_grid_flow_velocity_field = mpi_construct.grid.bcast(
        ref_lag_grid_flow_velocity_field, root=0
    )
    ref_lag_grid_velocity_mismatch_field = np.zeros_like(ref_lag_grid_velocity_field)
    # Generate reference solution field
    ref_virtual_boundary_forcing.compute_lag_grid_velocity_mismatch_field(
        lag_grid_velocity_mismatch_field=ref_lag_grid_velocity_mismatch_field,
        lag_grid_flow_velocity_field=ref_lag_grid_flow_velocity_field,
        lag_grid_body_velocity_field=ref_lag_grid_velocity_field,
    )
    # Compute solution field
    mpi_virtual_boundary_forcing.local_lag_grid_velocity_field = (
        ref_lag_grid_velocity_field[..., mask]
    )
    mpi_virtual_boundary_forcing.local_lag_grid_flow_velocity_field = (
        ref_lag_grid_flow_velocity_field[..., mask]
    )
    mpi_virtual_boundary_forcing.compute_lag_grid_velocity_mismatch_field(
        lag_grid_velocity_mismatch_field=mpi_virtual_boundary_forcing.local_lag_grid_velocity_mismatch_field,
        lag_grid_flow_velocity_field=mpi_virtual_boundary_forcing.local_lag_grid_flow_velocity_field,
        lag_grid_body_velocity_field=mpi_virtual_boundary_forcing.local_lag_grid_velocity_field,
    )
    # Compare and test solution field
    local_allclose_lag_grid_velocity_mismatch_field = np.allclose(
        ref_lag_grid_velocity_mismatch_field[..., mask],
        mpi_virtual_boundary_forcing.local_lag_grid_velocity_mismatch_field,
        atol=get_test_tol(precision),
    )
    # reduce to make sure each chunk of data in each rank is passing
    allclose_lag_grid_velocity_mismatch_field = mpi_construct.grid.allreduce(
        local_allclose_lag_grid_velocity_mismatch_field, op=MPI.LAND
    )
    # asserting this way ensures that if one rank fails (perhaps due to flaky test),
    # all ranks fail, and pytest can perform rerun without running into deadlock
    assert (
        allclose_lag_grid_velocity_mismatch_field
    ), f"lag grid velocity mismatch field failed [rank {mpi_construct.rank}]"


@pytest.mark.mpi(group="MPI_immersed_boundary_ops_2d", min_size=4)
@pytest.mark.parametrize("grid_dim", [2])
@pytest.mark.parametrize("ghost_size", [2, 3])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rank_distribution", [(1, 0), (0, 1)])
@pytest.mark.parametrize("aspect_ratio", [(1, 1), (1, 2), (2, 1)])
def test_mpi_update_lag_grid_position_mismatch_field_via_euler_forward(
    grid_dim, ghost_size, precision, rank_distribution, aspect_ratio
):
    n_values = 32
    # 1. Generate reference solution
    ref_virtual_boundary_forcing = ReferenceVirtualBoundaryForcing(
        n_values=n_values,
        aspect_ratio=aspect_ratio,
        grid_dim=grid_dim,
        precision=precision,
    )

    # 2. Initialize MPI related stuff to distribute lag nodes to corresponding ranks
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=ref_virtual_boundary_forcing.grid_size_y,
        grid_size_x=ref_virtual_boundary_forcing.grid_size_x,
        real_t=ref_virtual_boundary_forcing.real_t,
        rank_distribution=rank_distribution,
    )
    # Lagrangian grid inter-rank MPI communicator
    master_rank = 0
    mpi_lagrangian_field_communicator = MPILagrangianFieldCommunicator2D(
        eul_grid_dx=ref_virtual_boundary_forcing.dx,
        eul_grid_coord_shift=ref_virtual_boundary_forcing.eul_grid_coord_shift,
        mpi_construct=mpi_construct,
        master_rank=master_rank,
        real_t=ref_virtual_boundary_forcing.real_t,
    )
    # Map lagrangian nodes to ranks
    mpi_lagrangian_field_communicator.map_lagrangian_nodes_based_on_position(
        ref_virtual_boundary_forcing.lag_positions
    )
    rank_address = mpi_lagrangian_field_communicator.rank_address
    mask = np.where(rank_address == mpi_lagrangian_field_communicator.rank)[0]
    # Initialize MPI virtual boundary forcing
    if mpi_construct.rank == master_rank:
        global_ref_lag_grid_position_field = ref_virtual_boundary_forcing.lag_positions
    else:
        global_ref_lag_grid_position_field = None
    mpi_virtual_boundary_forcing = VirtualBoundaryForcingMPI(
        mpi_construct=mpi_construct,
        ghost_size=ghost_size,
        virtual_boundary_stiffness_coeff=ref_virtual_boundary_forcing.virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff=ref_virtual_boundary_forcing.virtual_boundary_damping_coeff,
        grid_dim=ref_virtual_boundary_forcing.grid_dim,
        dx=ref_virtual_boundary_forcing.dx,
        real_t=ref_virtual_boundary_forcing.real_t,
        start_time=ref_virtual_boundary_forcing.time,
        global_lag_grid_position_field=global_ref_lag_grid_position_field,
    )

    # 3. Generate reference fields and test against reference solutions
    # Initialize and broadcast solution for comparison later
    if mpi_construct.rank == 0:
        ref_lag_grid_position_mismatch_field = np.random.rand(
            grid_dim, ref_virtual_boundary_forcing.num_lag_nodes
        ).astype(ref_virtual_boundary_forcing.real_t)
        ref_lag_grid_velocity_mismatch_field = np.random.rand(
            grid_dim, ref_virtual_boundary_forcing.num_lag_nodes
        ).astype(ref_virtual_boundary_forcing.real_t)
    else:
        ref_lag_grid_position_mismatch_field = None
        ref_lag_grid_velocity_mismatch_field = None
    ref_lag_grid_position_mismatch_field = mpi_construct.grid.bcast(
        ref_lag_grid_position_mismatch_field, root=0
    )
    ref_lag_grid_velocity_mismatch_field = mpi_construct.grid.bcast(
        ref_lag_grid_velocity_mismatch_field, root=0
    )
    dt = ref_virtual_boundary_forcing.real_t(0.1)

    mpi_virtual_boundary_forcing.local_lag_grid_position_mismatch_field = (
        ref_lag_grid_position_mismatch_field[..., mask].copy()
    )
    # Compute reference solution after making a copy for testing
    ref_virtual_boundary_forcing.update_lag_grid_position_mismatch_field_via_euler_forward(
        lag_grid_position_mismatch_field=ref_lag_grid_position_mismatch_field,
        lag_grid_velocity_mismatch_field=ref_lag_grid_velocity_mismatch_field,
        dt=dt,
    )
    # Compute solution field
    mpi_virtual_boundary_forcing.local_lag_grid_velocity_mismatch_field = (
        ref_lag_grid_velocity_mismatch_field[..., mask]
    )
    mpi_virtual_boundary_forcing.update_lag_grid_position_mismatch_field_via_euler_forward(
        lag_grid_position_mismatch_field=mpi_virtual_boundary_forcing.local_lag_grid_position_mismatch_field,
        lag_grid_velocity_mismatch_field=mpi_virtual_boundary_forcing.local_lag_grid_velocity_mismatch_field,
        dt=dt,
    )
    # Compare and test solution field
    local_allclose_lag_grid_position_mismatch_field = np.allclose(
        ref_lag_grid_position_mismatch_field[..., mask],
        mpi_virtual_boundary_forcing.local_lag_grid_position_mismatch_field,
        atol=get_test_tol(precision),
    )
    # reduce to make sure each chunk of data in each rank is passing
    allclose_lag_grid_position_mismatch_field = mpi_construct.grid.allreduce(
        local_allclose_lag_grid_position_mismatch_field, op=MPI.LAND
    )
    # asserting this way ensures that if one rank fails (perhaps due to flaky test),
    # all ranks fail, and pytest can perform rerun without running into deadlock
    assert (
        allclose_lag_grid_position_mismatch_field
    ), f"lag grid position mismatch field failed [rank {mpi_construct.rank}]"


@pytest.mark.mpi(group="MPI_immersed_boundary_ops_2d", min_size=4)
@pytest.mark.parametrize("grid_dim", [2])
@pytest.mark.parametrize("ghost_size", [2, 3])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rank_distribution", [(1, 0), (0, 1)])
@pytest.mark.parametrize("aspect_ratio", [(1, 1), (1, 2), (2, 1)])
def test_mpi_compute_lag_grid_forcing_field(
    grid_dim, ghost_size, precision, rank_distribution, aspect_ratio
):
    n_values = 32
    # 1. Generate reference solution
    ref_virtual_boundary_forcing = ReferenceVirtualBoundaryForcing(
        n_values=n_values,
        aspect_ratio=aspect_ratio,
        grid_dim=grid_dim,
        precision=precision,
    )

    # 2. Initialize MPI related stuff to distribute lag nodes to corresponding ranks
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=ref_virtual_boundary_forcing.grid_size_y,
        grid_size_x=ref_virtual_boundary_forcing.grid_size_x,
        real_t=ref_virtual_boundary_forcing.real_t,
        rank_distribution=rank_distribution,
    )
    # Lagrangian grid inter-rank MPI communicator
    master_rank = 0
    mpi_lagrangian_field_communicator = MPILagrangianFieldCommunicator2D(
        eul_grid_dx=ref_virtual_boundary_forcing.dx,
        eul_grid_coord_shift=ref_virtual_boundary_forcing.eul_grid_coord_shift,
        mpi_construct=mpi_construct,
        master_rank=master_rank,
        real_t=ref_virtual_boundary_forcing.real_t,
    )
    # Map lagrangian nodes to ranks
    mpi_lagrangian_field_communicator.map_lagrangian_nodes_based_on_position(
        ref_virtual_boundary_forcing.lag_positions
    )
    rank_address = mpi_lagrangian_field_communicator.rank_address
    mask = np.where(rank_address == mpi_lagrangian_field_communicator.rank)[0]
    # Initialize MPI virtual boundary forcing
    if mpi_construct.rank == master_rank:
        global_ref_lag_grid_position_field = ref_virtual_boundary_forcing.lag_positions
    else:
        global_ref_lag_grid_position_field = None
    mpi_virtual_boundary_forcing = VirtualBoundaryForcingMPI(
        mpi_construct=mpi_construct,
        ghost_size=ghost_size,
        virtual_boundary_stiffness_coeff=ref_virtual_boundary_forcing.virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff=ref_virtual_boundary_forcing.virtual_boundary_damping_coeff,
        grid_dim=ref_virtual_boundary_forcing.grid_dim,
        dx=ref_virtual_boundary_forcing.dx,
        real_t=ref_virtual_boundary_forcing.real_t,
        start_time=ref_virtual_boundary_forcing.time,
        global_lag_grid_position_field=global_ref_lag_grid_position_field,
    )

    # 3. Generate reference fields and test against reference solutions
    # Initialize and broadcast solution for comparison later
    if mpi_construct.rank == 0:
        ref_lag_grid_position_mismatch_field = np.random.rand(
            grid_dim, ref_virtual_boundary_forcing.num_lag_nodes
        ).astype(ref_virtual_boundary_forcing.real_t)
        ref_lag_grid_velocity_mismatch_field = np.random.rand(
            grid_dim, ref_virtual_boundary_forcing.num_lag_nodes
        ).astype(ref_virtual_boundary_forcing.real_t)
    else:
        ref_lag_grid_position_mismatch_field = None
        ref_lag_grid_velocity_mismatch_field = None
    ref_lag_grid_position_mismatch_field = mpi_construct.grid.bcast(
        ref_lag_grid_position_mismatch_field, root=0
    )
    ref_lag_grid_velocity_mismatch_field = mpi_construct.grid.bcast(
        ref_lag_grid_velocity_mismatch_field, root=0
    )
    ref_lag_grid_forcing_field = np.zeros_like(ref_lag_grid_position_mismatch_field)
    # Compute reference solution
    ref_virtual_boundary_forcing.compute_lag_grid_forcing_field(
        lag_grid_forcing_field=ref_lag_grid_forcing_field,
        lag_grid_position_mismatch_field=ref_lag_grid_position_mismatch_field,
        lag_grid_velocity_mismatch_field=ref_lag_grid_velocity_mismatch_field,
        virtual_boundary_stiffness_coeff=ref_virtual_boundary_forcing.virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff=ref_virtual_boundary_forcing.virtual_boundary_damping_coeff,
    )
    # Compute solution field
    mpi_virtual_boundary_forcing.local_lag_grid_position_mismatch_field = (
        ref_lag_grid_position_mismatch_field[..., mask]
    )
    mpi_virtual_boundary_forcing.local_lag_grid_velocity_mismatch_field = (
        ref_lag_grid_velocity_mismatch_field[..., mask]
    )
    mpi_virtual_boundary_forcing.compute_lag_grid_forcing_field(
        lag_grid_forcing_field=mpi_virtual_boundary_forcing.local_lag_grid_forcing_field,
        lag_grid_position_mismatch_field=mpi_virtual_boundary_forcing.local_lag_grid_position_mismatch_field,
        lag_grid_velocity_mismatch_field=mpi_virtual_boundary_forcing.local_lag_grid_velocity_mismatch_field,
        virtual_boundary_stiffness_coeff=mpi_virtual_boundary_forcing.virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff=mpi_virtual_boundary_forcing.virtual_boundary_damping_coeff,
    )
    # Compare and test solution field
    local_allclose_lag_grid_forcing_field = np.allclose(
        ref_lag_grid_forcing_field[..., mask],
        mpi_virtual_boundary_forcing.local_lag_grid_forcing_field,
        atol=get_test_tol(precision),
    )
    # reduce to make sure each chunk of data in each rank is passing
    allclose_lag_grid_forcing_field = mpi_construct.grid.allreduce(
        local_allclose_lag_grid_forcing_field, op=MPI.LAND
    )
    # asserting this way ensures that if one rank fails (perhaps due to flaky test),
    # all ranks fail, and pytest can perform rerun without running into deadlock
    assert (
        allclose_lag_grid_forcing_field
    ), f"lag grid forcing field failed [rank {mpi_construct.rank}]"


@pytest.mark.mpi(group="MPI_immersed_boundary_ops_2d", min_size=4)
@pytest.mark.parametrize("grid_dim", [2])
@pytest.mark.parametrize("ghost_size", [2, 3])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rank_distribution", [(1, 0), (0, 1)])
@pytest.mark.parametrize("aspect_ratio", [(1, 1), (1, 2), (2, 1)])
def test_mpi_compute_interaction_force_on_lag_grid(
    grid_dim, ghost_size, precision, rank_distribution, aspect_ratio
):
    n_values = 32
    # 1. Generate reference solution
    ref_virtual_boundary_forcing = ReferenceVirtualBoundaryForcing(
        n_values=n_values,
        aspect_ratio=aspect_ratio,
        grid_dim=grid_dim,
        precision=precision,
    )

    # 2. Initialize MPI related stuff to distribute lag nodes to corresponding ranks
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=ref_virtual_boundary_forcing.grid_size_y,
        grid_size_x=ref_virtual_boundary_forcing.grid_size_x,
        real_t=ref_virtual_boundary_forcing.real_t,
        rank_distribution=rank_distribution,
    )
    # Lagrangian grid inter-rank MPI communicator
    master_rank = 0
    mpi_lagrangian_field_communicator = MPILagrangianFieldCommunicator2D(
        eul_grid_dx=ref_virtual_boundary_forcing.dx,
        eul_grid_coord_shift=ref_virtual_boundary_forcing.eul_grid_coord_shift,
        mpi_construct=mpi_construct,
        master_rank=master_rank,
        real_t=ref_virtual_boundary_forcing.real_t,
    )
    # Map lagrangian nodes to ranks
    mpi_lagrangian_field_communicator.map_lagrangian_nodes_based_on_position(
        ref_virtual_boundary_forcing.lag_positions
    )
    rank_address = mpi_lagrangian_field_communicator.rank_address
    mask = np.where(rank_address == mpi_lagrangian_field_communicator.rank)[0]
    # Initialize MPI virtual boundary forcing
    if mpi_construct.rank == master_rank:
        ref_global_lag_grid_position_field = ref_virtual_boundary_forcing.lag_positions
    else:
        ref_global_lag_grid_position_field = None
    mpi_virtual_boundary_forcing = VirtualBoundaryForcingMPI(
        mpi_construct=mpi_construct,
        ghost_size=ghost_size,
        virtual_boundary_stiffness_coeff=ref_virtual_boundary_forcing.virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff=ref_virtual_boundary_forcing.virtual_boundary_damping_coeff,
        grid_dim=ref_virtual_boundary_forcing.grid_dim,
        dx=ref_virtual_boundary_forcing.dx,
        real_t=ref_virtual_boundary_forcing.real_t,
        start_time=ref_virtual_boundary_forcing.time,
        global_lag_grid_position_field=ref_global_lag_grid_position_field,
    )

    # Field and ghost communicator for eul grid velocity field later
    mpi_ghost_exchange_communicator = MPIGhostCommunicator2D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )
    mpi_field_communicator = MPIFieldCommunicator2D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )
    scatter_global_field = mpi_field_communicator.scatter_global_field

    # 3. Generate reference fields and test against reference solutions
    # Initialize and broadcast solution for comparison later
    if mpi_construct.rank == 0:
        ref_lag_grid_velocity_field = np.random.rand(
            grid_dim, ref_virtual_boundary_forcing.num_lag_nodes
        ).astype(ref_virtual_boundary_forcing.real_t)
        ref_eul_grid_velocity_field = np.random.rand(
            grid_dim,
            ref_virtual_boundary_forcing.grid_size_y,
            ref_virtual_boundary_forcing.grid_size_x,
        ).astype(ref_virtual_boundary_forcing.real_t)
    else:
        ref_lag_grid_velocity_field = None
        ref_eul_grid_velocity_field = None
    ref_lag_grid_velocity_field = mpi_construct.grid.bcast(
        ref_lag_grid_velocity_field, root=0
    )
    ref_eul_grid_velocity_field = mpi_construct.grid.bcast(
        ref_eul_grid_velocity_field, root=0
    )
    ref_lag_grid_position_field = ref_virtual_boundary_forcing.lag_positions

    ref_virtual_boundary_forcing.compute_interaction_force_on_lag_grid(
        eul_grid_velocity_field=ref_eul_grid_velocity_field,
        lag_grid_position_field=ref_lag_grid_position_field,
        lag_grid_velocity_field=ref_lag_grid_velocity_field,
    )

    # Allocate local eul grid velocity field
    mpi_local_eul_grid_velocity_field = np.zeros(
        (
            grid_dim,
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
        )
    ).astype(ref_virtual_boundary_forcing.real_t)
    # scatter global reference eul grid velocity field
    scatter_global_field(
        mpi_local_eul_grid_velocity_field[0],
        ref_eul_grid_velocity_field[0],
        mpi_construct,
    )
    scatter_global_field(
        mpi_local_eul_grid_velocity_field[1],
        ref_eul_grid_velocity_field[1],
        mpi_construct,
    )
    # ghost the local field
    mpi_ghost_exchange_communicator.exchange_init(
        mpi_local_eul_grid_velocity_field[0], mpi_construct
    )
    mpi_ghost_exchange_communicator.exchange_finalise()
    mpi_ghost_exchange_communicator.exchange_init(
        mpi_local_eul_grid_velocity_field[1], mpi_construct
    )
    mpi_ghost_exchange_communicator.exchange_finalise()

    if mpi_construct.rank == master_rank:
        global_ref_lag_grid_position_field = ref_lag_grid_position_field
        global_ref_lag_grid_velocity_field = ref_lag_grid_velocity_field
    else:
        global_ref_lag_grid_position_field = None
        global_ref_lag_grid_velocity_field = None
    mpi_virtual_boundary_forcing.compute_interaction_force_on_lag_grid(
        local_eul_grid_velocity_field=mpi_local_eul_grid_velocity_field,
        global_lag_grid_position_field=global_ref_lag_grid_position_field,
        global_lag_grid_velocity_field=global_ref_lag_grid_velocity_field,
    )

    # Compare and test solution field
    (
        local_allclose_local_nearest_eul_grid_index_to_lag_grid,
        local_allclose_lag_grid_flow_velocity_field,
        local_allclose_lag_grid_velocity_mismatch_field,
        local_allclose_lag_grid_forcing_field,
    ) = ref_virtual_boundary_forcing.check_lag_grid_interaction_solution(
        virtual_boundary_forcing=mpi_virtual_boundary_forcing, mask=mask
    )
    # reduce to make sure each chunk of data in each rank is passing
    allclose_local_nearest_eul_grid_index_to_lag_grid = mpi_construct.grid.allreduce(
        local_allclose_local_nearest_eul_grid_index_to_lag_grid, op=MPI.LAND
    )
    allclose_lag_grid_flow_velocity_field = mpi_construct.grid.allreduce(
        local_allclose_lag_grid_flow_velocity_field, op=MPI.LAND
    )
    allclose_lag_grid_velocity_mismatch_field = mpi_construct.grid.allreduce(
        local_allclose_lag_grid_velocity_mismatch_field, op=MPI.LAND
    )
    allclose_lag_grid_forcing_field = mpi_construct.grid.allreduce(
        local_allclose_lag_grid_forcing_field, op=MPI.LAND
    )
    # asserting this way ensures that if one rank fails (perhaps due to flaky test),
    # all ranks fail, and pytest can perform rerun without running into deadlock
    assert (
        allclose_local_nearest_eul_grid_index_to_lag_grid
    ), f"local nearest eul grid index to lag grid failed [rank {mpi_construct.rank}]"
    assert (
        allclose_lag_grid_flow_velocity_field
    ), f"lag grid flow velocity field failed [rank {mpi_construct.rank}]"
    assert (
        allclose_lag_grid_velocity_mismatch_field
    ), f"lag grid velocity mismatch field failed [rank {mpi_construct.rank}]"
    assert (
        allclose_lag_grid_forcing_field
    ), f"lag grid forcing field failed [rank {mpi_construct.rank}]"


@pytest.mark.mpi(group="MPI_immersed_boundary_ops_2d", min_size=4)
@pytest.mark.parametrize("grid_dim", [2])
@pytest.mark.parametrize("ghost_size", [2, 3])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rank_distribution", [(1, 0), (0, 1)])
@pytest.mark.parametrize("aspect_ratio", [(1, 1), (1, 2), (2, 1)])
@pytest.mark.parametrize("enable_eul_grid_forcing_reset", [True, False])
def test_mpi_compute_interaction_force_on_eul_and_lag_grid(
    grid_dim,
    ghost_size,
    precision,
    rank_distribution,
    aspect_ratio,
    enable_eul_grid_forcing_reset,
):
    n_values = 32
    # 1. Generate reference solution
    ref_virtual_boundary_forcing = ReferenceVirtualBoundaryForcing(
        n_values=n_values,
        aspect_ratio=aspect_ratio,
        grid_dim=grid_dim,
        precision=precision,
        enable_eul_grid_forcing_reset=enable_eul_grid_forcing_reset,
    )

    # 2. Initialize MPI related stuff to distribute lag nodes to corresponding ranks
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=ref_virtual_boundary_forcing.grid_size_y,
        grid_size_x=ref_virtual_boundary_forcing.grid_size_x,
        real_t=ref_virtual_boundary_forcing.real_t,
        rank_distribution=rank_distribution,
    )
    # Lagrangian grid inter-rank MPI communicator
    master_rank = 0
    mpi_lagrangian_field_communicator = MPILagrangianFieldCommunicator2D(
        eul_grid_dx=ref_virtual_boundary_forcing.dx,
        eul_grid_coord_shift=ref_virtual_boundary_forcing.eul_grid_coord_shift,
        mpi_construct=mpi_construct,
        master_rank=master_rank,
        real_t=ref_virtual_boundary_forcing.real_t,
    )
    # Map lagrangian nodes to ranks
    mpi_lagrangian_field_communicator.map_lagrangian_nodes_based_on_position(
        ref_virtual_boundary_forcing.lag_positions
    )
    rank_address = mpi_lagrangian_field_communicator.rank_address
    mask = np.where(rank_address == mpi_lagrangian_field_communicator.rank)[0]
    # Initialize MPI virtual boundary forcing
    if mpi_construct.rank == master_rank:
        global_ref_lag_grid_position_field = ref_virtual_boundary_forcing.lag_positions
    else:
        global_ref_lag_grid_position_field = None
    mpi_virtual_boundary_forcing = VirtualBoundaryForcingMPI(
        mpi_construct=mpi_construct,
        ghost_size=ghost_size,
        virtual_boundary_stiffness_coeff=ref_virtual_boundary_forcing.virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff=ref_virtual_boundary_forcing.virtual_boundary_damping_coeff,
        grid_dim=ref_virtual_boundary_forcing.grid_dim,
        dx=ref_virtual_boundary_forcing.dx,
        real_t=ref_virtual_boundary_forcing.real_t,
        start_time=ref_virtual_boundary_forcing.time,
        enable_eul_grid_forcing_reset=enable_eul_grid_forcing_reset,
        global_lag_grid_position_field=global_ref_lag_grid_position_field,
    )

    # Field and ghost communicator for eul grid velocity field later
    mpi_ghost_exchange_communicator = MPIGhostCommunicator2D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )
    mpi_field_communicator = MPIFieldCommunicator2D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )
    scatter_global_field = mpi_field_communicator.scatter_global_field

    # 3. Generate reference fields and test against reference solutions
    # Initialize and broadcast solution for comparison later
    if mpi_construct.rank == 0:
        ref_lag_grid_velocity_field = np.random.rand(
            grid_dim, ref_virtual_boundary_forcing.num_lag_nodes
        ).astype(ref_virtual_boundary_forcing.real_t)
        ref_eul_grid_velocity_field = np.random.rand(
            grid_dim,
            ref_virtual_boundary_forcing.grid_size_y,
            ref_virtual_boundary_forcing.grid_size_x,
        ).astype(ref_virtual_boundary_forcing.real_t)
    else:
        ref_lag_grid_velocity_field = None
        ref_eul_grid_velocity_field = None
    ref_lag_grid_velocity_field = mpi_construct.grid.bcast(
        ref_lag_grid_velocity_field, root=0
    )
    ref_eul_grid_velocity_field = mpi_construct.grid.bcast(
        ref_eul_grid_velocity_field, root=0
    )
    ref_lag_grid_position_field = ref_virtual_boundary_forcing.lag_positions
    ref_eul_grid_forcing_field = np.zeros_like(ref_eul_grid_velocity_field)
    # Compute reference solution
    ref_virtual_boundary_forcing.compute_interaction_force_on_eul_and_lag_grid(
        eul_grid_forcing_field=ref_eul_grid_forcing_field,
        eul_grid_velocity_field=ref_eul_grid_velocity_field,
        lag_grid_position_field=ref_lag_grid_position_field,
        lag_grid_velocity_field=ref_lag_grid_velocity_field,
    )

    # Allocate local eul grid velocity field
    mpi_local_eul_grid_velocity_field = np.zeros(
        (
            grid_dim,
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
        )
    ).astype(ref_virtual_boundary_forcing.real_t)
    mpi_local_eul_grid_forcing_field = np.zeros_like(mpi_local_eul_grid_velocity_field)
    # scatter global reference eul grid velocity field
    scatter_global_field(
        mpi_local_eul_grid_velocity_field[0],
        ref_eul_grid_velocity_field[0],
        mpi_construct,
    )
    scatter_global_field(
        mpi_local_eul_grid_velocity_field[1],
        ref_eul_grid_velocity_field[1],
        mpi_construct,
    )
    # ghost the local field
    mpi_ghost_exchange_communicator.exchange_init(
        mpi_local_eul_grid_velocity_field[0], mpi_construct
    )
    mpi_ghost_exchange_communicator.exchange_finalise()
    mpi_ghost_exchange_communicator.exchange_init(
        mpi_local_eul_grid_velocity_field[1], mpi_construct
    )
    mpi_ghost_exchange_communicator.exchange_finalise()

    if mpi_construct.rank == master_rank:
        global_ref_lag_grid_velocity_field = ref_lag_grid_velocity_field
    else:
        global_ref_lag_grid_velocity_field = None
    # Compute solution field
    mpi_virtual_boundary_forcing.compute_interaction_force_on_eul_and_lag_grid(
        local_eul_grid_forcing_field=mpi_local_eul_grid_forcing_field,
        local_eul_grid_velocity_field=mpi_local_eul_grid_velocity_field,
        global_lag_grid_position_field=global_ref_lag_grid_position_field,
        global_lag_grid_velocity_field=global_ref_lag_grid_velocity_field,
    )

    # Compare and test solution field
    # Check lag grid solution
    (
        local_allclose_local_nearest_eul_grid_index_to_lag_grid,
        local_allclose_lag_grid_flow_velocity_field,
        local_allclose_lag_grid_velocity_mismatch_field,
        local_allclose_lag_grid_forcing_field,
    ) = ref_virtual_boundary_forcing.check_lag_grid_interaction_solution(
        virtual_boundary_forcing=mpi_virtual_boundary_forcing, mask=mask
    )
    # reduce to make sure each chunk of data in each rank is passing
    allclose_local_nearest_eul_grid_index_to_lag_grid = mpi_construct.grid.allreduce(
        local_allclose_local_nearest_eul_grid_index_to_lag_grid, op=MPI.LAND
    )
    allclose_lag_grid_flow_velocity_field = mpi_construct.grid.allreduce(
        local_allclose_lag_grid_flow_velocity_field, op=MPI.LAND
    )
    allclose_lag_grid_velocity_mismatch_field = mpi_construct.grid.allreduce(
        local_allclose_lag_grid_velocity_mismatch_field, op=MPI.LAND
    )
    allclose_lag_grid_forcing_field = mpi_construct.grid.allreduce(
        local_allclose_lag_grid_forcing_field, op=MPI.LAND
    )
    # asserting this way ensures that if one rank fails (perhaps due to flaky test),
    # all ranks fail, and pytest can perform rerun without running into deadlock
    assert (
        allclose_local_nearest_eul_grid_index_to_lag_grid
    ), f"local nearest eul grid index to lag grid failed [rank {mpi_construct.rank}]"
    assert (
        allclose_lag_grid_flow_velocity_field
    ), f"lag grid flow velocity field failed [rank {mpi_construct.rank}]"
    assert (
        allclose_lag_grid_velocity_mismatch_field
    ), f"lag grid velocity mismatch field failed [rank {mpi_construct.rank}]"
    assert (
        allclose_lag_grid_forcing_field
    ), f"lag grid forcing field failed [rank {mpi_construct.rank}]"
    # Check eul grid solution
    # Get corresponding local eul grid chunk of ref solution
    local_grid_size_y, local_grid_size_x = mpi_construct.local_grid_size
    mpi_substart_idx = mpi_construct.grid.coords * mpi_construct.local_grid_size
    mpi_local_sol_idx = (
        slice(None, None),
        slice(mpi_substart_idx[0], mpi_substart_idx[0] + local_grid_size_y),
        slice(mpi_substart_idx[1], mpi_substart_idx[1] + local_grid_size_x),
    )
    # Check eul grid solution
    local_allclose_eul_grid_forcing_field = np.allclose(
        ref_eul_grid_forcing_field[mpi_local_sol_idx],
        mpi_local_eul_grid_forcing_field[
            :, ghost_size:-ghost_size, ghost_size:-ghost_size
        ],
        atol=get_test_tol(precision),
    )
    allclose_eul_grid_forcing_field = mpi_construct.grid.allreduce(
        local_allclose_eul_grid_forcing_field, op=MPI.LAND
    )
    assert (
        allclose_eul_grid_forcing_field
    ), f"eul grid forcing field failed [rank {mpi_construct.rank}]"


@pytest.mark.mpi(group="MPI_immersed_boundary_ops_2d", min_size=4)
@pytest.mark.parametrize("grid_dim", [2])
@pytest.mark.parametrize("ghost_size", [2, 3])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rank_distribution", [(1, 0), (0, 1)])
@pytest.mark.parametrize("aspect_ratio", [(1, 1), (1, 2), (2, 1)])
def test_mpi_virtual_boundary_forcing_time_step(
    grid_dim, ghost_size, precision, rank_distribution, aspect_ratio
):

    n_values = 32
    # 1. Generate reference solution
    ref_virtual_boundary_forcing = ReferenceVirtualBoundaryForcing(
        n_values=n_values,
        aspect_ratio=aspect_ratio,
        grid_dim=grid_dim,
        precision=precision,
    )

    # 2. Initialize MPI related stuff to distribute lag nodes to corresponding ranks
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=ref_virtual_boundary_forcing.grid_size_y,
        grid_size_x=ref_virtual_boundary_forcing.grid_size_x,
        real_t=ref_virtual_boundary_forcing.real_t,
        rank_distribution=rank_distribution,
    )
    # Lagrangian grid inter-rank MPI communicator
    master_rank = 0
    mpi_lagrangian_field_communicator = MPILagrangianFieldCommunicator2D(
        eul_grid_dx=ref_virtual_boundary_forcing.dx,
        eul_grid_coord_shift=ref_virtual_boundary_forcing.eul_grid_coord_shift,
        mpi_construct=mpi_construct,
        master_rank=master_rank,
        real_t=ref_virtual_boundary_forcing.real_t,
    )
    # Map lagrangian nodes to ranks
    mpi_lagrangian_field_communicator.map_lagrangian_nodes_based_on_position(
        ref_virtual_boundary_forcing.lag_positions
    )
    rank_address = mpi_lagrangian_field_communicator.rank_address
    mask = np.where(rank_address == mpi_lagrangian_field_communicator.rank)[0]
    # Initialize MPI virtual boundary forcing
    if mpi_construct.rank == master_rank:
        global_ref_lag_grid_position_field = ref_virtual_boundary_forcing.lag_positions
    else:
        global_ref_lag_grid_position_field = None
    mpi_virtual_boundary_forcing = VirtualBoundaryForcingMPI(
        mpi_construct=mpi_construct,
        ghost_size=ghost_size,
        virtual_boundary_stiffness_coeff=ref_virtual_boundary_forcing.virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff=ref_virtual_boundary_forcing.virtual_boundary_damping_coeff,
        grid_dim=ref_virtual_boundary_forcing.grid_dim,
        dx=ref_virtual_boundary_forcing.dx,
        real_t=ref_virtual_boundary_forcing.real_t,
        start_time=ref_virtual_boundary_forcing.time,
        global_lag_grid_position_field=global_ref_lag_grid_position_field,
    )

    # 3. Generate reference fields and test against reference solutions
    # Initialize and broadcast solution for comparison later
    if mpi_construct.rank == 0:
        ref_lag_grid_position_mismatch_field = np.random.rand(
            grid_dim, ref_virtual_boundary_forcing.num_lag_nodes
        ).astype(ref_virtual_boundary_forcing.real_t)
        ref_lag_grid_velocity_mismatch_field = np.random.rand(
            grid_dim, ref_virtual_boundary_forcing.num_lag_nodes
        ).astype(ref_virtual_boundary_forcing.real_t)
    else:
        ref_lag_grid_position_mismatch_field = None
        ref_lag_grid_velocity_mismatch_field = None
    ref_lag_grid_position_mismatch_field = mpi_construct.grid.bcast(
        ref_lag_grid_position_mismatch_field, root=0
    )
    ref_lag_grid_velocity_mismatch_field = mpi_construct.grid.bcast(
        ref_lag_grid_velocity_mismatch_field, root=0
    )
    dt = ref_virtual_boundary_forcing.real_t(0.1)
    # Compute reference solution after making a copy for testing
    ref_virtual_boundary_forcing.lag_grid_position_mismatch_field = (
        ref_lag_grid_position_mismatch_field.copy()
    )
    ref_virtual_boundary_forcing.lag_grid_velocity_mismatch_field = (
        ref_lag_grid_velocity_mismatch_field.copy()
    )
    ref_virtual_boundary_forcing.time_step(dt=dt)
    # Compute mpi local solution
    mpi_virtual_boundary_forcing.local_lag_grid_position_mismatch_field = (
        ref_lag_grid_position_mismatch_field[..., mask]
    )
    mpi_virtual_boundary_forcing.local_lag_grid_velocity_mismatch_field = (
        ref_lag_grid_velocity_mismatch_field[..., mask]
    )
    mpi_virtual_boundary_forcing.time_step(dt=dt)
    # Compare and test solution fields
    local_allclose_lag_grid_position_mismatch_field = np.allclose(
        ref_virtual_boundary_forcing.lag_grid_position_mismatch_field[..., mask],
        mpi_virtual_boundary_forcing.local_lag_grid_position_mismatch_field,
        atol=get_test_tol(precision),
    )
    local_allclose_time = np.allclose(
        ref_virtual_boundary_forcing.time,
        mpi_virtual_boundary_forcing.time,
        atol=get_test_tol(precision),
    )
    allclose_lag_grid_position_mismatch_field = mpi_construct.grid.allreduce(
        local_allclose_lag_grid_position_mismatch_field, op=MPI.LAND
    )
    allclose_time = mpi_construct.grid.allreduce(local_allclose_time, op=MPI.LAND)
    assert (
        allclose_lag_grid_position_mismatch_field
    ), f"lag grid position mismatch field failed [rank {mpi_construct.rank}]"
    assert allclose_time, f"time failed [rank {mpi_construct.rank}]"
