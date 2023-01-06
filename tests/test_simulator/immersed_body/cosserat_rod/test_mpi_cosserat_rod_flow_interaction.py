import pytest
import numpy as np
import sopht_mpi.simulator as sps
from sopht.utils.precision import get_real_t
from sopht_mpi.utils import MPIConstruct2D, MPIGhostCommunicator2D
import elastica as ea
from sopht.simulator.immersed_body import CosseratRodElementCentricForcingGrid


def mock_straight_rod(n_elems, **kwargs):
    """Returns a straight rod aligned x = y = z plane for testing."""
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 1.0, 1.0])
    normal = np.array([0.0, -1.0, 1.0])
    rod_length = 1.0
    base_radius = kwargs.get("base_radius", 0.05)
    staight_rod = ea.CosseratRod.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        rod_length,
        base_radius,
        density=1e3,
        nu=0.0,  # internal damping constant, deprecated in v0.3.0
        youngs_modulus=1e6,
        shear_modulus=1e6 / (0.5 + 1.0),
    )
    n_nodes = n_elems + 1
    staight_rod.velocity_collection[...] = np.linspace(1, n_nodes, n_nodes)
    staight_rod.omega_collection[...] = np.linspace(1, n_elems, n_elems)
    return staight_rod


@pytest.mark.mpi(group="MPI_cosserat_rod_flow_interaction", min_size=4)
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("master_rank", [0, 1])
@pytest.mark.parametrize("n_elems", [8, 16])
def test_mpi_cosserat_rod_flow_interaction(precision, master_rank, n_elems):
    # Initialize minimal mpi constructs
    grid_size = (16, 16)
    real_t = get_real_t(precision)
    mpi_construct = MPIConstruct2D(
        grid_size_y=grid_size[0],
        grid_size_x=grid_size[1],
        real_t=real_t,
    )
    ghost_size = 2
    mpi_ghost_exchange_communicator = MPIGhostCommunicator2D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )

    # Initialize interactor
    cosserat_rod = mock_straight_rod(n_elems)
    forcing_grid_cls = CosseratRodElementCentricForcingGrid
    rod_flow_interactor = sps.CosseratRodFlowInteraction(
        mpi_construct=mpi_construct,
        mpi_ghost_exchange_communicator=mpi_ghost_exchange_communicator,
        cosserat_rod=cosserat_rod,
        eul_grid_forcing_field=np.zeros(grid_size),
        eul_grid_velocity_field=np.zeros(grid_size),
        virtual_boundary_stiffness_coeff=1.0,
        virtual_boundary_damping_coeff=1.0,
        dx=1.0,
        grid_dim=2,
        master_rank=master_rank,
        forcing_grid_cls=forcing_grid_cls,
    )
    rod_dim = 3
    np.testing.assert_allclose(
        rod_flow_interactor.body_flow_forces,
        np.zeros((rod_dim, cosserat_rod.n_elems + 1)),
    )
    np.testing.assert_allclose(
        rod_flow_interactor.body_flow_torques, np.zeros((rod_dim, cosserat_rod.n_elems))
    )
    if mpi_construct.rank == master_rank:
        assert isinstance(rod_flow_interactor.forcing_grid, forcing_grid_cls)
    else:
        assert isinstance(rod_flow_interactor.forcing_grid, sps.EmptyForcingGrid)
