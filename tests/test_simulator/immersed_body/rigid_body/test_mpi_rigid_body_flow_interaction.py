import pytest
import numpy as np
import sopht_mpi.simulator as sps
from sopht.utils.precision import get_real_t
from sopht_mpi.utils import MPIConstruct2D, MPIGhostCommunicator2D
import elastica as ea
from sopht.simulator.immersed_body import CircularCylinderForcingGrid


def mock_2d_cylinder():
    """Returns a mock 2D cylinder (from elastica) for testing"""
    cyl_radius = 0.1
    start = np.array([1.0, 2.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([1.0, 0.0, 0.0])
    base_length = 1.0
    cylinder = ea.Cylinder(
        start, direction, normal, base_length, cyl_radius, density=1e3
    )
    cylinder.velocity_collection[...] = 3.0
    cylinder.omega_collection[...] = 4.0
    return cylinder


@pytest.mark.mpi(group="MPI_rigid_body_flow_interaction", min_size=4)
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("master_rank", [0, 1])
def test_mpi_rigid_body_flow_interaction(precision, master_rank):
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
    cylinder = mock_2d_cylinder()
    forcing_grid_cls = CircularCylinderForcingGrid
    cylinder_flow_interactor = sps.RigidBodyFlowInteractionMPI(
        mpi_construct=mpi_construct,
        mpi_ghost_exchange_communicator=mpi_ghost_exchange_communicator,
        rigid_body=cylinder,
        eul_grid_forcing_field=np.zeros(grid_size),
        eul_grid_velocity_field=np.zeros(grid_size),
        virtual_boundary_stiffness_coeff=1.0,
        virtual_boundary_damping_coeff=1.0,
        dx=1.0,
        grid_dim=2,
        master_rank=master_rank,
        forcing_grid_cls=forcing_grid_cls,
        num_forcing_points=16,
    )
    rigid_body_dim = 3
    np.testing.assert_allclose(
        cylinder_flow_interactor.body_flow_forces, np.zeros((rigid_body_dim, 1))
    )
    np.testing.assert_allclose(
        cylinder_flow_interactor.body_flow_torques, np.zeros((rigid_body_dim, 1))
    )
    if mpi_construct.rank == master_rank:
        assert isinstance(cylinder_flow_interactor.forcing_grid, forcing_grid_cls)
    else:
        assert isinstance(cylinder_flow_interactor.forcing_grid, sps.EmptyForcingGrid)
