import logging
import numpy as np
import pytest
from sopht.utils.precision import get_real_t
import sopht_mpi.sopht_mpi_simulator as sps
from sopht_mpi.utils import MPIConstruct2D, MPIGhostCommunicator2D
from tests.test_simulator.immersed_body.rigid_body.test_rigid_body_forcing_grids import (
    mock_2d_cylinder,
)
from mpi4py import MPI


def mock_2d_cylinder_flow_interactor(
    num_forcing_points=16, master_rank=0, precision="single"
):
    """Returns a mock 2D cylinder flow interactor and related fields for testing"""
    grid_dim = 2
    ghost_size = 2
    real_t = get_real_t(precision)
    cylinder = mock_2d_cylinder()
    grid_size = (16,) * grid_dim
    mpi_construct = MPIConstruct2D(
        grid_size_y=grid_size[0], grid_size_x=grid_size[1], real_t=real_t
    )
    mpi_ghost_exchange_communicator = MPIGhostCommunicator2D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )
    local_eul_grid_velocity_field = np.random.rand(
        grid_dim, *(mpi_construct.local_grid_size + 2 * ghost_size)
    ).astype(real_t)
    local_eul_grid_forcing_field = np.zeros_like(local_eul_grid_velocity_field)
    # chosen so that cylinder lies within domain
    dx = cylinder.length / 4.0
    cylinder_flow_interactor = sps.RigidBodyFlowInteractionMPI(
        mpi_construct=mpi_construct,
        mpi_ghost_exchange_communicator=mpi_ghost_exchange_communicator,
        rigid_body=cylinder,
        eul_grid_forcing_field=local_eul_grid_forcing_field,
        eul_grid_velocity_field=local_eul_grid_velocity_field,
        virtual_boundary_stiffness_coeff=1.0,
        virtual_boundary_damping_coeff=1.0,
        dx=dx,
        grid_dim=grid_dim,
        real_t=real_t,
        master_rank=master_rank,
        forcing_grid_cls=sps.CircularCylinderForcingGrid,
        num_forcing_points=num_forcing_points,
    )
    return (
        cylinder_flow_interactor,
        local_eul_grid_forcing_field,
        local_eul_grid_velocity_field,
        dx,
    )


@pytest.mark.mpi(group="MPI_immersed_body_interaction", min_size=4)
@pytest.mark.parametrize("num_forcing_points", [1, 4, 64])
@pytest.mark.parametrize("master_rank", [0, 1])
def test_mpi_immersed_body_interactor_warnings(num_forcing_points, master_rank, caplog):
    with caplog.at_level(logging.WARNING):
        cylinder_flow_interactor, _, _, dx = mock_2d_cylinder_flow_interactor(
            num_forcing_points=num_forcing_points, master_rank=master_rank
        )
    max_lag_grid_dx = (
        cylinder_flow_interactor.forcing_grid.get_maximum_lagrangian_grid_spacing()
    )
    # Only master rank logs messages
    if cylinder_flow_interactor.mpi_construct.rank == master_rank:
        if max_lag_grid_dx > 2 * dx:
            warning_message = (
                f"Eulerian grid spacing (dx): {dx}"
                f"\nMax Lagrangian grid spacing: {max_lag_grid_dx} > 2 * dx"
                "\nThe Lagrangian grid of the body is too coarse relative to"
                "\nthe Eulerian grid of the flow, which can lead to unexpected"
                "\nconvergence. Please make the Lagrangian grid finer."
            )
        elif max_lag_grid_dx < 0.5 * dx:
            warning_message = (
                "==========================================================\n"
                f"Eulerian grid spacing (dx): {dx}"
                f"\nMax Lagrangian grid spacing: {max_lag_grid_dx} < 0.5 * dx"
                "\nThe Lagrangian grid of the body is too fine relative to"
                "\nthe Eulerian grid of the flow, which corresponds to redundant"
                "\nforcing points. Please make the Lagrangian grid coarser."
            )
        else:
            warning_message = (
                "Lagrangian grid is resolved almost the same as the Eulerian"
                "\ngrid of the flow."
            )
        # This is only written on the root process (i.e. echo_rank is 0 by default)
        if MPI.COMM_WORLD.Get_rank() == 0:
            assert warning_message in caplog.text


@pytest.mark.mpi(group="MPI_immersed_body_interaction", min_size=4)
@pytest.mark.parametrize("master_rank", [0, 1])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_mpi_immersed_body_interactor_call_method(master_rank, precision):
    (
        cylinder_flow_interactor,
        local_eul_grid_forcing_field,
        local_eul_grid_velocity_field,
        _,
    ) = mock_2d_cylinder_flow_interactor(master_rank=master_rank, precision=precision)
    cylinder_flow_interactor()

    ref_local_eul_grid_forcing_field = np.zeros_like(local_eul_grid_forcing_field)
    forcing_grid = cylinder_flow_interactor.forcing_grid
    # Perform the actual procedure below
    # 1. Compute forcing grid position and velocity
    forcing_grid.compute_lag_grid_position_field()
    forcing_grid.compute_lag_grid_velocity_field()
    # 2. Ghost the velocity field
    cylinder_flow_interactor.mpi_ghost_exchange_communicator.exchange_init(
        local_eul_grid_velocity_field[0], cylinder_flow_interactor.mpi_construct
    )
    cylinder_flow_interactor.mpi_ghost_exchange_communicator.exchange_init(
        local_eul_grid_velocity_field[1], cylinder_flow_interactor.mpi_construct
    )
    cylinder_flow_interactor.mpi_ghost_exchange_communicator.exchange_finalise()
    # 3. Compute interaction forcing
    cylinder_flow_interactor.compute_interaction_forcing(
        local_eul_grid_forcing_field=ref_local_eul_grid_forcing_field,
        local_eul_grid_velocity_field=local_eul_grid_velocity_field,
        global_lag_grid_position_field=forcing_grid.position_field,
        global_lag_grid_velocity_field=forcing_grid.velocity_field,
    )

    np.testing.assert_allclose(
        ref_local_eul_grid_forcing_field, local_eul_grid_forcing_field
    )


@pytest.mark.mpi(group="MPI_immersed_body_interaction", min_size=4)
@pytest.mark.parametrize("master_rank", [0, 1])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_mpi_immersed_body_interactor_compute_flow_forces_and_torques(
    master_rank, precision
):
    (
        cylinder_flow_interactor,
        _,
        local_eul_grid_velocity_field,
        _,
    ) = mock_2d_cylinder_flow_interactor(master_rank=master_rank)
    cylinder_flow_interactor.compute_flow_forces_and_torques()

    ref_body_flow_forces = np.zeros_like(cylinder_flow_interactor.body_flow_forces)
    ref_body_flow_torques = np.zeros_like(cylinder_flow_interactor.body_flow_torques)
    forcing_grid = cylinder_flow_interactor.forcing_grid
    # Perform the actual procedure below
    # 1. Compute forcing grid position and velocity
    forcing_grid.compute_lag_grid_position_field()
    forcing_grid.compute_lag_grid_velocity_field()
    # 2. Ghost the velocity field
    cylinder_flow_interactor.mpi_ghost_exchange_communicator.exchange_init(
        local_eul_grid_velocity_field[0], cylinder_flow_interactor.mpi_construct
    )
    cylinder_flow_interactor.mpi_ghost_exchange_communicator.exchange_init(
        local_eul_grid_velocity_field[1], cylinder_flow_interactor.mpi_construct
    )
    cylinder_flow_interactor.mpi_ghost_exchange_communicator.exchange_finalise()
    # 3. Compute interaction forcing on lag grid
    cylinder_flow_interactor.compute_interaction_force_on_lag_grid(
        local_eul_grid_velocity_field=local_eul_grid_velocity_field,
        global_lag_grid_position_field=forcing_grid.position_field,
        global_lag_grid_velocity_field=forcing_grid.velocity_field,
    )
    # 4. Transfer to body
    forcing_grid.transfer_forcing_from_grid_to_body(
        body_flow_forces=ref_body_flow_forces,
        body_flow_torques=ref_body_flow_torques,
        lag_grid_forcing_field=cylinder_flow_interactor.global_lag_grid_forcing_field,
    )
    np.testing.assert_allclose(
        ref_body_flow_forces, cylinder_flow_interactor.body_flow_forces
    )
    np.testing.assert_allclose(
        ref_body_flow_torques, cylinder_flow_interactor.body_flow_torques
    )


@pytest.mark.mpi(group="MPI_immersed_body_interaction", min_size=4)
@pytest.mark.parametrize("master_rank", [0, 1])
def test_mpi_immersed_body_interactor_get_grid_deviation_error_l2_norm(master_rank):
    cylinder_flow_interactor, _, _, _ = mock_2d_cylinder_flow_interactor()
    fixed_val = 2.0
    cylinder_flow_interactor.local_lag_grid_position_mismatch_field[...] = fixed_val
    grid_dev_error_l2_norm = cylinder_flow_interactor.get_grid_deviation_error_l2_norm()
    grid_dim = cylinder_flow_interactor.grid_dim
    ref_grid_dev_error_l2_norm = fixed_val * np.sqrt(grid_dim)
    np.testing.assert_allclose(grid_dev_error_l2_norm, ref_grid_dev_error_l2_norm)
