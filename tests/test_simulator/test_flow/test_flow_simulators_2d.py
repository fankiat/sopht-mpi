import numpy as np
import pytest
from sopht_mpi.utils import (
    MPIConstruct2D,
    MPIGhostCommunicator2D,
)
import sopht_mpi.simulator as sps
from sopht_mpi.numeric.eulerian_grid_ops import (
    gen_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_2d,
    gen_diffusion_timestep_euler_forward_pyst_mpi_kernel_2d,
    gen_penalise_field_boundary_pyst_mpi_kernel_2d,
    gen_outplane_field_curl_pyst_mpi_kernel_2d,
    gen_update_vorticity_from_velocity_forcing_pyst_mpi_kernel_2d,
    UnboundedPoissonSolverMPI2D,
)
from sopht.utils.precision import get_real_t, get_test_tol
from sopht.utils.field import VectorField


@pytest.mark.mpi(group="MPI_flow_simulator_2d", min_size=4)
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("with_free_stream", [True, False])
def test_mpi_flow_sim_2d_navier_stokes_with_forcing_timestep(
    precision, with_free_stream
):
    # Since we are testing the timestepper here, we can just preset some of the MPI
    # configuration, given that the kernels are extensively tested for in an MPI sense
    ghost_size = 2
    rank_distribution = None  # automatic decomposition
    aspect_ratio = (1, 1.5)
    n_values = 8
    grid_size_y, grid_size_x = (n_values * np.array(aspect_ratio)).astype(int)
    real_t = get_real_t(precision)

    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=grid_size_y,
        grid_size_x=grid_size_x,
        real_t=real_t,
        rank_distribution=rank_distribution,
    )
    # extra width needed for kernel computation
    mpi_ghost_exchange_communicator = MPIGhostCommunicator2D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )

    # Initialize flow variables
    grid_dim = mpi_construct.grid_dim
    x_range = 1.0
    nu = 1.0
    grid_size = (grid_size_y, grid_size_x)
    dx = real_t(x_range / grid_size_x)
    free_stream_velocity = np.array([3.0, 4.0])
    init_time = 1.0
    dt = 2.0

    # Initialize reference fields
    ref_vorticity_field = np.random.rand(
        mpi_construct.local_grid_size[0] + 2 * ghost_size,
        mpi_construct.local_grid_size[1] + 2 * ghost_size,
    ).astype(real_t)
    ref_velocity_field = np.random.rand(
        mpi_construct.grid_dim,
        mpi_construct.local_grid_size[0] + 2 * ghost_size,
        mpi_construct.local_grid_size[1] + 2 * ghost_size,
    ).astype(real_t)
    ref_eul_grid_forcing_field = np.random.rand(
        mpi_construct.grid_dim,
        mpi_construct.local_grid_size[0] + 2 * ghost_size,
        mpi_construct.local_grid_size[1] + 2 * ghost_size,
    ).astype(real_t)

    # Initialize and time step the flow simulator for testing
    flow_sim = sps.UnboundedFlowSimulator2D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        flow_type="navier_stokes_with_forcing",
        with_free_stream_flow=with_free_stream,
        real_t=real_t,
        time=init_time,
        rank_distribution=rank_distribution,
        ghost_size=ghost_size,
    )
    ref_time = init_time + dt
    # initialise flow sim state (vorticity and forcing)
    flow_sim.vorticity_field[...] = ref_vorticity_field.copy()
    flow_sim.velocity_field[...] = ref_velocity_field.copy()
    flow_sim.eul_grid_forcing_field[...] = ref_eul_grid_forcing_field.copy()
    # Timestep the flow simulator
    flow_sim.time_step(dt=dt, free_stream_velocity=free_stream_velocity)

    # Setup reference timestep manually
    # Compile necessary kernels
    diffusion_timestep = gen_diffusion_timestep_euler_forward_pyst_mpi_kernel_2d(
        real_t=real_t,
        mpi_construct=mpi_construct,
        ghost_exchange_communicator=mpi_ghost_exchange_communicator,
    )
    advection_timestep = (
        gen_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_2d(
            real_t=real_t,
            mpi_construct=mpi_construct,
            ghost_exchange_communicator=mpi_ghost_exchange_communicator,
        )
    )
    unbounded_poisson_solver = UnboundedPoissonSolverMPI2D(
        grid_size_y=grid_size_y,
        grid_size_x=grid_size_x,
        x_range=x_range,
        real_t=real_t,
        mpi_construct=mpi_construct,
        ghost_size=ghost_size,
    )
    curl = gen_outplane_field_curl_pyst_mpi_kernel_2d(
        real_t=real_t,
        mpi_construct=mpi_construct,
        ghost_exchange_communicator=mpi_ghost_exchange_communicator,
    )
    penalise_field_towards_boundary = gen_penalise_field_boundary_pyst_mpi_kernel_2d(
        width=flow_sim.penalty_zone_width,
        dx=dx,
        x_grid_field=flow_sim.position_field[VectorField.x_axis_idx()],
        y_grid_field=flow_sim.position_field[VectorField.y_axis_idx()],
        real_t=real_t,
        mpi_construct=mpi_construct,
        ghost_exchange_communicator=mpi_ghost_exchange_communicator,
    )
    update_vorticity_from_velocity_forcing = (
        gen_update_vorticity_from_velocity_forcing_pyst_mpi_kernel_2d(
            real_t=real_t,
            mpi_construct=mpi_construct,
            ghost_exchange_communicator=mpi_ghost_exchange_communicator,
        )
    )

    # manually timestep
    update_vorticity_from_velocity_forcing(
        vorticity_field=ref_vorticity_field,
        velocity_forcing_field=ref_eul_grid_forcing_field,
        prefactor=real_t(dt / (2 * dx)),
    )
    flux_buffer = np.zeros_like(ref_vorticity_field)
    advection_timestep(
        field=ref_vorticity_field,
        advection_flux=flux_buffer,
        velocity=ref_velocity_field,
        dt_by_dx=real_t(dt / dx),
    )
    diffusion_timestep(
        field=ref_vorticity_field,
        diffusion_flux=flux_buffer,
        nu_dt_by_dx2=real_t(nu * dt / dx / dx),
    )
    penalise_field_towards_boundary(field=ref_vorticity_field)
    stream_func_field = np.zeros_like(ref_vorticity_field)
    unbounded_poisson_solver.solve(
        solution_field=stream_func_field,
        rhs_field=ref_vorticity_field,
    )
    curl(
        curl=ref_velocity_field,
        field=stream_func_field,
        prefactor=real_t(0.5 / dx),
    )
    if with_free_stream:
        ref_velocity_field[...] += free_stream_velocity.reshape(grid_dim, 1, 1)

    assert flow_sim.time == ref_time
    np.testing.assert_allclose(flow_sim.eul_grid_forcing_field, 0.0)
    np.testing.assert_allclose(flow_sim.vorticity_field, ref_vorticity_field)
    np.testing.assert_allclose(flow_sim.velocity_field, ref_velocity_field)


@pytest.mark.mpi(group="MPI_flow_simulator_2d", min_size=4)
@pytest.mark.parametrize("precision", ["single", "double"])
def test_mpi_flow_sim_2d_compute_stable_timestep(precision):
    grid_size_x = 8
    grid_dim = 2
    x_range = 1.0
    nu = 1.0
    real_t = get_real_t(precision)
    grid_size = (grid_size_x,) * grid_dim
    dx = real_t(x_range / grid_size_x)
    cfl = 0.2
    flow_sim = sps.UnboundedFlowSimulator2D(
        grid_size=grid_size,
        x_range=x_range,
        CFL=cfl,
        kinematic_viscosity=nu,
        real_t=real_t,
    )
    flow_sim.velocity_field[...] = 2.0
    dt_prefac = 0.5
    sim_dt = flow_sim.compute_stable_timestep(dt_prefac=dt_prefac, precision=precision)
    # next compute reference value
    advection_limit_dt = (
        cfl * dx / (grid_dim * 2.0 + get_test_tol(precision))
    )  # max(sum(abs(velocity_field)))
    diffusion_limit_dt = 0.9 * dx**2 / 4 / nu
    ref_dt = dt_prefac * min(advection_limit_dt, diffusion_limit_dt)
    assert ref_dt == sim_dt


@pytest.mark.mpi(group="MPI_flow_simulator_2d", min_size=4)
@pytest.mark.parametrize("precision", ["single", "double"])
def test_mpi_flow_sim_2d_get_max_vorticity(precision):
    # Since we are testing the timestepper here, we can just preset some of the MPI
    # configuration, given that the kernels are extensively tested for in an MPI sense
    ghost_size = 2
    rank_distribution = None  # automatic decomposition
    aspect_ratio = (1, 1.5)
    n_values = 8
    grid_size_y, grid_size_x = (n_values * np.array(aspect_ratio)).astype(int)
    real_t = get_real_t(precision)

    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=grid_size_y,
        grid_size_x=grid_size_x,
        real_t=real_t,
        rank_distribution=rank_distribution,
    )

    # Initialize flow variables
    x_range = 1.0
    nu = 1e-2
    real_t = get_real_t(precision)
    grid_size = (grid_size_y, grid_size_x)
    flow_sim = sps.UnboundedFlowSimulator2D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        real_t=real_t,
        flow_type="navier_stokes_with_forcing",
        rank_distribution=rank_distribution,
        ghost_size=ghost_size,
    )

    # Initialize reference fields
    vorticity_field = np.random.rand(
        mpi_construct.local_grid_size[0] + 2 * ghost_size,
        mpi_construct.local_grid_size[1] + 2 * ghost_size,
    ).astype(real_t)
    ref_max_vort = 10.0  # a value more than random field above

    # set an artificial spike in one of the ranks
    if mpi_construct.rank == 0:
        vorticity_field[
            (mpi_construct.local_grid_size[0] + 2 * ghost_size) // 2,
            (mpi_construct.local_grid_size[1] + 2 * ghost_size) // 2,
        ] = ref_max_vort

    flow_sim.vorticity_field[...] = vorticity_field.copy()
    max_vort = flow_sim.get_max_vorticity()

    assert max_vort == ref_max_vort
