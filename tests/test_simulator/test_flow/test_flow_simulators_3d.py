import numpy as np
import pytest
from sopht_mpi.utils import (
    MPIConstruct3D,
    MPIGhostCommunicator3D,
)
import sopht_mpi.simulator as sps
from sopht_mpi.numeric.eulerian_grid_ops import (
    gen_diffusion_timestep_euler_forward_pyst_mpi_kernel_3d,
    gen_penalise_field_boundary_pyst_mpi_kernel_3d,
    gen_curl_pyst_mpi_kernel_3d,
    gen_update_vorticity_from_velocity_forcing_pyst_mpi_kernel_3d,
    gen_laplacian_filter_mpi_kernel_3d,
    UnboundedPoissonSolverMPI3D,
)
from sopht.numeric.eulerian_grid_ops import gen_elementwise_cross_product_pyst_kernel_3d
from sopht.utils.precision import get_real_t, get_test_tol
from sopht.utils.field import VectorField


@pytest.mark.mpi(group="MPI_flow_simulator_3d", min_size=4)
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("with_free_stream", [True, False])
@pytest.mark.parametrize("filter_vorticity", [True, False])
def test_mpi_flow_sim_3d_navier_stokes_with_forcing_timestep(
    precision, with_free_stream, filter_vorticity
):
    # Since we are testing the timestepper here, we can just preset some of the MPI
    # configuration, given that the kernels are extensively tested for in an MPI sense
    ghost_size = 2
    rank_distribution = (1, 1, 0)  # automatic decomposition
    aspect_ratio = (1, 1, 1)
    n_values = 8
    grid_size_z, grid_size_y, grid_size_x = (n_values * np.array(aspect_ratio)).astype(
        int
    )
    real_t = get_real_t(precision)

    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct3D(
        grid_size_z=grid_size_z,
        grid_size_y=grid_size_y,
        grid_size_x=grid_size_x,
        real_t=real_t,
        rank_distribution=rank_distribution,
    )
    # extra width needed for kernel computation
    mpi_ghost_exchange_communicator = MPIGhostCommunicator3D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )

    # Initialize flow variables
    grid_dim = mpi_construct.grid_dim
    x_range = 1.0
    nu = 1.0
    grid_size = (grid_size_z, grid_size_y, grid_size_x)
    dx = real_t(x_range / grid_size_x)
    free_stream_velocity = np.array([3.0, 4.0, 5.0])
    init_time = 1.0
    dt = 2.0

    # Initialize reference fields
    ref_vorticity_field = np.random.rand(
        mpi_construct.grid_dim,
        mpi_construct.local_grid_size[0] + 2 * ghost_size,
        mpi_construct.local_grid_size[1] + 2 * ghost_size,
        mpi_construct.local_grid_size[2] + 2 * ghost_size,
    ).astype(real_t)
    ref_velocity_field = np.random.rand(
        mpi_construct.grid_dim,
        mpi_construct.local_grid_size[0] + 2 * ghost_size,
        mpi_construct.local_grid_size[1] + 2 * ghost_size,
        mpi_construct.local_grid_size[2] + 2 * ghost_size,
    ).astype(real_t)
    ref_eul_grid_forcing_field = np.random.rand(
        mpi_construct.grid_dim,
        mpi_construct.local_grid_size[0] + 2 * ghost_size,
        mpi_construct.local_grid_size[1] + 2 * ghost_size,
        mpi_construct.local_grid_size[2] + 2 * ghost_size,
    ).astype(real_t)

    # Initialize and time step the flow simulator for testing
    flow_sim = sps.UnboundedFlowSimulator3D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        flow_type="navier_stokes_with_forcing",
        with_free_stream_flow=with_free_stream,
        filter_vorticity=filter_vorticity,
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
    diffusion_timestep = gen_diffusion_timestep_euler_forward_pyst_mpi_kernel_3d(
        mpi_construct=mpi_construct,
        ghost_exchange_communicator=mpi_ghost_exchange_communicator,
        real_t=real_t,
        field_type="vector",
    )
    unbounded_poisson_solver = UnboundedPoissonSolverMPI3D(
        grid_size_z=grid_size_z,
        grid_size_y=grid_size_y,
        grid_size_x=grid_size_x,
        x_range=x_range,
        real_t=real_t,
        mpi_construct=mpi_construct,
        ghost_size=ghost_size,
    )
    curl = gen_curl_pyst_mpi_kernel_3d(
        mpi_construct=mpi_construct,
        ghost_exchange_communicator=mpi_ghost_exchange_communicator,
        real_t=real_t,
    )
    penalise_field_towards_boundary = gen_penalise_field_boundary_pyst_mpi_kernel_3d(
        mpi_construct=mpi_construct,
        ghost_exchange_communicator=mpi_ghost_exchange_communicator,
        width=flow_sim.penalty_zone_width,
        dx=dx,
        x_grid_field=flow_sim.position_field[VectorField.x_axis_idx()],
        y_grid_field=flow_sim.position_field[VectorField.y_axis_idx()],
        z_grid_field=flow_sim.position_field[VectorField.z_axis_idx()],
        real_t=real_t,
        field_type="vector",
    )
    update_vorticity_from_velocity_forcing = (
        gen_update_vorticity_from_velocity_forcing_pyst_mpi_kernel_3d(
            mpi_construct=mpi_construct,
            ghost_exchange_communicator=mpi_ghost_exchange_communicator,
            real_t=real_t,
        )
    )
    elementwise_cross_product = gen_elementwise_cross_product_pyst_kernel_3d(
        real_t=real_t
    )

    # manually timestep
    update_vorticity_from_velocity_forcing(
        vorticity_field=ref_vorticity_field,
        velocity_forcing_field=ref_eul_grid_forcing_field,
        prefactor=real_t(dt / (2 * dx)),
    )
    velocity_cross_vorticity = np.zeros_like(ref_vorticity_field)
    elementwise_cross_product(
        result_field=velocity_cross_vorticity,
        field_1=ref_velocity_field,
        field_2=ref_vorticity_field,
    )
    update_vorticity_from_velocity_forcing(
        vorticity_field=ref_vorticity_field,
        velocity_forcing_field=velocity_cross_vorticity,
        prefactor=real_t(dt / (2 * dx)),
    )
    flux_buffer = np.zeros_like(ref_vorticity_field[0])
    diffusion_timestep(
        vector_field=ref_vorticity_field,
        diffusion_flux=flux_buffer,
        nu_dt_by_dx2=real_t(nu * dt / dx / dx),
    )
    if filter_vorticity:
        # use default filter vorticity settings, the filter kernels are tested elsewhere
        # set default values for the filter setting dictionary
        filter_setting_dict = {"order": 2, "type": "multiplicative"}
        filter_flux_buffer = np.zeros_like(ref_vorticity_field[0])
        field_buffer = np.zeros_like(ref_vorticity_field[1])
        filter_vector_field = gen_laplacian_filter_mpi_kernel_3d(
            mpi_construct=mpi_construct,
            ghost_exchange_communicator=mpi_ghost_exchange_communicator,
            filter_order=filter_setting_dict["order"],
            filter_flux_buffer=filter_flux_buffer,
            field_buffer=field_buffer,
            real_t=real_t,
            field_type="vector",
            filter_type=filter_setting_dict["type"],
        )
        filter_vector_field(vector_field=ref_vorticity_field)
    penalise_field_towards_boundary(vector_field=ref_vorticity_field)
    stream_func_field = np.zeros_like(ref_vorticity_field)
    unbounded_poisson_solver.vector_field_solve(
        solution_vector_field=stream_func_field,
        rhs_vector_field=ref_vorticity_field,
    )
    curl(
        curl=ref_velocity_field,
        field=stream_func_field,
        prefactor=real_t(0.5 / dx),
    )
    if with_free_stream:
        ref_velocity_field[...] += free_stream_velocity.reshape(grid_dim, 1, 1, 1)

    assert flow_sim.time == ref_time
    np.testing.assert_allclose(flow_sim.eul_grid_forcing_field, 0.0)
    # Since we reuse buffers in the simulator, and here we allocate separate buffers,
    # we test only the inner cells, which contains the correct flow quantities.
    inner_idx = (slice(None),) + (slice(ghost_size, -ghost_size),) * grid_dim
    np.testing.assert_allclose(
        flow_sim.vorticity_field[inner_idx], ref_vorticity_field[inner_idx]
    )
    np.testing.assert_allclose(
        flow_sim.velocity_field[inner_idx], ref_velocity_field[inner_idx]
    )


@pytest.mark.mpi(group="MPI_flow_simulator_3d", min_size=4)
@pytest.mark.parametrize("precision", ["single", "double"])
def test_mpi_flow_sim_3d_compute_stable_timestep(precision):
    grid_size_x = 8
    grid_dim = 3
    x_range = 1.0
    nu = 1.0
    real_t = get_real_t(precision)
    grid_size = (grid_size_x,) * grid_dim
    dx = real_t(x_range / grid_size_x)
    cfl = 0.2
    flow_sim = sps.UnboundedFlowSimulator3D(
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
    diffusion_limit_dt = 0.9 * dx**2 / 6 / (nu + get_test_tol(precision))
    ref_dt = dt_prefac * min(advection_limit_dt, diffusion_limit_dt)
    assert ref_dt == sim_dt
