import click
import elastica as ea
import numpy as np
import sopht_mpi.simulator as sps
import sopht.utils as spu
from sopht_mpi.utils import logger
from sopht_mpi.utils.mpi_io import MPIIO, CosseratRodMPIIO
from mpi4py import MPI
from typing import Optional
from sopht.simulator.immersed_body import CosseratRodSurfaceForcingGrid, FlowForces
import os


def flow_past_rod_case(
    non_dim_final_time: float,
    n_elem: int,
    grid_size: tuple[int, int, int],
    surface_grid_density_for_largest_element: int,
    cauchy_number: float,
    mass_ratio: float,
    base_length: float = 1.0,
    aspect_ratio: float = 20,
    poisson_ratio: float = 0.5,
    reynolds: float = 100.0,
    coupling_stiffness: float = -2e4,
    coupling_damping: float = -1e1,
    rod_start_incline_angle: float = 0.0,
    rank_distribution: Optional[tuple[int, int, int]] = None,
    precision: str = "single",
    save_data: bool = False,
    restart_simulation: bool = False,
) -> None:
    # =================COMMON SIMULATOR STUFF=======================
    grid_size_z, grid_size_y, grid_size_x = grid_size
    real_t = spu.get_real_t(precision)
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    z_axis_idx = spu.VectorField.z_axis_idx()

    rho_f = 1.0
    U_free_stream = 1.0
    x_range = 5.0 * base_length
    y_range = grid_size_y / grid_size_x * x_range
    z_range = grid_size_z / grid_size_x * x_range
    velocity_free_stream = [U_free_stream, 0.0, 0.0]
    # =================PYELASTICA STUFF BEGIN=====================

    class FlowPastRodSimulator(
        ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping
    ):
        ...

    flow_past_sim = FlowPastRodSimulator()
    start = np.array([0.08 * x_range, 0.502 * y_range, 0.502 * z_range])
    direction = np.array(
        [np.sin(rod_start_incline_angle), 0.0, -np.cos(rod_start_incline_angle)]
    )
    normal = np.array([0.0, 1.0, 0.0])
    base_diameter = base_length / aspect_ratio
    base_radius = base_diameter / 2.0
    # mass_ratio = rho_s / rho_f
    rho_s = mass_ratio * rho_f
    moment_of_inertia = np.pi / 4 * base_radius**4
    # Cau = (rho_f U^2 L^3 D) / EI
    youngs_modulus = (rho_f * U_free_stream**2 * base_length**3 * base_diameter) / (
        cauchy_number * moment_of_inertia
    )

    flow_past_rod = ea.CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        rho_s,
        0.0,  # internal damping constant, deprecated in v0.3.0
        youngs_modulus,
        shear_modulus=youngs_modulus / (poisson_ratio + 1.0),
    )
    flow_past_sim.append(flow_past_rod)

    # Constraint fixed end to allow for axial rotation
    flow_past_sim.constrain(flow_past_rod).using(
        ea.GeneralConstraint,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
        translational_constraint_selector=np.array([True, True, True]),
        rotational_constraint_selector=np.array([False, True, True]),
    )

    # add damping
    dl = base_length / n_elem
    rod_dt = 0.01 * dl
    damping_constant = 1e-3
    flow_past_sim.dampen(flow_past_rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=rod_dt,
    )
    # =================PYELASTICA STUFF END=====================

    # ==================FLOW SETUP START=========================
    kinematic_viscosity = U_free_stream * base_diameter / reynolds
    flow_sim = sps.UnboundedFlowSimulator3D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=kinematic_viscosity,
        flow_type="navier_stokes_with_forcing",
        with_free_stream_flow=True,
        real_t=real_t,
        rank_distribution=rank_distribution,
        filter_vorticity=True,
        filter_setting_dict={"order": 5, "type": "convolution"},
    )
    flow_sim.velocity_field += np.array(velocity_free_stream).reshape(3, 1, 1, 1)
    # ==================FLOW SETUP END=========================

    # ==================FLOW-ROD COMMUNICATOR SETUP START======
    master_rank = 0
    cosserat_rod_flow_interactor = sps.CosseratRodFlowInteraction(
        mpi_construct=flow_sim.mpi_construct,
        mpi_ghost_exchange_communicator=flow_sim.mpi_ghost_exchange_communicator,
        cosserat_rod=flow_past_rod,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=coupling_stiffness,
        virtual_boundary_damping_coeff=coupling_damping,
        dx=flow_sim.dx,
        grid_dim=flow_sim.grid_dim,
        forcing_grid_cls=CosseratRodSurfaceForcingGrid,
        surface_grid_density_for_largest_element=surface_grid_density_for_largest_element,
        master_rank=master_rank,
        with_cap=True,
    )
    flow_past_sim.add_forcing_to(flow_past_rod).using(
        FlowForces,
        cosserat_rod_flow_interactor,
    )
    # ==================FLOW-ROD COMMUNICATOR SETUP END======
    # =================TIMESTEPPING====================
    flow_past_sim.finalize()
    timestepper = ea.PositionVerlet()
    do_step, stages_and_updates = ea.extend_stepper_interface(
        timestepper, flow_past_sim
    )

    if save_data:
        restart_dir = "restart_data"
        os.makedirs(restart_dir, exist_ok=True)

        # setup IO
        # TODO: internalise this into flow simulator in a cleaner way
        origin_x = flow_sim.mpi_construct.grid.allreduce(
            flow_sim.position_field[
                x_axis_idx,
                flow_sim.ghost_size : -flow_sim.ghost_size,
                flow_sim.ghost_size : -flow_sim.ghost_size,
            ].min(),
            op=MPI.MIN,
        )
        origin_y = flow_sim.mpi_construct.grid.allreduce(
            flow_sim.position_field[
                y_axis_idx,
                flow_sim.ghost_size : -flow_sim.ghost_size,
                flow_sim.ghost_size : -flow_sim.ghost_size,
            ].min(),
            op=MPI.MIN,
        )
        origin_z = flow_sim.mpi_construct.grid.allreduce(
            flow_sim.position_field[
                z_axis_idx,
                flow_sim.ghost_size : -flow_sim.ghost_size,
                flow_sim.ghost_size : -flow_sim.ghost_size,
                flow_sim.ghost_size : -flow_sim.ghost_size,
            ].min(),
            op=MPI.MIN,
        )
        io_origin = np.array([origin_z, origin_y, origin_x])
        io_dx = flow_sim.dx * np.ones(flow_sim.grid_dim)
        io_grid_size = np.array(grid_size)
        io = MPIIO(mpi_construct=flow_sim.mpi_construct, real_dtype=real_t)
        io.define_eulerian_grid(
            origin=io_origin,
            dx=io_dx,
            grid_size=io_grid_size,
            ghost_size=flow_sim.ghost_size,
        )
        io.add_as_eulerian_fields_for_io(
            vorticity=flow_sim.vorticity_field, velocity=flow_sim.velocity_field
        )
        # Initialize rod IO
        rod_io = CosseratRodMPIIO(
            mpi_construct=flow_sim.mpi_construct,
            cosserat_rod=flow_past_rod,
            master_rank=cosserat_rod_flow_interactor.master_rank,
        )
        # Initialize rod mismatch field IO
        forcing_grid_io = MPIIO(
            mpi_construct=flow_sim.mpi_construct,
            real_dtype=cosserat_rod_flow_interactor.forcing_grid.position_field.dtype,
        )
        # Add vector field on lagrangian grid
        forcing_grid_io.add_as_lagrangian_fields_for_io(
            lagrangian_grid=cosserat_rod_flow_interactor.forcing_grid.position_field,
            lagrangian_grid_master_rank=cosserat_rod_flow_interactor.master_rank,
            lagrangian_grid_name="mismatch_field",
            position_mismatch=cosserat_rod_flow_interactor.global_lag_grid_position_mismatch_field,
            velocity_mismatch=cosserat_rod_flow_interactor.global_lag_grid_position_mismatch_field,
        )

        if restart_simulation:
            # find latest saved data
            iter_num = []
            for filename in os.listdir():
                if "flow" in filename and "h5" in filename:
                    iter_num.append(int(filename.split("_")[-1].split(".")[0]))
            latest = max(iter_num)
            # load sopht data
            flow_sim.time = io.load(h5_file_name=f"flow_{latest:05d}.h5")
            _ = rod_io.load(h5_file_name=f"rod_{latest:05d}.h5")
            _ = forcing_grid_io.load(
                h5_file_name=f"{restart_dir}/forcing_grid_{latest:05d}.h5"
            )
            if flow_sim.mpi_construct.rank == cosserat_rod_flow_interactor.master_rank:
                # load elastica data
                elastica_restart_time = ea.load_state(flow_past_sim, restart_dir, True)
                assert flow_sim.time == elastica_restart_time
            # Re-map the lagrangian nodes to correctly load rank-local mismatch fields
            cosserat_rod_flow_interactor.mpi_lagrangian_field_communicator.map_lagrangian_nodes_based_on_position(
                cosserat_rod_flow_interactor.forcing_grid.position_field
            )
            # Initialize buffer with new local size
            cosserat_rod_flow_interactor.local_num_lag_nodes = (
                cosserat_rod_flow_interactor.mpi_lagrangian_field_communicator.local_num_lag_nodes
            )
            cosserat_rod_flow_interactor._init_local_buffers(
                cosserat_rod_flow_interactor.local_num_lag_nodes
            )
            # Scatter the global mismatch fields correspondingly
            cosserat_rod_flow_interactor.mpi_lagrangian_field_communicator.scatter_global_field(
                local_lag_field=cosserat_rod_flow_interactor.local_lag_grid_position_mismatch_field,
                global_lag_field=cosserat_rod_flow_interactor.global_lag_grid_position_mismatch_field,
            )
            cosserat_rod_flow_interactor.mpi_lagrangian_field_communicator.scatter_global_field(
                local_lag_field=cosserat_rod_flow_interactor.local_lag_grid_velocity_mismatch_field,
                global_lag_field=cosserat_rod_flow_interactor.global_lag_grid_velocity_mismatch_field,
            )

    foto_timer = 0.0
    timescale = base_length / U_free_stream
    final_time = non_dim_final_time * timescale
    foto_timer_limit = timescale / 5

    # iterate
    while flow_sim.time < final_time:
        # Save data
        if foto_timer > foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0

            # Log some diagnostics
            logger.info(
                f"time: {flow_sim.time:.2f} ({(flow_sim.time/final_time*100):2.1f}%), "
                f"max_vort: {flow_sim.get_max_vorticity():.4f}, "
                f"vort divg. L2 norm: {flow_sim.get_vorticity_divergence_l2_norm():.4f},"
                " grid deviation L2 error: "
                f"{cosserat_rod_flow_interactor.get_grid_deviation_error_l2_norm():.6f} "
            )
            if save_data:
                # Save flow
                io.save(
                    h5_file_name="flow_" + str("%0.5d" % (flow_sim.time * 100)) + ".h5",
                    time=flow_sim.time,
                )
                # Save rod
                rod_io.save(
                    h5_file_name="rod_" + str("%0.5d" % (flow_sim.time * 100)) + ".h5",
                    time=flow_sim.time,
                )
                # Save interactor related stuff
                forcing_grid_io.save(
                    h5_file_name=f"{restart_dir}/forcing_grid_"
                    + str("%0.5d" % (flow_sim.time * 100))
                    + ".h5",
                    time=flow_sim.time,
                )
                # Save elastica simulator
                if (
                    flow_sim.mpi_construct.rank
                    == cosserat_rod_flow_interactor.master_rank
                ):
                    ea.save_state(flow_past_sim, restart_dir, flow_sim.time)

        # compute timestep
        flow_dt = flow_sim.compute_stable_timestep(dt_prefac=0.5)

        # timestep the rod, through the flow timestep
        rod_time_steps = int(flow_dt / min(flow_dt, rod_dt))
        local_rod_dt = flow_dt / rod_time_steps
        rod_time = flow_sim.time
        for i in range(rod_time_steps):
            rod_time = do_step(
                timestepper, stages_and_updates, flow_past_sim, rod_time, local_rod_dt
            )
            # timestep the cosserat_rod_flow_interactor
            cosserat_rod_flow_interactor.time_step(dt=local_rod_dt)
        # evaluate feedback/interaction between flow and rod
        cosserat_rod_flow_interactor()

        flow_sim.time_step(
            dt=flow_dt,
            free_stream_velocity=velocity_free_stream,
        )

        # update timer
        foto_timer += flow_dt

    if save_data and flow_sim.mpi_construct.rank == master_rank:
        spu.make_dir_and_transfer_h5_data(dir_name="flow_data")


if __name__ == "__main__":

    @click.command()
    @click.option("--nx", default=128, help="Number of grid points in x direction.")
    @click.option("--cauchy", default=1e2, help="Cauchy number.")
    @click.option("--reynolds", default=1e2, help="Reynolds number.")
    @click.option("--final_time", default=10.0, help="Final simulation time.")
    @click.option(
        "--restart", default=False, help="Specify if this is a restart simulation"
    )
    def simulate_flow_past_rod(
        nx: int, cauchy: float, reynolds: float, final_time: float, restart: bool
    ) -> None:
        ny = nx // 4
        nz = nx // 4
        # in order Z, Y, X
        grid_size = (nz, ny, nx)
        surface_grid_density_for_largest_element = nx // 25
        n_elem = nx // 8

        logger.info(f"Grid size:  {nz, ny, nx ,} ")
        logger.info(
            f"num forcing points around the surface:  {surface_grid_density_for_largest_element}"
        )
        logger.info(f"num rod elements: {n_elem}")
        logger.info(f"Cauchy number: {cauchy}")
        logger.info(f"Reynolds number: {reynolds}")

        mass_ratio = 1000
        rod_start_incline_angle = np.deg2rad(90)

        flow_past_rod_case(
            non_dim_final_time=final_time,
            cauchy_number=cauchy,
            mass_ratio=mass_ratio,
            reynolds=reynolds,
            grid_size=grid_size,
            surface_grid_density_for_largest_element=surface_grid_density_for_largest_element,
            n_elem=n_elem,
            rod_start_incline_angle=rod_start_incline_angle,
            save_data=True,
            restart_simulation=restart,
        )

    simulate_flow_past_rod()
