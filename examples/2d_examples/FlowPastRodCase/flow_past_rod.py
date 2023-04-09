import elastica as ea
import click
import numpy as np
import os
import sopht.utils as spu
import sopht_mpi.simulator as sps
from sopht_mpi.utils.mpi_utils_2d import MPIPlotter2D
from sopht_mpi.utils.mpi_io import MPIIO, CosseratRodMPIIO
from sopht_mpi.utils import logger
from mpi4py import MPI
from sopht.simulator.immersed_body import (
    CosseratRodElementCentricForcingGrid,
    FlowForces,
)


def flow_past_rod_case(
    nondim_final_time,
    grid_size,
    reynolds,
    nondim_bending_stiffness,
    nondim_mass_ratio,
    froude,
    rod_start_incline_angle=0.0,
    coupling_stiffness=-8e4,
    coupling_damping=-30,
    rank_distribution=None,
    precision="single",
    save_flow_data=False,
):
    # =================COMMON SIMULATOR STUFF=======================
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    grid_size_y, grid_size_x = grid_size

    velocity_free_stream = 1.0
    rho_f = 1.0
    base_length = 1.0
    x_range = 6.0 * base_length
    y_range = grid_size_y / grid_size_x * x_range
    # =================PYELASTICA STUFF BEGIN=====================
    class FlowPastRodSimulator(
        ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping
    ):
        pass

    flow_past_sim = FlowPastRodSimulator()
    # setting up test params
    n_elem = grid_size_x // 8
    start = np.array([base_length, 0.501 * y_range, 0.0])
    direction = np.array(
        [np.cos(rod_start_incline_angle), np.sin(rod_start_incline_angle), 0.0]
    )
    normal = np.array([0.0, 0.0, 1.0])
    base_radius = 0.01
    base_area = np.pi * base_radius**2
    z_axis_width = 1.0
    # nondim_mass_ratio = rod_line_density / (rho_f * base_length * z_axis_width)
    rod_line_density = nondim_mass_ratio * rho_f * base_length * z_axis_width
    density = rod_line_density / base_area
    moment_of_inertia = np.pi / 4 * base_radius**4
    # nondim_bending_stiffness = youngs_modulus * moment_of_inertia
    # / (rho_f vel_free_stream^2 base_length^3)
    youngs_modulus = (
        nondim_bending_stiffness
        * (rho_f * velocity_free_stream**2 * base_length**3 * z_axis_width)
        / moment_of_inertia
    )
    poisson_ratio = 0.5
    # Fr = gravitational_acc * base_length / velocity_free_stream ^ 2
    gravitational_acc = froude * velocity_free_stream**2 / base_length

    flow_past_rod = ea.CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        0.0,  # internal damping constant, deprecated in v0.3.0
        youngs_modulus,
        shear_modulus=youngs_modulus / (poisson_ratio + 1.0),
    )
    tip_start_position = flow_past_rod.position_collection[(x_axis_idx, y_axis_idx), -1]
    flow_past_sim.append(flow_past_rod)
    flow_past_sim.constrain(flow_past_rod).using(
        ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )
    # Add gravitational forces
    flow_past_sim.add_forcing_to(flow_past_rod).using(
        ea.GravityForces, acc_gravity=np.array([gravitational_acc, 0.0, 0.0])
    )
    # add damping
    dl = base_length / n_elem
    rod_dt = 0.01 * dl
    damping_constant = 0.5e-3
    flow_past_sim.dampen(flow_past_rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=rod_dt,
    )
    # =================PYELASTICA STUFF END=====================

    # ==================FLOW SETUP START=========================
    flow_solver_precision = precision
    real_t = spu.get_real_t(flow_solver_precision)
    # Flow parameters
    # Re = velocity_free_stream * base_length / nu
    nu = base_length * velocity_free_stream / reynolds
    flow_sim = sps.UnboundedFlowSimulator2D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        flow_type="navier_stokes_with_forcing",
        with_free_stream_flow=True,
        real_t=real_t,
        rank_distribution=rank_distribution,
        time=0.0,
    )
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
        forcing_grid_cls=CosseratRodElementCentricForcingGrid,
        master_rank=master_rank,
    )
    flow_past_sim.add_forcing_to(flow_past_rod).using(
        FlowForces,
        cosserat_rod_flow_interactor,
    )
    # ==================FLOW-ROD COMMUNICATOR SETUP END======

    if save_flow_data:
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
        io_origin = np.array([origin_y, origin_x])
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
            master_rank=master_rank,
        )

    # =================TIMESTEPPING====================

    flow_past_sim.finalize()
    timestepper = ea.PositionVerlet()
    do_step, stages_and_updates = ea.extend_stepper_interface(
        timestepper, flow_past_sim
    )
    foto_timer = 0.0
    timescale = base_length / velocity_free_stream
    final_time = nondim_final_time * timescale
    foto_timer_limit = final_time / 60

    # setup freestream ramping
    ramp_timescale = timescale
    velocity_free_stream_perturb = 0.5 * velocity_free_stream

    data_timer = 0.0
    data_timer_limit = 0.1 * timescale
    tip_time = []
    tip_position = []

    # Initialize field plotter
    mpi_plotter = MPIPlotter2D(
        flow_sim.mpi_construct,
        flow_sim.ghost_size,
    )

    while flow_sim.time < final_time:

        # Plot solution
        if foto_timer >= foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            mpi_plotter.ax.set_title(
                f"Vorticity, time: {flow_sim.time / timescale:.2f}"
            )
            mpi_plotter.contourf(
                flow_sim.position_field[x_axis_idx],
                flow_sim.position_field[y_axis_idx],
                flow_sim.vorticity_field,
                levels=np.linspace(-5, 5, 100),
                extend="both",
            )
            mpi_plotter.plot(
                cosserat_rod_flow_interactor.forcing_grid.position_field[x_axis_idx],
                cosserat_rod_flow_interactor.forcing_grid.position_field[y_axis_idx],
                linewidth=3,
                color="k",
            )
            mpi_plotter.savefig(
                file_name="snap_" + str("%0.4d" % (flow_sim.time * 100)) + ".png"
            )
            mpi_plotter.clearfig()

            # Compute some diagnostics to log
            grid_dev_error_l2_norm = (
                cosserat_rod_flow_interactor.get_grid_deviation_error_l2_norm()
            )
            max_vort = flow_sim.get_max_vorticity()
            logger.info(
                f"time: {flow_sim.time:.2f} ({(flow_sim.time / final_time*100):2.1f}%), "
                f"max_vort: {max_vort:.4f}, "
                f"grid deviation L2 error: {grid_dev_error_l2_norm:.8f}"
            )

            if save_flow_data:
                io.save(
                    h5_file_name="sopht_"
                    + str("%0.4d" % (flow_sim.time * 100))
                    + ".h5",
                    time=flow_sim.time,
                )
                rod_io.save(
                    h5_file_name="rod_" + str("%0.4d" % (flow_sim.time * 100)) + ".h5",
                    time=flow_sim.time,
                )

        # save diagnostic data
        if data_timer >= data_timer_limit or data_timer == 0:
            data_timer = 0.0
            if flow_sim.mpi_construct.rank == master_rank:
                tip_time.append(flow_sim.time / timescale)
                tip_position.append(
                    (
                        flow_past_rod.position_collection[(x_axis_idx, y_axis_idx), -1]
                        - tip_start_position
                    )
                    / base_length
                )

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

        ramp_factor = np.exp(-flow_sim.time / ramp_timescale)
        # timestep the flow
        flow_sim.time_step(
            dt=flow_dt,
            free_stream_velocity=[
                velocity_free_stream * (1.0 - ramp_factor),
                velocity_free_stream_perturb * ramp_factor,
            ],
        )

        # update timers
        foto_timer += flow_dt
        data_timer += flow_dt

    # compile video and other diagnostics plots
    if flow_sim.mpi_construct.rank == master_rank:
        os.system("rm -f flow.mp4")
        os.system(
            "ffmpeg -r 10 -s 3840x2160 -f image2 -pattern_type glob -i 'snap*.png' "
            "-vcodec libx264 -crf 15 -pix_fmt yuv420p -vf 'crop=trunc(iw/2)*2:trunc(ih/2)*2'"
            " flow.mp4"
        )
        os.system("rm -f snap*.png")

        tip_time = np.array(tip_time)
        tip_position = np.array(tip_position)

        np.savetxt(
            fname="rod_diagnostics_vs_time.csv",
            X=np.c_[
                tip_time,
                tip_position[..., x_axis_idx],
                tip_position[..., y_axis_idx],
            ],
            header="time, tip_x, tip_y",
            delimiter=",",
        )

        # Plot tip position
        mpi_plotter.ax.set_aspect(aspect="auto")
        mpi_plotter.ax.set_title("Rod tip deflection")
        mpi_plotter.plot(tip_time, tip_position[..., x_axis_idx], label="X")
        mpi_plotter.plot(tip_time, tip_position[..., y_axis_idx], label="Y")
        mpi_plotter.ax.legend()
        mpi_plotter.ax.set_xlabel("Non-dimensional time")
        mpi_plotter.ax.set_ylabel("Tip deflection")
        mpi_plotter.savefig("tip_position_vs_time.png")
        mpi_plotter.clearfig()


if __name__ == "__main__":
    # classical benchmark params
    bmk_reynolds = 200.0
    bmk_nondim_bending_stiffness = 1.5e-3
    bmk_nondim_mass_ratio = 1.5
    bmk_froude = 0.5

    @click.command()
    @click.option(
        "--sim_grid_size_x", default=256, help="Number of grid points in x direction."
    )
    @click.option(
        "--nondim_final_time",
        default=75.0,
        help="Non-dimensional final simulation time.",
    )
    @click.option("--reynolds", default=bmk_reynolds, help="Reynolds number.")
    @click.option(
        "--nondim_bending_stiffness",
        default=bmk_nondim_bending_stiffness,
        help="Non-dimensional bending stiffness",
    )
    @click.option(
        "--nondim_mass_ratio",
        default=bmk_nondim_mass_ratio,
        help="Non-dimensional mass ratio.",
    )
    @click.option(
        "--froude",
        default=bmk_froude,
        help="Froude number.",
    )
    def simulate_custom_flow_past_rod_case(
        sim_grid_size_x,
        nondim_final_time,
        reynolds,
        nondim_bending_stiffness,
        nondim_mass_ratio,
        froude,
    ):
        sim_grid_size_y = sim_grid_size_x // 2
        sim_grid_size = (sim_grid_size_y, sim_grid_size_x)
        logger.info(f"Grid size: {sim_grid_size}")
        logger.info(f"Reynolds number: {reynolds}")
        logger.info(f"Non-dimensional bending stiffness: {nondim_bending_stiffness}")
        logger.info(f"Non-dimensional mass ratio: {nondim_mass_ratio}")
        logger.info(f"Froude number: {froude}")

        flow_past_rod_case(
            nondim_final_time=nondim_final_time,
            grid_size=sim_grid_size,
            reynolds=reynolds,
            nondim_bending_stiffness=nondim_bending_stiffness,
            nondim_mass_ratio=nondim_mass_ratio,
            froude=froude,
        )

    simulate_custom_flow_past_rod_case()
