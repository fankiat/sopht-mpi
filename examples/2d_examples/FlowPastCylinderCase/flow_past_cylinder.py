import elastica as ea
import click
import numpy as np
import os
from sopht.utils.precision import get_real_t
import sopht_mpi.sopht_mpi_simulator as sps
from sopht_mpi.utils.mpi_utils_2d import MPIPlotter2D


def flow_past_cylinder_boundary_forcing_case(
    nondim_final_time,
    grid_size,
    reynolds,
    coupling_stiffness=-5e4,
    coupling_damping=-20,
    rank_distribution=None,
    precision="single",
    save_diagnostic=False,
):
    """
    This example considers a simple flow past cylinder using immersed
    boundary forcing.
    """
    real_t = get_real_t(precision)
    x_axis_idx = sps.VectorField.x_axis_idx()
    y_axis_idx = sps.VectorField.y_axis_idx()

    # Flow parameters
    velocity_scale = 1.0
    velocity_free_stream = np.zeros(2)
    velocity_free_stream[x_axis_idx] = velocity_scale
    cyl_radius = 0.03
    nu = cyl_radius * velocity_scale / reynolds
    x_range = 1.0

    flow_sim = sps.UnboundedFlowSimulator2D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        flow_type="navier_stokes_with_forcing",
        with_free_stream_flow=True,
        real_t=real_t,
        rank_distribution=rank_distribution,
    )

    # ==================FLOW-BODY COMMUNICATOR SETUP START======
    # Initialize fixed cylinder (elastica rigid body) with direction along Z
    x_cm = 2.5 * cyl_radius
    y_cm = 0.5 * flow_sim.grid_size_y / flow_sim.grid_size_x
    start = np.array([x_cm, y_cm, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([1.0, 0.0, 0.0])
    base_length = 1.0
    density = 1e3
    num_lag_nodes = 60
    cylinder = ea.Cylinder(start, direction, normal, base_length, cyl_radius, density)
    # Since the cylinder is fixed, we dont add it to pyelastica simulator,
    # and directly use it for setting up the flow interactor.
    master_rank = 0  # define master rank for lagrangian related grids
    cylinder_flow_interactor = sps.RigidBodyFlowInteractionMPI(
        mpi_construct=flow_sim.mpi_construct,
        mpi_ghost_exchange_communicator=flow_sim.mpi_ghost_exchange_communicator,
        rigid_body=cylinder,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=coupling_stiffness,
        virtual_boundary_damping_coeff=coupling_damping,
        dx=flow_sim.dx,
        grid_dim=flow_sim.grid_dim,
        real_t=real_t,
        moving_body=False,  # initialize as non-moving boundary
        master_rank=master_rank,
        forcing_grid_cls=sps.CircularCylinderForcingGrid,
        num_forcing_points=num_lag_nodes,
    )
    # ==================FLOW-BODY COMMUNICATOR SETUP END======

    # iterate
    timescale = cyl_radius / velocity_scale
    final_time = nondim_final_time * timescale  # dimensional end time
    time = 0.0
    foto_timer = 0.0
    foto_timer_limit = final_time / 50

    data_timer = 0.0
    data_timer_limit = 0.25 * timescale
    drag_coeffs_time = []
    drag_coeffs = []

    # Initialize field plotter
    mpi_plotter = MPIPlotter2D(
        flow_sim.mpi_construct,
        flow_sim.ghost_size,
        title=f"Vorticity, time: {time / timescale:.2f}",
        master_rank=master_rank,
    )

    while time < final_time:

        # Plot solution
        if foto_timer >= foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            mpi_plotter.ax.set_title(f"Vorticity, time: {time / timescale:.2f}")
            mpi_plotter.contourf(
                flow_sim.position_field[x_axis_idx],
                flow_sim.position_field[y_axis_idx],
                flow_sim.vorticity_field,
                levels=np.linspace(-25, 25, 100),
                extend="both",
            )
            mpi_plotter.scatter(
                cylinder_flow_interactor.forcing_grid.position_field[x_axis_idx],
                cylinder_flow_interactor.forcing_grid.position_field[y_axis_idx],
                s=4,
                color="k",
            )
            mpi_plotter.savefig(
                file_name="snap_" + str("%0.4d" % (time * 100)) + ".png"
            )
            mpi_plotter.clearfig()

            # Compute some diagnostics to log
            grid_dev_error_l2_norm = (
                cylinder_flow_interactor.get_grid_deviation_error_l2_norm()
            )
            max_vort = flow_sim.get_max_vorticity()
            # TODO: implement using logger when available
            if flow_sim.mpi_construct.rank == master_rank:
                print(
                    f"time: {time:.2f} ({(time / final_time * 100):2.1f}%), "
                    f"max_vort: {max_vort:.4f}, "
                    f"grid deviation L2 error: {grid_dev_error_l2_norm:.8f}"
                )

        # track diagnostic data
        if data_timer >= data_timer_limit or data_timer == 0:
            data_timer = 0.0
            if flow_sim.mpi_construct.rank == master_rank:
                drag_coeffs_time.append(time / timescale)
                # calculate drag
                F = np.sum(
                    cylinder_flow_interactor.global_lag_grid_forcing_field[
                        x_axis_idx, ...
                    ]
                )
                drag_coeff = np.fabs(F) / velocity_scale / velocity_scale / cyl_radius
                drag_coeffs.append(drag_coeff)

        dt = flow_sim.compute_stable_timestep()
        # compute flow forcing and timestep forcing
        cylinder_flow_interactor.time_step(dt=dt)
        cylinder_flow_interactor()

        # timestep the flow
        flow_sim.time_step(dt=dt, free_stream_velocity=velocity_free_stream)

        # update time
        time += dt
        foto_timer += dt
        data_timer += dt

    # Plot drag coefficients
    mpi_plotter.ax.set_aspect(aspect="auto")
    mpi_plotter.ax.set_title("Drag Coefficient, Cd")
    mpi_plotter.plot(np.array(drag_coeffs_time), np.array(drag_coeffs))
    mpi_plotter.ax.set_ylim([0.7, 1.7])
    mpi_plotter.ax.set_xlabel("Non-dimensional time")
    mpi_plotter.ax.set_ylabel("Drag coefficient, Cd")
    mpi_plotter.savefig(file_name="drag_vs_time.png")
    mpi_plotter.clearfig()

    # compile video and save diagnostics data
    if flow_sim.mpi_construct.rank == master_rank:
        os.system("rm -f flow.mp4")
        os.system(
            "ffmpeg -r 10 -s 3840x2160 -f image2 -pattern_type glob -i 'snap*.png' "
            "-vcodec libx264 -crf 15 -pix_fmt yuv420p -vf 'crop=trunc(iw/2)*2:trunc(ih/2)*2'"
            " flow.mp4"
        )
        os.system("rm -f snap*.png")

        if save_diagnostic:
            np.savetxt(
                "drag_vs_time.csv",
                np.c_[np.array(drag_coeffs_time), np.array(drag_coeffs)],
                delimiter=",",
            )


if __name__ == "__main__":

    @click.command()
    @click.option(
        "--sim_grid_size_x", default=256, help="Number of grid points in x direction."
    )
    @click.option(
        "--nondim_final_time",
        default=200.0,
        help="Non-dimensional final simulation time.",
    )
    @click.option("--reynolds", default=200.0, help="Reynolds number.")
    def simulate_custom_flow_past_cylinder_case(
        sim_grid_size_x, nondim_final_time, reynolds
    ):
        sim_grid_size_y = sim_grid_size_x // 2
        sim_grid_size = (sim_grid_size_y, sim_grid_size_x)
        # TODO: replace with mpi logger when available, for now its echoed on every rank
        click.echo(f"Grid size: {sim_grid_size}")
        click.echo(f"Reynolds number: {reynolds}")

        flow_past_cylinder_boundary_forcing_case(
            nondim_final_time=nondim_final_time,
            grid_size=sim_grid_size,
            reynolds=reynolds,
            save_diagnostic=True,
        )

    simulate_custom_flow_past_cylinder_case()
