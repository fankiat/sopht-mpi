import click
import elastica as ea
import numpy as np
import sopht_mpi.simulator as sps
import sopht.utils as spu
from sopht_mpi.utils import logger
from sopht_mpi.utils.mpi_io import MPIIO
from mpi4py import MPI
from typing import Optional
from sopht_mpi.utils.mpi_utils_2d import MPIPlotter2D
from sopht.simulator.immersed_body import SphereForcingGrid


def flow_past_sphere_case(
    nondim_time: float,
    grid_size: tuple[int, int, int],
    reynolds: float = 100.0,
    coupling_stiffness: float = -6e5 / 4,
    coupling_damping: float = -3.5e2 / 4,
    rank_distribution: Optional[tuple[int, int, int]] = None,
    precision: str = "single",
    save_flow_data: bool = False,
) -> None:
    """
    This example considers the case of flow past a sphere in 3D.
    """
    grid_size_z, grid_size_y, grid_size_x = grid_size
    real_t = spu.get_real_t(precision)
    x_axis_idx: int = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    z_axis_idx = spu.VectorField.z_axis_idx()
    x_range = 1.0
    far_field_velocity = 1.0
    sphere_diameter = 0.4 * min(grid_size_z, grid_size_y) / grid_size_x * x_range
    nu = far_field_velocity * sphere_diameter / reynolds
    flow_sim = sps.UnboundedFlowSimulator3D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        real_t=real_t,
        rank_distribution=rank_distribution,
        flow_type="navier_stokes_with_forcing",
        with_free_stream_flow=True,
        time=0.0,
    )
    rho_f = 1.0
    sphere_projected_area = 0.25 * np.pi * sphere_diameter**2
    drag_force_scale = 0.5 * rho_f * far_field_velocity**2 * sphere_projected_area

    # Initialize velocity = c in X direction
    velocity_free_stream = np.array([far_field_velocity, 0.0, 0.0])

    # Initialize fixed sphere (elastica rigid body)
    x_cm = 0.25 * flow_sim.x_range
    y_cm = 0.5 * flow_sim.y_range
    z_cm = 0.5 * flow_sim.z_range
    sphere_com = np.array([x_cm, y_cm, z_cm])
    density = 1e3
    sphere = ea.Sphere(
        center=sphere_com, base_radius=(sphere_diameter / 2.0), density=density
    )
    # Since the sphere is fixed, we don't add it to pyelastica simulator,
    # and directly use it for setting up the flow interactor.
    # ==================FLOW-BODY COMMUNICATOR SETUP START======
    num_forcing_points_along_equator = int(
        1.875 * sphere_diameter / x_range * grid_size_x
    )
    master_rank = 0  # define master rank for lagrangian related grids
    sphere_flow_interactor = sps.RigidBodyFlowInteractionMPI(
        mpi_construct=flow_sim.mpi_construct,
        mpi_ghost_exchange_communicator=flow_sim.mpi_ghost_exchange_communicator,
        rigid_body=sphere,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=coupling_stiffness,
        virtual_boundary_damping_coeff=coupling_damping,
        dx=flow_sim.dx,
        grid_dim=flow_sim.grid_dim,
        master_rank=master_rank,
        forcing_grid_cls=SphereForcingGrid,
        num_forcing_points_along_equator=num_forcing_points_along_equator,
    )
    # ==================FLOW-BODY COMMUNICATOR SETUP END======

    if save_flow_data:
        # setup IO
        # TODO internalise this in flow simulator as dump_fields
        origin_x = flow_sim.mpi_construct.grid.allreduce(
            flow_sim.position_field[
                x_axis_idx,
                flow_sim.ghost_size : -flow_sim.ghost_size,
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
        io_grid_size = np.array([grid_size_z, grid_size_y, grid_size_x])
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
        # Initialize sphere IO
        sphere_io = MPIIO(mpi_construct=flow_sim.mpi_construct, real_dtype=real_t)
        # Add vector field on lagrangian grid
        sphere_io.add_as_lagrangian_fields_for_io(
            lagrangian_grid=sphere_flow_interactor.forcing_grid.position_field,
            lagrangian_grid_master_rank=sphere_flow_interactor.master_rank,
            lagrangian_grid_name="sphere",
            vector_3d=sphere_flow_interactor.global_lag_grid_forcing_field,
        )

    timescale = sphere_diameter / far_field_velocity
    t_end_hat = nondim_time  # non-dimensional end time
    t_end = t_end_hat * timescale  # dimensional end time
    foto_timer = 0.0
    foto_timer_limit = timescale / 10

    time = []
    drag_coeffs = []

    # iterate
    while flow_sim.time < t_end:
        # Save data
        if foto_timer > foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            # calculate drag
            drag_coeff = 0.0
            if flow_sim.mpi_construct.rank == master_rank:
                drag_force = np.fabs(
                    np.sum(
                        sphere_flow_interactor.global_lag_grid_forcing_field[
                            x_axis_idx, ...
                        ]
                    )
                )
                drag_coeff = drag_force / drag_force_scale
                time.append(flow_sim.time)
                drag_coeffs.append(drag_coeff)

            if save_flow_data:
                io.save(
                    h5_file_name="flow_" + str("%0.4d" % (flow_sim.time * 100)) + ".h5",
                    time=flow_sim.time,
                )
                sphere_io.save(
                    h5_file_name="sphere_"
                    + str("%0.4d" % (flow_sim.time * 100))
                    + ".h5",
                    time=flow_sim.time,
                )

            # Log some diagnostics
            logger.info(
                f"time: {flow_sim.time:.2f} ({(flow_sim.time/t_end*100):2.1f}%), "
                f"max_vort: {flow_sim.get_max_vorticity():.4f}, "
                f"drag coeff: {drag_coeff:.4f}, "
                f"vort divg. L2 norm: {flow_sim.get_vorticity_divergence_l2_norm():.4f} "
                "grid deviation L2 error: "
                f"{sphere_flow_interactor.get_grid_deviation_error_l2_norm():.6f}"
            )

        dt = flow_sim.compute_stable_timestep(dt_prefac=0.5)

        # compute flow forcing and timestep forcing
        sphere_flow_interactor.time_step(dt=dt)
        sphere_flow_interactor()

        flow_sim.time_step(dt=dt, free_stream_velocity=velocity_free_stream)

        # update timers
        foto_timer += dt

    # Initialize mpi-supported plotter
    mpi_plotter = MPIPlotter2D(
        flow_sim.mpi_construct,
        flow_sim.ghost_size,
        title=f"Vorticity, time: {flow_sim.time / timescale:.2f}",
        master_rank=sphere_flow_interactor.master_rank,
    )
    mpi_plotter.ax.set_aspect(aspect="auto")
    mpi_plotter.plot(np.array(time), np.array(drag_coeffs), label="numerical")
    mpi_plotter.ax.set_xlabel("Time")
    mpi_plotter.ax.set_ylabel("Drag coefficient")
    mpi_plotter.savefig("drag_coeff_vs_time.png")

    if flow_sim.mpi_construct.rank == master_rank:
        np.savetxt(
            "drag_vs_time.csv",
            np.c_[np.array(time), np.array(drag_coeffs)],
            delimiter=",",
            header="time, drag_coeff",
        )

        if save_flow_data:
            spu.make_dir_and_transfer_h5_data(dir_name="flow_data_h5")


if __name__ == "__main__":

    @click.command()
    @click.option("--nx", default=128, help="Number of grid points in x direction.")
    @click.option("--reynolds", default=100.0, help="Reynolds number of flow.")
    def simulate_parallelised_flow_past_sphere(nx: int, reynolds: float) -> None:
        ny = nx // 2
        nz = nx // 2
        # in order Z, Y, X
        grid_size = (nz, ny, nx)
        logger.info(f"Grid size: {grid_size}")
        logger.info(f"Flow Reynolds number: {reynolds}")
        flow_past_sphere_case(
            nondim_time=10.0,
            grid_size=grid_size,
            reynolds=reynolds,
            save_flow_data=True,
        )

    simulate_parallelised_flow_past_sphere()
