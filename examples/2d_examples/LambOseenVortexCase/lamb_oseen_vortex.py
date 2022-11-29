import numpy as np
import os
import sopht.utils as spu
import sopht_mpi.simulator as sps
from sopht_mpi.utils.mpi_utils_2d import MPIPlotter2D
from lamb_oseen_helpers import compute_lamb_oseen_velocity, compute_lamb_oseen_vorticity
from mpi4py import MPI
from sopht_mpi.utils import logger


def lamb_oseen_vortex_flow_case(grid_size, precision="double", rank_distribution=None):
    """
    This example considers a simple case of Lamb-Oseen vortex, advecting with a
    constant velocity in 2D, while it diffuses in time, and involves solving
    the Navier-Stokes equation.
    """
    real_t = spu.get_real_t(precision)
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()

    # Consider a 1 by 1 2D domain
    x_range = 1.0
    nu = 1e-3
    # init vortex at (0.3 0.3)
    x_cm_start = 0.3
    y_cm_start = x_cm_start
    # start with non-zero to avoid singularity in Lamb-Oseen
    t_start = 1.0
    t_end = 1.4
    # to start with max circulation = 1
    gamma = 4 * np.pi * nu * t_start

    # Initialize unbounded flow simulator
    flow_sim = sps.UnboundedFlowSimulator2D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        flow_type="navier_stokes",
        with_free_stream_flow=True,
        real_t=real_t,
        rank_distribution=rank_distribution,
    )

    # Initialize vorticity, velocity fields and velocity free stream magnitudes
    flow_sim.vorticity_field[...] = compute_lamb_oseen_vorticity(
        x=flow_sim.position_field[x_axis_idx],
        y=flow_sim.position_field[y_axis_idx],
        x_cm=x_cm_start,
        y_cm=y_cm_start,
        nu=nu,
        gamma=gamma,
        t=t_start,
        real_t=real_t,
    )
    velocity_free_stream = np.ones(flow_sim.grid_dim, dtype=real_t)
    flow_sim.velocity_field[...] = compute_lamb_oseen_velocity(
        x=flow_sim.position_field[x_axis_idx],
        y=flow_sim.position_field[y_axis_idx],
        x_cm=x_cm_start,
        y_cm=y_cm_start,
        nu=nu,
        gamma=gamma,
        t=t_end,
        real_t=real_t,
    )

    # iterate
    t = t_start
    foto_timer = 0.0
    foto_timer_limit = (t_end - t_start) / 25

    # Initialize field plotter
    mpi_plotter = MPIPlotter2D(
        flow_sim.mpi_construct,
        flow_sim.ghost_size,
        title=f"Vorticity, time: {t:.2f}",
        master_rank=0,
    )

    while t < t_end:

        # Plot solution
        if foto_timer >= foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0

            # mpi_plotter will take care of gathering field to rank 0 and save
            mpi_plotter.ax.set_title(f"Vorticity, time: {t:.2f}")
            mpi_plotter.contourf(
                flow_sim.position_field[x_axis_idx],
                flow_sim.position_field[y_axis_idx],
                flow_sim.vorticity_field,
                extend="both",
                levels=100,
            )
            mpi_plotter.savefig(file_name="snap_" + str("%0.4d" % (t * 100)) + ".png")
            mpi_plotter.clearfig()

            max_vort = flow_sim.get_max_vorticity()
            logger.info(
                f"time: {t:.2f} ({((t-t_start)/(t_end-t_start)*100):2.1f}%), "
                f"max_vort: {max_vort:.4f}"
            )

        dt = flow_sim.compute_stable_timestep()
        flow_sim.time_step(dt=dt, free_stream_velocity=velocity_free_stream)

        # update time
        t = t + dt
        foto_timer += dt

    # compile video
    if flow_sim.mpi_construct.rank == 0:
        os.system("rm -f flow.mp4")
        os.system(
            "ffmpeg -r 16 -s 3840x2160 -f image2 -pattern_type glob -i 'snap*.png' "
            "-vcodec libx264 -crf 15 -pix_fmt yuv420p -vf 'crop=trunc(iw/2)*2:trunc(ih/2)*2'"
            " flow.mp4"
        )
        os.system("rm -f snap*.png")

    # final vortex field and error
    t_end = t
    x_cm_final = x_cm_start + velocity_free_stream[x_axis_idx] * (t_end - t_start)
    y_cm_final = y_cm_start + velocity_free_stream[y_axis_idx] * (t_end - t_start)
    final_analytical_vorticity_field = compute_lamb_oseen_vorticity(
        x=flow_sim.position_field[x_axis_idx],
        y=flow_sim.position_field[y_axis_idx],
        x_cm=x_cm_final,
        y_cm=y_cm_final,
        nu=nu,
        gamma=gamma,
        t=t_end,
        real_t=real_t,
    )
    # compute local errors and reduce to global later
    inner_idx = (
        slice(flow_sim.ghost_size, -flow_sim.ghost_size),
        slice(flow_sim.ghost_size, -flow_sim.ghost_size),
    )
    error_field = np.fabs(flow_sim.vorticity_field - final_analytical_vorticity_field)[
        inner_idx
    ]
    l2_error = (np.linalg.norm(error_field) * flow_sim.dx) ** 2
    l2_error = flow_sim.mpi_construct.grid.allreduce(l2_error, op=MPI.SUM)
    l2_error = np.sqrt(l2_error)
    linf_error = np.amax(error_field)
    linf_error = flow_sim.mpi_construct.grid.allreduce(linf_error, op=MPI.MAX)
    logger.info(f"Final vortex center location: ({x_cm_final}, {y_cm_final})")
    logger.info(f"vorticity L2 error: {l2_error}")
    logger.info(f"vorticity Linf error: {linf_error}")

    # final velocity field and error
    final_analytical_velocity_field = compute_lamb_oseen_velocity(
        x=flow_sim.position_field[x_axis_idx],
        y=flow_sim.position_field[y_axis_idx],
        x_cm=x_cm_final,
        y_cm=y_cm_final,
        nu=nu,
        gamma=gamma,
        t=t_end,
        real_t=real_t,
    )
    flow_sim.compute_velocity_from_vorticity()
    # compute local errors and reduce to global later
    inner_idx = (
        slice(None),
        slice(flow_sim.ghost_size, -flow_sim.ghost_size),
        slice(flow_sim.ghost_size, -flow_sim.ghost_size),
    )
    error_field = np.fabs(flow_sim.velocity_field - final_analytical_velocity_field)[
        inner_idx
    ]
    l2_error = (np.linalg.norm(error_field) * flow_sim.dx) ** 2
    l2_error = flow_sim.mpi_construct.grid.allreduce(l2_error, op=MPI.SUM)
    l2_error = np.sqrt(l2_error)
    linf_error = np.amax(error_field)
    linf_error = flow_sim.mpi_construct.grid.allreduce(linf_error, op=MPI.MAX)
    logger.info(f"velocity L2 error: {l2_error}")
    logger.info(f"velocity Linf error: {linf_error}")


if __name__ == "__main__":
    lamb_oseen_vortex_flow_case(grid_size=(256, 256))
