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


def flow_past_rod_case(
    non_dim_final_time: float,
    n_elem: int,
    grid_size: tuple[int, int, int],
    surface_grid_density_for_largest_element: int,
    cauchy_number: float,
    mass_ratio: float,
    froude_number: float,
    stretch_bending_ratio: float,
    poisson_ratio: float = 0.5,
    reynolds: float = 100.0,
    coupling_stiffness: float = -2e5,
    coupling_damping: float = -1e2,
    rod_start_incline_angle: float = 0.0,
    rank_distribution: Optional[tuple[int, int, int]] = None,
    precision: str = "single",
    save_data: bool = False,
) -> None:
    # =================COMMON SIMULATOR STUFF=======================
    grid_size_z, grid_size_y, grid_size_x = grid_size
    real_t = spu.get_real_t(precision)
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    z_axis_idx = spu.VectorField.z_axis_idx()
    rho_f = 1.0
    U_free_stream = 1.0
    base_length = 1.0
    x_range = 1.8 * base_length
    y_range = grid_size_y / grid_size_x * x_range
    z_range = grid_size_z / grid_size_x * x_range
    velocity_free_stream = [U_free_stream, 0.0, 0.0]
    # =================PYELASTICA STUFF BEGIN=====================

    class FlowPastRodSimulator(
        ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping
    ):
        ...

    flow_past_sim = FlowPastRodSimulator()
    start = np.array([0.2 * x_range, 0.5 * y_range, 0.75 * z_range])
    direction = np.array(
        [np.sin(rod_start_incline_angle), 0.0, -np.cos(rod_start_incline_angle)]
    )
    normal = np.array([0.0, 1.0, 0.0])
    base_diameter = y_range / 5.0
    base_radius = base_diameter / 2.0
    base_area = np.pi * base_radius**2
    # mass_ratio = rho_s / rho_f
    rho_s = mass_ratio * rho_f
    moment_of_inertia = np.pi / 4 * base_radius**4
    # Cau = (rho_f U^2 L^3 D) / EI
    youngs_modulus = (rho_f * U_free_stream**2 * base_length**3 * base_diameter) / (
        cauchy_number * moment_of_inertia
    )
    # Froude = g L / U^2
    gravitational_acc = froude_number * U_free_stream**2 / base_diameter
    # Stretch to Bending ratio EAL2 / EI
    Es_Eb = stretch_bending_ratio * moment_of_inertia / (base_area * base_length**2)

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
    flow_past_rod.shear_matrix[2, 2, :] *= Es_Eb
    flow_past_sim.append(flow_past_rod)
    flow_past_sim.constrain(flow_past_rod).using(
        ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )
    # Add gravitational forces
    flow_past_sim.add_forcing_to(flow_past_rod).using(
        ea.GravityForces, acc_gravity=np.array([0.0, 0.0, -gravitational_acc])
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
        filter_setting_dict={"order": 1, "type": "multiplicative"},
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
        forcing_grid_cls=CosseratRodSurfaceForcingGrid,
        surface_grid_density_for_largest_element=surface_grid_density_for_largest_element,
        master_rank=master_rank,
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

    foto_timer = 0.0
    timescale = base_length / U_free_stream
    final_time = non_dim_final_time * timescale
    foto_timer_limit = 1 / 30
    time_history = []
    rod_angle = []
    force_history = []

    def rod_incline_angle_with_horizon(rod: ea.CosseratRod):
        return np.rad2deg(
            np.fabs(
                np.arctan(
                    (
                        rod.position_collection[z_axis_idx, -1]
                        - rod.position_collection[z_axis_idx, 0]
                    )
                    / (
                        rod.position_collection[x_axis_idx, -1]
                        - rod.position_collection[x_axis_idx, 0]
                    )
                )
            )
        )

    # iterate
    while flow_sim.time < final_time:
        # Save data
        if foto_timer > foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            if save_data:
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

            forces = 0.0
            if flow_sim.mpi_construct.rank == cosserat_rod_flow_interactor.master_rank:
                time_history.append(flow_sim.time)
                rod_angle.append(rod_incline_angle_with_horizon(flow_past_rod))
                forces = np.sum(
                    cosserat_rod_flow_interactor.global_lag_grid_forcing_field, axis=1
                )
                force_history.append(forces.copy())

            # Log some diagnostics
            logger.info(
                f"time: {flow_sim.time:.2f} ({(flow_sim.time/final_time*100):2.1f}%), "
                f"max_vort: {flow_sim.get_max_vorticity():.4f}, "
                f"rod angle: {rod_incline_angle_with_horizon(flow_past_rod):2.2f}, "
                f"vort divg. L2 norm: {flow_sim.get_vorticity_divergence_l2_norm():.4f},"
                " grid deviation L2 error: "
                f"{cosserat_rod_flow_interactor.get_grid_deviation_error_l2_norm():.6f},"
                f" total force: {forces},"
                f" force norm: {np.linalg.norm(forces)}"
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

        flow_sim.time_step(
            dt=flow_dt,
            free_stream_velocity=velocity_free_stream,
        )

        # update timer
        foto_timer += flow_dt

    # Save data
    if flow_sim.mpi_construct.rank == master_rank:
        np.savetxt(
            "rod_angle_vs_time.csv",
            np.c_[np.array(time_history), np.array(rod_angle)],
            delimiter=",",
            header="time, rod_angle",
        )

        np.savetxt(
            "rod_forces_vs_time.csv",
            np.c_[
                np.array(time_history),
                np.array(force_history),
                np.linalg.norm(np.array(force_history), axis=1),
            ],
            delimiter=",",
            header="time, force x, force y, force z, force norm",
        )

        if save_data:
            spu.make_dir_and_transfer_h5_data(dir_name="flow_data_h5")


if __name__ == "__main__":

    @click.command()
    @click.option("--nx", default=128, help="Number of grid points in x direction.")
    @click.option("--u_free_stream", default=1.1, help="Free stream flow velocity.")
    def simulate_flow_past_rod(nx: int, u_free_stream: float) -> None:
        ny = nx // 4
        nz = nx
        # in order Z, Y, X
        grid_size = (nz, ny, nx)
        surface_grid_density_for_largest_element = nx // 8
        n_elem = 5 * nx // 16

        logger.info(f"Grid size:  {nz, ny, nx ,} ")
        logger.info(
            f"num forcing points around the surface:  {surface_grid_density_for_largest_element}"
        )
        logger.info(f"num rod elements: {n_elem}")
        logger.info(f"Free stream flow velocity: {u_free_stream}")

        final_time = 3.5

        exp_rho_s = 1e3  # kg/m3
        exp_rho_f = 1.21  # kg/m3
        exp_youngs_modulus = 2.25e5  # Pa
        exp_poisson_ratio = 0.01
        exp_base_length = 25e-3  # m
        exp_base_diameter = 0.4e-3  # m
        exp_kinematic_viscosity = 1.51e-5  # m2/s
        exp_U_free_stream = u_free_stream  # m/s
        exp_gravitational_acc = 9.80665  # m/s2

        exp_mass_ratio = exp_rho_s / exp_rho_f
        exp_base_radius = exp_base_diameter / 2
        exp_base_area = np.pi * exp_base_radius**2
        exp_moment_of_inertia = np.pi / 4 * exp_base_radius**4
        exp_bending_rigidity = exp_youngs_modulus * exp_moment_of_inertia
        exp_cauchy_number = (
            exp_rho_f
            * exp_U_free_stream**2
            * exp_base_length**3
            * exp_base_diameter
            / exp_bending_rigidity
        )
        # Froude = g D / U^2
        exp_froude_number = (
            exp_gravitational_acc * exp_base_diameter / exp_U_free_stream**2
        )
        exp_Re = exp_U_free_stream * exp_base_diameter / exp_kinematic_viscosity

        # stretch to bending ratio EAL2 / EI
        exp_Ks_Kb = (exp_youngs_modulus * exp_base_area * exp_base_length**2) / (
            exp_youngs_modulus * exp_moment_of_inertia
        )

        # Drag coefficient from Silvaleon 2018 Eq 10
        Cd = (1.13 + 11.4 / exp_Re**0.808) ** 0.952
        # Silvaleon 2018
        Ca_B = (
            (2 / np.pi) * Cd * (exp_rho_f / (exp_rho_s - exp_rho_f)) / exp_froude_number
        )
        # Final deflection angle Silvaleon 2018 Eq 15
        rod_start_incline_angle = np.deg2rad(
            90 - 90 / (1 + 3.32 * Ca_B**1.33) ** 0.407 + 0.5
        )

        logger.info(
            f"Re: {exp_Re}, Ca: {exp_cauchy_number}, Fr: {exp_froude_number}, Angle: {rod_start_incline_angle}"
        )

        flow_past_rod_case(
            non_dim_final_time=final_time,
            cauchy_number=exp_cauchy_number,
            mass_ratio=exp_mass_ratio,
            froude_number=exp_froude_number,
            poisson_ratio=exp_poisson_ratio,
            reynolds=exp_Re,
            stretch_bending_ratio=exp_Ks_Kb,
            grid_size=grid_size,
            surface_grid_density_for_largest_element=surface_grid_density_for_largest_element,
            n_elem=n_elem,
            rod_start_incline_angle=rod_start_incline_angle,
            save_data=True,
        )

    simulate_flow_past_rod()
