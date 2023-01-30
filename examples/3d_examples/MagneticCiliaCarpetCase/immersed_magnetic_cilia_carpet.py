import numpy as np
import sopht_mpi.simulator as sps
import sopht.utils as spu
from sopht_mpi.utils.mpi_io import MPIIO, CosseratRodMPIIO
from mpi4py import MPI
from magnetic_cilia_carpet import MagneticCiliaCarpetSimulator
from sopht.simulator.immersed_body import (
    CosseratRodElementCentricForcingGrid,
    FlowForces,
)
from typing import Optional
from sopht_mpi.utils import logger
import logging

# Remove redundant pyelastica warning for every rod
logging.getLogger().setLevel(logging.ERROR)


def immersed_magnetic_cilia_carpet_case(
    womersley: float,
    magnetic_elastic_ratio: float,
    num_rods_along_x: int,
    num_rods_along_y: int,
    num_cycles: float,
    rod_base_length: float = 1.5,
    angular_frequency=np.deg2rad(10.0),
    grid_size_x: int = 128,
    rod_elem_prefactor: float = 1.0,
    coupling_stiffness: float = -2e4,
    coupling_damping: float = -1e1,
    rank_distribution: Optional[tuple[int, int, int]] = (1, 0, 0),
    precision: str = "single",
    save_data: bool = True,
) -> None:
    """
    Example case for arrays of magnetic rods arranged in a grid.

    Note: Since the carpet are basically repeating units of cosserat rods, we can
    decompose the whole carpet into sub-carpets, which serves as a local carpet unit
    in each rank, such that the centroid of each local carpet unit is aligned with the
    centroid of the local eulerian domain. For the purpose of illustration, the rods are
    spaced out sufficiently such that the motion of the rods actuated by magnetic fields
    will stay within it's local rank eulerian grid. This allows us to bypass unnecessary
    communication during flow structure interaction.
    """
    # ==================Physical setup=========================
    if num_rods_along_x < 2 and num_rods_along_y < 2:
        raise ValueError("num_rod along x and y must be no less than 2")
    if num_rods_along_x % 2 != 0 or num_rods_along_y % 2 != 0:
        raise ValueError(
            "num_rods along x and y must be divisible by 2 "
            "for proper flow domain decomposition"
        )
    if rank_distribution[0] != 1:
        raise ValueError(
            "Domain decomposition along z-axis is currently not supported for this "
            "example case. Please read docstring for explanation."
        )

    # spacing rods sufficiently to contain within local eulerian grid
    spacing_between_rods = 2 * rod_base_length
    # Compute full global carpet length
    carpet_length_x = (num_rods_along_x - 1) * spacing_between_rods
    carpet_length_y = (num_rods_along_y - 1) * spacing_between_rods
    # get the flow domain range based on the carpet
    domain_x_range = carpet_length_x + spacing_between_rods
    domain_y_range = carpet_length_y + spacing_between_rods
    domain_z_range = 5 * rod_base_length
    # compute grid size y and z accordingly
    grid_size_y = round(domain_y_range / domain_x_range * grid_size_x)
    grid_size_z = round(domain_z_range / domain_x_range * grid_size_x)
    # align y and z domain range to grid size
    domain_y_range = grid_size_y / grid_size_x * domain_x_range
    domain_z_range = grid_size_z / grid_size_x * domain_x_range

    # ==================FLOW SETUP START=========================
    grid_dim = 3
    real_t = spu.get_real_t(precision)
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    z_axis_idx = spu.VectorField.z_axis_idx()
    # order Z, Y, X
    grid_size = (grid_size_z, grid_size_y, grid_size_x)
    logger.info(f"Flow grid size:{grid_size}")
    kinematic_viscosity = angular_frequency * rod_base_length**2 / womersley**2
    flow_sim = sps.UnboundedFlowSimulator3D(
        grid_size=grid_size,
        x_range=domain_x_range,
        kinematic_viscosity=kinematic_viscosity,
        flow_type="navier_stokes_with_forcing",
        real_t=real_t,
        rank_distribution=rank_distribution,
        filter_vorticity=True,
        filter_setting_dict={"order": 1, "type": "multiplicative"},
    )

    # Averaged fields
    avg_vorticity = np.zeros_like(flow_sim.vorticity_field)
    avg_velocity = np.zeros_like(flow_sim.velocity_field)

    # ==================FLOW SETUP END=========================

    # ==================ROD SIMULATOR START=========================
    # Get local cilia carpet centroid
    local_num_rods_along_x = num_rods_along_x // flow_sim.mpi_construct.grid_topology[2]
    local_num_rods_along_y = num_rods_along_y // flow_sim.mpi_construct.grid_topology[1]
    local_grid_size = flow_sim.mpi_construct.local_grid_size
    substart_idx = flow_sim.mpi_construct.grid.coords * local_grid_size
    subend_idx = substart_idx + local_grid_size
    substart = substart_idx * flow_sim.dx
    subend = subend_idx * flow_sim.dx
    grid_topology_y_axis_idx = 1  # grid topology is in zyx order!
    grid_topology_x_axis_idx = 2  # grid topology is in zyx order!
    local_carpet_base_centroid = np.array(
        [
            0.5
            * (substart[grid_topology_x_axis_idx] + subend[grid_topology_x_axis_idx]),
            0.5
            * (substart[grid_topology_y_axis_idx] + subend[grid_topology_y_axis_idx]),
            0.1 * domain_z_range,
        ]
    )
    n_elem_per_rod = int(grid_size_x * rod_elem_prefactor / num_rods_along_x)
    spatial_magnetisation_wavelength_x = spacing_between_rods * (num_rods_along_x - 1)
    spatial_magnetisation_wavelength_y = spacing_between_rods * (num_rods_along_y - 1)
    local_cilia_carpet_simulator = MagneticCiliaCarpetSimulator(
        magnetic_elastic_ratio=magnetic_elastic_ratio,
        rod_base_length=rod_base_length,
        n_elem_per_rod=n_elem_per_rod,
        num_rods_along_x=local_num_rods_along_x,
        num_rods_along_y=local_num_rods_along_y,
        spacing_between_rods=spacing_between_rods,
        spatial_magnetisation_wavelength_x=spatial_magnetisation_wavelength_x,
        spatial_magnetisation_wavelength_y=spatial_magnetisation_wavelength_y,
        num_cycles=num_cycles,
        carpet_base_centroid=local_carpet_base_centroid,
        plot_result=False,
    )
    # ==================ROD SIMULATOR END=========================

    # ==================FLOW-ROD COMMUNICATOR SETUP START======
    rod_flow_interactor_list = []
    master_rank = flow_sim.mpi_construct.rank  # each interactor is locally owned
    for magnetic_rod in local_cilia_carpet_simulator.magnetic_rod_list:
        rod_flow_interactor = sps.CosseratRodFlowInteraction(
            mpi_construct=flow_sim.mpi_construct,
            mpi_ghost_exchange_communicator=flow_sim.mpi_ghost_exchange_communicator,
            cosserat_rod=magnetic_rod,
            eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
            eul_grid_velocity_field=flow_sim.velocity_field,
            virtual_boundary_stiffness_coeff=coupling_stiffness,
            virtual_boundary_damping_coeff=coupling_damping,
            dx=flow_sim.dx,
            grid_dim=grid_dim,
            forcing_grid_cls=CosseratRodElementCentricForcingGrid,
            master_rank=master_rank,
            assume_data_locality=True,
            auto_ghosting=False,
        )
        rod_flow_interactor_list.append(rod_flow_interactor)
        local_cilia_carpet_simulator.magnetic_beam_sim.add_forcing_to(
            magnetic_rod
        ).using(FlowForces, rod_flow_interactor)
    # ==================FLOW-ROD COMMUNICATOR SETUP END======
    if save_data:
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

        # Setup average Eulerian field IO
        avg_io = MPIIO(mpi_construct=flow_sim.mpi_construct, real_dtype=real_t)
        avg_io.define_eulerian_grid(
            origin=io_origin,
            dx=io_dx,
            grid_size=io_grid_size,
            ghost_size=flow_sim.ghost_size,
        )
        avg_io.add_as_eulerian_fields_for_io(
            avg_vorticity=avg_vorticity,
            avg_velocity=avg_velocity,
        )

        # Initialize carpet IO
        carpet_io = CosseratRodMPIIO(
            mpi_construct=flow_sim.mpi_construct,
            real_dtype=real_t,
            master_rank=master_rank,
        )
        for magnetic_rod in local_cilia_carpet_simulator.magnetic_rod_list:
            carpet_io.add_cosserat_rod_for_io(cosserat_rod=magnetic_rod)

    local_cilia_carpet_simulator.finalize()
    # =================TIMESTEPPING====================
    foto_timer = 0.0
    period_timer = 0.0
    period_timer_limit = local_cilia_carpet_simulator.period
    foto_timer_limit = local_cilia_carpet_simulator.period / 800
    no_period = 0

    # iterate
    while flow_sim.time < local_cilia_carpet_simulator.final_time:
        # Save data
        if foto_timer > foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            if save_data:
                # update_carpet_lag_grid_fields()
                io.save(
                    h5_file_name="flow_" + str("%0.4d" % (flow_sim.time * 100)) + ".h5",
                    time=flow_sim.time,
                )
                carpet_io.save(
                    h5_file_name=f"carpet_"
                    # h5_file_name=f"carpet{flow_sim.mpi_construct.rank}_"
                    + str("%0.4d" % (flow_sim.time * 100)) + ".h5",
                    time=flow_sim.time,
                )

            # compute the grid dev error from independent interactors from each rank
            local_grid_dev_error = 0.0
            for flow_body_interactor in rod_flow_interactor_list:
                local_grid_dev_error += (
                    flow_body_interactor.get_grid_deviation_error_l2_norm(
                        compute_global=False
                    )
                )
            # sum up the grid dev error among all ranks to get the error globally
            grid_dev_error = flow_sim.mpi_construct.grid.allreduce(
                local_grid_dev_error, op=MPI.SUM
            )
            logger.info(
                f"time: {flow_sim.time:.2f} ({(flow_sim.time/local_cilia_carpet_simulator.final_time*100):2.1f}%), "
                f"cycle: {flow_sim.time / local_cilia_carpet_simulator.period:.2f}, "
                f"max_vort: {flow_sim.get_max_vorticity():.4f}, "
                f"vort divg. L2 norm: {flow_sim.get_vorticity_divergence_l2_norm():.4f}, "
                f"grid deviation L2 error: {grid_dev_error:.6f}"
            )

        # Save averaged vorticity field
        if period_timer >= period_timer_limit:
            period_timer = 0.0
            if save_data:
                avg_io.save(
                    h5_file_name=f"avg_flow_{no_period}.h5",
                    time=flow_sim.time,
                )

            avg_vorticity *= 0.0
            avg_velocity *= 0.0
            no_period += 1

        # compute timestep
        flow_dt = flow_sim.compute_stable_timestep(dt_prefac=0.25)

        # Average vorticity field
        avg_vorticity += flow_sim.vorticity_field * flow_dt / period_timer_limit
        avg_velocity += flow_sim.velocity_field * flow_dt / period_timer_limit

        # timestep the rod, through the flow timestep
        rod_time_steps = int(flow_dt / min(flow_dt, local_cilia_carpet_simulator.dt))
        local_rod_dt = flow_dt / rod_time_steps
        rod_time = flow_sim.time
        for i in range(rod_time_steps):
            # timestep the cilia simulator
            rod_time = local_cilia_carpet_simulator.time_step(
                time=rod_time, time_step=local_rod_dt
            )
            # timestep the rod_flow_interactors
            for rod_flow_interactor in rod_flow_interactor_list:
                rod_flow_interactor.time_step(dt=local_rod_dt)

        # evaluate feedback/interaction between flow and rod
        for rod_flow_interactor in rod_flow_interactor_list:
            rod_flow_interactor()

        flow_sim.time_step(dt=flow_dt)

        # update timer
        foto_timer += flow_dt
        period_timer += flow_dt


if __name__ == "__main__":
    immersed_magnetic_cilia_carpet_case(
        womersley=3.0,
        magnetic_elastic_ratio=3.3,
        num_rods_along_x=4,
        num_rods_along_y=8,
        num_cycles=2,
        grid_size_x=128,
    )
