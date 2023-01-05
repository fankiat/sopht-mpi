import numpy as np
from sopht_mpi.numeric.eulerian_grid_ops import (
    gen_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d,
    gen_diffusion_timestep_euler_forward_pyst_mpi_kernel_3d,
    gen_penalise_field_boundary_pyst_mpi_kernel_3d,
    gen_curl_pyst_mpi_kernel_3d,
    gen_update_vorticity_from_velocity_forcing_pyst_mpi_kernel_3d,
    gen_divergence_pyst_mpi_kernel_3d,
    gen_laplacian_filter_mpi_kernel_3d,
    UnboundedPoissonSolverMPI3D,
)
from sopht_mpi.utils import MPIConstruct3D, MPIGhostCommunicator3D, logger
from sopht.utils.field import VectorField
from sopht.numeric.eulerian_grid_ops import (
    gen_add_fixed_val_pyst_kernel_3d,
    gen_set_fixed_val_pyst_kernel_3d,
    gen_elementwise_cross_product_pyst_kernel_3d,
)
from sopht.utils.precision import get_test_tol
from mpi4py import MPI
from typing import Callable


class UnboundedFlowSimulator3D:
    """Class for MPI-supported 3D unbounded flow simulator"""

    def __init__(
        self,
        grid_size,
        x_range,
        kinematic_viscosity,
        time=0.0,
        CFL=0.1,
        flow_type="passive_scalar",
        filter_vorticity=False,
        real_t=np.float32,
        rank_distribution=None,
        ghost_size=2,
        **kwargs,
    ):
        """Class initialiser

        :param grid_size: Grid size of simulator
        :param x_range: Range of X coordinate of the grid
        :param kinematic_viscosity: kinematic viscosity of the fluid
        :param time: simulator time at initialisation
        :param CFL: Courant Freidrich Lewy number (advection timestep)
        :param flow_type: Nature of the simulator, can be "passive_scalar" (default value),
        "passive_vector", "navier_stokes" or "navier_stokes_with_forcing"
        :param filter_vorticity: flag to determine if vorticity should be filtered or not,
        needed for stability sometimes
        :param real_t: precision of the solver
        :param rank_distribution: distribution configuration of the grid
        :param ghost_size: ghost size for subdomains

        Notes
        -----
        Currently only supports Euler forward timesteps :(
        """
        self.grid_dim = 3
        self.grid_size = grid_size
        self.grid_size_z, self.grid_size_y, self.grid_size_x = self.grid_size
        self.x_range = x_range
        self.real_t = real_t
        self.flow_type = flow_type
        self.kinematic_viscosity = kinematic_viscosity
        self.CFL = CFL
        self.time = time
        self.filter_vorticity = filter_vorticity
        supported_flow_types = [
            "passive_scalar",
            "passive_vector",
            "navier_stokes",
            "navier_stokes_with_forcing",
        ]
        if self.flow_type not in supported_flow_types:
            raise ValueError("Invalid flow type given")

        # MPI-related variables
        self.rank_distribution = rank_distribution
        self.ghost_size = ghost_size

        self.init_mpi()
        self.init_domain()
        self.init_fields()

        if self.flow_type in ["navier_stokes", "navier_stokes_with_forcing"]:
            self.penalty_zone_width = kwargs.get("penalty_zone_width", 2)
            self.with_free_stream_flow = kwargs.get("with_free_stream_flow", False)
            if self.filter_vorticity:
                logger.warning(
                    "==============================================="
                    "\nVorticity filtering is turned on."
                )
                self.filter_setting_dict = kwargs.get("filter_setting_dict")
                if self.filter_setting_dict is None:
                    # set default values for the filter setting dictionary
                    self.filter_setting_dict = {"order": 2, "type": "multiplicative"}
                    logger.warning(
                        "Since a dict named filter_setting with keys "
                        "\n'order' and 'type' is not provided, setting "
                        f"\ndefault filter order = {self.filter_setting_dict['order']}"
                        f"\nand type: {self.filter_setting_dict['type']}"
                    )
                logger.warning("===============================================")
        self.compile_kernels()
        self.finalise_flow_timestep()

    def init_mpi(self):
        self.mpi_construct = MPIConstruct3D(
            grid_size_z=self.grid_size_z,
            grid_size_y=self.grid_size_y,
            grid_size_x=self.grid_size_x,
            real_t=self.real_t,
            rank_distribution=self.rank_distribution,
        )
        self.mpi_ghost_exchange_communicator = MPIGhostCommunicator3D(
            ghost_size=self.ghost_size, mpi_construct=self.mpi_construct
        )

    def init_domain(self):
        """Initialize the MPI local domain (with ghost cells)"""
        self.y_range = self.x_range * self.grid_size_y / self.grid_size_x
        self.z_range = self.x_range * self.grid_size_z / self.grid_size_x
        self.dx = self.real_t(self.x_range / self.grid_size_x)
        eul_grid_shift = self.dx / 2.0
        ghost_grid_shift = self.ghost_size * self.dx

        # Generate grid meshes for each corresponding rank based on local coords
        local_grid_size = self.mpi_construct.local_grid_size
        substart_idx = self.mpi_construct.grid.coords * local_grid_size
        subend_idx = substart_idx + local_grid_size
        substart_z, substart_y, substart_x = substart_idx * self.dx
        subend_z, subend_y, subend_x = subend_idx * self.dx
        local_grid_size_z, local_grid_size_y, local_grid_size_x = local_grid_size
        local_x = np.linspace(
            eul_grid_shift + substart_x - ghost_grid_shift,
            subend_x - eul_grid_shift + ghost_grid_shift,
            local_grid_size_x + 2 * self.ghost_size,
        ).astype(self.real_t)
        local_y = np.linspace(
            eul_grid_shift + substart_y - ghost_grid_shift,
            subend_y - eul_grid_shift + ghost_grid_shift,
            local_grid_size_y + 2 * self.ghost_size,
        ).astype(self.real_t)
        local_z = np.linspace(
            eul_grid_shift + substart_z - ghost_grid_shift,
            subend_z - eul_grid_shift + ghost_grid_shift,
            local_grid_size_z + 2 * self.ghost_size,
        ).astype(self.real_t)
        # flipud so that position field are ordered according to VectorField convention
        self.position_field = np.flipud(
            np.array(np.meshgrid(local_z, local_y, local_x, indexing="ij"))
        )
        self.local_grid_size_with_ghost = local_grid_size + 2 * self.ghost_size

        logger.info(
            "==============================================="
            f"\n{self.grid_dim}D flow domain initialized with:"
            f"\nX axis from 0.0 to {self.x_range}"
            f"\nY axis from 0.0 to {self.y_range}"
            f"\nZ axis from 0.0 to {self.z_range}"
            "\nPlease initialize bodies within these bounds!"
            "\n==============================================="
        )

    def init_fields(self):
        """Initialize the necessary field arrays, i.e. vorticity, velocity, etc."""
        # Initialize flow field
        self.primary_scalar_field = np.zeros(
            self.local_grid_size_with_ghost, dtype=self.real_t
        )
        self.velocity_field = np.zeros(
            (self.grid_dim, *self.local_grid_size_with_ghost), dtype=self.real_t
        )
        # we use the same buffer for advection, diffusion and velocity recovery
        self.buffer_scalar_field = np.zeros_like(self.primary_scalar_field)

        if self.flow_type in [
            "passive_vector",
            "navier_stokes",
            "navier_stokes_with_forcing",
        ]:
            self.primary_vector_field = np.zeros_like(self.velocity_field)
            del self.primary_scalar_field  # not needed

        if self.flow_type in ["navier_stokes", "navier_stokes_with_forcing"]:
            self.vorticity_field = self.primary_vector_field.view()
            self.stream_func_field = np.zeros_like(self.vorticity_field)
            self.buffer_vector_field = np.zeros_like(self.vorticity_field)
        if self.flow_type == "navier_stokes_with_forcing":
            # this one holds the forcing from bodies
            self.eul_grid_forcing_field = np.zeros_like(self.velocity_field)

    def compile_kernels(self):
        """Compile necessary kernels based on flow type"""
        if self.flow_type == "passive_scalar":
            self.diffusion_timestep = (
                gen_diffusion_timestep_euler_forward_pyst_mpi_kernel_3d(
                    mpi_construct=self.mpi_construct,
                    ghost_exchange_communicator=self.mpi_ghost_exchange_communicator,
                    real_t=self.real_t,
                    field_type="scalar",
                )
            )
            self.advection_timestep = gen_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d(
                mpi_construct=self.mpi_construct,
                ghost_exchange_communicator=self.mpi_ghost_exchange_communicator,
                real_t=self.real_t,
                field_type="scalar",
            )
        elif self.flow_type == "passive_vector":
            self.diffusion_timestep = (
                gen_diffusion_timestep_euler_forward_pyst_mpi_kernel_3d(
                    mpi_construct=self.mpi_construct,
                    ghost_exchange_communicator=self.mpi_ghost_exchange_communicator,
                    real_t=self.real_t,
                    field_type="vector",
                )
            )
            self.advection_timestep = gen_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d(
                mpi_construct=self.mpi_construct,
                ghost_exchange_communicator=self.mpi_ghost_exchange_communicator,
                real_t=self.real_t,
                field_type="vector",
            )

        if self.flow_type in ["navier_stokes", "navier_stokes_with_forcing"]:
            self.diffusion_timestep = (
                gen_diffusion_timestep_euler_forward_pyst_mpi_kernel_3d(
                    mpi_construct=self.mpi_construct,
                    ghost_exchange_communicator=self.mpi_ghost_exchange_communicator,
                    real_t=self.real_t,
                    field_type="vector",
                )
            )
            self.unbounded_poisson_solver = UnboundedPoissonSolverMPI3D(
                grid_size_z=self.grid_size_z,
                grid_size_y=self.grid_size_y,
                grid_size_x=self.grid_size_x,
                x_range=self.x_range,
                real_t=self.real_t,
                mpi_construct=self.mpi_construct,
                ghost_size=self.ghost_size,
            )
            self.curl = gen_curl_pyst_mpi_kernel_3d(
                mpi_construct=self.mpi_construct,
                ghost_exchange_communicator=self.mpi_ghost_exchange_communicator,
                real_t=self.real_t,
            )
            self.penalise_field_towards_boundary = (
                gen_penalise_field_boundary_pyst_mpi_kernel_3d(
                    mpi_construct=self.mpi_construct,
                    real_t=self.real_t,
                    ghost_exchange_communicator=self.mpi_ghost_exchange_communicator,
                    width=self.penalty_zone_width,
                    dx=self.dx,
                    x_grid_field=self.position_field[VectorField.x_axis_idx()],
                    y_grid_field=self.position_field[VectorField.y_axis_idx()],
                    z_grid_field=self.position_field[VectorField.z_axis_idx()],
                    field_type="vector",
                )
            )
            self.elementwise_cross_product = (
                gen_elementwise_cross_product_pyst_kernel_3d(real_t=self.real_t)
            )
            self.update_vorticity_from_velocity_forcing = (
                gen_update_vorticity_from_velocity_forcing_pyst_mpi_kernel_3d(
                    mpi_construct=self.mpi_construct,
                    ghost_exchange_communicator=self.mpi_ghost_exchange_communicator,
                    real_t=self.real_t,
                )
            )

            # check if vorticity stays divergence free
            self.compute_divergence = gen_divergence_pyst_mpi_kernel_3d(
                mpi_construct=self.mpi_construct,
                ghost_exchange_communicator=self.mpi_ghost_exchange_communicator,
                real_t=self.real_t,
            )

            # filter kernel compilation
            def filter_vector_field(vector_field):
                ...

            self.filter_vector_field = filter_vector_field
            if self.filter_vorticity and self.filter_setting_dict is not None:
                self.filter_vector_field = gen_laplacian_filter_mpi_kernel_3d(
                    mpi_construct=self.mpi_construct,
                    ghost_exchange_communicator=self.mpi_ghost_exchange_communicator,
                    filter_order=self.filter_setting_dict["order"],
                    filter_flux_buffer=self.buffer_vector_field[0],
                    field_buffer=self.buffer_vector_field[1],
                    real_t=self.real_t,
                    field_type="vector",
                    filter_type=self.filter_setting_dict["type"],
                )

        if self.flow_type == "navier_stokes_with_forcing":
            self.set_field = gen_set_fixed_val_pyst_kernel_3d(
                real_t=self.real_t, field_type="vector"
            )
        # free stream velocity stuff (only meaningful in navier stokes problems)
        if self.flow_type in ["navier_stokes", "navier_stokes_with_forcing"]:
            if self.with_free_stream_flow:
                add_fixed_val = gen_add_fixed_val_pyst_kernel_3d(
                    real_t=self.real_t, field_type="vector"
                )

                def update_velocity_with_free_stream(free_stream_velocity):
                    add_fixed_val(
                        sum_field=self.velocity_field,
                        vector_field=self.velocity_field,
                        fixed_vals=free_stream_velocity,
                    )

            else:

                def update_velocity_with_free_stream(free_stream_velocity):
                    ...

            self.update_velocity_with_free_stream = update_velocity_with_free_stream

    def finalise_navier_stokes_timestep(self):
        def default_navier_stokes_timestep(dt, free_stream_velocity):
            ...

        self.navier_stokes_timestep = default_navier_stokes_timestep
        if self.flow_type in ["navier_stokes", "navier_stokes_with_forcing"]:
            self.navier_stokes_timestep = self.rotational_form_navier_stokes_timestep

    def finalise_flow_timestep(self):
        self.finalise_navier_stokes_timestep()
        self.flow_time_step: Callable
        # default time step
        self.flow_time_step = self.scalar_advection_and_diffusion_timestep
        if self.flow_type == "passive_vector":
            self.flow_time_step = self.vector_advection_and_diffusion_timestep
        elif self.flow_type == "navier_stokes":
            self.flow_time_step = self.navier_stokes_timestep
        elif self.flow_type == "navier_stokes_with_forcing":
            self.flow_time_step = self.navier_stokes_with_forcing_timestep

    def update_simulator_time(self, dt):
        """Updates simulator time."""
        self.time += dt

    def time_step(self, dt, **kwargs):
        """Final simulator time step"""
        self.flow_time_step(dt=dt, **kwargs)
        self.update_simulator_time(dt=dt)

    def scalar_advection_and_diffusion_timestep(self, dt: float, **kwargs) -> None:
        self.advection_timestep(
            field=self.primary_scalar_field,
            advection_flux=self.buffer_scalar_field,
            velocity=self.velocity_field,
            dt_by_dx=self.real_t(dt / self.dx),
        )
        self.diffusion_timestep(
            field=self.primary_scalar_field,
            diffusion_flux=self.buffer_scalar_field,
            nu_dt_by_dx2=self.real_t(self.kinematic_viscosity * dt / self.dx / self.dx),
        )

    def vector_advection_and_diffusion_timestep(self, dt: float, **kwargs) -> None:
        self.advection_timestep(
            vector_field=self.primary_vector_field,
            advection_flux=self.buffer_scalar_field,
            velocity=self.velocity_field,
            dt_by_dx=self.real_t(dt / self.dx),
        )
        self.diffusion_timestep(
            vector_field=self.primary_vector_field,
            diffusion_flux=self.buffer_scalar_field,
            nu_dt_by_dx2=self.real_t(self.kinematic_viscosity * dt / self.dx / self.dx),
        )

    def compute_flow_velocity(self, free_stream_velocity):
        self.penalise_field_towards_boundary(vector_field=self.vorticity_field)
        self.unbounded_poisson_solver.vector_field_solve(
            solution_vector_field=self.stream_func_field,
            rhs_vector_field=self.vorticity_field,
        )
        self.curl(
            curl=self.velocity_field,
            field=self.stream_func_field,
            prefactor=self.real_t(0.5 / self.dx),
        )
        self.update_velocity_with_free_stream(free_stream_velocity=free_stream_velocity)

    def rotational_form_navier_stokes_timestep(self, dt, free_stream_velocity):
        velocity_cross_vorticity = self.buffer_vector_field.view()
        self.elementwise_cross_product(
            result_field=velocity_cross_vorticity,
            field_1=self.velocity_field,
            field_2=self.vorticity_field,
        )
        self.update_vorticity_from_velocity_forcing(
            vorticity_field=self.vorticity_field,
            velocity_forcing_field=velocity_cross_vorticity,
            prefactor=self.real_t(dt / (2 * self.dx)),
        )
        self.diffusion_timestep(
            vector_field=self.vorticity_field,
            diffusion_flux=self.buffer_scalar_field,
            nu_dt_by_dx2=self.real_t(self.kinematic_viscosity * dt / self.dx / self.dx),
        )
        self.filter_vector_field(vector_field=self.vorticity_field)
        self.compute_flow_velocity(free_stream_velocity=free_stream_velocity)

    def navier_stokes_with_forcing_timestep(self, dt, free_stream_velocity):
        self.update_vorticity_from_velocity_forcing(
            vorticity_field=self.vorticity_field,
            velocity_forcing_field=self.eul_grid_forcing_field,
            prefactor=self.real_t(dt / (2 * self.dx)),
        )
        self.navier_stokes_timestep(dt=dt, free_stream_velocity=free_stream_velocity)
        self.set_field(
            vector_field=self.eul_grid_forcing_field, fixed_vals=[0.0] * self.grid_dim
        )

    def compute_stable_timestep(self, dt_prefac=1, precision="single"):
        """Compute stable timestep based on advection and diffusion limits."""
        # This may need a numba or pystencil version
        velocity_mag_field = self.buffer_scalar_field.view()
        tol = get_test_tol(precision)
        velocity_mag_field[...] = np.sum(np.fabs(self.velocity_field), axis=0)
        dt = min(
            self.CFL
            * self.dx
            / (
                np.amax(
                    velocity_mag_field[
                        self.ghost_size : -self.ghost_size,
                        self.ghost_size : -self.ghost_size,
                        self.ghost_size : -self.ghost_size,
                    ]
                )
                + tol
            ),
            0.9 * self.dx**2 / (2 * self.grid_dim) / (self.kinematic_viscosity + tol),
        )
        # Get smallest timestep among all the ranks
        dt = self.mpi_construct.grid.allreduce(dt, op=MPI.MIN)
        return dt * dt_prefac

    def get_vorticity_divergence_l2_norm(self):
        """Return L2 norm of divergence of the vorticity field."""
        divergence_field = self.buffer_scalar_field.view()
        self.compute_divergence(
            divergence=divergence_field,
            field=self.vorticity_field,
            inv_dx=(1.0 / self.dx),
        )
        # Perform the expression below in an MPI context.
        # vorticity_divg_l2_norm = np.linalg.norm(divergence_field) * self.dx**1.5
        gs = self.ghost_size
        local_vorticity_divg_l2_norm_sq = (
            np.linalg.norm(divergence_field[gs:-gs, gs:-gs, gs:-gs]) ** 2
        )
        vorticity_divg_l2_norm_sq = self.mpi_construct.grid.allreduce(
            local_vorticity_divg_l2_norm_sq, op=MPI.SUM
        )
        vorticity_divg_l2_norm = np.sqrt(vorticity_divg_l2_norm_sq) * self.dx**1.5
        return vorticity_divg_l2_norm

    def get_max_vorticity(self):
        """Compute maximum vorticity"""
        gs = self.ghost_size
        max_vort_local = np.amax(self.vorticity_field[:, gs:-gs, gs:-gs, gs:-gs])
        max_vort = self.mpi_construct.grid.allreduce(max_vort_local, op=MPI.MAX)
        return max_vort
