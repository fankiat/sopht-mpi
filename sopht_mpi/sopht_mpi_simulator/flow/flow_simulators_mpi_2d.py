import numpy as np
from sopht_mpi.numeric.eulerian_grid_ops import (
    gen_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_2d,
    gen_diffusion_timestep_euler_forward_pyst_mpi_kernel_2d,
    gen_penalise_field_boundary_pyst_mpi_kernel_2d,
    gen_outplane_field_curl_pyst_mpi_kernel_2d,
    gen_update_vorticity_from_velocity_forcing_pyst_mpi_kernel_2d,
    UnboundedPoissonSolverMPI2D,
)
from sopht_mpi.utils import MPIConstruct2D, MPIGhostCommunicator2D
from sopht_mpi.sopht_mpi_simulator.utils.grid_utils import VectorField
from sopht.numeric.eulerian_grid_ops import (
    gen_add_fixed_val_pyst_kernel_2d,
    gen_set_fixed_val_pyst_kernel_2d,
)
from sopht.utils.precision import get_test_tol
from mpi4py import MPI


class UnboundedFlowSimulator2D:
    """Class for MPI-supported 2D unbounded flow simulator"""

    def __init__(
        self,
        grid_size,
        x_range,
        kinematic_viscosity,
        CFL=0.1,
        flow_type="passive_scalar",
        with_free_stream_flow=False,
        real_t=np.float32,
        rank_distribution=None,
        ghost_size=2,
        **kwargs,
    ):
        """Class initialiser

        :param grid_size: Grid size of simulator
        :param x_range: Range of X coordinate of the grid
        :param kinematic_viscosity: kinematic viscosity of the fluid
        :param CFL: Courant Freidrich Lewy number (advection timestep)
        :param flow_type: Nature of the simulator, can be "passive_scalar" (default value),
        "navier_stokes" or "navier_stokes_with_forcing"
        :param with_free_stream_flow: has free stream flow or not
        :param real_t: precision of the solver
        :param rank_distribution: distribution configuration of the grid

        Notes
        -----
        Currently only supports Euler forward timesteps :(
        """
        self.grid_dim = 2
        self.grid_size = grid_size
        self.grid_size_y, self.grid_size_x = self.grid_size
        self.x_range = x_range
        self.real_t = real_t
        self.flow_type = flow_type
        self.with_free_stream_flow = with_free_stream_flow
        self.kinematic_viscosity = kinematic_viscosity
        self.CFL = CFL
        supported_flow_types = [
            "passive_scalar",
            "navier_stokes",
            "navier_stokes_with_forcing",
        ]
        if self.flow_type not in supported_flow_types:
            raise ValueError("Invalid flow type given")
        if self.flow_type == "passive_scalar" and self.with_free_stream_flow:
            raise ValueError(
                "Free stream flow not defined for passive advection diffusion!"
            )

        # MPI-related variables
        self.rank_distribution = rank_distribution
        self.ghost_size = ghost_size

        self.init_mpi()
        self.init_domain()
        self.init_fields()
        if self.flow_type in ["navier_stokes", "navier_stokes_with_forcing"]:
            if "penalty_zone_width" in kwargs:
                self.penalty_zone_width = kwargs.get("penalty_zone_width")
            else:
                self.penalty_zone_width = 2
        try:
            self.compile_kernels()
        except Exception as error:
            print("Error with compiling kernels for simulator!")
            print(f"{type(error).__name__}: " + str(error))
        self.finalise_flow_timestep()

    def init_mpi(self):
        self.mpi_construct = MPIConstruct2D(
            grid_size_y=self.grid_size_y,
            grid_size_x=self.grid_size_x,
            real_t=self.real_t,
            rank_distribution=self.rank_distribution,
        )
        self.mpi_ghost_exchange_communicator = MPIGhostCommunicator2D(
            ghost_size=self.ghost_size, mpi_construct=self.mpi_construct
        )

    def init_domain(self):
        """Initialize the MPI local domain (with ghost cells)"""
        self.y_range = self.x_range * self.grid_size_y / self.grid_size_x
        self.dx = self.real_t(self.x_range / self.grid_size_x)
        eul_grid_shift = self.dx / 2.0
        ghost_grid_shift = self.ghost_size * self.dx

        # Generate grid meshes for each corresponding rank based on local coords
        local_grid_size = self.mpi_construct.local_grid_size
        substart_idx = self.mpi_construct.grid.coords * local_grid_size
        subend_idx = substart_idx + local_grid_size
        substart_y, substart_x = substart_idx * self.dx
        subend_y, subend_x = subend_idx * self.dx
        local_grid_size_y, local_grid_size_x = local_grid_size
        local_x = np.linspace(
            eul_grid_shift + substart_x - ghost_grid_shift,
            subend_x - eul_grid_shift + ghost_grid_shift,
            local_grid_size_x + self.grid_dim * self.ghost_size,
        ).astype(self.real_t)
        local_y = np.linspace(
            eul_grid_shift + substart_y - ghost_grid_shift,
            subend_y - eul_grid_shift + ghost_grid_shift,
            local_grid_size_y + self.grid_dim * self.ghost_size,
        ).astype(self.real_t)
        # flipud so that position field are ordered according to VectorField convention
        self.position_field = np.flipud(np.meshgrid(local_y, local_x, indexing="ij"))

        # TODO: (logger) refactor this after implementing a mpi-supported logger
        if self.mpi_construct.rank == 0:
            print(
                "==============================================="
                f"\n{self.grid_dim}D flow domain initialized with:"
                f"\nX axis from 0.0 to {self.x_range}"
                f"\nY axis from 0.0 to {self.y_range}"
                "\nPlease initialize bodies within these bounds!"
                "\n==============================================="
            )

    def init_fields(self):
        """Initialize the necessary field arrays, i.e. vorticity, velocity, etc."""
        # Initialize flow field
        self.primary_scalar_field = np.zeros(self.grid_size, dtype=self.real_t)
        self.velocity_field = np.zeros(
            (self.grid_dim, *self.grid_size), dtype=self.real_t
        )
        # we use the same buffer for advection, diffusion and velocity recovery
        self.buffer_scalar_field = np.zeros_like(self.primary_scalar_field)

        if self.flow_type in ["navier_stokes", "navier_stokes_with_forcing"]:
            self.vorticity_field = self.primary_scalar_field.view()
            self.stream_func_field = np.zeros_like(self.vorticity_field)
        if self.flow_type == "navier_stokes_with_forcing":
            # this one holds the forcing from bodies
            self.eul_grid_forcing_field = np.zeros_like(self.velocity_field)

    def compile_kernels(self):
        """Compile necessary kernels based on flow type"""
        self.diffusion_timestep = (
            gen_diffusion_timestep_euler_forward_pyst_mpi_kernel_2d(
                real_t=self.real_t,
                mpi_construct=self.mpi_construct,
                ghost_exchange_communicator=self.mpi_ghost_exchange_communicator,
            )
        )
        self.advection_timestep = (
            gen_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_2d(
                real_t=self.real_t,
                mpi_construct=self.mpi_construct,
                ghost_exchange_communicator=self.mpi_ghost_exchange_communicator,
            )
        )

        if self.flow_type in ["navier_stokes", "navier_stokes_with_forcing"]:
            self.unbounded_poisson_solver = UnboundedPoissonSolverMPI2D(
                grid_size_y=self.grid_size_y,
                grid_size_x=self.grid_size_x,
                x_range=self.x_range,
                real_t=self.real_t,
                mpi_construct=self.mpi_construct,
                ghost_size=self.ghost_size,
            )
            self.curl = gen_outplane_field_curl_pyst_mpi_kernel_2d(
                real_t=self.real_t,
                mpi_construct=self.mpi_construct,
                ghost_exchange_communicator=self.mpi_ghost_exchange_communicator,
            )
            self.penalise_field_towards_boundary = (
                gen_penalise_field_boundary_pyst_mpi_kernel_2d(
                    width=self.penalty_zone_width,
                    dx=self.dx,
                    x_grid_field=self.position_field[VectorField.x_axis_idx()],
                    y_grid_field=self.position_field[VectorField.y_axis_idx()],
                    real_t=self.real_t,
                    mpi_construct=self.mpi_construct,
                    ghost_exchange_communicator=self.mpi_ghost_exchange_communicator,
                )
            )

        if self.flow_type == "navier_stokes_with_forcing":
            self.update_vorticity_from_velocity_forcing = (
                gen_update_vorticity_from_velocity_forcing_pyst_mpi_kernel_2d(
                    real_t=self.real_t,
                    mpi_construct=self.mpi_construct,
                    ghost_exchange_communicator=self.mpi_ghost_exchange_communicator,
                )
            )
            self.set_field = gen_set_fixed_val_pyst_kernel_2d(
                real_t=self.real_t,
                field_type="vector",
            )
        # free stream velocity stuff
        if self.with_free_stream_flow:
            add_fixed_val = gen_add_fixed_val_pyst_kernel_2d(
                real_t=self.real_t,
                field_type="vector",
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

    def finalise_flow_timestep(self):
        # default time step
        self.time_step = self.advection_and_diffusion_timestep

        if self.flow_type == "navier_stokes":
            self.time_step = self.navier_stokes_timestep
        elif self.flow_type == "navier_stokes_with_forcing":
            self.time_step = self.navier_stokes_with_forcing_timestep

    def advection_and_diffusion_timestep(self, dt, **kwargs):
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

    def compute_velocity_from_vorticity(self):
        self.penalise_field_towards_boundary(field=self.vorticity_field)
        self.unbounded_poisson_solver.solve(
            solution_field=self.stream_func_field, rhs_field=self.vorticity_field
        )
        self.curl(
            curl=self.velocity_field,
            field=self.stream_func_field,
            prefactor=self.real_t(0.5 / self.dx),
        )

    def navier_stokes_timestep(self, dt, free_stream_velocity=(0.0, 0.0)):
        self.advection_and_diffusion_timestep(dt=dt)
        self.compute_velocity_from_vorticity()
        self.update_velocity_with_free_stream(free_stream_velocity=free_stream_velocity)

    def navier_stokes_with_forcing_timestep(self, dt, free_stream_velocity=(0.0, 0.0)):
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
        velocity_mag_field[...] = np.sum(np.fabs(self.velocity_field), axis=0)
        dt = min(
            self.CFL
            * self.dx
            / (
                np.amax(
                    velocity_mag_field[
                        self.ghost_size : -self.ghost_size,
                        self.ghost_size : -self.ghost_size,
                    ]
                )
                + get_test_tol(precision)
            ),
            0.9 * self.dx**2 / (2 * self.grid_dim) / (self.kinematic_viscosity),
        )
        # Get smallest timestep among all the ranks
        dt = self.mpi_construct.grid.allreduce(dt, op=MPI.MIN)
        return dt * dt_prefac

    def get_max_vorticity(self):
        """Compute maximum vorticity"""
        gs = self.ghost_size
        max_vort_local = np.amax(self.vorticity_field[gs:-gs, gs:-gs])
        max_vort = self.mpi_construct.grid.allreduce(max_vort_local, op=MPI.MAX)
        return max_vort
