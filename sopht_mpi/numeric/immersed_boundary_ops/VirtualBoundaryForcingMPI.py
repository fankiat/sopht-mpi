"""MPI-supported virtual boundary forcing for flow-body feedback."""
from numba import njit
import numpy as np
from sopht.numeric.eulerian_grid_ops.stencil_ops_2d.elementwise_ops_2d import (
    gen_set_fixed_val_pyst_kernel_2d,
)

# from sopht.numeric.eulerian_grid_ops.stencil_ops_3d.elementwise_ops_3d import (
#     gen_set_fixed_val_pyst_kernel_3d,
# )
from sopht_mpi.numeric.immersed_boundary_ops.EulerianLagrangianGridCommunicatorMPI2D import (
    EulerianLagrangianGridCommunicatorMPI2D,
)

# from sopht.numeric.immersed_boundary_ops.EulerianLagrangianGridCommunicator3D import (
#     EulerianLagrangianGridCommunicator3D,
# )
from sopht_mpi.utils.mpi_utils_2d import MPILagrangianFieldCommunicator2D
from mpi4py import MPI


class VirtualBoundaryForcingMPI:
    """
    Class for MPI-supported virtual boundary forcing.

    Virtual boundary forcing class for computing feedback between the
    Lagrangian body and Eulerian grid flow, using virtual boundary method
    Refer to Goldstein 1993, JCP for details on the penalty force computation.

    Note: For clarify and explicit naming, buffer variables here are prefixed with
    'local_*' and 'global_*' to indicate mpi local and global variables, respectively.
    """

    def __init__(
        self,
        mpi_construct,
        ghost_size,
        virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff,
        grid_dim,
        dx,
        real_t,
        eul_grid_coord_shift=None,
        interp_kernel_width=None,
        enable_eul_grid_forcing_reset=True,
        start_time=0.0,
        master_rank=0,
        global_lag_grid_position_field=None,
        moving_boundary=True,
    ):
        """Class initialiser.

        Input arguments:
        virtual_boundary_stiffness_coeff: stiffness coefficient for computing penalty
        force, set to a high values
        virtual_boundary_damping_coeff: damping coefficient for computing penalty
        force, added for stabilising force
        grid_dim: dimensions of the grid
        dx: Eulerian grid spacing
        eul_grid_coord_shift: shift of the coordinates of the Eulerian grid start from
        0 (usually dx / 2)
        global_num_lag_nodes: total number of Lagrangian grid nodes
        interp_kernel_width: width of interpolation kernel
        real_t: numerical precision
        enable_eul_grid_forcing_reset : flag for enabling option of feedback step
        with resetting of eul_grid_forcing_field
        start_time: start time of the forcing
        master_rank: rank that owns all the lagrangian nodes and serves as main comm hub
        global_lag_grid_position_field: initial lagrangian grid position
        moving_boundary: flag for moving boundary, used in buffer reinitialization

        """
        if grid_dim != 2 and grid_dim != 3:
            raise ValueError("Invalid grid dimensions for virtual boundary forcing!")
        self.grid_dim = grid_dim
        self.virtual_boundary_stiffness_coeff = virtual_boundary_stiffness_coeff
        self.virtual_boundary_damping_coeff = virtual_boundary_damping_coeff
        self.time = start_time
        self.real_t = real_t
        self.moving_boundary = moving_boundary

        # these are rather invariant hence pushed to fixed kwargs
        if eul_grid_coord_shift is None:
            eul_grid_coord_shift = real_t(dx / 2)
        self.interp_kernel_width = interp_kernel_width
        if interp_kernel_width is None:
            self.interp_kernel_width = 2
        self.ghost_size = ghost_size

        if self.interp_kernel_width > ghost_size:
            raise ValueError(
                f"Field ghost size {ghost_size} needs to be larger than "
                f"interpolation kernel width {self.interp_kernel_width}"
            )

        # Initialize MPI related stuff
        self.mpi_construct = mpi_construct
        self.mpi_lagrangian_field_communicator = MPILagrangianFieldCommunicator2D(
            eul_grid_dx=dx,
            eul_grid_coord_shift=eul_grid_coord_shift,
            mpi_construct=self.mpi_construct,
            master_rank=master_rank,
            real_t=real_t,
        )
        self.mpi_lagrangian_field_communicator.map_lagrangian_nodes_based_on_position(
            global_lag_positions=global_lag_grid_position_field
        )
        self.local_num_lag_nodes = (
            self.mpi_lagrangian_field_communicator.local_num_lag_nodes
        )
        self.global_num_lag_nodes = (
            self.mpi_lagrangian_field_communicator.rank_address.shape[-1]
        )

        # creating buffers...
        self._init_local_buffers(self.local_num_lag_nodes)
        self._init_global_buffers()

        if grid_dim == 2:
            self.eul_lag_grid_communicator = EulerianLagrangianGridCommunicatorMPI2D(
                dx=dx,
                eul_grid_coord_shift=eul_grid_coord_shift,
                interp_kernel_width=self.interp_kernel_width,
                real_t=real_t,
                n_components=grid_dim,
                mpi_construct=mpi_construct,
                ghost_size=ghost_size,
            )
        # elif grid_dim == 3:
        #     self.eul_lag_grid_communicator = EulerianLagrangianGridCommunicatorMPI3D(
        #         dx=dx,
        #         eul_grid_coord_shift=eul_grid_coord_shift,
        #         interp_kernel_width=self.interp_kernel_width,
        #         real_t=real_t,
        #         n_components=grid_dim,
        #         mpi_construct=mpi_construct,
        #         ghost_size=ghost_size,
        #     )

        if enable_eul_grid_forcing_reset:
            if grid_dim == 2:
                self.set_eul_grid_vector_field = gen_set_fixed_val_pyst_kernel_2d(
                    real_t=real_t,
                    field_type="vector",
                )

            # elif grid_dim == 3:
            #     self.set_eul_grid_vector_field = gen_set_fixed_val_pyst_kernel_3d(
            #         real_t=real_t,
            #         field_type="vector",
            #     )
            self.compute_interaction_forcing = (
                self.compute_interaction_force_on_eul_and_lag_grid_with_eul_grid_forcing_reset
            )
        else:
            self.compute_interaction_forcing = (
                self.compute_interaction_force_on_eul_and_lag_grid
            )

    def _init_global_buffers(self):
        self.global_lag_grid_position_mismatch_field = np.zeros(
            (self.grid_dim, self.global_num_lag_nodes), dtype=self.real_t
        )
        self.global_lag_grid_velocity_mismatch_field = np.zeros_like(
            self.global_lag_grid_position_mismatch_field
        )
        self.global_lag_grid_forcing_field = np.zeros_like(
            self.global_lag_grid_position_mismatch_field
        )

    def _init_local_buffers(self, num_lag_nodes):
        self.local_nearest_eul_grid_index_to_lag_grid = np.empty(
            (self.grid_dim, num_lag_nodes), dtype=int
        )
        eul_grid_support_of_lag_grid_shape = (
            (self.grid_dim,)
            + (2 * self.interp_kernel_width,) * self.grid_dim
            + (num_lag_nodes,)
        )
        self.local_local_eul_grid_support_of_lag_grid = np.empty(
            eul_grid_support_of_lag_grid_shape, dtype=self.real_t
        )
        interp_weights_shape = (2 * self.interp_kernel_width,) * self.grid_dim + (
            num_lag_nodes,
        )
        self.local_interp_weights = np.empty(interp_weights_shape, dtype=self.real_t)
        self.local_lag_grid_flow_velocity_field = np.zeros(
            (self.grid_dim, num_lag_nodes), dtype=self.real_t
        )
        self.local_lag_grid_position_mismatch_field = np.zeros_like(
            self.local_lag_grid_flow_velocity_field
        )
        self.local_lag_grid_velocity_mismatch_field = np.zeros_like(
            self.local_lag_grid_position_mismatch_field
        )
        self.local_lag_grid_forcing_field = np.zeros_like(
            self.local_lag_grid_velocity_mismatch_field
        )
        self.local_lag_grid_position_field = np.zeros_like(
            self.local_lag_grid_position_mismatch_field
        )
        self.local_lag_grid_velocity_field = np.zeros_like(
            self.local_lag_grid_position_field
        )

    def update_buffers(self, global_lag_grid_position_field):
        # First gather the local mismatch fields for storing and redistribution later
        # This needs to be done before remapping.
        self.mpi_lagrangian_field_communicator.gather_local_field(
            global_lag_field=self.global_lag_grid_position_mismatch_field,
            local_lag_field=self.local_lag_grid_position_mismatch_field[
                ..., : self.local_num_lag_nodes
            ],
        )
        self.mpi_lagrangian_field_communicator.gather_local_field(
            global_lag_field=self.global_lag_grid_velocity_mismatch_field,
            local_lag_field=self.local_lag_grid_velocity_mismatch_field[
                ..., : self.local_num_lag_nodes
            ],
        )
        # Re-map the lagrangian nodes
        self.mpi_lagrangian_field_communicator.map_lagrangian_nodes_based_on_position(
            global_lag_grid_position_field
        )
        # Check if local buffers needs to be updated
        update_buffer_flag = (
            self.local_num_lag_nodes
            != self.mpi_lagrangian_field_communicator.local_num_lag_nodes
        )
        update_buffer_flag = self.mpi_construct.grid.allreduce(
            update_buffer_flag, op=MPI.LOR
        )
        # Reinitialize buffers if needed
        if update_buffer_flag:
            # Initialize buffer with new local size
            self.local_num_lag_nodes = (
                self.mpi_lagrangian_field_communicator.local_num_lag_nodes
            )
            self._init_local_buffers(self.local_num_lag_nodes)
            # Scatter the global mismatch fields correspondingly
            self.mpi_lagrangian_field_communicator.scatter_global_field(
                local_lag_field=self.local_lag_grid_position_mismatch_field,
                global_lag_field=self.global_lag_grid_position_mismatch_field,
            )
            self.mpi_lagrangian_field_communicator.scatter_global_field(
                local_lag_field=self.local_lag_grid_velocity_mismatch_field,
                global_lag_field=self.global_lag_grid_velocity_mismatch_field,
            )

    @staticmethod
    @njit(cache=True, fastmath=True)
    def compute_lag_grid_velocity_mismatch_field(
        lag_grid_velocity_mismatch_field,
        lag_grid_flow_velocity_field,
        lag_grid_body_velocity_field,
    ):
        """Compute Lagrangian grid velocity mismatch, between the flow and body.

        We can use pystencils for this but seems like it will be O(N) work, and wont be
        the limiter at least for few rods.

        """
        lag_grid_velocity_mismatch_field[...] = (
            lag_grid_flow_velocity_field - lag_grid_body_velocity_field
        )

    @staticmethod
    @njit(cache=True, fastmath=True)
    def update_lag_grid_position_mismatch_field_via_euler_forward(
        lag_grid_position_mismatch_field,
        lag_grid_velocity_mismatch_field,
        dt,
    ):
        """Update Lagrangian grid position mismatch, between the flow and body.

        We can use pystencils for this but seems like it will be O(N) work, and wont be
        the limiter at least for few rods.

        """
        lag_grid_position_mismatch_field[...] = (
            lag_grid_position_mismatch_field + dt * lag_grid_velocity_mismatch_field
        )

    @staticmethod
    @njit(cache=True, fastmath=True)
    def compute_lag_grid_forcing_field(
        lag_grid_forcing_field,
        lag_grid_position_mismatch_field,
        lag_grid_velocity_mismatch_field,
        virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff,
    ):
        """Compute penalty force on Lagrangian grid, defined via virtual boundary method.

        Refer to Goldstein 1993, JCP for details on the penalty force computation.
        We can use pystencils for this but seems like it will be O(N) work, and wont be
        the limiter at least for few rods.

        """
        lag_grid_forcing_field[...] = (
            virtual_boundary_stiffness_coeff * lag_grid_position_mismatch_field
            + virtual_boundary_damping_coeff * lag_grid_velocity_mismatch_field
        )

    def compute_interaction_force_on_lag_grid(
        self,
        local_eul_grid_velocity_field,
        global_lag_grid_position_field,
        global_lag_grid_velocity_field,
    ):
        """Virtual boundary: compute interaction force on Lagrangian grid."""
        # Start by checking and mapping global lag grid fields data
        # Check and make sure buffer is enough to store all local lagrangian quantities
        if self.moving_boundary:
            self.update_buffers(
                global_lag_grid_position_field=global_lag_grid_position_field
            )
        # Distribute the global lag grid fields
        self.mpi_lagrangian_field_communicator.scatter_global_field(
            local_lag_field=self.local_lag_grid_position_field,
            global_lag_field=global_lag_grid_position_field,
        )
        self.mpi_lagrangian_field_communicator.scatter_global_field(
            local_lag_field=self.local_lag_grid_velocity_field,
            global_lag_field=global_lag_grid_velocity_field,
        )

        # Start working on the local chunks of data
        # 1. Find Eulerian grid local support of the Lagrangian grid
        self.eul_lag_grid_communicator.local_eulerian_grid_support_of_lagrangian_grid_kernel(
            local_eul_grid_support_of_lag_grid=self.local_local_eul_grid_support_of_lag_grid,
            nearest_eul_grid_index_to_lag_grid=self.local_nearest_eul_grid_index_to_lag_grid,
            lag_positions=self.local_lag_grid_position_field,
        )

        # 2. Compute interpolation weights based on local Eulerian grid support
        self.eul_lag_grid_communicator.interpolation_weights_kernel(
            interp_weights=self.local_interp_weights,
            local_eul_grid_support_of_lag_grid=self.local_local_eul_grid_support_of_lag_grid,
        )

        # 3. Interpolate Eulerian flow velocity onto the Lagrangian grid
        self.eul_lag_grid_communicator.eulerian_to_lagrangian_grid_interpolation_kernel(
            lag_grid_field=self.local_lag_grid_flow_velocity_field,
            eul_grid_field=local_eul_grid_velocity_field,
            interp_weights=self.local_interp_weights,
            nearest_eul_grid_index_to_lag_grid=self.local_nearest_eul_grid_index_to_lag_grid,
        )

        # 4. Compute velocity mismatch between flow and body on Lagrangian grid
        self.compute_lag_grid_velocity_mismatch_field(
            lag_grid_velocity_mismatch_field=self.local_lag_grid_velocity_mismatch_field,
            lag_grid_flow_velocity_field=self.local_lag_grid_flow_velocity_field,
            lag_grid_body_velocity_field=self.local_lag_grid_velocity_field,
        )

        # 5. Compute penalty force using virtual boundary forcing formulation
        # on Lagrangian grid
        self.compute_lag_grid_forcing_field(
            lag_grid_forcing_field=self.local_lag_grid_forcing_field,
            lag_grid_position_mismatch_field=self.local_lag_grid_position_mismatch_field,
            lag_grid_velocity_mismatch_field=self.local_lag_grid_velocity_mismatch_field,
            virtual_boundary_stiffness_coeff=self.virtual_boundary_stiffness_coeff,
            virtual_boundary_damping_coeff=self.virtual_boundary_damping_coeff,
        )

    def compute_interaction_force_on_eul_and_lag_grid(
        self,
        local_eul_grid_forcing_field,
        local_eul_grid_velocity_field,
        global_lag_grid_position_field,
        global_lag_grid_velocity_field,
    ):
        """Virtual boundary: compute interaction on Eulerian grid."""
        # 1. Compute penalty force using virtual boundary forcing formulation
        # on Lagrangian grid
        self.compute_interaction_force_on_lag_grid(
            local_eul_grid_velocity_field,
            global_lag_grid_position_field,
            global_lag_grid_velocity_field,
        )
        # 2. Interpolate penalty forcing from Lagrangian onto the Eulerian grid
        self.eul_lag_grid_communicator.lagrangian_to_eulerian_grid_interpolation_kernel(
            eul_grid_field=local_eul_grid_forcing_field,
            lag_grid_field=self.local_lag_grid_forcing_field[
                ..., : self.local_num_lag_nodes
            ],
            interp_weights=self.local_interp_weights[..., : self.local_num_lag_nodes],
            nearest_eul_grid_index_to_lag_grid=self.local_nearest_eul_grid_index_to_lag_grid[
                ..., : self.local_num_lag_nodes
            ],
        )

    def compute_interaction_force_on_eul_and_lag_grid_with_eul_grid_forcing_reset(
        self,
        local_eul_grid_forcing_field,
        local_eul_grid_velocity_field,
        global_lag_grid_position_field,
        global_lag_grid_velocity_field,
    ):
        """Virtual boundary: compute interaction on Eulerian grid.

        Resets eul_grid_forcing_field.
        """
        self.set_eul_grid_vector_field(
            vector_field=local_eul_grid_forcing_field, fixed_vals=([0] * self.grid_dim)
        )
        self.compute_interaction_force_on_eul_and_lag_grid(
            local_eul_grid_forcing_field,
            local_eul_grid_velocity_field,
            global_lag_grid_position_field,
            global_lag_grid_velocity_field,
        )

    def time_step(self, dt):
        """Virtual boundary forcing time step, updates grid deviation."""
        self.update_lag_grid_position_mismatch_field_via_euler_forward(
            lag_grid_position_mismatch_field=self.local_lag_grid_position_mismatch_field[
                ..., : self.local_num_lag_nodes
            ],
            lag_grid_velocity_mismatch_field=self.local_lag_grid_velocity_mismatch_field[
                ..., : self.local_num_lag_nodes
            ],
            dt=dt,
        )
        self.time += dt
