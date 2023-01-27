import numpy as np
from sopht_mpi.numeric.immersed_boundary_ops import VirtualBoundaryForcingMPI
from sopht_mpi.simulator.immersed_body.immersed_body_forcing_grid import (
    ImmersedBodyForcingGrid,
)
from mpi4py import MPI
from sopht_mpi.utils import logger


class ImmersedBodyFlowInteractionMPI(VirtualBoundaryForcingMPI):
    """Base class for immersed body flow interaction."""

    # These are meant to be initialised in the derived classes
    body_flow_forces: np.ndarray
    body_flow_torques: np.ndarray
    forcing_grid: type(ImmersedBodyForcingGrid)
    master_rank: int

    def __init__(
        self,
        mpi_construct,
        mpi_ghost_exchange_communicator,
        eul_grid_forcing_field,
        eul_grid_velocity_field,
        virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff,
        dx,
        grid_dim,
        eul_grid_coord_shift=None,
        interp_kernel_width=None,
        enable_eul_grid_forcing_reset=False,
        start_time=0.0,
        assume_data_locality=False,
        auto_ghosting=True,
    ):
        """Class initialiser."""
        # ghost exchange communicator for ghosting vel field before computing interactor
        self.mpi_ghost_exchange_communicator = mpi_ghost_exchange_communicator

        # these hold references to Eulerian fields
        self.eul_grid_forcing_field = eul_grid_forcing_field.view()
        self.eul_grid_velocity_field = eul_grid_velocity_field.view()
        # this class should only "view" the flow velocity
        self.eul_grid_velocity_field.flags.writeable = False

        # check relative resolutions of the Lagrangian and Eulerian grids
        max_lag_grid_dx = self.forcing_grid.get_maximum_lagrangian_grid_spacing()
        max_lag_grid_dx = mpi_construct.grid.bcast(
            max_lag_grid_dx, root=self.master_rank
        )
        grid_type = type(self.forcing_grid).__name__
        logger.warning(
            "==========================================================\n"
            f"For {grid_type}:"
        )
        if (
            max_lag_grid_dx > 2 * dx
        ):  # 2 here since the support of delta function is 2 grid points
            logger.warning(
                f"Eulerian grid spacing (dx): {dx}"
                f"\nMax Lagrangian grid spacing: {max_lag_grid_dx} > 2 * dx"
                "\nThe Lagrangian grid of the body is too coarse relative to"
                "\nthe Eulerian grid of the flow, which can lead to unexpected"
                "\nconvergence. Please make the Lagrangian grid finer."
            )
        elif max_lag_grid_dx < 0.5 * dx:  # reverse case of the above condition
            logger.warning(
                "==========================================================\n"
                f"Eulerian grid spacing (dx): {dx}"
                f"\nMax Lagrangian grid spacing: {max_lag_grid_dx} < 0.5 * dx"
                "\nThe Lagrangian grid of the body is too fine relative to"
                "\nthe Eulerian grid of the flow, which corresponds to redundant"
                "\nforcing points. Please make the Lagrangian grid coarser."
            )
        else:
            logger.warning(
                "Lagrangian grid is resolved almost the same as the Eulerian"
                "\ngrid of the flow."
            )
        logger.warning("==========================================================")

        # rescale the virtual boundary coeffs by grid spacings
        # (based on previous penalty immersed boundary method works)
        virtual_boundary_stiffness_coeff *= max_lag_grid_dx ** (grid_dim - 1)
        virtual_boundary_damping_coeff *= max_lag_grid_dx ** (grid_dim - 1)

        # initialising super class
        super().__init__(
            mpi_construct=mpi_construct,
            ghost_size=self.mpi_ghost_exchange_communicator.ghost_size,
            virtual_boundary_stiffness_coeff=virtual_boundary_stiffness_coeff,
            virtual_boundary_damping_coeff=virtual_boundary_damping_coeff,
            grid_dim=grid_dim,
            dx=dx,
            eul_grid_coord_shift=eul_grid_coord_shift,
            interp_kernel_width=interp_kernel_width,
            enable_eul_grid_forcing_reset=enable_eul_grid_forcing_reset,
            start_time=start_time,
            master_rank=self.master_rank,
            global_lag_grid_position_field=self.forcing_grid.position_field,
            assume_data_locality=assume_data_locality,
        )

        if auto_ghosting:
            self.compute_full_interaction = self._compute_full_interaction_with_ghosting
            self.compute_interaction_on_lag_grid = (
                self._compute_interaction_on_lag_grid_with_ghosting
            )
        else:
            logger.warning(
                "==========================================================\n"
                "Auto ghosting of velocity field is disabled for interactor.\n"
                "Please ensure ghosting is done before calling interactor functions.\n"
                "=========================================================="
            )
            self.compute_full_interaction = (
                self._compute_full_interaction_without_ghosting
            )
            self.compute_interaction_on_lag_grid = (
                self._compute_interaction_on_lag_grid_without_ghosting
            )

    def __call__(self):
        self.compute_full_interaction()

    def _ghost_velocity_field_for_interaction(self):
        """Ghost eulerian grid velocity field before computing interaction"""
        self.eul_grid_velocity_field.flags.writeable = True
        self.mpi_ghost_exchange_communicator.exchange_vector_field_init(
            self.eul_grid_velocity_field
        )
        self.mpi_ghost_exchange_communicator.exchange_finalise()
        self.eul_grid_velocity_field.flags.writeable = False

    def _compute_interaction_on_lag_grid_without_ghosting(self):
        """Compute interaction forces on the Lagrangian forcing grid."""
        # 1. Compute forcing grid position and velocity
        self.forcing_grid.compute_lag_grid_position_field()
        self.forcing_grid.compute_lag_grid_velocity_field()
        # 2. Compute interaction forcing
        self.compute_interaction_force_on_lag_grid(
            local_eul_grid_velocity_field=self.eul_grid_velocity_field,
            global_lag_grid_position_field=self.forcing_grid.position_field,
            global_lag_grid_velocity_field=self.forcing_grid.velocity_field,
        )

    def _compute_interaction_on_lag_grid_with_ghosting(self):
        self._ghost_velocity_field_for_interaction()
        self._compute_interaction_on_lag_grid_without_ghosting()

    def _compute_full_interaction_without_ghosting(self):
        """Call the full interaction (eul and lag field force computation)"""
        # 1. Compute forcing grid position and velocity
        self.forcing_grid.compute_lag_grid_position_field()
        self.forcing_grid.compute_lag_grid_velocity_field()
        # 3. Compute interaction forcing
        self.compute_interaction_forcing(
            local_eul_grid_forcing_field=self.eul_grid_forcing_field,
            local_eul_grid_velocity_field=self.eul_grid_velocity_field,
            global_lag_grid_position_field=self.forcing_grid.position_field,
            global_lag_grid_velocity_field=self.forcing_grid.velocity_field,
        )

    def _compute_full_interaction_with_ghosting(self):
        self._ghost_velocity_field_for_interaction()
        self._compute_full_interaction_without_ghosting()

    def compute_flow_forces_and_torques(self):
        """Compute flow forces and torques on the body from forces on Lagrangian grid."""
        self.compute_interaction_on_lag_grid()
        self.forcing_grid.transfer_forcing_from_grid_to_body(
            body_flow_forces=self.body_flow_forces,
            body_flow_torques=self.body_flow_torques,
            lag_grid_forcing_field=self.global_lag_grid_forcing_field,
        )

    def get_grid_deviation_error_l2_norm(self, compute_global=True):
        """
        Computes and returns L2 norm of deviation error between flow
        and body grids.
        """
        if not compute_global:
            local_grid_dev_error_l2_norm = np.linalg.norm(
                self.local_lag_grid_position_mismatch_field
            ) / np.sqrt(self.forcing_grid.num_lag_nodes)
            return local_grid_dev_error_l2_norm
        else:
            local_grid_dev_error_l2_norm = (
                np.linalg.norm(self.local_lag_grid_position_mismatch_field) ** 2
            )
            grid_dev_error_l2_norm = self.mpi_construct.grid.reduce(
                local_grid_dev_error_l2_norm, op=MPI.SUM, root=self.master_rank
            )
            if self.mpi_construct.rank == self.master_rank:
                grid_dev_error_l2_norm = np.sqrt(grid_dev_error_l2_norm) / np.sqrt(
                    self.forcing_grid.num_lag_nodes
                )
            grid_dev_error_l2_norm = self.mpi_construct.grid.bcast(
                grid_dev_error_l2_norm, root=self.master_rank
            )

            return grid_dev_error_l2_norm
