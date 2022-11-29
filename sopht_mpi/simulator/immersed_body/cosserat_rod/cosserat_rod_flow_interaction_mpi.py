__all__ = ["CosseratRodFlowInteraction"]
from elastica.rod.cosserat_rod import CosseratRod
import numpy as np
from sopht_mpi.simulator.immersed_body import (
    ImmersedBodyForcingGrid,
    EmptyForcingGrid,
    ImmersedBodyFlowInteractionMPI,
)


class CosseratRodFlowInteraction(ImmersedBodyFlowInteractionMPI):
    """Class for Cosserat rod flow interaction."""

    def __init__(
        self,
        mpi_construct,
        mpi_ghost_exchange_communicator,
        cosserat_rod: type(CosseratRod),
        eul_grid_forcing_field,
        eul_grid_velocity_field,
        virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff,
        dx,
        grid_dim,
        forcing_grid_cls: type(ImmersedBodyForcingGrid),
        real_t=np.float64,
        eul_grid_coord_shift=None,
        interp_kernel_width=None,
        enable_eul_grid_forcing_reset=False,
        start_time=0.0,
        master_rank=0,
        **forcing_grid_kwargs,
    ):
        """Class initialiser."""
        self.body_flow_forces = np.zeros(
            (3, cosserat_rod.n_elems + 1),
        )
        self.body_flow_torques = np.zeros(
            (3, cosserat_rod.n_elems),
        )
        # Initialize forcing grid based on the master owning the immersed body
        self.master_rank = master_rank
        if mpi_construct.rank == self.master_rank:
            self.forcing_grid = forcing_grid_cls(
                grid_dim=grid_dim,
                cosserat_rod=cosserat_rod,
                real_t=real_t,
                **forcing_grid_kwargs,
            )
        else:
            self.forcing_grid = EmptyForcingGrid(grid_dim=grid_dim, real_t=real_t)

        # initialising super class
        super().__init__(
            mpi_construct=mpi_construct,
            mpi_ghost_exchange_communicator=mpi_ghost_exchange_communicator,
            eul_grid_forcing_field=eul_grid_forcing_field,
            eul_grid_velocity_field=eul_grid_velocity_field,
            virtual_boundary_stiffness_coeff=virtual_boundary_stiffness_coeff,
            virtual_boundary_damping_coeff=virtual_boundary_damping_coeff,
            dx=dx,
            grid_dim=grid_dim,
            real_t=real_t,
            eul_grid_coord_shift=eul_grid_coord_shift,
            interp_kernel_width=interp_kernel_width,
            enable_eul_grid_forcing_reset=enable_eul_grid_forcing_reset,
            start_time=start_time,
        )
