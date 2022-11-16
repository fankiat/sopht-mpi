__all__ = ["ImmersedBodyForcingGrid", "EmptyForcingGrid"]
from abc import ABC, abstractmethod
import logging
import numpy as np
from sopht_mpi.utils import logger


class ImmersedBodyForcingGrid(ABC):
    """
    This is the base class for forcing grid in immersed body-flow coupling.

    Notes
    -----
    Every new forcing grid class must be derived from ImmersedBodyForcingGrid.

    """

    # Will be set in derived classes
    num_lag_nodes: int = NotImplementedError
    real_t: NotImplementedError

    def __init__(self, grid_dim):
        self.grid_dim = grid_dim
        self.position_field = np.zeros(
            (self.grid_dim, self.num_lag_nodes), dtype=self.real_t
        )
        self.velocity_field = np.zeros_like(self.position_field)
        # right now its printing on all ranks since it doesnt have mpi context!
        if grid_dim == 2:
            logger.warning(
                "=========================================================="
                "\n2D body forcing grid generated, this assumes the body"
                "\nmoves in XY plane! Please initialize your body such that"
                "\nensuing dynamics are constrained in XY plane!"
                "\n=========================================================="
            )

    @abstractmethod
    def compute_lag_grid_position_field(self):
        """Computes location of forcing grid for the Cosserat rod"""

    @abstractmethod
    def compute_lag_grid_velocity_field(self):
        """Computes velocity of forcing grid points for the Cosserat rod"""

    @abstractmethod
    def transfer_forcing_from_grid_to_body(
        self,
        body_flow_forces,
        body_flow_torques,
        lag_grid_forcing_field,
    ):
        """Transfer forcing from lagrangian forcing grid to the cosserat rod"""

    @abstractmethod
    def get_maximum_lagrangian_grid_spacing(self):
        """Get the maximum Lagrangian grid spacing"""


class EmptyForcingGrid(ImmersedBodyForcingGrid):
    """
    An empty forcing grid class derived from the base class for the use of
    non-master ranks (i.e. ranks that don't have information of the global
    lagrangian quantities).
    """

    def __init__(self, grid_dim, real_t):
        self.grid_dim = grid_dim
        self.num_lag_nodes = 0
        self.real_t = real_t
        super().__init__(grid_dim=grid_dim)

    def compute_lag_grid_position_field(self):
        pass

    def compute_lag_grid_velocity_field(self):
        pass

    def transfer_forcing_from_grid_to_body(
        self, body_flow_forces, body_flow_torques, lag_grid_forcing_field
    ):
        pass

    def get_maximum_lagrangian_grid_spacing(self):
        pass
