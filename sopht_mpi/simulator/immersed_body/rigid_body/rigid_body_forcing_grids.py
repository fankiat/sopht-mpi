__all__ = [
    "TwoDimensionalCylinderForcingGrid",
    "CircularCylinderForcingGrid",
]

from elastica import Cylinder
import numpy as np
from sopht_mpi.simulator.immersed_body import ImmersedBodyForcingGrid


class TwoDimensionalCylinderForcingGrid(ImmersedBodyForcingGrid):
    """Class for forcing grid of a 2D cylinder with cross-section
    in XY plane.

    """

    def __init__(
        self,
        grid_dim,
        rigid_body: type(Cylinder),
    ):
        if grid_dim != 2:
            raise ValueError(
                "Invalid grid dimensions. 2D cylinder forcing grid is only "
                "defined for grid_dim=2"
            )
        self.cylinder = rigid_body
        super().__init__(grid_dim)
        self.local_frame_relative_position_field = np.zeros_like(self.position_field)
        self.global_frame_relative_position_field = np.zeros_like(self.position_field)

    def compute_lag_grid_position_field(self):
        """Computes location of forcing grid for the cylinder boundary"""
        self.global_frame_relative_position_field[...] = np.dot(
            self.cylinder.director_collection[: self.grid_dim, : self.grid_dim, 0].T,
            self.local_frame_relative_position_field,
        )
        self.position_field[...] = (
            self.cylinder.position_collection[: self.grid_dim]
            + self.global_frame_relative_position_field
        )

    def compute_lag_grid_velocity_field(self):
        """Computes velocity of forcing grid points for the cylinder boundary"""
        # d3 aligned along Z while d1 and d2 along XY plane...
        # Can be shown that omega local and global lie along d3 (Z axis)
        global_frame_omega_z = (
            self.cylinder.director_collection[self.grid_dim, self.grid_dim, 0]
            * self.cylinder.omega_collection[self.grid_dim, 0]
        )
        self.velocity_field[0] = (
            self.cylinder.velocity_collection[0]
            - global_frame_omega_z * self.global_frame_relative_position_field[1]
        )
        self.velocity_field[1] = (
            self.cylinder.velocity_collection[1]
            + global_frame_omega_z * self.global_frame_relative_position_field[0]
        )

    def transfer_forcing_from_grid_to_body(
        self,
        body_flow_forces,
        body_flow_torques,
        lag_grid_forcing_field,
    ):
        """Transfer forcing from lagrangian forcing grid to the cylinder"""
        # negative sign due to Newtons third law
        body_flow_forces[: self.grid_dim] = -np.sum(
            lag_grid_forcing_field, axis=1
        ).reshape(-1, 1)

        # torque from grid forcing
        # Q @ (0, 0, torque) = d3 dot (0, 0, torque) = Q[2, 2] * (0, 0, torque)
        body_flow_torques[self.grid_dim] = self.cylinder.director_collection[
            self.grid_dim, self.grid_dim, 0
        ] * np.sum(
            -self.global_frame_relative_position_field[0] * lag_grid_forcing_field[1]
            + self.global_frame_relative_position_field[1] * lag_grid_forcing_field[0]
        )

    def get_maximum_lagrangian_grid_spacing(self):
        """Get the maximum Lagrangian grid spacing"""


class CircularCylinderForcingGrid(TwoDimensionalCylinderForcingGrid):
    """Class for forcing grid of a 2D circular cylinder with cross-section
    in XY plane.

    """

    def __init__(
        self,
        grid_dim,
        rigid_body: type(Cylinder),
        num_forcing_points,
        real_t=np.float64,
    ):
        self.num_lag_nodes = num_forcing_points
        self.real_t = real_t
        super().__init__(
            grid_dim=grid_dim,
            rigid_body=rigid_body,
        )

        self.initialize_local_frame_relative_position_field()
        # to ensure position/velocity are consistent during initialisation
        self.compute_lag_grid_position_field()
        self.compute_lag_grid_velocity_field()

    def initialize_local_frame_relative_position_field(self):
        dtheta = 2.0 * np.pi / self.num_lag_nodes
        theta = np.linspace(
            0 + dtheta / 2.0, 2.0 * np.pi - dtheta / 2.0, self.num_lag_nodes
        )
        self.local_frame_relative_position_field[0, :] = self.cylinder.radius * np.cos(
            theta
        )
        self.local_frame_relative_position_field[1, :] = self.cylinder.radius * np.sin(
            theta
        )

    def get_maximum_lagrangian_grid_spacing(self):
        """Get the maximum Lagrangian grid spacing"""
        # ds = radius * dtheta
        return self.cylinder.radius * (2.0 * np.pi / self.num_lag_nodes)
