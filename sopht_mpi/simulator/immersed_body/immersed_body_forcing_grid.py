from sopht.simulator.immersed_body import ImmersedBodyForcingGrid


class EmptyForcingGrid(ImmersedBodyForcingGrid):
    """
    An empty forcing grid class derived from the base class for the use of
    non-master ranks (i.e. ranks that don't have information of the global
    lagrangian quantities).
    """

    def __init__(self, grid_dim):
        super().__init__(grid_dim=grid_dim, num_lag_nodes=0)

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
