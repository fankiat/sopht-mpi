import logging
import numpy as np
import pytest
import sopht_mpi.sopht_mpi_simulator as sps
from sopht.utils.precision import get_real_t


class EmptyDerivedForcingGrid(sps.ImmersedBodyForcingGrid):
    """
    An empty forcing grid class derived from the base class for the use of
    non-master ranks (i.e. ranks that don't have information of the global
    lagrangian quantities).
    """

    def __init__(self, grid_dim, num_lag_nodes, real_t):
        self.grid_dim = grid_dim
        self.num_lag_nodes = num_lag_nodes
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


@pytest.mark.parametrize("grid_dim", [2, 3])
@pytest.mark.parametrize("num_lag_nodes", [8, 16])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_immersed_body_forcing_grid(grid_dim, num_lag_nodes, precision, caplog):
    real_t = get_real_t(precision)
    with caplog.at_level(logging.WARNING):
        forcing_grid = EmptyDerivedForcingGrid(
            grid_dim=grid_dim, num_lag_nodes=num_lag_nodes, real_t=real_t
        )
    assert forcing_grid.grid_dim == grid_dim
    assert forcing_grid.num_lag_nodes == num_lag_nodes
    correct_forcing_grid_field = np.zeros((grid_dim, num_lag_nodes))
    np.testing.assert_allclose(forcing_grid.position_field, correct_forcing_grid_field)
    np.testing.assert_allclose(forcing_grid.velocity_field, correct_forcing_grid_field)
    if grid_dim == 2:
        warning_message = (
            "=========================================================="
            "\n2D body forcing grid generated, this assumes the body"
            "\nmoves in XY plane! Please initialize your body such that"
            "\nensuing dynamics are constrained in XY plane!"
            "\n=========================================================="
        )
        assert warning_message in caplog.text
