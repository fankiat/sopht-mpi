from mpi4py import MPI
import numpy as np


class MPIConstruct3D:
    """
    Sets up MPI main construct which stores the 3D grid topology and domain decomp
    information, has exclusive MPI info, and will be the one whose interface would
    be provided to the user.
    """

    def __init__(
        self,
        grid_size_z,
        grid_size_y,
        grid_size_x,
        periodic_flag=False,
        real_t=np.float64,
        rank_distribution=None,
    ):
        # grid/problem dimensions
        self.grid_dim = 3
        # Set the MPI dtype generator based on precision
        self.dtype_generator = MPI.FLOAT if real_t == np.float32 else MPI.DOUBLE
        # Setup MPI environment
        self.world = MPI.COMM_WORLD
        # Automatically create topologies
        if rank_distribution is None:
            self.rank_distribution = [0] * self.grid_dim
            self.rank_distribution[
                -1
            ] = 1  # to align at least one dimension for fft operations
        else:
            self.rank_distribution = rank_distribution
        assert (
            1 in self.rank_distribution
        ), f"Rank distribution {self.rank_distribution} needs to be aligned in at least one direction for fft"
        self.grid_topology = MPI.Compute_dims(
            self.world.Get_size(), dims=self.rank_distribution
        )
        # Check for proper domain distribution and assign local domain size
        self.global_grid_size = np.array((grid_size_z, grid_size_y, grid_size_x))
        if np.any(self.global_grid_size % self.grid_topology):
            print(
                "Cannot divide grid evenly to processors in x, y and/or z directions!"
            )
            print(
                f"{self.global_grid_size / self.grid_topology} x {self.grid_topology} "
                f"!= {self.global_grid_size}"
            )
            raise RuntimeError("Invalid domain decomposition")
        else:
            self.local_grid_size = (self.global_grid_size / self.grid_topology).astype(
                int
            )

        # Create Cartesian grid communicator
        self.grid = self.world.Create_cart(
            self.grid_topology, periods=periodic_flag, reorder=False
        )
        # Determine neighbours in all directions
        self.previous_grid_along = np.zeros(self.grid_dim).astype(int)
        self.next_grid_along = np.zeros(self.grid_dim).astype(int)
        for dim in range(self.grid_dim):
            (
                self.previous_grid_along[dim],
                self.next_grid_along[dim],
            ) = self.grid.Shift(dim, 1)
        self.size = self.grid.Get_size()
        self.rank = self.grid.Get_rank()

        if self.rank == 0:
            print(f"Initializing a {self.grid_dim}D simulation with")
            print(f"global_grid_size : {self.global_grid_size.tolist()}")
            print(f"processes : {self.grid_topology}")
            print(f"local_grid_size : {self.local_grid_size.tolist()}")
