from mpi4py import MPI
import numpy as np


class MPIConstruct2D:
    """
    Sets up MPI main construct which stores the 2D grid topology and domain decomp
    information, has exclusive MPI info, and will be the one whose interface would
    be provided to the user.
    """

    def __init__(
        self,
        grid_size_y,
        grid_size_x,
        periodic_flag=False,
        real_t=np.float64,
        rank_distribution=None,
    ):
        # grid/problem dimensions
        self.grid_dim = 2
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
        if 1 not in self.rank_distribution:
            raise ValueError(
                f"Rank distribution {self.rank_distribution} needs to be"
                "aligned in at least one direction for fft"
            )
        self.grid_topology = MPI.Compute_dims(
            self.world.Get_size(), dims=self.rank_distribution
        )
        # Check for proper domain distribution and assign local domain size
        self.global_grid_size = np.array((grid_size_y, grid_size_x))
        if np.any(self.global_grid_size % self.grid_topology):
            print("Cannot divide grid evenly to processors in x and/or y directions!")
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


class MPIGhostCommunicator2D:
    """
    Class exclusive for ghost communication across ranks, initialises data types
    that will be used for comm. in both blocking and non-blocking styles.
    Builds dtypes based on ghost_size (determined from stencil width of the kernel)
    This class wont be seen by the user, rather based on stencils we determine
    the properties here.
    """

    def __init__(self, ghost_size, mpi_construct):
        # extra width needed for kernel computation
        assert ghost_size > 0, "ghost_size has to be > 0 for calling ghost comm."
        self.ghost_size = ghost_size
        # define field_size variable for local field size (which includes ghost)
        self.field_size = mpi_construct.local_grid_size + 2 * self.ghost_size

        # Set datatypes for ghost communication
        # For row we use contiguous type
        self.row_type = mpi_construct.dtype_generator.Create_contiguous(
            count=self.ghost_size * self.field_size[1]
        )
        self.row_type.Commit()
        # For column we use strided vector
        self.column_type = mpi_construct.dtype_generator.Create_vector(
            count=self.field_size[0],
            blocklength=self.ghost_size,
            stride=self.field_size[1],
        )
        self.column_type.Commit()

        # non-blocking comm, stuff
        self.num_requests = (
            mpi_construct.grid_dim * 2 * 2
        )  # dimension * 2 request for send/recv * 2 directions along each axis
        # Better to initialize the requests array?
        self.comm_requests = [
            0,
        ] * self.num_requests

    def non_blocking_exchange_init(self, local_field, mpi_construct):
        """
        Non-blocking exchange ghost data between neighbors.
        """
        # Lines below to make code more literal
        y_axis = 0
        x_axis = 1
        # Along Y: send to previous block, receive from next block
        self.comm_requests[0] = mpi_construct.grid.Isend(
            (
                local_field[self.ghost_size : 2 * self.ghost_size, :],
                1,
                self.row_type,
            ),
            dest=mpi_construct.previous_grid_along[y_axis],
        )
        self.comm_requests[1] = mpi_construct.grid.Irecv(
            (
                local_field[-self.ghost_size : local_field.shape[0], :],
                1,
                self.row_type,
            ),
            source=mpi_construct.next_grid_along[y_axis],
        )

        # Along Y: send to next block, receive from previous block
        self.comm_requests[2] = mpi_construct.grid.Isend(
            (
                local_field[-2 * self.ghost_size : -self.ghost_size, :],
                1,
                self.row_type,
            ),
            dest=mpi_construct.next_grid_along[y_axis],
        )
        self.comm_requests[3] = mpi_construct.grid.Irecv(
            (
                local_field[0 : self.ghost_size, :],
                1,
                self.row_type,
            ),
            source=mpi_construct.previous_grid_along[y_axis],
        )

        # Along X: send to previous block, receive from next block
        self.comm_requests[4] = mpi_construct.grid.Isend(
            (
                local_field.ravel()[self.ghost_size :],
                1,
                self.column_type,
            ),
            dest=mpi_construct.previous_grid_along[x_axis],
        )
        self.comm_requests[5] = mpi_construct.grid.Irecv(
            (
                local_field.ravel()[local_field.shape[1] - self.ghost_size :],
                1,
                self.column_type,
            ),
            source=mpi_construct.next_grid_along[x_axis],
        )

        # Along X: send to next block, receive from previous block
        self.comm_requests[6] = mpi_construct.grid.Isend(
            (
                local_field.ravel()[local_field.shape[1] - 2 * self.ghost_size :],
                1,
                self.column_type,
            ),
            dest=mpi_construct.next_grid_along[x_axis],
        )
        self.comm_requests[7] = mpi_construct.grid.Irecv(
            (
                local_field.ravel()[0:],
                1,
                self.column_type,
            ),
            source=mpi_construct.previous_grid_along[x_axis],
        )

    def non_blocking_exchange_finalise(self):
        """
        Finalizing non-blocking exchange ghost data between neighbors.
        """
        MPI.Request.Waitall(self.comm_requests)


class MPIFieldIOCommunicator2D:
    """
    Class exclusive for field communication across ranks, initialises data types
    that will be used for scattering global fields and aggregating local fields.
    Builds dtypes based on field_offset (determined from local memory offset of field)
    This class wont be seen by the user, rather based on field metadata we determine
    the properties here.
    """

    def __init__(self, field_offset, mpi_construct):
        # Use offset to define indices for inner cell (actual data without halo)
        assert field_offset >= 0, "field offset has to be >= 0"
        self.field_offset = field_offset
        if self.field_offset == 0:
            self.inner_idx = ...
        else:
            self.inner_idx = (
                slice(self.field_offset, -self.field_offset),
            ) * mpi_construct.grid_dim
        # Datatypes for subdomain used in gather and scatter
        field_sub_sizes = mpi_construct.local_grid_size
        # Rank 0 uses datatype for receiving sub arrays in full array
        if mpi_construct.rank == 0:
            field_sizes = mpi_construct.global_grid_size
            field_offsets = [0, 0]
        # Other ranks use datatype for sending sub arrays
        else:
            field_sizes = mpi_construct.local_grid_size + 2 * self.field_offset
            field_offsets = [self.field_offset, self.field_offset]
        self.sub_array_type = mpi_construct.dtype_generator.Create_subarray(
            sizes=field_sizes, subsizes=field_sub_sizes, starts=field_offsets
        )
        self.sub_array_type.Commit()

    def gather_local_field(self, global_field, local_field, mpi_construct):
        """
        Gather local fields from all ranks and return a global field in rank 0
        """
        if mpi_construct.rank == 0:
            # Fill in field values for rank 0 on the edge
            global_field[
                : mpi_construct.local_grid_size[0],
                : mpi_construct.local_grid_size[1],
            ] = local_field[self.inner_idx]
            # Receiving from other ranks as contiguous array
            for rank_idx in range(1, mpi_construct.size):
                coords = mpi_construct.grid.Get_coords(rank_idx)
                idx = np.ravel_multi_index(
                    coords * mpi_construct.local_grid_size,
                    mpi_construct.global_grid_size,
                )
                mpi_construct.grid.Recv(
                    (global_field.ravel()[idx:], 1, self.sub_array_type),
                    source=rank_idx,
                )
        else:
            # Sending as contiguous chunks
            mpi_construct.grid.Send((local_field, 1, self.sub_array_type), dest=0)

    def scatter_global_field(self, local_field, global_field, mpi_construct):
        """
        Scatter a global field in rank 0 to corresponding ranks into local fields
        """
        # Fill in field values for rank 0 on the edge
        if mpi_construct.rank == 0:
            local_field[self.inner_idx] = global_field[
                : mpi_construct.local_grid_size[0],
                : mpi_construct.local_grid_size[1],
            ]
            # Sending to other ranks as contiguous array
            for rank_idx in range(1, mpi_construct.size):
                coords = mpi_construct.grid.Get_coords(rank_idx)
                idx = np.ravel_multi_index(
                    coords * mpi_construct.local_grid_size,
                    mpi_construct.global_grid_size,
                )
                mpi_construct.grid.Send(
                    (global_field.ravel()[idx:], 1, self.sub_array_type),
                    dest=rank_idx,
                )
        else:
            # Receiving from rank 0 as contiguous array
            mpi_construct.grid.Recv((local_field, 1, self.sub_array_type), source=0)
