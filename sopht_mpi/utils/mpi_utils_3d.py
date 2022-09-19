from mpi4py import MPI
import numpy as np


class MPIConstruct3D:
    """
    Sets up MPI main construct which stores the 3D grid topology and domain
    decomp information, has exclusive MPI info, and will be the one whose
    interface would be provided to the user.
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
        if 1 not in self.rank_distribution:
            raise ValueError(
                f"Rank distribution {self.rank_distribution} needs to be"
                "aligned in at least one direction for fft"
            )
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


class MPIGhostCommunicator3D:
    """
    Class exclusive for ghost communication across ranks, initialises data types
    that will be used for comm. in both blocking and non-blocking styles. Builds
    dtypes based on ghost_size (determined from stencil width of the kernel)
    This class wont be seen by the user, rather based on stencils we determine
    the properties here.
    """

    def __init__(self, ghost_size, mpi_construct):
        # extra width needed for kernel computation
        if ghost_size <= 0 and not isinstance(ghost_size, int):
            raise ValueError(
                f"Ghost size {ghost_size} needs to be an integer > 0"
                "for calling ghost communication."
            )
        self.ghost_size = ghost_size
        # define field_size variable for local field size (which includes ghost)
        self.field_size = mpi_construct.local_grid_size + 2 * self.ghost_size

        # Set datatypes for ghost communication
        # Note: these can be written in a more involved, but perhaps faster way.
        # Keeping this for now for its readibility and easy implementation.
        # Using the Create_subarray approach, each type for sending / receiving
        # needs to be initialized based on their starting index location. In
        # each dimension, we have 2 ghost layers to be sent (to next & prev) and
        # 2 corresponding receiving layers (from next & prev). This amounts to
        # (2 type for send/recv) * (2 dir along each dim) * (3 dim) = 12 type
        # Along X (next)
        self.send_next_along_x_type = mpi_construct.dtype_generator.Create_subarray(
            sizes=self.field_size,
            subsizes=[self.field_size[0], self.field_size[1], self.ghost_size],
            starts=[0, 0, self.field_size[2] - 2 * self.ghost_size],
        )
        self.send_next_along_x_type.Commit()
        self.recv_next_along_x_type = mpi_construct.dtype_generator.Create_subarray(
            sizes=self.field_size,
            subsizes=[self.field_size[0], self.field_size[1], self.ghost_size],
            starts=[0, 0, self.field_size[2] - self.ghost_size],
        )
        self.recv_next_along_x_type.Commit()
        # Along X (prev)
        self.send_previous_along_x_type = mpi_construct.dtype_generator.Create_subarray(
            sizes=self.field_size,
            subsizes=[self.field_size[0], self.field_size[1], self.ghost_size],
            starts=[0, 0, self.ghost_size],
        )
        self.send_previous_along_x_type.Commit()
        self.recv_previous_along_x_type = mpi_construct.dtype_generator.Create_subarray(
            sizes=self.field_size,
            subsizes=[self.field_size[0], self.field_size[1], self.ghost_size],
            starts=[0, 0, 0],
        )
        self.recv_previous_along_x_type.Commit()
        # Along Y (next)
        self.send_next_along_y_type = mpi_construct.dtype_generator.Create_subarray(
            sizes=self.field_size,
            subsizes=[self.field_size[0], self.ghost_size, self.field_size[2]],
            starts=[0, self.field_size[1] - 2 * self.ghost_size, 0],
        )
        self.send_next_along_y_type.Commit()
        self.recv_next_along_y_type = mpi_construct.dtype_generator.Create_subarray(
            sizes=self.field_size,
            subsizes=[self.field_size[0], self.ghost_size, self.field_size[2]],
            starts=[0, self.field_size[1] - self.ghost_size, 0],
        )
        self.recv_next_along_y_type.Commit()
        # Along Y (prev)
        self.send_previous_along_y_type = mpi_construct.dtype_generator.Create_subarray(
            sizes=self.field_size,
            subsizes=[self.field_size[0], self.ghost_size, self.field_size[2]],
            starts=[0, self.ghost_size, 0],
        )
        self.send_previous_along_y_type.Commit()
        self.recv_previous_along_y_type = mpi_construct.dtype_generator.Create_subarray(
            sizes=self.field_size,
            subsizes=[self.field_size[0], self.ghost_size, self.field_size[2]],
            starts=[0, 0, 0],
        )
        self.recv_previous_along_y_type.Commit()

        # Along Z (next)
        self.send_next_along_z_type = mpi_construct.dtype_generator.Create_subarray(
            sizes=self.field_size,
            subsizes=[self.ghost_size, self.field_size[1], self.field_size[2]],
            starts=[self.field_size[0] - 2 * self.ghost_size, 0, 0],
        )
        self.send_next_along_z_type.Commit()
        self.recv_next_along_z_type = mpi_construct.dtype_generator.Create_subarray(
            sizes=self.field_size,
            subsizes=[self.ghost_size, self.field_size[1], self.field_size[2]],
            starts=[self.field_size[0] - self.ghost_size, 0, 0],
        )
        self.recv_next_along_z_type.Commit()
        # Along Z (prev)
        self.send_previous_along_z_type = mpi_construct.dtype_generator.Create_subarray(
            sizes=self.field_size,
            subsizes=[self.ghost_size, self.field_size[1], self.field_size[2]],
            starts=[self.ghost_size, 0, 0],
        )
        self.send_previous_along_z_type.Commit()
        self.recv_previous_along_z_type = mpi_construct.dtype_generator.Create_subarray(
            sizes=self.field_size,
            subsizes=[self.ghost_size, self.field_size[1], self.field_size[2]],
            starts=[0, 0, 0],
        )
        self.recv_previous_along_z_type.Commit()

        # non-blocking comm, stuff
        self.num_requests = (
            mpi_construct.grid_dim * 2 * 2
        )  # dimension * 2 request for send/recv * 2 directions along each axis
        # initialize the requests array
        self.comm_requests = [
            0,
        ] * self.num_requests

    def exchange_init(self, local_field, mpi_construct):
        """
        Exchange ghost data between neighbors.
        """
        # Lines below to make code more literal
        z_axis = 0
        y_axis = 1
        x_axis = 2
        # Along X: send to previous block, receive from next block
        self.comm_requests[0] = mpi_construct.grid.Isend(
            (local_field, self.send_previous_along_x_type),
            dest=mpi_construct.previous_grid_along[x_axis],
        )
        self.comm_requests[1] = mpi_construct.grid.Irecv(
            (local_field, self.recv_next_along_x_type),
            source=mpi_construct.next_grid_along[x_axis],
        )
        # Along X: send to next block, receive from previous block
        self.comm_requests[2] = mpi_construct.grid.Isend(
            (local_field, self.send_next_along_x_type),
            dest=mpi_construct.next_grid_along[x_axis],
        )
        self.comm_requests[3] = mpi_construct.grid.Irecv(
            (local_field, self.recv_previous_along_x_type),
            source=mpi_construct.previous_grid_along[x_axis],
        )

        # Along Y: send to previous block, receive from next block
        self.comm_requests[4] = mpi_construct.grid.Isend(
            (local_field, self.send_previous_along_y_type),
            dest=mpi_construct.previous_grid_along[y_axis],
        )
        self.comm_requests[5] = mpi_construct.grid.Irecv(
            (local_field, self.recv_next_along_y_type),
            source=mpi_construct.next_grid_along[y_axis],
        )
        # Along Y: send to next block, receive from previous block
        self.comm_requests[6] = mpi_construct.grid.Isend(
            (local_field, self.send_next_along_y_type),
            dest=mpi_construct.next_grid_along[y_axis],
        )
        self.comm_requests[7] = mpi_construct.grid.Irecv(
            (local_field, self.recv_previous_along_y_type),
            source=mpi_construct.previous_grid_along[y_axis],
        )

        # Along Z: send to previous block, receive from next block
        self.comm_requests[8] = mpi_construct.grid.Isend(
            (local_field, self.send_previous_along_z_type),
            dest=mpi_construct.previous_grid_along[z_axis],
        )
        self.comm_requests[9] = mpi_construct.grid.Irecv(
            (local_field, self.recv_next_along_z_type),
            source=mpi_construct.next_grid_along[z_axis],
        )
        # Along Z: send to next block, receive from previous block
        self.comm_requests[10] = mpi_construct.grid.Isend(
            (local_field, self.send_next_along_z_type),
            dest=mpi_construct.next_grid_along[z_axis],
        )
        self.comm_requests[11] = mpi_construct.grid.Irecv(
            (local_field, self.recv_previous_along_z_type),
            source=mpi_construct.previous_grid_along[z_axis],
        )

    def exchange_finalise(self):
        """
        Finalizing non-blocking exchange ghost data between neighbors.
        """
        MPI.Request.Waitall(self.comm_requests)


class MPIFieldIOCommunicator3D:
    """
    Class exclusive for field communication across ranks, initialises data types
    that will be used for scattering global fields and aggregating local fields.
    Builds dtypes based on ghost_size (determined from stencil width of the
    employed kernel). This class wont be seen by the user, rather based on field
    metadata we determine the properties here.
    """

    def __init__(self, ghost_size, mpi_construct):
        # Use ghost_size to define indices for inner cell (actual data without
        # halo)
        if ghost_size < 0:
            raise ValueError("ghost size has to be >= 0")
        self.ghost_size = ghost_size
        if self.ghost_size == 0:
            self.inner_idx = ...
        else:
            self.inner_idx = (
                slice(self.ghost_size, -self.ghost_size),
            ) * mpi_construct.grid_dim
        # Datatypes for subdomain used in gather and scatter
        field_sub_size = mpi_construct.local_grid_size
        # Rank 0 uses datatype for receiving sub arrays in full array
        if mpi_construct.rank == 0:
            field_size = mpi_construct.global_grid_size
            self.sub_array_type = mpi_construct.dtype_generator.Create_subarray(
                sizes=field_size,
                subsizes=field_sub_size,
                starts=[0] * mpi_construct.grid_dim,
            )
        # Other ranks use datatype for sending sub arrays
        else:
            field_size = mpi_construct.local_grid_size + 2 * self.ghost_size
            self.sub_array_type = mpi_construct.dtype_generator.Create_subarray(
                sizes=field_size,
                subsizes=field_sub_size,
                starts=[self.ghost_size] * mpi_construct.grid_dim,
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
                : mpi_construct.local_grid_size[2],
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
        Scatter a global field in rank 0 to corresponding ranks into local
        fields
        """
        # Fill in field values for rank 0 on the edge
        if mpi_construct.rank == 0:
            local_field[self.inner_idx] = global_field[
                : mpi_construct.local_grid_size[0],
                : mpi_construct.local_grid_size[1],
                : mpi_construct.local_grid_size[2],
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
