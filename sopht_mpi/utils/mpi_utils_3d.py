from mpi4py import MPI
import numpy as np
from sopht_mpi.utils.mpi_logger import logger
from sopht.utils.field import VectorField


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
        self.real_t = real_t
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
            logger.error(
                "Cannot divide grid evenly to processors in x, y and/or z directions!\n"
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

        logger.debug(
            f"Initializing a {self.grid_dim}D simulation with\n"
            f"global_grid_size : {self.global_grid_size.tolist()}\n"
            f"processes : {self.grid_topology}\n"
            f"local_grid_size : {self.local_grid_size.tolist()}\n"
        )


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
        self.mpi_construct = mpi_construct
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

        # Initialize requests list for non-blocking comm
        self.comm_requests = []

    def exchange_scalar_field_init(self, local_field):
        """
        Exchange ghost data between neighbors.
        """
        # Lines below to make code more literal
        z_axis = 0
        y_axis = 1
        x_axis = 2
        # Along X: send to previous block, receive from next block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_previous_along_x_type),
                dest=self.mpi_construct.previous_grid_along[x_axis],
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_next_along_x_type),
                source=self.mpi_construct.next_grid_along[x_axis],
            )
        )
        # Along X: send to next block, receive from previous block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_next_along_x_type),
                dest=self.mpi_construct.next_grid_along[x_axis],
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_previous_along_x_type),
                source=self.mpi_construct.previous_grid_along[x_axis],
            )
        )

        # Along Y: send to previous block, receive from next block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_previous_along_y_type),
                dest=self.mpi_construct.previous_grid_along[y_axis],
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_next_along_y_type),
                source=self.mpi_construct.next_grid_along[y_axis],
            )
        )
        # Along Y: send to next block, receive from previous block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_next_along_y_type),
                dest=self.mpi_construct.next_grid_along[y_axis],
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_previous_along_y_type),
                source=self.mpi_construct.previous_grid_along[y_axis],
            )
        )

        # Along Z: send to previous block, receive from next block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_previous_along_z_type),
                dest=self.mpi_construct.previous_grid_along[z_axis],
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_next_along_z_type),
                source=self.mpi_construct.next_grid_along[z_axis],
            )
        )
        # Along Z: send to next block, receive from previous block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_next_along_z_type),
                dest=self.mpi_construct.next_grid_along[z_axis],
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_previous_along_z_type),
                source=self.mpi_construct.previous_grid_along[z_axis],
            )
        )

    def exchange_vector_field_init(self, local_vector_field):
        self.exchange_scalar_field_init(
            local_field=local_vector_field[VectorField.x_axis_idx()]
        )
        self.exchange_scalar_field_init(
            local_field=local_vector_field[VectorField.y_axis_idx()]
        )
        self.exchange_scalar_field_init(
            local_field=local_vector_field[VectorField.z_axis_idx()]
        )

    def exchange_finalise(self):
        """
        Finalizing non-blocking exchange ghost data between neighbors.
        """
        MPI.Request.Waitall(self.comm_requests)
        # reset the list of requests
        self.comm_requests = []


class MPIFieldCommunicator3D:
    """
    Class exclusive for field communication across ranks, initialises data types
    that will be used for scattering global fields and aggregating local fields.
    Builds dtypes based on ghost_size (determined from stencil width of the
    employed kernel). This class wont be seen by the user, rather based on field
    metadata we determine the properties here.
    """

    def __init__(self, ghost_size, mpi_construct, master_rank=0):
        # Use ghost_size to define indices for inner cell (actual data without
        # halo)
        if ghost_size < 0 and not isinstance(ghost_size, int):
            raise ValueError(
                f"Ghost size {ghost_size} needs to be an integer >= 0"
                "for field IO communication."
            )
        self.ghost_size = ghost_size
        self.mpi_construct = mpi_construct
        if self.ghost_size == 0:
            self.inner_idx = ...
        else:
            self.inner_idx = (
                slice(self.ghost_size, -self.ghost_size),
            ) * mpi_construct.grid_dim
        # Datatypes for subdomain used in gather and scatter
        field_sub_size = mpi_construct.local_grid_size
        # master rank uses datatype for receiving sub arrays in full array
        self.master_rank = master_rank
        self.slave_ranks = set(np.arange(mpi_construct.size)) - set([self.master_rank])
        if mpi_construct.rank == self.master_rank:
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

    def gather_local_scalar_field(self, global_field, local_field):
        """
        Gather local scalar fields from all ranks and return a global scalar field in
        rank 0
        """
        if self.mpi_construct.rank == self.master_rank:
            # Fill in field values for master rank
            coords = self.mpi_construct.grid.Get_coords(self.master_rank)
            local_chunk_idx = (
                slice(
                    coords[0] * self.mpi_construct.local_grid_size[0],
                    (coords[0] + 1) * self.mpi_construct.local_grid_size[0],
                ),
                slice(
                    coords[1] * self.mpi_construct.local_grid_size[1],
                    (coords[1] + 1) * self.mpi_construct.local_grid_size[1],
                ),
                slice(
                    coords[2] * self.mpi_construct.local_grid_size[2],
                    (coords[2] + 1) * self.mpi_construct.local_grid_size[2],
                ),
            )
            global_field[local_chunk_idx] = local_field[self.inner_idx]
            # Receiving from other ranks as contiguous array
            for rank_idx in self.slave_ranks:
                coords = self.mpi_construct.grid.Get_coords(rank_idx)
                idx = np.ravel_multi_index(
                    coords * self.mpi_construct.local_grid_size,
                    self.mpi_construct.global_grid_size,
                )
                self.mpi_construct.grid.Recv(
                    (global_field.ravel()[idx:], 1, self.sub_array_type),
                    source=rank_idx,
                )
        else:
            # Sending as contiguous chunks
            self.mpi_construct.grid.Send(
                (local_field, 1, self.sub_array_type), dest=self.master_rank
            )

    def gather_local_vector_field(self, global_vector_field, local_vector_field):
        """
        Gather local vector fields from all ranks and return a global vector field in
        rank 0
        """
        self.gather_local_scalar_field(
            global_field=global_vector_field[VectorField.x_axis_idx()],
            local_field=local_vector_field[VectorField.x_axis_idx()],
        )
        self.gather_local_scalar_field(
            global_field=global_vector_field[VectorField.y_axis_idx()],
            local_field=local_vector_field[VectorField.y_axis_idx()],
        )
        self.gather_local_scalar_field(
            global_field=global_vector_field[VectorField.z_axis_idx()],
            local_field=local_vector_field[VectorField.z_axis_idx()],
        )

    def scatter_global_scalar_field(self, local_field, global_field):
        """
        Scatter a global scalar field in rank 0 into local scalar fields in each
        corresponding ranks
        """
        # Fill in field values for master rank on the edge
        if self.mpi_construct.rank == self.master_rank:
            coords = self.mpi_construct.grid.Get_coords(self.master_rank)
            local_chunk_idx = (
                slice(
                    coords[0] * self.mpi_construct.local_grid_size[0],
                    (coords[0] + 1) * self.mpi_construct.local_grid_size[0],
                ),
                slice(
                    coords[1] * self.mpi_construct.local_grid_size[1],
                    (coords[1] + 1) * self.mpi_construct.local_grid_size[1],
                ),
                slice(
                    coords[2] * self.mpi_construct.local_grid_size[2],
                    (coords[2] + 1) * self.mpi_construct.local_grid_size[2],
                ),
            )
            local_field[self.inner_idx] = global_field[local_chunk_idx]
            # Sending to other ranks as contiguous array
            for rank_idx in self.slave_ranks:
                coords = self.mpi_construct.grid.Get_coords(rank_idx)
                idx = np.ravel_multi_index(
                    coords * self.mpi_construct.local_grid_size,
                    self.mpi_construct.global_grid_size,
                )
                self.mpi_construct.grid.Send(
                    (global_field.ravel()[idx:], 1, self.sub_array_type),
                    dest=rank_idx,
                )
        else:
            # Receiving from rank 0 as contiguous array
            self.mpi_construct.grid.Recv(
                (local_field, 1, self.sub_array_type), source=self.master_rank
            )

    def scatter_global_vector_field(self, local_vector_field, global_vector_field):
        """
        Scatter a global vector field in master rank into local vector fields in each
        corresponding ranks
        """
        self.scatter_global_scalar_field(
            local_field=local_vector_field[VectorField.x_axis_idx()],
            global_field=global_vector_field[VectorField.x_axis_idx()],
        )
        self.scatter_global_scalar_field(
            local_field=local_vector_field[VectorField.y_axis_idx()],
            global_field=global_vector_field[VectorField.y_axis_idx()],
        )
        self.scatter_global_scalar_field(
            local_field=local_vector_field[VectorField.z_axis_idx()],
            global_field=global_vector_field[VectorField.z_axis_idx()],
        )


class MPILagrangianFieldCommunicator3D:
    """
    Class exclusive for 3D lagrangian field communication across ranks, and takes care
    of scattering global lagrangian fields and aggregating local lagrangian fields.

    Notes:
    - VirtualBoundaryForcing, and subsequently, Eulerian-Lagrangian communicator will
      initialize and use this internally.
    - Then each initialized virtual boundary instance will have its own communicator
    - Virtual boundary forcing can then call synchronization utils from this class
      and proceed with calculating quantities related to lag nodes in respective ranks
    """

    def __init__(
        self,
        eul_grid_dx,
        eul_grid_coord_shift,
        mpi_construct,
        master_rank=0,
        real_t=np.float64,
    ):
        self.grid_dim = 3
        self.mpi_construct = mpi_construct
        self.master_rank = master_rank
        self.rank = self.mpi_construct.rank
        self.eul_subblock_dx = eul_grid_dx * self.mpi_construct.local_grid_size
        self.eul_grid_coord_shift = eul_grid_coord_shift

        self.real_t = real_t

        # Construct block-to-rank map for convenient access later to identify which
        # block (and hence rank) a lag node reside in
        # Note: this equivalent to comm.Get_cart_rank, which cannot accept multiple
        # coords and so cannot be called in a vectorized fashion.
        # This map is a workaround for that.
        self.rank_map = np.zeros(self.mpi_construct.grid_topology, dtype=np.int32)
        for i in range(self.mpi_construct.grid_topology[0]):
            for j in range(self.mpi_construct.grid_topology[1]):
                for k in range(self.mpi_construct.grid_topology[2]):
                    self.rank_map[i, j, k] = self.mpi_construct.grid.Get_cart_rank(
                        [i, j, k]
                    )

    def _compute_lag_nodes_rank_address(self, global_lag_positions):
        """
        Locate corresponding blocks (and ranks) the lagrangian nodes reside in such that
        they stay in the half-open interval [subblock_lower_bound, subblock_upper_bound)
        """
        # Note: Lagrangian positions follow xy order in grid_dim here. while quantities
        # derived from mpi_construct follow yx ordering (follows from MPI cart comm)
        eul_subblock_coords_z = (
            (global_lag_positions[2, ...] - self.eul_grid_coord_shift)
            / self.eul_subblock_dx[0]
        ).astype(np.int32)
        eul_subblock_coords_y = (
            (global_lag_positions[1, ...] - self.eul_grid_coord_shift)
            / self.eul_subblock_dx[1]
        ).astype(np.int32)
        eul_subblock_coords_x = (
            (global_lag_positions[0, ...] - self.eul_grid_coord_shift)
            / self.eul_subblock_dx[2]
        ).astype(np.int32)
        if (
            (np.any(eul_subblock_coords_x >= self.mpi_construct.grid_topology[2]))
            or (np.any(eul_subblock_coords_y >= self.mpi_construct.grid_topology[1]))
            or (np.any(eul_subblock_coords_z >= self.mpi_construct.grid_topology[0]))
        ):
            # python error handling exception would not work here because it halts the
            # process before abort is called
            logger.error("Lagrangian node is found outside of Eulerian domain!")
            self.mpi_construct.grid.Abort()

        lag_nodes_rank_address = self.rank_map[
            eul_subblock_coords_z, eul_subblock_coords_y, eul_subblock_coords_x
        ]
        return lag_nodes_rank_address

    def map_lagrangian_nodes_based_on_position(self, global_lag_positions):
        if self.rank == self.master_rank:
            if global_lag_positions.shape[0] != self.grid_dim:
                # python error handling exception would not work here because it halts the
                # process before abort is called
                logger.error(
                    f"global_lag_positions needs to be shape ({self.grid_dim}, ...)"
                )
                self.mpi_construct.grid.Abort()
            rank_address = self._compute_lag_nodes_rank_address(global_lag_positions)
        else:
            rank_address = None
        self.rank_address = self.mpi_construct.grid.bcast(
            rank_address, root=self.master_rank
        )

        self.local_nodes_idx = np.where(self.rank_address == self.rank)
        self.local_num_lag_nodes = np.count_nonzero(
            self.rank_address == self.mpi_construct.rank
        )
        self.slave_ranks_containing_lag_nodes = set(self.rank_address) - set(
            [self.master_rank]
        )

    def scatter_global_field(self, local_lag_field, global_lag_field):
        """
        Scatter lagrangian nodes to ranks that are involved.

        Note: This assumes that lag nodes are already correctly mapped. If the nodes are
        moving, a re-mapping is needed.
        """
        # Send from master rank to other ranks containing the lagrangian grid
        if self.rank == self.master_rank:
            # first set the local field for the master rank
            idx = np.where(self.rank_address == self.rank)[0]
            local_lag_field[...] = global_lag_field[:, idx]
            # then send the other chunks to other ranks
            for rank_i in self.slave_ranks_containing_lag_nodes:
                idx = np.where(self.rank_address == rank_i)[0]
                self.mpi_construct.grid.Send(
                    global_lag_field[:, idx].ravel(), dest=rank_i
                )

        # Other ranks containing the lagrangian grid receives array from master rank
        if self.rank in self.slave_ranks_containing_lag_nodes:
            self.mpi_construct.grid.Recv(local_lag_field, source=self.master_rank)

    def gather_local_field(self, global_lag_field, local_lag_field):
        """
        Gather lagrangian nodes to master rank

        Note: This assumes that lag nodes are already correctly mapped. If the nodes are
        moving, a re-mapping is needed.
        """
        # Slave ranks send their corresponding lagrangian grid to master rank
        if self.rank in self.slave_ranks_containing_lag_nodes:
            self.mpi_construct.grid.Send(local_lag_field, dest=self.master_rank)

        # Master rank receives from other ranks containing the lagrangian grid
        if self.rank == self.master_rank:
            # first set the chunk of global field for the master rank
            idx = np.where(self.rank_address == self.rank)[0]
            global_lag_field[:, idx] = local_lag_field
            # then receive other chunks of the global field from other involved ranks
            for rank_i in self.slave_ranks_containing_lag_nodes:
                idx = np.where(self.rank_address == rank_i)[0]
                expected_num_lag_nodes = len(idx)
                recv_buffer = np.empty(
                    self.grid_dim * expected_num_lag_nodes, dtype=self.real_t
                )
                self.mpi_construct.grid.Recv(recv_buffer, source=rank_i)
                global_lag_field[:, idx] = recv_buffer.reshape(
                    self.grid_dim, expected_num_lag_nodes
                )
