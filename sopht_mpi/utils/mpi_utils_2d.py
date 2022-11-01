from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from sopht_mpi.utils.lab_cmap import lab_cmap


class MPIConstruct2D:
    """
    Sets up MPI main construct which stores the 2D grid topology and domain
    decomp information, has exclusive MPI info, and will be the one whose
    interface would be provided to the user.
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
        # Initialize the requests array?
        self.comm_requests = [
            0,
        ] * self.num_requests

    def exchange_init(self, local_field, mpi_construct):
        """
        Exchange ghost data between neighbors.
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

    def exchange_finalise(self):
        """
        Finalizing non-blocking exchange ghost data between neighbors.
        """
        MPI.Request.Waitall(self.comm_requests)


class MPIFieldCommunicator2D:
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
        if ghost_size < 0 and not isinstance(ghost_size, int):
            raise ValueError(
                f"Ghost size {ghost_size} needs to be an integer >= 0"
                "for field IO communication."
            )
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


class MPILagrangianFieldCommunicator2D:
    """ "
    Class exclusive for lagrangian field communication across ranks, and takes care of
    scattering global lagrangian fields and aggregating local lagrangian fields.

    Notes:
    - VirtualBoundaryForcing, and subsequently, Eulerian-Lagrangian communicator will
      initialize and use this internally.
    - Then each initialized virtual boundary instance will have its own communicator
    - Virtual boundary forcing can then call synchronization utils from this class
      and proceed with calculating quantities related to lag nodes in respective ranks
    - elastica simulator lives only on one rank? what do we do when we have more than 1?
    -
    """

    def __init__(
        self,
        eul_grid_dx,
        eul_grid_shift,
        mpi_construct,
        master_rank=0,
        real_t=np.float64,
    ):
        self.grid_dim = 2
        self.mpi_construct = mpi_construct
        self.master_rank = master_rank
        self.rank = self.mpi_construct.rank
        self.eul_subblock_dx = eul_grid_dx * self.mpi_construct.local_grid_size
        self.eul_grid_shift = eul_grid_shift

        self.real_t = real_t

        # Construct block-to-rank map for convenient access later to identify which
        # block (and hence rank) a lag node reside in
        # Note: this equivalent to comm.Get_cart_rank, which cannot accept multiple
        # coords and so cannot be called in a vectorized fashion.
        # This map is a workaround for that.
        self.rank_map = np.zeros(self.mpi_construct.grid_topology, dtype=np.int32)
        for i in range(self.mpi_construct.grid_topology[0]):
            for j in range(self.mpi_construct.grid_topology[1]):
                self.rank_map[i, j] = self.mpi_construct.grid.Get_cart_rank([i, j])

    def _compute_lag_nodes_rank_address(self, global_lag_positions):
        """
        Locate corresponding blocks (and ranks) the lagrangian nodes reside in such that
        they stay in the half-open interval [subblock_lower_bound, subblock_upper_bound)
        """
        # Note: Lagrangian positions follow xy order in grid_dim here. while quantities
        # derived from mpi_construct follow yx ordering (follows from MPI cart comm)
        eul_subblock_coords_y = (
            (global_lag_positions[1, ...] - self.eul_grid_shift)
            / self.eul_subblock_dx[0]
        ).astype(np.int32)
        eul_subblock_coords_x = (
            (global_lag_positions[0, ...] - self.eul_grid_shift)
            / self.eul_subblock_dx[1]
        ).astype(np.int32)
        if (np.any(eul_subblock_coords_x >= self.mpi_construct.grid_topology[1])) or (
            np.any(eul_subblock_coords_y >= self.mpi_construct.grid_topology[0])
        ):
            # TODO: replace this with logger message
            # python error handling exception would not work here because it halts the
            # process before abort is called
            print("Lagrangian node is found outside of Eulerian domain!")
            self.mpi_construct.grid.Abort()

        lag_nodes_rank_address = self.rank_map[
            eul_subblock_coords_y, eul_subblock_coords_x
        ]
        return lag_nodes_rank_address

    def map_lagrangian_nodes_based_on_position(self, global_lag_positions):
        if self.rank == self.master_rank:
            if global_lag_positions.shape[0] != self.grid_dim:
                # TODO: replace this with logger message
                # python error handling exception would not work here because it halts the
                # process before abort is called
                print(f"global_lag_positions needs to be shape ({self.grid_dim}, ...)")
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
                    2, expected_num_lag_nodes
                )


class MPIPlotter2D:
    """
    Minimal plotting tool for MPI 2D flow simulator.
    Currently supports only contourf functionality.
    TODO: maybe we will implement line plot functions if needed

    Warning: Use this only for quick visualization and debugging on problem with
    small grid size (preferably <256). Performance may suffer when problem size
    becomes large, since all plotting is gathered and done on a single rank.
    """

    def __init__(self, mpi_construct, ghost_size, fig_aspect_ratio=1.0):
        self.mpi_construct = mpi_construct
        self.ghost_size = ghost_size

        # Initialize communicator for gather
        self.mpi_field_comm = MPIFieldCommunicator2D(
            ghost_size=self.ghost_size, mpi_construct=self.mpi_construct
        )
        self.gather_local_field = self.mpi_field_comm.gather_local_field

        # Initialize global buffers for plotting
        self.field_io = np.zeros(self.mpi_construct.global_grid_size).astype(
            self.mpi_construct.real_t
        )
        self.x_grid_io = np.zeros_like(self.field_io)
        self.y_grid_io = np.zeros_like(self.field_io)

        # Initialize figure
        self.create_figure_and_axes(fig_aspect_ratio)

    @staticmethod
    def execute_only_on_root(func):
        def wrapper(*args, **kwargs):
            self = args[0]
            if self.mpi_construct.rank == 0:
                func(*args, **kwargs)
            else:
                pass

        return wrapper

    def create_figure_and_axes(self, fig_aspect_ratio):
        """Creates figure and axes for plotting contour fields (on all ranks)"""
        plt.style.use("seaborn")
        self.fig = plt.figure(frameon=True, dpi=150)
        self.ax = self.fig.add_subplot(111)
        if fig_aspect_ratio == "default":
            pass
        else:
            self.ax.set_aspect(aspect=fig_aspect_ratio)

    def contourf(
        self,
        x_grid,
        y_grid,
        field,
        title="",
        levels=np.linspace(0, 1, 50),
        cmap=lab_cmap,
        *args,
        **kwargs,
    ):
        """
        Plot contour fields.

        Note: this runs on every rank, but since we gather the field to rank 0,
        only rank 0 contains useful information. This will be saved later when
        `save_and_clear_fig(...)` is called, which runs only on rank 0. The
        plotting here is done on all rank since they have to wait for rank 0
        anyway, and gather field needs to be called on all rank otherwise we run
        into deadlock situations.
        """
        self.gather_local_field(self.field_io, field, self.mpi_construct)
        self.gather_local_field(self.x_grid_io, x_grid, self.mpi_construct)
        self.gather_local_field(self.y_grid_io, y_grid, self.mpi_construct)
        self.ax.set_title(title)
        contourf_obj = self.ax.contourf(
            self.x_grid_io,
            self.y_grid_io,
            self.field_io,
            levels=levels,
            cmap=cmap,
            *args,
            **kwargs,
        )
        self.cbar = self.fig.colorbar(mappable=contourf_obj, ax=self.ax)

    @execute_only_on_root
    def savefig(self, file_name, *args, **kwargs):
        """Save figure (only on root)"""
        self.fig.savefig(
            file_name,
            bbox_inches="tight",
            pad_inches=0,
            *args,
            **kwargs,
        )

    def clearfig(self):
        """Clears figure (on all ranks) for next iteration"""
        self.ax.cla()
        if self.cbar is not None:
            self.cbar.remove()
