"""MPI-supported Eulerian-Lagrangian grid communicator in 2D."""
from numba import njit
import numpy as np
from sopht.utils.field import VectorField


class EulerianLagrangianGridCommunicatorMPI2D:
    """Class for MPI-supported communication between Eulerian and Lagrangian grids in 2D

    Sets up a MPI-supported communicator between Eulerian and Lagrangian grids in the
    domain, which consists of:
    1. Find grid intersections (nearest indices)
    2. Interpolate fields back and forth
    3. Compute interpolation weights for interpolation
    """

    def __init__(
        self,
        dx,
        eul_grid_coord_shift,
        interp_kernel_width,
        real_t,
        mpi_construct,
        ghost_size,
        n_components=1,
        interp_kernel_type="cosine",
    ):
        """Class initialiser."""
        # Check that ghost size is enough for interp_kernel_width
        if ghost_size < interp_kernel_width:
            raise ValueError(
                f"ghost size ({ghost_size}) needs to be >= interp kernel width "
                f"({interp_kernel_width})"
            )
        # MPI-related variables
        self.mpi_construct = mpi_construct
        self.mpi_ghost_sum_comm = MPIGhostSumCommunicator2D(
            ghost_size=ghost_size, mpi_construct=self.mpi_construct
        )
        # Compute mpi rank local eul grid coord shift (accounting also for ghost cell)
        self.mpi_substart_idx = np.flip(
            mpi_construct.grid.coords * mpi_construct.local_grid_size
        )
        # store local index shift due to subdomain grid shift and ghost cell in ints
        # for offsetting the local nearest eul grid index to lag grid accordingly later
        self.mpi_local_substart_coord_shift = self.mpi_substart_idx - ghost_size

        # Kernel generation
        # Local eulerian grid support (nearest indices)
        self.local_eulerian_grid_support_of_lagrangian_grid_kernel = (
            generate_local_eulerian_grid_support_of_lagrangian_grid_kernel_2d(
                dx=dx,
                eul_grid_coord_shift=eul_grid_coord_shift,
                interp_kernel_width=interp_kernel_width,
                mpi_local_substart_coord_shift=self.mpi_local_substart_coord_shift,
            )
        )

        # Eulerian to lagrangian grid interpolation
        self.eulerian_to_lagrangian_grid_interpolation_kernel = (
            generate_eulerian_to_lagrangian_grid_interpolation_kernel_2d(
                dx=dx,
                interp_kernel_width=interp_kernel_width,
                n_components=n_components,
            )
        )

        # Lagrangian to eulerian grid interpolation
        self.lagrangian_to_eulerian_grid_interpolation_kernel_without_ghost_sum = (
            generate_lagrangian_to_eulerian_grid_interpolation_kernel_2d(
                interp_kernel_width=interp_kernel_width,
                n_components=n_components,
            )
        )

        # Eulerian grid ghost sum kernel
        self.eulerian_grid_ghost_sum = generate_eulerian_grid_ghost_sum_2d(
            n_components, self.mpi_ghost_sum_comm
        )

        if interp_kernel_type == "peskin":
            self.interpolation_weights_kernel = (
                generate_peskin_interpolation_weights_kernel_2d(
                    dx=dx, interp_kernel_width=interp_kernel_width, real_t=real_t
                )
            )
        elif interp_kernel_type == "cosine":
            self.interpolation_weights_kernel = (
                generate_cosine_interpolation_weights_kernel_2d(
                    dx=dx, interp_kernel_width=interp_kernel_width, real_t=real_t
                )
            )
        else:
            raise ValueError(
                "Invalid interpolation kernel type. Currently supported types are"
                "'cosine' and 'peskin'."
            )

    def lagrangian_to_eulerian_grid_interpolation_kernel(
        self,
        eul_grid_field,
        lag_grid_field,
        interp_weights,
        nearest_eul_grid_index_to_lag_grid,
    ):
        self.lagrangian_to_eulerian_grid_interpolation_kernel_without_ghost_sum(
            eul_grid_field,
            lag_grid_field,
            interp_weights,
            nearest_eul_grid_index_to_lag_grid,
        )
        # Add ghost cell contribution to neighbors and zero out ghost cells
        self.eulerian_grid_ghost_sum(local_field=eul_grid_field)


def generate_local_eulerian_grid_support_of_lagrangian_grid_kernel_2d(
    dx, eul_grid_coord_shift, interp_kernel_width, mpi_local_substart_coord_shift
):
    """
    Generate kernel that computes local Eulerian support of Lagrangian grid.

    Inputs:
    interp_kernel_width: width of interpolation kernel
    eul_grid_coord_shift: shift of the coordinates of the local Eulerian grid start
    (i.e. starting point of current local Eulerian grid)
    dx: Eulerian grid spacing

    """
    # grid/problem dimensions
    grid_dim = 2
    x = np.arange(-interp_kernel_width + 1, interp_kernel_width + 1)
    x_grid, y_grid = np.meshgrid(x, x)
    local_eul_grid_support_idx = np.stack((x_grid, y_grid))

    @njit(fastmath=True)
    def local_eulerian_grid_support_of_lagrangian_grid_kernel_2d(
        local_eul_grid_support_of_lag_grid,
        nearest_eul_grid_index_to_lag_grid,
        lag_positions,
    ):
        """Compute local Eulerian support of Lagrangian grid.

        Return nearest_eul_grid_index_to_lag_grid: size (grid_dim, num_lag_nodes)
        local_eul_grid_support_of_lag_grid: local Eulerian grid support of the Lagrangian grid
        (grid_dim, 2 * interp_kernel_width, 2 * interp_kernel_width, num_lag_nodes)
        from input params:
        lag_positions: (grid_dim, num_lag_nodes)

        """
        # dtype of nearest_grid_index takes care of type casting to int
        # The approach for nearest eul grid index:
        # (1) Compute the nearest eul grid index as done in `sopht`, thus
        # achieving the precision in computed index.
        # (2) Offset the computed index with the MPI related coords (i.e. local domain
        # index and ghost cells)
        nearest_eul_grid_index_to_lag_grid[...] = (
            lag_positions - eul_grid_coord_shift
        ) // dx - mpi_local_substart_coord_shift.reshape(grid_dim, 1)

        # reshape done to broadcast
        local_eul_grid_support_of_lag_grid[...] = (
            (
                nearest_eul_grid_index_to_lag_grid.reshape(grid_dim, 1, 1, -1)
                + local_eul_grid_support_idx.reshape(
                    grid_dim, 2 * interp_kernel_width, 2 * interp_kernel_width, 1
                )
                + mpi_local_substart_coord_shift.reshape(grid_dim, 1, 1, -1)
            )
            * dx
            + eul_grid_coord_shift
            - lag_positions.reshape(grid_dim, 1, 1, -1)
        )

    return local_eulerian_grid_support_of_lagrangian_grid_kernel_2d


def generate_eulerian_to_lagrangian_grid_interpolation_kernel_2d(
    dx, interp_kernel_width, n_components
):
    """Generate kernel that interpolates a field from an Eulerian grid to a Lagrangian grid.

    Inputs:
    interp_kernel_width: width of interpolation kernel
    dx: Eulerian grid spacing
    n_components : number of components in Lagrangian field

    """
    assert (
        n_components == 1 or n_components == 2
    ), "invalid number of components for interpolation!"
    # grid/problem dimensions
    grid_dim = 2

    if n_components == 1:

        @njit(fastmath=True)
        def eulerian_to_lagrangian_grid_interpolation_kernel_2d(
            lag_grid_field,
            eul_grid_field,
            interp_weights,
            nearest_eul_grid_index_to_lag_grid,
        ):
            """Interpolate an Eulerian field onto a Lagrangian field.

            Inputs:
            the nearest_eul_grid_index_to_lag_grid(grid_dim, num_lag_nodes) and
            interpolation weights interp_weights of
            shape (2 * interp_kernel_width, 2 * interp_kernel_width, num_lag_nodes)

            """
            num_lag_nodes = lag_grid_field.shape[-1]
            for i in range(0, num_lag_nodes):
                lag_grid_field[i] = np.sum(
                    eul_grid_field[
                        nearest_eul_grid_index_to_lag_grid[1, i]
                        - interp_kernel_width
                        + 1 : nearest_eul_grid_index_to_lag_grid[1, i]
                        + interp_kernel_width
                        + 1,
                        nearest_eul_grid_index_to_lag_grid[0, i]
                        - interp_kernel_width
                        + 1 : nearest_eul_grid_index_to_lag_grid[0, i]
                        + interp_kernel_width
                        + 1,
                    ]
                    * interp_weights[..., i]
                ) * (dx**grid_dim)

        return eulerian_to_lagrangian_grid_interpolation_kernel_2d
    else:

        @njit(fastmath=True)
        def vector_field_eulerian_to_lagrangian_grid_interpolation_kernel_2d(
            lag_grid_field,
            eul_grid_field,
            interp_weights,
            nearest_eul_grid_index_to_lag_grid,
        ):
            """Interpolate an Eulerian vector field onto a Lagrangian vector field.

            Inputs:
            the nearest_eul_grid_index_to_lag_grid(grid_dim, num_lag_nodes) and
            interpolation weights interp_weights of
            shape (2 * interp_kernel_width, 2 * interp_kernel_width, num_lag_nodes)

            """
            num_lag_nodes = lag_grid_field.shape[-1]
            for i in range(0, num_lag_nodes):
                # numba doesnt allow multiple axes for np.sum :/,
                # hence needs to be done serially
                lag_grid_field[0, i] = np.sum(
                    eul_grid_field[
                        0,
                        nearest_eul_grid_index_to_lag_grid[1, i]
                        - interp_kernel_width
                        + 1 : nearest_eul_grid_index_to_lag_grid[1, i]
                        + interp_kernel_width
                        + 1,
                        nearest_eul_grid_index_to_lag_grid[0, i]
                        - interp_kernel_width
                        + 1 : nearest_eul_grid_index_to_lag_grid[0, i]
                        + interp_kernel_width
                        + 1,
                    ]
                    * interp_weights[..., i]
                ) * (dx**grid_dim)
                lag_grid_field[1, i] = np.sum(
                    eul_grid_field[
                        1,
                        nearest_eul_grid_index_to_lag_grid[1, i]
                        - interp_kernel_width
                        + 1 : nearest_eul_grid_index_to_lag_grid[1, i]
                        + interp_kernel_width
                        + 1,
                        nearest_eul_grid_index_to_lag_grid[0, i]
                        - interp_kernel_width
                        + 1 : nearest_eul_grid_index_to_lag_grid[0, i]
                        + interp_kernel_width
                        + 1,
                    ]
                    * interp_weights[..., i]
                ) * (dx**grid_dim)

        return vector_field_eulerian_to_lagrangian_grid_interpolation_kernel_2d


def generate_lagrangian_to_eulerian_grid_interpolation_kernel_2d(
    interp_kernel_width, n_components=1
):
    """Generate kernel that interpolates a field from a Lagrangian grid to an Eulerian grid.

    Inputs:
    interp_kernel_width: width of interpolation kernel
    n_components : number of components in Lagrangian field

    """
    assert (
        n_components == 1 or n_components == 2
    ), "invalid number of components for interpolation!"

    if n_components == 1:

        @njit(fastmath=True)
        def lagrangian_to_eulerian_grid_interpolation_kernel_2d(
            eul_grid_field,
            lag_grid_field,
            interp_weights,
            nearest_eul_grid_index_to_lag_grid,
        ):
            """Interpolate a Lagrangian field onto an Eulerian field.

            Inputs:
            the nearest_eul_grid_index_to_lag_grid(grid_dim, num_lag_nodes) and
            interpolation weights interp_weights of
            shape (2 * interp_kernel_width, 2 * interp_kernel_width, num_lag_nodes)

            """
            num_lag_nodes = lag_grid_field.shape[-1]
            for i in range(0, num_lag_nodes):
                eul_grid_field[
                    nearest_eul_grid_index_to_lag_grid[1, i]
                    - interp_kernel_width
                    + 1 : nearest_eul_grid_index_to_lag_grid[1, i]
                    + interp_kernel_width
                    + 1,
                    nearest_eul_grid_index_to_lag_grid[0, i]
                    - interp_kernel_width
                    + 1 : nearest_eul_grid_index_to_lag_grid[0, i]
                    + interp_kernel_width
                    + 1,
                ] += (
                    lag_grid_field[..., i] * interp_weights[..., i]
                )

        return lagrangian_to_eulerian_grid_interpolation_kernel_2d
    else:

        @njit(fastmath=True)
        def vector_field_lagrangian_to_eulerian_grid_interpolation_kernel_2d(
            eul_grid_field,
            lag_grid_field,
            interp_weights,
            nearest_eul_grid_index_to_lag_grid,
        ):
            """Interpolate a Lagrangian vector field onto an Eulerian field.

            Inputs:
            the nearest_eul_grid_index_to_lag_grid(grid_dim, num_lag_nodes) and
            interpolation weights interp_weights of
            shape (2 * interp_kernel_width, 2 * interp_kernel_width, num_lag_nodes)

            """
            # TODO We need to add boundary exception handling! where the Lagrangian
            #  node goes in `interp_kernel_width` boundary zone of the Eulerian grid
            num_lag_nodes = lag_grid_field.shape[-1]
            for i in range(0, num_lag_nodes):
                eul_grid_field[
                    ...,
                    nearest_eul_grid_index_to_lag_grid[1, i]
                    - interp_kernel_width
                    + 1 : nearest_eul_grid_index_to_lag_grid[1, i]
                    + interp_kernel_width
                    + 1,
                    nearest_eul_grid_index_to_lag_grid[0, i]
                    - interp_kernel_width
                    + 1 : nearest_eul_grid_index_to_lag_grid[0, i]
                    + interp_kernel_width
                    + 1,
                ] += (
                    np.ascontiguousarray(lag_grid_field[..., i]).reshape(-1, 1, 1)
                    * interp_weights[..., i]
                )

        return vector_field_lagrangian_to_eulerian_grid_interpolation_kernel_2d


def generate_eulerian_grid_ghost_sum_2d(n_components, mpi_ghost_sum_comm):
    if n_components == 1:
        return mpi_ghost_sum_comm.ghost_sum
    else:

        def vector_field_ghost_sum(local_field):
            mpi_ghost_sum_comm.ghost_sum(local_field[VectorField.x_axis_idx()])
            mpi_ghost_sum_comm.ghost_sum(local_field[VectorField.y_axis_idx()])

        return vector_field_ghost_sum


def generate_cosine_interpolation_weights_kernel_2d(dx, interp_kernel_width, real_t):
    """Generate the kernel for computing interpolation weights using 2D cosine delta function.

    Inputs:
    interp_kernel_width: width of interpolation kernel
    dx : Eulerian grid spacing

    """
    # grid/problem dimensions
    grid_dim = 2
    assert (
        interp_kernel_width == 2
    ), "Interpolation kernel inconsistent with interpolation kernel width!"

    @njit(fastmath=True)
    def cosine_interpolation_weights_kernel_2d(
        interp_weights, local_eul_grid_support_of_lag_grid
    ):
        """Compute the interpolation weights using 2D cosine delta function.

        Result stored in interp_weights of shape
        (2 * interp_kernel_width, 2 * interp_kernel_width, ...) with
        input as eul_grid_support_of_lag_grid of shape
        (grid_dim, 2 * interp_kernel_width, 2 * interp_kernel_width, ...)
        Applicable for interp_kernel_width = 2
        """
        local_eul_grid_support_of_lag_grid /= dx
        interp_weights[...] = (
            real_t((0.25 / dx) ** grid_dim)
            * (
                real_t(1.0)
                + np.cos(real_t(0.5 * np.pi) * local_eul_grid_support_of_lag_grid[0])
            )
            * (
                real_t(1.0)
                + np.cos(real_t(0.5 * np.pi) * local_eul_grid_support_of_lag_grid[1])
            )
        )

    return cosine_interpolation_weights_kernel_2d


def generate_peskin_interpolation_weights_kernel_2d(dx, interp_kernel_width, real_t):
    """Generate the kernel for computing interpolation weights proposed by Peskin, 2002, 6.27.

    Inputs:
    interp_kernel_width: width of interpolation kernel
    dx : Eulerian grid spacing

    """
    # grid/problem dimensions
    grid_dim = 2
    assert (
        interp_kernel_width == 2
    ), "Interpolation kernel inconsistent with interpolation kernel width!"

    @njit(fastmath=True)
    def peskin_interpolation_weights_kernel_2d(
        interp_weights, local_eul_grid_support_of_lag_grid
    ):
        """Compute the interpolation weights using 2D delta function by Peskin, 2002.

        Result stored in interp_weights of shape
        (2 * interp_kernel_width, 2 * interp_kernel_width, ...) with
        input as eul_grid_support_of_lag_grid of shape
        (grid_dim, 2 * interp_kernel_width, 2 * interp_kernel_width, ...)
        Applicable for interp_kernel_width = 2
        """
        local_eul_grid_support_of_lag_grid[...] = (
            np.fabs(local_eul_grid_support_of_lag_grid) / dx
        )
        interp_weights[...] = (
            (0.125 / dx) ** grid_dim
            * (
                (local_eul_grid_support_of_lag_grid[0] < 1.0)
                * (
                    3.0
                    - 2 * local_eul_grid_support_of_lag_grid[0]
                    + np.sqrt(
                        np.fabs(
                            1
                            + 4 * local_eul_grid_support_of_lag_grid[0]
                            - 4 * local_eul_grid_support_of_lag_grid[0] ** 2
                        )
                    )
                )
                + (local_eul_grid_support_of_lag_grid[0] >= 1.0)
                * (local_eul_grid_support_of_lag_grid[0] < 2.0)
                * (
                    5.0
                    - 2 * local_eul_grid_support_of_lag_grid[0]
                    - np.sqrt(
                        np.fabs(
                            -7
                            + 12 * local_eul_grid_support_of_lag_grid[0]
                            - 4 * local_eul_grid_support_of_lag_grid[0] ** 2
                        )
                    )
                )
            )
            * (
                (local_eul_grid_support_of_lag_grid[1] < 1.0)
                * (
                    3.0
                    - 2 * local_eul_grid_support_of_lag_grid[1]
                    + np.sqrt(
                        np.fabs(
                            1
                            + 4 * local_eul_grid_support_of_lag_grid[1]
                            - 4 * local_eul_grid_support_of_lag_grid[1] ** 2
                        )
                    )
                )
                + (local_eul_grid_support_of_lag_grid[1] >= 1.0)
                * (local_eul_grid_support_of_lag_grid[1] < 2.0)
                * (
                    5.0
                    - 2 * local_eul_grid_support_of_lag_grid[1]
                    - np.sqrt(
                        np.fabs(
                            -7
                            + 12 * local_eul_grid_support_of_lag_grid[1]
                            - 4 * local_eul_grid_support_of_lag_grid[1] ** 2
                        )
                    )
                )
            )
        ).astype(real_t)

    return peskin_interpolation_weights_kernel_2d


class MPIGhostSumCommunicator2D:
    """
    Class exclusive for ghost field communication between ranks in eulerian lagrangian
    grid context.
    This communicator enables the influence of lagrangian nodes on eulerian nodes that
    resides in the ghost cell to be transferred and accumulated over to neighbour ranks.
    """

    def __init__(self, ghost_size, mpi_construct):
        self.mpi_construct = mpi_construct
        # Use ghost_size to define offset indices for inner cell
        if ghost_size < 0 and not isinstance(ghost_size, int):
            raise ValueError(
                f"Ghost size {ghost_size} needs to be an integer >= 0"
                "for field communication."
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

        # Create buffers for receiving
        self.row_buffer = np.zeros(
            (self.ghost_size, self.field_size[1]), dtype=mpi_construct.real_t
        )
        self.col_buffer = np.zeros(
            (self.field_size[0], self.ghost_size), dtype=mpi_construct.real_t
        )

    def ghost_sum(self, local_field):
        """
        Add ghost data in each rank to corresponding neighboring rank, followed by
        clearing out the ghost cell values.

        Note: We use blocking calls here to ensure proper accumulation of ghost cells to
        their corresponding ranks.
        """
        # Lines below to make code more literal
        y_axis = 0
        x_axis = 1

        # Along Y: send to next block, receive from previous block
        self.mpi_construct.grid.Send(
            (
                local_field[-self.ghost_size : local_field.shape[0], :],
                1,
                self.row_type,
            ),
            dest=self.mpi_construct.next_grid_along[y_axis],
        )
        self.mpi_construct.grid.Recv(
            self.row_buffer,
            source=self.mpi_construct.previous_grid_along[y_axis],
        )
        # add to field
        local_field[self.ghost_size : 2 * self.ghost_size, :] += self.row_buffer
        # clear buffer values
        self.row_buffer *= 0

        # # Along X: send to next block, receive from previous block
        self.mpi_construct.grid.Send(
            (
                local_field.ravel()[local_field.shape[1] - self.ghost_size :],
                1,
                self.column_type,
            ),
            dest=self.mpi_construct.next_grid_along[x_axis],
        )
        self.mpi_construct.grid.Recv(
            self.col_buffer,
            source=self.mpi_construct.previous_grid_along[x_axis],
        )
        # add to field
        local_field[:, self.ghost_size : 2 * self.ghost_size] += self.col_buffer
        # clear buffer values
        self.col_buffer *= 0

        # Along Y: send to previous block, receive from next block
        self.mpi_construct.grid.Send(
            (
                local_field[0 : self.ghost_size, :],
                1,
                self.row_type,
            ),
            dest=self.mpi_construct.previous_grid_along[y_axis],
        )
        self.mpi_construct.grid.Recv(
            self.row_buffer,
            source=self.mpi_construct.next_grid_along[y_axis],
        )
        # add to field
        local_field[-2 * self.ghost_size : -self.ghost_size, :] += self.row_buffer
        self.row_buffer *= 0

        # Along X: send to previous block, receive from next block
        self.mpi_construct.grid.Send(
            (
                local_field.ravel()[0:],
                1,
                self.column_type,
            ),
            dest=self.mpi_construct.previous_grid_along[x_axis],
        )
        self.mpi_construct.grid.Recv(
            self.col_buffer,
            source=self.mpi_construct.next_grid_along[x_axis],
        )
        # add to field
        local_field[:, -2 * self.ghost_size : -self.ghost_size] += self.col_buffer
        self.col_buffer *= 0

        # zero out ghost cells
        self.clear_ghost_cells(local_field)

    def clear_ghost_cells(self, field):
        field[: self.ghost_size, :] *= 0
        field[-self.ghost_size :, :] *= 0
        field[:, : self.ghost_size] *= 0
        field[:, -self.ghost_size :] *= 0
