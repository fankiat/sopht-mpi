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
        periodic_domain=False,
        real_t=np.float64,
        rank_distribution=None,
    ):
        # grid/problem dimensions
        self.grid_dim = 3
        self.real_t = real_t
        # Set the MPI dtype generator based on precision
        self.dtype_generator = MPI.FLOAT if real_t == np.float32 else MPI.DOUBLE
        # Setup MPI environment
        self.periodic_domain = periodic_domain
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
            # Log warning here for more generic use.
            # mpi4py-fft will take care of throwing errors if fft is invoked later.
            logger.warning(
                f"Rank distribution {self.rank_distribution} needs to be "
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
            self.grid_topology, periods=self.periodic_domain, reorder=False
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
    for communication based on ghost_size (determined from stencil width of the kernel).

    Here we refer the ghost cells following the terminologies for cubes
    (https://en.wikipedia.org/wiki/Cube): "The has 6 faces, 12 edges, and 8 vertices."
    Consider a face of the cube in the ghost zone as illustrated below.

    v e e e e e v <- vertex (corner)
    e f f f f f e
    e f f f f f e <- edge (side)
    e f f f f f e
    v e e e e e v

    (v) vertex ghost cell
    (e) edge ghost cell
    (f) face ghost cell

    Note
    ---
    `full_exchange` allows for exchanges ghost cells on the vertices, edges and faces of
    the local domain (see illustration above). The full exchange mode is required only
    when Eulerian-to-Lagrangian (structured-to-unstructured) grid interpolation is
    performed. Similar to the 2D ghost communicator, we won't need the full exchange
    mode even when the interpolation is performed if the domain decomposition mode
    is slabs (unless a periodic flow simulator is employed). In the case where domain
    decomposition mode is pencils (in 3D, mpi4py-fft allows for both slabs and pencils),
    full exchange is needed if the structured-to-unstructured grid interpolation is
    performed.
    """

    def __init__(self, ghost_size, mpi_construct, full_exchange=True):
        # extra width needed for kernel computation
        if ghost_size <= 0 and not isinstance(ghost_size, int):
            raise ValueError(
                f"Ghost size {ghost_size} needs to be an integer > 0"
                "for calling ghost communication."
            )
        self.ghost_size = ghost_size
        self.mpi_construct = mpi_construct
        self.full_exchange = full_exchange
        self.grid_coord = np.array(self.mpi_construct.grid.coords)

        # Initialize data types
        self.init_datatypes()

        # Initialize requests list for non-blocking comm
        self.comm_requests = []

        if self.full_exchange:
            self.exchange_scalar_field_init = self.exchange_scalar_field_full_init
        else:
            self.exchange_scalar_field_init = self.exchange_scalar_field_faces_init

    def init_datatypes(self):
        """
        Set datatypes for ghost communication on 6 faces, 12 edges and 8 vertices.
        Each datatype will have a send and recv, giving a total of
        (6 + 12 + 8) * 2 = 52 datatypes.

        Note
        ---
        Here we are using the Create_subarray approach, each type for send/recv needs to
        be initialized based on their starting index location. Alternatively, we can use
        Create_vector approach which is more involved, but perhaps faster way.
        Keeping subarray approach for now for its readibility and easy implementation.

        For simplicity and readability, we use coordinate system (z, y, x) as used in
        grid topology to define direction for send and recv. For example, when sending
        along the (0, -1, +1) direction (i.e. along XY-plane in the positive X and
        negative Y direction), we denote the data type name as `send_along_0_ny_px_type`
        where 0, ny and px denotes 0 in Z, negative in Y, and positive in X coord shift,
        respectively.
        """
        # define field_size variable for local field size (which includes ghost)
        self.field_size_with_ghost = (
            self.mpi_construct.local_grid_size + 2 * self.ghost_size
        )
        self.field_size = self.mpi_construct.local_grid_size

        # (1) For faces data
        # Since there are 6 faces on a cube, each with a send and recv datatypes, we
        # need 6 * 2 = 12 data types
        # Comm. along +X direction: send to (0, 0, +1) with recv from (0, 0, -1) block
        self.send_to_0_0_px_type = self.mpi_construct.dtype_generator.Create_subarray(
            sizes=self.field_size_with_ghost,
            subsizes=[self.field_size[0], self.field_size[1], self.ghost_size],
            starts=[
                self.ghost_size,
                self.ghost_size,
                self.field_size_with_ghost[2] - 2 * self.ghost_size,
            ],
        )
        self.send_to_0_0_px_type.Commit()
        self.recv_from_0_0_nx_type = self.mpi_construct.dtype_generator.Create_subarray(
            sizes=self.field_size_with_ghost,
            subsizes=[self.field_size[0], self.field_size[1], self.ghost_size],
            starts=[self.ghost_size, self.ghost_size, 0],
        )
        self.recv_from_0_0_nx_type.Commit()
        # Comm. along -X direction: send to (-1, 0, 0) with recv from (+1, 0, 0) block
        self.send_to_0_0_nx_type = self.mpi_construct.dtype_generator.Create_subarray(
            sizes=self.field_size_with_ghost,
            subsizes=[self.field_size[0], self.field_size[1], self.ghost_size],
            starts=[self.ghost_size, self.ghost_size, self.ghost_size],
        )
        self.send_to_0_0_nx_type.Commit()
        self.recv_from_0_0_px_type = self.mpi_construct.dtype_generator.Create_subarray(
            sizes=self.field_size_with_ghost,
            subsizes=[self.field_size[0], self.field_size[1], self.ghost_size],
            starts=[
                self.ghost_size,
                self.ghost_size,
                self.field_size_with_ghost[2] - self.ghost_size,
            ],
        )
        self.recv_from_0_0_px_type.Commit()

        # Comm. along +Y direction: send to (0, +1, 0) with recv from (0, -1, 0) block
        self.send_to_0_py_0_type = self.mpi_construct.dtype_generator.Create_subarray(
            sizes=self.field_size_with_ghost,
            subsizes=[self.field_size[0], self.ghost_size, self.field_size[2]],
            starts=[
                self.ghost_size,
                self.field_size_with_ghost[1] - 2 * self.ghost_size,
                self.ghost_size,
            ],
        )
        self.send_to_0_py_0_type.Commit()
        self.recv_from_0_ny_0_type = self.mpi_construct.dtype_generator.Create_subarray(
            sizes=self.field_size_with_ghost,
            subsizes=[self.field_size[0], self.ghost_size, self.field_size[2]],
            starts=[self.ghost_size, 0, self.ghost_size],
        )
        self.recv_from_0_ny_0_type.Commit()
        # Comm. along -Y direction: send to (0, -1, 0) with recv from (0, +1, 0) block
        self.send_to_0_ny_0_type = self.mpi_construct.dtype_generator.Create_subarray(
            sizes=self.field_size_with_ghost,
            subsizes=[self.field_size[0], self.ghost_size, self.field_size[2]],
            starts=[self.ghost_size, self.ghost_size, self.ghost_size],
        )
        self.send_to_0_ny_0_type.Commit()
        self.recv_from_0_py_0_type = self.mpi_construct.dtype_generator.Create_subarray(
            sizes=self.field_size_with_ghost,
            subsizes=[self.field_size[0], self.ghost_size, self.field_size[2]],
            starts=[
                self.ghost_size,
                self.field_size_with_ghost[1] - self.ghost_size,
                self.ghost_size,
            ],
        )
        self.recv_from_0_py_0_type.Commit()

        # Comm. along +Z direction: send to (+1, 0, 0) with recv from (-1, 0, 0) block
        self.send_to_pz_0_0_type = self.mpi_construct.dtype_generator.Create_subarray(
            sizes=self.field_size_with_ghost,
            subsizes=[self.ghost_size, self.field_size[1], self.field_size[2]],
            starts=[
                self.field_size_with_ghost[0] - 2 * self.ghost_size,
                self.ghost_size,
                self.ghost_size,
            ],
        )
        self.send_to_pz_0_0_type.Commit()
        self.recv_from_nz_0_0_type = self.mpi_construct.dtype_generator.Create_subarray(
            sizes=self.field_size_with_ghost,
            subsizes=[self.ghost_size, self.field_size[1], self.field_size[2]],
            starts=[0, self.ghost_size, self.ghost_size],
        )
        self.recv_from_nz_0_0_type.Commit()
        # Comm. along -Z direction: send to (-1, 0, 0) with recv from (+1, 0, 0) block
        self.send_to_nz_0_0_type = self.mpi_construct.dtype_generator.Create_subarray(
            sizes=self.field_size_with_ghost,
            subsizes=[self.ghost_size, self.field_size[1], self.field_size[2]],
            starts=[self.ghost_size, self.ghost_size, self.ghost_size],
        )
        self.send_to_nz_0_0_type.Commit()
        self.recv_from_pz_0_0_type = self.mpi_construct.dtype_generator.Create_subarray(
            sizes=self.field_size_with_ghost,
            subsizes=[self.ghost_size, self.field_size[1], self.field_size[2]],
            starts=[
                self.field_size_with_ghost[0] - self.ghost_size,
                self.ghost_size,
                self.ghost_size,
            ],
        )
        self.recv_from_pz_0_0_type.Commit()

        if self.full_exchange:
            # (2) For edges
            # Since there are 12 edges on a cube, each with a send and recv datatypes,
            # we need 12 * 2 = 24 datatypes
            # Comm. along +Y, +X: send to (0, +1, +1) with recv from (0, -1, -1) block
            self.send_to_0_py_px_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.field_size[0], self.ghost_size, self.ghost_size],
                    starts=[
                        self.ghost_size,
                        self.field_size_with_ghost[1] - 2 * self.ghost_size,
                        self.field_size_with_ghost[2] - 2 * self.ghost_size,
                    ],
                )
            )
            self.send_to_0_py_px_type.Commit()
            self.recv_from_0_ny_nx_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.field_size[0], self.ghost_size, self.ghost_size],
                    starts=[self.ghost_size, 0, 0],
                )
            )
            self.recv_from_0_ny_nx_type.Commit()
            # Comm. along -Y, +X: send to (0, -1, +1) with recv from (0, +1, -1) block
            self.send_to_0_ny_px_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.field_size[0], self.ghost_size, self.ghost_size],
                    starts=[
                        self.ghost_size,
                        self.ghost_size,
                        self.field_size_with_ghost[2] - 2 * self.ghost_size,
                    ],
                )
            )
            self.send_to_0_ny_px_type.Commit()
            self.recv_from_0_py_nx_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.field_size[0], self.ghost_size, self.ghost_size],
                    starts=[
                        self.ghost_size,
                        self.field_size_with_ghost[1] - self.ghost_size,
                        0,
                    ],
                )
            )
            self.recv_from_0_py_nx_type.Commit()
            # Comm. along +Y, -X: send to (0, +1, -1) with recv from (0, -1, +1) block
            self.send_to_0_py_nx_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.field_size[0], self.ghost_size, self.ghost_size],
                    starts=[
                        self.ghost_size,
                        self.field_size_with_ghost[1] - 2 * self.ghost_size,
                        self.ghost_size,
                    ],
                )
            )
            self.send_to_0_py_nx_type.Commit()
            self.recv_from_0_ny_px_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.field_size[0], self.ghost_size, self.ghost_size],
                    starts=[
                        self.ghost_size,
                        0,
                        self.field_size_with_ghost[2] - self.ghost_size,
                    ],
                )
            )
            self.recv_from_0_ny_px_type.Commit()
            # Comm. along -Y, -X: send to (0, -1, -1) with recv from (0, +1, +1) block
            self.send_to_0_ny_nx_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.field_size[0], self.ghost_size, self.ghost_size],
                    starts=[self.ghost_size, self.ghost_size, self.ghost_size],
                )
            )
            self.send_to_0_ny_nx_type.Commit()
            self.recv_from_0_py_px_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.field_size[0], self.ghost_size, self.ghost_size],
                    starts=[
                        self.ghost_size,
                        self.field_size_with_ghost[1] - self.ghost_size,
                        self.field_size_with_ghost[2] - self.ghost_size,
                    ],
                )
            )
            self.recv_from_0_py_px_type.Commit()

            # Comm. along +Z, +X: send to (+1, 0, +1) with recv from (-1, 0, -1) block
            self.send_to_pz_0_px_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.field_size[1], self.ghost_size],
                    starts=[
                        self.field_size_with_ghost[0] - 2 * self.ghost_size,
                        self.ghost_size,
                        self.field_size_with_ghost[2] - 2 * self.ghost_size,
                    ],
                )
            )
            self.send_to_pz_0_px_type.Commit()
            self.recv_from_nz_0_nx_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.field_size[1], self.ghost_size],
                    starts=[0, self.ghost_size, 0],
                )
            )
            self.recv_from_nz_0_nx_type.Commit()
            # Comm. along -Z, +X: send to (-1, 0, +1) with recv from (+1, 0, -1) block
            self.send_to_nz_0_px_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.field_size[1], self.ghost_size],
                    starts=[
                        self.ghost_size,
                        self.ghost_size,
                        self.field_size_with_ghost[2] - 2 * self.ghost_size,
                    ],
                )
            )
            self.send_to_nz_0_px_type.Commit()
            self.recv_from_pz_0_nx_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.field_size[1], self.ghost_size],
                    starts=[
                        self.field_size_with_ghost[0] - self.ghost_size,
                        self.ghost_size,
                        0,
                    ],
                )
            )
            self.recv_from_pz_0_nx_type.Commit()
            # Comm. along +Z, -X: send to (+1, 0, -1) with recv from (-1, 0, +1) block
            self.send_to_pz_0_nx_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.field_size[1], self.ghost_size],
                    starts=[
                        self.field_size_with_ghost[0] - 2 * self.ghost_size,
                        self.ghost_size,
                        self.ghost_size,
                    ],
                )
            )
            self.send_to_pz_0_nx_type.Commit()
            self.recv_from_nz_0_px_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.field_size[1], self.ghost_size],
                    starts=[
                        0,
                        self.ghost_size,
                        self.field_size_with_ghost[2] - self.ghost_size,
                    ],
                )
            )
            self.recv_from_nz_0_px_type.Commit()
            # Comm. along -Z, -X: send to (-1, 0, -1) with recv from (+1, 0, +1) block
            self.send_to_nz_0_nx_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.field_size[1], self.ghost_size],
                    starts=[self.ghost_size, self.ghost_size, self.ghost_size],
                )
            )
            self.send_to_nz_0_nx_type.Commit()
            self.recv_from_pz_0_px_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.field_size[1], self.ghost_size],
                    starts=[
                        self.field_size_with_ghost[0] - self.ghost_size,
                        self.ghost_size,
                        self.field_size_with_ghost[2] - self.ghost_size,
                    ],
                )
            )
            self.recv_from_pz_0_px_type.Commit()

            # Comm. along +Z, +Y: send to (+1, +1, 0) with recv from (-1, -1, 0) block
            self.send_to_pz_py_0_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.ghost_size, self.field_size[2]],
                    starts=[
                        self.field_size_with_ghost[0] - 2 * self.ghost_size,
                        self.field_size_with_ghost[1] - 2 * self.ghost_size,
                        self.ghost_size,
                    ],
                )
            )
            self.send_to_pz_py_0_type.Commit()
            self.recv_from_nz_ny_0_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.ghost_size, self.field_size[2]],
                    starts=[0, 0, self.ghost_size],
                )
            )
            self.recv_from_nz_ny_0_type.Commit()
            # Comm. along -Z, +Y: send to (-1, +1, 0) with recv from (+1, -1, 0) block
            self.send_to_nz_py_0_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.ghost_size, self.field_size[2]],
                    starts=[
                        self.ghost_size,
                        self.field_size_with_ghost[1] - 2 * self.ghost_size,
                        self.ghost_size,
                    ],
                )
            )
            self.send_to_nz_py_0_type.Commit()
            self.recv_from_pz_ny_0_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.ghost_size, self.field_size[2]],
                    starts=[
                        self.field_size_with_ghost[0] - self.ghost_size,
                        0,
                        self.ghost_size,
                    ],
                )
            )
            self.recv_from_pz_ny_0_type.Commit()
            # Comm. along +Z, -Y: send to (+1, -1, 0) with recv from (-1, +1, 0) block
            self.send_to_pz_ny_0_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.ghost_size, self.field_size[2]],
                    starts=[
                        self.field_size_with_ghost[0] - 2 * self.ghost_size,
                        self.ghost_size,
                        self.ghost_size,
                    ],
                )
            )
            self.send_to_pz_ny_0_type.Commit()
            self.recv_from_nz_py_0_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.ghost_size, self.field_size[2]],
                    starts=[
                        0,
                        self.field_size_with_ghost[1] - self.ghost_size,
                        self.ghost_size,
                    ],
                )
            )
            self.recv_from_nz_py_0_type.Commit()
            # Comm. along -Z, -Y: send to (-1, -1, 0) with recv from (+1, +1, 0) block
            self.send_to_nz_ny_0_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.ghost_size, self.field_size[2]],
                    starts=[self.ghost_size, self.ghost_size, self.ghost_size],
                )
            )
            self.send_to_nz_ny_0_type.Commit()
            self.recv_from_pz_py_0_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.ghost_size, self.field_size[2]],
                    starts=[
                        self.field_size_with_ghost[0] - self.ghost_size,
                        self.field_size_with_ghost[1] - self.ghost_size,
                        self.ghost_size,
                    ],
                )
            )
            self.recv_from_pz_py_0_type.Commit()

            # (3) For vertices
            # Since there are 8 vertices on a cube, each with a send and recv datatypes,
            # we need 8 * 2 = 16 datatypes
            # Comm. along +Z, +Y, +X: send to (+1, +1, +1) with recv from (-1, -1, -1) block
            self.send_to_pz_py_px_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.ghost_size, self.ghost_size],
                    starts=[
                        self.field_size_with_ghost[0] - 2 * self.ghost_size,
                        self.field_size_with_ghost[1] - 2 * self.ghost_size,
                        self.field_size_with_ghost[2] - 2 * self.ghost_size,
                    ],
                )
            )
            self.send_to_pz_py_px_type.Commit()
            self.recv_from_nz_ny_nx_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.ghost_size, self.ghost_size],
                    starts=[0, 0, 0],
                )
            )
            self.recv_from_nz_ny_nx_type.Commit()
            # Comm. along -Z, +Y, +X: send to (-1, +1, +1) with recv from (+1, -1, -1) block
            self.send_to_nz_py_px_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.ghost_size, self.ghost_size],
                    starts=[
                        self.ghost_size,
                        self.field_size_with_ghost[1] - 2 * self.ghost_size,
                        self.field_size_with_ghost[2] - 2 * self.ghost_size,
                    ],
                )
            )
            self.send_to_nz_py_px_type.Commit()
            self.recv_from_pz_ny_nx_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.ghost_size, self.ghost_size],
                    starts=[self.field_size_with_ghost[0] - self.ghost_size, 0, 0],
                )
            )
            self.recv_from_pz_ny_nx_type.Commit()
            # Comm. along +Z, -Y, +X: send to (+1, -1, +1) with recv from (-1, +1, -1) block
            self.send_to_pz_ny_px_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.ghost_size, self.ghost_size],
                    starts=[
                        self.field_size_with_ghost[0] - 2 * self.ghost_size,
                        self.ghost_size,
                        self.field_size_with_ghost[2] - 2 * self.ghost_size,
                    ],
                )
            )
            self.send_to_pz_ny_px_type.Commit()
            self.recv_from_nz_py_nx_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.ghost_size, self.ghost_size],
                    starts=[0, self.field_size_with_ghost[1] - self.ghost_size, 0],
                )
            )
            self.recv_from_nz_py_nx_type.Commit()
            # Comm. along +Z, +Y, -X: send to (+1, +1, -1) with recv from (-1, -1, +1) block
            self.send_to_pz_py_nx_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.ghost_size, self.ghost_size],
                    starts=[
                        self.field_size_with_ghost[0] - 2 * self.ghost_size,
                        self.field_size_with_ghost[1] - 2 * self.ghost_size,
                        self.ghost_size,
                    ],
                )
            )
            self.send_to_pz_py_nx_type.Commit()
            self.recv_from_nz_ny_px_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.ghost_size, self.ghost_size],
                    starts=[0, 0, self.field_size_with_ghost[2] - self.ghost_size],
                )
            )
            self.recv_from_nz_ny_px_type.Commit()
            # Comm. along -Z, -Y, +X: send to (-1, -1, +1) with recv from (+1, +1, -1) block
            self.send_to_nz_ny_px_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.ghost_size, self.ghost_size],
                    starts=[
                        self.ghost_size,
                        self.ghost_size,
                        self.field_size_with_ghost[2] - 2 * self.ghost_size,
                    ],
                )
            )
            self.send_to_nz_ny_px_type.Commit()
            self.recv_from_pz_py_nx_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.ghost_size, self.ghost_size],
                    starts=[
                        self.field_size_with_ghost[0] - self.ghost_size,
                        self.field_size_with_ghost[1] - self.ghost_size,
                        0,
                    ],
                )
            )
            self.recv_from_pz_py_nx_type.Commit()
            # Comm. along -Z, +Y, -X: send to (-1, +1, -1) with recv from (+1, -1, +1) block
            self.send_to_nz_py_nx_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.ghost_size, self.ghost_size],
                    starts=[
                        self.ghost_size,
                        self.field_size_with_ghost[1] - 2 * self.ghost_size,
                        self.ghost_size,
                    ],
                )
            )
            self.send_to_nz_py_nx_type.Commit()
            self.recv_from_pz_ny_px_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.ghost_size, self.ghost_size],
                    starts=[
                        self.field_size_with_ghost[0] - self.ghost_size,
                        0,
                        self.field_size_with_ghost[2] - self.ghost_size,
                    ],
                )
            )
            self.recv_from_pz_ny_px_type.Commit()
            # Comm. along +Z, -Y, -X: send to (+1, -1, -1) with recv from (-1, +1, +1) block
            self.send_to_pz_ny_nx_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.ghost_size, self.ghost_size],
                    starts=[
                        self.field_size_with_ghost[0] - 2 * self.ghost_size,
                        self.ghost_size,
                        self.ghost_size,
                    ],
                )
            )
            self.send_to_pz_ny_nx_type.Commit()
            self.recv_from_nz_py_px_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.ghost_size, self.ghost_size],
                    starts=[
                        0,
                        self.field_size_with_ghost[1] - self.ghost_size,
                        self.field_size_with_ghost[2] - self.ghost_size,
                    ],
                )
            )
            self.recv_from_nz_py_px_type.Commit()
            # Comm. along -Z, -Y, -X: send to (-1, -1, -1) with recv from (+1, +1, +1) block
            self.send_to_nz_ny_nx_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.ghost_size, self.ghost_size],
                    starts=[self.ghost_size, self.ghost_size, self.ghost_size],
                )
            )
            self.send_to_nz_ny_nx_type.Commit()
            self.recv_from_pz_py_px_type = (
                self.mpi_construct.dtype_generator.Create_subarray(
                    sizes=self.field_size_with_ghost,
                    subsizes=[self.ghost_size, self.ghost_size, self.ghost_size],
                    starts=[
                        self.field_size_with_ghost[0] - self.ghost_size,
                        self.field_size_with_ghost[1] - self.ghost_size,
                        self.field_size_with_ghost[2] - self.ghost_size,
                    ],
                )
            )
            self.recv_from_pz_py_px_type.Commit()

    def _get_diagonally_shifted_coord_rank(self, coord_shift):
        """Helper function to get diagonally shifted coords"""
        shifted_coord = self.grid_coord + np.array(coord_shift)
        if not self.mpi_construct.periodic_domain and (
            np.any(shifted_coord >= self.mpi_construct.grid_topology)
            or np.any(shifted_coord < 0)
        ):
            # The shifted coord is out of non-periodic domain
            rank = MPI.PROC_NULL
        else:
            # Periodic domain is automatically taken care of in mpi cartersian grid
            rank = self.mpi_construct.grid.Get_cart_rank(shifted_coord)
        return rank

    def exchange_scalar_field_faces_init(self, local_field):
        """
        Exchange scalar field ghost data on faces between neighbors.
        """
        # Lines below to make code more literal
        z_axis = 0
        y_axis = 1
        x_axis = 2

        # Comm. along +X direction: send to (0, 0, +1) with recv from (0, 0, -1) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_0_0_px_type),
                dest=self.mpi_construct.next_grid_along[x_axis],
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_0_0_nx_type),
                source=self.mpi_construct.previous_grid_along[x_axis],
            )
        )
        # Comm. along -X direction: send to (0, 0, -1) with recv from (0, 0, +1) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_0_0_nx_type),
                dest=self.mpi_construct.previous_grid_along[x_axis],
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_0_0_px_type),
                source=self.mpi_construct.next_grid_along[x_axis],
            )
        )
        # Comm. along +Y direction: send to (0, +1, 0) with recv from (0, -1, 0) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_0_py_0_type),
                dest=self.mpi_construct.next_grid_along[y_axis],
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_0_ny_0_type),
                source=self.mpi_construct.previous_grid_along[y_axis],
            )
        )
        # Comm. along -Y direction: send to (0, -1, 0) with recv from (0, +1, 0) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_0_ny_0_type),
                dest=self.mpi_construct.previous_grid_along[y_axis],
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_0_py_0_type),
                source=self.mpi_construct.next_grid_along[y_axis],
            )
        )
        # Comm. along +Z direction: send to (+1, 0, 0) with recv from (-1, 0, 0) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_pz_0_0_type),
                dest=self.mpi_construct.next_grid_along[z_axis],
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_nz_0_0_type),
                source=self.mpi_construct.previous_grid_along[z_axis],
            )
        )
        # Comm. along -Z direction: send to (-1, 0, 0) with recv from (+1, 0, 0) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_nz_0_0_type),
                dest=self.mpi_construct.previous_grid_along[z_axis],
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_pz_0_0_type),
                source=self.mpi_construct.next_grid_along[z_axis],
            )
        )

    def exchange_scalar_field_edges_init(self, local_field):
        """
        Exchange scalar field ghost data on edges between neighbors.
        """
        # Comm. along +Y, +X: send to (0, +1, +1) with recv from (0, -1, -1) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_0_py_px_type),
                dest=self._get_diagonally_shifted_coord_rank(coord_shift=[0, 1, 1]),
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_0_ny_nx_type),
                source=self._get_diagonally_shifted_coord_rank(coord_shift=[0, -1, -1]),
            )
        )
        # Comm. along -Y, +X: send to (0, -1, +1) with recv from (0, +1, -1) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_0_ny_px_type),
                dest=self._get_diagonally_shifted_coord_rank(coord_shift=[0, -1, 1]),
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_0_py_nx_type),
                source=self._get_diagonally_shifted_coord_rank(coord_shift=[0, 1, -1]),
            )
        )
        # Comm. along +Y, -X: send to (0, +1, -1) with recv from (0, -1, +1) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_0_py_nx_type),
                dest=self._get_diagonally_shifted_coord_rank(coord_shift=[0, 1, -1]),
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_0_ny_px_type),
                source=self._get_diagonally_shifted_coord_rank(coord_shift=[0, -1, 1]),
            )
        )
        # Comm. along -Y, -X: send to (0, -1, -1) with recv from (0, +1, +1) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_0_ny_nx_type),
                dest=self._get_diagonally_shifted_coord_rank(coord_shift=[0, -1, -1]),
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_0_py_px_type),
                source=self._get_diagonally_shifted_coord_rank(coord_shift=[0, 1, 1]),
            )
        )
        # Comm. along +Z, +X: send to (+1, 0, +1) with recv from (-1, 0, -1) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_pz_0_px_type),
                dest=self._get_diagonally_shifted_coord_rank(coord_shift=[1, 0, 1]),
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_nz_0_nx_type),
                source=self._get_diagonally_shifted_coord_rank(coord_shift=[-1, 0, -1]),
            )
        )
        # Comm. along -Z, +X: send to (-1, 0, +1) with recv from (+1, 0, -1) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_nz_0_px_type),
                dest=self._get_diagonally_shifted_coord_rank(coord_shift=[-1, 0, 1]),
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_pz_0_nx_type),
                source=self._get_diagonally_shifted_coord_rank(coord_shift=[1, 0, -1]),
            )
        )
        # Comm. along +Z, -X: send to (+1, 0, -1) with recv from (-1, 0, +1) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_pz_0_nx_type),
                dest=self._get_diagonally_shifted_coord_rank(coord_shift=[1, 0, -1]),
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_nz_0_px_type),
                source=self._get_diagonally_shifted_coord_rank(coord_shift=[-1, 0, 1]),
            )
        )
        # Comm. along -Z, -X: send to (-1, 0, -1) with recv from (+1, 0, +1) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_nz_0_nx_type),
                dest=self._get_diagonally_shifted_coord_rank(coord_shift=[-1, 0, -1]),
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_pz_0_px_type),
                source=self._get_diagonally_shifted_coord_rank(coord_shift=[1, 0, 1]),
            )
        )
        # Comm. along +Z, +Y: send to (+1, +1, 0) with recv from (-1, -1, 0) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_pz_py_0_type),
                dest=self._get_diagonally_shifted_coord_rank(coord_shift=[1, 1, 0]),
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_nz_ny_0_type),
                source=self._get_diagonally_shifted_coord_rank(coord_shift=[-1, -1, 0]),
            )
        )
        # Comm. along -Z, +Y: send to (-1, +1, 0) with recv from (+1, -1, 0) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_nz_py_0_type),
                dest=self._get_diagonally_shifted_coord_rank(coord_shift=[-1, 1, 0]),
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_pz_ny_0_type),
                source=self._get_diagonally_shifted_coord_rank(coord_shift=[1, -1, 0]),
            )
        )
        # Comm. along +Z, -Y: send to (+1, -1, 0) with recv from (-1, +1, 0) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_pz_ny_0_type),
                dest=self._get_diagonally_shifted_coord_rank(coord_shift=[1, -1, 0]),
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_nz_py_0_type),
                source=self._get_diagonally_shifted_coord_rank(coord_shift=[-1, 1, 0]),
            )
        )
        # Comm. along -Z, -Y: send to (-1, -1, 0) with recv from (+1, +1, 0) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_nz_ny_0_type),
                dest=self._get_diagonally_shifted_coord_rank(coord_shift=[-1, -1, 0]),
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_pz_py_0_type),
                source=self._get_diagonally_shifted_coord_rank(coord_shift=[1, 1, 0]),
            )
        )

    def exchange_scalar_field_vertices_init(self, local_field):
        """
        Exchange scalar field ghost data on vertices between neighbors.
        """
        # Comm. along +Z, +Y, +X: send to (+1, +1, +1) with recv from (-1, -1, -1) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_pz_py_px_type),
                dest=self._get_diagonally_shifted_coord_rank(coord_shift=[1, 1, 1]),
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_nz_ny_nx_type),
                source=self._get_diagonally_shifted_coord_rank(
                    coord_shift=[-1, -1, -1]
                ),
            )
        )
        # Comm. along -Z, +Y, +X: send to (-1, +1, +1) with recv from (+1, -1, -1) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_nz_py_px_type),
                dest=self._get_diagonally_shifted_coord_rank(coord_shift=[-1, 1, 1]),
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_pz_ny_nx_type),
                source=self._get_diagonally_shifted_coord_rank(coord_shift=[1, -1, -1]),
            )
        )
        # Comm. along +Z, -Y, +X: send to (+1, -1, +1) with recv from (-1, +1, -1) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_pz_ny_px_type),
                dest=self._get_diagonally_shifted_coord_rank(coord_shift=[1, -1, 1]),
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_nz_py_nx_type),
                source=self._get_diagonally_shifted_coord_rank(coord_shift=[-1, 1, -1]),
            )
        )
        # Comm. along +Z, +Y, -X: send to (+1, +1, -1) with recv from (-1, -1, +1) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_pz_py_nx_type),
                dest=self._get_diagonally_shifted_coord_rank(coord_shift=[1, 1, -1]),
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_nz_ny_px_type),
                source=self._get_diagonally_shifted_coord_rank(coord_shift=[-1, -1, 1]),
            )
        )
        # Comm. along -Z, -Y, +X: send to (-1, -1, +1) with recv from (+1, +1, -1) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_nz_ny_px_type),
                dest=self._get_diagonally_shifted_coord_rank(coord_shift=[-1, -1, 1]),
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_pz_py_nx_type),
                source=self._get_diagonally_shifted_coord_rank(coord_shift=[1, 1, -1]),
            )
        )
        # Comm. along -Z, +Y, -X: send to (-1, +1, -1) with recv from (+1, -1, +1) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_nz_py_nx_type),
                dest=self._get_diagonally_shifted_coord_rank(coord_shift=[-1, 1, -1]),
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_pz_ny_px_type),
                source=self._get_diagonally_shifted_coord_rank(coord_shift=[1, -1, 1]),
            )
        )
        # Comm. along +Z, -Y, -X: send to (+1, -1, -1) with recv from (-1, +1, +1) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_pz_ny_nx_type),
                dest=self._get_diagonally_shifted_coord_rank(coord_shift=[1, -1, -1]),
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_nz_py_px_type),
                source=self._get_diagonally_shifted_coord_rank(coord_shift=[-1, 1, 1]),
            )
        )
        # Comm. along -Z, -Y, -X: send to (-1, -1, -1) with recv from (+1, +1, +1) block
        self.comm_requests.append(
            self.mpi_construct.grid.Isend(
                (local_field, self.send_to_nz_ny_nx_type),
                dest=self._get_diagonally_shifted_coord_rank(coord_shift=[-1, -1, -1]),
            )
        )
        self.comm_requests.append(
            self.mpi_construct.grid.Irecv(
                (local_field, self.recv_from_pz_py_px_type),
                source=self._get_diagonally_shifted_coord_rank(coord_shift=[1, 1, 1]),
            )
        )

    def exchange_scalar_field_full_init(self, local_field):
        """
        Exchange scalar field ghost data including all faces, edges and vertices between
        neighbors.
        """
        self.exchange_scalar_field_faces_init(local_field)
        self.exchange_scalar_field_edges_init(local_field)
        self.exchange_scalar_field_vertices_init(local_field)

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
