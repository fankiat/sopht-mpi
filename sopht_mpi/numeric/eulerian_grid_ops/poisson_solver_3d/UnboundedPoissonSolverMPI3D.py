"""MPI-supported unbounded Poisson solver kernels 3D via mpi4py-fft."""
import numpy as np
from mpi4py import MPI
from mpi4py_fft import newDistArray
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d.elementwise_ops_3d import (
    gen_elementwise_complex_product_pyst_kernel_3d,
    gen_set_fixed_val_pyst_kernel_3d,
)
from sopht_mpi.numeric.eulerian_grid_ops.poisson_solver_3d.fft_mpi_3d import FFTMPI3D
import itertools


class UnboundedPoissonSolverMPI3D:
    """
    MPI-supported class for solving unbounded Poisson in 3D via mpi4py-fft.

    Note: We need ghost size here to maintain contiguous memory when passing in
    local fields with ghost cells for poisson solve.
    """

    def __init__(
        self,
        grid_size_z,
        grid_size_y,
        grid_size_x,
        mpi_construct,
        ghost_size,
        x_range=1.0,
        real_t=np.float64,
    ):
        """Class initialiser."""
        self.mpi_construct = mpi_construct
        self.grid_size_z = grid_size_z
        self.grid_size_y = grid_size_y
        self.grid_size_x = grid_size_x
        self.x_range = x_range
        self.y_range = x_range * (grid_size_y / grid_size_x)
        self.z_range = x_range * (grid_size_z / grid_size_x)
        self.dx = real_t(x_range / grid_size_x)
        self.real_t = real_t
        self.mpi_domain_doubling_comm = MPIDomainDoublingCommunicator3D(
            ghost_size=ghost_size, mpi_construct=self.mpi_construct
        )

        self.fft_construct = FFTMPI3D(
            # 2 because FFTs taken on doubled domain
            grid_size_z=2 * grid_size_z,
            grid_size_y=2 * grid_size_y,
            grid_size_x=2 * grid_size_x,
            mpi_construct=mpi_construct,
            real_t=real_t,
        )
        self.rfft = self.fft_construct.forward
        self.irfft = self.fft_construct.backward
        self.domain_doubled_buffer = self.fft_construct.field_buffer
        self.domain_doubled_fourier_buffer = self.fft_construct.fourier_field_buffer
        self.convolution_buffer = newDistArray(
            pfft=self.fft_construct.fft, forward_output=True
        )
        self.construct_fourier_greens_function_field()
        self.fourier_greens_function_times_dx_cubed = (
            self.domain_doubled_fourier_buffer * (self.dx**3)
        )
        self.gen_elementwise_operation_kernels()

    def construct_fourier_greens_function_field(self):
        """Construct the local grid of unbounded Greens function."""
        # Lines below to make code more literal
        z_axis = 0
        y_axis = 1
        x_axis = 2

        # get start and end indices of local grid relative to global grid
        # this information is stored in domain_doubled_buffer, which is a distarray
        # initialized in FFTMPI3D
        global_start_idx = np.array(self.domain_doubled_buffer.substart)
        local_grid_size = self.domain_doubled_buffer.shape
        global_end_idx = global_start_idx + local_grid_size

        # Generate local xyz mesh based on local grid location
        local_x = np.linspace(
            global_start_idx[x_axis] * self.dx,
            (global_end_idx[x_axis] - 1) * self.dx,
            local_grid_size[x_axis],
        ).astype(self.real_t)
        local_y = np.linspace(
            global_start_idx[y_axis] * self.dx,
            (global_end_idx[y_axis] - 1) * self.dx,
            local_grid_size[y_axis],
        ).astype(self.real_t)
        local_z = np.linspace(
            global_start_idx[z_axis] * self.dx,
            (global_end_idx[z_axis] - 1) * self.dx,
            local_grid_size[z_axis],
        ).astype(self.real_t)
        local_z_grid, local_y_grid, local_x_grid = np.meshgrid(
            local_z, local_y, local_x, indexing="ij"
        )

        # Generate greens function field
        even_reflected_distance_field = np.sqrt(
            np.minimum(local_x_grid, 2 * self.x_range - local_x_grid) ** 2
            + np.minimum(local_y_grid, 2 * self.y_range - local_y_grid) ** 2
            + np.minimum(local_z_grid, 2 * self.z_range - local_z_grid) ** 2
        )
        greens_function_field = newDistArray(
            pfft=self.fft_construct.fft, forward_output=False
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            greens_function_field = (1 / even_reflected_distance_field) / (4 * np.pi)
        # Regularization term (straight from PPM)
        if np.all(global_start_idx == 0):
            greens_function_field[0, 0, 0] = 1 / (4 * np.pi * self.dx)

        # take forward transform of greens function field
        self.rfft(
            field=greens_function_field,
            fourier_field=self.domain_doubled_fourier_buffer,
        )

    def gen_elementwise_operation_kernels(self):
        """Compile funcs for elementwise ops on buffers."""
        # this operate on domain doubled arrays
        self.set_fixed_val_kernel_3d = gen_set_fixed_val_pyst_kernel_3d(
            real_t=self.real_t
        )
        # this one operates on fourier buffer
        self.elementwise_complex_product_kernel_3d = (
            gen_elementwise_complex_product_pyst_kernel_3d(real_t=self.real_t)
        )

    def solve(self, solution_field, rhs_field):
        """Unbounded Poisson solver method.

        Solves Poisson equation in 3D: -del^2(solution_field) = rhs_field
        for unbounded domain using Greens function convolution and
        domain doubling trick (Hockney and Eastwood).
        """

        self.set_fixed_val_kernel_3d(field=self.domain_doubled_buffer, fixed_val=0)
        self.mpi_domain_doubling_comm.copy_to_doubled_domain(
            local_field=rhs_field,
            local_doubled_field=self.domain_doubled_buffer,
        )

        self.rfft(
            field=self.domain_doubled_buffer,
            fourier_field=self.domain_doubled_fourier_buffer,
        )

        # Greens function convolution
        self.elementwise_complex_product_kernel_3d(
            product_field=self.convolution_buffer,
            field_1=self.domain_doubled_fourier_buffer,
            field_2=self.fourier_greens_function_times_dx_cubed,
        )

        self.irfft(
            fourier_field=self.convolution_buffer,
            inv_fourier_field=self.domain_doubled_buffer,
        )

        self.mpi_domain_doubling_comm.copy_from_doubled_domain(
            local_doubled_field=self.domain_doubled_buffer,
            local_field=solution_field,
        )

    def vector_field_solve(self, solution_vector_field, rhs_vector_field):
        """Unbounded Poisson solver method (vector field solve).

        Solves 3 Poisson equations in 3D for each component:
        -del^2(solution_vector_field) = rhs_vector_field for unbounded domain using
        Greens function convolution and domain doubling trick (Hockney and Eastwood).
        """
        self.solve(
            solution_field=solution_vector_field[0], rhs_field=rhs_vector_field[0]
        )
        self.solve(
            solution_field=solution_vector_field[1], rhs_field=rhs_vector_field[1]
        )
        self.solve(
            solution_field=solution_vector_field[2], rhs_field=rhs_vector_field[2]
        )


class MPIDomainDoublingCommunicator3D:
    """
    Class exclusive for field communication between actual and doubled domain.

    Note: Since mpi4py-fft always require that at least one axis of a multidimensional
    array remains aligned (non-distributed), in 3D that translates to either slab or
    pencil decomposition. Nonetheless, the communication operations here is extensible
    to higher dimensions such as block decomposition.
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

        # define local grid sizes for actual and doubled domain
        self.field_grid_size = mpi_construct.local_grid_size
        self.field_doubled_grid_size = mpi_construct.local_grid_size * 2

        self.grid_topology = np.array(mpi_construct.grid_topology)
        # Handles the case when only a single process is spawned
        if self.mpi_construct.size == 1:
            self.distributed_dim = []
        else:
            self.distributed_dim = np.where(self.grid_topology != 1)[0]
        num_distributed_dim = len(self.distributed_dim)
        if num_distributed_dim >= mpi_construct.grid_dim:
            raise ValueError(
                f"Distributed in {num_distributed_dim} dimensions."
                "Only a max of 2 dimensions are allowed to be distributed in 3D"
                "(slab/pencil decomp)"
            )

        # Actual-to-doubled domain copy -> (2 ** num_distributed_dim) receives, 1 send
        # Doubled-to-actual domain copy -> (2 ** num_distributed_dim) sends, 1 receive
        # Total requests needed = (2 ** num_distributed_dim) + 1 requests
        self.num_requests = (2**num_distributed_dim) + 1
        self.comm_requests = [0] * self.num_requests

        # Get the Cartesian product of which direction to offset the subarray.
        # We need this so that we can call communication consistently for both slab /
        # pencil decomposition.
        # For slab decomposition, we will have only first half and second half, which we
        # denote as [0] and [1] in the distributed direction
        # For pencil decomp, we will have 4 quarters, which we will denote as [0,0],
        # [0,1], [1,0], and [1,1], with elements in the pairs denoting offset in each
        # distributed directions.
        # These numbers denote in which direction to offset the subarrays and thus tells
        # us where each data to send to / recv from.
        # Note that this approach is extensible to blocks decomp (distributed in all 3
        # directions), but we won't be needing that here since mpi4py-fft requires min
        # one axis to be aligned.
        self.subarray_start_offsets = []
        for offset in itertools.product([0, 1], repeat=num_distributed_dim):
            self.subarray_start_offsets.append(offset)

        # communication datatypes initialization
        # (1) Buffers for actual -> doubled domain communication
        # For sending from actual domain. Local actual domain contains ghost cells
        starts = [self.ghost_size] * mpi_construct.grid_dim
        self.send_from_actual_to_doubled_domain_type = (
            self.mpi_construct.dtype_generator.Create_subarray(
                sizes=self.field_grid_size + 2 * self.ghost_size,
                subsizes=self.field_grid_size,
                starts=starts,
            )
        )
        self.send_from_actual_to_doubled_domain_type.Commit()
        # For recving into doubled domain, it depends on the grid decomposition
        # Slab: two halves of the doubled block of data in doubled domain.
        # Pencil: four quarters of the doubled block of data in double domain.
        self.recv_from_actual_to_doubled_domain_type = {}
        # Initialize data type for each subarray with the corresponding offset
        for offsets in self.subarray_start_offsets:
            # Get start offset
            starts = [0] * self.mpi_construct.grid_dim
            for i, dim in enumerate(self.distributed_dim):
                starts[dim] = self.field_grid_size[dim] * offsets[i]
            # Initialize datatype
            self.recv_from_actual_to_doubled_domain_type[
                offsets
            ] = self.mpi_construct.dtype_generator.Create_subarray(
                sizes=self.field_doubled_grid_size,
                subsizes=self.field_grid_size,
                starts=starts,
            )
            self.recv_from_actual_to_doubled_domain_type[offsets].Commit()

        # (2) Buffers for doubled -> actual domain communication
        # For sending from doubled domain, it depends on the grid decomposition
        # For slab decomp, we have one each for every half of the doubled block of data
        # in doubled domain.
        # For pencil decomp, we have one each for every quarter of the doubled block of
        # data in double domain.
        self.send_from_doubled_to_actual_domain_type = {}
        # Initialize data type for each subarray with the corresponding offset
        for offsets in self.subarray_start_offsets:
            starts = [0] * self.mpi_construct.grid_dim
            for j, dim in enumerate(self.distributed_dim):
                starts[dim] = self.field_grid_size[dim] * offsets[j]
            self.send_from_doubled_to_actual_domain_type[
                offsets
            ] = self.mpi_construct.dtype_generator.Create_subarray(
                sizes=self.field_doubled_grid_size,
                subsizes=self.field_grid_size,
                starts=starts,
            )
            self.send_from_doubled_to_actual_domain_type[offsets].Commit()
        # For recving into actual domain. Local actual domain contains ghost cells
        starts = [self.ghost_size] * mpi_construct.grid_dim
        self.recv_from_doubled_to_actual_domain_type = (
            self.mpi_construct.dtype_generator.Create_subarray(
                sizes=self.field_grid_size + 2 * self.ghost_size,
                subsizes=self.field_grid_size,
                starts=starts,
            )
        )
        self.recv_from_doubled_to_actual_domain_type.Commit()

    def copy_to_doubled_domain(self, local_field, local_doubled_field):
        """
        One send for each rank from actual domain,
        two (slab) / four (pencil) receives for each rank in doubled domain.
        """
        coord = np.array(self.mpi_construct.grid.coords)

        # Send current local data to corresponding rank
        dest_coord = coord // 2
        dest = self.mpi_construct.grid.Get_cart_rank(dest_coord)
        self.comm_requests[0] = self.mpi_construct.grid.Isend(
            (local_field, self.send_from_actual_to_doubled_domain_type),
            dest=dest,
        )
        # Receive data from corresponding ranks
        if np.all(coord < self.grid_topology / 2):
            for i, offsets in enumerate(self.subarray_start_offsets):
                source_coord = coord.copy()
                for j, dim in enumerate(self.distributed_dim):
                    source_coord[dim] = 2 * coord[dim] + offsets[j]
                source = self.mpi_construct.grid.Get_cart_rank(source_coord)
                self.comm_requests[i + 1] = self.mpi_construct.grid.Irecv(
                    (
                        local_doubled_field,
                        self.recv_from_actual_to_doubled_domain_type[offsets],
                    ),
                    source=source,
                )
        else:
            # Nothing to receive for ranks lying beyond the actual domain
            for i in range(1, self.num_requests):
                self.comm_requests[i] = MPI.REQUEST_NULL

        MPI.Request.Waitall(self.comm_requests)

    def copy_from_doubled_domain(self, local_doubled_field, local_field):
        """
        Two (slab) / four (pencil) sends for each rank in doubled domain,
        one receive for each rank in actual domain.
        """
        coord = np.array(self.mpi_construct.grid.coords)

        # Send data to corresponding ranks
        if np.all(coord < self.grid_topology / 2):
            for i, offsets in enumerate(self.subarray_start_offsets):
                send_coord = coord.copy()
                for j, dim in enumerate(self.distributed_dim):
                    send_coord[dim] = 2 * coord[dim] + offsets[j]
                dest = self.mpi_construct.grid.Get_cart_rank(send_coord)
                self.comm_requests[i] = self.mpi_construct.grid.Isend(
                    (
                        local_doubled_field,
                        self.send_from_doubled_to_actual_domain_type[offsets],
                    ),
                    dest=dest,
                )
        else:
            # Nothing to send for ranks lying beyond the actual domain
            for i in range(self.num_requests - 1):
                self.comm_requests[i] = MPI.REQUEST_NULL

        # Receive data from corresponding rank to local array
        source_coord = coord // 2
        source = self.mpi_construct.grid.Get_cart_rank(source_coord)
        self.comm_requests[-1] = self.mpi_construct.grid.Irecv(
            (local_field, self.recv_from_doubled_to_actual_domain_type),
            source=source,
        )
        MPI.Request.Waitall(self.comm_requests)
