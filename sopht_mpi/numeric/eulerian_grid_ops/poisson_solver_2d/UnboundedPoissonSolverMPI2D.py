"""MPI-supported unbounded Poisson solver kernels 2D via mpi4py-fft."""
import numpy as np
from mpi4py import MPI
from mpi4py_fft import newDistArray
from sopht.numeric.eulerian_grid_ops.stencil_ops_2d.elementwise_ops_2d import (
    gen_elementwise_complex_product_pyst_kernel_2d,
    gen_set_fixed_val_pyst_kernel_2d,
)
from sopht_mpi.numeric.eulerian_grid_ops.poisson_solver_2d.fft_mpi_2d import FFTMPI2D


class UnboundedPoissonSolverMPI2D:
    """
    MPI-supported class for solving unbounded Poisson in 2D via mpi4py-fft.

    Note: We need ghost size here to maintain contiguous memory when passing in
    local fields with ghost cells for poisson solve.
    """

    def __init__(
        self,
        grid_size_y,
        grid_size_x,
        mpi_construct,
        ghost_size,
        x_range=1.0,
        real_t=np.float64,
    ):
        """Class initialiser."""
        self.mpi_construct = mpi_construct
        self.grid_size_y = grid_size_y
        self.grid_size_x = grid_size_x
        self.x_range = x_range
        self.y_range = x_range * (grid_size_y / grid_size_x)
        self.dx = real_t(x_range / grid_size_x)
        self.real_t = real_t
        self.mpi_domain_doubling_comm = MPIDomainDoublingCommunicator2D(
            ghost_size=ghost_size, mpi_construct=self.mpi_construct
        )

        self.fft_construct = FFTMPI2D(
            # 2 because FFTs taken on doubled domain
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
        self.fourier_greens_function_times_dx_squared = (
            self.domain_doubled_fourier_buffer * (self.dx**2)
        )
        self.gen_elementwise_operation_kernels()

    def construct_fourier_greens_function_field(self):
        """Construct the local grid of unbounded Greens function."""
        # Lines below to make code more literal
        y_axis = 0
        x_axis = 1

        # define local xy-coord mesh
        local_x_grid = newDistArray(pfft=self.fft_construct.fft, forward_output=False)
        local_y_grid = newDistArray(pfft=self.fft_construct.fft, forward_output=False)

        # get start and end indices of local grid relative to global grid
        global_start_idx = np.array(local_x_grid.substart)
        local_grid_size = local_x_grid.shape
        global_end_idx = global_start_idx + local_grid_size

        # Generate local xy mesh based on local grid location
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
        local_x_grid, local_y_grid = np.meshgrid(local_x, local_y)

        # Generate greens function field
        even_reflected_distance_field = np.sqrt(
            np.minimum(local_x_grid, 2 * self.x_range - local_x_grid) ** 2
            + np.minimum(local_y_grid, 2 * self.y_range - local_y_grid) ** 2
        )
        greens_function_field = newDistArray(
            pfft=self.fft_construct.fft, forward_output=False
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            greens_function_field = -np.log(even_reflected_distance_field) / (2 * np.pi)
        # Regularization term
        if np.all(global_start_idx == 0):
            greens_function_field[0, 0] = -(
                2 * np.log(self.dx / np.sqrt(np.pi)) - 1
            ) / (4 * np.pi)

        # take forward transform of greens function field
        self.rfft(
            field=greens_function_field,
            fourier_field=self.domain_doubled_fourier_buffer,
        )

    def gen_elementwise_operation_kernels(self):
        """Compile funcs for elementwise ops on buffers."""
        # this operate on domain doubled arrays
        self.set_fixed_val_kernel_2d = gen_set_fixed_val_pyst_kernel_2d(
            real_t=self.real_t
        )
        # this one operates on fourier buffer
        self.elementwise_complex_product_kernel_2d = (
            gen_elementwise_complex_product_pyst_kernel_2d(real_t=self.real_t)
        )

    def solve(self, solution_field, rhs_field):
        """Unbounded Poisson solver method.
        Solves Poisson equation in 2D: -del^2(solution_field) = rhs_field
        for unbounded domain using Greens function convolution and
        domain doubling trick (Hockney and Eastwood).
        """

        self.set_fixed_val_kernel_2d(field=self.domain_doubled_buffer, fixed_val=0)
        self.mpi_domain_doubling_comm.copy_to_doubled_domain(
            local_field=rhs_field,
            local_doubled_field=self.domain_doubled_buffer,
        )

        self.rfft(
            field=self.domain_doubled_buffer,
            fourier_field=self.domain_doubled_fourier_buffer,
        )

        # Greens function convolution
        self.elementwise_complex_product_kernel_2d(
            product_field=self.convolution_buffer,
            field_1=self.domain_doubled_fourier_buffer,
            field_2=self.fourier_greens_function_times_dx_squared,
        )

        self.irfft(
            fourier_field=self.convolution_buffer,
            inv_fourier_field=self.domain_doubled_buffer,
        )

        self.mpi_domain_doubling_comm.copy_from_doubled_domain(
            local_doubled_field=self.domain_doubled_buffer,
            local_field=solution_field,
        )


class MPIDomainDoublingCommunicator2D:
    """
    Class exclusive for field communication between actual and doubled domain.
    Since mpi4py-fft always require that at least one axis of a multidimensional
    array remains aligned (non-distributed), in 2D that translates to slab
    decomposition, which will be assumed for the communication operations here.
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

        distributed_dim = np.where(np.array(mpi_construct.grid_topology) != 1)[0]
        if len(distributed_dim) > 1:
            raise ValueError(
                f"Distributed in {len(distributed_dim)} dimensions."
                "Only 1 dimension is allowed to be distributed in 2D (slab decomp)"
            )
        self.distributed_dim = distributed_dim[0]
        # Copying from actual to doubled domain -> two receives, one send
        # Copying from doubled to actual domain -> two sends, one receive
        # Total requests needed = 3 requests
        self.num_requests = 3
        self.comm_requests = [
            0,
        ] * self.num_requests

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
        # For recving into doubled domain, one each for (a) first and (b) second
        # half of the doubled block of data in doubled domain
        # (a) First half (start offset assumes slab decomp)
        starts = [0] * mpi_construct.grid_dim
        self.recv_from_actual_to_doubled_domain_first_half_type = (
            self.mpi_construct.dtype_generator.Create_subarray(
                sizes=self.field_doubled_grid_size,
                subsizes=self.field_grid_size,
                starts=starts,
            )
        )
        self.recv_from_actual_to_doubled_domain_first_half_type.Commit()
        # (b) Second half (start offset assumes slab decomp)
        starts = [0, 0]
        starts[self.distributed_dim] = self.field_grid_size[self.distributed_dim]
        self.recv_from_actual_to_doubled_domain_second_half_type = (
            self.mpi_construct.dtype_generator.Create_subarray(
                sizes=self.field_doubled_grid_size,
                subsizes=self.field_grid_size,
                starts=starts,
            )
        )
        self.recv_from_actual_to_doubled_domain_second_half_type.Commit()

        # (2) Buffers for doubled -> actual domain communication
        # For sending from doubled domain, one each for (a) first and (b) second
        # half of the doubled block of data in doubled domain
        # (a) First half (start offset assumes slab decomp)
        starts = [0] * mpi_construct.grid_dim
        self.send_from_doubled_to_actual_domain_first_half_type = (
            self.mpi_construct.dtype_generator.Create_subarray(
                sizes=self.field_doubled_grid_size,
                subsizes=self.field_grid_size,
                starts=starts,
            )
        )
        self.send_from_doubled_to_actual_domain_first_half_type.Commit()
        # (b) Second half (start offset assumes slab decomp)
        starts = [0, 0]
        starts[self.distributed_dim] = self.field_grid_size[self.distributed_dim]
        self.send_from_doubled_to_actual_domain_second_half_type = (
            self.mpi_construct.dtype_generator.Create_subarray(
                sizes=self.field_doubled_grid_size,
                subsizes=self.field_grid_size,
                starts=starts,
            )
        )
        self.send_from_doubled_to_actual_domain_second_half_type.Commit()

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
        two receives for each rank in doubled domain.
        """
        coord = self.mpi_construct.grid.coords[self.distributed_dim]
        send_offset = coord // 2 + coord % 2
        dest = self.mpi_construct.grid.Shift(self.distributed_dim, send_offset)[0]
        self.comm_requests[0] = self.mpi_construct.grid.Isend(
            (local_field, self.send_from_actual_to_doubled_domain_type),
            dest=dest,
        )
        if coord < self.mpi_construct.grid_topology[self.distributed_dim] / 2:
            recv_offset_first_half = 2 * coord - coord
            source_first_half = self.mpi_construct.grid.Shift(
                self.distributed_dim, recv_offset_first_half
            )[1]
            self.comm_requests[1] = self.mpi_construct.grid.Irecv(
                (
                    local_doubled_field,
                    self.recv_from_actual_to_doubled_domain_first_half_type,
                ),
                source=source_first_half,
            )
            recv_offset_second_half = 2 * coord + 1 - coord
            source_second_half = self.mpi_construct.grid.Shift(
                self.distributed_dim, recv_offset_second_half
            )[1]
            self.comm_requests[2] = self.mpi_construct.grid.Irecv(
                (
                    local_doubled_field,
                    self.recv_from_actual_to_doubled_domain_second_half_type,
                ),
                source=source_second_half,
            )
        else:
            self.comm_requests[1] = MPI.REQUEST_NULL
            self.comm_requests[2] = MPI.REQUEST_NULL

        MPI.Request.Waitall(self.comm_requests)

    def copy_from_doubled_domain(self, local_doubled_field, local_field):
        """
        Two sends for each rank from doubled domain,
        one receive for each rank in actual domain.
        """
        coord = self.mpi_construct.grid.coords[self.distributed_dim]
        if coord < self.mpi_construct.grid_topology[self.distributed_dim] / 2:
            send_offset_first_half = 2 * coord - coord
            dest_first_half = self.mpi_construct.grid.Shift(
                self.distributed_dim, send_offset_first_half
            )[1]
            self.comm_requests[0] = self.mpi_construct.grid.Isend(
                (
                    local_doubled_field,
                    self.send_from_doubled_to_actual_domain_first_half_type,
                ),
                dest=dest_first_half,
            )
            send_offset_second_half = 2 * coord + 1 - coord
            dest_second_half = self.mpi_construct.grid.Shift(
                self.distributed_dim, send_offset_second_half
            )[1]
            self.comm_requests[1] = self.mpi_construct.grid.Isend(
                (
                    local_doubled_field,
                    self.send_from_doubled_to_actual_domain_second_half_type,
                ),
                dest=dest_second_half,
            )
        else:
            self.comm_requests[0] = MPI.REQUEST_NULL
            self.comm_requests[1] = MPI.REQUEST_NULL
        recv_offset = coord // 2 + coord % 2
        source = self.mpi_construct.grid.Shift(self.distributed_dim, recv_offset)[0]
        self.comm_requests[2] = self.mpi_construct.grid.Irecv(
            (local_field, self.recv_from_doubled_to_actual_domain_type),
            source=source,
        )
        MPI.Request.Waitall(self.comm_requests)
