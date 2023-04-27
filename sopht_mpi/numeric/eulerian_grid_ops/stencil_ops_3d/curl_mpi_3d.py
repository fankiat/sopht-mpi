"""MPI-supported kernels for computing curl 3D."""
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d import (
    gen_curl_pyst_kernel_3d,
    gen_set_fixed_val_pyst_kernel_3d,
)
from sopht_mpi.utils.mpi_utils import check_valid_ghost_size_and_kernel_support
from mpi4py import MPI


def gen_curl_pyst_mpi_kernel_3d(real_t, mpi_construct, ghost_exchange_communicator):
    """MPI-supported 3D curl kernel generator."""
    curl_pyst_kernel_3d = gen_curl_pyst_kernel_3d(real_t=real_t, reset_ghost_zone=False)
    kernel_support = 1
    # define this here so that ghost size and kernel support is checked during
    # generation phase itself
    gen_curl_pyst_mpi_kernel_3d.kernel_support = kernel_support
    check_valid_ghost_size_and_kernel_support(
        ghost_size=ghost_exchange_communicator.ghost_size,
        kernel_support=gen_curl_pyst_mpi_kernel_3d.kernel_support,
    )

    # for setting values at physical domain boundary
    z_next, y_next, x_next = mpi_construct.next_grid_along
    z_previous, y_previous, x_previous = mpi_construct.previous_grid_along
    set_fixed_val_kernel_3d = gen_set_fixed_val_pyst_kernel_3d(
        real_t=real_t, field_type="vector"
    )

    def curl_pyst_mpi_kernel_3d(curl, field, prefactor):
        """MPI-supported curl in 3D.
        Computes curl (3D vector field) for a 3D vector field
        Assumes curl and vector field is (3, n, n, n) and dx = dy = dz.
        """
        # define kernel support for kernel
        curl_pyst_mpi_kernel_3d.kernel_support = (
            gen_curl_pyst_mpi_kernel_3d.kernel_support
        )
        # define variable for use later
        ghost_size = ghost_exchange_communicator.ghost_size
        # begin ghost comm.
        ghost_exchange_communicator.exchange_vector_field_init(field)

        # crunch interior stencil
        curl_pyst_kernel_3d(
            curl=curl[:, ghost_size:-ghost_size, ghost_size:-ghost_size],
            field=field[:, ghost_size:-ghost_size, ghost_size:-ghost_size],
            prefactor=prefactor,
        )
        # finalise ghost comm.
        ghost_exchange_communicator.exchange_finalise()

        # crunch boundary numbers
        # NOTE: we pass in arrays of width 3 * kernel support size because the
        # interior stencil computation leaves out a width of kernel_support.
        # Since the support needed by the kernel is kernel_support on each side,
        # we need to pass an array of width 3 * kernel_support, starting from
        # index +/-(ghost_size - kernel_support) on the lower and upper end.
        # Pystencils then automatically sets the kernel comp. bounds and
        # crunches numbers in the kernel_support thickness zone at the boundary.
        # Start of Z axis
        curl_pyst_kernel_3d(
            curl=curl[
                :,
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
                ghost_size:-ghost_size,
                ghost_size:-ghost_size,
            ],
            field=field[
                :,
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
                ghost_size:-ghost_size,
                ghost_size:-ghost_size,
            ],
            prefactor=prefactor,
        )
        # End of Z axis
        curl_pyst_kernel_3d(
            curl=curl[
                :,
                -(ghost_size + 2 * kernel_support) : curl.shape[1]
                - (ghost_size - kernel_support),
                ghost_size:-ghost_size,
                ghost_size:-ghost_size,
            ],
            field=field[
                :,
                -(ghost_size + 2 * kernel_support) : field.shape[1]
                - (ghost_size - kernel_support),
                ghost_size:-ghost_size,
                ghost_size:-ghost_size,
            ],
            prefactor=prefactor,
        )
        # Start of Y axis
        curl_pyst_kernel_3d(
            curl=curl[
                :,
                :,
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
                ghost_size:-ghost_size,
            ],
            field=field[
                :,
                :,
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
                ghost_size:-ghost_size,
            ],
            prefactor=prefactor,
        )
        # End of Y axis
        curl_pyst_kernel_3d(
            curl=curl[
                :,
                :,
                -(ghost_size + 2 * kernel_support) : curl.shape[2]
                - (ghost_size - kernel_support),
                ghost_size:-ghost_size,
            ],
            field=field[
                :,
                :,
                -(ghost_size + 2 * kernel_support) : field.shape[2]
                - (ghost_size - kernel_support),
                ghost_size:-ghost_size,
            ],
            prefactor=prefactor,
        )
        # Start of X axis
        curl_pyst_kernel_3d(
            curl=curl[
                :,
                :,
                :,
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
            ],
            field=field[
                :,
                :,
                :,
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
            ],
            prefactor=prefactor,
        )
        # End of X axis
        curl_pyst_kernel_3d(
            curl=curl[
                :,
                :,
                :,
                -(ghost_size + 2 * kernel_support) : curl.shape[3]
                - (ghost_size - kernel_support),
            ],
            field=field[
                :,
                :,
                :,
                -(ghost_size + 2 * kernel_support) : field.shape[3]
                - (ghost_size - kernel_support),
            ],
            prefactor=prefactor,
        )

        # Set physical domain boundary curl to zero based on neighboring block
        boundary_width = 1
        if x_previous == MPI.PROC_NULL:
            set_fixed_val_kernel_3d(
                vector_field=curl[:, :, :, : ghost_size + boundary_width],
                fixed_vals=[0.0, 0.0, 0.0],
            )
        if x_next == MPI.PROC_NULL:
            set_fixed_val_kernel_3d(
                vector_field=curl[:, :, :, -ghost_size - boundary_width :],
                fixed_vals=[0.0, 0.0, 0.0],
            )
        if y_previous == MPI.PROC_NULL:
            set_fixed_val_kernel_3d(
                vector_field=curl[:, :, : ghost_size + boundary_width, :],
                fixed_vals=[0.0, 0.0, 0.0],
            )
        if y_next == MPI.PROC_NULL:
            set_fixed_val_kernel_3d(
                vector_field=curl[:, :, -ghost_size - boundary_width :, :],
                fixed_vals=[0.0, 0.0, 0.0],
            )
        if z_previous == MPI.PROC_NULL:
            set_fixed_val_kernel_3d(
                vector_field=curl[:, : ghost_size + boundary_width, :, :],
                fixed_vals=[0.0, 0.0, 0.0],
            )
        if z_next == MPI.PROC_NULL:
            set_fixed_val_kernel_3d(
                vector_field=curl[:, -ghost_size - boundary_width :, :, :],
                fixed_vals=[0.0, 0.0, 0.0],
            )

    return curl_pyst_mpi_kernel_3d
