"""MPI-supported kernels for computing curl of outplane field in 2D."""
from sopht.numeric.eulerian_grid_ops.stencil_ops_2d import (
    gen_outplane_field_curl_pyst_kernel_2d,
    gen_set_fixed_val_pyst_kernel_2d,
)
from sopht_mpi.utils.mpi_utils import check_valid_ghost_size_and_kernel_support
from mpi4py import MPI


def gen_outplane_field_curl_pyst_mpi_kernel_2d(
    real_t, mpi_construct, ghost_exchange_communicator
):
    """MPI-supported 2D outplane field curl kernel generator."""
    outplane_field_curl_pyst_kernel_2d = gen_outplane_field_curl_pyst_kernel_2d(
        real_t=real_t
    )
    kernel_support = 1
    # define this here so that ghost size and kernel support is checked during
    # generation phase itself
    gen_outplane_field_curl_pyst_mpi_kernel_2d.kernel_support = kernel_support
    check_valid_ghost_size_and_kernel_support(
        ghost_size=ghost_exchange_communicator.ghost_size,
        kernel_support=gen_outplane_field_curl_pyst_mpi_kernel_2d.kernel_support,
    )

    # for setting values at physical domain boundary
    y_next, x_next = mpi_construct.next_grid_along
    y_previous, x_previous = mpi_construct.previous_grid_along
    set_fixed_val_kernel_2d = gen_set_fixed_val_pyst_kernel_2d(
        real_t=real_t, field_type="vector"
    )

    def outplane_field_curl_pyst_mpi_kernel_2d(curl, field, prefactor):
        """MPI-supporteed outplane field curl in 2D.
        Computes curl of outplane 2D vector field (field)
        into vector 2D inplane field (curl_x, curl_y).
        Used for psi ---> velocity
        Assumes curl field is (2, n, n).
        """
        # define kernel support for kernel
        outplane_field_curl_pyst_mpi_kernel_2d.kernel_support = (
            gen_outplane_field_curl_pyst_mpi_kernel_2d.kernel_support
        )
        # define variable for use later
        ghost_size = ghost_exchange_communicator.ghost_size
        # begin ghost comm.
        ghost_exchange_communicator.exchange_init(field, mpi_construct)

        # crunch interior stencil
        outplane_field_curl_pyst_kernel_2d(
            curl=curl[:, ghost_size:-ghost_size, ghost_size:-ghost_size],
            field=field[ghost_size:-ghost_size, ghost_size:-ghost_size],
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
        # Start of Y axis
        outplane_field_curl_pyst_kernel_2d(
            curl=curl[
                :,
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
                ghost_size:-ghost_size,
            ],
            field=field[
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
                ghost_size:-ghost_size,
            ],
            prefactor=prefactor,
        )
        # End of Y axis
        outplane_field_curl_pyst_kernel_2d(
            curl=curl[
                :,
                -(ghost_size + 2 * kernel_support) : field.shape[0]
                - (ghost_size - kernel_support),
                ghost_size:-ghost_size,
            ],
            field=field[
                -(ghost_size + 2 * kernel_support) : field.shape[0]
                - (ghost_size - kernel_support),
                ghost_size:-ghost_size,
            ],
            prefactor=prefactor,
        )
        # Start of X axis
        outplane_field_curl_pyst_kernel_2d(
            curl=curl[
                :,
                :,
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
            ],
            field=field[
                :,
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
            ],
            prefactor=prefactor,
        )
        # End of X axis
        outplane_field_curl_pyst_kernel_2d(
            curl=curl[
                :,
                :,
                -(ghost_size + 2 * kernel_support) : field.shape[1]
                - (ghost_size - kernel_support),
            ],
            field=field[
                :,
                -(ghost_size + 2 * kernel_support) : field.shape[1]
                - (ghost_size - kernel_support),
            ],
            prefactor=prefactor,
        )

        # Code below works
        # NOTE: disable set boundary in outfield curl kernel
        # set zero after taking curl of inner cell

        # crunch interior stencil
        # outplane_field_curl_pyst_kernel_2d(
        #     curl=curl,
        #     field=field,
        #     prefactor=prefactor,
        # )

        # Set boundary curl to zero when neighboring block is MPI.PROC_NULL
        # first along X
        if x_previous == MPI.PROC_NULL:
            # curl[:, :, : ghost_size + 1] = 0.0
            set_fixed_val_kernel_2d(
                vector_field=curl[:, :, : ghost_size + 1], fixed_vals=[0.0, 0.0]
            )
        if x_next == MPI.PROC_NULL:
            # curl[:, :, -ghost_size - 1 :] = 0.0
            set_fixed_val_kernel_2d(
                vector_field=curl[:, :, -ghost_size - 1 :], fixed_vals=[0.0, 0.0]
            )

        # then along Y
        if y_previous == MPI.PROC_NULL:
            # curl[:, : ghost_size + 1, :] = 0.0
            set_fixed_val_kernel_2d(
                vector_field=curl[:, : ghost_size + 1, :], fixed_vals=[0.0, 0.0]
            )
        if y_next == MPI.PROC_NULL:
            # curl[:, -ghost_size - 1 :, :] = 0.0
            set_fixed_val_kernel_2d(
                vector_field=curl[:, -ghost_size - 1 :, :], fixed_vals=[0.0, 0.0]
            )

    return outplane_field_curl_pyst_mpi_kernel_2d
