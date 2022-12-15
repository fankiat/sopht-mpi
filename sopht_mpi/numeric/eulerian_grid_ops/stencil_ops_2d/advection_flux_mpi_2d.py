"""MPI-supported kernels for computing advection flux in 2D."""
from sopht.numeric.eulerian_grid_ops.stencil_ops_2d import (
    gen_advection_flux_conservative_eno3_pyst_kernel_2d,
)
from sopht_mpi.utils.mpi_utils import check_valid_ghost_size_and_kernel_support


def gen_advection_flux_conservative_eno3_pyst_mpi_kernel_2d(
    real_t, mpi_construct, ghost_exchange_communicator
):
    advection_flux_pyst_kernel = gen_advection_flux_conservative_eno3_pyst_kernel_2d(
        real_t=real_t
    )
    kernel_support = 2
    # define this here so that ghost size and kernel support is checked during
    # generation phase itself
    gen_advection_flux_conservative_eno3_pyst_mpi_kernel_2d.kernel_support = (
        kernel_support
    )
    check_valid_ghost_size_and_kernel_support(
        ghost_size=ghost_exchange_communicator.ghost_size,
        kernel_support=gen_advection_flux_conservative_eno3_pyst_mpi_kernel_2d.kernel_support,
    )

    def advection_flux_conservative_eno3_pyst_mpi_kernel_2d(
        advection_flux, field, velocity, inv_dx
    ):
        # define kernel support for kernel
        advection_flux_conservative_eno3_pyst_mpi_kernel_2d.kernel_support = (
            gen_advection_flux_conservative_eno3_pyst_mpi_kernel_2d.kernel_support
        )
        # define variable for use later
        ghost_size = ghost_exchange_communicator.ghost_size
        # begin ghost comm.
        ghost_exchange_communicator.exchange_scalar_field_init(field)
        ghost_exchange_communicator.exchange_vector_field_init(velocity)

        # crunch interior stencil
        advection_flux_pyst_kernel(
            advection_flux=advection_flux[
                ghost_size:-ghost_size, ghost_size:-ghost_size
            ],
            field=field[ghost_size:-ghost_size, ghost_size:-ghost_size],
            velocity=velocity[:, ghost_size:-ghost_size, ghost_size:-ghost_size],
            inv_dx=inv_dx,
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
        advection_flux_pyst_kernel(
            advection_flux=advection_flux[
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
                ghost_size:-ghost_size,
            ],
            field=field[
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
                ghost_size:-ghost_size,
            ],
            velocity=velocity[
                :,
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
                ghost_size:-ghost_size,
            ],
            inv_dx=inv_dx,
        )
        # End of Y axis
        advection_flux_pyst_kernel(
            advection_flux=advection_flux[
                -(ghost_size + 2 * kernel_support) : field.shape[0]
                - (ghost_size - kernel_support),
                ghost_size:-ghost_size,
            ],
            field=field[
                -(ghost_size + 2 * kernel_support) : field.shape[0]
                - (ghost_size - kernel_support),
                ghost_size:-ghost_size,
            ],
            velocity=velocity[
                :,
                -(ghost_size + 2 * kernel_support) : field.shape[0]
                - (ghost_size - kernel_support),
                ghost_size:-ghost_size,
            ],
            inv_dx=inv_dx,
        )
        # Start of X axis
        advection_flux_pyst_kernel(
            advection_flux=advection_flux[
                :,
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
            ],
            field=field[
                :,
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
            ],
            velocity=velocity[
                :,
                :,
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
            ],
            inv_dx=inv_dx,
        )
        # End of X axis
        advection_flux_pyst_kernel(
            advection_flux=advection_flux[
                :,
                -(ghost_size + 2 * kernel_support) : field.shape[1]
                - (ghost_size - kernel_support),
            ],
            field=field[
                :,
                -(ghost_size + 2 * kernel_support) : field.shape[1]
                - (ghost_size - kernel_support),
            ],
            velocity=velocity[
                :,
                :,
                -(ghost_size + 2 * kernel_support) : field.shape[1]
                - (ghost_size - kernel_support),
            ],
            inv_dx=inv_dx,
        )

    return advection_flux_conservative_eno3_pyst_mpi_kernel_2d
