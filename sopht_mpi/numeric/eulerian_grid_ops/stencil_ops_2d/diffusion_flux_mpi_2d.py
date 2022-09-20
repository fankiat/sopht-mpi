"""MPI-supported kernels for computing diffusion flux in 2D."""
from sopht.numeric.eulerian_grid_ops.stencil_ops_2d import (
    gen_diffusion_flux_pyst_kernel_2d,
)
from sopht_mpi.utils.mpi_utils import check_valid_ghost_size_and_kernel_support


def gen_diffusion_flux_pyst_mpi_kernel_2d(
    real_t, mpi_construct, ghost_exchange_communicator
):
    # Note currently I'm generating these for arbit size arrays, we ca optimise this
    # more by generating fixed size for the interior stencil and arbit size for
    # boundary crunching
    diffusion_flux_pyst_kernel = gen_diffusion_flux_pyst_kernel_2d(real_t=real_t)

    def diffusion_flux_pyst_mpi_kernel_2d(
        diffusion_flux,
        field,
        prefactor,
    ):
        # define kernel support for kernel
        diffusion_flux_pyst_mpi_kernel_2d.kernel_support = 1
        # define variable for use later
        kernel_support = diffusion_flux_pyst_mpi_kernel_2d.kernel_support
        ghost_size = ghost_exchange_communicator.ghost_size
        check_valid_ghost_size_and_kernel_support(
            ghost_size=ghost_size,
            kernel_support=kernel_support,
        )

        # begin ghost comm.
        ghost_exchange_communicator.exchange_init(field, mpi_construct)

        # crunch interior stencil
        diffusion_flux_pyst_kernel(
            diffusion_flux=diffusion_flux[
                ghost_size:-ghost_size, ghost_size:-ghost_size
            ],
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
        diffusion_flux_pyst_kernel(
            diffusion_flux=diffusion_flux[
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
        diffusion_flux_pyst_kernel(
            diffusion_flux=diffusion_flux[
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
        diffusion_flux_pyst_kernel(
            diffusion_flux=diffusion_flux[
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
        diffusion_flux_pyst_kernel(
            diffusion_flux=diffusion_flux[
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

    return diffusion_flux_pyst_mpi_kernel_2d
