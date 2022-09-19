from sopht.numeric.eulerian_grid_ops.stencil_ops_2d import (
    gen_diffusion_flux_pyst_kernel_2d,
)


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
        # begin ghost comm.
        ghost_exchange_communicator.exchange_init(field, mpi_construct)

        # crunch interior stencil
        ghost_size = ghost_exchange_communicator.ghost_size
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
        # NOTE we pass in arrays of width 3 * ghost_size because the
        # interior stencil computation leaves out a width of ghost_size.
        # Since the support needed by the kernel is 2 * ghost_size on both
        # sides, we need to pass an array of width 3 * ghost_size.
        # Pystencils then automatically sets the kernel comp. bounds and
        # crunches numbers in the ghost_size thickness zone at the boundary.
        ghost_size = ghost_exchange_communicator.ghost_size
        kernel_max_range_y = field.shape[0]
        kernel_max_range_x = field.shape[1]

        # Start of Y axis
        diffusion_flux_pyst_kernel(
            diffusion_flux=diffusion_flux[0 : 3 * ghost_size, ghost_size:-ghost_size],
            field=field[0 : 3 * ghost_size, ghost_size:-ghost_size],
            prefactor=prefactor,
        )
        # End of Y axis
        diffusion_flux_pyst_kernel(
            diffusion_flux=diffusion_flux[
                -3 * ghost_size : kernel_max_range_y, ghost_size:-ghost_size
            ],
            field=field[-3 * ghost_size : kernel_max_range_y, ghost_size:-ghost_size],
            prefactor=prefactor,
        )
        # Start of X axis
        diffusion_flux_pyst_kernel(
            diffusion_flux=diffusion_flux[:, 0 : 3 * ghost_size],
            field=field[:, 0 : 3 * ghost_size],
            prefactor=prefactor,
        )
        # End of X axis
        diffusion_flux_pyst_kernel(
            diffusion_flux=diffusion_flux[:, -3 * ghost_size : kernel_max_range_x],
            field=field[:, -3 * ghost_size : kernel_max_range_x],
            prefactor=prefactor,
        )

    return diffusion_flux_pyst_mpi_kernel_2d
