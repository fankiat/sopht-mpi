"""MPI-supported kernels for updating vorticity based on velocity forcing in 2D."""
from sopht.numeric.eulerian_grid_ops.stencil_ops_2d import (
    gen_update_vorticity_from_velocity_forcing_pyst_kernel_2d,
)
from sopht_mpi.utils.mpi_utils import check_valid_ghost_size_and_kernel_support


def gen_update_vorticity_from_velocity_forcing_pyst_mpi_kernel_2d(
    real_t, mpi_construct, ghost_exchange_communicator
):
    """MPI-supported update vorticity based on velocity forcing in 2D kernel generator."""
    update_vorticity_from_velocity_forcing_pyst_kernel_2d = (
        gen_update_vorticity_from_velocity_forcing_pyst_kernel_2d(real_t=real_t)
    )
    kernel_support = 1
    # define this here so that ghost size and kernel support is checked during
    # generation phase itself
    gen_update_vorticity_from_velocity_forcing_pyst_mpi_kernel_2d.kernel_support = (
        kernel_support
    )
    check_valid_ghost_size_and_kernel_support(
        ghost_size=ghost_exchange_communicator.ghost_size,
        kernel_support=gen_update_vorticity_from_velocity_forcing_pyst_mpi_kernel_2d.kernel_support,
    )

    def update_vorticity_from_velocity_forcing_pyst_mpi_kernel_2d(
        vorticity_field, velocity_forcing_field, prefactor
    ):
        """MPI-supported kernel for updating vorticity based on velocity forcing
        Updates vorticity_field based on velocity_forcing_field
        vorticity_field += prefactor * curl(velocity_forcing_field)
        prefactor: grid spacing factored out, along with any other multiplier
        Assumes velocity_forcing_field is (2, n, n).
        """

        # define kernel support for kernel
        update_vorticity_from_velocity_forcing_pyst_mpi_kernel_2d.kernel_support = (
            gen_update_vorticity_from_velocity_forcing_pyst_mpi_kernel_2d.kernel_support
        )
        # define variable for use later
        ghost_size = ghost_exchange_communicator.ghost_size
        # begin ghost comm.
        ghost_exchange_communicator.exchange_vector_field_init(velocity_forcing_field)

        # crunch interior stencil
        update_vorticity_from_velocity_forcing_pyst_kernel_2d(
            vorticity_field=vorticity_field[
                ghost_size:-ghost_size, ghost_size:-ghost_size
            ],
            velocity_forcing_field=velocity_forcing_field[
                :, ghost_size:-ghost_size, ghost_size:-ghost_size
            ],
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
        update_vorticity_from_velocity_forcing_pyst_kernel_2d(
            vorticity_field=vorticity_field[
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
                ghost_size:-ghost_size,
            ],
            velocity_forcing_field=velocity_forcing_field[
                :,
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
                ghost_size:-ghost_size,
            ],
            prefactor=prefactor,
        )
        # End of Y axis
        update_vorticity_from_velocity_forcing_pyst_kernel_2d(
            vorticity_field=vorticity_field[
                -(ghost_size + 2 * kernel_support) : vorticity_field.shape[0]
                - (ghost_size - kernel_support),
                ghost_size:-ghost_size,
            ],
            velocity_forcing_field=velocity_forcing_field[
                :,
                -(ghost_size + 2 * kernel_support) : vorticity_field.shape[0]
                - (ghost_size - kernel_support),
                ghost_size:-ghost_size,
            ],
            prefactor=prefactor,
        )
        # Start of X axis
        update_vorticity_from_velocity_forcing_pyst_kernel_2d(
            vorticity_field=vorticity_field[
                :,
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
            ],
            velocity_forcing_field=velocity_forcing_field[
                :,
                :,
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
            ],
            prefactor=prefactor,
        )
        # End of X axis
        update_vorticity_from_velocity_forcing_pyst_kernel_2d(
            vorticity_field=vorticity_field[
                :,
                -(ghost_size + 2 * kernel_support) : vorticity_field.shape[1]
                - (ghost_size - kernel_support),
            ],
            velocity_forcing_field=velocity_forcing_field[
                :,
                :,
                -(ghost_size + 2 * kernel_support) : vorticity_field.shape[1]
                - (ghost_size - kernel_support),
            ],
            prefactor=prefactor,
        )

    return update_vorticity_from_velocity_forcing_pyst_mpi_kernel_2d
