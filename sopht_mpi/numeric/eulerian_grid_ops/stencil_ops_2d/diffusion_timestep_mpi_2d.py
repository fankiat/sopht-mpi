"""MPI-supported kernels for performing diffusion timestep in 2D."""
from sopht_mpi.numeric.eulerian_grid_ops.stencil_ops_2d.diffusion_flux_mpi_2d import (
    gen_diffusion_flux_pyst_mpi_kernel_2d,
)
from sopht.numeric.eulerian_grid_ops.stencil_ops_2d.elementwise_ops_2d import (
    gen_elementwise_sum_pyst_kernel_2d,
    gen_set_fixed_val_pyst_kernel_2d,
)
from sopht_mpi.utils.mpi_utils import check_valid_ghost_size_and_kernel_support


def gen_diffusion_timestep_euler_forward_pyst_mpi_kernel_2d(
    real_t,
    mpi_construct,
    ghost_exchange_communicator,
):
    """MPI-supported 2D diffusion euler forward timestep generator"""
    elementwise_sum_pyst_kernel_2d = gen_elementwise_sum_pyst_kernel_2d(real_t=real_t)
    set_fixed_val_pyst_kernel_2d = gen_set_fixed_val_pyst_kernel_2d(real_t=real_t)
    diffusion_flux_mpi_kernel_2d = gen_diffusion_flux_pyst_mpi_kernel_2d(
        real_t=real_t,
        mpi_construct=mpi_construct,
        ghost_exchange_communicator=ghost_exchange_communicator,
    )
    # Since the generated function has not been called yet, we cannot access the
    # kernel support variable. However, because of the way we implemented the
    # generator function, we can retrieve it from the generator function.
    # Define the variables below so that we check ghost size and kernel support
    # during generation phase itself
    kernel_support = gen_diffusion_flux_pyst_mpi_kernel_2d.kernel_support
    gen_diffusion_timestep_euler_forward_pyst_mpi_kernel_2d.kernel_support = (
        kernel_support
    )
    check_valid_ghost_size_and_kernel_support(
        ghost_size=ghost_exchange_communicator.ghost_size,
        kernel_support=gen_diffusion_timestep_euler_forward_pyst_mpi_kernel_2d.kernel_support,
    )

    def diffusion_timestep_euler_forward_pyst_mpi_kernel_2d(
        field, diffusion_flux, nu_dt_by_dx2
    ):
        """2D Diffusion Euler forward timestep.

        Performs an inplace diffusion timestep in 2D using Euler forward,
        for a 2D field (n, n).
        """
        # define and store kernel support size
        diffusion_timestep_euler_forward_pyst_mpi_kernel_2d.kernel_support = (
            gen_diffusion_flux_pyst_mpi_kernel_2d.kernel_support
        )
        set_fixed_val_pyst_kernel_2d(field=diffusion_flux, fixed_val=0)
        diffusion_flux_mpi_kernel_2d(
            diffusion_flux=diffusion_flux,
            field=field,
            prefactor=nu_dt_by_dx2,
        )
        elementwise_sum_pyst_kernel_2d(
            sum_field=field, field_1=field, field_2=diffusion_flux
        )

    return diffusion_timestep_euler_forward_pyst_mpi_kernel_2d
