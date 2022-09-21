"""MPI-supported kernels for performing advection timestep in 2D."""
from sopht_mpi.numeric.eulerian_grid_ops.stencil_ops_2d.advection_flux_mpi_2d import (
    gen_advection_flux_conservative_eno3_pyst_mpi_kernel_2d,
)
from sopht.numeric.eulerian_grid_ops.stencil_ops_2d.elementwise_ops_2d import (
    gen_elementwise_sum_pyst_kernel_2d,
    gen_set_fixed_val_pyst_kernel_2d,
)
from sopht_mpi.utils.mpi_utils import check_valid_ghost_size_and_kernel_support


def gen_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_2d(
    real_t, mpi_construct, ghost_exchange_communicator
):
    """MPI-supported 2D Advection (ENO3 stencil) Euler forward timestep generator"""
    elementwise_sum_pyst_kernel_2d = gen_elementwise_sum_pyst_kernel_2d(real_t=real_t)
    set_fixed_val_pyst_kernel_2d = gen_set_fixed_val_pyst_kernel_2d(real_t=real_t)
    advection_flux_conservative_eno3_pyst_mpi_kernel_2d = (
        gen_advection_flux_conservative_eno3_pyst_mpi_kernel_2d(
            real_t=real_t,
            mpi_construct=mpi_construct,
            ghost_exchange_communicator=ghost_exchange_communicator,
        )
    )
    # Define the variables below so that we check ghost size and kernel support
    # during generation phase itself
    kernel_support = (
        gen_advection_flux_conservative_eno3_pyst_mpi_kernel_2d.kernel_support
    )
    gen_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_2d.kernel_support = (
        kernel_support
    )
    check_valid_ghost_size_and_kernel_support(
        ghost_size=ghost_exchange_communicator.ghost_size,
        kernel_support=gen_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_2d.kernel_support,
    )

    def advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_2d(
        field, advection_flux, velocity, dt_by_dx
    ):
        """MPI-supported 2D Advection (ENO3 stencil) Euler forward timestep.
        Performs an inplace advection timestep (using ENO3 stencil)
        in 2D using Euler forward, for a 2D field (n, n).
        """
        # define and store kernel support size
        advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_2d.kernel_support = (
            gen_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_2d.kernel_support
        )
        set_fixed_val_pyst_kernel_2d(field=advection_flux, fixed_val=0)
        advection_flux_conservative_eno3_pyst_mpi_kernel_2d(
            advection_flux=advection_flux,
            field=field,
            velocity=velocity,
            inv_dx=-dt_by_dx,
        )
        elementwise_sum_pyst_kernel_2d(
            sum_field=field, field_1=field, field_2=advection_flux
        )

    return advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_2d
