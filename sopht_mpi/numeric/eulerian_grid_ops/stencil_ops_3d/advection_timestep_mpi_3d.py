"""MPI-supported kernels for performing advection timestep in 3D."""
from sopht_mpi.numeric.eulerian_grid_ops.stencil_ops_3d.advection_flux_mpi_3d import (
    gen_advection_flux_conservative_eno3_pyst_mpi_kernel_3d,
)
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d.elementwise_ops_3d import (
    gen_elementwise_sum_pyst_kernel_3d,
    gen_set_fixed_val_pyst_kernel_3d,
)
from sopht_mpi.utils.mpi_utils import check_valid_ghost_size_and_kernel_support


def gen_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d(
    real_t, mpi_construct, ghost_exchange_communicator, field_type="scalar"
):
    """MPI-supported 3D Advection (ENO3 stencil) Euler forward timestep generator"""
    if field_type != "scalar" and field_type != "vector":
        raise ValueError("Invalid field type")
    elementwise_sum_pyst_kernel_3d = gen_elementwise_sum_pyst_kernel_3d(real_t=real_t)
    set_fixed_val_pyst_kernel_3d = gen_set_fixed_val_pyst_kernel_3d(real_t=real_t)
    advection_flux_conservative_eno3_pyst_mpi_kernel_3d = (
        gen_advection_flux_conservative_eno3_pyst_mpi_kernel_3d(
            real_t=real_t,
            mpi_construct=mpi_construct,
            ghost_exchange_communicator=ghost_exchange_communicator,
        )
    )
    # Define the variables below so that we check ghost size and kernel support
    # during generation phase itself
    kernel_support = (
        gen_advection_flux_conservative_eno3_pyst_mpi_kernel_3d.kernel_support
    )
    gen_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d.kernel_support = (
        kernel_support
    )
    check_valid_ghost_size_and_kernel_support(
        ghost_size=ghost_exchange_communicator.ghost_size,
        kernel_support=gen_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d.kernel_support,
    )

    def advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d(
        field, advection_flux, velocity, dt_by_dx
    ):
        """MPI-supported 3D Advection (ENO3 stencil) Euler forward timestep.
        Performs an inplace advection timestep (using ENO3 stencil)
        in 3D using Euler forward, for a 3D scalar field (n, n, n).
        """
        # define and store kernel support size
        advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d.kernel_support = (
            gen_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d.kernel_support
        )
        set_fixed_val_pyst_kernel_3d(field=advection_flux, fixed_val=0)
        advection_flux_conservative_eno3_pyst_mpi_kernel_3d(
            advection_flux=advection_flux,
            field=field,
            velocity=velocity,
            inv_dx=-dt_by_dx,
        )
        elementwise_sum_pyst_kernel_3d(
            sum_field=field, field_1=field, field_2=advection_flux
        )

    if field_type == "scalar":
        return advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d
    elif field_type == "vector":

        def vector_field_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d(
            vector_field, advection_flux, velocity, dt_by_dx
        ):
            """MPI-supported 3D Advection (ENO3 stencil) Euler forward timestep.
            Performs an inplace advection timestep (using ENO3 stencil)
            in 3D using Euler forward, for a 3D vector field (3, n, n, n).
            """
            vector_field_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d.kernel_support = (
                gen_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d.kernel_support
            )
            advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d(
                field=vector_field[0],
                advection_flux=advection_flux,
                velocity=velocity,
                dt_by_dx=dt_by_dx,
            )
            advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d(
                field=vector_field[1],
                advection_flux=advection_flux,
                velocity=velocity,
                dt_by_dx=dt_by_dx,
            )
            advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d(
                field=vector_field[2],
                advection_flux=advection_flux,
                velocity=velocity,
                dt_by_dx=dt_by_dx,
            )

        return vector_field_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d
