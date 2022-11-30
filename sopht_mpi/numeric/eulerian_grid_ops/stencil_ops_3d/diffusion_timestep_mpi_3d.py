"""MPI-supported kernels for performing diffusion timestep in 3D."""
from sopht_mpi.numeric.eulerian_grid_ops.stencil_ops_3d.diffusion_flux_mpi_3d import (
    gen_diffusion_flux_pyst_mpi_kernel_3d,
)
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d.elementwise_ops_3d import (
    gen_elementwise_sum_pyst_kernel_3d,
)
from sopht_mpi.utils.mpi_utils import check_valid_ghost_size_and_kernel_support


def gen_diffusion_timestep_euler_forward_pyst_mpi_kernel_3d(
    real_t,
    mpi_construct,
    ghost_exchange_communicator,
    field_type="scalar",
):
    """MPI-supported 3D diffusion euler forward timestep generator"""
    if field_type != "scalar" and field_type != "vector":
        raise ValueError("Invalid field type")
    elementwise_sum_pyst_kernel_3d = gen_elementwise_sum_pyst_kernel_3d(real_t=real_t)
    diffusion_flux_mpi_kernel_3d = gen_diffusion_flux_pyst_mpi_kernel_3d(
        real_t=real_t,
        mpi_construct=mpi_construct,
        ghost_exchange_communicator=ghost_exchange_communicator,
    )
    # Since the generated function has not been called yet, we cannot access the
    # kernel support variable. However, because of the way we implemented the
    # generator function, we can retrieve it from the generator function.
    # Define the variables below so that we check ghost size and kernel support
    # during generation phase itself
    kernel_support = gen_diffusion_flux_pyst_mpi_kernel_3d.kernel_support
    gen_diffusion_timestep_euler_forward_pyst_mpi_kernel_3d.kernel_support = (
        kernel_support
    )
    check_valid_ghost_size_and_kernel_support(
        ghost_size=ghost_exchange_communicator.ghost_size,
        kernel_support=gen_diffusion_timestep_euler_forward_pyst_mpi_kernel_3d.kernel_support,
    )

    def diffusion_timestep_euler_forward_pyst_mpi_kernel_3d(
        field, diffusion_flux, nu_dt_by_dx2
    ):
        """MPI-supported 3D Diffusion Euler forward timestep.

        Performs an inplace diffusion timestep in 3D using Euler forward,
        for a 3D field (n, n, n).
        """
        # define and store kernel support size
        diffusion_timestep_euler_forward_pyst_mpi_kernel_3d.kernel_support = (
            gen_diffusion_flux_pyst_mpi_kernel_3d.kernel_support
        )
        diffusion_flux_mpi_kernel_3d(
            diffusion_flux=diffusion_flux,
            field=field,
            prefactor=nu_dt_by_dx2,
        )
        elementwise_sum_pyst_kernel_3d(
            sum_field=field, field_1=field, field_2=diffusion_flux
        )

    if field_type == "scalar":
        return diffusion_timestep_euler_forward_pyst_mpi_kernel_3d
    elif field_type == "vector":

        def vector_field_diffusion_timestep_euler_forward_pyst_mpi_kernel_3d(
            vector_field, diffusion_flux, nu_dt_by_dx2
        ):
            """MPI-supported 3D Diffusion Euler forward timestep (vector field).

            Performs an inplace diffusion timestep in 3D using Euler forward,
            for a 3D vector field (3, n, n, n).
            """
            vector_field_diffusion_timestep_euler_forward_pyst_mpi_kernel_3d.kernel_support = (
                gen_diffusion_flux_pyst_mpi_kernel_3d.kernel_support
            )
            diffusion_timestep_euler_forward_pyst_mpi_kernel_3d(
                field=vector_field[0],
                diffusion_flux=diffusion_flux,
                nu_dt_by_dx2=nu_dt_by_dx2,
            )
            diffusion_timestep_euler_forward_pyst_mpi_kernel_3d(
                field=vector_field[1],
                diffusion_flux=diffusion_flux,
                nu_dt_by_dx2=nu_dt_by_dx2,
            )
            diffusion_timestep_euler_forward_pyst_mpi_kernel_3d(
                field=vector_field[2],
                diffusion_flux=diffusion_flux,
                nu_dt_by_dx2=nu_dt_by_dx2,
            )

        return vector_field_diffusion_timestep_euler_forward_pyst_mpi_kernel_3d
