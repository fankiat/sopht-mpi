"""MPI-supported kernels for Brinkmann penalisation in 2D."""
from sopht.numeric.eulerian_grid_ops.stencil_ops_2d import (
    gen_brinkmann_penalise_pyst_kernel_2d,
)


def gen_brinkmann_penalise_pyst_mpi_kernel_2d(real_t, field_type="scalar"):
    """MPI-supported Brinkmann penalisation 2D kernel generator."""
    if field_type != "scalar" and field_type != "vector":
        raise ValueError("Invalid field type")

    # define kernel support here, no need to check since kernel_support = 0
    # and ghost size is guaranteed to be >= 0 when ghost comm is created
    gen_brinkmann_penalise_pyst_mpi_kernel_2d.kernel_support = 0

    if field_type == "scalar":
        brinkmann_penalise_pyst_kernel_2d = gen_brinkmann_penalise_pyst_kernel_2d(
            real_t=real_t,
            field_type="scalar",
        )

        def brinkmann_penalise_scalar_field_pyst_mpi_kernel_2d(
            penalised_field, penalty_factor, char_field, penalty_field, field
        ):
            """MPI-supported Brinkmann penalisation for 2D scalar field."""
            brinkmann_penalise_scalar_field_pyst_mpi_kernel_2d.kernel_support = (
                gen_brinkmann_penalise_pyst_mpi_kernel_2d.kernel_support
            )
            brinkmann_penalise_pyst_kernel_2d(
                penalised_field=penalised_field,
                penalty_factor=penalty_factor,
                char_field=char_field,
                penalty_field=penalty_field,
                field=field,
            )

        return brinkmann_penalise_scalar_field_pyst_mpi_kernel_2d

    elif field_type == "vector":
        brinkmann_penalise_pyst_kernel_2d = gen_brinkmann_penalise_pyst_kernel_2d(
            real_t=real_t,
            field_type="vector",
        )

        def brinkmann_penalise_vector_field_pyst_mpi_kernel_2d(
            penalised_vector_field,
            penalty_factor,
            char_field,
            penalty_vector_field,
            vector_field,
        ):
            """MPI-supported Brinkmann penalisation for 2D vector field."""
            brinkmann_penalise_vector_field_pyst_mpi_kernel_2d.kernel_support = (
                gen_brinkmann_penalise_pyst_mpi_kernel_2d.kernel_support
            )
            brinkmann_penalise_pyst_kernel_2d(
                penalised_vector_field=penalised_vector_field,
                penalty_factor=penalty_factor,
                char_field=char_field,
                penalty_vector_field=penalty_vector_field,
                vector_field=vector_field,
            )

        return brinkmann_penalise_vector_field_pyst_mpi_kernel_2d
