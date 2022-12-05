"""MPI-supported kernels for computing characteristic function from level set
field in 3D."""
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d import (
    gen_char_func_from_level_set_via_sine_heaviside_pyst_kernel_3d,
)


def gen_char_func_from_level_set_via_sine_heaviside_pyst_mpi_kernel_3d(
    blend_width, real_t
):
    """MPI-supported level set --> characteristic function 3D kernel generator.
    Generate function that computes characteristic function field
    from the level set field, via a smooth sine Heaviside function.
    """
    # define kernel support here, no need to check since kernel_support = 0
    # and ghost size is guaranteed to be >= 0 when ghost comm is created
    gen_char_func_from_level_set_via_sine_heaviside_pyst_mpi_kernel_3d.kernel_support = (
        0
    )
    char_func_from_level_set_via_sine_heaviside_pyst_mpi_kernel_3d = (
        gen_char_func_from_level_set_via_sine_heaviside_pyst_kernel_3d(
            blend_width=blend_width, real_t=real_t
        )
    )
    char_func_from_level_set_via_sine_heaviside_pyst_mpi_kernel_3d.kernel_support = (
        gen_char_func_from_level_set_via_sine_heaviside_pyst_mpi_kernel_3d.kernel_support
    )
    return char_func_from_level_set_via_sine_heaviside_pyst_mpi_kernel_3d
