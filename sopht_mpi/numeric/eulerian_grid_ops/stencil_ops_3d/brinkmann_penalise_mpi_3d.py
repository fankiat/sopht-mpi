"""MPI-supported kernels for Brinkmann penalisation in 3D."""
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d import (
    gen_brinkmann_penalise_pyst_kernel_3d,
)


def gen_brinkmann_penalise_pyst_mpi_kernel_3d(real_t, field_type="scalar"):
    """MPI-supported Brinkmann penalisation 3D kernel generator."""
    # define kernel support here, no need to check since kernel_support = 0
    # and ghost size is guaranteed to be >= 0 when ghost comm is created
    gen_brinkmann_penalise_pyst_mpi_kernel_3d.kernel_support = 0
    brinkmann_penalise_pyst_mpi_kernel_3d = gen_brinkmann_penalise_pyst_kernel_3d(
        real_t=real_t,
        field_type=field_type,
    )
    brinkmann_penalise_pyst_mpi_kernel_3d.kernel_support = (
        gen_brinkmann_penalise_pyst_mpi_kernel_3d.kernel_support
    )
    return brinkmann_penalise_pyst_mpi_kernel_3d
