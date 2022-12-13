import numpy as np
import pytest
from sopht.utils.precision import get_real_t, get_test_tol
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d import (
    gen_char_func_from_level_set_via_sine_heaviside_pyst_kernel_3d,
)
from sopht_mpi.utils import MPIConstruct3D, MPIFieldCommunicator3D
from sopht_mpi.numeric.eulerian_grid_ops.stencil_ops_3d import (
    gen_char_func_from_level_set_via_sine_heaviside_pyst_mpi_kernel_3d,
)


@pytest.mark.mpi(group="MPI_stencil_ops_3d", min_size=4)
@pytest.mark.parametrize("ghost_size", [1, 2])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize(
    "rank_distribution",
    [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
)
@pytest.mark.parametrize("aspect_ratio", [(1, 1, 1), (1, 1.5, 2)])
def test_mpi_char_func_from_level_set_3d(
    ghost_size, precision, rank_distribution, aspect_ratio
):
    n_values = 8
    grid_size_z, grid_size_y, grid_size_x = (n_values * np.array(aspect_ratio)).astype(
        int
    )
    real_t = get_real_t(precision)
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct3D(
        grid_size_z=grid_size_z,
        grid_size_y=grid_size_y,
        grid_size_x=grid_size_x,
        real_t=real_t,
        rank_distribution=rank_distribution,
    )

    # Initialize field communicator
    # No need for ghost communicator, since no ghost exchange is needed
    mpi_field_communicator = MPIFieldCommunicator3D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )
    gather_local_field = mpi_field_communicator.gather_local_field
    scatter_global_field = mpi_field_communicator.scatter_global_field

    # Allocate local field
    local_char_func_field = np.zeros(
        (
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
            mpi_construct.local_grid_size[2] + 2 * ghost_size,
        )
    ).astype(real_t)
    local_level_set_field = np.zeros_like(local_char_func_field)

    # Initialize and broadcast solution for comparison later
    if mpi_construct.rank == 0:
        ref_level_set_field = np.random.rand(
            grid_size_z, grid_size_y, grid_size_x
        ).astype(real_t)
        blend_width = real_t(0.2)
    else:
        ref_level_set_field = None
        blend_width = None
    blend_width = mpi_construct.grid.bcast(blend_width, root=0)

    # scatter global field
    scatter_global_field(local_level_set_field, ref_level_set_field, mpi_construct)

    # compute the char func from level set
    char_func_from_level_set_via_sine_heaviside_pyst_mpi_kernel = (
        gen_char_func_from_level_set_via_sine_heaviside_pyst_mpi_kernel_3d(
            blend_width=blend_width, real_t=real_t
        )
    )

    char_func_from_level_set_via_sine_heaviside_pyst_mpi_kernel(
        char_func_field=local_char_func_field, level_set_field=local_level_set_field
    )

    # gather back the field globally after diffusion timestep
    global_char_func_field = np.zeros_like(ref_level_set_field)
    gather_local_field(global_char_func_field, local_char_func_field, mpi_construct)

    # assert correct
    if mpi_construct.rank == 0:
        char_func_from_level_set_via_sine_heaviside_pyst_kernel = (
            gen_char_func_from_level_set_via_sine_heaviside_pyst_kernel_3d(
                blend_width=blend_width, real_t=real_t
            )
        )
        ref_char_func_field = np.zeros_like(ref_level_set_field)
        char_func_from_level_set_via_sine_heaviside_pyst_kernel(
            char_func_field=ref_char_func_field, level_set_field=ref_level_set_field
        )
        kernel_support = (
            char_func_from_level_set_via_sine_heaviside_pyst_mpi_kernel.kernel_support
        )
        # check kernel_support for the diffusion kernel
        assert kernel_support == 0, "Incorrect kernel support!"
        # check field correctness
        np.testing.assert_allclose(
            ref_char_func_field,
            global_char_func_field,
            atol=get_test_tol(precision),
        )
