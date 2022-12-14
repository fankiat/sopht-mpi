import numpy as np
import pytest
from sopht.utils.precision import get_real_t, get_test_tol
from sopht.utils.field import VectorField
from sopht.numeric.eulerian_grid_ops.stencil_ops_2d import (
    gen_penalise_field_boundary_pyst_kernel_2d,
)
from sopht_mpi.utils import (
    MPIConstruct2D,
    MPIGhostCommunicator2D,
    MPIFieldCommunicator2D,
)
from sopht_mpi.numeric.eulerian_grid_ops.stencil_ops_2d import (
    gen_penalise_field_boundary_pyst_mpi_kernel_2d,
)


@pytest.mark.mpi(group="MPI_stencil_ops_2d", min_size=4)
@pytest.mark.parametrize("ghost_size", [1, 2])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rank_distribution", [(1, 0), (0, 1)])
@pytest.mark.parametrize("aspect_ratio", [(1, 1), (1, 1.5)])
def test_mpi_penalise_field_boundary_pyst_2d(
    ghost_size, precision, rank_distribution, aspect_ratio
):
    n_values = 16
    grid_size_y, grid_size_x = (n_values * np.array(aspect_ratio)).astype(int)
    real_t = get_real_t(precision)
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=grid_size_y,
        grid_size_x=grid_size_x,
        real_t=real_t,
        rank_distribution=rank_distribution,
    )

    # extra width needed for kernel computation
    mpi_ghost_exchange_communicator = MPIGhostCommunicator2D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )
    mpi_field_communicator = MPIFieldCommunicator2D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )
    gather_local_scalar_field = mpi_field_communicator.gather_local_scalar_field
    scatter_global_scalar_field = mpi_field_communicator.scatter_global_scalar_field
    scatter_global_vector_field = mpi_field_communicator.scatter_global_vector_field

    # Allocate local field
    local_field = np.zeros(
        (
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
        )
    ).astype(real_t)
    local_grid_field = np.zeros(
        (
            mpi_construct.grid_dim,
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
        )
    ).astype(real_t)

    # Initialize and broadcast solution for comparison later
    if mpi_construct.rank == 0:
        ref_field = np.random.rand(grid_size_y, grid_size_x).astype(real_t)
        width = 4
        dx = real_t(0.1)
        grid_coord_shift = real_t(dx / 2)
        x = np.linspace(grid_coord_shift, 1 - grid_coord_shift, grid_size_x).astype(
            real_t
        )
        y = np.linspace(grid_coord_shift, 1 - grid_coord_shift, grid_size_y).astype(
            real_t
        )
        ref_grid_field = np.flipud(np.meshgrid(y, x, indexing="ij"))
    else:
        ref_field = None
        width = None
        dx = None
        ref_grid_field = (None, None)
    dx = mpi_construct.grid.bcast(dx, root=0)
    width = mpi_construct.grid.bcast(width, root=0)

    # scatter global field
    scatter_global_scalar_field(local_field, ref_field)
    scatter_global_vector_field(local_grid_field, ref_grid_field)

    # compute the field boundary penalisation
    penalise_field_boundary_pyst_mpi_kernel = (
        gen_penalise_field_boundary_pyst_mpi_kernel_2d(
            width=width,
            dx=dx,
            x_grid_field=local_grid_field[VectorField.x_axis_idx()],
            y_grid_field=local_grid_field[VectorField.y_axis_idx()],
            real_t=real_t,
            mpi_construct=mpi_construct,
            ghost_exchange_communicator=mpi_ghost_exchange_communicator,
        )
    )

    penalise_field_boundary_pyst_mpi_kernel(field=local_field)

    # gather back the penalised field globally
    global_penalised_field = np.zeros_like(ref_field)
    gather_local_scalar_field(global_penalised_field, local_field)

    # assert correct
    if mpi_construct.rank == 0:
        penalise_field_boundary_pyst_kernel = (
            gen_penalise_field_boundary_pyst_kernel_2d(
                width=width,
                dx=dx,
                x_grid_field=ref_grid_field[VectorField.x_axis_idx()],
                y_grid_field=ref_grid_field[VectorField.y_axis_idx()],
                real_t=real_t,
            )
        )
        ref_penalised_field = ref_field.copy()
        penalise_field_boundary_pyst_kernel(field=ref_penalised_field)
        kernel_support = penalise_field_boundary_pyst_mpi_kernel.kernel_support
        # check kernel_support for the diffusion kernel
        assert kernel_support == 0, "Incorrect kernel support!"
        # check field correctness
        np.testing.assert_allclose(
            ref_penalised_field,
            global_penalised_field,
            atol=get_test_tol(precision),
        )
