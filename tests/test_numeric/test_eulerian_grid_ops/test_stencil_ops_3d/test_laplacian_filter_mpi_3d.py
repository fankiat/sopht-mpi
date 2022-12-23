import numpy as np
import pytest
from sopht.utils.precision import get_real_t, get_test_tol
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d import (
    gen_laplacian_filter_kernel_3d,
)
from sopht_mpi.utils import (
    MPIConstruct3D,
    MPIGhostCommunicator3D,
    MPIFieldCommunicator3D,
)
from sopht_mpi.numeric.eulerian_grid_ops.stencil_ops_3d import (
    gen_laplacian_filter_mpi_kernel_3d,
)


@pytest.mark.mpi(group="MPI_stencil_ops_3d", min_size=4)
@pytest.mark.parametrize("ghost_size", [1, 2])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize(
    "rank_distribution",
    [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
)
@pytest.mark.parametrize("aspect_ratio", [(1, 1, 1), (1, 1.5, 2)])
@pytest.mark.parametrize("field_type", ["scalar", "vector"])
@pytest.mark.parametrize("filter_type", ["multiplicative", "convolution"])
def test_mpi_laplacian_filter_3d(
    ghost_size, precision, rank_distribution, aspect_ratio, field_type, filter_type
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

    # extra width needed for kernel computation
    mpi_ghost_exchange_communicator = MPIGhostCommunicator3D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )
    mpi_field_communicator = MPIFieldCommunicator3D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )
    gather_local_field = (
        mpi_field_communicator.gather_local_scalar_field
        if field_type == "scalar"
        else mpi_field_communicator.gather_local_vector_field
    )
    scatter_global_field = (
        mpi_field_communicator.scatter_global_scalar_field
        if field_type == "scalar"
        else mpi_field_communicator.scatter_global_vector_field
    )

    # Allocate local field
    local_field_shape = mpi_construct.local_grid_size + 2 * ghost_size
    if field_type == "vector":
        local_field_shape = np.insert(local_field_shape, 0, mpi_construct.grid_dim)
    local_test_field = np.zeros(local_field_shape).astype(real_t)
    local_field_buffer = (
        np.zeros_like(local_test_field)
        if field_type == "scalar"
        else np.zeros_like(local_test_field[0])
    )
    local_filter_flux_buffer = np.zeros_like(local_field_buffer)

    # Initialize and broadcast solution for comparison later
    if mpi_construct.rank == 0:
        ref_field_shape = (
            (grid_size_z, grid_size_y, grid_size_x)
            if field_type == "scalar"
            else (mpi_construct.grid_dim, grid_size_z, grid_size_y, grid_size_x)
        )
        ref_test_field = np.random.rand(*ref_field_shape).astype(real_t)
    else:
        ref_test_field = (
            None if field_type == "scalar" else (None,) * mpi_construct.grid_dim
        )

    # scatter global field
    scatter_global_field(local_test_field, ref_test_field)

    # generate and apply laplacian filter kernel
    filter_order = 2
    laplacian_filter_mpi_kernel = gen_laplacian_filter_mpi_kernel_3d(
        real_t=real_t,
        mpi_construct=mpi_construct,
        ghost_exchange_communicator=mpi_ghost_exchange_communicator,
        filter_order=filter_order,
        field_buffer=local_field_buffer,
        filter_flux_buffer=local_filter_flux_buffer,
        field_type=field_type,
        filter_type=filter_type,
    )
    laplacian_filter_mpi_kernel(local_test_field)

    # gather back the diffusion flux globally
    global_test_field = np.zeros_like(ref_test_field)
    gather_local_field(global_test_field, local_test_field)

    # assert correct
    if mpi_construct.rank == 0:
        ref_field_buffer = (
            np.zeros_like(ref_test_field)
            if field_type == "scalar"
            else np.zeros_like(ref_test_field[0])
        )
        ref_filter_flux_buffer = np.zeros_like(ref_field_buffer)
        laplacian_filter = gen_laplacian_filter_kernel_3d(
            filter_order=filter_order,
            field_buffer=ref_field_buffer,
            filter_flux_buffer=ref_filter_flux_buffer,
            real_t=real_t,
            field_type=field_type,
            filter_type=filter_type,
        )
        laplacian_filter(ref_test_field)

        # check kernel_support for the diffusion kernel
        kernel_support = laplacian_filter_mpi_kernel.kernel_support
        assert kernel_support == 1, "Incorrect kernel support!"

        # check field correctness
        np.testing.assert_allclose(
            ref_test_field,
            global_test_field,
            atol=get_test_tol(precision),
        )
