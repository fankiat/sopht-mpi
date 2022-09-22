import numpy as np
import pytest
from sopht.utils.precision import get_real_t, get_test_tol
from sopht.numeric.eulerian_grid_ops.stencil_ops_2d import (
    gen_brinkmann_penalise_pyst_kernel_2d,
)
from sopht_mpi.utils import MPIConstruct2D, MPIFieldCommunicator2D
from sopht_mpi.numeric.eulerian_grid_ops.stencil_ops_2d import (
    gen_brinkmann_penalise_pyst_mpi_kernel_2d,
)


@pytest.mark.mpi(group="MPI_stencil_ops_2d", min_size=2)
@pytest.mark.parametrize("ghost_size", [0, 1, 2, 3])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rank_distribution", [(1, 0), (0, 1)])
@pytest.mark.parametrize("aspect_ratio", [(1, 1), (1, 2), (2, 1)])
def test_mpi_brinkmann_penalise_scalar_field_2d(
    ghost_size, precision, rank_distribution, aspect_ratio
):
    n_values = 32
    real_t = get_real_t(precision)
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=n_values * aspect_ratio[0],
        grid_size_x=n_values * aspect_ratio[1],
        real_t=real_t,
        rank_distribution=rank_distribution,
    )

    # Initialize field communicator
    # No need for ghost communicator, since no ghost exchange is needed
    mpi_field_io_communicator = MPIFieldCommunicator2D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )
    gather_local_field = mpi_field_io_communicator.gather_local_field
    scatter_global_field = mpi_field_io_communicator.scatter_global_field

    # Allocate local field
    local_field = np.zeros(
        (
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
        )
    ).astype(real_t)
    local_penalty_field = np.zeros_like(local_field)
    local_char_field = np.zeros_like(local_field)
    local_penalised_field = np.zeros_like(local_field)

    # Initialize and broadcast solution for comparison later
    if mpi_construct.rank == 0:
        ref_field = np.random.rand(
            n_values * aspect_ratio[0], n_values * aspect_ratio[1]
        ).astype(real_t)
        ref_penalty_field = np.random.rand(
            n_values * aspect_ratio[0], n_values * aspect_ratio[1]
        ).astype(real_t)
        ref_char_field = np.random.rand(
            n_values * aspect_ratio[0], n_values * aspect_ratio[1]
        ).astype(real_t)
        penalty_factor = real_t(0.1)
    else:
        ref_field = None
        ref_penalty_field = None
        ref_char_field = None
        penalty_factor = None
    ref_field = mpi_construct.grid.bcast(ref_field, root=0)
    ref_penalty_field = mpi_construct.grid.bcast(ref_penalty_field, root=0)
    ref_char_field = mpi_construct.grid.bcast(ref_char_field, root=0)
    penalty_factor = mpi_construct.grid.bcast(penalty_factor, root=0)

    # scatter global field
    scatter_global_field(local_field, ref_field, mpi_construct)
    scatter_global_field(local_penalty_field, ref_penalty_field, mpi_construct)
    scatter_global_field(local_char_field, ref_char_field, mpi_construct)

    # compute the brinkmann penalisation
    brinkmann_penalise_scalar_field_pyst_mpi_kernel = (
        gen_brinkmann_penalise_pyst_mpi_kernel_2d(real_t=real_t, field_type="scalar")
    )

    brinkmann_penalise_scalar_field_pyst_mpi_kernel(
        penalised_field=local_penalised_field,
        penalty_factor=penalty_factor,
        char_field=local_char_field,
        penalty_field=local_penalty_field,
        field=local_field,
    )

    # gather back the field globally after diffusion timestep
    global_penalised_field = np.zeros_like(ref_field)
    gather_local_field(global_penalised_field, local_penalised_field, mpi_construct)

    # assert correct
    if mpi_construct.rank == 0:
        brinkmann_penalise_pyst_kernel = gen_brinkmann_penalise_pyst_kernel_2d(
            real_t=real_t, field_type="scalar"
        )
        ref_penalised_field = np.ones_like(ref_field)
        brinkmann_penalise_pyst_kernel(
            penalised_field=ref_penalised_field,
            penalty_factor=penalty_factor,
            char_field=ref_char_field,
            penalty_field=ref_penalty_field,
            field=ref_field,
        )
        kernel_support = brinkmann_penalise_scalar_field_pyst_mpi_kernel.kernel_support
        # check kernel_support for the diffusion kernel
        assert kernel_support == 0, "Incorrect kernel support!"
        # check field correctness
        inner_idx = (slice(kernel_support, -kernel_support),) * 2
        np.testing.assert_allclose(
            ref_penalised_field[inner_idx],
            global_penalised_field[inner_idx],
            atol=get_test_tol(precision),
        )


@pytest.mark.mpi(group="MPI_stencil_ops_2d", min_size=2)
@pytest.mark.parametrize("ghost_size", [0, 1, 2, 3])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rank_distribution", [(1, 0), (0, 1)])
@pytest.mark.parametrize("aspect_ratio", [(1, 1), (1, 2), (2, 1)])
def test_mpi_brinkmann_penalise_vector_field_2d(
    ghost_size, precision, rank_distribution, aspect_ratio
):
    n_values = 32
    real_t = get_real_t(precision)
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=n_values * aspect_ratio[0],
        grid_size_x=n_values * aspect_ratio[1],
        real_t=real_t,
        rank_distribution=rank_distribution,
    )

    # Initialize field communicator
    # No need for ghost communicator, since no ghost exchange is needed
    mpi_field_io_communicator = MPIFieldCommunicator2D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )
    gather_local_field = mpi_field_io_communicator.gather_local_field
    scatter_global_field = mpi_field_io_communicator.scatter_global_field

    # Allocate local field
    local_vector_field = np.zeros(
        (
            2,
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
        )
    ).astype(real_t)
    local_penalty_vector_field = np.zeros_like(local_vector_field)
    local_char_field = np.zeros_like(local_vector_field[0])
    local_penalised_vector_field = np.zeros_like(local_vector_field)

    # Initialize and broadcast solution for comparison later
    if mpi_construct.rank == 0:
        ref_vector_field = np.random.rand(
            2, n_values * aspect_ratio[0], n_values * aspect_ratio[1]
        ).astype(real_t)
        ref_penalty_vector_field = np.random.rand(
            2, n_values * aspect_ratio[0], n_values * aspect_ratio[1]
        ).astype(real_t)
        ref_char_field = np.random.rand(
            n_values * aspect_ratio[0], n_values * aspect_ratio[1]
        ).astype(real_t)
        penalty_factor = real_t(0.1)
    else:
        ref_vector_field = None
        ref_penalty_vector_field = None
        ref_char_field = None
        penalty_factor = None
    ref_vector_field = mpi_construct.grid.bcast(ref_vector_field, root=0)
    ref_penalty_vector_field = mpi_construct.grid.bcast(
        ref_penalty_vector_field, root=0
    )
    ref_char_field = mpi_construct.grid.bcast(ref_char_field, root=0)
    penalty_factor = mpi_construct.grid.bcast(penalty_factor, root=0)

    # scatter global field
    scatter_global_field(local_vector_field[0], ref_vector_field[0], mpi_construct)
    scatter_global_field(local_vector_field[1], ref_vector_field[1], mpi_construct)
    scatter_global_field(
        local_penalty_vector_field[0], ref_penalty_vector_field[0], mpi_construct
    )
    scatter_global_field(
        local_penalty_vector_field[1], ref_penalty_vector_field[1], mpi_construct
    )
    scatter_global_field(local_char_field, ref_char_field, mpi_construct)

    # compute the brinkmann penalisation
    brinkmann_penalise_vector_field_pyst_mpi_kernel = (
        gen_brinkmann_penalise_pyst_mpi_kernel_2d(real_t=real_t, field_type="vector")
    )

    brinkmann_penalise_vector_field_pyst_mpi_kernel(
        penalised_vector_field=local_penalised_vector_field,
        penalty_factor=penalty_factor,
        char_field=local_char_field,
        penalty_vector_field=local_penalty_vector_field,
        vector_field=local_vector_field,
    )

    # gather back the vector field globally after diffusion timestep
    global_penalised_vector_field = np.zeros_like(ref_vector_field)
    gather_local_field(
        global_penalised_vector_field[0], local_penalised_vector_field[0], mpi_construct
    )
    gather_local_field(
        global_penalised_vector_field[1], local_penalised_vector_field[1], mpi_construct
    )

    # assert correct
    if mpi_construct.rank == 0:
        brinkmann_penalise_pyst_kernel = gen_brinkmann_penalise_pyst_kernel_2d(
            real_t=real_t, field_type="vector"
        )
        ref_penalised_vector_field = np.ones_like(ref_vector_field)
        brinkmann_penalise_pyst_kernel(
            penalised_vector_field=ref_penalised_vector_field,
            penalty_factor=penalty_factor,
            char_field=ref_char_field,
            penalty_vector_field=ref_penalty_vector_field,
            vector_field=ref_vector_field,
        )
        kernel_support = brinkmann_penalise_vector_field_pyst_mpi_kernel.kernel_support
        # check kernel_support for the diffusion kernel
        assert kernel_support == 0, "Incorrect kernel support!"
        # check field correctness
        inner_idx = (slice(kernel_support, -kernel_support),) * 2
        np.testing.assert_allclose(
            ref_penalised_vector_field[inner_idx],
            global_penalised_vector_field[inner_idx],
            atol=get_test_tol(precision),
        )
