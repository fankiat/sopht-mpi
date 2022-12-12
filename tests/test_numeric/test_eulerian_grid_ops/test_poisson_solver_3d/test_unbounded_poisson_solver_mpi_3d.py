import numpy as np
import pytest
from sopht.utils.precision import get_real_t, get_test_tol
from sopht.numeric.eulerian_grid_ops.poisson_solver_3d import (
    UnboundedPoissonSolverPYFFTW3D,
)
from sopht_mpi.utils import MPIConstruct3D, MPIFieldCommunicator3D
from sopht_mpi.numeric.eulerian_grid_ops.poisson_solver_3d import (
    UnboundedPoissonSolverMPI3D,
)


@pytest.mark.mpi(group="MPI_unbounded_poisson_solve_3d", min_size=4)
@pytest.mark.parametrize("ghost_size", [1, 2])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize(
    "rank_distribution",
    [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
)
@pytest.mark.parametrize("aspect_ratio", [(1, 1, 1), (1, 1.5, 2)])
def test_mpi_unbounded_poisson_solve_3d(
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

    # Create unbounded poisson solver
    unbounded_poisson_solver = UnboundedPoissonSolverMPI3D(
        grid_size_z=grid_size_z,
        grid_size_y=grid_size_y,
        grid_size_x=grid_size_x,
        real_t=real_t,
        mpi_construct=mpi_construct,
        ghost_size=ghost_size,
    )

    # Initialize communicator for scatter and gather
    mpi_field_comm = MPIFieldCommunicator3D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )
    gather_local_field = mpi_field_comm.gather_local_field
    scatter_global_field = mpi_field_comm.scatter_global_field

    # Allocate local field
    local_rhs_field = np.zeros(
        (
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
            mpi_construct.local_grid_size[2] + 2 * ghost_size,
        )
    ).astype(real_t)
    local_solution_field = np.zeros_like(local_rhs_field)

    # Initialize and broadcast solution for comparison later
    if mpi_construct.rank == 0:
        ref_rhs_field = np.random.rand(grid_size_z, grid_size_y, grid_size_x).astype(
            real_t
        )
    else:
        ref_rhs_field = None

    # scatter global field
    scatter_global_field(local_rhs_field, ref_rhs_field, mpi_construct)

    # compute the unbounded poisson solve
    unbounded_poisson_solver.solve(
        solution_field=local_solution_field, rhs_field=local_rhs_field
    )

    # gather back the solution field globally
    global_solution_field = np.zeros_like(ref_rhs_field)
    gather_local_field(global_solution_field, local_solution_field, mpi_construct)

    # assert correct
    if mpi_construct.rank == 0:
        ref_unbounded_poisson_solver = UnboundedPoissonSolverPYFFTW3D(
            grid_size_z=grid_size_z,
            grid_size_y=grid_size_y,
            grid_size_x=grid_size_x,
            real_t=real_t,
        )
        ref_solution_field = np.zeros_like(ref_rhs_field)
        ref_unbounded_poisson_solver.solve(
            solution_field=ref_solution_field, rhs_field=ref_rhs_field
        )
        np.testing.assert_allclose(
            ref_solution_field,
            global_solution_field,
            atol=get_test_tol(precision),
        )


@pytest.mark.mpi(group="MPI_unbounded_poisson_solve_3d", min_size=4)
@pytest.mark.parametrize("ghost_size", [1, 2])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize(
    "rank_distribution",
    [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
)
@pytest.mark.parametrize("aspect_ratio", [(1, 1, 1), (1, 1.5, 2)])
def test_mpi_unbounded_vector_field_poisson_solve_3d(
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

    # Create unbounded poisson solver
    unbounded_poisson_solver = UnboundedPoissonSolverMPI3D(
        grid_size_z=grid_size_z,
        grid_size_y=grid_size_y,
        grid_size_x=grid_size_x,
        real_t=real_t,
        mpi_construct=mpi_construct,
        ghost_size=ghost_size,
    )

    # Initialize communicator for scatter and gather
    mpi_field_comm = MPIFieldCommunicator3D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )
    gather_local_field = mpi_field_comm.gather_local_field
    scatter_global_field = mpi_field_comm.scatter_global_field

    # Allocate local field
    local_rhs_vector_field = np.zeros(
        (
            mpi_construct.grid_dim,
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
            mpi_construct.local_grid_size[2] + 2 * ghost_size,
        )
    ).astype(real_t)
    local_solution_vector_field = np.zeros_like(local_rhs_vector_field)

    # Initialize and broadcast solution for comparison later
    if mpi_construct.rank == 0:
        ref_rhs_vector_field = np.random.rand(
            mpi_construct.grid_dim, grid_size_z, grid_size_y, grid_size_x
        ).astype(real_t)
    else:
        ref_rhs_vector_field = (None, None, None)

    # scatter global field
    # TODO: replace with vector scatter when available
    scatter_global_field(
        local_rhs_vector_field[0], ref_rhs_vector_field[0], mpi_construct
    )
    scatter_global_field(
        local_rhs_vector_field[1], ref_rhs_vector_field[1], mpi_construct
    )
    scatter_global_field(
        local_rhs_vector_field[2], ref_rhs_vector_field[2], mpi_construct
    )

    # compute the unbounded poisson solve
    unbounded_poisson_solver.vector_field_solve(
        solution_vector_field=local_solution_vector_field,
        rhs_vector_field=local_rhs_vector_field,
    )

    # gather back the solution field globally
    global_solution_vector_field = np.zeros_like(ref_rhs_vector_field)
    # TODO: replace with vector gather when available
    gather_local_field(
        global_solution_vector_field[0], local_solution_vector_field[0], mpi_construct
    )
    gather_local_field(
        global_solution_vector_field[1], local_solution_vector_field[1], mpi_construct
    )
    gather_local_field(
        global_solution_vector_field[2], local_solution_vector_field[2], mpi_construct
    )

    # assert correct
    if mpi_construct.rank == 0:
        ref_unbounded_poisson_solver = UnboundedPoissonSolverPYFFTW3D(
            grid_size_z=grid_size_z,
            grid_size_y=grid_size_y,
            grid_size_x=grid_size_x,
            real_t=real_t,
        )
        ref_solution_vector_field = np.zeros_like(ref_rhs_vector_field)
        ref_unbounded_poisson_solver.vector_field_solve(
            solution_vector_field=ref_solution_vector_field,
            rhs_vector_field=ref_rhs_vector_field,
        )
        np.testing.assert_allclose(
            ref_solution_vector_field,
            global_solution_vector_field,
            atol=get_test_tol(precision),
        )
