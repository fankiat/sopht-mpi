import numpy as np
import pytest
from sopht.utils.precision import get_real_t, get_test_tol
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d import (
    gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d,
)
from sopht_mpi.utils import (
    MPIConstruct3D,
    MPIGhostCommunicator3D,
    MPIFieldCommunicator3D,
)
from sopht_mpi.numeric.eulerian_grid_ops.stencil_ops_3d import (
    gen_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d,
)


@pytest.mark.mpi(group="MPI_stencil_ops_3d", min_size=4)
@pytest.mark.parametrize("ghost_size", [pytest.param(1, marks=pytest.mark.xfail), 2])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize(
    "rank_distribution",
    [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
)
@pytest.mark.parametrize("aspect_ratio", [(1, 1, 1), (1, 1.5, 2)])
def test_mpi_advection_timestep_eno3_euler_forward_3d(
    ghost_size, precision, rank_distribution, aspect_ratio
):
    n_values = 16
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
    gather_local_scalar_field = mpi_field_communicator.gather_local_scalar_field
    scatter_global_scalar_field = mpi_field_communicator.scatter_global_scalar_field
    scatter_global_vector_field = mpi_field_communicator.scatter_global_vector_field

    # Allocate local field
    local_field = np.zeros(
        (
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
            mpi_construct.local_grid_size[2] + 2 * ghost_size,
        )
    ).astype(real_t)
    local_velocity = np.zeros(
        (
            mpi_construct.grid_dim,
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
            mpi_construct.local_grid_size[2] + 2 * ghost_size,
        )
    ).astype(real_t)
    local_advection_flux = np.zeros_like(local_field).astype(real_t)

    # Initialize and broadcast solution for comparison later
    if mpi_construct.rank == 0:
        ref_field = np.random.rand(grid_size_z, grid_size_y, grid_size_x).astype(real_t)
        ref_velocity = np.random.rand(
            mpi_construct.grid_dim, grid_size_z, grid_size_y, grid_size_x
        ).astype(real_t)
        inv_dx = real_t(0.2)
        dt = real_t(0.1)
    else:
        ref_field = None
        ref_velocity = (None,) * mpi_construct.grid_dim
        inv_dx = None
        dt = None
    inv_dx = mpi_construct.grid.bcast(inv_dx, root=0)
    dt = mpi_construct.grid.bcast(dt, root=0)
    dt_by_dx = real_t(dt * inv_dx)

    # scatter global field
    scatter_global_scalar_field(local_field, ref_field)
    scatter_global_vector_field(local_velocity, ref_velocity)

    # compute the advection timestep
    advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d = (
        gen_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d(
            real_t=real_t,
            mpi_construct=mpi_construct,
            ghost_exchange_communicator=mpi_ghost_exchange_communicator,
        )
    )

    advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d(
        advection_flux=local_advection_flux,
        field=local_field,
        velocity=local_velocity,
        dt_by_dx=dt_by_dx,
    )

    # gather back the field globally after advection timestep
    global_field = np.zeros_like(ref_field)
    gather_local_scalar_field(global_field, local_field)

    # assert correct
    if mpi_construct.rank == 0:
        advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d = (
            gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d(
                real_t=real_t,
            )
        )
        ref_advection_flux = np.ones_like(ref_field)
        advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d(
            advection_flux=ref_advection_flux,
            field=ref_field,
            velocity=ref_velocity,
            dt_by_dx=dt_by_dx,
        )
        kernel_support = (
            advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d.kernel_support
        )
        # check kernel_support for the advection kernel
        assert kernel_support == 2, "Incorrect kernel support!"
        # check field correctness
        inner_idx = (slice(kernel_support, -kernel_support),) * mpi_construct.grid_dim
        np.testing.assert_allclose(
            ref_field[inner_idx],
            global_field[inner_idx],
            atol=get_test_tol(precision),
        )


@pytest.mark.mpi(group="MPI_stencil_ops_3d", min_size=4)
@pytest.mark.parametrize("ghost_size", [pytest.param(1, marks=pytest.mark.xfail), 2])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize(
    "rank_distribution",
    [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
)
@pytest.mark.parametrize("aspect_ratio", [(1, 1, 1), (1, 1.5, 2)])
def test_mpi_vector_field_advection_timestep_eno3_euler_forward_3d(
    ghost_size, precision, rank_distribution, aspect_ratio
):
    n_values = 16
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
    gather_local_vector_field = mpi_field_communicator.gather_local_vector_field
    scatter_global_vector_field = mpi_field_communicator.scatter_global_vector_field

    # Allocate local field
    local_vector_field = np.zeros(
        (
            mpi_construct.grid_dim,
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
            mpi_construct.local_grid_size[2] + 2 * ghost_size,
        )
    ).astype(real_t)
    local_velocity = np.zeros_like(local_vector_field)
    local_advection_flux = np.zeros(
        (
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
            mpi_construct.local_grid_size[2] + 2 * ghost_size,
        )
    ).astype(real_t)

    # Initialize and broadcast solution for comparison later
    if mpi_construct.rank == 0:
        ref_vector_field = np.random.rand(
            mpi_construct.grid_dim, grid_size_z, grid_size_y, grid_size_x
        ).astype(real_t)
        ref_velocity = np.random.rand(
            mpi_construct.grid_dim, grid_size_z, grid_size_y, grid_size_x
        ).astype(real_t)
        inv_dx = real_t(0.2)
        dt = real_t(0.1)
    else:
        ref_vector_field = (None,) * mpi_construct.grid_dim
        ref_velocity = (None,) * mpi_construct.grid_dim
        inv_dx = None
        dt = None
    inv_dx = mpi_construct.grid.bcast(inv_dx, root=0)
    dt = mpi_construct.grid.bcast(dt, root=0)
    dt_by_dx = real_t(dt * inv_dx)

    # scatter global field
    scatter_global_vector_field(local_vector_field, ref_vector_field)
    scatter_global_vector_field(local_velocity, ref_velocity)

    # compute the advection timestep
    vector_field_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d = gen_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d(
        real_t=real_t,
        mpi_construct=mpi_construct,
        ghost_exchange_communicator=mpi_ghost_exchange_communicator,
        field_type="vector",
    )

    vector_field_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d(
        advection_flux=local_advection_flux,
        vector_field=local_vector_field,
        velocity=local_velocity,
        dt_by_dx=dt_by_dx,
    )

    # gather back the field globally after advection timestep
    global_vector_field = np.zeros_like(ref_vector_field)
    gather_local_vector_field(global_vector_field, local_vector_field)

    # assert correct
    if mpi_construct.rank == 0:
        vector_field_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d = gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d(
            real_t=real_t, field_type="vector"
        )
        ref_advection_flux = np.ones((grid_size_z, grid_size_y, grid_size_x)).astype(
            real_t
        )
        vector_field_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d(
            advection_flux=ref_advection_flux,
            vector_field=ref_vector_field,
            velocity=ref_velocity,
            dt_by_dx=dt_by_dx,
        )
        kernel_support = (
            vector_field_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_3d.kernel_support
        )
        # check kernel_support for the advection timestep kernel
        assert kernel_support == 2, "Incorrect kernel support!"
        # check field correctness
        inner_idx = (slice(None),) + (
            slice(kernel_support, -kernel_support),
        ) * mpi_construct.grid_dim
        np.testing.assert_allclose(
            ref_vector_field[inner_idx],
            global_vector_field[inner_idx],
            atol=get_test_tol(precision),
        )
