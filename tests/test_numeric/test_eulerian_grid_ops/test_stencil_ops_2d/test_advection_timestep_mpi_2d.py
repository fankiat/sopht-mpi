import numpy as np
import pytest
from sopht.utils.precision import get_real_t, get_test_tol
from sopht.numeric.eulerian_grid_ops.stencil_ops_2d import (
    gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_2d,
)
from sopht_mpi.utils import (
    MPIConstruct2D,
    MPIGhostCommunicator2D,
    MPIFieldCommunicator2D,
)
from sopht_mpi.numeric.eulerian_grid_ops.stencil_ops_2d import (
    gen_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_2d,
)


@pytest.mark.mpi(group="MPI_stencil_ops_2d", min_size=2)
@pytest.mark.parametrize("ghost_size", [pytest.param(1, marks=pytest.mark.xfail), 2, 3])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rank_distribution", [(1, 0), (0, 1)])
@pytest.mark.parametrize("aspect_ratio", [(1, 1), (1, 2), (2, 1)])
def test_mpi_advection_timestep_eno3_euler_forward_2d(
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

    # extra width needed for kernel computation
    mpi_ghost_exchange_communicator = MPIGhostCommunicator2D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )
    mpi_field_communicator = MPIFieldCommunicator2D(
        ghost_size=ghost_size, mpi_construct=mpi_construct
    )
    gather_local_field = mpi_field_communicator.gather_local_field
    scatter_global_field = mpi_field_communicator.scatter_global_field

    # Allocate local field
    local_field = np.zeros(
        (
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
        )
    ).astype(real_t)
    local_velocity_x = np.zeros_like(local_field).astype(real_t)
    local_velocity_y = np.zeros_like(local_field).astype(real_t)
    local_advection_flux = np.zeros_like(local_field).astype(real_t)

    # Initialize and broadcast solution for comparison later
    if mpi_construct.rank == 0:
        ref_field = np.random.rand(
            n_values * aspect_ratio[0], n_values * aspect_ratio[1]
        ).astype(real_t)
        ref_velocity = np.random.rand(
            2, n_values * aspect_ratio[0], n_values * aspect_ratio[1]
        ).astype(real_t)
        inv_dx = real_t(0.2)
        dt = real_t(0.1)
    else:
        ref_field = None
        ref_velocity = None
        inv_dx = None
        dt = None
    ref_field = mpi_construct.grid.bcast(ref_field, root=0)
    ref_velocity = mpi_construct.grid.bcast(ref_velocity, root=0)
    inv_dx = mpi_construct.grid.bcast(inv_dx, root=0)
    dt = mpi_construct.grid.bcast(dt, root=0)
    dt_by_dx = real_t(dt * inv_dx)

    # scatter global field
    scatter_global_field(local_field, ref_field, mpi_construct)
    scatter_global_field(local_velocity_x, ref_velocity[0], mpi_construct)
    scatter_global_field(local_velocity_y, ref_velocity[1], mpi_construct)

    local_velocity = np.zeros(
        (
            2,
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
        )
    ).astype(real_t)
    local_velocity[0] = local_velocity_x
    local_velocity[1] = local_velocity_y

    # compute the advection timestep
    advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_2d = (
        gen_advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_2d(
            real_t=real_t,
            mpi_construct=mpi_construct,
            ghost_exchange_communicator=mpi_ghost_exchange_communicator,
        )
    )

    advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_2d(
        advection_flux=local_advection_flux,
        field=local_field,
        velocity=local_velocity,
        dt_by_dx=dt_by_dx,
    )

    # gather back the field globally after advection timestep
    global_field = np.zeros_like(ref_field)
    gather_local_field(global_field, local_field, mpi_construct)

    # assert correct
    if mpi_construct.rank == 0:
        advection_timestep_euler_forward_conservative_eno3_pyst_kernel_2d = (
            gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_2d(
                real_t=real_t,
            )
        )
        ref_advection_flux = np.ones_like(ref_field)
        advection_timestep_euler_forward_conservative_eno3_pyst_kernel_2d(
            advection_flux=ref_advection_flux,
            field=ref_field,
            velocity=ref_velocity,
            dt_by_dx=dt_by_dx,
        )
        kernel_support = (
            advection_timestep_euler_forward_conservative_eno3_pyst_mpi_kernel_2d.kernel_support
        )
        # check kernel_support for the advection kernel
        assert kernel_support == 2, "Incorrect kernel support!"
        # check field correctness
        inner_idx = (slice(kernel_support, -kernel_support),) * 2
        np.testing.assert_allclose(
            ref_field[inner_idx],
            global_field[inner_idx],
            atol=get_test_tol(precision),
        )
