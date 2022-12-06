import numpy as np
import pytest
from sopht.utils.precision import get_real_t, get_test_tol
from sopht.numeric.eulerian_grid_ops.stencil_ops_2d import (
    gen_diffusion_timestep_euler_forward_pyst_kernel_2d,
)
from sopht_mpi.utils import (
    MPIConstruct2D,
    MPIGhostCommunicator2D,
    MPIFieldCommunicator2D,
)
from sopht_mpi.numeric.eulerian_grid_ops.stencil_ops_2d import (
    gen_diffusion_timestep_euler_forward_pyst_mpi_kernel_2d,
)


@pytest.mark.mpi(group="MPI_stencil_ops_2d", min_size=4)
@pytest.mark.parametrize("ghost_size", [1, 2])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rank_distribution", [(1, 0), (0, 1)])
@pytest.mark.parametrize("aspect_ratio", [(1, 1), (1, 1.5)])
def test_mpi_diffusion_timestep_2d(
    ghost_size, precision, rank_distribution, aspect_ratio
):
    n_values = 8
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
    gather_local_field = mpi_field_communicator.gather_local_field
    scatter_global_field = mpi_field_communicator.scatter_global_field

    # Allocate local field
    local_field = np.zeros(
        (
            mpi_construct.local_grid_size[0] + 2 * ghost_size,
            mpi_construct.local_grid_size[1] + 2 * ghost_size,
        )
    ).astype(real_t)
    local_diffusion_flux = np.ones_like(local_field)

    # Initialize and broadcast solution for comparison later
    if mpi_construct.rank == 0:
        ref_field = np.random.rand(grid_size_y, grid_size_x).astype(real_t)
        nu_dt_by_dx2 = real_t(0.1)
    else:
        ref_field = None
        nu_dt_by_dx2 = None
    ref_field = mpi_construct.grid.bcast(ref_field, root=0)
    nu_dt_by_dx2 = mpi_construct.grid.bcast(nu_dt_by_dx2, root=0)

    # scatter global field
    scatter_global_field(local_field, ref_field, mpi_construct)

    # compute the diffusion timestep
    diffusion_timestep_euler_forward_pyst_mpi_kernel = (
        gen_diffusion_timestep_euler_forward_pyst_mpi_kernel_2d(
            real_t=real_t,
            mpi_construct=mpi_construct,
            ghost_exchange_communicator=mpi_ghost_exchange_communicator,
        )
    )

    diffusion_timestep_euler_forward_pyst_mpi_kernel(
        diffusion_flux=local_diffusion_flux,
        field=local_field,
        nu_dt_by_dx2=nu_dt_by_dx2,
    )

    # gather back the field globally after diffusion timestep
    global_field = np.zeros_like(ref_field)
    gather_local_field(global_field, local_field, mpi_construct)

    # assert correct
    if mpi_construct.rank == 0:
        diffusion_timestep_euler_forward_pyst_kernel = (
            gen_diffusion_timestep_euler_forward_pyst_kernel_2d(
                real_t=real_t,
            )
        )
        ref_diffusion_flux = np.ones_like(ref_field)
        diffusion_timestep_euler_forward_pyst_kernel(
            diffusion_flux=ref_diffusion_flux,
            field=ref_field,
            nu_dt_by_dx2=nu_dt_by_dx2,
        )

        # check kernel_support for the diffusion kernel
        assert (
            diffusion_timestep_euler_forward_pyst_mpi_kernel.kernel_support == 1
        ), "Incorrect kernel support!"
        # check field correctness
        np.testing.assert_allclose(
            ref_field,
            global_field,
            atol=get_test_tol(precision),
        )
