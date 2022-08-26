import numpy as np
import pytest
from sopht.utils.precision import get_real_t, get_test_tol
from sopht.numeric.eulerian_grid_ops.stencil_ops_2d import (
    gen_diffusion_flux_pyst_kernel_2d,
)
from sopht_mpi.utils import (
    MPIConstruct2D,
    MPIGhostCommunicator2D,
    MPIFieldIOCommunicator2D,
)
from sopht_mpi.numeric.eulerian_grid_ops.stencil_ops_2d import (
    gen_diffusion_flux_pyst_mpi_blocking_kernel_2d,
    gen_diffusion_flux_pyst_mpi_non_blocking_kernel_2d,
)


@pytest.mark.mpi(group="MPI_stencil_ops_2d")
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
@pytest.mark.parametrize("comm_type", ["blocking", "non_blocking"])
def test_mpi_diffusion_flux_2d(n_values, precision, comm_type):
    real_t = get_real_t(precision)
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct2D(
        grid_size_y=n_values,
        grid_size_x=n_values,
        real_t=real_t,
    )

    # extra width needed for kernel computation
    ghost_size = 1
    # extra width involved in the field storage (>= ghost_size)
    field_offset = 1 * ghost_size
    mpi_ghost_exchange_communicator = MPIGhostCommunicator2D(
        ghost_size=ghost_size, field_offset=field_offset, mpi_construct=mpi_construct
    )
    mpi_field_io_comm_with_offset_size_1 = MPIFieldIOCommunicator2D(
        field_offset=field_offset, mpi_construct=mpi_construct
    )
    gather_local_field = mpi_field_io_comm_with_offset_size_1.gather_local_field
    scatter_global_field = mpi_field_io_comm_with_offset_size_1.scatter_global_field

    # Allocate local field
    local_field = np.zeros(
        (
            mpi_construct.local_grid_size[0] + 2 * field_offset,
            mpi_construct.local_grid_size[1] + 2 * field_offset,
        )
    ).astype(real_t)
    local_diffusion_flux = np.zeros_like(local_field)

    # Initialize and broadcast solution for comparison later
    if mpi_construct.rank == 0:
        ref_field = np.random.rand(n_values, n_values).astype(real_t)
        prefactor = real_t(0.1)
    else:
        ref_field = None
        prefactor = None
    ref_field = mpi_construct.grid.bcast(ref_field, root=0)
    prefactor = mpi_construct.grid.bcast(prefactor, root=0)

    # scatter global field
    scatter_global_field(local_field, ref_field, mpi_construct)

    # compute the diffusion flux
    if comm_type == "blocking":
        diffusion_flux_pyst_mpi_kernel = gen_diffusion_flux_pyst_mpi_blocking_kernel_2d(
            real_t=real_t,
            mpi_construct=mpi_construct,
            ghost_exchange_communicator=mpi_ghost_exchange_communicator,
        )
    elif comm_type == "non_blocking":
        diffusion_flux_pyst_mpi_kernel = (
            gen_diffusion_flux_pyst_mpi_non_blocking_kernel_2d(
                real_t=real_t,
                mpi_construct=mpi_construct,
                ghost_exchange_communicator=mpi_ghost_exchange_communicator,
            )
        )
    else:
        raise ValueError("Invalid communication type!")
    diffusion_flux_pyst_mpi_kernel(
        diffusion_flux=local_diffusion_flux,
        field=local_field,
        prefactor=prefactor,
    )

    # gather back the diffusion flux globally
    global_diffusion_flux = np.zeros_like(ref_field)
    gather_local_field(global_diffusion_flux, local_diffusion_flux, mpi_construct)

    # assert correct
    if mpi_construct.rank == 0:
        diffusion_flux_pyst_kernel = gen_diffusion_flux_pyst_kernel_2d(
            real_t=real_t,
        )
        ref_diffusion_flux = np.zeros_like(ref_field)
        diffusion_flux_pyst_kernel(
            diffusion_flux=ref_diffusion_flux,
            field=ref_field,
            prefactor=prefactor,
        )
        inner_idx = (slice(ghost_size, -ghost_size),) * 2
        np.testing.assert_allclose(
            ref_diffusion_flux[inner_idx],
            global_diffusion_flux[inner_idx],
            atol=get_test_tol(precision),
        )
