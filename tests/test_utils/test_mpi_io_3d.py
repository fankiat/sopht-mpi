import pytest
import sopht.utils as spu
import numpy as np
import os
from sopht_mpi.utils import MPIConstruct3D, MPIIO
from sopht_mpi.utils.mpi_io import CosseratRodMPIIO
import elastica as ea


@pytest.mark.mpi(group="MPI_utils_IO_3d", min_size=4)
@pytest.mark.parametrize("ghost_size", [1, 2])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize(
    "rank_distribution",
    [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
)
@pytest.mark.parametrize("aspect_ratio", [(1, 1, 1), (1, 1.5, 2)])
def test_mpi_eulerian_grid_scalar_field_io(
    ghost_size, precision, rank_distribution, aspect_ratio
):
    real_t = spu.get_real_t(precision)
    n_values = 8
    grid_size = (n_values * np.array(aspect_ratio)).astype(int)
    grid_size_z, grid_size_y, grid_size_x = grid_size
    x_range = 1.0
    dx = x_range / grid_size_x
    eul_grid_shift = dx / 2
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct3D(
        grid_size_z=grid_size_z,
        grid_size_y=grid_size_y,
        grid_size_x=grid_size_x,
        real_t=real_t,
        rank_distribution=rank_distribution,
    )

    # Allocate local scalar field
    local_scalar_field = np.random.rand(
        mpi_construct.local_grid_size[0] + 2 * ghost_size,
        mpi_construct.local_grid_size[1] + 2 * ghost_size,
        mpi_construct.local_grid_size[2] + 2 * ghost_size,
    ).astype(real_t)
    time = 0.1

    # Initialize IO
    origin_io = eul_grid_shift * np.ones(mpi_construct.grid_dim)
    dx_io = dx * np.ones(mpi_construct.grid_dim)
    grid_size_io = grid_size
    io = MPIIO(mpi_construct=mpi_construct, real_dtype=real_t)
    io.define_eulerian_grid(
        origin=origin_io, dx=dx_io, grid_size=grid_size_io, ghost_size=ghost_size
    )

    # Save field
    io.add_as_eulerian_fields_for_io(scalar_field=local_scalar_field)
    io.save(h5_file_name="test_eulerian_grid_scalar_field.h5", time=time)

    # Load saved HDF5 file for checking
    del io
    local_scalar_field_saved = local_scalar_field.copy()
    local_scalar_field_loaded = np.zeros_like(local_scalar_field)
    io = MPIIO(mpi_construct=mpi_construct, real_dtype=real_t)
    io.define_eulerian_grid(
        origin=origin_io, dx=dx_io, grid_size=grid_size_io, ghost_size=ghost_size
    )
    io.add_as_eulerian_fields_for_io(scalar_field=local_scalar_field_loaded)
    time_loaded = io.load(h5_file_name="test_eulerian_grid_scalar_field.h5")

    # Check values
    inner_idx = (slice(ghost_size, -ghost_size),) * mpi_construct.grid_dim
    np.testing.assert_array_equal(
        local_scalar_field_saved[inner_idx], local_scalar_field_loaded[inner_idx]
    )
    np.testing.assert_equal(time, time_loaded)
    # Cleanup saved output files
    os.system("rm -f *h5 *xmf")


@pytest.mark.mpi(group="MPI_utils_IO_3d", min_size=4)
@pytest.mark.parametrize("ghost_size", [1, 2])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize(
    "rank_distribution",
    [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
)
@pytest.mark.parametrize("aspect_ratio", [(1, 1, 1), (1, 1.5, 2)])
def test_mpi_eulerian_grid_vector_field_io(
    ghost_size, precision, rank_distribution, aspect_ratio
):
    real_t = spu.get_real_t(precision)
    n_values = 8
    grid_size = (n_values * np.array(aspect_ratio)).astype(int)
    grid_size_z, grid_size_y, grid_size_x = grid_size
    x_range = 1.0
    dx = x_range / grid_size_x
    eul_grid_shift = dx / 2
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct3D(
        grid_size_z=grid_size_z,
        grid_size_y=grid_size_y,
        grid_size_x=grid_size_x,
        real_t=real_t,
        rank_distribution=rank_distribution,
    )

    # Allocate local scalar field
    local_vector_field = np.random.rand(
        mpi_construct.grid_dim,
        mpi_construct.local_grid_size[0] + 2 * ghost_size,
        mpi_construct.local_grid_size[1] + 2 * ghost_size,
        mpi_construct.local_grid_size[2] + 2 * ghost_size,
    ).astype(real_t)
    time = 0.1

    # Initialize IO
    origin_io = eul_grid_shift * np.ones(mpi_construct.grid_dim)
    dx_io = dx * np.ones(mpi_construct.grid_dim)
    grid_size_io = grid_size
    io = MPIIO(mpi_construct=mpi_construct, real_dtype=real_t)
    io.define_eulerian_grid(
        origin=origin_io, dx=dx_io, grid_size=grid_size_io, ghost_size=ghost_size
    )

    # Save field
    io.add_as_eulerian_fields_for_io(vector_field=local_vector_field)
    io.save(h5_file_name="test_eulerian_grid_vector_field.h5", time=time)

    # Load saved HDF5 file for checking
    del io
    local_vector_field_saved = local_vector_field.copy()
    local_vector_field_loaded = np.zeros_like(local_vector_field)
    io = MPIIO(mpi_construct=mpi_construct, real_dtype=real_t)
    io.define_eulerian_grid(
        origin=origin_io, dx=dx_io, grid_size=grid_size_io, ghost_size=ghost_size
    )
    io.add_as_eulerian_fields_for_io(vector_field=local_vector_field_loaded)
    time_loaded = io.load(h5_file_name="test_eulerian_grid_vector_field.h5")

    # Check values
    inner_idx = (slice(None),) + (
        slice(ghost_size, -ghost_size),
    ) * mpi_construct.grid_dim
    np.testing.assert_array_equal(
        local_vector_field_saved[inner_idx], local_vector_field_loaded[inner_idx]
    )
    np.testing.assert_equal(time, time_loaded)
    # Cleanup saved output files
    os.system("rm -f *h5 *xmf")


@pytest.mark.mpi(group="MPI_utils_IO_3d", min_size=4)
@pytest.mark.parametrize("ghost_size", [1, 2])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize(
    "rank_distribution",
    [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
)
@pytest.mark.parametrize("aspect_ratio", [(1, 1, 1), (1, 1.5, 2)])
def test_mpi_eulerian_grid_multiple_field_io(
    ghost_size, precision, rank_distribution, aspect_ratio
):
    real_t = spu.get_real_t(precision)
    n_values = 8
    grid_size = (n_values * np.array(aspect_ratio)).astype(int)
    grid_size_z, grid_size_y, grid_size_x = grid_size
    x_range = 1.0
    dx = x_range / grid_size_x
    eul_grid_shift = dx / 2
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct3D(
        grid_size_z=grid_size_z,
        grid_size_y=grid_size_y,
        grid_size_x=grid_size_x,
        real_t=real_t,
        rank_distribution=rank_distribution,
    )

    # Allocate local scalar field
    local_scalar_field = np.random.rand(
        mpi_construct.local_grid_size[0] + 2 * ghost_size,
        mpi_construct.local_grid_size[1] + 2 * ghost_size,
        mpi_construct.local_grid_size[2] + 2 * ghost_size,
    ).astype(real_t)
    local_vector_field = np.random.rand(
        mpi_construct.grid_dim,
        mpi_construct.local_grid_size[0] + 2 * ghost_size,
        mpi_construct.local_grid_size[1] + 2 * ghost_size,
        mpi_construct.local_grid_size[2] + 2 * ghost_size,
    ).astype(real_t)
    time = 0.1

    # Initialize IO
    origin_io = eul_grid_shift * np.ones(mpi_construct.grid_dim)
    dx_io = dx * np.ones(mpi_construct.grid_dim)
    grid_size_io = grid_size
    io = MPIIO(mpi_construct=mpi_construct, real_dtype=real_t)
    io.define_eulerian_grid(
        origin=origin_io, dx=dx_io, grid_size=grid_size_io, ghost_size=ghost_size
    )

    # Save field
    io.add_as_eulerian_fields_for_io(scalar_field=local_scalar_field)
    io.add_as_eulerian_fields_for_io(vector_field=local_vector_field)
    io.save(h5_file_name="test_eulerian_grid_multiple_field.h5", time=time)

    # Load saved HDF5 file for checking
    del io
    local_scalar_field_saved = local_scalar_field.copy()
    local_scalar_field_loaded = np.zeros_like(local_scalar_field)
    local_vector_field_saved = local_vector_field.copy()
    local_vector_field_loaded = np.zeros_like(local_vector_field)
    io = MPIIO(mpi_construct=mpi_construct, real_dtype=real_t)
    io.define_eulerian_grid(
        origin=origin_io, dx=dx_io, grid_size=grid_size_io, ghost_size=ghost_size
    )
    io.add_as_eulerian_fields_for_io(scalar_field=local_scalar_field_loaded)
    io.add_as_eulerian_fields_for_io(vector_field=local_vector_field_loaded)
    time_loaded = io.load(h5_file_name="test_eulerian_grid_multiple_field.h5")

    # Check values
    scalar_inner_idx = (slice(ghost_size, -ghost_size),) * mpi_construct.grid_dim
    vector_inner_idx = (slice(None),) + (
        slice(ghost_size, -ghost_size),
    ) * mpi_construct.grid_dim
    np.testing.assert_array_equal(
        local_scalar_field_saved[scalar_inner_idx],
        local_scalar_field_loaded[scalar_inner_idx],
    )
    np.testing.assert_array_equal(
        local_vector_field_saved[vector_inner_idx],
        local_vector_field_loaded[vector_inner_idx],
    )
    np.testing.assert_equal(time, time_loaded)
    # Cleanup saved output files
    os.system("rm -f *h5 *xmf")


@pytest.mark.mpi(group="MPI_utils_IO_3d", min_size=4)
@pytest.mark.parametrize("precision", ["single", "double"])
def test_mpi_lagrangian_grid_scalar_field_io(precision):
    real_t = spu.get_real_t(precision)
    n_values = 8
    dim = 3
    grid_size_z, grid_size_y, grid_size_x = (n_values,) * dim
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct3D(
        grid_size_z=grid_size_z,
        grid_size_y=grid_size_y,
        grid_size_x=grid_size_x,
        real_t=real_t,
        rank_distribution=None,
    )

    # Initialize lagrangian grid
    # Here we consider a grid with positions along a outward spiraling coil/helix
    master_rank = 0
    if mpi_construct.rank == master_rank:
        num_lagrangian_nodes = 64
    else:
        num_lagrangian_nodes = 0
    drdt = 1.0
    num_revolutions = 4
    theta = np.linspace(0, num_revolutions * 2 * np.pi, num_lagrangian_nodes)
    radius = np.linspace(0, num_revolutions, num_lagrangian_nodes) * drdt
    lagrangian_grid_position = np.zeros(
        (mpi_construct.grid_dim, num_lagrangian_nodes)
    ).astype(real_t)
    lagrangian_grid_position[spu.VectorField.x_axis_idx(), :] = radius * np.cos(theta)
    lagrangian_grid_position[spu.VectorField.y_axis_idx(), :] = radius * np.sin(theta)
    dzdt = 1.0
    z = np.linspace(0, num_revolutions, num_lagrangian_nodes) * dzdt
    lagrangian_grid_position[spu.VectorField.z_axis_idx(), :] = z

    # Initialize scalar field
    scalar_field = np.linspace(0, 1, num_lagrangian_nodes).astype(real_t)
    time = 0.1

    # Initialize IO
    io = MPIIO(mpi_construct=mpi_construct, real_dtype=real_t)
    # Add scalar field on lagrangian grid
    io.add_as_lagrangian_fields_for_io(
        lagrangian_grid=lagrangian_grid_position,
        lagrangian_grid_master_rank=master_rank,
        lagrangian_grid_name="test_lagrangian_grid",
        scalar_field=scalar_field,
    )
    # Save field
    io.save(h5_file_name="test_lagrangian_grid_scalar_field.h5", time=time)

    # Load saved HDF5 file for checking
    del io
    lagrangian_grid_position_saved = lagrangian_grid_position.copy()
    lagrangian_grid_position_loaded = np.zeros_like(lagrangian_grid_position)
    scalar_field_saved = scalar_field.copy()
    scalar_field_loaded = np.zeros_like(scalar_field)
    io = MPIIO(mpi_construct=mpi_construct, real_dtype=real_t)
    io.add_as_lagrangian_fields_for_io(
        lagrangian_grid=lagrangian_grid_position_loaded,
        lagrangian_grid_master_rank=master_rank,
        lagrangian_grid_name="test_lagrangian_grid",
        scalar_field=scalar_field_loaded,
    )
    time_loaded = io.load(h5_file_name="test_lagrangian_grid_scalar_field.h5")

    # Check values
    if mpi_construct.rank == master_rank:
        all_equal_grid = np.array_equal(
            lagrangian_grid_position_saved, lagrangian_grid_position_loaded
        )
        all_equal_scalar_field = np.array_equal(
            scalar_field_saved,
            scalar_field_loaded,
        )
    else:
        all_equal_grid = None
        all_equal_scalar_field = None

    all_equal_grid = mpi_construct.grid.bcast(all_equal_grid, root=master_rank)
    all_equal_scalar_field = mpi_construct.grid.bcast(
        all_equal_scalar_field, root=master_rank
    )
    assert all_equal_grid, "Lagrangian grid mismatch!"
    assert all_equal_scalar_field, "Lagrangian scalar field mismatch!"
    np.testing.assert_equal(time, time_loaded)
    os.system("rm -f *h5 *xmf")


@pytest.mark.mpi(group="MPI_utils_IO_3d", min_size=4)
@pytest.mark.parametrize("precision", ["single", "double"])
def test_mpi_lagrangian_grid_vector_field_io(precision):
    real_t = spu.get_real_t(precision)
    n_values = 8
    dim = 3
    grid_size_z, grid_size_y, grid_size_x = (n_values,) * dim
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct3D(
        grid_size_z=grid_size_z,
        grid_size_y=grid_size_y,
        grid_size_x=grid_size_x,
        real_t=real_t,
        rank_distribution=None,
    )

    # Initialize lagrangian grid
    # Here we consider a grid with positions along a outward spiraling coil/helix
    master_rank = 0
    if mpi_construct.rank == master_rank:
        num_lagrangian_nodes = 64
    else:
        num_lagrangian_nodes = 0
    drdt = 1.0
    num_revolutions = 4
    theta = np.linspace(0, num_revolutions * 2 * np.pi, num_lagrangian_nodes)
    radius = np.linspace(0, num_revolutions, num_lagrangian_nodes) * drdt
    lagrangian_grid_position = np.zeros(
        (mpi_construct.grid_dim, num_lagrangian_nodes)
    ).astype(real_t)
    lagrangian_grid_position[spu.VectorField.x_axis_idx(), :] = radius * np.cos(theta)
    lagrangian_grid_position[spu.VectorField.y_axis_idx(), :] = radius * np.sin(theta)
    dzdt = 1.0
    z = np.linspace(0, num_revolutions, num_lagrangian_nodes) * dzdt
    lagrangian_grid_position[spu.VectorField.z_axis_idx(), :] = z

    # Initialize vector field
    vector_field = np.zeros_like(lagrangian_grid_position)
    vector_field[spu.VectorField.x_axis_idx(), :] = -radius * np.sin(theta)
    vector_field[spu.VectorField.y_axis_idx(), :] = radius * np.cos(theta)
    vector_field[spu.VectorField.z_axis_idx(), :] = dzdt
    time = 0.1

    # Initialize IO
    io = MPIIO(mpi_construct=mpi_construct, real_dtype=real_t)
    # Add scalar field on lagrangian grid
    io.add_as_lagrangian_fields_for_io(
        lagrangian_grid=lagrangian_grid_position,
        lagrangian_grid_master_rank=master_rank,
        lagrangian_grid_name="test_lagrangian_grid",
        vector_field=vector_field,
    )
    # Save field
    io.save(h5_file_name="test_lagrangian_grid_vector_field.h5", time=time)

    # Load saved HDF5 file for checking
    del io
    lagrangian_grid_position_saved = lagrangian_grid_position.copy()
    lagrangian_grid_position_loaded = np.zeros_like(lagrangian_grid_position)
    vector_field_saved = vector_field.copy()
    vector_field_loaded = np.zeros_like(vector_field)
    io = MPIIO(mpi_construct=mpi_construct, real_dtype=real_t)
    io.add_as_lagrangian_fields_for_io(
        lagrangian_grid=lagrangian_grid_position_loaded,
        lagrangian_grid_master_rank=master_rank,
        lagrangian_grid_name="test_lagrangian_grid",
        vector_field=vector_field_loaded,
    )
    time_loaded = io.load(h5_file_name="test_lagrangian_grid_vector_field.h5")

    # Check values
    if mpi_construct.rank == master_rank:
        all_equal_grid = np.array_equal(
            lagrangian_grid_position_saved, lagrangian_grid_position_loaded
        )
        all_equal_vector_field = np.array_equal(vector_field_saved, vector_field_loaded)
    else:
        all_equal_grid = None
        all_equal_vector_field = None

    all_equal_grid = mpi_construct.grid.bcast(all_equal_grid, root=master_rank)
    all_equal_vector_field = mpi_construct.grid.bcast(
        all_equal_vector_field, root=master_rank
    )
    assert all_equal_grid, "Lagrangian grid mismatch!"
    assert all_equal_vector_field, "Lagrangian vector field mismatch!"
    np.testing.assert_equal(time, time_loaded)
    os.system("rm -f *h5 *xmf")


@pytest.mark.mpi(group="MPI_utils_IO_3d", min_size=4)
@pytest.mark.parametrize("precision", ["single", "double"])
def test_mpi_lagrangian_grid_multiple_field_io(precision):
    real_t = spu.get_real_t(precision)
    n_values = 8
    dim = 3
    grid_size_z, grid_size_y, grid_size_x = (n_values,) * dim
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct3D(
        grid_size_z=grid_size_z,
        grid_size_y=grid_size_y,
        grid_size_x=grid_size_x,
        real_t=real_t,
        rank_distribution=None,
    )

    # Initialize lagrangian grid
    # Here we consider a grid with positions along a outward spiraling coil/helix
    master_rank = 0
    if mpi_construct.rank == master_rank:
        num_lagrangian_nodes = 64
    else:
        num_lagrangian_nodes = 0
    drdt = 1.0
    num_revolutions = 4
    theta = np.linspace(0, num_revolutions * 2 * np.pi, num_lagrangian_nodes)
    radius = np.linspace(0, num_revolutions, num_lagrangian_nodes) * drdt
    lagrangian_grid_position = np.zeros(
        (mpi_construct.grid_dim, num_lagrangian_nodes)
    ).astype(real_t)
    lagrangian_grid_position[spu.VectorField.x_axis_idx(), :] = radius * np.cos(theta)
    lagrangian_grid_position[spu.VectorField.y_axis_idx(), :] = radius * np.sin(theta)
    dzdt = 1.0
    z = np.linspace(0, num_revolutions, num_lagrangian_nodes) * dzdt
    lagrangian_grid_position[spu.VectorField.z_axis_idx(), :] = z

    # Initialize scalar and vector field
    scalar_field = np.linspace(0, 1, num_lagrangian_nodes).astype(real_t)
    vector_field = np.zeros_like(lagrangian_grid_position)
    vector_field[spu.VectorField.x_axis_idx(), :] = -radius * np.sin(theta)
    vector_field[spu.VectorField.y_axis_idx(), :] = radius * np.cos(theta)
    vector_field[spu.VectorField.z_axis_idx(), :] = dzdt
    time = 0.1

    # Initialize IO
    io = MPIIO(mpi_construct=mpi_construct, real_dtype=real_t)
    # Add scalar field on lagrangian grid
    io.add_as_lagrangian_fields_for_io(
        lagrangian_grid=lagrangian_grid_position,
        lagrangian_grid_master_rank=master_rank,
        lagrangian_grid_name="test_lagrangian_grid",
        scalar_field=scalar_field,
        vector_field=vector_field,
    )
    # Save field
    io.save(h5_file_name="test_lagrangian_grid_multiple_field.h5", time=time)

    # Load saved HDF5 file for checking
    del io
    lagrangian_grid_position_saved = lagrangian_grid_position.copy()
    lagrangian_grid_position_loaded = np.zeros_like(lagrangian_grid_position)
    scalar_field_saved = scalar_field.copy()
    scalar_field_loaded = np.zeros_like(scalar_field)
    vector_field_saved = vector_field.copy()
    vector_field_loaded = np.zeros_like(vector_field)
    io = MPIIO(mpi_construct=mpi_construct, real_dtype=real_t)
    io.add_as_lagrangian_fields_for_io(
        lagrangian_grid=lagrangian_grid_position_loaded,
        lagrangian_grid_master_rank=master_rank,
        lagrangian_grid_name="test_lagrangian_grid",
        scalar_field=scalar_field_loaded,
        vector_field=vector_field_loaded,
    )
    time_loaded = io.load(h5_file_name="test_lagrangian_grid_multiple_field.h5")

    # Check values
    if mpi_construct.rank == master_rank:
        all_equal_grid = np.array_equal(
            lagrangian_grid_position_saved, lagrangian_grid_position_loaded
        )
        all_equal_scalar_field = np.array_equal(scalar_field_saved, scalar_field_loaded)
        all_equal_vector_field = np.array_equal(vector_field_saved, vector_field_loaded)
    else:
        all_equal_grid = None
        all_equal_scalar_field = None
        all_equal_vector_field = None

    all_equal_grid = mpi_construct.grid.bcast(all_equal_grid, root=master_rank)
    all_equal_scalar_field = mpi_construct.grid.bcast(
        all_equal_scalar_field, root=master_rank
    )
    all_equal_vector_field = mpi_construct.grid.bcast(
        all_equal_vector_field, root=master_rank
    )
    assert all_equal_grid, "Lagrangian grid mismatch!"
    assert all_equal_scalar_field, "Lagrangian scalar field mismatch!"
    assert all_equal_vector_field, "Lagrangian vector field mismatch!"
    np.testing.assert_equal(time, time_loaded)
    os.system("rm -f *h5 *xmf")


@pytest.mark.mpi(group="MPI_utils_IO_3d", min_size=4)
@pytest.mark.parametrize("precision", ["single", "double"])
def test_mpi_cosserat_rod_io(precision):
    real_t = spu.get_real_t(precision)
    n_values = 16
    dim = 3
    grid_size_z, grid_size_y, grid_size_x = (n_values,) * dim
    # Generate the MPI topology minimal object
    mpi_construct = MPIConstruct3D(
        grid_size_z=grid_size_z,
        grid_size_y=grid_size_y,
        grid_size_x=grid_size_x,
        real_t=real_t,
        rank_distribution=None,
    )

    # Initialize mock rod
    n_element = 16
    rod_incline_angle = np.pi / 4.0
    start = np.zeros(3)
    direction = np.zeros_like(start)
    direction[spu.VectorField.x_axis_idx()] = np.cos(rod_incline_angle)
    direction[spu.VectorField.y_axis_idx()] = np.sin(rod_incline_angle)
    normal = np.array([0.0, 0.0, 1.0])
    rod_length = 1.0
    rod_element_radius = np.linspace(0.01, 0.5, n_element)
    density = 1.0
    nu = 1.0
    youngs_modulus = 1.0
    rod = ea.CosseratRod.straight_rod(
        n_element,
        start,
        direction,
        normal,
        rod_length,
        rod_element_radius,
        density,
        nu,
        youngs_modulus,
    )
    time = 0.1

    # Initialize cosserat rod io
    master_rank = 0  # we only care about the rod instance on master_rank
    rod_io = CosseratRodMPIIO(
        mpi_construct=mpi_construct,
        master_rank=master_rank,
        cosserat_rod=rod,
    )
    # Save rod
    rod_io.save(h5_file_name="test_cosserat_rod_io.h5", time=time)

    # Load saved HDF5 file for checking
    del rod_io
    rod_element_position_saved = 0.5 * (
        rod.position_collection[:dim, 1:] + rod.position_collection[:dim, :-1]
    )
    rod_element_position_loaded = np.zeros((dim, n_element))
    rod_element_radius_saved = rod.radius.copy()
    rod_element_radius_loaded = np.zeros(n_element)
    rod_real_t = rod.position_collection.dtype  # pyelastica is always double
    base_io = MPIIO(mpi_construct=mpi_construct, real_dtype=rod_real_t)
    base_io.add_as_lagrangian_fields_for_io(
        lagrangian_grid=rod_element_position_loaded,
        lagrangian_grid_master_rank=master_rank,
        lagrangian_grid_name="rod",
        scalar_3d=rod_element_radius_loaded,
    )
    time_loaded = base_io.load(h5_file_name="test_cosserat_rod_io.h5")

    # Check values
    # We only care about the values on master rank
    if mpi_construct.rank == master_rank:
        all_equal_rod_position = np.array_equal(
            rod_element_position_saved, rod_element_position_loaded
        )
        all_equal_rod_radius = np.array_equal(
            rod_element_radius_saved, rod_element_radius_loaded
        )
    else:
        all_equal_rod_position = None
        all_equal_rod_radius = None

    all_equal_rod_position = mpi_construct.grid.bcast(
        all_equal_rod_position, root=master_rank
    )
    all_equal_rod_radius = mpi_construct.grid.bcast(
        all_equal_rod_radius, root=master_rank
    )

    assert all_equal_rod_position, "Rod position mismatch!"
    assert all_equal_rod_radius, "Rod radius mismatch!"
    np.testing.assert_equal(time, time_loaded)
    os.system("rm -f *h5 *xmf")
