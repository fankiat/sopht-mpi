"""Module for Input/Output via HD5 format."""
import h5py
import numpy as np
from mpi4py import MPI
from elastica.rod.cosserat_rod import CosseratRod


class MPIIO:
    r"""MPI-supported IO class for field save and load.

    Notes
    ----------
    Currently, the XDMF files are written for visualization in Paraview.
    For Eulerian (structured-grid) fields, read and write are done in parallel.
    For Lagrangian (unstructured-grid) fields, read and write are done by `master_rank`.

    Attributes
    ----------
    mpi_construct: MPI construct
    real_dtype: data type
        Data type for typecasting real values and setting precision of XDMF description file.

    The HDF5 file roughly follows the hierarchy below:
                    __________________/root___________________
                   /                                          \
             Eulerian                               _______Lagrangian_______
            /   |                                  /                        \
          /     |     \                        Grid_1    ..............    Grid_N
     Scalar   Vector  Parameters              /  |  \                      /  |  \
      /|\       /|\                         /    |    \                   /   |   \
    (Fields)  (Fields)                Scalar  Vector  Grid           Scalar Vector Grid
                                       /|\        /|\                 /|\     /|\
                                    (Fields)   (Fields)            (Fields) (Fields)
    """

    def __init__(self, mpi_construct, real_dtype=np.float64):
        """Class initializer."""
        self.mpi_construct = mpi_construct
        self.dim = mpi_construct.grid_dim
        assert self.dim == 2 or self.dim == 3, "Invalid dimension (only 2D and 3D)"
        self.real_dtype = real_dtype
        self.precision = 8 if real_dtype is np.float32 else 16

        # Initialize dictionaries for fields for IO and their
        # corresponding field_type ('Scalar' or 'Vector') Eulerian grid
        self.eulerian_grid_defined = False
        self.eulerian_fields = {}
        self.eulerian_fields_type = {}

        # Lagrangian grid
        self.lagrangian_fields = {}
        self.lagrangian_fields_type = {}
        self.lagrangian_grids = {}
        self.all_lagrangian_grids = {}
        self.lagrangian_fields_with_grid_name = {}
        self.lagrangian_grid_count = 0
        self.lagrangian_grid_connection = {}
        self.lagrangian_grid_master_rank = {}
        self.lagrangian_grid_num_node = {}

    def define_eulerian_grid(self, origin, dx, grid_size, ghost_size):
        """
        Define the Eulerian grid mesh.

        Attributes
        ----------
        origin: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Array containing origin position (min of coordinate values) in z-y-x ordering
        dx: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Array containing dx in each dimension following z-y-x ordering.
        grid_size: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Array containing grid_size in each dimension following z-y-x ordering.
        ghost_size: int
            Integer describing ghost size of local eulerian fields.
        """
        assert isinstance(origin, np.ndarray)
        assert isinstance(dx, np.ndarray)
        assert isinstance(grid_size, np.ndarray)
        if ghost_size < 0 and not isinstance(ghost_size, int):
            raise ValueError(
                f"Ghost size {ghost_size} needs to be an integer >= 0"
                "for eulerian field IO."
            )

        # Define global domain parameters
        self.eulerian_origin = origin
        self.eulerian_dx = dx
        self.eulerian_grid_size = grid_size  # z,y,x
        self.eulerian_grid_defined = True

        # Define local indices for read/write location on global domain dataset
        mpi_substart_index = (
            self.mpi_construct.grid.coords * self.mpi_construct.local_grid_size
        )
        mpi_subend_index = (np.array(self.mpi_construct.grid.coords) + 1) * np.array(
            self.mpi_construct.local_grid_size
        )
        # Generate local eulerian index for accessing local chunk in global dataset
        # Pre-fix with ellipsis to take care of any added dimension. See notes on
        # Eulerian save in _save() for additional clarification on why this is needed.
        self.local_eulerian_index = (...,)
        for dim in range(self.dim):
            self.local_eulerian_index += (
                slice(mpi_substart_index[dim], mpi_subend_index[dim]),
            )

        # Define local eulerian field related parameters for convenience
        self.ghost_size = ghost_size
        self.local_eulerian_grid_size = self.mpi_construct.local_grid_size
        self.local_eulerian_grid_size_with_ghost = (
            self.local_eulerian_grid_size + 2 * self.ghost_size
        )
        if self.ghost_size == 0:
            self.eulerian_field_inner_index = ...
        else:
            self.eulerian_field_inner_index = (
                slice(ghost_size, -ghost_size),
            ) * self.dim

    def add_as_eulerian_fields_for_io(self, **fields_for_io):
        """Add local Eulerian fields to be saved/loaded.

        Eulerian grid needs to be defined first using `define_eulerian_grid(...)` call.

        Attributes
        ----------
        **fields_for_io: keyword arguments used for storing local eulerian fields to file.

        Each field will be saved to the output file with its corresponding
        keyword name, similar to numpy savez function.
        https://numpy.org/doc/stable/reference/generated/numpy.savez.html#numpy.savez
        """
        assert self.eulerian_grid_defined, "Eulerian mesh is not defined!"

        for field_name in fields_for_io:
            # Add each field into local dictionary
            field = fields_for_io[field_name]
            self.eulerian_fields[field_name] = field

            # Assign field types
            if field.shape == (*self.local_eulerian_grid_size_with_ghost,):
                self.eulerian_fields_type[field_name] = "Scalar"
            elif field.shape == (self.dim, *self.local_eulerian_grid_size_with_ghost):
                self.eulerian_fields_type[field_name] = "Vector"
            else:
                raise ValueError(
                    f"Unable to identify eulerian field type "
                    f"(scalar / vector) based on field dimension {field.shape}"
                )

    def add_as_lagrangian_fields_for_io(
        self,
        lagrangian_grid_master_rank,
        lagrangian_grid,
        lagrangian_grid_name=None,
        lagrangian_grid_connect=False,
        **fields_for_io,
    ):
        """
        Add lagrangian fields to be saved/loaded.

        Notes
        ----------
        In sopht-mpi, the lagrangian grid is typically stored in a single rank. This
        information is made known to MPIIO through `lagrangian_grid_master_rank`.

        Attributes
        ----------
        lagrangian_grid_master_rank: int
            Integer describing the rank that owns lagrangian grid data.
        lagrangian_grid: numpy.ndarray
            2D (dim, N) array containing data with 'float' type.
            Array containing lagrangian grid positions.
        lagrangian_grid_name: str
            Optional naming for the lagrangian grid used by added
            fields to identify which grid they belong to.
            Otherwise default naming is used with lagrangian_grid_count.
        **fields_for_io: keyword arguments used for storing lagrangian fields to file.

        Each field will be saved to the output file with its corresponding
        keyword name, similar to numpy savez function.
        https://numpy.org/doc/stable/reference/generated/numpy.savez.html#numpy.savez
        """
        # Update relevant metadata on master rank
        if self.mpi_construct.rank == lagrangian_grid_master_rank:
            assert (
                len(lagrangian_grid.shape) == 2
            ), "lagrangian grid has to be a 2D (dim, N) array."
            assert (
                lagrangian_grid.shape[0] == self.dim
            ), "Invalid lagrangian grid dimension (only 2D and 3D)"

            if lagrangian_grid_name is None:
                lagrangian_grid_name = f"Lagrangian_grid_{self.lagrangian_grid_count}_rank{self.mpi_construct.rank}"
                self.lagrangian_grid_count += 1

            num_lagrangian_nodes = lagrangian_grid.shape[1]
            self.lagrangian_grid_num_node[lagrangian_grid_name] = num_lagrangian_nodes

            # Save `lagrangian_grid_master_rank` for lagrangian grid with grid name `lagrangian_grid_name`
            self.lagrangian_grid_master_rank[
                lagrangian_grid_name
            ] = lagrangian_grid_master_rank

            if lagrangian_grid_connect:
                self.lagrangian_grid_connection[lagrangian_grid_name] = np.arange(
                    num_lagrangian_nodes
                )

            # Save `lagrangian_grid` with grid name `lagrangian_grid_name` to `lagrangian_grids`
            self.lagrangian_grids[lagrangian_grid_name] = lagrangian_grid

            # Create a list of to store names of fields that lie on
            # grid with grid name `lagrangian_grid_name`
            self.lagrangian_fields_with_grid_name[lagrangian_grid_name] = []
            for field_name in fields_for_io:
                # Add each field into local dictionary
                field = fields_for_io[field_name]
                self.lagrangian_fields[field_name] = field
                self.lagrangian_fields_with_grid_name[lagrangian_grid_name].append(
                    field_name
                )
                # Assign field types
                if field.shape[0] == lagrangian_grid.shape[1]:
                    field_type = "Scalar"
                elif field.shape == lagrangian_grid.shape:
                    field_type = "Vector"
                else:
                    raise ValueError(
                        f"Unable to identify lagrangian field type "
                        f"(scalar / vector) based on field dimension {field.shape}"
                    )
                self.lagrangian_fields_type[field_name] = field_type

        # Update metadata on all ranks so each rank knows the data structure during save
        # Note: we need a separate all_lagrangian_grids dictionary to preserve the original
        # reference of lagrangian_grids to the actual numpy array. The all_lagrangian_grids
        # serves to facilitate the construction of metadata structure and collective
        # writing in MPIIO. Other dictionaries can remain local to the master rank,
        # since only master rank writes the relevant data in the _save function.
        self.all_lagrangian_grids = self._allreduce_dictionary(self.lagrangian_grids)
        self.lagrangian_grid_num_node = self._allreduce_dictionary(
            self.lagrangian_grid_num_node
        )
        self.lagrangian_grid_master_rank = self._allreduce_dictionary(
            self.lagrangian_grid_master_rank
        )
        self.lagrangian_grid_connection = self._allreduce_dictionary(
            self.lagrangian_grid_connection
        )
        self.lagrangian_fields_with_grid_name = self._allreduce_dictionary(
            self.lagrangian_fields_with_grid_name
        )
        self.lagrangian_fields_type = self._allreduce_dictionary(
            self.lagrangian_fields_type
        )

    def _allreduce_dictionary(self, dictionary):
        """Helper function for allreduce operation on dictionaries."""
        dictionary_as_list = [(k, v) for k, v in dictionary.items()]
        updated_list = self.mpi_construct.grid.allreduce(dictionary_as_list, op=MPI.SUM)
        updated_dictionary = {k: v for (k, v) in updated_list}
        return updated_dictionary

    def save(self, h5_file_name, time=0.0):  # noqa: C901
        """
        This is a wrapper function to call _save function.

        Attributes
        ----------
        h5_file_name: str
            String containing name of the hdf5 file.
        time: real_dtype
            Time at which the fields are saved.
        """

        self._save(h5_file_name, time)

    def _save(self, h5_file_name, time=0.0):  # noqa: C901
        """
        Save added fields to hdf5 file.

        Attributes
        ----------
        h5_file_name: str
            String containing name of the hdf5 file.
        time: real_dtype
            Time at which the fields are saved.
        """
        # 1. Create hdf5 file.
        # 2. Initialize groups for Eulerian and Lagrangian grids.
        # 3. For Eulerian group, initialize groups for scalar and
        #    vector fields (grid is already defined).
        # 4. For Lagrangian group, initialize individual groups for different lagrangian grids.
        #    In each of the lagrangian grid group, store the grid information and initialize
        #    groups for scalar and vector field.
        # 5. Go over the fields in the dictionary and save them in their corresponding location.

        with h5py.File(h5_file_name, "w", driver="mpio", comm=MPI.COMM_WORLD) as f:
            # Save time stamp
            f.attrs["time"] = time

            # Eulerian save
            if self.eulerian_grid_defined and self.eulerian_fields:
                eulerian_grp = f.create_group("Eulerian")
                # 'Scalar' and 'Vector' fields
                eulerian_scalar_grp = eulerian_grp.create_group("Scalar")
                eulerian_vector_grp = eulerian_grp.create_group("Vector")
                # Go over and save all fields that lie on the common eulerian grid
                # Note : Paraview renders 2D eulerian field on the YZ plane, inconsistently with
                # lagrangian fields rendered on XY plane. This could be a bug in Paraview and
                # I have opened up a question on Paraview's official forum
                # https://discourse.paraview.org/t/2dcorectmesh-displaying-on-yz-plane-instead-of-xy-plane/9535
                # As a workaround, here we extend the dimension of the 2D field, so that
                # it appears as a slice in a 3D space, and Paraview can
                # correctly render the field on the XY plane. We can remove the
                # reshaping when Paraview resolve the issue.
                for field_name in self.eulerian_fields:
                    field = self.eulerian_fields[field_name]
                    field_type = self.eulerian_fields_type[field_name]
                    if field_type == "Scalar":
                        # Set up dataset with global shape
                        dset = eulerian_scalar_grp.create_dataset(
                            field_name, shape=(1, *self.eulerian_grid_size)
                        )
                        # Write the local chunk of data
                        dset[self.local_eulerian_index] = field[
                            self.eulerian_field_inner_index
                        ].reshape(1, *self.local_eulerian_grid_size)
                    elif field_type == "Vector":
                        # Decompose vector fields into individual component as scalar fields
                        for idx_dim in range(self.dim):
                            # Set up dataset with global shape
                            dset = eulerian_vector_grp.create_dataset(
                                f"{field_name}_{idx_dim}",
                                shape=(1, *self.eulerian_grid_size),
                            )
                            # Write the local chunk of data
                            dset[self.local_eulerian_index] = field[idx_dim][
                                self.eulerian_field_inner_index
                            ].reshape(1, *self.local_eulerian_grid_size)
                    else:
                        raise ValueError(
                            "Unsupported eulerian_field_type ('Scalar' and 'Vector' only)"
                        )
                # Save eulerian simulation parameters
                eulerian_params_grp = eulerian_grp.create_group("Parameters")
                eulerian_params_grp.attrs["origin"] = self.eulerian_origin
                eulerian_params_grp.attrs["dx"] = self.eulerian_dx
                eulerian_params_grp.attrs["grid_size"] = self.eulerian_grid_size

                # Only a single rank generates xdmf. Here we use rank 0.
                if self.mpi_construct.rank == 0:
                    self.generate_xdmf_eulerian(h5_file_name=h5_file_name, time=time)

            # Lagrangian save
            # Note: We need to reverse the order from (dim, ...) -> (..., dim) for Paraview.
            # For eulerian fields, we mitigate this by splitting each vector
            # component into scalar fields. For lagrangian fields, since N is small
            # compared to the N in Eulerian grid, I have decided to stick with the
            # tranpose/moveaxis approach for now. This pays off later as convenience
            # during post-processing and visualizing these lagrangian points in Paraview.
            lagrangian_grp = f.create_group("Lagrangian")
            # Go over all lagrangian grids
            for lagrangian_grid_name in self.all_lagrangian_grids:
                lagrangian_grid_grp = lagrangian_grp.create_group(lagrangian_grid_name)
                is_master_rank = (
                    self.mpi_construct.rank
                    == self.lagrangian_grid_master_rank[lagrangian_grid_name]
                )

                # Dataset for lagrangian grid
                dset_grid = lagrangian_grid_grp.create_dataset(
                    "Grid",
                    shape=(
                        self.lagrangian_grid_num_node[lagrangian_grid_name],
                        self.dim,
                    ),
                )
                # write only on master_rank where lagrangian grid resides
                if is_master_rank:
                    lagrangian_grid = self.lagrangian_grids[lagrangian_grid_name]
                    dset_grid[...] = np.transpose(lagrangian_grid)

                # Dataset for lagrangian grid connection, if any
                if lagrangian_grid_name in self.lagrangian_grid_connection:
                    dset_connection = lagrangian_grid_grp.create_dataset(
                        "Connection",
                        shape=(self.lagrangian_grid_num_node[lagrangian_grid_name],),
                    )
                    # write only on master_rank
                    if is_master_rank:
                        dset_connection[...] = self.lagrangian_grid_connection[
                            lagrangian_grid_name
                        ]

                # Dataset for 'Scalar' and 'Vector' fields
                lagrangian_scalar_grp = lagrangian_grid_grp.create_group("Scalar")
                lagrangian_vector_grp = lagrangian_grid_grp.create_group("Vector")
                # Go over and save all fields that lie on the current lagrangian grid
                for field_name in self.lagrangian_fields_with_grid_name[
                    lagrangian_grid_name
                ]:
                    field_type = self.lagrangian_fields_type[field_name]
                    if field_type == "Scalar":
                        dset = lagrangian_scalar_grp.create_dataset(
                            field_name,
                            shape=(
                                self.lagrangian_grid_num_node[lagrangian_grid_name],
                            ),
                        )
                        # write only on master_rank
                        if is_master_rank:
                            field = self.lagrangian_fields[field_name]
                            dset[...] = field
                    elif field_type == "Vector":
                        dset = lagrangian_vector_grp.create_dataset(
                            field_name,
                            shape=(
                                self.lagrangian_grid_num_node[lagrangian_grid_name],
                                self.dim,
                            ),
                        )
                        # write only on master_rank
                        if is_master_rank:
                            field = self.lagrangian_fields[field_name]
                            dset[...] = np.moveaxis(field, 0, -1)
                    else:
                        raise ValueError(
                            "Unsupported lagrangian_field_type ('Scalar' and 'Vector' only)"
                        )

                # only master_rank owning the lagrangian grid generates the xdmf file
                if self.mpi_construct.rank == 0:
                    self.generate_xdmf_lagrangian(h5_file_name=h5_file_name, time=time)

    def load(self, h5_file_name):  # noqa: C901
        """Load fields from hdf5 file.

        Field arrays need to be allocated and added to `eulerian_fields` and/or
        `lagrangian_fields` for proper loading and field recovery.

        Attributes
        ----------
        h5_file_name: str
            String containing name of the hdf5 file.
        """
        with h5py.File(h5_file_name, "r", driver="mpio", comm=MPI.COMM_WORLD) as f:
            keys = []
            f.visit(keys.append)

            # Load time
            time = f.attrs["time"]

            # Load Eulerian fields
            if self.eulerian_fields:
                assert self.eulerian_grid_defined, "Eulerian grid is not defined!"
                for field_name in self.eulerian_fields:
                    field_type = self.eulerian_fields_type[field_name]
                    if field_type == "Scalar":
                        assert (
                            f"Eulerian/{field_type}/{field_name}" in keys
                        ), f"Unable to find scalar field {field_name} in loaded file!"
                        self.eulerian_fields[field_name][
                            self.eulerian_field_inner_index
                        ] = f["Eulerian"][field_type][field_name][
                            self.local_eulerian_index
                        ]
                    elif field_type == "Vector":
                        for idx_dim in range(self.dim):
                            assert (
                                f"Eulerian/{field_type}/{field_name}_{idx_dim}" in keys
                            ), (
                                f"Unable to find vector field {field_name}_{idx_dim} "
                                f"in loaded file!"
                            )
                            self.eulerian_fields[field_name][idx_dim][
                                self.eulerian_field_inner_index
                            ] = f["Eulerian"][field_type][f"{field_name}_{idx_dim}"][
                                self.local_eulerian_index
                            ]
                    else:
                        raise ValueError(
                            "Unsupported lagrangian_field_type ('Scalar' and 'Vector' only)"
                        )

                # Load 'Parameters' and assert equals for restart consistency
                np.testing.assert_allclose(
                    self.eulerian_origin,
                    f["Eulerian"]["Parameters"].attrs["origin"],
                )
                np.testing.assert_allclose(
                    self.eulerian_dx, f["Eulerian"]["Parameters"].attrs["dx"]
                )
                np.testing.assert_allclose(
                    self.eulerian_grid_size,
                    f["Eulerian"]["Parameters"].attrs["grid_size"],
                )

            # Load Lagrangian fields
            if self.lagrangian_fields:
                # First loop over and load each of the lagrangian grids
                for lagrangian_grid_name in self.all_lagrangian_grids:
                    # Load only on master rank owning the lagrangian grid
                    if (
                        self.mpi_construct.rank
                        == self.lagrangian_grid_master_rank[lagrangian_grid_name]
                    ):
                        assert (
                            f"Lagrangian/{lagrangian_grid_name}/Grid" in keys
                        ), f"Unable to find grid '{lagrangian_grid_name}' in loaded file!"
                        self.lagrangian_grids[lagrangian_grid_name][...] = np.transpose(
                            f["Lagrangian"][lagrangian_grid_name]["Grid"][...]
                        )

                        if f"Lagrangian/{lagrangian_grid_name}/Connection" in keys:
                            self.lagrangian_grid_connection[lagrangian_grid_name] = f[
                                "Lagrangian"
                            ][lagrangian_grid_name]["Connection"][...]

                        # Load all the fields living on the current lagrangian grid
                        for field_name in self.lagrangian_fields_with_grid_name[
                            lagrangian_grid_name
                        ]:
                            field_type = self.lagrangian_fields_type[field_name]
                            if field_type == "Scalar":
                                assert (
                                    f"Lagrangian/{lagrangian_grid_name}/{field_type}/{field_name}"
                                    in keys
                                ), (
                                    f"Unable to find scalar field {field_name} on "
                                    f"grid {lagrangian_grid_name} in loaded file!"
                                )
                                self.lagrangian_fields[field_name][...] = f[
                                    "Lagrangian"
                                ][lagrangian_grid_name][field_type][field_name][...]
                            elif field_type == "Vector":
                                assert (
                                    f"Lagrangian/{lagrangian_grid_name}/{field_type}/{field_name}"
                                    in keys
                                ), (
                                    f"Unable to find vector field {field_name} on grid "
                                    f"{lagrangian_grid_name} in loaded file!"
                                )
                                self.lagrangian_fields[field_name][...] = np.moveaxis(
                                    f["Lagrangian"][lagrangian_grid_name][field_type][
                                        field_name
                                    ][...],
                                    -1,
                                    0,
                                )
                            else:
                                raise ValueError(
                                    "Unsupported lagrangian_field_type "
                                    "('Scalar' and 'Vector' only)"
                                )

        return time

    def generate_xdmf_eulerian(self, h5_file_name, time=0.0):
        """Generate XDMF description file for Eulerian fields.

        Currently, the XDMF file is generated for Paraview visualization only.

        Attributes
        ----------
        h5_file_name: str
            String containing name of the corresponding hdf5 file.
        time: real_dtype
            Time at which the fields are saved.

        """
        # We use 3DCORECTMESH for both 2D and 3D fields since paraview
        # does not render on XY plane in 2DCORECTMESH
        topology_type = "3DCORECTMesh"
        geometry_type = "ORIGIN_DXDYDZ"

        grid_size = (
            self.eulerian_grid_size
            if self.dim == 3
            else np.insert(self.eulerian_grid_size, 0, 1)
        )
        origin = (
            self.eulerian_origin
            if self.dim == 3
            else np.insert(self.eulerian_origin, 0, 0.0)
        )
        dx = self.eulerian_dx if self.dim == 3 else np.insert(self.eulerian_dx, 0, 0.0)

        grid_size_string = np.array2string(
            grid_size, precision=self.precision, separator="    "
        )[1:-1]
        origin_string = np.array2string(
            origin, precision=self.precision, separator="    "
        )[1:-1]
        dx_string = np.array2string(dx, precision=self.precision, separator="    ")[
            1:-1
        ]

        def generate_field_entry(file_name, field_name, field_type):
            if field_type == "Scalar":
                entry = f"""<Attribute Name="{field_name}" Active="1"
                AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="{grid_size_string}"
                    NumberType="Float" Precision="{self.precision}" Format="HDF">
                        {file_name}:/Eulerian/{field_type}/{field_name}
                    </DataItem>
                </Attribute>

                """
            elif field_type == "Vector":
                entry = ""
                for idx_dim in range(self.dim):
                    entry += f"""<Attribute Name="{field_name}_{idx_dim}"
                    Active="1" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="{grid_size_string}" NumberType="Float"
                    Precision="{self.precision}" Format="HDF">
                        {file_name}:/Eulerian/{field_type}/{field_name}_{idx_dim}
                    </DataItem>
                </Attribute>

                """
            return entry

        field_entries = ""
        for field_name in self.eulerian_fields:
            field_type = self.eulerian_fields_type[field_name]
            field_entries += generate_field_entry(h5_file_name, field_name, field_type)

        xdmffile = f"""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">
    <Domain>
        <Grid GridType="Uniform">
            <Time Value="{time}"/>
            <Topology TopologyType="{topology_type}" Dimensions="{grid_size_string}"/>
            <Geometry GeometryType="{geometry_type}">
                <DataItem Name="Origin" Dimensions="{self.dim}"
                NumberType="Float" Precision="{self.precision}" Format="XML">
                    {origin_string}
                </DataItem>
                <DataItem Name="Spacing" Dimensions="{self.dim}"
                NumberType="Float" Precision="{self.precision}" Format="XML">
                    {dx_string}
                </DataItem>
            </Geometry>

            {field_entries}
        </Grid>
    </Domain>
</Xdmf>
"""
        with open(h5_file_name.replace(".h5", ".xmf"), "w") as f:
            f.write(xdmffile)

    def generate_xdmf_lagrangian(self, h5_file_name, time):
        """Generate XDMF description file for Lagrangian fields.

        Currently, the XDMF file is generated for Paraview visualization only.

        Attributes
        ----------
        h5_file_name: str
            String containing name of the corresponding hdf5 file.
        time: real_dtype
            Time at which the fields are saved.
        """
        geometry_type = "XYZ" if self.dim == 3 else "XY"

        def generate_lagrangian_field_entry(
            h5_file_name,
            field_name,
            field_type,
            field_grid_size,
            lagrangian_grid_name,
        ):
            entry = f"""<Attribute Name="{field_name}" Active="1"
            AttributeType="{field_type}" Center="Node">
                <DataItem Dimensions="{field_grid_size}" NumberType="Float"
                Precision="{self.precision}" Format="HDF">
                    {h5_file_name}:/Lagrangian/{lagrangian_grid_name}/{field_type}/{field_name}
                </DataItem>
            </Attribute>

                """
            return entry

        grid_entries = ""
        for lagrangian_grid_name in self.all_lagrangian_grids:
            xmf_file_name = f"{h5_file_name.replace('.h5', '.xmf')}"
            field_entries = ""
            lagrangian_grid = self.all_lagrangian_grids[lagrangian_grid_name]
            lagrangian_grid_size = np.flip(np.array(lagrangian_grid.shape))
            lagrangian_grid_size_string = np.array2string(
                lagrangian_grid_size,
                precision=self.precision,
                separator="    ",
            )[1:-1]

            for field_name in self.lagrangian_fields_with_grid_name[
                lagrangian_grid_name
            ]:
                field_type = self.lagrangian_fields_type[field_name]

                if field_type == "Scalar":
                    field_grid_size_string = lagrangian_grid_size[0]
                elif field_type == "Vector":
                    field_grid_size_string = np.array2string(
                        lagrangian_grid_size,
                        precision=self.precision,
                        separator="    ",
                    )[1:-1]

                field_entries += generate_lagrangian_field_entry(
                    h5_file_name,
                    field_name,
                    field_type,
                    field_grid_size_string,
                    lagrangian_grid_name=lagrangian_grid_name,
                )

            if lagrangian_grid_name in self.lagrangian_grid_connection:
                topology = f"""<Topology TopologyType="Polyline">
                <DataItem DataType="Int" Dimensions="1
                {lagrangian_grid_size[0]}" Format="HDF" Precision="{self.precision}">
                    {h5_file_name}:/Lagrangian/{lagrangian_grid_name}/Connection
                </DataItem>
            </Topology>"""
            else:
                topology = f"""<Topology TopologyType="Polyvertex"
                NumberOfElements="{lagrangian_grid_size[0]}"/>"""

            grid_entries += f"""<Grid GridType="Uniform">
            <Time Value="{time}"/>
            {topology}
            <Geometry GeometryType="{geometry_type}">
                <DataItem Dimensions="{lagrangian_grid_size_string}"
                NumberType="Float" Precision="{self.precision}" Format="HDF">
                    {h5_file_name}:/Lagrangian/{lagrangian_grid_name}/Grid
                </DataItem>
            </Geometry>

            {field_entries}
        </Grid>\n
        """

        xdmffile = f"""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">
    <Domain>
        {grid_entries}
    </Domain>
</Xdmf>
"""

        with open(xmf_file_name, "w") as f:
            f.write(xdmffile)


class CosseratRodMPIIO(MPIIO):
    """
    Derived MPIIO class for Cosserat rod IO.
    """

    def __init__(
        self,
        mpi_construct,
        real_dtype=np.float64,
        master_rank=0,
    ):
        super().__init__(mpi_construct, real_dtype)
        # Initialize list for storing rods and corresponding element positions
        self.cosserat_rods = []
        self.rod_element_position = []
        self.master_rank = master_rank

    def save(self, h5_file_name, time=0.0):
        self._update_rod_element_position()
        self._save(h5_file_name=h5_file_name, time=time)

    def _update_rod_element_position(self):
        for i, rod in enumerate(self.cosserat_rods):
            self.rod_element_position[i][...] = 0.5 * (
                rod.position_collection[: self.dim, 1:]
                + rod.position_collection[: self.dim, :-1]
            )

    def add_cosserat_rod_for_io(
        self, cosserat_rod: CosseratRod, name=None, **fields_for_io
    ):
        self.cosserat_rods.append(cosserat_rod)
        self.rod_element_position.append(np.zeros((self.dim, cosserat_rod.n_elems)))
        if name is None:
            name = f"rank{self.mpi_construct.rank}_rod{len(self.cosserat_rods)}"
        self.add_as_lagrangian_fields_for_io(
            lagrangian_grid=self.rod_element_position[-1],
            lagrangian_grid_master_rank=self.master_rank,
            lagrangian_grid_name=name,
            radius=self.cosserat_rods[-1].radius,
            lagrangian_grid_connect=True,
            **fields_for_io,
        )
