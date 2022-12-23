"""Mpi-supported kernels applying laplacian filter on 3d scalar and vector fields"""
import pystencils as ps
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d import (
    gen_elementwise_copy_pyst_kernel_3d,
    gen_elementwise_saxpby_pyst_kernel_3d,
    gen_set_fixed_val_pyst_kernel_3d,
)
from sopht_mpi.utils.mpi_utils import check_valid_ghost_size_and_kernel_support
from sopht.utils.pyst_kernel_config import get_pyst_dtype, get_pyst_kernel_config
from sopht.utils.field import VectorField
from mpi4py import MPI


def gen_laplacian_filter_mpi_kernel_3d(  # noqa: C901
    mpi_construct,
    ghost_exchange_communicator,
    filter_order,
    filter_flux_buffer,
    field_buffer,
    real_t,
    field_type="scalar",
    filter_type="multiplicative",
    filter_flux_buffer_boundary_width=1,
):
    """
    MPI-supported Laplacian filter kernel generator. Based on the field type
    filter kernels for both scalar and vectorial field can be constructed.
    One dimensional laplacian filter applied on the field in 3D.

    Notes
    -----
    For details regarding the numerics behind the filtering, refer to [1]_, [2]_.
    .. [1] Jeanmart, H., & Winckelmans, G. (2007). Investigation of eddy-viscosity
       models modified using discrete filters: a simplified “regularized variational
       multiscale model” and an “enhanced field model”. Physics of fluids, 19(5), 055110.
    .. [2] Lorieul, G. (2018). Development and validation of a 2D Vortex Particle-Mesh
       method for incompressible multiphase flows (Doctoral dissertation,
       Université Catholique de Louvain).
    """

    if filter_order < 0 or not isinstance(filter_order, int):
        raise ValueError("Invalid filter order")
    if field_type != "scalar" and field_type != "vector":
        raise ValueError("Invalid field type")
    if filter_flux_buffer_boundary_width <= 0 or not isinstance(
        filter_flux_buffer_boundary_width, int
    ):
        raise ValueError("Invalid value for filter flux buffer boundary zone")


    kernel_support = 1
    # define this here so that ghost size and kernel support is checked during
    # generation phase itself
    gen_laplacian_filter_mpi_kernel_3d.kernel_support = kernel_support
    check_valid_ghost_size_and_kernel_support(
        ghost_size=ghost_exchange_communicator.ghost_size,
        kernel_support=gen_laplacian_filter_mpi_kernel_3d.kernel_support,
    )

    pyst_dtype = get_pyst_dtype(real_t)
    kernel_config = get_pyst_kernel_config(real_t, num_threads=False)
    grid_info = "3D"

    # Compile laplacian filter kernels in each direction
    @ps.kernel
    def _laplacian_filter_3d_x():
        filter_flux, field = ps.fields(
            f"filter_flux, field : {pyst_dtype}[{grid_info}]"
        )
        filter_flux[0, 0, 0] @= 0.25 * (
            -field[0, 0, 1] - field[0, 0, -1] + 2 * field[0, 0, 0]
        )

    laplacian_filter_3d_x = ps.create_kernel(
        _laplacian_filter_3d_x, config=kernel_config
    ).compile()

    @ps.kernel
    def _laplacian_filter_3d_y():
        filter_flux, field = ps.fields(
            f"filter_flux, field : {pyst_dtype}[{grid_info}]"
        )
        filter_flux[0, 0, 0] @= 0.25 * (
            -field[0, 1, 0] - field[0, -1, 0] + 2 * field[0, 0, 0]
        )

    laplacian_filter_3d_y = ps.create_kernel(
        _laplacian_filter_3d_y, config=kernel_config
    ).compile()

    @ps.kernel
    def _laplacian_filter_3d_z():
        filter_flux, field = ps.fields(
            f"filter_flux, field : {pyst_dtype}[{grid_info}]"
        )
        filter_flux[0, 0, 0] @= 0.25 * (
            -field[1, 0, 0] - field[-1, 0, 0] + 2 * field[0, 0, 0]
        )

    laplacian_filter_3d_z = ps.create_kernel(
        _laplacian_filter_3d_z, config=kernel_config
    ).compile()

    # Compile other elementwise kernels
    elementwise_copy_3d = gen_elementwise_copy_pyst_kernel_3d(real_t=real_t)
    elementwise_saxpby_3d = gen_elementwise_saxpby_pyst_kernel_3d(real_t=real_t)
    # to set boundary zone = 0 at physical domain boundary
    z_next, y_next, x_next = mpi_construct.next_grid_along
    z_previous, y_previous, x_previous = mpi_construct.previous_grid_along
    set_fixed_val_kernel_3d = gen_set_fixed_val_pyst_kernel_3d(real_t=real_t)

    # some MPI-supported helper functions to reduce redundant code
    def _clear_val_at_physical_domain_boundary_kernel_3d(field):
        """Set physical domain boundary of a scalar field to zero"""
        boundary_width = filter_flux_buffer_boundary_width
        ghost_size = ghost_exchange_communicator.ghost_size
        if x_previous == MPI.PROC_NULL:
            set_fixed_val_kernel_3d(
                field=field[:, :, : ghost_size + boundary_width],
                fixed_val=0,
            )
        if x_next == MPI.PROC_NULL:
            set_fixed_val_kernel_3d(
                field=field[:, :, -ghost_size - boundary_width :],
                fixed_val=0,
            )
        if y_previous == MPI.PROC_NULL:
            set_fixed_val_kernel_3d(
                field=field[:, : ghost_size + boundary_width, :],
                fixed_val=0,
            )
        if y_next == MPI.PROC_NULL:
            set_fixed_val_kernel_3d(
                field=field[:, -ghost_size - boundary_width :, :],
                fixed_val=0,
            )
        if z_previous == MPI.PROC_NULL:
            set_fixed_val_kernel_3d(
                field=field[: ghost_size + boundary_width, :, :],
                fixed_val=0,
            )
        if z_next == MPI.PROC_NULL:
            set_fixed_val_kernel_3d(
                field=field[-ghost_size - boundary_width :, :, :],
                fixed_val=0,
            )

    def _laplacian_filter_3d_mpi(filter_kernel, field_buffer, filter_flux_buffer):
        """
        Generic filtering steps in MPI given a filter kernel as below:
        (1) begin ghosting
        (2) apply filter in inner cells
        (3) wait for ghosting to complete
        (4) apply filter on outer cells (with width kernel_support adjacent to ghost)
        """

        ghost_size = ghost_exchange_communicator.ghost_size
        # begin ghost comm.
        ghost_exchange_communicator.exchange_scalar_field_init(field_buffer)

        # crunch interior stencil
        filter_kernel(
            filter_flux=filter_flux_buffer[
                ghost_size:-ghost_size,
                ghost_size:-ghost_size,
                ghost_size:-ghost_size,
            ],
            field=field_buffer[
                ghost_size:-ghost_size,
                ghost_size:-ghost_size,
                ghost_size:-ghost_size,
            ],
        )
        # finalise ghost comm.
        ghost_exchange_communicator.exchange_finalise()

        # crunch boundary numbers
        # NOTE: we pass in arrays of width 3 * kernel support size because the
        # interior stencil computation leaves out a width of kernel_support.
        # Since the support needed by the kernel is kernel_support on each side,
        # we need to pass an array of width 3 * kernel_support, starting from
        # index +/-(ghost_size - kernel_support) on the lower and upper end.
        # Pystencils then automatically sets the kernel comp. bounds and
        # crunches numbers in the kernel_support thickness zone at the boundary.
        # Start of Z axis
        filter_kernel(
            filter_flux=filter_flux_buffer[
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
                ghost_size:-ghost_size,
                ghost_size:-ghost_size,
            ],
            field=field_buffer[
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
                ghost_size:-ghost_size,
                ghost_size:-ghost_size,
            ],
        )
        # End of Z axis
        filter_kernel(
            filter_flux=filter_flux_buffer[
                -(ghost_size + 2 * kernel_support) : filter_flux_buffer.shape[0]
                - (ghost_size - kernel_support),
                ghost_size:-ghost_size,
                ghost_size:-ghost_size,
            ],
            field=field_buffer[
                -(ghost_size + 2 * kernel_support) : field_buffer.shape[0]
                - (ghost_size - kernel_support),
                ghost_size:-ghost_size,
                ghost_size:-ghost_size,
            ],
        )
        # Start of Y axis
        filter_kernel(
            filter_flux=filter_flux_buffer[
                :,
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
                ghost_size:-ghost_size,
            ],
            field=field_buffer[
                :,
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
                ghost_size:-ghost_size,
            ],
        )
        # End of Y axis
        filter_kernel(
            filter_flux=filter_flux_buffer[
                :,
                -(ghost_size + 2 * kernel_support) : filter_flux_buffer.shape[1]
                - (ghost_size - kernel_support),
                ghost_size:-ghost_size,
            ],
            field=field_buffer[
                :,
                -(ghost_size + 2 * kernel_support) : field_buffer.shape[1]
                - (ghost_size - kernel_support),
                ghost_size:-ghost_size,
            ],
        )
        # Start of X axis
        filter_kernel(
            filter_flux=filter_flux_buffer[
                :,
                :,
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
            ],
            field=field_buffer[
                :,
                :,
                ghost_size - kernel_support : ghost_size + 2 * kernel_support,
            ],
        )
        # End of X axis
        filter_kernel(
            filter_flux=filter_flux_buffer[
                :,
                :,
                -(ghost_size + 2 * kernel_support) : filter_flux_buffer.shape[2]
                - (ghost_size - kernel_support),
            ],
            field=field_buffer[
                :,
                :,
                -(ghost_size + 2 * kernel_support) : field_buffer.shape[2]
                - (ghost_size - kernel_support),
            ],
        )

    def scalar_field_multiplicative_filter_mpi_kernel_3d(scalar_field):
        """
        Applies multiplicative Laplacian filter on any scalar field.
        """
        # define kernel support for kernel
        scalar_field_multiplicative_filter_mpi_kernel_3d.kernel_support = (
            gen_laplacian_filter_mpi_kernel_3d.kernel_support
        )

        # Set physical boundary domain to zero for filter_flux_buffer
        _clear_val_at_physical_domain_boundary_kernel_3d(field=filter_flux_buffer)
        # Copy scalar field into field buffer
        elementwise_copy_3d(field=field_buffer, rhs_field=scalar_field)

        # Apply laplacian filtering
        for _ in range(filter_order):
            # (1) Laplacian filter in x direction
            _laplacian_filter_3d_mpi(
                filter_kernel=laplacian_filter_3d_x,
                field_buffer=field_buffer,
                filter_flux_buffer=filter_flux_buffer,
            )
            # clear out noise from physical domain boundary from ghosting before copy
            _clear_val_at_physical_domain_boundary_kernel_3d(field=filter_flux_buffer)
            elementwise_copy_3d(field=field_buffer, rhs_field=filter_flux_buffer)

            # Laplacian filter in y direction
            _laplacian_filter_3d_mpi(
                filter_kernel=laplacian_filter_3d_y,
                field_buffer=field_buffer,
                filter_flux_buffer=filter_flux_buffer,
            )
            # clear out noise from physical domain boundary from ghosting before copy
            _clear_val_at_physical_domain_boundary_kernel_3d(field=filter_flux_buffer)
            elementwise_copy_3d(field=field_buffer, rhs_field=filter_flux_buffer)

            # Laplacian filter in z direction
            _laplacian_filter_3d_mpi(
                filter_kernel=laplacian_filter_3d_z,
                field_buffer=field_buffer,
                filter_flux_buffer=filter_flux_buffer,
            )
            # clear out noise from physical domain boundary from ghosting before copy
            _clear_val_at_physical_domain_boundary_kernel_3d(field=filter_flux_buffer)
            elementwise_copy_3d(field=field_buffer, rhs_field=filter_flux_buffer)

        elementwise_saxpby_3d(
            sum_field=scalar_field,
            field_1=scalar_field,
            field_1_prefac=1.0,
            field_2=filter_flux_buffer,
            field_2_prefac=-1.0,
        )

    def scalar_field_convolution_filter_mpi_kernel_3d(scalar_field):
        """
        Applies convolution Laplacian filter on any scalar field.
        """
        # define kernel support for kernel
        scalar_field_convolution_filter_mpi_kernel_3d.kernel_support = (
            gen_laplacian_filter_mpi_kernel_3d.kernel_support
        )

        # Set physical boundary domain to zero for filter_flux_buffer
        _clear_val_at_physical_domain_boundary_kernel_3d(field=filter_flux_buffer)

        # Laplacian filter in x direction
        elementwise_copy_3d(field=field_buffer, rhs_field=scalar_field)
        for _ in range(filter_order):
            _laplacian_filter_3d_mpi(
                filter_kernel=laplacian_filter_3d_x,
                field_buffer=field_buffer,
                filter_flux_buffer=filter_flux_buffer,
            )
            _clear_val_at_physical_domain_boundary_kernel_3d(field=filter_flux_buffer)
            elementwise_copy_3d(field=field_buffer, rhs_field=filter_flux_buffer)
        elementwise_saxpby_3d(
            sum_field=scalar_field,
            field_1=scalar_field,
            field_1_prefac=1.0,
            field_2=filter_flux_buffer,
            field_2_prefac=-1.0,
        )

        # Laplacian filter in y direction
        elementwise_copy_3d(field=field_buffer, rhs_field=scalar_field)
        for _ in range(filter_order):
            _laplacian_filter_3d_mpi(
                filter_kernel=laplacian_filter_3d_y,
                field_buffer=field_buffer,
                filter_flux_buffer=filter_flux_buffer,
            )
            _clear_val_at_physical_domain_boundary_kernel_3d(field=filter_flux_buffer)
            elementwise_copy_3d(field=field_buffer, rhs_field=filter_flux_buffer)
        elementwise_saxpby_3d(
            sum_field=scalar_field,
            field_1=scalar_field,
            field_1_prefac=1.0,
            field_2=filter_flux_buffer,
            field_2_prefac=-1.0,
        )

        # Laplacian filter in z direction
        elementwise_copy_3d(field=field_buffer, rhs_field=scalar_field)
        for _ in range(filter_order):
            _laplacian_filter_3d_mpi(
                filter_kernel=laplacian_filter_3d_z,
                field_buffer=field_buffer,
                filter_flux_buffer=filter_flux_buffer,
            )
            _clear_val_at_physical_domain_boundary_kernel_3d(field=filter_flux_buffer)
            elementwise_copy_3d(field=field_buffer, rhs_field=filter_flux_buffer)
        elementwise_saxpby_3d(
            sum_field=scalar_field,
            field_1=scalar_field,
            field_1_prefac=1.0,
            field_2=filter_flux_buffer,
            field_2_prefac=-1.0,
        )

    match filter_type:
        case "multiplicative":
            scalar_field_filter_mpi_kernel_3d = (
                scalar_field_multiplicative_filter_mpi_kernel_3d
            )
        case "convolution":
            scalar_field_filter_mpi_kernel_3d = (
                scalar_field_convolution_filter_mpi_kernel_3d
            )
        case _:
            raise ValueError("Invalid filter type")

    # Depending on the field type return the relevant filter implementation
    match field_type:
        case "scalar":
            return scalar_field_filter_mpi_kernel_3d
        case "vector":
            x_axis_idx = VectorField.x_axis_idx()
            y_axis_idx = VectorField.y_axis_idx()
            z_axis_idx = VectorField.z_axis_idx()

            def vector_field_filter_kernel_3d(vector_field) -> None:
                """
                Applies laplacian filter on any vector field.
                """
                vector_field_filter_kernel_3d.kernel_support = (
                    gen_laplacian_filter_mpi_kernel_3d.kernel_support
                )
                scalar_field_filter_mpi_kernel_3d(scalar_field=vector_field[x_axis_idx])
                scalar_field_filter_mpi_kernel_3d(scalar_field=vector_field[y_axis_idx])
                scalar_field_filter_mpi_kernel_3d(scalar_field=vector_field[z_axis_idx])

            return vector_field_filter_kernel_3d
        case _:
            raise ValueError("Invalid field type")
