"""MPI-supported kernels for penalising field boundary in 2D."""
import numpy as np
import pystencils as ps
import sympy as sp
from sopht.utils.pyst_kernel_config import get_pyst_dtype, get_pyst_kernel_config
from mpi4py import MPI


def gen_penalise_field_boundary_pyst_mpi_kernel_2d(
    width,
    dx,
    x_grid_field,
    y_grid_field,
    real_t,
    mpi_construct,
    ghost_exchange_communicator,
):
    """MPI-supported 2D penalise field boundary kernel generator."""
    if width < 0 or not isinstance(width, int):
        raise ValueError("invalid zone width")

    gen_penalise_field_boundary_pyst_mpi_kernel_2d.kernel_support = 0

    if width == 0:
        # bypass option to prevent penalisation, done this way since by
        # default to avoid artifacts one must use penalisation...
        def penalise_field_boundary_pyst_mpi_kernel_2d(field):
            pass

    else:
        # get rank on neighboring block
        y_next, x_next = mpi_construct.next_grid_along
        y_previous, x_previous = mpi_construct.previous_grid_along
        # get ghost size for use in indexing later
        ghost_size = ghost_exchange_communicator.ghost_size

        pyst_dtype = get_pyst_dtype(real_t)
        grid_info = "2D"
        x_grid_field_start = x_grid_field[ghost_size, ghost_size]
        y_grid_field_start = y_grid_field[ghost_size, ghost_size]
        x_grid_field_end = x_grid_field[ghost_size, -(ghost_size + 1)]
        y_grid_field_end = y_grid_field[-(ghost_size + 1), ghost_size]

        sine_prefactor = (np.pi / 2) / (width * dx)

        x_front_boundary_slice = ps.make_slice[:, : (width + ghost_size)]
        x_front_boundary_kernel_config = get_pyst_kernel_config(
            pyst_dtype=real_t,
            num_threads=False,
            iteration_slice=x_front_boundary_slice,
        )
        x_back_boundary_slice = ps.make_slice[:, -(width + ghost_size) :]
        x_back_boundary_kernel_config = get_pyst_kernel_config(
            pyst_dtype=real_t,
            num_threads=False,
            iteration_slice=x_back_boundary_slice,
        )

        @ps.kernel
        def penalise_field_x_front_boundary_stencil_2d():
            field, x_grid_field = ps.fields(
                f"field, x_grid_field : {pyst_dtype}[{grid_info}]"
            )
            field[0, 0] @= field[0, 0] * sp.sin(
                sine_prefactor * (x_grid_field[0, 0] - x_grid_field_start)
            )

        penalise_field_x_front_boundary_kernel_2d = ps.create_kernel(
            penalise_field_x_front_boundary_stencil_2d,
            config=x_front_boundary_kernel_config,
        ).compile()

        @ps.kernel
        def penalise_field_x_back_boundary_stencil_2d():
            field, x_grid_field = ps.fields(
                f"field, x_grid_field : {pyst_dtype}[{grid_info}]"
            )
            field[0, 0] @= field[0, 0] * sp.sin(
                sine_prefactor * (x_grid_field_end - x_grid_field[0, 0])
            )

        penalise_field_x_back_boundary_kernel_2d = ps.create_kernel(
            penalise_field_x_back_boundary_stencil_2d,
            config=x_back_boundary_kernel_config,
        ).compile()

        y_front_boundary_slice = ps.make_slice[: (ghost_size + width), :]
        y_front_boundary_kernel_config = get_pyst_kernel_config(
            pyst_dtype=real_t,
            num_threads=False,
            iteration_slice=y_front_boundary_slice,
        )
        y_back_boundary_slice = ps.make_slice[-(ghost_size + width) :, :]
        y_back_boundary_kernel_config = get_pyst_kernel_config(
            pyst_dtype=real_t,
            num_threads=False,
            iteration_slice=y_back_boundary_slice,
        )

        @ps.kernel
        def penalise_field_y_front_boundary_stencil_2d():
            field, y_grid_field = ps.fields(
                f"field, y_grid_field : {pyst_dtype}[{grid_info}]"
            )
            field[0, 0] @= field[0, 0] * sp.sin(
                sine_prefactor * (y_grid_field[0, 0] - y_grid_field_start)
            )

        penalise_field_y_front_boundary_kernel_2d = ps.create_kernel(
            penalise_field_y_front_boundary_stencil_2d,
            config=y_front_boundary_kernel_config,
        ).compile()

        @ps.kernel
        def penalise_field_y_back_boundary_stencil_2d():
            field, y_grid_field = ps.fields(
                f"field, y_grid_field : {pyst_dtype}[{grid_info}]"
            )
            field[0, 0] @= field[0, 0] * sp.sin(
                sine_prefactor * (y_grid_field_end - y_grid_field[0, 0])
            )

        penalise_field_y_back_boundary_kernel_2d = ps.create_kernel(
            penalise_field_y_back_boundary_stencil_2d,
            config=y_back_boundary_kernel_config,
        ).compile()

        def penalise_field_boundary_pyst_mpi_kernel_2d(field):
            """2D penalise field boundary kernel.
            Penalises field on the boundaries in a sine wave fashion
            in the given width in X and Y direction
            field: field to be penalised
            """
            penalise_field_boundary_pyst_mpi_kernel_2d.kernel_support = (
                gen_penalise_field_boundary_pyst_mpi_kernel_2d.kernel_support
            )
            # Penalize boundary when neighboring block is MPI.PROC_NULL
            # first along X
            if x_previous == MPI.PROC_NULL:
                field[:, : (ghost_size + width)] = field[
                    :, (ghost_size + width - 1) : (ghost_size + width)
                ]
                penalise_field_x_front_boundary_kernel_2d(
                    field=field, x_grid_field=x_grid_field
                )

            if x_next == MPI.PROC_NULL:
                field[:, -(ghost_size + width) :] = field[
                    :, -(ghost_size + width) : (-(ghost_size + width) + 1)
                ]
                penalise_field_x_back_boundary_kernel_2d(
                    field=field, x_grid_field=x_grid_field
                )

            # then along Y
            if y_previous == MPI.PROC_NULL:
                field[: (ghost_size + width), :] = field[
                    (ghost_size + width - 1) : (ghost_size + width), :
                ]
                penalise_field_y_front_boundary_kernel_2d(
                    field=field, y_grid_field=y_grid_field
                )

            if y_next == MPI.PROC_NULL:
                field[-(ghost_size + width) :, :] = field[
                    -(ghost_size + width) : (-(ghost_size + width) + 1), :
                ]
                penalise_field_y_back_boundary_kernel_2d(
                    field=field, y_grid_field=y_grid_field
                )

    return penalise_field_boundary_pyst_mpi_kernel_2d
