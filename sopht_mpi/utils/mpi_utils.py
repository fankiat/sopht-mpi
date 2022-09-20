"""Generic MPI utils for both 2D and 3D"""


def is_valid_ghost_size_and_kernel_support(ghost_size, kernel_support):
    """Check if ghost size and kernel support is valid"""
    return ghost_size >= kernel_support
