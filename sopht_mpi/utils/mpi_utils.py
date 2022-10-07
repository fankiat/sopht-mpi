"""Generic MPI utils for both 2D and 3D"""
import inspect


def get_caller_name():
    return inspect.currentframe().f_back.f_back.f_code.co_name


def check_valid_ghost_size_and_kernel_support(ghost_size, kernel_support):
    """Check if ghost size and kernel support is valid"""
    if ghost_size < kernel_support:
        raise ValueError(
            f"Inconsistent ghost_size={ghost_size} and kernel_support="
            f"{kernel_support} for kernel {get_caller_name()}. "
            "Need to have ghost_size >= kernel_support"
        )
