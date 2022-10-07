"""Generic MPI utils for both 2D and 3D"""
import inspect


def _get_caller_name(steps=1):
    """
    Get calling function's name
    """
    frame = inspect.currentframe()
    # step back once here to account for _get_caller_name function itself
    frame = frame.f_back
    for i in range(steps):
        frame = frame.f_back
    return frame.f_code.co_name


def check_valid_ghost_size_and_kernel_support(ghost_size, kernel_support):
    """Check if ghost size and kernel support is valid"""
    if ghost_size < kernel_support:
        raise ValueError(
            f"Inconsistent ghost_size={ghost_size} and kernel_support="
            f"{kernel_support} for kernel {_get_caller_name(steps=1)}. "
            "Need to have ghost_size >= kernel_support"
        )
