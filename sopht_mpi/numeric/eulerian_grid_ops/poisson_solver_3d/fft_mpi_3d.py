from mpi4py_fft import PFFT, newDistArray, DistArray
import numpy as np


class FFTMPI3D:
    def __init__(
        self, grid_size_z, grid_size_y, grid_size_x, mpi_construct, real_t=np.float64
    ):
        """
        Creates PFFT instance based on rank distribution from mpi_construct
        """
        self.grid_size_z = grid_size_z
        self.grid_size_y = grid_size_y
        self.grid_size_x = grid_size_x
        self.real_dtype = real_t
        self.complex_dtype = np.complex64 if real_t == np.float32 else np.complex128

        # Create buffer for field distributed based on mpi_construct
        # This is the easiest way to generate a parallel fft plan consistent
        # with mpi_construct rank distribution
        self.field_buffer = DistArray(
            global_shape=(grid_size_z, grid_size_y, grid_size_x),
            subcomm=mpi_construct.rank_distribution,
            dtype=self.real_dtype,
        )
        # Use generated distributed array to create parallel fft plan
        self.fft = PFFT(
            mpi_construct.world, dtype=self.real_dtype, darray=self.field_buffer
        )
        # Create buffer for fourier field
        # set forward_output=True for the correct array shape in fourier space
        self.fourier_field_buffer = newDistArray(pfft=self.fft, forward_output=True)

    def forward(self, field, fourier_field):
        """
        Forward fft transform (real-to-complex)
        Normalisation flags set to match fftw/scipy convention
        """
        self.fft.forward(input_array=field, output_array=fourier_field, normalize=False)

    def backward(self, fourier_field, inv_fourier_field):
        """
        Backward fft transform (complex-to-real)
        Normalisation flags set to match fftw/scipy convention
        """
        self.fft.backward(
            input_array=fourier_field, output_array=inv_fourier_field, normalize=True
        )
