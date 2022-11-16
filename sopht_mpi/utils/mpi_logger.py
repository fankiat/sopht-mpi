from mpi4py import MPI
import logging
import os
import time


class MPIFileHandler(logging.FileHandler):
    """
    Custom file handler for logging output stream in MPI environment to a file.
    The code in this class is mostly taken from
    https://gist.github.com/muammar/2baec60fa8c7e62978720686895cdb9f
    """

    def __init__(
        self,
        filename,
        mode=MPI.MODE_WRONLY | MPI.MODE_CREATE | MPI.MODE_APPEND,
        encoding="utf-8",
        comm=MPI.COMM_WORLD,
    ):
        self.baseFilename = os.path.abspath(filename)
        self.mode = mode
        self.encoding = encoding
        self.comm = comm
        # initialize handler
        logging.StreamHandler.__init__(self, self._open())

    def _open(self):
        stream = MPI.File.Open(self.comm, self.baseFilename, self.mode)
        stream.Set_atomicity(True)
        return stream

    def emit(self, record):
        """
        Emit a record.

        If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.  If
        exception information is present, it is formatted using
        traceback.print_exception and appended to the stream.  If the stream
        has an 'encoding' attribute, it is used to determine how to do the
        output to the stream.

        Modification:
            stream is MPI.File, so it must use `Write_shared` method rather
            than `write` method. And `Write_shared` method only accept
            bytestring, so `encode` is used. `Write_shared` should be invoked
            only once in each all of this emit function to keep atomicity.
        """
        try:
            msg = self.format(record)
            self.stream.Write_shared((msg + self.terminator).encode(self.encoding))
        except Exception:
            self.handleError(record)

    def close(self):
        if self.stream:
            self.stream.Sync()
            self.stream.Close()
            self.stream = None


class MPILogger:
    def __init__(
        self, level=logging.INFO, echo_rank: list[int] = [], with_rank_id=True
    ):
        self.rank = MPI.COMM_WORLD.Get_rank()

        # Set which rank to log from
        self.echo_rank = echo_rank
        # Default logging on all ranks, unless specified ranks are chosen
        if len(self.echo_rank) == 0:
            self.verbose = True
        else:
            self.verbose = self.rank in self.echo_rank

        # Get logger
        self.level = level
        self.logger = logging.getLogger("rank[%i]" % self.rank)
        self.logger.setLevel(self.level)

        # Remove all currently available handler
        self.remove_and_close_all_handlers()

        # Set formatter
        if with_rank_id:
            format = "[ {levelname:<7} ] ({name:<7}) :: {message}"
        else:
            format = "[ {levelname:<7} ]  {message}"
        self.formatter = logging.Formatter(format, style="{")

        # Get stream handler for output stream to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)

    def remove_and_close_all_handlers(self):
        """
        Release filehandles to logfiles since it doesnt do that until the script is done
        running. We only need to do this here because in the test file, we are creating
        multiple loggers for different test cases, and the handlers accumulate in the
        logging under the hood.
        https://stackoverflow.com/questions/15435652/python-does-not-release-filehandles-to-logfile
        """
        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)
            handler.close()

    def enable_write_to_logfile(self, filename="logfile", timestamp=False):
        """
        Create logfile to dump all outputs. This is done separately to allow for case
        where output log to file is not needed.
        """
        # Get file handler for output stream to file
        if timestamp:
            log_filename = f'{filename}-{time.strftime("%Y%m%d-%H%M%S")}.log'
        else:
            log_filename = f"{filename}.log"
        self.filename = log_filename
        mpi_file_handler = MPIFileHandler(self.filename)
        mpi_file_handler.setLevel(self.level)
        mpi_file_handler.setFormatter(self.formatter)
        self.logger.addHandler(mpi_file_handler)

    def debug(self, msg):
        if self.verbose:
            self.logger.debug(msg)

    def info(self, msg):
        if self.verbose:
            self.logger.info(msg)

    def warning(self, msg):
        if self.verbose:
            self.logger.warning(msg)

    def error(self, msg):
        if self.verbose:
            self.logger.error(msg)

    def critical(self, msg):
        if self.verbose:
            self.logger.critical(msg)


# Initialize a default logger here
logger = MPILogger(echo_rank=[0], with_rank_id=False)
