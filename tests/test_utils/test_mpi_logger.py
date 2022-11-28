import pytest
from sopht_mpi.utils import MPIConstruct2D, MPILogger
import os
import glob
import logging
from mpi4py import MPI


@pytest.mark.mpi(group="MPI_utils", min_size=4)
@pytest.mark.parametrize("echo_rank", [[0], [1], [0, 1]])
@pytest.mark.parametrize(
    "level", [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]
)
@pytest.mark.parametrize("filename_timestamp", [True, False])
def test_mpi_logger_init(echo_rank, level, filename_timestamp):
    mpi_construct = MPIConstruct2D(grid_size_y=8, grid_size_x=8)
    mpi_logger = MPILogger(level=level, echo_rank=echo_rank)
    # Check that mpi construct has the same ranks as mpi logger
    # This allows independent use of mpi logger without first initializing mpi construct
    assert mpi_construct.rank == mpi_logger.rank
    assert mpi_logger.verbose == (mpi_logger.rank in echo_rank)

    # Check logging level for all handlers
    for handler in mpi_logger.logger.handlers:
        assert handler.level == level

    # Check that log file should not be created before enabling write to logfile
    assert not glob.glob("*.log")
    mpi_logger.enable_write_to_logfile(timestamp=filename_timestamp)
    assert glob.glob("*.log")

    # Release and close handlers so that next test case can start anew
    mpi_logger.remove_and_close_all_handlers()
    # remove any existing log files so as to not interfere with other test cases
    os.system("rm -rf *.log")


@pytest.mark.mpi(group="MPI_utils", min_size=4)
@pytest.mark.parametrize("echo_rank", [[0], [1], [0, 1]])
@pytest.mark.parametrize(
    "level", [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]
)
@pytest.mark.parametrize("with_rank_id", [True, False])
@pytest.mark.parametrize("with_level_name", [True, False])
def test_mpi_logger_output_to_file(echo_rank, level, with_rank_id, with_level_name):
    # Initialize mpi logger
    mpi_logger = MPILogger(
        level=level,
        echo_rank=echo_rank,
        with_rank_id=with_rank_id,
        with_level_name=with_level_name,
    )
    mpi_logger.enable_write_to_logfile()
    rank = mpi_logger.rank
    filename = mpi_logger.filename

    # Log some messages
    messages = {
        "DEBUG": f"Hello debug from {rank}",
        "INFO": f"Hello info from {rank}",
        "WARNING": f"Hello warning from {rank}",
        "ERROR": f"Hello error from {rank}",
    }
    mpi_logger.debug(messages["DEBUG"])
    mpi_logger.info(messages["INFO"])
    mpi_logger.warning(messages["WARNING"])
    mpi_logger.error(messages["ERROR"])

    # Compute expected line count
    if mpi_logger.level == logging.DEBUG:
        expected_line_count = len(messages) * len(echo_rank)
    elif mpi_logger.level == logging.INFO:
        expected_line_count = (len(messages) - 1) * len(echo_rank)
    elif mpi_logger.level == logging.WARNING:
        expected_line_count = (len(messages) - 2) * len(echo_rank)
    elif mpi_logger.level == logging.ERROR:
        expected_line_count = (len(messages) - 3) * len(echo_rank)

    # Check log outputs in file
    local_count = 0
    # Count output lines coming from each rank
    if rank in echo_rank:
        with open(filename, encoding="utf-8") as f:
            lines = f.readlines()

        # Check outputs are coming from the correct rank
        for msg_type in messages.keys():
            rank_id = f"rank[{rank}]"
            if with_rank_id:
                if with_level_name:
                    msg = f"[ {msg_type:<7} ] ({rank_id:<7}) :: {messages[msg_type]}\n"
                else:
                    msg = f"({rank_id:<7}) :: {messages[msg_type]}\n"
            else:
                if with_level_name:
                    msg = f"[ {msg_type:<7} ]  {messages[msg_type]}\n"
                else:
                    msg = f"{messages[msg_type]}\n"
            # count matching log output
            msg_count = lines.count(msg)
            local_count += msg_count

    # Get total count of log output lines
    count = MPI.COMM_WORLD.allreduce(local_count, op=MPI.SUM)
    # Check for equality
    assert count == expected_line_count

    # Release and close handlers so that next test case can start anew
    mpi_logger.remove_and_close_all_handlers()
    # remove any existing log files so as to not interfere with other test cases
    os.system("rm -rf *.log")
