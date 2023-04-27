"""Script for generating submission files for cluster jobs"""
from typing import Optional, Literal

SUPPORTED_CLUSTERS = Literal["stampede", "expanse", "bridges"]
STAMPEDE_INFO_DICT = {"account": "TG-MCB190004", "compute": "normal"}
EXPANSE_INFO_DICT = {"account": "uic409", "shared": "shared", "compute": "compute"}
BRIDGES_INFO_DICT = {"account": "mcb200029p", "shared": "RM-shared", "compute": "RM"}
CLUSTER_MAP = {
    "stampede": STAMPEDE_INFO_DICT,
    "expanse": EXPANSE_INFO_DICT,
    "bridges": BRIDGES_INFO_DICT,
}


def create_submit_file(
    program_name: str,
    environment_name: str,
    cluster_name: SUPPORTED_CLUSTERS,
    output_file_name: Optional[str] = "%x_%j.out",
    error_file_name: Optional[str] = "%x_%j.err",
    partition: str = "compute",
    num_nodes: int = 1,
    num_tasks_per_node: int = 4,
    memory: int = 249325,
    time: str = "48:00:00",
    verbose: bool = False,
    mail_user: Optional[str] = None,
    mail_type: Optional[str] = "ALL",
    other_cli_arguments: str = "",
    submit_filename: str = "submit.sh",
) -> None:
    """
    The submission script can be roughly divided into the following parts:
    1. sbatch directives / headers
    2. printing details of launched job
    3. module and environment preparation
    4. running executable
    """
    cluster_info_dict = CLUSTER_MAP[cluster_name]
    f = open(submit_filename, "w")

    # (1) Write sbatch directives
    f.writelines(
        [
            "#!/bin/bash\n",
            "\n",
            f"#SBATCH -J {program_name.replace('.py', '')}\n",
            f"#SBATCH -p {cluster_info_dict.get(partition)}\n",
            f"#SBATCH -N {num_nodes}\n",
            f"#SBATCH --ntasks-per-node={num_tasks_per_node}\n",
            f"#SBATCH -t {time}\n",
            f"#SBATCH --account={cluster_info_dict.get('account')}\n",
            f"#SBATCH -o {output_file_name}\n",
            f"#SBATCH -e {error_file_name}\n",
        ]
    )
    if mail_user:
        if "@" not in mail_user:
            print(
                f"WARNING: mail notification might not work without full email domain."
            )
        f.write(f"#SBATCH --mail-user={mail_user}\n")
        f.write(f"#SBATCH --mail-type={mail_type}\n")
    # only expanse can set memory
    if cluster_name == "expanse":
        f.write(f"#SBATCH --mem={memory}M\n")
    if verbose:
        f.write("#SBATCH -v\n")
    f.writelines(["\n"])

    # (2) Print details about launched job
    f.writelines(
        [
            "date\n",
            "echo Job name: $SLURM_JOB_NAME\n",
            "echo Execution dir: $SLURM_SUBMIT_DIR\n",
            "echo Number of processes: $SLURM_NTASKS\n",
            "\n",
        ]
    )

    # (3) Prepare modules and environment
    f.write(f"module reset\n")
    if cluster_name == "stampede":
        # Other mpi libraries are loaded by default (intel mpi)
        f.writelines([f"module unload python2\n", f"module load phdf5\n"])
    elif cluster_name == "expanse":
        f.write("module load gcc openmpi fftw hdf5 anaconda3\n")
    elif cluster_name == "bridges":
        f.write(
            "module load anaconda3 gcc/10.2.0 openmpi/4.0.5-gcc10.2.0 "
            "phdf5/1.10.7-openmpi4.0.5-gcc10.2.0 fftw\n"
        )
    # activate python environment
    f.write(f"source deactivate\n")
    f.write(f"source activate {environment_name}\n")

    # (4) Run executable
    f.write("\n")
    if cluster_name == "stampede":
        f.write(f"ibrun python -u {program_name} {other_cli_arguments}\n")
    else:
        f.write(
            f"mpiexec -n ${{SLURM_NTASKS}} python -u {program_name} {other_cli_arguments}\n"
        )

    f.close()

    return submit_filename


if __name__ == "__main__":
    PROGRAM_NAME = "flow_past_sphere_case.py"
    ENVIRONMENT_NAME = "sopht-mpi-env"
    PARTITION = "compute"
    TIME = "06:00:00"
    NUM_NODES = 2
    NUM_TASKS_PER_NODE = 32
    MAIL_USER = "email@email.com"
    CLUSTER_NAME: SUPPORTED_CLUSTERS = "stampede"

    submit_filename = create_submit_file(
        program_name=PROGRAM_NAME,
        environment_name=ENVIRONMENT_NAME,
        cluster_name=CLUSTER_NAME,
        time=TIME,
        partition=PARTITION,
        num_nodes=NUM_NODES,
        num_tasks_per_node=NUM_TASKS_PER_NODE,
        mail_user=MAIL_USER,
    )

    print(f"Generated submission script file : {submit_filename}")
