import os
import pandas as pd

# Directories setup
job_directory = f"{os.getcwd()}/.job"
#scratch = os.environ["SCRATCH"]
#data_dir = os.path.join(scratch, "/MAGEMin/runs")
data_dir = os.path.join("runs")

os.mkdir(job_directory)
os.mkdir(data_dir)

# SLURM setup
account_name = "geodyn"
#partition = "gm_geodyn"
partition = "gm_base"
nodes = 1
n_tasks = 28

# MAGEMin setup: 128 x 128 grid
P_range = [10, 110, 0.78125]
T_range = [500, 2500, 15.625]
source = "earthchem"
strategy = "batch"
parallel = True

# Earthchem batches setup
df = pd.read_csv("assets/data/earthchem-ig.csv")
num_observations = df.shape[0]
batch_size = 28
batches = num_observations // batch_size

# Submit jobs
for k in range(batches):
    # Create batch-specific job file and directory
    batch_name = f"ec-batch-{k}"
    job_file = os.path.join(job_directory, f"{batch_name}.job")
    out_dir = os.path.join(data_dir, batch_name)

    # Create sample directories
    os.mkdir(out_dir)

    with open(job_file, 'w') as fh:  # Open file in write mode
        fh.write("#!/bin/bash\n")
        fh.write(f"#SBATCH --job-name={batch_name}.job\n")
        fh.write(f"#SBATCH --account={account_name}\n")
        fh.write(f"#SBATCH --partition={partition}\n")
        fh.write(f"#SBATCH --nodes={nodes}\n")
        fh.write(f"#SBATCH --n-tasks={n_tasks}\n")
        fh.write(f"#SBATCH --n-tasks-per-node={n_tasks / nodes}\n")
        fh.write(f"#SBATCH --n-tasks-per-core=1\n")
        fh.write(f"#SBATCH --output=.out/{batch_name}.out\n")
        fh.write(f"#SBATCH --error=.out/{batch_name}.err\n")
        fh.write(f"#SBATCH --time=04:00:00\n")
        fh.write(f"#SBATCH --qos=normal\n")
        fh.write(f"#SBATCH --mail-type=ALL\n")
        fh.write(f"#SBATCH --mail-user=buchanan.kerswell@umontpellier.fr\n")
        fh.write(f"module purge\n")
        fh.write(f"module load cv-standard\n")
        fh.write(f"module load openmpi\n")
        fh.write(f"module load Nlopt/2.6.1\n")
        fh.write(f"echo 'Running on: $SLURM_NODELIST'\n")
        fh.write(f"echo 'SLURM_NTASKS=$SLURM_NTASKS'\n")
        fh.write(f"echo 'SLURM_NTASKS_PER_NODE=$SLURM_NTASKS_PER_NODE'\n")
        fh.write(f"echo 'SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK'\n")
        fh.write(f"echo 'SLURM_NNODES=$SLURM_NNODES'\n")
        fh.write(f"echo 'SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE'\n")
        fh.write(f"cd kerswell_et_al_madnn\n")
        fh.write(
            f"make build_database "
            f"PRANGE=\"{P_range}\" "
            f"TRANGE=\"{T_range}\" "
            f"DATASOURCE={source} "
            f"SAMPLESTRATEGY={strategy} "
            f"N={batch_size} "
            f"K={k} "
            f"PARALLEL={parallel} "
            f"NPROCS={n_tasks} "
            f"OUTDIR={data_dir}"
        )

    #os.system(f"sbatch {job_file}")