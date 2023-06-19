import os
import pandas as pd
from magemin import parse_arguments_build_db

# Setup directories
job_directory = f"{os.getcwd()}/.job"
#scratch = os.environ["SCRATCH"]
#data_dir = os.path.join(scratch, "/magemin")
data_dir = os.path.join("runs")

# Output directories
os.mkdir(job_directory)
os.mkdir(data_dir)

# Batches
df = pd.read_csv("data/earthchem-ig.csv")
num_observations = df.shape[0]
batch_size = 28
batches = num_observations // batch_size

# MAGEMin parameters
P_range = [10, 110, 10]
T_range = [500, 2500, 200]
type = "batch"
parallel = True
nprocs = 28

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
        fh.write(f"#SBATCH --output=.out/{batch_name}.out\n")
        fh.write(f"#SBATCH --error=.out/{batch_name}.err\n")
        fh.write(f"#SBATCH --time=2-00:00\n")
        fh.write(f"#SBATCH --mem=12000\n")
        fh.write(f"#SBATCH --qos=normal\n")
        fh.write(f"#SBATCH --mail-type=ALL\n")
        fh.write(f"#SBATCH --mail-user=buchanan.kerswell@umontpellier.fr\n")
        fh.write(
            f"make build_database "
            f"PRANGE=\"{P_range}\" "
            f"TRANGE=\"{T_range}\" "
            f"TYPE={type} "
            f"N={batch_size} "
            f"K={k} "
            f"PARALLEL={parallel} "
            f"NPROCS={nprocs} "
            f"OUTDIR={data_dir}"
        )

    #os.system(f"sbatch {job_file}")