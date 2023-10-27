import glob
from scripting import parse_arguments, check_arguments
from gfem import get_sampleids, build_gfem_models
from visualize import (visualize_gfem_design, visualize_gfem_efficiency, visualize_gfem,
                       visualize_gfem_diff, compose_dataset_plots)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "build-gfem-models.py")

# Load valid arguments
locals().update(valid_args)

# Get samples filepath
if "benchmark" in source:
    files = ["assets/data/benchmark-samples-normalized.csv"]
else:
    files = sorted(glob.glob(f"assets/data/synthetic*.csv"))

# Initialize gfem models
gfem_models = []

# Build GFEM models
for source in files:
    gfem_models.extend(build_gfem_models(source))

# Visualize GFEM models
visualize_gfem(gfem_models)
visualize_gfem_diff(gfem_models)
compose_dataset_plots(gfem_models)
visualize_gfem_design()
visualize_gfem_efficiency()

print("GFEM models visualized!")