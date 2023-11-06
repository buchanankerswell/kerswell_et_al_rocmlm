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
    source = "assets/data/benchmark-samples-normalized.csv"
    programs = ["magemin", "perplex"]
    nbatches = 1
    batch = 0
else:
    source = f"assets/data/synthetic-samples.csv"
    programs = ["perplex"]
    nbatches = 32
    batch = 0

# Initialize gfem models
gfem_models = []

# Build GFEM models
gfem_models.extend(build_gfem_models(source, programs, batch=batch, nbatches=nbatches))

# Visualize GFEM models
visualize_gfem(gfem_models)
visualize_gfem_diff(gfem_models)
compose_dataset_plots(gfem_models)
visualize_gfem_design()
visualize_gfem_efficiency()

print("GFEM models visualized!")