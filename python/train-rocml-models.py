import glob
from scripting import parse_arguments, check_arguments
from gfem import build_gfem_models
from rocml import train_rocml_models
from visualize import visualize_rocml_model, compose_rocml_plots

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "train-rocml-models.py")

# Load valid arguments
locals().update(valid_args)

# Get samples filepath
if "benchmark" in source:
    files = ["assets/data/benchmark-samples-normalized.csv"]
    ml_models = ["DT", "KN", "RF", "NN1", "NN2", "NN3"]
else:
    files = sorted(glob.glob(f"assets/data/synthetic*.csv"))
    ml_models = ["DT", "KN"]

# Initialize gfem models
gfem_models = []

# Build GFEM models
for source in files:
    gfem_models.extend(build_gfem_models(source))

# Build RocML models
rocml_models = train_rocml_models(gfem_models, ml_models)

# Visualize RocML models
for model in rocml_models:
    visualize_rocml_model(model)
    compose_rocml_plots(model)

print("RocML visualized!")