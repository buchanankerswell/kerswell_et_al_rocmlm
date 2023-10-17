from scripting import parse_arguments, check_arguments
from gfem import build_gfem_models
from rocml import train_rocml_models
from visualize import visualize_rocml_model, compose_rocml_plots

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "train-rocml-models.py")

# Load valid arguments
locals().update(valid_args)

# Build GFEM models
gfem_models = build_gfem_models(source)

# Build RocML models
rocml_models = train_rocml_models(gfem_models)

for model in rocml_models:
    visualize_rocml_model(model)
    compose_rocml_plots(model)
print("RocML visualized!")