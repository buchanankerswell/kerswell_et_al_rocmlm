from scripting import parse_arguments, check_arguments
from gfem import build_gfem_models
from rocml import build_rocml_model
from visualize import visualize_rocml_model, compose_rocml_plots

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "train-rocml-models.py")

# Load valid arguments
locals().update(valid_args)

# Get GFEM models
gfem_models = build_gfem_models(programs=programs, source=source, batch=batch,
                                nbatches=nbatches, res=res, nprocs=nprocs, debug=debug,
                                verbose=verbose)

# Compile training dataset features and targets
rocml = build_rocml_model(gfem_models, "DT")

# Train RocML models
rocml.train()

print("RocML trained!")

if visualize:
    visualize_rocml_model(rocml)
    compose_rocml_plots(rocml)

print("RocML visualized!")