from scripting import parse_arguments, check_arguments
from gfem import get_sampleids, build_gfem_models
from visualize import (visualize_training_dataset_design,
                       visualize_gfem_efficiency,
                       visualize_training_dataset,
                       visualize_training_dataset_diff,
                       compose_dataset_plots)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "build-gfem-models.py")

# Load valid arguments
locals().update(valid_args)

# Build GFEM models
models = build_gfem_models(programs=["perplex"], source=source, batch=batch, res=res, debug=debug)

if visualize:
    # Visualize GFEM models
    for model in models:
        visualize_training_dataset(model)

    # Parse GFEM models
    mage_models = [model for model in models if model.program == "magemin"]
    perp_models = [model for model in models if model.program == "perplex"]

    # Visualize GFEM model differences
    for magemin, perplex in zip(mage_models, perp_models):
        visualize_training_dataset_diff(magemin, perplex)

    # Compose plots
    compose_dataset_plots(mage_models, perp_models)

    # Visualize RocML training dataset design
    visualize_training_dataset_design()

    # Visualize GFEM efficiency
    visualize_gfem_efficiency()

    print("GFEM models visualized!")