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
models = build_gfem_models(programs=programs, source=source, batch=batch, nbatches=nbatches,
                           res=res, nprocs=nprocs, debug=debug, verbose=verbose)

if visualize:
    # Visualize GFEM models (training datasets only)
    for model in [m if m.dataset == "train" else None for m in models]:
        visualize_training_dataset(model, verbose)

    # Parse GFEM models (training datasets only)
    mage_models = [m if m.program == "magemin" and m.dataset == "train" else
                   None for m in models]
    perp_models = [m if m.program == "perplex" and m.dataset == "train" else
                   None for m in models]

    # Visualize GFEM model differences (training datasets only)
    for magemin, perplex in zip(mage_models, perp_models):
        if magemin is not None and perplex is not None:
            visualize_training_dataset_diff(magemin, perplex, verbose)

    # Compose plots (training datasets only)
    compose_dataset_plots(mage_models, perp_models)

    # Visualize RocML training dataset design
    visualize_training_dataset_design()

    # Visualize GFEM efficiency
    visualize_gfem_efficiency()

    print("GFEM models visualized!")