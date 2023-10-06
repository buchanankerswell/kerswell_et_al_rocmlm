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

# Get samples
if benchmarks:
    source = "assets/data/benchmark-samples.csv"
    sampleids = ["PUM", "DMM", "NMORB", "RE46"]
elif not benchmarks:
    source = f"assets/data/synthetic-samples-pca{npca}-clusters13.csv"
    sampleids = get_sampleids(source, nsamples, seed)

# Build GFEM models
models = []
for p in ["magemin", "perplex"]:
    models.extend(build_gfem_models(p, Pmin, Pmax, Tmin, Tmax, res, source, sampleids, normox,
                                    targets, maskgeotherm, parallel, nprocs, verbose))

# Visualize GFEM models
for m in models:
    if not m.model_build_error:
        visualize_training_dataset(m, palette)

# Parse GFEM models
mage_models = [m for m in models if m.program == "magemin"]
perp_models = [m for m in models if m.program == "perplex"]

# Visualize GFEM model differences
for m, p in zip(mage_models, perp_models):
    visualize_training_dataset_diff(m, p, palette)

# Compose plots
for s in sampleids:
    for d in ["train", "valid"]:
        magemin = True if mage_models else False
        perplex = True if perp_models else False

        compose_dataset_plots(magemin, perplex, s, d, res, targets,
                              f"{figdir}/{s}_{res}", verbose)

# Visualize RocML training dataset design
visualize_training_dataset_design(Pmin, Pmax, Tmin, Tmax, f"{figdir}/other")

# Visualize GFEM efficiency
visualize_gfem_efficiency(f"{figdir}/other", "gfem-efficiency.png")

print("build-gfem-models.py done!")