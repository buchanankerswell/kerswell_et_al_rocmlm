from scripting import parse_arguments, check_arguments
from gfem import get_sampleids, build_gfem_models
from rocmlm import train_rocmlms, evaluate_lut_efficiency
from visualize import (visualize_rocmlm_tradeoffs, visualize_rocmlm, compose_rocmlm_plots)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "train-rocmlms.py")

# Load valid arguments
locals().update(valid_args)

# Initialize models
gfems, rocmlms = {}, {}

# Sample sources
sources = {"benchmark": "assets/data/benchmark-samples-pca.csv",
           "top": "assets/data/synthetic-samples-mixing-tops.csv",
           "bottom": "assets/data/synthetic-samples-mixing-bottoms.csv",
           "random": "assets/data/synthetic-samples-mixing-random.csv"}

# Build GFEM models
for name, source in sources.items():
    sids = get_sampleids(source, "all")
    gfems[name] = build_gfem_models(source, sids)

# Train RocMLMs
for name, models in gfems.items():
    rocmlms[name] = train_rocmlms(models, ["KN", "DT", "NN1", "NN2", "NN3"])

    # Clock LUT efficiency
    evaluate_lut_efficiency(name, models)

# Visualize RocMLM performance tradeoffs
visualize_rocmlm_tradeoffs()

# Visualize RocMLMs
for name, models in rocmlms.items():
    if name == "benchmark":
        for model in models:
            visualize_rocmlm(model)
            compose_rocmlm_plots(model)
    else:
        filtered_models = [model for model in models if "S128-W128" in model.model_prefix]

        for model in filtered_models:
            visualize_rocmlm(model, skip=32)
            compose_rocmlm_plots(model, skip=32)

print("RocMLMs visualized!")