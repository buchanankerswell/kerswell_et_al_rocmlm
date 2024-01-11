import glob
from scripting import parse_arguments, check_arguments
from gfem import get_sampleids, build_gfem_models
from rocmlm import train_rocmlms, evaluate_lut_efficiency
from visualize import visualize_rocmlm_performance, visualize_rocmlm, compose_rocmlm_plots

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
    if name == "benchmark":
        mlms = ["KN", "DT", "RF", "NN1", "NN2", "NN3"]
    else:
        mlms = ["KN", "DT"]

    # Train RocMLMs
    rocmlms[name] = train_rocmlms(models, mlms)

    # Clock LUT efficiency
    if name == "top":
        evaluate_lut_efficiency(models)

# Visualize rocmlm performance
targets, res  = ["rho", "Vp", "Vs"], 128
visualize_rocmlm_performance(targets, res)

# Visualize RocMLMs
for name, models in rocmlms.items():
    for model in models:
        visualize_rocmlm(model)
        compose_rocmlm_plots(model)

print("RocMLMs visualized!")