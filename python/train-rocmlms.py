import glob
from scripting import parse_arguments, check_arguments
from gfem import get_sampleids, build_gfem_models
from rocmlm import train_rocmlms
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
    if name == "benchmark":
        sids = get_sampleids(source, "all")
    else:
        sids = get_sampleids(source, "all")[::32]
    gfems[name] = build_gfem_models(source, sids, res=64)

# Train RocMLMs
for name, models in gfems.items():
    if name == "benchmark":
        #rocmlms[name] = train_rocmlms(models, ["KN", "DT", "RF", "NN1", "NN2", "NN3"])
        rocmlms[name] = train_rocmlms(models, ["KN", "DT"])
    else:
        rocmlms[name] = train_rocmlms(models, ["DT", "KN"])

# Visualize rocmlm performance
visualize_rocmlm_performance(["rho", "Vp", "Vs"], 128)

# Visualize RocMLMs
for name, models in rocmlms.items():
    for model in models:
        visualize_rocmlm(model)
        compose_rocmlm_plots(model)

print("RocMLMs visualized!")