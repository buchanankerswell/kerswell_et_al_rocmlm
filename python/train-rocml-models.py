import glob
from scripting import parse_arguments, check_arguments
from gfem import get_sampleids, build_gfem_models
from rocml import train_rocml_models
from visualize import visualize_rocml_performance, visualize_rocml_model, compose_rocml_plots

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "train-rocml-models.py")

# Load valid arguments
locals().update(valid_args)

# Initialize models
gfems, rocmls = {}, {}

# Sample sources
sources = {"benchmark": "assets/data/benchmark-samples-pca.csv",
           "top": "assets/data/synthetic-samples-mixing-tops.csv",
           "bottom": "assets/data/synthetic-samples-mixing-bottoms.csv",
           "random": "assets/data/synthetic-samples-mixing-random.csv"}

# Build GFEM models
for name, source in sources.items():
    sids = get_sampleids(source, "all")
    gfems[name] = build_gfem_models(source, sids)

# Train RocML models
for name, models in gfems.items():
    if name == "benchmark":
        rocmls[name] = train_rocml_models(models, ["KN", "DT"])
#        rocmls[name] = train_rocml_models(models, ["KN", "DT", "RF", "NN1", "NN2", "NN3"])
    else:
        rocmls[name] = train_rocml_models(models, ["DT", "KN"])

# Visualize rocml performance
visualize_rocml_performance(["rho", "Vp", "Vs"], 128)

# Visualize RocML models
for name, models in rocmls.items():
    for model in models:
        visualize_rocml_model(model)
        compose_rocml_plots(model)

print("RocML visualized!")