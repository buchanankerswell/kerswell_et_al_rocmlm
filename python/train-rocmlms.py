from scripting import parse_arguments, check_arguments
from gfem import get_sampleids, build_gfem_models
from rocmlm import train_rocmlms, evaluate_lut_efficiency
from visualize import (visualize_rocmlm_performance, visualize_rocmlm, compose_rocmlm_plots)

# Parse and check arguments
valid_args = check_arguments(parse_arguments(), "train-rocmlms.py")
locals().update(valid_args)

# Sample sources
sources = {"benchmark": "assets/data/benchmark-samples-pca.csv",
           "middle": "assets/data/synthetic-samples-mixing-middle.csv",
           "random": "assets/data/synthetic-samples-mixing-random.csv"}

# Build GFEM models
gfems = {}
for name, source in sources.items():
    sids = get_sampleids(source, "all")
    gfems[name] = build_gfem_models(source, sids)

# Combine synthetic models for RocMLM training
training_data = {"benchmark": gfems["benchmark"],
                 "synthetic": gfems["middle"] + gfems["random"]}

# Train RocMLMs
rocmlms = {}
for name, models in training_data.items():
    rocmlms[name] = train_rocmlms(models)
    evaluate_lut_efficiency(name, models)

# Visualize RocMLMs
visualize_rocmlm_performance()

for name, models in rocmlms.items():
    for model in models:
        visualize_rocmlm(model)
        compose_rocmlm_plots(model)

print("RocMLMs trained and visualized!")