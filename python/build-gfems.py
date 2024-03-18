from scripting import parse_arguments, check_arguments
from gfem import get_sampleids, build_gfem_models
from visualize import (visualize_gfem_pt_range, visualize_gfem, visualize_gfem_diff,
                       visualize_gfem_accuracy_vs_prem, compose_dataset_plots,
                       compose_prem_plots, create_dataset_movies)

# Parse and check arguments
valid_args = check_arguments(parse_arguments(), "build-gfems.py")
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

# Visualize GFEM models
visualize_gfem_pt_range(gfems["benchmark"][0])
isualize_gfem_accuracy_vs_prem()

for name, models in gfems.items():
    visualize_gfem(models)
    visualize_gfem_diff(models)

compose_prem_plots([gfems["middle"][0], gfems["middle"][128], gfems["benchmark"][2]])

for name, models in gfems.items():
    compose_dataset_plots(models)
    create_dataset_movies(models)

print("GFEM models built and visualized!")