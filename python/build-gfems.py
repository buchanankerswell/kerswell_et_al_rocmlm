from scripting import parse_arguments, check_arguments
from gfem import get_sampleids, build_gfem_models
from visualize import (visualize_gfem_pt_range, visualize_gfem, visualize_gfem_diff,
                       visualize_prem_comps, compose_dataset_plots)

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
visualize_prem_comps(gfems["middle"] + gfems["random"])

for name, models in gfems.items():
    visualize_gfem(models)
    visualize_gfem_diff(models)
    compose_dataset_plots(models)

print("GFEM models built and visualized!")