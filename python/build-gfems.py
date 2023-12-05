from scripting import parse_arguments, check_arguments
from gfem import get_sampleids, build_gfem_models, analyze_gfem_model
from visualize import (visualize_gfem_design, visualize_gfem_efficiency, visualize_gfem,
                       visualize_gfem_diff, visualize_gfem_analysis, compose_dataset_plots,
                       create_dataset_movies)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "build-gfem-models.py")

# Load valid arguments
locals().update(valid_args)

# Initialize GFEM models
gfems = {}

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

# Visualize GFEM models
for name, models in gfems.items():
    visualize_gfem(models)
    visualize_gfem_diff(models)
    compose_dataset_plots(models)
    create_dataset_movies(models)

visualize_gfem_analysis()
visualize_gfem_design()
visualize_gfem_efficiency()

print("GFEM models visualized!")