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

# GFEM analysis csv
csv_file = "assets/data/gfem-analysis.csv"

# Build benchmark GFEM models
#source = "assets/data/benchmark-samples-pca.csv"
#gfem_benchmark = []
#gfem_benchmark.extend(build_gfem_models(source, programs=programs, res=res))
#for model in [model for model in gfem_benchmark if model.dataset == "train"]:
#    analyze_gfem_model(model, csv_file)

# Build GFEM models sampled along synthetic mixing arrays
#source = "assets/data/synthetic-samples-mixing-tops.csv"
#sids = get_sampleids(source, "all")
#gfem_top = []
#gfem_top.extend(build_gfem_models(source, sids, programs=programs, res=res))
#for model in [model for model in gfem_top if model.dataset == "train"]:
#    analyze_gfem_model(model, csv_file)

#source = "assets/data/synthetic-samples-mixing-bottoms.csv"
#sids = get_sampleids(source, "all")[::8]
#gfem_bottom = []
#gfem_bottom.extend(build_gfem_models(source, sids, programs=programs, res=res))
#for model in [model for model in gfem_bottom if model.dataset == "train"]:
#    analyze_gfem_model(model, csv_file)

source = f"assets/data/synthetic-samples-mixing-random.csv"
sids = get_sampleids(source, "all")[::8]
gfem_random = []
gfem_random.extend(build_gfem_models(source, sids, programs=programs, res=res))
for model in [model for model in gfem_random if model.dataset == "train"]:
    analyze_gfem_model(model, csv_file)

# Visualize GFEM models
#gfems = [gfem_benchmark, gfem_top, gfem_bottom, gfem_random]
#for models in gfems:
#    visualize_gfem(models)
#    visualize_gfem_diff(models)
#    compose_dataset_plots(models)
#    create_dataset_movies(models)

#visualize_gfem_analysis()
#visualize_gfem_design()
#visualize_gfem_efficiency()

print("GFEM models visualized!")