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

# Build benchmark GFEM models
source = "assets/data/benchmark-samples-pca.csv"
gfem_benchmark = []
gfem_benchmark.extend(build_gfem_models(source, programs=programs, res=res))

# Build GFEM models sampled along synthetic mixing arrays
source = "assets/data/synthetic-samples-mixing-tops.csv"
sids = get_sampleids(source, "all")
gfem_top = []
gfem_top.extend(build_gfem_models(source, sids, programs=programs, res=res))

source = "assets/data/synthetic-samples-mixing-bottoms.csv"
sids = get_sampleids(source, "all")
gfem_bottom = []
gfem_bottom.extend(build_gfem_models(source, sids, programs=programs, res=res))

source = f"assets/data/synthetic-samples-mixing-random.csv"
sids = get_sampleids(source, "all")
gfem_random = []
gfem_random.extend(build_gfem_models(source, sids, programs=programs, res=res))

# Train benchmark RocML models
ml_models_benchmark = ["DT", "KN", "RF", "NN1", "NN2", "NN3"]
rocml_benchmark = train_rocml_models(gfem_benchmark, ml_models_benchmark)

# Train synthetic RocML models
ml_models_synthetic = ["DT", "KN"]
rocml_top = train_rocml_models(gfem_top, ml_models_synthetic)
rocml_bottom = train_rocml_models(gfem_bottom, ml_models_synthetic)
rocml_random = train_rocml_models(gfem_random, ml_models_synthetic)

# Visualize rocml performance
visualize_rocml_performance(["rho", "Vp", "Vs", "melt"], res, "figs/rocml", "rocml")

# Visualize RocML models
for model in rocml_benchmark + rocml_top + rocml_bottom + rocml_random:
    visualize_rocml_model(model)
    compose_rocml_plots(model)

print("RocML visualized!")