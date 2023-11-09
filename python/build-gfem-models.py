from scripting import parse_arguments, check_arguments
from gfem import get_sampleids, build_gfem_models
from visualize import (visualize_gfem_design, visualize_gfem_efficiency, visualize_gfem,
                       visualize_gfem_diff, compose_dataset_plots)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "build-gfem-models.py")

# Load valid arguments
locals().update(valid_args)

# Initialize gfem models
gfem_models = []

# Configure and build GFEM models
if "benchmark" in source:
    programs = ["magemin", "perplex"]
    source = "assets/data/benchmark-samples-normalized.csv"

    gfem_models.extend(build_gfem_models(source, programs))

else:
    programs = ["perplex"]

    # Build synthetic benchmark models
    sources = ["assets/data/synthetic-samples-mixing-tops.csv",
               "assets/data/synthetic-samples-mixing-middle.csv",
               "assets/data/synthetic-samples-mixing-bottoms.csv"]
    sampleids = [["st12000", "st23000", "st23127"],
                 ["sm12000", "sm23000", "sm23127"],
                 ["sb12000", "sb23000", "sb23127"]]

    for source, sids in zip(sources, sampleids):
        gfem_models.extend(build_gfem_models(source, sids, programs))

#    # Random samples along mixing array
#    source = f"assets/data/synthetic-samples-mixing-random.csv"
#    sampleids.append(get_sampleids(source, "all"))
#
#    gfem_models.extend(build_gfem_models(source, programs))

# Visualize GFEM models
#visualize_gfem(gfem_models)
#visualize_gfem_diff(gfem_models)
#compose_dataset_plots(gfem_models)
#visualize_gfem_design()
#visualize_gfem_efficiency()

#print("GFEM models visualized!")