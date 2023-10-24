from scripting import parse_arguments, check_arguments
from gfem import get_sampleids, build_gfem_models
from visualize import (visualize_gfem_design, visualize_gfem_efficiency, visualize_gfem,
                       visualize_gfem_diff, compose_dataset_plots)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "build-gfem-models.py")

# Load valid arguments
locals().update(valid_args)

# Build GFEM models
gfem_models = build_gfem_models(source, debug=True)

# Visualize GFEM models
visualize_gfem(gfem_models)
visualize_gfem_diff(gfem_models)
compose_dataset_plots(gfem_models)
visualize_gfem_design()
visualize_gfem_efficiency()

print("GFEM models visualized!")