from scripting import parse_arguments, check_arguments
from pca import MixingArray
from visualize import visualize_pca_loadings, visualize_kmeans_clusters, visualize_mixing_array

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "create-mixing-array.py")

# Load valid arguments
locals().update(valid_args)

# Create mixing array
mixing_array = MixingArray(res)
mixing_array.create_mixing_array()
print("Mixing array created!")

# Visualize mixing array
visualize_pca_loadings(mixing_array)
visualize_kmeans_clusters(mixing_array)
visualize_mixing_array(mixing_array)
print("Mixing array visualized!")