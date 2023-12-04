from scripting import parse_arguments, check_arguments
from pca import MixingArray, samples_to_csv
from visualize import visualize_pca_loadings, visualize_harker_diagrams

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "create-mixing-array.py")

# Load valid arguments
locals().update(valid_args)

# Create mixing array
mixing_array = MixingArray()
mixing_array.create_mixing_array()
print("Mixing array created!")

# Csv file for synthetic benchmark samples
filename = "assets/data/synthetic-samples-benchmarks.csv"

# Save synthetic benchmark models
sources = ["assets/data/synthetic-samples-mixing-tops.csv",
           "assets/data/synthetic-samples-mixing-middle.csv",
           "assets/data/synthetic-samples-mixing-bottoms.csv"]
sampleids = [["st12000", "st12127"], ["sm12000", "sm12127"], ["sb12000", "sb12127"]]

for source, sids in zip(sources, sampleids):
    samples_to_csv(sids, source, filename)

# Visualize mixing array
visualize_pca_loadings(mixing_array)
visualize_harker_diagrams(mixing_array)
print("Mixing array visualized!")