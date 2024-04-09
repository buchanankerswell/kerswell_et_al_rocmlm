from scripting import parse_arguments, check_arguments
from pca import MixingArray, samples_to_csv
from visualize import visualize_mixing_array, visualize_harker_diagrams

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
sampleids = [["st000", "st128"], ["sm000", "sm128"], ["sb000", "sb128"]]

for source, sids in zip(sources, sampleids):
    samples_to_csv(sids, source, filename)

# Visualize mixing array
visualize_harker_diagrams(mixing_array)
visualize_mixing_array(mixing_array)

print("Mixing array visualized!")