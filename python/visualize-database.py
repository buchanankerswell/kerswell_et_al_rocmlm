import os
from magemin import (
    parse_arguments_visualize_db,
    process_MAGEMin_files,
    encode_phases,
    create_PT_grid,
    plot_pseudosection
)

# Parse arguments
args = parse_arguments_visualize_db()

# Directory of MAGEMin output
out_dir = args.out_dir

# Process MAGEMin output files
if out_dir is None:
    runs = os.listdir("runs")
    runs = [run for run in runs if run != '.DS_Store']
else:
    runs = os.listdir(out_dir + "/runs")
    runs = [run for run in runs if run != '.DS_Store']

# Parameters to visualize
parameters = [
    "Point",
    "Status",
    "Gibbs",
    "BrNorm",
    "Vp",
    "Vs",
    "Entropy",
    "StableSolutions",
    "LiquidFraction",
    "DensityOfFullAssemblage",
    "DensityOfLiquid",
    "DensityOfSolid",
    "DensityOfMixture"
]

for run in runs:
    print(f"Plotting results for {run} ...")

    results = process_MAGEMin_files(run, out_dir)

    # Get PT values
    P = results["P"]
    T = results["T"]

    for parameter in parameters:
        # Transform results into 2D numpy arrays
        if parameter == "StableSolutions":
            # Encode unique phase assemblages
            encoded, unique = encode_phases(results[parameter])
            grid = create_PT_grid(P, T, encoded)
        else:
            grid = create_PT_grid(P, T, results[parameter])

        # Plot PT grids
        plot_pseudosection(
            P, T, grid, parameter,
            title=run.replace("_", " ") + ": " + parameter,
            palette="grey",
            filename=f"{run}-{parameter}.png"
        )