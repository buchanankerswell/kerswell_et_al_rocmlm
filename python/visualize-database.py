import os
from magemin import (
    parse_arguments_visualize_db,
    plot_harker_diagram,
    process_MAGEMin_files,
    encode_phases,
    create_PT_grid,
    plot_pseudosection
)

# Parse arguments
args = parse_arguments_visualize_db()

# Get argument values
parameters = args.params
y_oxide = args.oxides
out_dir = args.outdir
fig_dir = args.figdir

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Plotting with the following parameters:")
print("Physical properties:")
for param in parameters:
    print(f"    {param}")
print("Oxides (for Harker diagrams):")
for oxide in y_oxide:
    print(f"    {oxide}")
print(f"out_dir: {out_dir}")
print(f"fig_dir: {fig_dir}")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Process MAGEMin output files
MAGEMin_output = os.listdir(out_dir)
runs = [run for run in MAGEMin_output if run != ".DS_Store" and not run.endswith(".dat")]

# Plot MAGEMin output
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
            filename=f"{run}-{parameter}.png",
            fig_dir=fig_dir
        )

# Plot Earthchem data
plot_harker_diagram(
    datafile="assets/data/earthchem-samples.csv",
    x_oxide="SiO2",
    y_oxide=y_oxide,
    fig_dir=fig_dir
)