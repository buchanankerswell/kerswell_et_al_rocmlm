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
P_min = args.Pmin
P_max = args.Pmax
P_res = args.Pres
T_min = args.Tmin
T_max = args.Tmax
T_res = args.Tres
sample_id = args.sampleid
parameters = args.params
y_oxide = args.figox
out_dir = args.outdir
fig_dir = args.figdir

# Run name
run_name = sample_id + f"-{T_res}x{P_res}"

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(f"Plotting sample {run_name} the following parameters:")
print("Physical properties:")
for param in parameters:
    print(f"    {param}")
print("Oxides (for Harker diagrams):")
for oxide in y_oxide:
    print(f"    {oxide}")
print(f"out_dir: {out_dir}")
print(f"fig_dir: {fig_dir}")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Plot MAGEMin output
if len(os.listdir(out_dir + "/" + run_name)) != 0:
    print(f"Plotting results for {run_name} ...")

    results = process_MAGEMin_files(run_name, out_dir)

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

        # Use discrete colorscale
        if parameter in ["StableSolutions", "StableVariance"]:
            color_discrete = True
        else:
            color_discrete = False

        # Reverse color scale
        if parameter in ["StableVariance"]:
            color_reverse = True
        else:
            color_reverse = False

        # Plot PT grids
        plot_pseudosection(
            P, T, grid, parameter,
            #title=run_name.replace("_", " ") + ": " + parameter,
            palette="grey",
            color_discrete=color_discrete,
            color_reverse=color_reverse,
            filename=f"{parameter}.png",
            fig_dir=fig_dir
        )

# Plot Earthchem data
plot_harker_diagram(
    datafile="assets/data/earthchem-samples.csv",
    x_oxide="SiO2",
    y_oxide=y_oxide,
    fig_dir="figs"
)