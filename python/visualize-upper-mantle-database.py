from magemin import process_MAGEMin_files, encode_phases, create_PT_grid, plot_pseudosection

# Process MAGEMin output files
run_name = "test"
results = process_MAGEMin_files(run_name=run_name)

# Plot
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
        title=run_name.replace("_", " "),
        filename=f"{run_name}-{parameter}.png"
    )