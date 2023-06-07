from magemin import process_MAGEMin_files, plot_pseudosection

# Process MAGEMin output files
results = process_MAGEMin_files(run_name="test")

# Plot
parameters = [
    "Point",
    "Status",
    "P",
    "T",
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

for parameter in parameters:
    plot_pseudosection(results, parameter, filename=f"{parameter}.png")