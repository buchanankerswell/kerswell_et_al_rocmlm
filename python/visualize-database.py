import os
import numpy as np
from magemin import (
    visualize_GFEM,
    parse_arguments,
    check_arguments,
    print_filepaths,
    visualize_GFEM_diff,
    combine_plots_vertically,
    combine_plots_horizontally
)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "visualize-database.py")

# Load valid arguments
locals().update(valid_args)

# Test for results
mgm_results = len(os.listdir(outdir + "/" + sampleid)) != 0
ppx_results = os.path.exists(f"assets/benchmark/{sampleid}/{sampleid}_grid.tab")

print(f"Plotting results for {sampleid}:")

# Plot MAGEMin output
if mgm_results:
    visualize_GFEM("MAGEMin", sampleid, params, True, colormap, outdir, figdir, datadir)

# Plot Perple_X output
if ppx_results:
    visualize_GFEM("Perple_X", sampleid, params, True, colormap, outdir, figdir, datadir)

# Plot MAGEMin Perple_X difference
if (mgm_results and ppx_results):
    visualize_GFEM_diff(sampleid, params, True, colormap, outdir, figdir, datadir)

    # Plot MAGEMin Perple_X compositions
    for parameter in params:
        # Create composition for continuous variables
        if parameter not in ["StableSolutions", "StableVariance"]:

            # First row
            combine_plots_horizontally(
                f"{figdir}/MAGEMin-{sampleid}-{parameter}.png",
                f"{figdir}/Perple_X-{sampleid}-{parameter}.png",
                f"{figdir}/temp1.png",
                caption1="a)",
                caption2="b)"
            )

            os.remove(f"{figdir}/MAGEMin-{sampleid}-{parameter}.png")
            os.remove(f"{figdir}/Perple_X-{sampleid}-{parameter}.png")

            # Second row
            if parameter in ["DensityOfFullAssemblage", "Vp", "Vs"]:
                combine_plots_horizontally(
                    f"{figdir}/diff-{sampleid}-{parameter}.png",
                    f"{figdir}/prem-{sampleid}-{parameter}.png",
                    f"{figdir}/temp2.png",
                    caption1="c)",
                    caption2="d)"
                )

                os.remove(f"{figdir}/diff-{sampleid}-{parameter}.png")
                os.remove(f"{figdir}/prem-{sampleid}-{parameter}.png")

            # Stack rows
            combine_plots_vertically(
                f"{figdir}/temp1.png",
                f"{figdir}/temp2.png",
                f"{figdir}/image-{sampleid}-{parameter}.png",
                caption1="",
                caption2=""
            )

            os.remove(f"{figdir}/temp1.png")
            os.remove(f"{figdir}/temp2.png")

        # Create composition for discrete variables
        if parameter in ["StableSolutions", "StableVariance"]:
            # First row
            combine_plots_horizontally(
                f"{figdir}/MAGEMin-{sampleid}-{parameter}.png",
                f"{figdir}/Perple_X-{sampleid}-{parameter}.png",
                f"{figdir}/image-{sampleid}-{parameter}.png",
                caption1="a)",
                caption2="b)"
            )

            os.remove(f"{figdir}/MAGEMin-{sampleid}-{parameter}.png")
            os.remove(f"{figdir}/Perple_X-{sampleid}-{parameter}.png")

# Print figure filepaths
print_filepaths(figdir)