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
mgm_results_train = len(os.listdir(f"{outdir}/{sampleid}/magemin_train_{res}")) != 0
mgm_results_valid = len(os.listdir(f"{outdir}/{sampleid}/magemin_valid_{res}")) != 0
ppx_results_train = len(os.listdir(f"{outdir}/{sampleid}/perplex_train_{res}")) != 0
ppx_results_valid = len(os.listdir(f"{outdir}/{sampleid}/perplex_valid_{res}")) != 0

print(f"Plotting results for {sampleid}:")

# Plot MAGEMin output
if (mgm_results_train and mgm_results_valid):
    visualize_GFEM(
        "MAGEMin", sampleid, res, "train", params, True, colormap, outdir, figdir, datadir
    )
    visualize_GFEM(
        "MAGEMin", sampleid, res, "valid", params, True, colormap, outdir, figdir, datadir
    )

# Plot Perple_X output
if (ppx_results_train and ppx_results_valid):
    visualize_GFEM(
        "Perple_X", sampleid, res, "train", params, True, colormap, outdir, figdir, datadir
    )
    visualize_GFEM(
        "Perple_X", sampleid, res, "valid", params, True, colormap, outdir, figdir, datadir
    )

# Plot MAGEMin Perple_X difference
if (mgm_results_train and mgm_results_valid and ppx_results_train and ppx_results_valid):
    visualize_GFEM_diff(
        sampleid, res, "train", params, True, colormap, outdir, figdir, datadir
    )
    visualize_GFEM_diff(
        sampleid, res, "valid", params, True, colormap, outdir, figdir, datadir
    )

    # Plot MAGEMin Perple_X compositions
    for parameter in params:
        # Create composition for continuous variables
        if parameter not in ["StableSolutions", "StableVariance"]:
            for ds in ["train", "valid"]:
                # First row
                combine_plots_horizontally(
                    f"{figdir}/MAGEMin-{sampleid}-{ds}-{parameter}.png",
                    f"{figdir}/Perple_X-{sampleid}-{ds}-{parameter}.png",
                    f"{figdir}/temp1.png",
                    caption1="a)",
                    caption2="b)"
                )

                os.remove(f"{figdir}/MAGEMin-{sampleid}-{ds}-{parameter}.png")
                os.remove(f"{figdir}/Perple_X-{sampleid}-{ds}-{parameter}.png")

                # Second row
                if parameter in ["DensityOfFullAssemblage", "Vp", "Vs"]:
                    combine_plots_horizontally(
                        f"{figdir}/diff-{sampleid}-{ds}-{parameter}.png",
                        f"{figdir}/prem-{sampleid}-{ds}-{parameter}.png",
                        f"{figdir}/temp2.png",
                        caption1="c)",
                        caption2="d)"
                    )

                    os.remove(f"{figdir}/diff-{sampleid}-{ds}-{parameter}.png")
                    os.remove(f"{figdir}/prem-{sampleid}-{ds}-{parameter}.png")

                # Stack rows
                combine_plots_vertically(
                    f"{figdir}/temp1.png",
                    f"{figdir}/temp2.png",
                    f"{figdir}/image-{sampleid}-{ds}-{parameter}.png",
                    caption1="",
                    caption2=""
                )

                os.remove(f"{figdir}/temp1.png")
                os.remove(f"{figdir}/temp2.png")

        # Create composition for discrete variables
        if parameter in ["StableSolutions", "StableVariance"]:
            # First row
            combine_plots_horizontally(
                f"{figdir}/MAGEMin-{sampleid}-train-{parameter}.png",
                f"{figdir}/Perple_X-{sampleid}-train-{parameter}.png",
                f"{figdir}/temp1.png",
                caption1="a)",
                caption2="b)"
            )

            os.remove(f"{figdir}/MAGEMin-{sampleid}-train-{parameter}.png")
            os.remove(f"{figdir}/Perple_X-{sampleid}-train-{parameter}.png")

            # Second row
            combine_plots_horizontally(
                f"{figdir}/MAGEMin-{sampleid}-valid-{parameter}.png",
                f"{figdir}/Perple_X-{sampleid}-valid-{parameter}.png",
                f"{figdir}/temp2.png",
                caption1="c)",
                caption2="d)"
            )

            os.remove(f"{figdir}/MAGEMin-{sampleid}-valid-{parameter}.png")
            os.remove(f"{figdir}/Perple_X-{sampleid}-valid-{parameter}.png")

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

# Print figure filepaths
print_filepaths(figdir)