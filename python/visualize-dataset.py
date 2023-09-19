import os
import numpy as np
from rocml import (
    visualize_training_dataset,
    parse_arguments,
    check_arguments,
    visualize_training_dataset_diff,
    combine_plots_vertically,
    combine_plots_horizontally
)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "visualize-dataset.py")

# Load valid arguments
locals().update(valid_args)

# Test for results
mgm_results_train = os.path.exists(f"{outdir}/magemin_{sampleid}_train_{res}")
mgm_results_valid = os.path.exists(f"{outdir}/magemin_{sampleid}_valid_{res}")
ppx_results_train = os.path.exists(f"{outdir}/perplex_{sampleid}_train_{res}")
ppx_results_valid = os.path.exists(f"{outdir}/perplex_{sampleid}_valid_{res}")

print(f"Plotting results for {sampleid}:")

# Plot MAGEMin output
if (mgm_results_train and mgm_results_valid):
    visualize_training_dataset("MAGEMin", sampleid, res, "train", params)
    visualize_training_dataset("MAGEMin", sampleid, res, "valid", params)

# Plot Perple_X output
if (ppx_results_train and ppx_results_valid):
    visualize_training_dataset("Perple_X", sampleid, res, "train", params)
    visualize_training_dataset("Perple_X", sampleid, res, "valid", params)

# Plot MAGEMin Perple_X difference
if (mgm_results_train and mgm_results_valid and ppx_results_train and ppx_results_valid):
    visualize_training_dataset_diff(sampleid, res, "train", params)
    visualize_training_dataset_diff(sampleid, res, "valid", params)

    # Plot MAGEMin Perple_X compositions
    for p in params:
        # Create composition for continuous variables
        if p not in ["StableSolutions", "StableVariance"]:
            for ds in ["train", "valid"]:
                # First row
                combine_plots_horizontally(
                    f"{figdir}/MAGEMin-{sampleid}-{ds}-{p}.png",
                    f"{figdir}/Perple_X-{sampleid}-{ds}-{p}.png",
                    f"{figdir}/temp1.png",
                    caption1="a)",
                    caption2="b)"
                )

                os.remove(f"{figdir}/MAGEMin-{sampleid}-{ds}-{p}.png")
                os.remove(f"{figdir}/Perple_X-{sampleid}-{ds}-{p}.png")

                # Second row
                if p in ["DensityOfFullAssemblage", "Vp", "Vs"]:
                    combine_plots_horizontally(
                        f"{figdir}/diff-{sampleid}-{ds}-{p}.png",
                        f"{figdir}/prem-{sampleid}-{ds}-{p}.png",
                        f"{figdir}/temp2.png",
                        caption1="c)",
                        caption2="d)"
                    )

                    os.remove(f"{figdir}/diff-{sampleid}-{ds}-{p}.png")
                    os.remove(f"{figdir}/prem-{sampleid}-{ds}-{p}.png")

                # Stack rows
                combine_plots_vertically(
                    f"{figdir}/temp1.png",
                    f"{figdir}/temp2.png",
                    f"{figdir}/image-{sampleid}-{ds}-{p}.png",
                    caption1="",
                    caption2=""
                )

                os.remove(f"{figdir}/temp1.png")
                os.remove(f"{figdir}/temp2.png")

        # Create composition for discrete variables
        if p in ["StableSolutions", "StableVariance"]:
            # First row
            combine_plots_horizontally(
                f"{figdir}/MAGEMin-{sampleid}-train-{p}.png",
                f"{figdir}/Perple_X-{sampleid}-train-{p}.png",
                f"{figdir}/temp1.png",
                caption1="a)",
                caption2="b)"
            )

            os.remove(f"{figdir}/MAGEMin-{sampleid}-train-{p}.png")
            os.remove(f"{figdir}/Perple_X-{sampleid}-train-{p}.png")

            # Second row
            combine_plots_horizontally(
                f"{figdir}/MAGEMin-{sampleid}-valid-{p}.png",
                f"{figdir}/Perple_X-{sampleid}-valid-{p}.png",
                f"{figdir}/temp2.png",
                caption1="c)",
                caption2="d)"
            )

            os.remove(f"{figdir}/MAGEMin-{sampleid}-valid-{p}.png")
            os.remove(f"{figdir}/Perple_X-{sampleid}-valid-{p}.png")

            # Stack rows
            combine_plots_vertically(
                f"{figdir}/temp1.png",
                f"{figdir}/temp2.png",
                f"{figdir}/image-{sampleid}-{p}.png",
                caption1="",
                caption2=""
            )

            os.remove(f"{figdir}/temp1.png")
            os.remove(f"{figdir}/temp2.png")

print("visualize-dataset.py done!")