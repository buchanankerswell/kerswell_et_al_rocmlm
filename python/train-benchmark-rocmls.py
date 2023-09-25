import os
import glob
from rocml import (
    parse_arguments,
    check_arguments,
    train_rocml,
    combine_plots_vertically,
    combine_plots_horizontally
)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "train-benchmark-rocmls.py")

# Load valid arguments
locals().update(valid_args)

# Test for results
mgm_results_train = os.path.exists(f"runs/magemin_{sampleid}_train_{res}")
mgm_results_valid = os.path.exists(f"runs/magemin_{sampleid}_valid_{res}")
ppx_results_train = os.path.exists(f"runs/perplex_{sampleid}_train_{res}")
ppx_results_valid = os.path.exists(f"runs/perplex_{sampleid}_valid_{res}")

if (mgm_results_train and ppx_results_train and mgm_results_valid and ppx_results_valid):
    for model in models:
        for program in ["magemin", "perplex"]:
            # Train rocml
            train_rocml(program, sampleid, res, targets, maskgeotherm, model, tune, kfolds,
                        parallel, nprocs, seed, colormap, figdir, verbose)

        # Combine plots
        for target in targets:
            target = target.replace('_', '-')

            if target in ["rho", "Vp", "Vs"]:
                # First row
                combine_plots_horizontally(
                    f"{figdir}/magemin-{sampleid}-{model}-{target}-prem.png",
                    f"{figdir}/perplex-{sampleid}-{model}-{target}-prem.png",
                    f"{figdir}/prem-{sampleid}-{model}-{target}.png",
                    caption1="a)",
                    caption2="b)"
                )

            # First row
            combine_plots_horizontally(
                f"{figdir}/magemin-{sampleid}-{model}-{target}-targets-surf.png",
                f"{figdir}/perplex-{sampleid}-{model}-{target}-targets-surf.png",
                f"{figdir}/temp1.png",
                caption1="a)",
                caption2="b)"
            )

            # Second row
            combine_plots_horizontally(
                f"{figdir}/magemin-{sampleid}-{model}-{target}-surf.png",
                f"{figdir}/perplex-{sampleid}-{model}-{target}-surf.png",
                f"{figdir}/temp2.png",
                caption1="c)",
                caption2="d)"
            )

            # Third row
            combine_plots_horizontally(
                f"{figdir}/magemin-{sampleid}-{model}-{target}-diff-surf.png",
                f"{figdir}/perplex-{sampleid}-{model}-{target}-diff-surf.png",
                f"{figdir}/temp4.png",
                caption1="e)",
                caption2="f)"
            )

            # Stack rows
            combine_plots_vertically(
                f"{figdir}/temp1.png",
                f"{figdir}/temp2.png",
                f"{figdir}/temp3.png",
                caption1="",
                caption2=""
            )

            # Stack rows
            combine_plots_vertically(
                f"{figdir}/temp3.png",
                f"{figdir}/temp4.png",
                f"{figdir}/surf-{sampleid}-{model}-{target}.png",
                caption1="",
                caption2=""
            )

            # First row
            combine_plots_horizontally(
                f"{figdir}/magemin-{sampleid}-{model}-{target}-targets.png",
                f"{figdir}/perplex-{sampleid}-{model}-{target}-targets.png",
                f"{figdir}/temp1.png",
                caption1="a)",
                caption2="b)"
            )

            # Second row
            combine_plots_horizontally(
                f"{figdir}/magemin-{sampleid}-{model}-{target}-predictions.png",
                f"{figdir}/perplex-{sampleid}-{model}-{target}-predictions.png",
                f"{figdir}/temp2.png",
                caption1="c)",
                caption2="d)"
            )

            # Third row
            combine_plots_horizontally(
                f"{figdir}/magemin-{sampleid}-{model}-{target}-diff.png",
                f"{figdir}/perplex-{sampleid}-{model}-{target}-diff.png",
                f"{figdir}/temp4.png",
                caption1="e)",
                caption2="f)"
            )

            # Stack rows
            combine_plots_vertically(
                f"{figdir}/temp1.png",
                f"{figdir}/temp2.png",
                f"{figdir}/temp3.png",
                caption1="",
                caption2=""
            )

            # Stack rows
            combine_plots_vertically(
                f"{figdir}/temp3.png",
                f"{figdir}/temp4.png",
                f"{figdir}/image-{sampleid}-{model}-{target}.png",
                caption1="",
                caption2=""
            )

    os.remove(f"{figdir}/temp1.png")
    os.remove(f"{figdir}/temp2.png")
    os.remove(f"{figdir}/temp3.png")
    os.remove(f"{figdir}/temp4.png")

    if len(models) == 6:

        # First row
        combine_plots_horizontally(
            f"{figdir}/magemin-{sampleid}-{target}-{model[0]}-surf.png",
            f"{figdir}/magemin-{sampleid}-{target}-{model[1]}-surf.png",
            f"{figdir}/temp1.png",
            caption1="a)",
            caption2="b)"
        )

        # Second row
        combine_plots_horizontally(
            f"{figdir}/magemin-{sampleid}-{target}-{model[2]}-surf.png",
            f"{figdir}/magemin-{sampleid}-{target}-{model[3]}-surf.png",
            f"{figdir}/temp2.png",
            caption1="c)",
            caption2="d)"
        )

        # Stack rows
        combine_plots_vertically(
            f"{figdir}/temp1.png",
            f"{figdir}/temp2.png",
            f"{figdir}/temp3.png",
            caption1="",
            caption2=""
        )

        # Third row
        combine_plots_horizontally(
            f"{figdir}/magemin-{sampleid}-{target}-{model[4]}-surf.png",
            f"{figdir}/magemin-{sampleid}-{target}-{model[5]}-surf.png",
            f"{figdir}/temp4.png",
            caption1="e)",
            caption2="f)"
        )

        # Stack rows
        combine_plots_vertically(
            f"{figdir}/temp3.png",
            f"{figdir}/temp4.png",
            f"{figdir}/magemin-{sampleid}-{target}-model-comp-surf.png",
            caption1="",
            caption2=""
        )

        # First row
        combine_plots_horizontally(
            f"{figdir}/perplex-{sampleid}-{target}-{model[0]}-surf.png",
            f"{figdir}/perplex-{sampleid}-{target}-{model[1]}-surf.png",
            f"{figdir}/temp1.png",
            caption1="g)",
            caption2="h)"
        )

        # Second row
        combine_plots_horizontally(
            f"{figdir}/perplex-{sampleid}-{target}-{model[2]}-surf.png",
            f"{figdir}/perplex-{sampleid}-{target}-{model[3]}-surf.png",
            f"{figdir}/temp2.png",
            caption1="i)",
            caption2="j)"
        )

        # Stack rows
        combine_plots_vertically(
            f"{figdir}/temp1.png",
            f"{figdir}/temp2.png",
            f"{figdir}/temp3.png",
            caption1="",
            caption2=""
        )

        # Third row
        combine_plots_horizontally(
            f"{figdir}/perplex-{sampleid}-{target}-{model[4]}-surf.png",
            f"{figdir}/perplex-{sampleid}-{target}-{model[5]}-surf.png",
            f"{figdir}/temp4.png",
            caption1="k)",
            caption2="l)"
        )

        # Stack rows
        combine_plots_vertically(
            f"{figdir}/temp3.png",
            f"{figdir}/temp4.png",
            f"{figdir}/perplex-{sampleid}-{target}-model-comp-surf.png",
            caption1="",
            caption2=""
        )

        # Stack rows
        combine_plots_horizontally(
            f"{figdir}/magemin-{sampleid}-{target}-model-comp-surf.png",
            f"{figdir}/perplex-{sampleid}-{target}-model-comp-surf.png",
            f"{figdir}/all-surf-{sampleid}-{target}.png",
            caption1="",
            caption2=""
        )

        # First row
        combine_plots_horizontally(
            f"{figdir}/magemin-{sampleid}-{target}-{model[0]}-predictions.png",
            f"{figdir}/magemin-{sampleid}-{target}-{model[1]}-predictions.png",
            f"{figdir}/temp1.png",
            caption1="a)",
            caption2="b)"
        )

        # Second row
        combine_plots_horizontally(
            f"{figdir}/magemin-{sampleid}-{target}-{model[2]}-predictions.png",
            f"{figdir}/magemin-{sampleid}-{target}-{model[3]}-predictions.png",
            f"{figdir}/temp2.png",
            caption1="c)",
            caption2="d)"
        )

        # Stack rows
        combine_plots_vertically(
            f"{figdir}/temp1.png",
            f"{figdir}/temp2.png",
            f"{figdir}/temp3.png",
            caption1="",
            caption2=""
        )

        # Third row
        combine_plots_horizontally(
            f"{figdir}/magemin-{sampleid}-{target}-{model[4]}-predictions.png",
            f"{figdir}/magemin-{sampleid}-{target}-{model[5]}-predictions.png",
            f"{figdir}/temp4.png",
            caption1="e)",
            caption2="f)"
        )

        # Stack rows
        combine_plots_vertically(
            f"{figdir}/temp3.png",
            f"{figdir}/temp4.png",
            f"{figdir}/magemin-{sampleid}-{target}-model-comp-image.png",
            caption1="",
            caption2=""
        )

        # First row
        combine_plots_horizontally(
            f"{figdir}/perplex-{sampleid}-{target}-{model[0]}-predictions.png",
            f"{figdir}/perplex-{sampleid}-{target}-{model[1]}-predictions.png",
            f"{figdir}/temp1.png",
            caption1="g)",
            caption2="h)"
        )

        # Second row
        combine_plots_horizontally(
            f"{figdir}/perplex-{sampleid}-{target}-{model[2]}-predictions.png",
            f"{figdir}/perplex-{sampleid}-{target}-{model[3]}-predictions.png",
            f"{figdir}/temp2.png",
            caption1="i)",
            caption2="j)"
        )

        # Stack rows
        combine_plots_vertically(
            f"{figdir}/temp1.png",
            f"{figdir}/temp2.png",
            f"{figdir}/temp3.png",
            caption1="",
            caption2=""
        )

        # Third row
        combine_plots_horizontally(
            f"{figdir}/perplex-{sampleid}-{target}-{model[4]}-predictions.png",
            f"{figdir}/perplex-{sampleid}-{target}-{model[5]}-predictions.png",
            f"{figdir}/temp4.png",
            caption1="k)",
            caption2="l)"
        )

        # Stack rows
        combine_plots_vertically(
            f"{figdir}/temp3.png",
            f"{figdir}/temp4.png",
            f"{figdir}/perplex-{sampleid}-{target}-model-comp-image.png",
            caption1="",
            caption2=""
        )

        # Stack rows
        combine_plots_horizontally(
            f"{figdir}/magemin-{sampleid}-{target}-model-comp-image.png",
            f"{figdir}/perplex-{sampleid}-{target}-model-comp-image.png",
            f"{figdir}/all-image-{sampleid}-{target}.png",
            caption1="",
            caption2=""
        )

        # First row
        combine_plots_horizontally(
            f"{figdir}/magemin-{sampleid}-{target}-{model[0]}-prem.png",
            f"{figdir}/magemin-{sampleid}-{target}-{model[1]}-prem.png",
            f"{figdir}/temp1.png",
            caption1="a)",
            caption2="b)"
        )

        # Second row
        combine_plots_horizontally(
            f"{figdir}/magemin-{sampleid}-{target}-{model[2]}-prem.png",
            f"{figdir}/magemin-{sampleid}-{target}-{model[3]}-prem.png",
            f"{figdir}/temp2.png",
            caption1="c)",
            caption2="d)"
        )

        # Stack rows
        combine_plots_vertically(
            f"{figdir}/temp1.png",
            f"{figdir}/temp2.png",
            f"{figdir}/temp3.png",
            caption1="",
            caption2=""
        )

        # Third row
        combine_plots_horizontally(
            f"{figdir}/magemin-{sampleid}-{target}-{model[4]}-prem.png",
            f"{figdir}/magemin-{sampleid}-{target}-{model[5]}-prem.png",
            f"{figdir}/temp4.png",
            caption1="e)",
            caption2="f)"
        )

        # Stack rows
        combine_plots_vertically(
            f"{figdir}/temp3.png",
            f"{figdir}/temp4.png",
            f"{figdir}/magemin-{sampleid}-{target}-model-comp-prem.png",
            caption1="",
            caption2=""
        )

        # First row
        combine_plots_horizontally(
            f"{figdir}/perplex-{sampleid}-{target}-{model[0]}-prem.png",
            f"{figdir}/perplex-{sampleid}-{target}-{model[1]}-prem.png",
            f"{figdir}/temp1.png",
            caption1="g)",
            caption2="h)"
        )

        # Second row
        combine_plots_horizontally(
            f"{figdir}/perplex-{sampleid}-{target}-{model[2]}-prem.png",
            f"{figdir}/perplex-{sampleid}-{target}-{model[3]}-prem.png",
            f"{figdir}/temp2.png",
            caption1="i)",
            caption2="j)"
        )

        # Stack rows
        combine_plots_vertically(
            f"{figdir}/temp1.png",
            f"{figdir}/temp2.png",
            f"{figdir}/temp3.png",
            caption1="",
            caption2=""
        )

        # Third row
        combine_plots_horizontally(
            f"{figdir}/perplex-{sampleid}-{target}-{model[4]}-prem.png",
            f"{figdir}/perplex-{sampleid}-{target}-{model[5]}-prem.png",
            f"{figdir}/temp4.png",
            caption1="k)",
            caption2="l)"
        )

        # Stack rows
        combine_plots_vertically(
            f"{figdir}/temp3.png",
            f"{figdir}/temp4.png",
            f"{figdir}/perplex-{sampleid}-{target}-model-comp-prem.png",
            caption1="",
            caption2=""
        )

        # Stack rows
        combine_plots_horizontally(
            f"{figdir}/magemin-{sampleid}-{target}-model-comp-prem.png",
            f"{figdir}/perplex-{sampleid}-{target}-model-comp-prem.png",
            f"{figdir}/all-prem-{sampleid}-{target}.png",
            caption1="",
            caption2=""
        )

    # Clean up directory
    tmp_files = glob.glob(f"{figdir}/temp*.png")
    mgm_files = glob.glob(f"{figdir}/magemin*.png")
    ppx_files = glob.glob(f"{figdir}/perplex*.png")

    for file in tmp_files + mgm_files + ppx_files:
        os.remove(file)

print("train-benchmark-rocmls.py done!")