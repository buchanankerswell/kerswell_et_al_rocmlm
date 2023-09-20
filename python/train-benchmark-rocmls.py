import os
import glob
from rocml import (
    parse_arguments,
    check_arguments,
    run_rocml_training,
    combine_plots_vertically,
    combine_plots_horizontally
)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "train-benchmark-rocmls.py")

# Load valid arguments
locals().update(valid_args)

# Test for results
mgm_results_train = os.path.exists(f"{outdir}/magemin_{sampleid}_train_{res}")
mgm_results_valid = os.path.exists(f"{outdir}/magemin_{sampleid}_valid_{res}")
ppx_results_train = os.path.exists(f"{outdir}/perplex_{sampleid}_train_{res}")
ppx_results_valid = os.path.exists(f"{outdir}/perplex_{sampleid}_valid_{res}")

if (mgm_results_train and ppx_results_train and mgm_results_valid and ppx_results_valid):
    for m in models:
        # Run support vector regression
        run_rocml_training(
            sampleid, res, params, True, True, True, m, tune, kfolds,
            parallel, nprocs, seed, colormap, outdir, figdir, datadir
        )

        for p in params:
            if p in ["rho", "Vp", "Vs"]:
                # First row
                combine_plots_horizontally(
                    f"{figdir}/MAGEMin-{sampleid}-{m}-{p}-prem.png",
                    f"{figdir}/Perple_X-{sampleid}-{m}-{p}-prem.png",
                    f"{figdir}/prem-{sampleid}-{m}-{p}.png",
                    caption1="a)",
                    caption2="b)"
                )

            # First row
            combine_plots_horizontally(
                f"{figdir}/MAGEMin-{sampleid}-{m}-{p}-targets-surf.png",
                f"{figdir}/Perple_X-{sampleid}-{m}-{p}-targets-surf.png",
                f"{figdir}/temp1.png",
                caption1="a)",
                caption2="b)"
            )

            # Second row
            combine_plots_horizontally(
                f"{figdir}/MAGEMin-{sampleid}-{m}-{p}-surf.png",
                f"{figdir}/Perple_X-{sampleid}-{m}-{p}-surf.png",
                f"{figdir}/temp2.png",
                caption1="c)",
                caption2="d)"
            )

            # Third row
            combine_plots_horizontally(
                f"{figdir}/MAGEMin-{sampleid}-{m}-{p}-diff-surf.png",
                f"{figdir}/Perple_X-{sampleid}-{m}-{p}-diff-surf.png",
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
                f"{figdir}/surf-{sampleid}-{m}-{p}.png",
                caption1="",
                caption2=""
            )

            # First row
            combine_plots_horizontally(
                f"{figdir}/MAGEMin-{sampleid}-{m}-{p}-targets.png",
                f"{figdir}/Perple_X-{sampleid}-{m}-{p}-targets.png",
                f"{figdir}/temp1.png",
                caption1="a)",
                caption2="b)"
            )

            # Second row
            combine_plots_horizontally(
                f"{figdir}/MAGEMin-{sampleid}-{m}-{p}-predictions.png",
                f"{figdir}/Perple_X-{sampleid}-{m}-{p}-predictions.png",
                f"{figdir}/temp2.png",
                caption1="c)",
                caption2="d)"
            )

            # Third row
            combine_plots_horizontally(
                f"{figdir}/MAGEMin-{sampleid}-{m}-{p}-diff.png",
                f"{figdir}/Perple_X-{sampleid}-{m}-{p}-diff.png",
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
                f"{figdir}/image-{sampleid}-{m}-{p}.png",
                caption1="",
                caption2=""
            )

    os.remove(f"{figdir}/temp1.png")
    os.remove(f"{figdir}/temp2.png")
    os.remove(f"{figdir}/temp3.png")
    os.remove(f"{figdir}/temp4.png")

    if len(m) == 6:

        # First row
        combine_plots_horizontally(
            f"{figdir}/MAGEMin-{sampleid}-{p}-{m[0]}-surf.png",
            f"{figdir}/MAGEMin-{sampleid}-{p}-{m[1]}-surf.png",
            f"{figdir}/temp1.png",
            caption1="a)",
            caption2="b)"
        )

        # Second row
        combine_plots_horizontally(
            f"{figdir}/MAGEMin-{sampleid}-{p}-{m[2]}-surf.png",
            f"{figdir}/MAGEMin-{sampleid}-{p}-{m[3]}-surf.png",
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
            f"{figdir}/MAGEMin-{sampleid}-{p}-{m[4]}-surf.png",
            f"{figdir}/MAGEMin-{sampleid}-{p}-{m[5]}-surf.png",
            f"{figdir}/temp4.png",
            caption1="e)",
            caption2="f)"
        )

        # Stack rows
        combine_plots_vertically(
            f"{figdir}/temp3.png",
            f"{figdir}/temp4.png",
            f"{figdir}/MAGEMin-{sampleid}-{p}-model-comp-surf.png",
            caption1="",
            caption2=""
        )

        # First row
        combine_plots_horizontally(
            f"{figdir}/Perple_X-{sampleid}-{p}-{m[0]}-surf.png",
            f"{figdir}/Perple_X-{sampleid}-{p}-{m[1]}-surf.png",
            f"{figdir}/temp1.png",
            caption1="g)",
            caption2="h)"
        )

        # Second row
        combine_plots_horizontally(
            f"{figdir}/Perple_X-{sampleid}-{p}-{m[2]}-surf.png",
            f"{figdir}/Perple_X-{sampleid}-{p}-{m[3]}-surf.png",
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
            f"{figdir}/Perple_X-{sampleid}-{p}-{m[4]}-surf.png",
            f"{figdir}/Perple_X-{sampleid}-{p}-{m[5]}-surf.png",
            f"{figdir}/temp4.png",
            caption1="k)",
            caption2="l)"
        )

        # Stack rows
        combine_plots_vertically(
            f"{figdir}/temp3.png",
            f"{figdir}/temp4.png",
            f"{figdir}/Perple_X-{sampleid}-{p}-model-comp-surf.png",
            caption1="",
            caption2=""
        )

        # Stack rows
        combine_plots_horizontally(
            f"{figdir}/MAGEMin-{sampleid}-{p}-model-comp-surf.png",
            f"{figdir}/Perple_X-{sampleid}-{p}-model-comp-surf.png",
            f"{figdir}/all-surf-{sampleid}-{p}.png",
            caption1="",
            caption2=""
        )

        # First row
        combine_plots_horizontally(
            f"{figdir}/MAGEMin-{sampleid}-{p}-{m[0]}-predictions.png",
            f"{figdir}/MAGEMin-{sampleid}-{p}-{m[1]}-predictions.png",
            f"{figdir}/temp1.png",
            caption1="a)",
            caption2="b)"
        )

        # Second row
        combine_plots_horizontally(
            f"{figdir}/MAGEMin-{sampleid}-{p}-{m[2]}-predictions.png",
            f"{figdir}/MAGEMin-{sampleid}-{p}-{m[3]}-predictions.png",
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
            f"{figdir}/MAGEMin-{sampleid}-{p}-{m[4]}-predictions.png",
            f"{figdir}/MAGEMin-{sampleid}-{p}-{m[5]}-predictions.png",
            f"{figdir}/temp4.png",
            caption1="e)",
            caption2="f)"
        )

        # Stack rows
        combine_plots_vertically(
            f"{figdir}/temp3.png",
            f"{figdir}/temp4.png",
            f"{figdir}/MAGEMin-{sampleid}-{p}-model-comp-image.png",
            caption1="",
            caption2=""
        )

        # First row
        combine_plots_horizontally(
            f"{figdir}/Perple_X-{sampleid}-{p}-{m[0]}-predictions.png",
            f"{figdir}/Perple_X-{sampleid}-{p}-{m[1]}-predictions.png",
            f"{figdir}/temp1.png",
            caption1="g)",
            caption2="h)"
        )

        # Second row
        combine_plots_horizontally(
            f"{figdir}/Perple_X-{sampleid}-{p}-{m[2]}-predictions.png",
            f"{figdir}/Perple_X-{sampleid}-{p}-{m[3]}-predictions.png",
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
            f"{figdir}/Perple_X-{sampleid}-{p}-{m[4]}-predictions.png",
            f"{figdir}/Perple_X-{sampleid}-{p}-{m[5]}-predictions.png",
            f"{figdir}/temp4.png",
            caption1="k)",
            caption2="l)"
        )

        # Stack rows
        combine_plots_vertically(
            f"{figdir}/temp3.png",
            f"{figdir}/temp4.png",
            f"{figdir}/Perple_X-{sampleid}-{p}-model-comp-image.png",
            caption1="",
            caption2=""
        )

        # Stack rows
        combine_plots_horizontally(
            f"{figdir}/MAGEMin-{sampleid}-{p}-model-comp-image.png",
            f"{figdir}/Perple_X-{sampleid}-{p}-model-comp-image.png",
            f"{figdir}/all-image-{sampleid}-{p}.png",
            caption1="",
            caption2=""
        )

        # First row
        combine_plots_horizontally(
            f"{figdir}/MAGEMin-{sampleid}-{p}-{m[0]}-prem.png",
            f"{figdir}/MAGEMin-{sampleid}-{p}-{m[1]}-prem.png",
            f"{figdir}/temp1.png",
            caption1="a)",
            caption2="b)"
        )

        # Second row
        combine_plots_horizontally(
            f"{figdir}/MAGEMin-{sampleid}-{p}-{m[2]}-prem.png",
            f"{figdir}/MAGEMin-{sampleid}-{p}-{m[3]}-prem.png",
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
            f"{figdir}/MAGEMin-{sampleid}-{p}-{m[4]}-prem.png",
            f"{figdir}/MAGEMin-{sampleid}-{p}-{m[5]}-prem.png",
            f"{figdir}/temp4.png",
            caption1="e)",
            caption2="f)"
        )

        # Stack rows
        combine_plots_vertically(
            f"{figdir}/temp3.png",
            f"{figdir}/temp4.png",
            f"{figdir}/MAGEMin-{sampleid}-{p}-model-comp-prem.png",
            caption1="",
            caption2=""
        )

        # First row
        combine_plots_horizontally(
            f"{figdir}/Perple_X-{sampleid}-{p}-{m[0]}-prem.png",
            f"{figdir}/Perple_X-{sampleid}-{p}-{m[1]}-prem.png",
            f"{figdir}/temp1.png",
            caption1="g)",
            caption2="h)"
        )

        # Second row
        combine_plots_horizontally(
            f"{figdir}/Perple_X-{sampleid}-{p}-{m[2]}-prem.png",
            f"{figdir}/Perple_X-{sampleid}-{p}-{m[3]}-prem.png",
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
            f"{figdir}/Perple_X-{sampleid}-{p}-{m[4]}-prem.png",
            f"{figdir}/Perple_X-{sampleid}-{p}-{m[5]}-prem.png",
            f"{figdir}/temp4.png",
            caption1="k)",
            caption2="l)"
        )

        # Stack rows
        combine_plots_vertically(
            f"{figdir}/temp3.png",
            f"{figdir}/temp4.png",
            f"{figdir}/Perple_X-{sampleid}-{p}-model-comp-prem.png",
            caption1="",
            caption2=""
        )

        # Stack rows
        combine_plots_horizontally(
            f"{figdir}/MAGEMin-{sampleid}-{p}-model-comp-prem.png",
            f"{figdir}/Perple_X-{sampleid}-{p}-model-comp-prem.png",
            f"{figdir}/all-prem-{sampleid}-{p}.png",
            caption1="",
            caption2=""
        )

    # Clean up directory
    tmp_files = glob.glob(f"{figdir}/temp*.png")
    mgm_files = glob.glob(f"{figdir}/MAGEMin*.png")
    ppx_files = glob.glob(f"{figdir}/Perple_X*.png")

    for file in tmp_files + mgm_files + ppx_files:
        os.remove(file)

print("train-benchmark-rocmls.py done!")