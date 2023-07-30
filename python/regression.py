import os
from magemin import (
    parse_arguments,
    check_arguments,
    print_filepaths,
    run_ml_regression,
    combine_plots_vertically,
    combine_plots_horizontally
)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "svr.py")

# Load valid arguments
locals().update(valid_args)

# Test for results
mgm_results = len(os.listdir(outdir + "/" + sampleid)) != 0
ppx_results = os.path.exists(f"assets/benchmark/{sampleid}/{sampleid}_grid.tab")

if (mgm_results and ppx_results):
    for m in models:
        # Run support vector regression
        run_ml_regression(
            sampleid, params, True, True, m, kfolds, parallel, nprocs, seed, colormap,
            outdir, figdir, datadir
        )

        # Create compositions
        print(f"Plotting results for: {sampleid} {m}:")
        print(f"    sample: {sampleid}")
        print(f"     model: {m}")

        # Change model string for filename
        mlab = m.replace(" ", "-")

        for p in params:
            if p in ["DensityOfFullAssemblage", "Vp", "Vs"]:
                # First row
                combine_plots_horizontally(
                    f"{figdir}/MAGEMin-{sampleid}-{p}-{mlab}-prem.png",
                    f"{figdir}/Perple_X-{sampleid}-{p}-{mlab}-prem.png",
                    f"{figdir}/prem-{sampleid}-{p}-{mlab}.png",
                    caption1="a)",
                    caption2="b)"
                )

#                os.remove(f"{figdir}/MAGEMin-{sampleid}-{p}-{mlab}-prem.png")
#                os.remove(f"{figdir}/Perple_X-{sampleid}-{p}-{mlab}-prem.png")
#                os.remove(f"{figdir}/MAGEMin-{sampleid}-{p}-{mlab}-performance.png")
#                os.remove(f"{figdir}/Perple_X-{sampleid}-{p}-{mlab}-performance.png")

            # First row
            combine_plots_horizontally(
                f"{figdir}/MAGEMin-{sampleid}-{p}-{mlab}-targets-surf.png",
                f"{figdir}/Perple_X-{sampleid}-{p}-{mlab}-targets-surf.png",
                f"{figdir}/temp1.png",
                caption1="a)",
                caption2="b)"
            )

#            os.remove(f"{figdir}/MAGEMin-{sampleid}-{p}-{mlab}-targets-surf.png")
#            os.remove(f"{figdir}/Perple_X-{sampleid}-{p}-{mlab}-targets-surf.png")

            # Second row
            combine_plots_horizontally(
                f"{figdir}/MAGEMin-{sampleid}-{p}-{mlab}-surf.png",
                f"{figdir}/Perple_X-{sampleid}-{p}-{mlab}-surf.png",
                f"{figdir}/temp2.png",
                caption1="c)",
                caption2="d)"
            )

#            os.remove(f"{figdir}/MAGEMin-{sampleid}-{p}-{mlab}-surf.png")
#            os.remove(f"{figdir}/Perple_X-{sampleid}-{p}-{mlab}-surf.png")

            # Third row
            combine_plots_horizontally(
                f"{figdir}/MAGEMin-{sampleid}-{p}-{mlab}-diff-surf.png",
                f"{figdir}/Perple_X-{sampleid}-{p}-{mlab}-diff-surf.png",
                f"{figdir}/temp4.png",
                caption1="e)",
                caption2="f)"
            )

#            os.remove(f"{figdir}/MAGEMin-{sampleid}-{p}-{mlab}-diff-surf.png")
#            os.remove(f"{figdir}/Perple_X-{sampleid}-{p}-{mlab}-diff-surf.png")

            # Stack rows
            combine_plots_vertically(
                f"{figdir}/temp1.png",
                f"{figdir}/temp2.png",
                f"{figdir}/temp3.png",
                caption1="",
                caption2=""
            )

            os.remove(f"{figdir}/temp1.png")
            os.remove(f"{figdir}/temp2.png")

            # Stack rows
            combine_plots_vertically(
                f"{figdir}/temp3.png",
                f"{figdir}/temp4.png",
                f"{figdir}/surf-{sampleid}-{p}-{mlab}.png",
                caption1="",
                caption2=""
            )

            os.remove(f"{figdir}/temp3.png")
            os.remove(f"{figdir}/temp4.png")

            # First row
            combine_plots_horizontally(
                f"{figdir}/MAGEMin-{sampleid}-{p}-{mlab}-targets.png",
                f"{figdir}/Perple_X-{sampleid}-{p}-{mlab}-targets.png",
                f"{figdir}/temp1.png",
                caption1="a)",
                caption2="b)"
            )

#            os.remove(f"{figdir}/MAGEMin-{sampleid}-{p}-{mlab}-targets.png")
#            os.remove(f"{figdir}/Perple_X-{sampleid}-{p}-{mlab}-targets.png")

            # Second row
            combine_plots_horizontally(
                f"{figdir}/MAGEMin-{sampleid}-{p}-{mlab}-predictions.png",
                f"{figdir}/Perple_X-{sampleid}-{p}-{mlab}-predictions.png",
                f"{figdir}/temp2.png",
                caption1="c)",
                caption2="d)"
            )

#            os.remove(f"{figdir}/MAGEMin-{sampleid}-{p}-{mlab}-predictions.png")
#            os.remove(f"{figdir}/Perple_X-{sampleid}-{p}-{mlab}-predictions.png")

            # Third row
            combine_plots_horizontally(
                f"{figdir}/MAGEMin-{sampleid}-{p}-{mlab}-diff.png",
                f"{figdir}/Perple_X-{sampleid}-{p}-{mlab}-diff.png",
                f"{figdir}/temp4.png",
                caption1="e)",
                caption2="f)"
            )

#            os.remove(f"{figdir}/MAGEMin-{sampleid}-{p}-{mlab}-diff.png")
#            os.remove(f"{figdir}/Perple_X-{sampleid}-{p}-{mlab}-diff.png")

            # Stack rows
            combine_plots_vertically(
                f"{figdir}/temp1.png",
                f"{figdir}/temp2.png",
                f"{figdir}/temp3.png",
                caption1="",
                caption2=""
            )

            os.remove(f"{figdir}/temp1.png")
            os.remove(f"{figdir}/temp2.png")

            # Stack rows
            combine_plots_vertically(
                f"{figdir}/temp3.png",
                f"{figdir}/temp4.png",
                f"{figdir}/image-{sampleid}-{p}-{mlab}.png",
                caption1="",
                caption2=""
            )

            os.remove(f"{figdir}/temp3.png")
            os.remove(f"{figdir}/temp4.png")

# Print figure filepaths
print_filepaths(figdir)