import os
from magemin import (
    run_svr,
    parse_arguments,
    check_arguments,
    print_filepaths,
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

# SVR kernels and scalers
#kernels = ["rbf", "linear"]
#scalers = ["standard", "minmax"]
kernels = ["rbf", "linear"]
scalers = ["standard", "minmax"]

if (mgm_results and ppx_results):
    # Run support vector regression
    run_svr(
        sampleid, params, True, True, kernels, scalers, kfolds, parallel,
        nprocs, seed, colormap, outdir, figdir, datadir
    )

    # Create compositions
    print(f"Plotting SVR composition for: {sampleid}:")

    for p in params:
        for k in kernels:
            for s in scalers:
                if p in ["DensityOfFullAssemblage", "Vp", "Vs"]:
                    # First row
                    combine_plots_horizontally(
                        f"{figdir}/MAGEMin-{sampleid}-{p}-{k}-{s}-prem.png",
                        f"{figdir}/Perple_X-{sampleid}-{p}-{k}-{s}-prem.png",
                        f"{figdir}/prem-{sampleid}-{p}-{k}-{s}.png",
                        caption1="a)",
                        caption2="b)"
                    )

                    os.remove(f"{figdir}/MAGEMin-{sampleid}-{p}-{k}-{s}-prem.png")
                    os.remove(f"{figdir}/Perple_X-{sampleid}-{p}-{k}-{s}-prem.png")
                    os.remove(f"{figdir}/MAGEMin-{sampleid}-{p}-{k}-{s}.png")
                    os.remove(f"{figdir}/Perple_X-{sampleid}-{p}-{k}-{s}.png")
                else:
                    # First row
                    combine_plots_horizontally(
                        f"{figdir}/MAGEMin-{sampleid}-{p}-{k}-{s}.png",
                        f"{figdir}/Perple_X-{sampleid}-{p}-{k}-{s}.png",
                        f"{figdir}/performance-{sampleid}-{p}-{k}-{s}.png",
                        caption1="e)",
                        caption2="f)"
                    )

                    os.remove(f"{figdir}/MAGEMin-{sampleid}-{p}-{k}-{s}.png")
                    os.remove(f"{figdir}/Perple_X-{sampleid}-{p}-{k}-{s}.png")

                # First row
                combine_plots_horizontally(
                    f"{figdir}/MAGEMin-{sampleid}-{p}-{k}-{s}-targets-surf.png",
                    f"{figdir}/Perple_X-{sampleid}-{p}-{k}-{s}-targets-surf.png",
                    f"{figdir}/temp1.png",
                    caption1="a)",
                    caption2="b)"
                )

                os.remove(f"{figdir}/MAGEMin-{sampleid}-{p}-{k}-{s}-targets-surf.png")
                os.remove(f"{figdir}/Perple_X-{sampleid}-{p}-{k}-{s}-targets-surf.png")

                # Second row
                combine_plots_horizontally(
                    f"{figdir}/MAGEMin-{sampleid}-{p}-{k}-{s}-svr-surf.png",
                    f"{figdir}/Perple_X-{sampleid}-{p}-{k}-{s}-svr-surf.png",
                    f"{figdir}/temp2.png",
                    caption1="c)",
                    caption2="d)"
                )

                os.remove(f"{figdir}/MAGEMin-{sampleid}-{p}-{k}-{s}-svr-surf.png")
                os.remove(f"{figdir}/Perple_X-{sampleid}-{p}-{k}-{s}-svr-surf.png")

                # Third row
                combine_plots_horizontally(
                    f"{figdir}/MAGEMin-{sampleid}-{p}-{k}-{s}-diff-surf.png",
                    f"{figdir}/Perple_X-{sampleid}-{p}-{k}-{s}-diff-surf.png",
                    f"{figdir}/temp4.png",
                    caption1="e)",
                    caption2="f)"
                )

                os.remove(f"{figdir}/MAGEMin-{sampleid}-{p}-{k}-{s}-diff-surf.png")
                os.remove(f"{figdir}/Perple_X-{sampleid}-{p}-{k}-{s}-diff-surf.png")

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
                    f"{figdir}/surf-{sampleid}-{p}-{k}-{s}.png",
                    caption1="",
                    caption2=""
                )

                os.remove(f"{figdir}/temp3.png")
                os.remove(f"{figdir}/temp4.png")

                # First row
                combine_plots_horizontally(
                    f"{figdir}/MAGEMin-{sampleid}-{p}-{k}-{s}-targets.png",
                    f"{figdir}/Perple_X-{sampleid}-{p}-{k}-{s}-targets.png",
                    f"{figdir}/temp1.png",
                    caption1="a)",
                    caption2="b)"
                )

                os.remove(f"{figdir}/MAGEMin-{sampleid}-{p}-{k}-{s}-targets.png")
                os.remove(f"{figdir}/Perple_X-{sampleid}-{p}-{k}-{s}-targets.png")

                # Second row
                combine_plots_horizontally(
                    f"{figdir}/MAGEMin-{sampleid}-{p}-{k}-{s}-svr.png",
                    f"{figdir}/Perple_X-{sampleid}-{p}-{k}-{s}-svr.png",
                    f"{figdir}/temp2.png",
                    caption1="c)",
                    caption2="d)"
                )

                os.remove(f"{figdir}/MAGEMin-{sampleid}-{p}-{k}-{s}-svr.png")
                os.remove(f"{figdir}/Perple_X-{sampleid}-{p}-{k}-{s}-svr.png")

                # Third row
                combine_plots_horizontally(
                    f"{figdir}/MAGEMin-{sampleid}-{p}-{k}-{s}-diff.png",
                    f"{figdir}/Perple_X-{sampleid}-{p}-{k}-{s}-diff.png",
                    f"{figdir}/temp4.png",
                    caption1="e)",
                    caption2="f)"
                )

                os.remove(f"{figdir}/MAGEMin-{sampleid}-{p}-{k}-{s}-diff.png")
                os.remove(f"{figdir}/Perple_X-{sampleid}-{p}-{k}-{s}-diff.png")

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
                    f"{figdir}/image-{sampleid}-{p}-{k}-{s}.png",
                    caption1="",
                    caption2=""
                )

                os.remove(f"{figdir}/temp3.png")
                os.remove(f"{figdir}/temp4.png")

                os.remove(f"{figdir}/MAGEMin-{sampleid}-{p}-{k}-{s}-P.png")
                os.remove(f"{figdir}/MAGEMin-{sampleid}-{p}-{k}-{s}-T.png")
                os.remove(f"{figdir}/Perple_X-{sampleid}-{p}-{k}-{s}-P.png")
                os.remove(f"{figdir}/Perple_X-{sampleid}-{p}-{k}-{s}-T.png")


# Print figure filepaths
print_filepaths(figdir)