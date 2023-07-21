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
kernels = ["rbf"]
scalers = ["standard", "minmax"]

if mgm_results:
    run_svr("MAGEMin", sampleid, params, kernels, scalers, colormap, outdir, figdir, datadir)

if ppx_results:
    run_svr("Perple_X", sampleid, params, kernels, scalers, colormap, outdir, figdir, datadir)

if (mgm_results and ppx_results):
    # Create compositions
    print(f"Plotting SVR results for: {sampleid}:")

    for parameter in params:
        # First row surface
        combine_plots_horizontally(
            f"{figdir}/MAGEMin-{sampleid}-{parameter}-histogram.png",
            f"{figdir}/Perple_X-{sampleid}-{parameter}-histogram.png",
            f"{figdir}/comp-histogram-{sampleid}-{parameter}.png",
            caption1="a)",
            caption2="b)"
        )

        # First row scatter
        combine_plots_horizontally(
            f"{figdir}/MAGEMin-{sampleid}-{parameter}-scatter-P.png",
            f"{figdir}/Perple_X-{sampleid}-{parameter}-scatter-P.png",
            f"{figdir}/temp1.png",
            caption1="a)",
            caption2="c)"
        )
        # Second row scatter
        combine_plots_horizontally(
            f"{figdir}/MAGEMin-{sampleid}-{parameter}-scatter-T.png",
            f"{figdir}/Perple_X-{sampleid}-{parameter}-scatter-T.png",
            f"{figdir}/temp2.png",
            caption1="b)",
            caption2="d)"
        )
        # Stack rows
        combine_plots_vertically(
            f"{figdir}/temp1.png",
            f"{figdir}/temp2.png",
            f"{figdir}/comp-scatter-{sampleid}-{parameter}.png",
            caption1="",
            caption2=""
        )

        # Cleanup dir
        if os.path.exists(f"{figdir}/temp1.png"):
            os.remove(f"{figdir}/temp1.png")
        if os.path.exists(f"{figdir}/temp2.png"):
            os.remove(f"{figdir}/temp2.png")

        # First row SVR MGM
        combine_plots_horizontally(
            f"{figdir}/MAGEMin-{sampleid}-{parameter}-{kernels[0]}-{scalers[0]}.png",
            f"{figdir}/Perple_X-{sampleid}-{parameter}-{kernels[0]}-{scalers[0]}.png",
            f"{figdir}/temp1.png",
            caption1="a)",
            caption2="c)"
        )
        # Second row SVR PPX
        combine_plots_horizontally(
            f"{figdir}/MAGEMin-{sampleid}-{parameter}-{kernels[0]}-{scalers[1]}.png",
            f"{figdir}/Perple_X-{sampleid}-{parameter}-{kernels[0]}-{scalers[1]}.png",
            f"{figdir}/temp2.png",
            caption1="b)",
            caption2="d)"
        )
        # Stack rows
        combine_plots_vertically(
            f"{figdir}/temp1.png",
            f"{figdir}/temp2.png",
            f"{figdir}/comp-svr-{sampleid}-{parameter}.png",
            caption1="",
            caption2=""
        )

        # Cleanup dir
        if os.path.exists(f"{figdir}/temp1.png"):
            os.remove(f"{figdir}/temp1.png")
        if os.path.exists(f"{figdir}/temp2.png"):
            os.remove(f"{figdir}/temp2.png")

# Print figure filepaths
print_filepaths(figdir)