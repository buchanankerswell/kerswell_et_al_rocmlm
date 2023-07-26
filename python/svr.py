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
kernels = ["linear", "rbf"]
scalers = ["standard", "minmax"]

if (mgm_results and ppx_results):
    # Run support vector regression
    run_svr(
        sampleid, params, True, True, kernels, scalers, kfolds, parallel,
        nprocs, seed, colormap, outdir, figdir, datadir
    )

    # Create compositions
    print(f"Plotting SVR composition for: {sampleid}:")

    for parameter in params:
        for kernel in kernels:
            for scaler in scalers:
                if parameter in ["DensityOfFullAssemblage", "Vp", "Vs"]:
                    # First row
                    combine_plots_horizontally(
                        f"{figdir}/MAGEMin-{parameter}-{kernel}-{scaler}.png",
                        f"{figdir}/Perple_X-{parameter}-{kernel}-{scaler}.png",
                        f"{figdir}/temp1.png",
                        caption1="a)",
                        caption2="c)"
                    )

                    os.remove(f"{figdir}/MAGEMin-{parameter}-{kernel}-{scaler}.png")
                    os.remove(f"{figdir}/Perple_X-{parameter}-{kernel}-{scaler}.png")

                    # Second row
                    combine_plots_horizontally(
                        f"{figdir}/MAGEMin-{parameter}-{kernel}-{scaler}-prem.png",
                        f"{figdir}/Perple_X-{parameter}-{kernel}-{scaler}-prem.png",
                        f"{figdir}/temp2.png",
                        caption1="b)",
                        caption2="d)"
                    )

                    os.remove(f"{figdir}/MAGEMin-{parameter}-{kernel}-{scaler}-prem.png")
                    os.remove(f"{figdir}/Perple_X-{parameter}-{kernel}-{scaler}-prem.png")

                    # Stack rows
                    combine_plots_vertically(
                        f"{figdir}/temp1.png",
                        f"{figdir}/temp2.png",
                        f"{figdir}/performance-{parameter}-{kernel}-{scaler}.png",
                        caption1="",
                        caption2=""
                    )
                else:
                    # First row
                    combine_plots_horizontally(
                        f"{figdir}/MAGEMin-{parameter}-{kernel}-{scaler}.png",
                        f"{figdir}/Perple_X-{parameter}-{kernel}-{scaler}.png",
                        f"{figdir}/performance-{parameter}-{kernel}-{scaler}.png",
                        caption1="a)",
                        caption2="c)"
                    )

                    os.remove(f"{figdir}/MAGEMin-{parameter}-{kernel}-{scaler}.png")
                    os.remove(f"{figdir}/Perple_X-{parameter}-{kernel}-{scaler}.png")

                # First row
                combine_plots_horizontally(
                    f"{figdir}/MAGEMin-{parameter}-{kernel}-{scaler}-targets-surf.png",
                    f"{figdir}/Perple_X-{parameter}-{kernel}-{scaler}-targets-surf.png",
                    f"{figdir}/temp1.png",
                    caption1="a)",
                    caption2="d)"
                )

                os.remove(f"{figdir}/MAGEMin-{parameter}-{kernel}-{scaler}-targets-surf.png")
                os.remove(f"{figdir}/Perple_X-{parameter}-{kernel}-{scaler}-targets-surf.png")

                # Second row
                combine_plots_horizontally(
                    f"{figdir}/MAGEMin-{parameter}-{kernel}-{scaler}-svr-surf.png",
                    f"{figdir}/Perple_X-{parameter}-{kernel}-{scaler}-svr-surf.png",
                    f"{figdir}/temp2.png",
                    caption1="b)",
                    caption2="e)"
                )

                os.remove(f"{figdir}/MAGEMin-{parameter}-{kernel}-{scaler}-svr-surf.png")
                os.remove(f"{figdir}/Perple_X-{parameter}-{kernel}-{scaler}-svr-surf.png")

                # Third row
                combine_plots_horizontally(
                    f"{figdir}/MAGEMin-{parameter}-{kernel}-{scaler}-diff-surf.png",
                    f"{figdir}/Perple_X-{parameter}-{kernel}-{scaler}-diff-surf.png",
                    f"{figdir}/temp4.png",
                    caption1="c)",
                    caption2="f)"
                )

                os.remove(f"{figdir}/MAGEMin-{parameter}-{kernel}-{scaler}-diff-surf.png")
                os.remove(f"{figdir}/Perple_X-{parameter}-{kernel}-{scaler}-diff-surf.png")

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
                    f"{figdir}/surf-{parameter}-{kernel}-{scaler}.png",
                    caption1="",
                    caption2=""
                )

                os.remove(f"{figdir}/temp3.png")
                os.remove(f"{figdir}/temp4.png")

                # First row
                combine_plots_horizontally(
                    f"{figdir}/MAGEMin-{parameter}-{kernel}-{scaler}-targets.png",
                    f"{figdir}/Perple_X-{parameter}-{kernel}-{scaler}-targets.png",
                    f"{figdir}/temp1.png",
                    caption1="a)",
                    caption2="d)"
                )

                os.remove(f"{figdir}/MAGEMin-{parameter}-{kernel}-{scaler}-targets.png")
                os.remove(f"{figdir}/Perple_X-{parameter}-{kernel}-{scaler}-targets.png")

                # Second row
                combine_plots_horizontally(
                    f"{figdir}/MAGEMin-{parameter}-{kernel}-{scaler}-svr.png",
                    f"{figdir}/Perple_X-{parameter}-{kernel}-{scaler}-svr.png",
                    f"{figdir}/temp2.png",
                    caption1="b)",
                    caption2="e)"
                )

                os.remove(f"{figdir}/MAGEMin-{parameter}-{kernel}-{scaler}-svr.png")
                os.remove(f"{figdir}/Perple_X-{parameter}-{kernel}-{scaler}-svr.png")

                # Third row
                combine_plots_horizontally(
                    f"{figdir}/MAGEMin-{parameter}-{kernel}-{scaler}-diff.png",
                    f"{figdir}/Perple_X-{parameter}-{kernel}-{scaler}-diff.png",
                    f"{figdir}/temp4.png",
                    caption1="c)",
                    caption2="f)"
                )

                os.remove(f"{figdir}/MAGEMin-{parameter}-{kernel}-{scaler}-diff.png")
                os.remove(f"{figdir}/Perple_X-{parameter}-{kernel}-{scaler}-diff.png")

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
                    f"{figdir}/image-{parameter}-{kernel}-{scaler}.png",
                    caption1="",
                    caption2=""
                )

                os.remove(f"{figdir}/temp3.png")
                os.remove(f"{figdir}/temp4.png")

                os.remove(f"{figdir}/MAGEMin-{parameter}-{kernel}-{scaler}-P.png")
                os.remove(f"{figdir}/MAGEMin-{parameter}-{kernel}-{scaler}-T.png")
                os.remove(f"{figdir}/Perple_X-{parameter}-{kernel}-{scaler}-P.png")
                os.remove(f"{figdir}/Perple_X-{parameter}-{kernel}-{scaler}-T.png")


# Print figure filepaths
print_filepaths(figdir)