import os
import numpy as np
from magemin import (
    encode_phases,
    visualize_MAD,
    create_PT_grid,
    visualize_PREM,
    parse_arguments,
    check_arguments,
    print_filepaths,
    process_MAGEMin_grid,
    process_perplex_grid,
    combine_plots_vertically,
    combine_plots_horizontally,
    process_perplex_assemblage
)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "visualize-benchmark.py")

# Load valid arguments
locals().update(valid_args)

# Plot benchmark comparison
if (len(os.listdir(f"{outdir}/{sampleid}")) != 0 and
    os.path.exists(f"assets/benchmark/{sampleid}/{sampleid}_grid.tab")):
    print(f"Plotting benchmark comparison for {sampleid}:")

    # Get MAGEMin results
    results_mgm = process_MAGEMin_grid(sampleid, outdir)

    # Get perplex results
    file_path_results_ppx = f"assets/benchmark/{sampleid}/{sampleid}_grid.tab"
    file_path_assemblage_ppx = f"assets/benchmark/{sampleid}/{sampleid}_assemblages.txt"
    results_ppx = process_perplex_grid(file_path_results_ppx, file_path_assemblage_ppx)

    # Get PT values MAGEMin and transform units
    P_mgm = [P / 10 for P in results_mgm["P"]]
    T_mgm = [T + 273 for T in results_mgm["T"]]

    # Get PT values perplex and transform units
    P_ppx = [P / 10 for P in results_ppx["P"]]
    T_ppx = [T + 273 for T in results_ppx["T"]]

    for parameter in params:
        # Transform results into 2D numpy arrays
        if parameter == "StableSolutions":
            # Encode unique phase assemblages MAGEMin
            encoded_mgm, unique_mgm = encode_phases(
                results_mgm[parameter],
                filename=f"assets/benchmark/{sampleid}/{sampleid}_mgm_assemblages.csv"
            )
            grid_mgm = create_PT_grid(P_mgm, T_mgm, encoded_mgm)

            # Encode unique phase assemblages perplex
            encoded_ppx, unique_ppx = encode_phases(
                results_ppx[parameter],
                filename=f"assets/benchmark/{sampleid}/{sampleid}_ppx_assemblages.csv"
            )
            grid_ppx = create_PT_grid(P_ppx, T_ppx, encoded_ppx)
        else:
            grid_mgm = create_PT_grid(P_mgm, T_mgm, results_mgm[parameter])
            grid_ppx = create_PT_grid(P_ppx, T_ppx, results_ppx[parameter])

        # Change zero liquid fraction to nan in MAGEMin predictions for better comparison
        if parameter == "LiquidFraction":
            grid_ppx = np.where(np.isnan(grid_ppx), 0, grid_ppx)

        # Transform units
        if parameter == "DensityOfFullAssemblage":
            grid_mgm = grid_mgm / 1000
            grid_ppx = grid_ppx / 1000

        # Use discrete colorscale
        if parameter in ["StableSolutions", "StableVariance"]:
            color_discrete = True
        else:
            color_discrete = False

        # Reverse color scale
        if colormap in ["grey"]:
            if parameter in ["StableVariance"]:
                color_reverse = True
            else:
                color_reverse = False
        else:
            if parameter in ["StableVariance"]:
                color_reverse = False
            else:
                color_reverse = True

        # Set colorbar limits for better comparisons
        if not color_discrete:
            vmin_mgm=np.min(grid_mgm[np.logical_not(np.isnan(grid_mgm))])
            vmax_mgm=np.max(grid_mgm[np.logical_not(np.isnan(grid_mgm))])
            vmin_ppx=np.min(grid_ppx[np.logical_not(np.isnan(grid_ppx))])
            vmax_ppx=np.max(grid_ppx[np.logical_not(np.isnan(grid_ppx))])

            vmin = min(vmin_mgm, vmin_ppx)
            vmax = max(vmax_mgm, vmax_ppx)
        else:
            num_colors_mgm = len(np.unique(grid_mgm))
            num_colors_ppx = len(np.unique(grid_ppx))
            vmin = 1
            vmax = max(num_colors_mgm, num_colors_ppx) + 1

        # Plot PT grid MAGEMin
        visualize_MAD(
            P_mgm, T_mgm, grid_mgm, parameter,
            geotherm=True,
            geotherm_linetype="-.",
            geotherm_color=0,
            title="MAGEMin",
            palette=colormap,
            color_discrete=color_discrete,
            color_reverse=color_reverse,
            vmin=vmin,
            vmax=vmax,
            filename=f"MAGEMin-{sampleid}-{parameter}.png",
            fig_dir=figdir
        )

        # Plot PT grid perplex
        visualize_MAD(
            P_ppx, T_ppx, grid_ppx, parameter,
            geotherm=True,
            geotherm_linetype="--",
            geotherm_color=1,
            title="Perple_X",
            palette=colormap,
            color_discrete=color_discrete,
            color_reverse=color_reverse,
            vmin=vmin,
            vmax=vmax,
            filename=f"Perple_X-{sampleid}-{parameter}.png",
            fig_dir=figdir
        )

        if not color_discrete:
            # Comppute normalized diff
            mask = ~np.isnan(grid_mgm) & ~np.isnan(grid_ppx)
            max_diff = np.max(np.abs(grid_mgm[mask] - grid_ppx[mask]))
            diff_norm = (grid_mgm - grid_ppx) / max_diff
            diff_norm[~mask] = np.nan

            # Plot PT grid normalized diff mgm-ppx
            visualize_MAD(
                P_ppx, T_ppx, diff_norm, parameter,
                title="Normalized Difference",
                palette="seismic",
                color_discrete=color_discrete,
                color_reverse=False,
                filename=f"diff-norm-{sampleid}-{parameter}.png",
                fig_dir=figdir
            )

            # Compute the absolute gradient along the rows and columns
            gradient_rows = np.abs(np.diff(diff_norm, axis=0))
            gradient_cols = np.abs(np.diff(diff_norm, axis=1))

            # Pad the gradients to match the original size
            gradient_rows_padded = np.pad(
                gradient_rows, ((0, 1), (0, 0)),
                mode="constant",
                constant_values=np.nan
            )
            gradient_cols_padded = np.pad(
                gradient_cols, ((0, 0), (0, 1)),
                mode="constant",
                constant_values=np.nan
            )

            # Compute the maximum gradient between rows and columns
            max_gradient = np.maximum(gradient_rows_padded, gradient_cols_padded)

            # Plot PT grid max gradient
            visualize_MAD(
                P_ppx, T_ppx, max_gradient, parameter,
                title="Difference Gradient",
                palette=colormap,
                color_discrete=color_discrete,
                color_reverse=color_reverse,
                filename=f"max-grad-{sampleid}-{parameter}.png",
                fig_dir=figdir
            )

            # Plot PREM comparisons
            if parameter == "DensityOfFullAssemblage":
                visualize_PREM(
                    "assets/data/prem.csv",
                    results_mgm,
                    results_ppx,
                    parameter=parameter,
                    param_unit="g/cm$^3$",
                    geotherm_threshold=0.1,
                    title="PREM Comparison",
                    filename=f"prem-{sampleid}-{parameter}.png",
                    fig_dir=figdir
                )

            if parameter in ["Vp", "Vs"]:
                visualize_PREM(
                    "assets/data/prem.csv",
                    results_mgm,
                    results_ppx,
                    parameter=parameter,
                    param_unit="km/s",
                    geotherm_threshold=0.1,
                    title="PREM Comparison",
                    filename=f"prem-{sampleid}-{parameter}.png",
                    fig_dir=figdir
                )

            # Create composition for continuous variables

            # First row
            combine_plots_horizontally(
                f"{figdir}/MAGEMin-{sampleid}-{parameter}.png",
                f"{figdir}/Perple_X-{sampleid}-{parameter}.png",
                f"{figdir}/temp1.png",
                caption1="a",
                caption2="b"
            )

            # Second row
            if parameter in ["DensityOfFullAssemblage", "Vp", "Vs"]:
                combine_plots_horizontally(
                    f"{figdir}/diff-norm-{sampleid}-{parameter}.png",
                    f"{figdir}/prem-{sampleid}-{parameter}.png",
                    f"{figdir}/temp2.png",
                    caption1="c",
                    caption2="d"
                )
            else:
                combine_plots_horizontally(
                    f"{figdir}/diff-norm-{sampleid}-{parameter}.png",
                    f"{figdir}/max-grad-{sampleid}-{parameter}.png",
                    f"{figdir}/temp2.png",
                    caption1="c",
                    caption2="d"
                )

            # Stack rows
            combine_plots_vertically(
                f"{figdir}/temp1.png",
                f"{figdir}/temp2.png",
                f"{figdir}/comp-{sampleid}-{parameter}.png",
                caption1="",
                caption2=""
            )

        # Create composition for discrete variables
        if color_discrete:
            # First row
            combine_plots_horizontally(
                f"{figdir}/MAGEMin-{sampleid}-{parameter}.png",
                f"{figdir}/Perple_X-{sampleid}-{parameter}.png",
                f"{figdir}/comp-{sampleid}-{parameter}.png",
                caption1="a",
                caption2="b"
            )

        # Cleanup dir
        if os.path.exists(f"{figdir}/temp1.png"):
            os.remove(f"{figdir}/temp1.png")
        if os.path.exists(f"{figdir}/temp2.png"):
            os.remove(f"{figdir}/temp2.png")

    # Print figure filepaths
    print_filepaths(figdir)

