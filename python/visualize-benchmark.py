import os
import numpy as np
from magemin import (
    parse_arguments_visualize_db,
    visualize_benchmark_comp_times,
    process_MAGEMin_grid,
    process_perplex_assemblage,
    process_perplex_grid,
    encode_phases,
    create_PT_grid,
    plot_MAD,
    combine_plots_horizontally,
    combine_plots_vertically
)

# Parse arguments
args = parse_arguments_visualize_db()

# Get argument values
sample_id = args.sampleid
parameters = args.params
palette = args.colormap
out_dir = args.outdir
fig_dir = args.figdir

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(f"Plotting sample {sample_id} the following parameters:")
print("Physical properties:")
for param in parameters:
    print(f"    {param}")
print("Oxides (for Harker diagrams):")
print(f"out_dir: {out_dir}")
print(f"fig_dir: {fig_dir}")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Plot benchmark comparison
if (len(os.listdir(f"{out_dir}/{sample_id}")) != 0 and
    os.path.exists(f"assets/benchmark/{sample_id}/{sample_id}_grid.tab")):
    print(f"Plotting benchmark comparison for {sample_id} ...")

    # Get MAGEMin results
    results_mgm = process_MAGEMin_grid(sample_id, out_dir)

    # Get perplex results
    file_path_results_ppx = f"assets/benchmark/{sample_id}/{sample_id}_grid.tab"
    file_path_assemblage_ppx = f"assets/benchmark/{sample_id}/{sample_id}_assemblages.txt"
    results_ppx = process_perplex_grid(file_path_results_ppx, file_path_assemblage_ppx)

    # Get PT values MAGEMin and transform units
    P_mgm = [P / 10 for P in results_mgm["P"]]
    T_mgm = [T + 273 for T in results_mgm["T"]]

    # Get PT values perplex and transform units
    P_ppx = [P / 10 for P in results_ppx["P"]]
    T_ppx = [T + 273 for T in results_ppx["T"]]

    for parameter in parameters:

        # Transform results into 2D numpy arrays
        if parameter == "StableSolutions":
            # Encode unique phase assemblages MAGEMin
            encoded_mgm, unique_mgm = encode_phases(
                results_mgm[parameter],
                filename=f"assets/benchmark/{sample_id}/{sample_id}_mgm_assemblages.csv"
            )
            grid_mgm = create_PT_grid(P_mgm, T_mgm, encoded_mgm)

            # Encode unique phase assemblages perplex
            encoded_ppx, unique_ppx = encode_phases(
                results_ppx[parameter],
                filename=f"assets/benchmark/{sample_id}/{sample_id}_ppx_assemblages.csv"
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
            grid_mgm = grid_mgm/1000
            grid_ppx = grid_ppx/1000

        # Use discrete colorscale
        if parameter in ["StableSolutions", "StableVariance"]:
            color_discrete = True
        else:
            color_discrete = False

        # Reverse color scale
        if palette in ["grey"]:
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
        plot_MAD(
            P_mgm, T_mgm, grid_mgm, parameter,
            title="MAGEMin",
            palette=palette,
            color_discrete=color_discrete,
            color_reverse=color_reverse,
            vmin=vmin,
            vmax=vmax,
            filename=f"MAGEMin-{sample_id}-{parameter}.png",
            fig_dir=fig_dir
        )

        # Plot PT grid perplex
        plot_MAD(
            P_ppx, T_ppx, grid_ppx, parameter,
            title="Perple_X",
            palette=palette,
            color_discrete=color_discrete,
            color_reverse=color_reverse,
            vmin=vmin,
            vmax=vmax,
            filename=f"Perple_X-{sample_id}-{parameter}.png",
            fig_dir=fig_dir
        )

        if not color_discrete:
            # Comppute normalized diff
            mask = ~np.isnan(grid_mgm) & ~np.isnan(grid_ppx)
            max_diff = np.max(np.abs(grid_mgm[mask] - grid_ppx[mask]))
            diff_norm = (grid_mgm - grid_ppx) / max_diff
            diff_norm[~mask] = np.nan

            # Plot PT grid normalized diff mgm-ppx
            plot_MAD(
                P_ppx, T_ppx, diff_norm, parameter,
                title="Normalized Difference",
                palette="seismic",
                color_discrete=color_discrete,
                color_reverse=False,
                filename=f"diff-norm-{sample_id}-{parameter}.png",
                fig_dir=fig_dir
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
            plot_MAD(
                P_ppx, T_ppx, max_gradient, parameter,
                title="Difference Gradient",
                palette=palette,
                color_discrete=color_discrete,
                color_reverse=color_reverse,
                filename=f"max-grad-{sample_id}-{parameter}.png",
                fig_dir=fig_dir
            )

            # Create composition for continuous variables
            # First row
            combine_plots_horizontally(
                f"{fig_dir}/MAGEMin-{sample_id}-{parameter}.png",
                f"{fig_dir}/Perple_X-{sample_id}-{parameter}.png",
                f"{fig_dir}/temp1.png",
                caption1="a",
                caption2="b"
            )
            # Second row
            combine_plots_horizontally(
                f"{fig_dir}/diff-norm-{sample_id}-{parameter}.png",
                f"{fig_dir}/max-grad-{sample_id}-{parameter}.png",
                f"{fig_dir}/temp2.png",
                caption1="c",
                caption2="d"
            )
            # Stack rows
            combine_plots_vertically(
                f"{fig_dir}/temp1.png",
                f"{fig_dir}/temp2.png",
                f"{fig_dir}/comp-{sample_id}-{parameter}.png",
            )

        # Create composition for discrete variables
        if color_discrete:
            # First row
            combine_plots_horizontally(
                f"{fig_dir}/MAGEMin-{sample_id}-{parameter}.png",
                f"{fig_dir}/Perple_X-{sample_id}-{parameter}.png",
                f"{fig_dir}/comp-{sample_id}-{parameter}.png",
                caption1="a",
                caption2="b"
            )

        # Cleanup dir
        if os.path.exists(f"{fig_dir}/temp1.png"):
            os.remove(f"{fig_dir}/temp1.png")
        if os.path.exists(f"{fig_dir}/temp2.png"):
            os.remove(f"{fig_dir}/temp2.png")
