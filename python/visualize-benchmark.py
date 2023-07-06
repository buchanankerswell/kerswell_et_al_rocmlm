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
    plot_histogram,
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

    # Get PT values MAGEMin
    P_mgm = results_mgm["P"]
    T_mgm = results_mgm["T"]

    # Get PT values perplex
    P_ppx = results_ppx["P"]
    T_ppx = results_ppx["T"]

    for parameter in parameters:

        # Transform results into 2D numpy arrays
        if parameter == "StableSolutions":
            # Encode unique phase assemblages MAGEMin
            encoded_mgm, unique_mgm = encode_phases(results_mgm[parameter])
            grid_mgm = create_PT_grid(P_mgm, T_mgm, encoded_mgm)

            # Encode unique phase assemblages perplex
            encoded_ppx, unique_ppx = encode_phases(results_ppx[parameter])
            grid_ppx = create_PT_grid(P_ppx, T_ppx, encoded_ppx)

        else:
            # MAGEMin grid
            grid_mgm = create_PT_grid(P_mgm, T_mgm, results_mgm[parameter])

            # perplex grid
            grid_ppx = create_PT_grid(P_ppx, T_ppx, results_ppx[parameter])

        # Change zero liquid fraction to nan in MAGEMin predictions for better comparison
        if parameter == "LiquidFraction":
            grid_mgm = np.where(grid_mgm == 0, np.nan, grid_mgm)

        # Use discrete colorscale
        if parameter in ["StableSolutions", "StableVariance"]:
            color_discrete = True
        else:
            color_discrete = False

        # Reverse color scale
        if parameter in ["StableVariance"]:
            color_reverse = True
        else:
            color_reverse = False

        if not color_discrete:
            # Colorbar limits
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
            # Plot distributions
            plot_histogram(
                grid_mgm,
                grid_ppx,
                parameter,
                bins=30,
                title="Model Predictions",
                filename=f"hist-{sample_id}-{parameter}.png",
                fig_dir=fig_dir
            )

            # Plot PT grid diff mgm-ppx
            plot_MAD(
                P_ppx, T_ppx, grid_mgm - grid_ppx, parameter,
                title="Difference (MGM - PPX)",
                palette="seismic",
                color_discrete=color_discrete,
                color_reverse=False,
                filename=f"diff-{sample_id}-{parameter}.png",
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
                f"{fig_dir}/diff-{sample_id}-{parameter}.png",
                f"{fig_dir}/hist-{sample_id}-{parameter}.png",
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

# Read benchmark data
if os.path.exists("assets/data"):
    comp_times = "assets/data/benchmark-comp-times.csv"
    # Plot benchmark comp times
    visualize_benchmark_comp_times(comp_times, fig_dir="figs/benchmark")
