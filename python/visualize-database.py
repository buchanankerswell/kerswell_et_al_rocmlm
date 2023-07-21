import os
import numpy as np
from magemin import (
    encode_phases,
    visualize_MAD,
    create_PT_grid,
    parse_arguments,
    check_arguments,
    print_filepaths,
    process_MAGEMin_grid
)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "visualize-database.py")

# Load valid arguments
locals().update(valid_args)

# Plot MAGEMin output
if len(os.listdir(outdir + "/" + sampleid)) != 0:
    print(f"Plotting results for {sampleid}:")

    results_mgm = process_MAGEMin_grid(sampleid, outdir)

    # Get PT values MAGEMin and transform units
    P_mgm = [P / 10 for P in results_mgm["P"]]
    T_mgm = [T + 273 for T in results_mgm["T"]]

    for parameter in params:
        # Transform results into 2D numpy arrays
        if parameter == "StableSolutions":
            # Encode unique phase assemblages MAGEMin
            encoded_mgm, unique_mgm = encode_phases(results_mgm[parameter])
            grid_mgm = create_PT_grid(P_mgm, T_mgm, encoded_mgm)
        else:
            grid_mgm = create_PT_grid(P_mgm, T_mgm, results_mgm[parameter])

        # Transform units
        if parameter == "DensityOfFullAssemblage":
            grid_mgm = grid_mgm / 1000

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
            vmin=np.min(grid_mgm[np.logical_not(np.isnan(grid_mgm))])
            vmax=np.max(grid_mgm[np.logical_not(np.isnan(grid_mgm))])
        else:
            num_colors_mgm = len(np.unique(grid_mgm))
            vmin = 1
            vmax = num_colors_mgm + 1

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