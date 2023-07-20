import os
import matplotlib.pyplot as plt
import numpy as np
from magemin import (
    parse_arguments_visualize_db,
    process_MAGEMin_grid,
    process_perplex_assemblage,
    process_perplex_grid,
    create_PT_grid,
    run_svr_regression,
    combine_plots_horizontally,
    combine_plots_vertically,
    visualize_input_data,
    append_to_csv
)

# Parse arguments
#args = parse_arguments_visualize_db()

# Get argument values
#sample_id = args.sampleid
#parameters = args.params
#out_dir = args.outdir
#fig_dir = args.figdir

sample_id = "DMM-128x128"
parameters = ["DensityOfFullAssemblage", "LiquidFraction", "Vp", "Vs"]
out_dir = "runs"
fig_dir = f"figs/svr/{sample_id}"
data_dir = "assets/data"

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(f"Running SVR on {sample_id} ...")
print("Physical properties:")
for param in parameters:
    print(f"    {param}")
print(f"out_dir:  {out_dir}")
print(f"fig_dir:  {fig_dir}")
print(f"data_dir: {data_dir}")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

if (len(os.listdir(f"{out_dir}/{sample_id}")) != 0):

    print("=============================================")
    print("Processing MAGEMin and Perple_X results from:")
    print(f"    MAGEMin: {out_dir}/{sample_id}")
    print(f"    Perple_X: assets/benchmark/{sample_id}/{sample_id}_grid.tab")
    print(f"    Perple_X: assets/benchmark/{sample_id}/{sample_id}_assemblages.txt")

    # Get results
    results_mgm = process_MAGEMin_grid(sample_id, out_dir)
    file_path_results_ppx = f"assets/benchmark/{sample_id}/{sample_id}_grid.tab"
    file_path_assemblage_ppx = f"assets/benchmark/{sample_id}/{sample_id}_assemblages.txt"
    results_ppx = process_perplex_grid(file_path_results_ppx, file_path_assemblage_ppx)

    print("Preprocessing features array (P and T):")
    print("    Transforming units to K and GPa")

    # Get PT values MAGEMin and transform units
    P_mgm = [P / 10 for P in results_mgm["P"]]
    T_mgm = [T + 273 for T in results_mgm["T"]]
    P_ppx = [P / 10 for P in results_ppx["P"]]
    T_ppx = [T + 273 for T in results_ppx["T"]]

    # Reshape into (W, 1) arrays
    P_array_mgm = np.unique(np.array(P_mgm)).reshape(-1, 1)
    T_array_mgm = np.unique(np.array(T_mgm)).reshape(1, -1)
    P_array_ppx = np.unique(np.array(P_ppx)).reshape(-1, 1)
    T_array_ppx = np.unique(np.array(T_ppx)).reshape(1, -1)

    # Get array dimensions
    W_mgm = P_array_mgm.shape[0]
    W_ppx = P_array_ppx.shape[0]

    print(f"    Reshaping P array to {1, W_mgm}")
    print(f"    Reshaping T array to {W_mgm, 1}")

    # Reshape into (W, W) arrays by repeating values
    P_grid_mgm = np.tile(P_array_mgm, (1, W_mgm))
    T_grid_mgm = np.tile(T_array_mgm, (W_mgm, 1))
    P_grid_ppx = np.tile(P_array_ppx, (1, W_ppx))
    T_grid_ppx = np.tile(T_array_ppx, (W_ppx, 1))

    print(f"    Combining PT arrays into feature array with shape {W_mgm, W_mgm, 2}")

    # Combine P and T grids into a single feature set with shape (W, W, 2)
    features_array_mgm = np.stack((P_grid_mgm, T_grid_mgm), axis=-1)
    features_array_ppx = np.stack((P_grid_ppx, T_grid_ppx), axis=-1)

    print("=============================================")

    for parameter in parameters:

        print(f"Preprocessing target array ({parameter}):")

        # Units
        if parameter == "DensityOfFullAssemblage":
            print("    Transforming units from Kg/m^3 to g/cm^3")
            units = "g/cm$^3$"
        if parameter == "LiquidFraction":
            units = None
        if parameter in ["Vp", "Vs"]:
            units = "km/s"

        print(f"    Creating target array with shape {W_mgm, W_mgm}")

        # Target array with shape (W, W)
        target_array_mgm = create_PT_grid(P_mgm, T_mgm, results_mgm[parameter])
        target_array_ppx = create_PT_grid(P_ppx, T_ppx, results_ppx[parameter])

        # Change zero liquid fraction to nan in MAGEMin predictions for better comparison
        if parameter == "LiquidFraction":
            print(f" Setting zero liquid fraction to NaN for better comparisons")
            target_array_mgm = np.where(np.isnan(target_array_ppx), np.nan, target_array_ppx)

        # Get min max of target array to plot colorbars on the same scales
        min_mgm = np.min(np.abs(target_array_mgm[np.logical_not(np.isnan(target_array_mgm))]))
        min_ppx = np.min(np.abs(target_array_ppx[np.logical_not(np.isnan(target_array_ppx))]))
        vmin = min(min_mgm, min_ppx)
        max_mgm = np.max(np.abs(target_array_mgm[np.logical_not(np.isnan(target_array_mgm))]))
        max_ppx = np.max(np.abs(target_array_ppx[np.logical_not(np.isnan(target_array_ppx))]))
        vmax = max(max_mgm, max_ppx)

        # Transform units
        if parameter == "DensityOfFullAssemblage":
            vmin = vmin / 1000
            vmax = vmax / 1000

        print(f"    Finding min, max {round(vmin, 2), round(vmax, 2)}")

        print(f"Plotting input data to: {fig_dir}")

        # Visualizations: input data
        visualize_input_data(
            results_mgm,
            features_array_mgm,
            target_array_mgm,
            sample_id,
            parameter,
            units,
            program="MAGEMin",
            palette="bone",
            vmin=vmin,
            vmax=vmax,
            fig_dir=fig_dir
        )
        visualize_input_data(
            results_ppx,
            features_array_ppx,
            target_array_ppx,
            sample_id,
            parameter,
            units,
            program="Perple_X",
            palette="bone",
            vmin=vmin,
            vmax=vmax,
            fig_dir=fig_dir
        )

        # Run SVR on MAGEMin dataset
        kernels = ["rbf"]
        scalers = ["standard", "minmax"]

        print("Running SVR ...")
        print("=============================================")

        for kernel in kernels:
            for scaler in scalers:
                # Run SVR for MAGEMin
                model_mgm, info_mgm = run_svr_regression(
                    features_array_mgm,
                    target_array_mgm,
                    parameter,
                    units=units,
                    vmin=vmin,
                    vmax=vmax,
                    program="MAGEMin",
                    kernel=kernel,
                    scaler=scaler,
                    filename=f"MAGEMin-{sample_id}-{parameter}-{kernel}-{scaler}.png",
                    fig_dir=fig_dir
                )

                # Write SVR config and performance info to csv
                append_to_csv("assets/data/svr-info.csv", info_mgm)

                # Run SVR for MAGEMin
                model_ppx, info_ppx = run_svr_regression(
                    features_array_ppx,
                    target_array_ppx,
                    parameter,
                    units=units,
                    vmin=vmin,
                    vmax=vmax,
                    program="Perple_X",
                    kernel=kernel,
                    scaler=scaler,
                    filename=f"Perple_X-{sample_id}-{parameter}-{kernel}-{scaler}.png",
                    fig_dir=fig_dir
                )

                # Write SVR config and performance info to csv
                append_to_csv(f"{data_dir}/svr-info.csv", info_ppx)

        print("=============================================")

    # Create compositions
    print(f"Writing SVR results to: {data_dir}/svr-info.csv")
    print(f"Plotting SVR results to: {fig_dir}")

    for parameter in parameters:
        # First row surface
        combine_plots_horizontally(
            f"{fig_dir}/MAGEMin-{sample_id}-{parameter}-histogram.png",
            f"{fig_dir}/Perple_X-{sample_id}-{parameter}-histogram.png",
            f"{fig_dir}/comp-histogram-{sample_id}-{parameter}.png",
            caption1="a)",
            caption2="b)"
        )

        # First row scatter
        combine_plots_horizontally(
            f"{fig_dir}/MAGEMin-{sample_id}-{parameter}-scatter-P.png",
            f"{fig_dir}/Perple_X-{sample_id}-{parameter}-scatter-P.png",
            f"{fig_dir}/temp1.png",
            caption1="a)",
            caption2="c)"
        )
        # Second row scatter
        combine_plots_horizontally(
            f"{fig_dir}/MAGEMin-{sample_id}-{parameter}-scatter-T.png",
            f"{fig_dir}/Perple_X-{sample_id}-{parameter}-scatter-T.png",
            f"{fig_dir}/temp2.png",
            caption1="b)",
            caption2="d)"
        )
        # Stack rows
        combine_plots_vertically(
            f"{fig_dir}/temp1.png",
            f"{fig_dir}/temp2.png",
            f"{fig_dir}/comp-scatter-{sample_id}-{parameter}.png",
            caption1="",
            caption2=""
        )

        # Cleanup dir
        if os.path.exists(f"{fig_dir}/temp1.png"):
            os.remove(f"{fig_dir}/temp1.png")
        if os.path.exists(f"{fig_dir}/temp2.png"):
            os.remove(f"{fig_dir}/temp2.png")

        # First row SVR MGM
        combine_plots_horizontally(
            f"{fig_dir}/MAGEMin-{sample_id}-{parameter}-{kernels[0]}-{scalers[0]}.png",
            f"{fig_dir}/Perple_X-{sample_id}-{parameter}-{kernels[0]}-{scalers[0]}.png",
            f"{fig_dir}/temp1.png",
            caption1="a)",
            caption2="c)"
        )
        # Second row SVR PPX
        combine_plots_horizontally(
            f"{fig_dir}/MAGEMin-{sample_id}-{parameter}-{kernels[0]}-{scalers[1]}.png",
            f"{fig_dir}/Perple_X-{sample_id}-{parameter}-{kernels[0]}-{scalers[1]}.png",
            f"{fig_dir}/temp2.png",
            caption1="b)",
            caption2="d)"
        )
        # Stack rows
        combine_plots_vertically(
            f"{fig_dir}/temp1.png",
            f"{fig_dir}/temp2.png",
            f"{fig_dir}/comp-svr-{sample_id}-{parameter}.png",
            caption1="",
            caption2=""
        )

        # Cleanup dir
        if os.path.exists(f"{fig_dir}/temp1.png"):
            os.remove(f"{fig_dir}/temp1.png")
        if os.path.exists(f"{fig_dir}/temp2.png"):
            os.remove(f"{fig_dir}/temp2.png")

    print("=============================================")