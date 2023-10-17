#######################################################
## .0.              Load Libraries               !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# utilities !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import glob
import shutil
import warnings
import subprocess

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# dataframes and arrays !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import cv2
import numpy as np
import pandas as pd

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GFEM models !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from gfem import GFEMModel

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# plotting !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import seaborn as sns
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageFont
from matplotlib.colors import ListedColormap

#######################################################
## .1.              Visualizations               !!! ##
#######################################################

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ .1.1            Helper Functions              !!! ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# get geotherm !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_geotherm(results, target, threshold, thermal_gradient=0.5, mantle_potential_T=1573):
    """
    """
    # Get PT and target values and transform units
    df = pd.DataFrame({"P": results["P"], "T": results["T"],
                       target: results[target]}).sort_values(by="P")

    # Calculate geotherm
    df["geotherm_P"] = (df["T"] - mantle_potential_T) / (thermal_gradient * 35)

    # Subset df along geotherm
    df_geotherm = df[abs(df["P"] - df["geotherm_P"]) < threshold]

    # Extract the three vectors
    P_values = df_geotherm["P"].values
    T_values = df_geotherm["T"].values
    targets = df_geotherm[target].values

    return P_values, T_values, targets

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# combine plots horizontally !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def combine_plots_horizontally(image1_path, image2_path, output_path, caption1, caption2,
                               font_size=150, caption_margin=25, dpi=330):
    """
    """
    # Open the images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Determine the maximum height between the two images
    max_height = max(image1.height, image2.height)

    # Create a new image with twice the width and the maximum height
    combined_width = image1.width + image2.width
    combined_image = Image.new("RGB", (combined_width, max_height), (255, 255, 255))

    # Set the DPI metadata
    combined_image.info["dpi"] = (dpi, dpi)

    # Paste the first image on the left
    combined_image.paste(image1, (0, 0))

    # Paste the second image on the right
    combined_image.paste(image2, (image1.width, 0))

    # Add captions
    draw = ImageDraw.Draw(combined_image)
    font = ImageFont.truetype("Arial", font_size)
    caption_margin = caption_margin

    # Add caption
    draw.text((caption_margin, caption_margin), caption1, font=font, fill="black")

    # Add caption "b"
    draw.text((image1.width + caption_margin, caption_margin), caption2, font=font,
              fill="black")

    # Save the combined image with captions
    combined_image.save(output_path, dpi=(dpi, dpi))

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# combine plots vertically !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def combine_plots_vertically(image1_path, image2_path, output_path, caption1, caption2,
                             font_size=150, caption_margin=25, dpi=330):
    """
    """
    # Open the images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Determine the maximum width between the two images
    max_width = max(image1.width, image2.width)

    # Create a new image with the maximum width and the sum of the heights
    combined_height = image1.height + image2.height
    combined_image = Image.new("RGB", (max_width, combined_height), (255, 255, 255))

    # Paste the first image on the top
    combined_image.paste(image1, (0, 0))

    # Paste the second image below the first
    combined_image.paste(image2, (0, image1.height))

    # Add captions
    draw = ImageDraw.Draw(combined_image)
    font = ImageFont.truetype("Arial", font_size)
    caption_margin = caption_margin

    # Add caption
    draw.text((caption_margin, caption_margin), caption1, font=font, fill="black")

    # Add caption "b"
    draw.text((caption_margin, image1.height + caption_margin), caption2, font=font,
              fill="black")

    # Save the combined image with captions
    combined_image.save(output_path, dpi=(dpi, dpi))

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# compose dataset plots !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compose_dataset_plots(gfem_models):
    """
    """
    # Parse models
    magemin_models = [m if m.program == "magemin" and m.dataset == "train" else
                      None for m in gfem_models]
    perplex_models = [m if m.program == "perplex" and m.dataset == "train" else
                      None for m in gfem_models]

    # Iterate through all models
    for magemin_model, perplex_model in zip(magemin_models, perplex_models):
        magemin = True if magemin_model is not None else False
        perplex = True if perplex_model is not None else False

        if not magemin and not perplex:
            print("No models to visualize!")

            break

        if magemin and perplex:
            # Get model data
            if magemin_model.sample_id == perplex_model.sample_id:
                sample_id = magemin_model.sample_id
            else:
                raise ValueError("Model samples are not the same!")
            if magemin_model.res == perplex_model.res:
                res = magemin_model.res
            else:
                raise ValueError("Model resolutions are not the same!")
            if magemin_model.dataset == perplex_model.dataset:
                dataset = magemin_model.dataset
            else:
                raise ValueError("Model datasets are not the same!")
            if magemin_model.targets == perplex_model.targets:
                targets = magemin_model.targets
            else:
                raise ValueError("Model targets are not the same!")
            if magemin_model.fig_dir == perplex_model.fig_dir:
                fig_dir = magemin_model.fig_dir
            else:
                raise ValueError("Model figure directories are not the same!")
            if magemin_model.verbose == perplex_model.verbose:
                verbose = magemin_model.verbose
            else:
                verbose = 1

        elif magemin and not perplex:
            # Get model data
            sample_id = magemin_model.sample_id
            res = magemin_model.res
            dataset = magemin_model.dataset
            targets = magemin_model.targets
            fig_dir = magemin_model.fig_dir
            verbose = magemin_model.verbose

        elif perplex and not magemin:
            # Get model data
            sample_id = perplex_model.sample_id
            res = perplex_model.res
            dataset = perplex_model.dataset
            targets = perplex_model.targets
            fig_dir = perplex_model.fig_dir
            verbose = perplex_model.verbose

        # Set geotherm threshold for extracting depth profiles
        if res <= 8:
            geotherm_threshold = 4

        elif res <= 16:
            geotherm_threshold = 2

        elif res <= 32:
            geotherm_threshold = 1

        elif res <= 64:
            geotherm_threshold = 0.5

        elif res <= 128:
            geotherm_threshold = 0.25

        else:
            geotherm_threshold = 0.125

        # Rename targets
        targets_rename = [target.replace("_", "-") for target in targets]

        print(f"Composing plots: {fig_dir}")

        if magemin and perplex:
            for target in targets_rename:
                if target not in ["assemblage", "variance"]:
                    combine_plots_horizontally(
                        f"{fig_dir}/magemin-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/perplex-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/temp1.png",
                        caption1="a)",
                        caption2="b)"
                    )

                    combine_plots_horizontally(
                        f"{fig_dir}/temp1.png",
                        f"{fig_dir}/diff-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/image3-{sample_id}-{dataset}-{target}.png",
                        caption1="",
                        caption2="c)"
                    )

                if target in ["rho", "Vp", "Vs"]:
                    combine_plots_horizontally(
                        f"{fig_dir}/magemin-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/perplex-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/temp1.png",
                        caption1="a)",
                        caption2="b)"
                    )

                    combine_plots_horizontally(
                        f"{fig_dir}/diff-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/prem-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/temp2.png",
                        caption1="c)",
                        caption2="d)"
                    )

                    combine_plots_vertically(
                        f"{fig_dir}/temp1.png",
                        f"{fig_dir}/temp2.png",
                        f"{fig_dir}/image4-{sample_id}-{dataset}-{target}.png",
                        caption1="",
                        caption2=""
                    )

        elif magemin and not perplex:
            for target in targets_rename:
                if target not in ["assemblage", "variance"]:
                    combine_plots_horizontally(
                        f"{fig_dir}/magemin-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/grad-magemin-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/image2-{sample_id}-{dataset}-{target}.png",
                        caption1="a)",
                        caption2="b)"
                    )

                    if target in ["rho", "Vp", "Vs"]:
                        combine_plots_horizontally(
                            f"{fig_dir}/magemin-{sample_id}-{dataset}-{target}.png",
                            f"{fig_dir}/grad-magemin-{sample_id}-{dataset}-{target}.png",
                            f"{fig_dir}/temp1.png",
                            caption1="a)",
                            caption2="b)"
                        )

                        combine_plots_horizontally(
                            f"{fig_dir}/temp1.png",
                            f"{fig_dir}/prem-{sample_id}-{dataset}-{target}.png",
                            f"{fig_dir}/image3-{sample_id}-{dataset}-{target}.png",
                            caption1="",
                            caption2="c)"
                        )

        elif perplex and not magemin:
            for target in targets_rename:
                if target not in ["assemblage", "variance"]:
                    combine_plots_horizontally(
                        f"{fig_dir}/perplex-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/grad-perplex-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/image2-{sample_id}-{dataset}-{target}.png",
                        caption1="a)",
                        caption2="b)"
                    )

                    if target in ["rho", "Vp", "Vs"]:
                        combine_plots_horizontally(
                            f"{fig_dir}/perplex-{sample_id}-{dataset}-{target}.png",
                            f"{fig_dir}/grad-perplex-{sample_id}-{dataset}-{target}.png",
                            f"{fig_dir}/temp1.png",
                            caption1="a)",
                            caption2="b)"
                        )

                        combine_plots_horizontally(
                            f"{fig_dir}/temp1.png",
                            f"{fig_dir}/prem-{sample_id}-{dataset}-{target}.png",
                            f"{fig_dir}/image3-{sample_id}-{dataset}-{target}.png",
                            caption1="",
                            caption2="c)"
                        )

            if all(item in targets_rename for item in ["rho", "Vp", "Vs"]):
                captions = [("a", "b", "c"), ("d", "e", "f"), ("g", "h", "i")]
                targets = ["rho", "Vp", "Vs"]

                for i, target in enumerate(targets):
                    combine_plots_horizontally(
                        f"{fig_dir}/perplex-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/grad-perplex-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/temp1.png",
                        caption1=captions[i][0],
                        caption2=captions[i][1]
                    )

                    combine_plots_horizontally(
                        f"{fig_dir}/temp1.png",
                        f"{fig_dir}/prem-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/temp-{target}.png",
                        caption1="",
                        caption2=captions[i][2]
                    )

                combine_plots_vertically(
                    f"{fig_dir}/temp-rho.png",
                    f"{fig_dir}/temp-Vp.png",
                    f"{fig_dir}/temp1.png",
                    caption1="",
                    caption2=""
                )

                combine_plots_vertically(
                    f"{fig_dir}/temp1.png",
                    f"{fig_dir}/temp-Vs.png",
                    f"{fig_dir}/image9-{sample_id}-{dataset}.png",
                    caption1="",
                    caption2=""
                )

        # Clean up directory
        tmp_files = glob.glob(f"{fig_dir}/temp*.png")
        prem_files = glob.glob(f"{fig_dir}/prem*.png")
        grad_files = glob.glob(f"{fig_dir}/grad*.png")
        diff_files = glob.glob(f"{fig_dir}/diff*.png")
        mgm_files = glob.glob(f"{fig_dir}/mage*.png")
        ppx_files = glob.glob(f"{fig_dir}/perp*.png")

        for file in tmp_files + prem_files + grad_files + mgm_files + ppx_files:
            os.remove(file)

    # Check for multiple movie frames
    if sample_id not in ["PUM", "DMM", "NMORB", "RE46"]:
        # Make movies
        print("Creating mp4 movies ...")

        if os.path.exists("figs/movies"):
            shutil.rmtree("figs/movies")
            os.makedirs("figs/movies", exist_ok=True)
        else:
            os.makedirs("figs/movies", exist_ok=True)

        for target in targets_rename:
            ffmpeg = (f"ffmpeg -framerate 15 -pattern_type glob -i "
                      f"'figs/s?????_{res}/image2-s?????-{dataset}-{target}.png' "
                      f"-vf 'scale=3915:1432' -c:v h264 -pix_fmt yuv420p "
                      f"'figs/movies/image2-{target}.mp4'")

            try:
                subprocess.run(ffmpeg, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, shell=True)
                print(f"Image2 [{target}] movie created!")

            except subprocess.CalledProcessError as e:
                print(f"Error running FFmpeg command: {e}")

        if all(item in targets_rename for item in ["rho", "Vp", "Vs"]):
            for target in ["rho", "Vp", "Vs"]:
                ffmpeg = (f"ffmpeg -framerate 15 -pattern_type glob -i "
                          f"'figs/s?????_{res}/image3-s?????-{dataset}-{target}.png' "
                          f"-vf 'scale=5832:1432' -c:v h264 -pix_fmt yuv420p "
                          f"'figs/movies/image3-{target}.mp4'")

                try:
                    subprocess.run(ffmpeg, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, shell=True)
                    print(f"Image3 [{target}] movie created!")

                except subprocess.CalledProcessError as e:
                    print(f"Error running FFmpeg command: {e}")

            ffmpeg = (f"ffmpeg -framerate 15 -pattern_type glob -i "
                      f"'figs/s?????_{res}/image9-s?????-{dataset}.png' "
                      f"-vf 'scale=5842:4296' -c:v h264 -pix_fmt yuv420p "
                      f"'figs/movies/image9.mp4'")

            try:
                subprocess.run(ffmpeg, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, shell=True)
                print(f"Image9 movie created!")

            except subprocess.CalledProcessError as e:
                print(f"Error running FFmpeg command: {e}")

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# compose rocml plots !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compose_rocml_plots(rocml_model):
    """
    """
    # Get ml model attributes
    program = rocml_model.program
    model_prefix = rocml_model.model_prefix
    sample_ids = rocml_model.sample_ids
    res = rocml_model.res
    targets = rocml_model.targets
    fig_dir = rocml_model.fig_dir

    # Rename targets
    targets_rename = [target.replace("_", "-") for target in targets]

    print(f"Composing plots: {fig_dir}")

    for target in targets_rename:
        visualize_rocml_performance(target.replace("-", "_"), res, fig_dir, "rocml")

        combine_plots_horizontally(
            f"{fig_dir}/rocml-inference-time-mean-{res}.png",
            f"{fig_dir}/rocml-training-time-mean-{res}.png",
            f"{fig_dir}/temp1.png",
            caption1="a)",
            caption2="b)"
        )

        os.remove(f"{fig_dir}/rocml-inference-time-mean-{res}.png")
        os.remove(f"{fig_dir}/rocml-training-time-mean-{res}.png")

        combine_plots_horizontally(
            f"{fig_dir}/rocml-rmse-test-mean-{target}-{res}.png",
            f"{fig_dir}/rocml-rmse-val-mean-{target}-{res}.png",
            f"{fig_dir}/temp2.png",
            caption1="c)",
            caption2="d)"
        )

        os.remove(f"{fig_dir}/rocml-rmse-test-mean-{target}-{res}.png")
        os.remove(f"{fig_dir}/rocml-rmse-val-mean-{target}-{res}.png")

        combine_plots_vertically(
            f"{fig_dir}/temp1.png",
            f"{fig_dir}/temp2.png",
            f"{fig_dir}/performance-{target}.png",
            caption1="",
            caption2=""
        )

        # Iterate through all samples
        for sample_id in rocml_model.sample_ids:

            if target in ["rho", "Vp", "Vs"]:
                combine_plots_horizontally(
                    f"{fig_dir}/{model_prefix}-{sample_id}-{target}-targets.png",
                    f"{fig_dir}/{model_prefix}-{sample_id}-{target}-predictions.png",
                    f"{fig_dir}/temp1.png",
                    caption1="a)",
                    caption2="b)"
                )

                combine_plots_horizontally(
                    f"{fig_dir}/temp1.png",
                    f"{fig_dir}/{model_prefix}-{sample_id}-{target}-prem.png",
                    f"{fig_dir}/prem-{sample_id}-{target}.png",
                    caption1="",
                    caption2="c)"
                )

            combine_plots_horizontally(
                f"{fig_dir}/{model_prefix}-{sample_id}-{target}-targets-surf.png",
                f"{fig_dir}/{model_prefix}-{sample_id}-{target}-surf.png",
                f"{fig_dir}/temp1.png",
                caption1="a)",
                caption2="b)"
            )

            combine_plots_horizontally(
                f"{fig_dir}/temp1.png",
                f"{fig_dir}/{model_prefix}-{sample_id}-{target}-diff-surf.png",
                f"{fig_dir}/surf-{sample_id}-{target}.png",
                caption1="",
                caption2="c)"
            )

            combine_plots_horizontally(
                f"{fig_dir}/{model_prefix}-{sample_id}-{target}-targets.png",
                f"{fig_dir}/{model_prefix}-{sample_id}-{target}-predictions.png",
                f"{fig_dir}/temp1.png",
                caption1="a)",
                caption2="b)"
            )

            combine_plots_horizontally(
                f"{fig_dir}/temp1.png",
                f"{fig_dir}/{model_prefix}-{sample_id}-{target}-diff.png",
                f"{fig_dir}/image-{sample_id}-{target}.png",
                caption1="",
                caption2="c)"
            )

    # Clean up directory
    rocml_files = glob.glob(f"{fig_dir}/rocml*.png")
    tmp_files = glob.glob(f"{fig_dir}/temp*.png")
    program_files = glob.glob(f"{fig_dir}/{program[:4]}*.png")

    for file in rocml_files + tmp_files + program_files:
        os.remove(file)

    return None

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ .1.2          Plotting Functions              !!! ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize gfem design !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_gfem_design(P_min=1, P_max=28, T_min=773, T_max=2273,
                                      fig_dir="figs/other", T_mantle1=273, T_mantle2=1773,
                                      grad_mantle1=1, grad_mantle2=0.5, fontsize=12,
                                      figwidth=6.3, figheight=3.54):
    """
    """
    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # T range
    T = np.arange(0, T_max + 728)

    # Olivine --> Ringwoodite Clapeyron slopes
    references_410 = {"[410] Akaogi89": [0.001, 0.002], "[410] Katsura89": [0.0025],
                      "[410] Morishima94": [0.0034, 0.0038]}

    # Ringwoodite --> Bridgmanite + Ferropericlase Clapeyron slopes
    references_660 = {"[660] Ito82": [-0.002], "[660] Ito89 & Hirose02": [-0.0028],
                      "[660] Ito90": [-0.002, -0.006], "[660] Katsura03": [-0.0004, -0.002],
                      "[660] Akaogi07": [-0.0024, -0.0028]}

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = "False"
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = "True"
    plt.rcParams["figure.dpi"] = 330
    plt.rcParams["savefig.bbox"] = "tight"

    # Legend colors
    colormap = plt.cm.get_cmap("tab10")
    colors = [colormap(i) for i in range(9)]

    # Calculate phase boundaries:
    # Olivine --> Ringwoodite
    lines_410 = []
    labels_410 = set()

    for i, (ref, c_values) in enumerate(references_410.items()):
        ref_lines = []

        for j, c in enumerate(c_values):
            P = (T - 1758) * c + 13.4

            ref_lines.append(P)

            label = f"{ref}"
            labels_410.add(label)

        lines_410.append(ref_lines)

    # Ringwoodite --> Bridgmanite + Ferropericlase
    lines_660 = []
    labels_660 = set()

    for i, (ref, c_values) in enumerate(references_660.items()):
        ref_lines = []

        for j, c in enumerate(c_values):
            P = (T - 1883) * c + 23.0

            ref_lines.append(P)

            label = f"{ref}"
            labels_660.add(label)

        lines_660.append(ref_lines)

    # Plotting
    plt.figure()

    # Map labels to colors
    label_color_mapping = {}

    # Olivine --> Ringwoodite
    for i, (ref, ref_lines) in enumerate(zip(references_410.keys(), lines_410)):
        color = colors[i % len(colors)]

        for j, line in enumerate(ref_lines):
            label = f"{ref}" if j == 0 else None

            plt.plot(T[(T >= 1200) & (T <= 2000)], line[(T >= 1200) & (T <= 2000)],
                     color=color, label=label)

            if label not in label_color_mapping:
                label_color_mapping[label] = color

    # Ringwoodite --> Bridgmanite + Ferropericlase
    for j, (ref, ref_lines) in enumerate(zip(references_660.keys(), lines_660)):
        color = colors[j + i + 1 % len(colors)]

        for j, line in enumerate(ref_lines):
            label = f"{ref}" if j == 0 else None

            plt.plot(T[(T >= 1200) & (T <= 2000)], line[(T >= 1200) & (T <= 2000)],
                     color=color, label=label)

            if label not in label_color_mapping:
                label_color_mapping[label] = color

    # Plot shaded rectangle for PT range of training dataset
    fill = plt.fill_between(T, P_min, P_max, where=(T >= T_min) & (T <= T_max),
                            color="gray", alpha=0.2)

    # Calculate mantle geotherms
    geotherm1 = (T - T_mantle1) / (grad_mantle1 * 35)
    geotherm2 = (T - T_mantle2) / (grad_mantle2 * 35)

    # Find boundaries
    T1_Pmax = (P_max * grad_mantle1 * 35) + T_mantle1
    P1_Tmin = (T_min - T_mantle1) / (grad_mantle1 * 35)
    T2_Pmin = (P_min * grad_mantle2 * 35) + T_mantle2
    T2_Pmax = (P_max * grad_mantle2 * 35) + T_mantle2

    # Crop geotherms
    geotherm1_cropped = geotherm1[geotherm1 >= P1_Tmin]
    geotherm1_cropped = geotherm1_cropped[geotherm1_cropped <= P_max]
    geotherm2_cropped = geotherm2[geotherm2 >= P_min]
    geotherm2_cropped = geotherm2_cropped[geotherm2_cropped <= P_max]

    # Crop T vectors
    T_cropped_geotherm1= T[T >= T_min]
    T_cropped_geotherm1 = T_cropped_geotherm1[T_cropped_geotherm1 <= T1_Pmax]
    T_cropped_geotherm2= T[T >= T2_Pmin]
    T_cropped_geotherm2 = T_cropped_geotherm2[T_cropped_geotherm2 <= T2_Pmax]

    # Plot mantle geotherms
    plt.plot(T_cropped_geotherm1, geotherm1_cropped, "-", color="black")
    plt.plot(T_cropped_geotherm2, geotherm2_cropped, "--", color="black")

    # Interpolate the geotherms to have the same length as temperature vectors
    geotherm1_interp = np.interp(T_cropped_geotherm1, T, geotherm1)
    geotherm2_interp = np.interp(T_cropped_geotherm2, T, geotherm2)

    # Define the vertices for the polygon
    vertices = np.vstack(
        (
            np.vstack((T_cropped_geotherm1, geotherm1_interp)).T,
            (T_cropped_geotherm2[-1], geotherm2_interp[-1]),
            np.vstack((T_cropped_geotherm2[::-1], geotherm2_interp[::-1])).T,
            np.array([T_min, P_min]),
            (T_cropped_geotherm1[0], geotherm1_interp[0])
        )
    )

    # Fill the area within the polygon
    plt.fill(vertices[:, 0], vertices[:, 1], facecolor="blue", edgecolor="black", alpha=0.1)

    # Geotherm legend handles
    geotherm1_handle = mlines.Line2D([], [], linestyle="-", color="black",
                                     label="Geotherm 1")
    geotherm2_handle = mlines.Line2D([], [], linestyle="--", color="black",
                                     label="Geotherm 2")

    # Phase boundaries legend handles
    ref_line_handles = [
        mlines.Line2D([], [], color=color, label=label)
        for label, color in label_color_mapping.items() if label
    ]

    # Add geotherms to legend handles
    ref_line_handles.extend([geotherm1_handle, geotherm2_handle])

    db_data_handle = mpatches.Patch(color="gray", alpha=0.2, label="Training Data Range")

    labels_660.add("Training Data Range")
    label_color_mapping["Training Data Range"] = "gray"

    training_data_handle = mpatches.Patch(facecolor="blue", edgecolor="black", alpha=0.1,
                                          label="Mantle Conditions")

    labels_660.add("Mantle Conditions")
    label_color_mapping["Mantle Conditions"] = "gray"

    # Define the desired order of the legend items
    desired_order = ["Training Data Range", "Mantle Conditions", "[410] Akaogi89",
                     "[410] Katsura89", "[410] Morishima94", "[660] Ito82",
                     "[660] Ito89 & Hirose02", "[660] Ito90", "[660] Katsura03",
                     "[660] Akaogi07", "Geotherm 1", "Geotherm 2"]

    # Sort the legend handles based on the desired order
    legend_handles = sorted(ref_line_handles + [db_data_handle, training_data_handle],
                            key=lambda x: desired_order.index(x.get_label()))

    plt.xlabel("Temperature (K)")
    plt.ylabel("Pressure (GPa)")
    plt.title("RocML Traning Dataset Design")
    plt.xlim(T_min - 100, T_max + 100)
    plt.ylim(P_min - 1, P_max + 1)

    # Move the legend outside the plot to the right
    plt.legend(title="", handles=legend_handles, loc="center left",
               bbox_to_anchor=(1.02, 0.5))

    # Adjust the figure size
    fig = plt.gcf()
    fig.set_size_inches(figwidth, figheight)

    # Save the plot to a file
    plt.savefig(f"{fig_dir}/training-dataset-design.png")

    # Close device
    plt.close()

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize gfem efficiency !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_gfem_efficiency(fig_dir="figs/other", filename="gfem-efficiency.png",
                              fontsize=12, figwidth=6.3, figheight=3.54):
    """
    """
    # Data assets dir
    data_dir = "assets/data"

    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Read data
    data = pd.read_csv(f"{data_dir}/gfem-efficiency.csv")

    # Filter out validation dataset
    data = data[data["dataset"] == "train"]

    # Filter out non-benchmark samples
    data = data[data["sample"].isin(["DMM", "NMORB", "PUM", "RE46"])]

    # Arrange data by resolution and sample
    data.sort_values(by=["size", "sample", "program"], inplace=True)

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.dpi"] = 330
    plt.rcParams["savefig.bbox"] = "tight"

    # Create a dictionary to map samples to colors using a colormap
    colormap = plt.cm.get_cmap("tab10")
    sample_colors = {"DMM": colormap(0), "NMORB": colormap(1),
                     "PUM": colormap(2), "RE46": colormap(3)}

    # Group the data by sample
    grouped_data = data.groupby(["sample"])

    # Create a list to store the legend labels and handles
    legend_labels = []
    legend_handles = []

    # Plot the data for MAGEMin lines
    for group, group_data in grouped_data:
        sample_id = group[0]
        color_val = sample_colors[sample_id]

        # Filter out rows with missing time values for mgm column
        mgm_data = group_data[group_data["program"] == "magemin"]
        mgm_x = mgm_data["size"]
        mgm_y = mgm_data["time"]

        # Filter out rows with missing time values for ppx column
        ppx_data = group_data[group_data["program"] == "perplex"]
        ppx_x = ppx_data["size"]
        ppx_y = ppx_data["time"]

        # Plot mgm data points and connect them with lines
        line_mgm, = plt.plot(mgm_x, mgm_y, marker="o", color=color_val,
                             linestyle="-", label=f"[MAGEMin] {sample_id}")

        legend_handles.append(line_mgm)
        legend_labels.append(f"[MAGEMin] {sample_id}")

        # Plot ppx data points and connect them with lines
        line_ppx, = plt.plot(ppx_x, ppx_y, marker="s", color=color_val,
                             linestyle="--", label=f"[Perple_X] {sample_id}")

        legend_handles.append(line_ppx)
        legend_labels.append(f"[Perple_X] {sample_id}")

    # Set labels and title
    plt.xlabel("Number of Minimizations (PT Points)")
    plt.ylabel("Elapsed Time (s)")
    plt.title("Solution Efficiency")
    plt.xscale("log")
    plt.yscale("log")

    # Create the legend with the desired order
    plt.legend(legend_handles, legend_labels, title="",
               bbox_to_anchor=(1.02, 0.5), loc="center left")

    # Adjust the figure size
    fig = plt.gcf()
    fig.set_size_inches(figwidth, figheight)

    # Save the plot to a file if a filename is provided
    if filename:
        plt.savefig(f"{fig_dir}/{filename}")

    else:
        # Print plot
        plt.show()

    # Close device
    plt.close()

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize prem !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_prem(program, sample_id, dataset, res, target, target_unit, results_mgm=None,
                   results_ppx=None, results_ml=None, model=None, geotherm_threshold=0.1,
                   metrics=None, title=None, fig_dir="figs", filename=None, figwidth=6.3,
                   figheight=4.725, fontsize=22):
    """
    """
    # Data asset dir
    data_dir = "assets/data"

    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Read PREM data
    df_prem = pd.read_csv(f"{data_dir}/prem.csv")

    # Extract depth and target values
    target_prem = df_prem[target]
    depth_prem = df_prem["depth"]

    # Transform depth to pressure
    P_prem = depth_prem / 30

    # Initialize geotherms
    P_mgm, P_ppx, P_ml = None, None, None
    P_pum, P_nmorb, P_dmm, P_re46 = None, None, None, None
    target_mgm, target_ppx, target_ml = None, None, None
    target_pum, target_nmorb, target_dmm, target_re46 = None, None, None, None

    # Get benchmark models
    pum_path = f"runs/{program[:4]}_PUM_{dataset[0]}{res}/results.csv"
    nmorb_path = f"runs/{program[:4]}_NMORB_{dataset[0]}{res}/results.csv"
    dmm_path = f"runs/{program[:4]}_DMM_{dataset[0]}{res}/results.csv"
    re46_path = f"runs/{program[:4]}_RE46_{dataset[0]}{res}/results.csv"
    source = "assets/data/benchmark-samples.csv"
    targets = ["rho", "Vp", "Vs"]

    if os.path.exists(pum_path) and sample_id != "PUM":
        pum_model = GFEMModel(program, dataset, "PUM", source, res, 1, 28, 773, 2273, "all",
                              targets, False, 0, False)
        results_pum = pum_model.results
    else:
        results_pum = None
    if os.path.exists(nmorb_path) and sample_id != "NMORB":
        nmorb_model = GFEMModel(program, dataset, "NMORB", source, res, 1, 28, 773, 2273,
                                "all", targets, False, 0, False)
        results_nmorb = nmorb_model.results
    else:
        results_nmorb = None
    if os.path.exists(dmm_path) and sample_id != "DMM":
        dmm_model = GFEMModel(program, dataset, "DMM", source, res, 1, 28, 773, 2273,
                                "all", targets, False, 0, False)
        results_dmm = dmm_model.results
    else:
        results_dmm = None
    if os.path.exists(re46_path) and sample_id != "RE46":
        re46_model = GFEMModel(program, dataset, "RE46", source, res, 1, 28, 773, 2273,
                                "all", targets, False, 0, False)
        results_re46 = re46_model.results
    else:
        results_re46 = None

    # Extract target values along a geotherm
    if results_mgm:
        P_mgm, _, target_mgm = get_geotherm(results_mgm, target, geotherm_threshold)
    if results_ppx:
        P_ppx, _, target_ppx = get_geotherm(results_ppx, target, geotherm_threshold)
    if results_ml:
        P_ml, _, target_ml = get_geotherm(results_ml, target, geotherm_threshold)
    if results_pum:
        P_pum, _, target_pum = get_geotherm(results_pum, target, geotherm_threshold)
    if results_nmorb:
        P_nmorb, _, target_nmorb = get_geotherm(results_nmorb, target, geotherm_threshold)
    if results_dmm:
        P_dmm, _, target_dmm = get_geotherm(results_dmm, target, geotherm_threshold)
    if results_re46:
        P_re46, _, target_re46 = get_geotherm(results_re46, target, geotherm_threshold)

    # Get min and max P from geotherms
    P_min = min(np.nanmin(P) for P in [P_mgm, P_ppx, P_ml, P_pum, P_nmorb, P_dmm, P_re46] if
                P is not None)
    P_max = max(np.nanmax(P) for P in [P_mgm, P_ppx, P_ml, P_pum, P_nmorb, P_dmm, P_re46] if
                P is not None)

    # Create cropping mask for prem
    mask_prem = (P_prem >= P_min) & (P_prem <= P_max)

    # Crop pressure and target values
    P_prem, target_prem = P_prem[mask_prem], target_prem[mask_prem]

    # Crop results
    if results_mgm:
        mask_mgm = (P_mgm >= P_min) & (P_mgm <= P_max)
        P_mgm, target_mgm = P_mgm[mask_mgm], target_mgm[mask_mgm]
    if results_ppx:
        mask_ppx = (P_ppx >= P_min) & (P_ppx <= P_max)
        P_ppx, target_ppx = P_ppx[mask_ppx], target_ppx[mask_ppx]
    if results_ml:
        mask_ml = (P_ml >= P_min) & (P_ml <= P_max)
        P_ml, target_ml = P_ml[mask_ml], target_ml[mask_ml]
    if results_pum:
        mask_pum = (P_pum >= P_min) & (P_pum <= P_max)
        P_pum, target_pum = P_pum[mask_pum], target_pum[mask_pum]
    if results_nmorb:
        mask_nmorb = (P_nmorb >= P_min) & (P_nmorb <= P_max)
        P_nmorb, target_nmorb = P_nmorb[mask_nmorb], target_nmorb[mask_nmorb]
    if results_dmm:
        mask_dmm = (P_dmm >= P_min) & (P_dmm <= P_max)
        P_dmm, target_dmm = P_dmm[mask_dmm], target_dmm[mask_dmm]
    if results_re46:
        mask_re46 = (P_re46 >= P_min) & (P_re46 <= P_max)
        P_re46, target_re46 = P_re46[mask_re46], target_re46[mask_re46]

    # Get min max
    target_min = min(min(np.nanmin(lst) for lst in
                         [target_mgm, target_ppx, target_ml, target_pum, target_nmorb,
                          target_dmm, target_re46] if lst is not None), min(target_prem))
    target_max = max(max(np.nanmax(lst) for lst in
                         [target_mgm, target_ppx, target_ml, target_pum, target_nmorb,
                          target_dmm, target_re46] if lst is not None), max(target_prem))

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = "False"
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = "True"
    plt.rcParams["figure.dpi"] = 330
    plt.rcParams["savefig.bbox"] = "tight"

    # Colormap
    colormap = plt.cm.get_cmap("tab10")

    # Plotting
    fig, ax1 = plt.subplots(figsize=(figwidth, figheight))

    # Plot PREM data on the primary y-axis
    ax1.plot(target_prem, P_prem, "-", linewidth=2, color="black", label="PREM")

#    if results_pum:
#        ax1.plot(target_pum, P_pum, "-", linewidth=1.5, color=colormap(0), label="PUM")
#    if results_nmorb:
#        ax1.plot(target_nmorb, P_nmorb, "-", linewidth=1.5, color=colormap(1), label="NMORB")
#    if results_dmm:
#        ax1.plot(target_dmm, P_dmm, "-", linewidth=1.5, color=colormap(2), label="DMM")
#    if results_re46:
#        ax1.plot(target_re46, P_re46, "-", linewidth=1.5, color=colormap(3), label="RE46")
    if results_mgm:
        ax1.plot(target_mgm, P_mgm, "-", linewidth=3, color=colormap(0), label=sample_id)
    if results_ppx:
        ax1.plot(target_ppx, P_ppx, "-", linewidth=3, color=colormap(1), label=sample_id)
    if results_ml:
        ax1.plot(target_ml, P_ml, "-", linewidth=3, color=colormap(2), label=model)

    if target == "rho":
        target_label = "Density"
    else:
        target_label = target

    ax1.set_xlabel(f"{target_label } ({target_unit})")
    ax1.set_ylabel("P (GPa)")
    ax1.set_xlim(target_min - (target_min * 0.05), target_max + (target_max * 0.05))
    ax1.set_xticks(np.linspace(target_min, target_max, num=4))

    if target in ["Vp", "Vs", "rho"]:
        ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    if metrics is not None:
        # Vertical text spacing
        text_margin_x = 0.04
        text_margin_y = 0.15
        text_spacing_y = 0.1

        # Get metrics
        rmse_mean, r2_mean = metrics

        # Add R-squared and RMSE values as text annotations in the plot
        plt.text(1 - text_margin_x, text_margin_y - (text_spacing_y * 0),
                 f"R$^2$: {r2_mean:.3f}", transform=plt.gca().transAxes,
                 fontsize=fontsize * 0.833, horizontalalignment="right",
                 verticalalignment="bottom")
        plt.text(1 - text_margin_x, text_margin_y - (text_spacing_y * 1),
                 f"RMSE: {rmse_mean:.3f}", transform=plt.gca().transAxes,
                 fontsize=fontsize * 0.833, horizontalalignment="right",
                 verticalalignment="bottom")

    # Convert the primary y-axis data (pressure) to depth
    depth_conversion = lambda P: P * 30
    depth_values = depth_conversion(P_prem)

    # Create the secondary y-axis and plot depth on it
    ax2 = ax1.secondary_yaxis("right", functions=(depth_conversion, depth_conversion))
    ax2.set_yticks([410, 660])
    ax2.set_ylabel("Depth (km)")

    plt.legend()

    if title:
        plt.title(title)

    # Save the plot to a file if a filename is provided
    if filename:
        plt.savefig(f"{fig_dir}/{filename}")

    else:
        # Print plot
        plt.show()

    # Close device
    plt.close()

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize target array  !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_target_array(P, T, target_array, target, title, palette, color_discrete,
                           color_reverse, vmin, vmax, fig_dir, filename, figwidth=6.3,
                           figheight=4.725, fontsize=22):
    """
    """
    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.loc"] = "upper left"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = "False"
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = "True"
    plt.rcParams["figure.dpi"] = 330
    plt.rcParams["savefig.bbox"] = "tight"

    if color_discrete:
        # Discrete color palette
        num_colors = vmax - vmin + 1

        if palette == "viridis":
            if color_reverse:
                pal = plt.cm.get_cmap("viridis_r", num_colors)

            else:
                pal = plt.cm.get_cmap("viridis", num_colors)

        elif palette == "bone":
            if color_reverse:
                pal = plt.cm.get_cmap("bone_r", num_colors)

            else:
                pal = plt.cm.get_cmap("bone", num_colors)

        elif palette == "pink":
            if color_reverse:
                pal = plt.cm.get_cmap("pink_r", num_colors)

            else:
                pal = plt.cm.get_cmap("pink", num_colors)

        elif palette == "seismic":
            if color_reverse:
                pal = plt.cm.get_cmap("seismic_r", num_colors)

            else:
                pal = plt.cm.get_cmap("seismic", num_colors)

        elif palette == "grey":
            if color_reverse:
                pal = plt.cm.get_cmap("Greys_r", num_colors)

            else:
                pal = plt.cm.get_cmap("Greys", num_colors)

        elif palette not in ["viridis", "grey", "bone", "pink", "seismic"]:
            if color_reverse:
                pal = plt.cm.get_cmap("Blues_r", num_colors)

            else:
                pal = plt.cm.get_cmap("Blues", num_colors)

        # Descritize
        color_palette = pal(np.linspace(0, 1, num_colors))
        cmap = ListedColormap(color_palette)

        # Set nan color
        cmap.set_bad(color="white")

        # Plot as a raster using imshow
        fig, ax = plt.subplots(figsize=(figwidth, figheight))

        im = ax.imshow(target_array, extent=[np.nanmin(T), np.nanmax(T), np.nanmin(P),
                                             np.nanmax(P)],
                       aspect="auto", cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_xlabel("T (K)")
        ax.set_ylabel("P (GPa)")
        plt.colorbar(im, ax=ax, ticks=np.arange(vmin, vmax, num_colors // 4), label="")

    else:
        # Continuous color palette
        if palette == "viridis":
            if color_reverse:
                cmap = "viridis_r"

            else:
                cmap = "viridis"

        elif palette == "bone":
            if color_reverse:
                cmap = "bone_r"

            else:
                cmap = "bone"

        elif palette == "pink":
            if color_reverse:
                cmap = "pink_r"

            else:
                cmap = "pink"

        elif palette == "seismic":
            if color_reverse:
                cmap = "seismic_r"

            else:
                cmap = "seismic"

        elif palette == "grey":
            if color_reverse:
                cmap = "Greys_r"

            else:
                cmap = "Greys"

        elif palette not in ["viridis", "grey", "bone", "pink", "seismic"]:
            if color_reverse:
                cmap="Blues_r"

            else:
                cmap="Blues"

        # Adjust diverging colorscale to center on zero
        if palette == "seismic":
            vmin=-np.max(np.abs(target_array[np.logical_not(np.isnan(target_array))]))
            vmax=np.max(np.abs(target_array[np.logical_not(np.isnan(target_array))]))

        else:
            vmin, vmax = vmin, vmax

            # Adjust vmin close to zero
            if vmin <= 1e-4:
                vmin = 0

            # Set melt fraction to 0–100 %
            if target == "melt":
                vmin, vmax = 0, 100

        # Set nan color
        cmap = plt.cm.get_cmap(cmap)
        cmap.set_bad(color="white")

        # Plot as a raster using imshow
        fig, ax = plt.subplots()

        im = ax.imshow(target_array, extent=[np.nanmin(T), np.nanmax(T), np.nanmin(P),
                                             np.nanmax(P)],
                       aspect="auto", cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_xlabel("T (K)")
        ax.set_ylabel("P (GPa)")

        # Diverging colorbar
        if palette == "seismic":
            cbar = plt.colorbar(im, ax=ax, ticks=[vmin, 0, vmax], label="")

        # Continuous colorbar
        else:
            cbar = plt.colorbar(im, ax=ax, ticks=np.linspace(vmin, vmax, num=4), label="")

        # Set colorbar limits and number formatting
        if target == "rho":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        elif target == "Vp":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2g"))
        elif target == "Vs":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        elif target == "melt":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
        elif target == "assemblage":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
        elif target == "variance":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))

    # Add title
    if title:
        plt.title(title)

    # Save the plot to a file if a filename is provided
    if filename:
        plt.savefig(f"{fig_dir}/{filename}")

    else:
        # Print plot
        plt.show()

    # Close device
    plt.close()

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize target surf !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_target_surf(P, T, target_array, target, title, palette, color_discrete,
                          color_reverse, vmin, vmax, fig_dir, filename, figwidth=6.3,
                          figheight=4.725, fontsize=22):
    """
    """
    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.loc"] = "upper left"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = "False"
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = "True"
    plt.rcParams["figure.dpi"] = 330
    plt.rcParams["savefig.bbox"] = "tight"

    if color_discrete:
        # Discrete color palette
        num_colors = vmax - vmin + 1

        if palette == "viridis":
            if color_reverse:
                pal = plt.cm.get_cmap("viridis_r", num_colors)

            else:
                pal = plt.cm.get_cmap("viridis", num_colors)

        elif palette == "bone":
            if color_reverse:
                pal = plt.cm.get_cmap("bone_r", num_colors)

            else:
                pal = plt.cm.get_cmap("bone", num_colors)

        elif palette == "pink":
            if color_reverse:
                pal = plt.cm.get_cmap("pink_r", num_colors)

            else:
                pal = plt.cm.get_cmap("pink", num_colors)

        elif palette == "seismic":
            if color_reverse:
                pal = plt.cm.get_cmap("seismic_r", num_colors)

            else:
                pal = plt.cm.get_cmap("seismic", num_colors)

        elif palette == "grey":
            if color_reverse:
                pal = plt.cm.get_cmap("Greys_r", num_colors)

            else:
                pal = plt.cm.get_cmap("Greys", num_colors)

        elif palette not in ["viridis", "grey", "bone", "pink", "seismic"]:
            if color_reverse:
                pal = plt.cm.get_cmap("Blues_r", num_colors)

            else:
                pal = plt.cm.get_cmap("Blues", num_colors)

        # Descritize
        color_palette = pal(np.linspace(0, 1, num_colors))
        cmap = ListedColormap(color_palette)

        # Set nan color
        cmap.set_bad(color="white")

        # 3D surface
        fig = plt.figure(figsize=(figwidth, figheight), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(T, P, target_array, cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_xlabel("T (K)", labelpad=18)
        ax.set_ylabel("P (GPa)", labelpad=18)
        ax.set_zlabel("")
        ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        plt.tick_params(axis="x", which="major")
        plt.tick_params(axis="y", which="major")
        plt.title(title, y=0.95)
        ax.view_init(20, -145)
        ax.set_box_aspect((1.5, 1.5, 1), zoom=1)
        ax.set_facecolor("white")
        cbar = fig.colorbar(surf, ax=ax, ticks=np.arange(vmin, vmax, num_colors // 4),
                            label="", shrink=0.6)
        cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
        cbar.ax.set_ylim(vmax, vmin)

    else:
        # Continuous color palette
        if palette == "viridis":
            if color_reverse:
                cmap = "viridis_r"

            else:
                cmap = "viridis"

        elif palette == "bone":
            if color_reverse:
                cmap = "bone_r"

            else:
                cmap = "bone"

        elif palette == "pink":
            if color_reverse:
                cmap = "pink_r"

            else:
                cmap = "pink"

        elif palette == "seismic":
            if color_reverse:
                cmap = "seismic_r"

            else:
                cmap = "seismic"

        elif palette == "grey":
            if color_reverse:
                cmap = "Greys_r"

            else:
                cmap = "Greys"

        elif palette not in ["viridis", "grey", "bone", "pink", "seismic"]:
            if color_reverse:
                cmap="Blues_r"

            else:
                cmap="Blues"

        # Adjust diverging colorscale to center on zero
        if palette == "seismic":
            vmin=-np.max(np.abs(target_array[np.logical_not(np.isnan(target_array))]))
            vmax=np.max(np.abs(target_array[np.logical_not(np.isnan(target_array))]))

        else:
            vmin, vmax = vmin, vmax

            # Adjust vmin close to zero
            if vmin <= 1e-4:
                vmin = 0

            # Set melt fraction to 0–100 %
            if target == "melt":
                vmin, vmax = 0, 100

        # Set nan color
        cmap = plt.cm.get_cmap(cmap)
        cmap.set_bad(color="white")

        # 3D surface
        fig = plt.figure(figsize=(figwidth, figheight), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(T, P, target_array, cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_xlabel("T (K)", labelpad=18)
        ax.set_ylabel("P (GPa)", labelpad=18)
        ax.set_zlabel("")
        ax.set_zlim(vmin - (vmin * 0.05), vmax + (vmax * 0.05))
        plt.tick_params(axis="x", which="major")
        plt.tick_params(axis="y", which="major")
        plt.title(title, y=0.95)
        ax.view_init(20, -145)
        ax.set_box_aspect((1.5, 1.5, 1), zoom=1)
        ax.set_facecolor("white")

        # Diverging colorbar
        if palette == "seismic":
            cbar = fig.colorbar(surf, ax=ax, ticks=[vmin, 0, vmax], label="", shrink=0.6)

        # Continous colorbar
        else:
            cbar = fig.colorbar(surf, ax=ax, ticks=np.linspace(vmin, vmax, num=4),
                                label="", shrink=0.6)

        # Set z and colorbar limits and number formatting
        if target == "rho":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
            ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        elif target == "Vp":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2g"))
            ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.2g"))
        elif target == "Vs":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
            ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        elif target == "melt":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
            ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
        elif target == "assemblage":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
            ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
        elif target == "variance":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
            ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))

        cbar.ax.set_ylim(vmin, vmax)

    # Save fig
    plt.savefig(f"{fig_dir}/{filename}")

    # Close fig
    plt.close()

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize gfem !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_gfem(gfem_models, edges=True, palette="bone", verbose=1):
    """
    """
    for model in [m if m.dataset == "train" else None for m in gfem_models]:

        # Check for model
        if model is None:
            if verbose >= 2:
                print("No model to visualize!")

            return None

        # Get model data
        program = model.program
        sample_id = model.sample_id
        model_prefix = model.model_prefix
        res = model.res
        dataset = model.dataset
        targets = model.targets
        mask_geotherm = model.mask_geotherm
        results = model.results
        P, T = results["P"], results["T"]
        target_array = model.target_array
        fig_dir = model.fig_dir
        verbose = model.verbose

        if program == "magemin":
            program_title = "MAGEMin"

        elif program == "perplex":
            program_title = "Perple_X"

        print(f"Visualizing {model_prefix} [{program}] ...")

        for i, target in enumerate(targets):
            # Reshape targets into square array
            square_target = target_array[:, i].reshape(res + 1, res + 1)

            # Use discrete colorscale
            if target in ["assemblage", "variance"]:
                color_discrete = True

            else:
                color_discrete = False

            # Reverse color scale
            if palette in ["grey"]:
                if target in ["variance"]:
                    color_reverse = True

                else:
                    color_reverse = False

            else:
                if target in ["variance"]:
                    color_reverse = False

                else:
                    color_reverse = True

            # Set colorbar limits for better comparisons
            if not color_discrete:
                vmin=np.min(square_target[np.logical_not(np.isnan(square_target))])
                vmax=np.max(square_target[np.logical_not(np.isnan(square_target))])

            else:
                vmin = int(np.nanmin(np.unique(square_target)))
                vmax = int(np.nanmax(np.unique(square_target)))

            # Rename target
            target_rename = target.replace("_", "-")

            # Print filepath
            if verbose >= 2:
                print(f"Saving figure: {program}-{sample_id}-{dataset}-{target_rename}.png")

            # Plot targets
            visualize_target_array(P, T, square_target, target, program_title, palette,
                                   color_discrete, color_reverse, vmin, vmax, fig_dir,
                                   f"{program}-{sample_id}-{dataset}-{target_rename}.png")
            if edges:
                original_image = square_target.copy()

                # Apply Sobel edge detection
                edges_x = cv2.Sobel(original_image, cv2.CV_64F, 1, 0, ksize=3)
                edges_y = cv2.Sobel(original_image, cv2.CV_64F, 0, 1, ksize=3)

                # Calculate the magnitude of the gradient
                magnitude = np.sqrt(edges_x**2 + edges_y**2) / np.nanmax(original_image)

                if not color_discrete:
                    vmin_mag = np.min(magnitude[np.logical_not(np.isnan(magnitude))])
                    vmax_mag = np.max(magnitude[np.logical_not(np.isnan(magnitude))])

                else:
                    vmin_mag = int(np.nanmin(np.unique(magnitude)))
                    vmax_mag = int(np.nanmax(np.unique(magnitude)))

                visualize_target_array(P, T, magnitude, target, "Gradient", palette,
                                       color_discrete, color_reverse, vmin_mag, vmax_mag,
                                       fig_dir, f"grad-{program}-{sample_id}-{dataset}-"
                                       f"{target_rename}.png")

            # Set geotherm threshold for extracting depth profiles
            if res <= 8:
                geotherm_threshold = 4

            elif res <= 16:
                geotherm_threshold = 2

            elif res <= 32:
                geotherm_threshold = 1

            elif res <= 64:
                geotherm_threshold = 0.5

            elif res <= 128:
                geotherm_threshold = 0.25

            else:
                geotherm_threshold = 0.125

            # Plot PREM comparisons
            if target == "rho":
                # Print filepath
                if verbose >= 2:
                    print(f"Saving figure: prem-{sample_id}-{dataset}-{target_rename}.png")

                if program == "magemin":
                    results_mgm = results
                    results_ppx = None
                    visualize_prem(program, sample_id, dataset, res, target, "g/cm$^3$",
                                   results_mgm, results_ppx,
                                   geotherm_threshold=geotherm_threshold,
                                   title="PREM Comparison", fig_dir=fig_dir,
                                   filename=f"prem-{sample_id}-{dataset}-{target_rename}.png")

                elif program == "perplex":
                    results_mgm = None
                    results_ppx = results
                    visualize_prem(program, sample_id, dataset, res, target, "g/cm$^3$",
                                   results_mgm, results_ppx,
                                   geotherm_threshold=geotherm_threshold,
                                   title="PREM Comparison", fig_dir=fig_dir,
                                   filename=f"prem-{sample_id}-{dataset}-{target_rename}.png")

            if target in ["Vp", "Vs"]:
                # Print filepath
                if verbose >= 2:
                    print(f"Saving figure: prem-{sample_id}-{dataset}-{target_rename}.png")

                if program == "magemin":
                    results_mgm = results
                    results_ppx = None
                    visualize_prem(program, sample_id, dataset, res, target, "km/s",
                                   results_mgm, results_ppx,
                                   geotherm_threshold=geotherm_threshold,
                                   title="PREM Comparison", fig_dir=fig_dir,
                                   filename=f"prem-{sample_id}-{dataset}-{target_rename}.png")

                elif program == "perplex":
                    results_mgm = None
                    results_ppx = results
                    visualize_prem(program, sample_id, dataset, res, target, "km/s",
                                   results_mgm, results_ppx,
                                   geotherm_threshold=geotherm_threshold,
                                   title="PREM Comparison", fig_dir=fig_dir,
                                   filename=f"prem-{sample_id}-{dataset}-{target_rename}.png")

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize gfem diff !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_gfem_diff(gfem_models, palette="bone", verbose=1):
    """
    """
    # Parse models
    magemin_models = [m if m.program == "magemin" and m.dataset == "train" else
                      None for m in gfem_models]
    perplex_models = [m if m.program == "perplex" and m.dataset == "train" else
                      None for m in gfem_models]

    # Iterate through models
    for magemin_model, perplex_model in zip(magemin_models, perplex_models):
        # Check for model
        if magemin_model is None or perplex_model is None:
            if verbose >= 2:
                print("No models to visualize!")

            break

        # Get model data
        if magemin_model.sample_id == perplex_model.sample_id:
            sample_id = magemin_model.sample_id
        else:
            raise ValueError("Model samples are not the same!")
        if magemin_model.res == perplex_model.res:
            res = magemin_model.res
        else:
            raise ValueError("Model resolutions are not the same!")
        if magemin_model.dataset == perplex_model.dataset:
            dataset = magemin_model.dataset
        else:
            raise ValueError("Model datasets are not the same!")
        if magemin_model.targets == perplex_model.targets:
            targets = magemin_model.targets
        else:
            raise ValueError("Model datasets are not the same!")
        if magemin_model.mask_geotherm == perplex_model.mask_geotherm:
            mask_geotherm = magemin_model.mask_geotherm
        else:
            raise ValueError("Model geotherm masks are not the same!")
        if magemin_model.fig_dir == perplex_model.fig_dir:
            fig_dir = magemin_model.fig_dir
        else:
            raise ValueError("Model fig dir are not the same!")
        if magemin_model.verbose == perplex_model.verbose:
            verbose = magemin_model.verbose
        else:
            raise ValueError("Model verbosity settings are not the same!")

        results_mgm, results_ppx = magemin_model.results, perplex_model.results
        P_mgm, T_mgm = results_mgm["P"], results_mgm["T"]
        P_ppx, T_ppx = results_ppx["P"], results_ppx["T"]
        target_array_mgm = magemin_model.target_array
        target_array_ppx = perplex_model.target_array

        for i, target in enumerate(targets):
            # Reshape targets into square array
            square_array_mgm = target_array_mgm[:, i].reshape(res + 1, res + 1)
            square_array_ppx = target_array_ppx[:, i].reshape(res + 1, res + 1)

            # Use discrete colorscale
            if target in ["assemblage", "variance"]:
                color_discrete = True

            else:
                color_discrete = False

            # Reverse color scale
            if palette in ["grey"]:
                if target in ["variance"]:
                    color_reverse = True

                else:
                    color_reverse = False

            else:
                if target in ["variance"]:
                    color_reverse = False

                else:
                    color_reverse = True

            # Set colorbar limits for better comparisons
            if not color_discrete:
                vmin_mgm=np.min(square_array_mgm[np.logical_not(np.isnan(square_array_mgm))])
                vmax_mgm=np.max(square_array_mgm[np.logical_not(np.isnan(square_array_mgm))])
                vmin_ppx=np.min(square_array_ppx[np.logical_not(np.isnan(square_array_ppx))])
                vmax_ppx=np.max(square_array_ppx[np.logical_not(np.isnan(square_array_ppx))])

                vmin = min(vmin_mgm, vmin_ppx)
                vmax = max(vmax_mgm, vmax_ppx)

            else:
                num_colors_mgm = len(np.unique(square_array_mgm))
                num_colors_ppx = len(np.unique(square_array_ppx))

                vmin = 1
                vmax = max(num_colors_mgm, num_colors_ppx) + 1

            if not color_discrete:
                # Define a filter to ignore the specific warning
                warnings.filterwarnings("ignore", message="invalid value encountered in divide")

                # Create nan mask
                mask = ~np.isnan(square_array_mgm) & ~np.isnan(square_array_ppx)

                # Compute normalized diff
                diff = square_array_mgm - square_array_ppx

                # Add nans to match original target arrays
                diff[~mask] = np.nan

                # Rename target
                target_rename = target.replace('_', '-')

                # Print filepath
                if verbose >= 2:
                    print(f"Saving figure: diff-{sample_id}-{dataset}-{target_rename}.png")

                # Plot target array normalized diff mgm-ppx
                visualize_target_array(P_ppx, T_ppx, diff, target, "Residuals", "seismic",
                                       color_discrete, False, vmin, vmax, fig_dir,
                                       f"diff-{sample_id}-{dataset}-{target_rename}.png")

                # Set geotherm threshold for extracting depth profiles
                if res <= 8:
                    geotherm_threshold = 4

                elif res <= 16:
                    geotherm_threshold = 2

                elif res <= 32:
                    geotherm_threshold = 1

                elif res <= 64:
                    geotherm_threshold = 0.5

                elif res <= 128:
                    geotherm_threshold = 0.25

                else:
                    geotherm_threshold = 0.125

                # Plot PREM comparisons
                if target == "rho":
                    # Print filepath
                    if verbose >= 2:
                        print(f"Saving figure: prem-{sample_id}-{dataset}-{target_rename}.png")

                    visualize_prem(program, sample_id, dataset, res, target, "g/cm$^3$",
                                   results_mgm, results_ppx,
                                   geotherm_threshold=geotherm_threshold,
                                   title="PREM Comparison", fig_dir=fig_dir,
                                   filename=f"prem-{sample_id}-{dataset}-{target_rename}.png")

                if target in ["Vp", "Vs"]:
                    # Print filepath
                    if verbose >= 2:
                        print(f"Saving figure: prem-{sample_id}-{dataset}-{target_rename}.png")

                    visualize_prem(program, sample_id, dataset, res, target, "km/s", results_mgm,
                                   results_ppx, geotherm_threshold=geotherm_threshold,
                                   title="PREM Comparison", fig_dir=fig_dir,
                                   filename=f"prem-{sample_id}-{dataset}-{target_rename}.png")

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize rocml model !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_rocml_model(rocml_model, figwidth=6.3, figheight=4.725, fontsize=22):
    """
    """
    # Get ml model attributes
    program = rocml_model.program
    if program == "perplex":
        program_label = "Perple_X"
    elif program == "magemin":
        program_label = "MAGEMin"
    model_label_full = rocml_model.ml_model_label_full
    model_label = rocml_model.ml_model_label
    model_prefix = rocml_model.model_prefix
    sample_ids = rocml_model.sample_ids
    feature_arrays = rocml_model.feature_square
    target_arrays = rocml_model.target_square
    pred_arrays = rocml_model.prediction_square
    cv_info = rocml_model.cv_info
    targets = rocml_model.targets
    fig_dir = rocml_model.fig_dir
    palette = rocml_model.palette
    n_oxides = feature_arrays.shape[-1] - 2
    n_models = feature_arrays.shape[0]
    w = feature_arrays.shape[1]
    verbose = rocml_model.verbose

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.loc"] = "upper left"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = "False"
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = "True"
    plt.rcParams["figure.constrained_layout.use"] = "True"
    plt.rcParams["figure.dpi"] = 330
    plt.rcParams["savefig.bbox"] = "tight"

    for s, sample_id in enumerate(sample_ids):
        if verbose >= 1:
            print(f"Visualizing {sample_id} ...")
        # Slice arrays
        feature_array = feature_arrays[s, :, :, :]
        target_array = target_arrays[s, :, :, :]
        pred_array = pred_arrays[s, :, :, :]

        # Plotting variables
        units = []
        units_labels = []
        vmin = []
        vmax = []

        # Get units and colorbar limits
        for i, target in enumerate(targets):
            # Units
            if target == "rho":
                units.append("g/cm$^3$")

            elif target in ["Vp", "Vs"]:
                units.append("km/s")

            elif target == "melt":
                units.append("%")

            else:
                units.append("")

            # Get min max of target array
            vmin.append(np.nanmin(target_array[:, :, i]))
            vmax.append(np.nanmax(target_array[:, :, i]))

        units_labels = [f"({unit})" if unit is not None else "" for unit in units]

        # Colormap
        cmap = plt.cm.get_cmap("bone_r")
        cmap.set_bad(color="white")

        for i, target in enumerate(targets):
            # Rename target
            target_rename = target.replace("_", "-")

            # Create nan mask for validation set targets
            mask = np.isnan(target_array[:, :, i])

            # Match nans between validation set predictions and original targets
            pred_array[:, :, i][mask] = np.nan

            # Compute normalized diff
            diff = target_array[:,:,i] - pred_array[:,:,i]

            # Make nans consistent
            diff[mask] = np.nan

            # Plot training data distribution and ML model predictions
            colormap = plt.cm.get_cmap("tab10")

            # Reverse color scale
            if palette in ["grey"]:
                color_reverse = False

            else:
                color_reverse = True

            # Plot target array 2d
            P = feature_array[:, :, 0 + n_oxides]
            T = feature_array[:, :, 1 + n_oxides]
            t = target_array[:, :, i]
            p = pred_array[:, :, i]

            visualize_target_array(P, T, t, target, program_label, palette, False,
                                   color_reverse, vmin[i], vmax[i], fig_dir,
                                   f"{model_prefix}-{sample_id}-{target_rename}-targets.png")
            # Plot target array 3d
            visualize_target_surf(P, T, t, target, program_label, palette, False,
                                  color_reverse, vmin[i], vmax[i], fig_dir,
                                  f"{model_prefix}-{sample_id}-{target_rename}"
                                  f"-targets-surf.png")

            # Plot ML model predictions array 2d
            visualize_target_array(P, T, p, target, model_label_full, palette, False,
                                   color_reverse, vmin[i], vmax[i], fig_dir,
                                   f"{model_prefix}-{sample_id}-{target_rename}"
                                   f"-predictions.png")

            # Plot ML model predictions array 3d
            visualize_target_surf(P, T, p, target, model_label_full, palette, False,
                                  color_reverse, vmin[i], vmax[i], fig_dir,
                                  f"{model_prefix}-{sample_id}-{target_rename}-surf.png")

            # Plot PT normalized diff targets vs. ML model predictions 2d
            visualize_target_array(P, T, diff, target, "Residuals", "seismic", False, False,
                                   vmin[i], vmax[i], fig_dir, f"{model_prefix}-{sample_id}"
                                   f"-{target_rename}-diff.png")

            # Plot PT normalized diff targets vs. ML model predictions 3d
            visualize_target_surf(P, T, diff, target, "Residuals", "seismic", False, False,
                                  vmin[i], vmax[i], fig_dir, f"{model_prefix}-{sample_id}"
                                  f"-{target_rename}-diff-surf.png")

            # Reshape results and transform units for MAGEMin
            if program == "magemin":
                results_mgm = {"P": P.flatten().tolist(), "T": T.flatten().tolist(),
                               target: t.flatten().tolist()}

                results_ppx = None

            # Reshape results and transform units for Perple_X
            elif program == "perplex":
                results_ppx = {"P": P.flatten().tolist(), "T": T.flatten().tolist(),
                               target: t.flatten().tolist()}

                results_mgm = None

            # Reshape results and transform units for ML model
            results_rocml = {"P": P.flatten().tolist(), "T": T.flatten().tolist(),
                             target: p.flatten().tolist()}

            # Get relevant metrics for PREM plot
            rmse = cv_info[f"rmse_val_mean_{target}"]
            r2 = cv_info[f"r2_val_mean_{target}"]

            metrics = [rmse[0], r2[0]]

            # Set geotherm threshold for extracting depth profiles
            res = w - 1

            if res <= 8:
                geotherm_threshold = 4

            elif res <= 16:
                geotherm_threshold = 2

            elif res <= 32:
                geotherm_threshold = 1

            elif res <= 64:
                geotherm_threshold = 0.5

            elif res <= 128:
                geotherm_threshold = 0.25

            else:
                geotherm_threshold = 0.125

            # Plot PREM comparisons
            if target == "rho":
                visualize_prem(program, sample_id, "train", res, target, "g/cm$^3$",
                               results_mgm, results_ppx, results_rocml, model_label,
                               geotherm_threshold, metrics, title=model_label_full,
                               fig_dir=fig_dir, filename=f"{model_prefix}-{sample_id}"
                               f"-{target_rename}-prem.png")

            if target in ["Vp", "Vs"]:
                visualize_prem(program, sample_id, "train", res, target, "km/s", results_mgm,
                               results_ppx, results_rocml, model_label, geotherm_threshold,
                               metrics, title=model_label_full, fig_dir=fig_dir,
                               filename=f"{model_prefix}-{sample_id}-{target_rename}"
                               f"-prem.png")
    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize rocml performance !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_rocml_performance(target, res, fig_dir, filename, fontsize=22,
                                figwidth=6.3, figheight=4.725):
    """
    """
    # Data assets dir
    data_dir = "assets/data"

    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Read regression data
    data = pd.read_csv(f"{data_dir}/rocml-performance.csv")

    # Summarize data
    numeric_columns = data.select_dtypes(include=[float, int]).columns
    summary_df = data.groupby("model")[numeric_columns].mean().reset_index()

    # Get MAGEMin and Perple_X benchmark times
    benchmark_times = pd.read_csv(f"{data_dir}/gfem-efficiency.csv")

    filtered_times = benchmark_times[(benchmark_times["sample"] == "PUM") &
                                     (benchmark_times["size"] == res**2)]

    time_mgm = np.mean(
        filtered_times[filtered_times["program"] == "magemin"]["time"].values /
        filtered_times[filtered_times["program"] == "magemin"]["size"].values *
        1000
    )
    time_ppx = np.mean(
        filtered_times[filtered_times["program"] == "perplex"]["time"].values /
        filtered_times[filtered_times["program"] == "perplex"]["size"].values *
        1000
    )

    # Define the metrics to plot
    metrics = ["training_time_mean", "inference_time_mean",
               f"rmse_test_mean_{target}", f"rmse_val_mean_{target}"]
    metric_names = ["Training Efficiency", "Prediction Efficiency",
                    "Training Error", "Validation Error"]

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.dpi"] = 330
    plt.rcParams["savefig.bbox"] = "tight"

    # Define the colors for the programs
    colormap = plt.cm.get_cmap("tab10")
    models = ["KN", "DT", "RF", "NN1", "NN2", "NN3"]

    # Create a dictionary to map each model to a specific color
    color_mapping = {"KN": colormap(2), "DT": colormap(0), "RF": colormap(4),
                     "NN1": colormap(1), "NN2": colormap(3), "NN3": colormap(5)}

    # Get the corresponding colors for each model
    colors = [color_mapping[model] for model in models]

    # Define units
    if target == "rho":
        unit = "(kg/m$^3$)"
    elif target in ["Vp", "Vs"]:
        unit = "(km/s)"
    elif target == "melt":
        unit = "(%)"
    else:
        unit = ""

    # Loop through each metric and create a subplot
    for i, metric in enumerate(metrics):
        # Create the facet barplot
        plt.figure(figsize=(figwidth, figheight))

        # Define the offset for side-by-side bars
        bar_width = 0.45

        # Get order of sorted bars
        order = summary_df[metric].sort_values().index
        models_order = summary_df.loc[order]["model"].tolist()

        # Bar positions
        x_positions = np.arange(len(summary_df[metric]))

        # Show MAGEMin and Perple_X compute times
        if metric == "inference_time_mean":
            mgm_line = plt.axhline(time_mgm, color="black", linestyle="-", label="MAGEMin")
            ppx_line = plt.axhline(time_ppx, color="black", linestyle="--", label="Perple_X")

        # Plot the bars for each program
        bars = plt.bar(x_positions * bar_width, summary_df.loc[order][metric],
                       edgecolor="black", width=bar_width,
                       color=[color_mapping[model] for model in models_order],
                       label=models_order if i == 1 else "")

        plt.gca().set_xticks([])
        plt.gca().set_xticklabels([])

        # Plot titles
        if metric == "training_time_mean":
            plt.title(f"{metric_names[i]}")
            plt.ylabel("Elapsed Time (ms)")
            plt.yscale("log")

        elif metric == "inference_time_mean":
            plt.title(f"{metric_names[i]}")
            plt.ylabel("Elapsed Time (ms)")
            plt.yscale("log")
            handles = [mgm_line, ppx_line]
            labels = [handle.get_label() for handle in handles]
            legend = plt.legend(fontsize="x-small")
            legend.set_bbox_to_anchor((0.44, 0.89))

        elif metric == f"rmse_test_mean_{target}":
            # Calculate limits
            max_error = np.max(np.concatenate([
                summary_df.loc[order][f"rmse_test_std_{target}"].values * 2,
                summary_df.loc[order][f"rmse_val_std_{target}"].values * 2
            ]))

            max_mean = np.max(np.concatenate([
                summary_df.loc[order][f"rmse_test_mean_{target}"].values,
                summary_df.loc[order][f"rmse_val_mean_{target}"].values
            ]))

            vmax = max_mean + max_error + ((max_mean + max_error) * 0.05)

            plt.errorbar(x_positions * bar_width,
                         summary_df.loc[order][f"rmse_test_mean_{target}"],
                         yerr=summary_df.loc[order][f"rmse_test_std_{target}"] * 2,
                         fmt="none", capsize=5, color="black", linewidth=2)

            plt.title(f"{metric_names[i]}")
            plt.ylabel(f"RMSE {unit}")
            plt.ylim(0, vmax)

            if target != "melt":
                plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

            else:
                plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

        elif metric == f"rmse_val_mean_{target}":
            # Calculate limits
            max_error = np.max(np.concatenate([
                summary_df.loc[order][f"rmse_test_std_{target}"].values * 2,
                summary_df.loc[order][f"rmse_val_std_{target}"].values * 2
            ]))

            max_mean = np.max(np.concatenate([
                summary_df.loc[order][f"rmse_test_mean_{target}"].values,
                summary_df.loc[order][f"rmse_val_mean_{target}"].values
            ]))

            vmax = max_mean + max_error + ((max_mean + max_error) * 0.05)

            plt.errorbar(x_positions * bar_width,
                         summary_df.loc[order][f"rmse_val_mean_{target}"],
                         yerr=summary_df.loc[order][f"rmse_val_std_{target}"] * 2,
                         fmt="none", capsize=5, color="black", linewidth=2)

            plt.title(f"{metric_names[i]}")
            plt.ylabel(f"RMSE {unit}")
            plt.ylim(0, vmax)

            if target != "melt":
                plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

            else:
                plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

        # Save the plot to a file if a filename is provided
        if filename:
            plt.savefig(f"{fig_dir}/{filename}-{metric.replace('_', '-')}-{res}.png")

        else:
            # Print plot
            plt.show()

        # Close device
        plt.close()

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize pca loadings !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_pca_loadings(mixing_array, fig_dir="figs/other", filename="earthchem",
                           figwidth=6.3, figheight=6.3, fontsize=22):
    """
    """
    # Get mixing array attributes
    pca = mixing_array.pca_model
    oxides = mixing_array.oxides
    n_pca_components = mixing_array.n_pca_components
    data = mixing_array.earthchem_pca

    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.loc"] = "upper left"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = "False"
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = "True"
    plt.rcParams["figure.dpi"] = 330
    plt.rcParams["savefig.bbox"] = "tight"

    loadings = pd.DataFrame((pca.components_.T * np.sqrt(pca.explained_variance_)).T,
                            columns=oxides)

    # Colormap
    colormap = plt.cm.get_cmap("tab10")

    # Plot PCA loadings
    fig = plt.figure(figsize=(figheight, figwidth))

    for i in [0, 1]:
        ax = fig.add_subplot(2, 1, i+1)

        ax.bar(oxides, loadings.iloc[i], color=colormap(i))
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        ax.set_xlabel("")
        ax.set_ylim([-1, 1])
        ax.set_ylabel("")
        ax.xaxis.set_major_locator(ticker.FixedLocator(range(len(oxides))))
        ax.set_xticklabels(oxides, rotation=90)
        plt.title(f"PC{i+1} Loadings")

        if i == 0:
            ax.set_xticks([])

    # Save the plot to a file if a filename is provided
    if filename:
        plt.savefig(f"{fig_dir}/{filename}-pca-loadings.png")

    else:
        # Print plot
        plt.show()

    # Close device
    plt.close()

    for n in range(n_pca_components - 1):

        fig = plt.figure(figsize=(figwidth, figheight))
        ax = fig.add_subplot(111)

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

        for i, comp in enumerate(["ultramafic", "mafic"]):
            indices = data.loc[data["COMPOSITION"] == comp].index

            scatter = ax.scatter(data.loc[indices, f"PC{n + 1}"],
                                 data.loc[indices, f"PC{n + 2}"], edgecolors="none",
                                 color=colormap(i), marker=".", label=comp)

        for oxide in oxides:
            ax.arrow(0, 0, loadings.at[n, oxide] * 2, loadings.at[n + 1, oxide] * 2,
                     width=0.02, head_width=0.14, color="black")
            ax.text((loadings.at[n, oxide] * 2) + (loadings.at[n, oxide] * 1),
                    (loadings.at[n + 1, oxide] * 2) + (loadings.at[n + 1, oxide] * 1),
                    oxide, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8,
                                     pad=0.1),
                    fontsize=fontsize * 0.579, color="black", ha = "center", va = "center")

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, columnspacing=0,
                  markerscale=3, handletextpad=-0.5, fontsize=fontsize * 0.694)
        ax.set_xlabel(f"PC{n + 1}")
        ax.set_ylabel(f"PC{n + 2}")
        plt.title("Earthchem Samples")

        # Save the plot to a file if a filename is provided
        if filename:
            plt.savefig(f"{fig_dir}/{filename}-pca{n + 1}{n + 2}.png")

        else:
            # Print plot
            plt.show()

        # Close device
        plt.close()

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize kmeans clusters !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_kmeans_clusters(mixing_array, fig_dir="figs/other", filename="earthchem",
                              figwidth=6.3, figheight=6.3, fontsize=22):
    """
    """
    # Get mixing array attributes
    res = mixing_array.res
    pca = mixing_array.pca_model
    n_pca_components = mixing_array.n_pca_components
    k_pca_clusters = mixing_array.k_pca_clusters
    data = mixing_array.earthchem_cluster
    kmeans = mixing_array.kmeans_model

    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.loc"] = "upper left"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = "False"
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = "True"
    plt.rcParams["figure.dpi"] = 330
    plt.rcParams["savefig.bbox"] = "tight"

    # Colormap
    colormap = plt.cm.get_cmap("tab10")

    # Get centroids
    centroids = kmeans.cluster_centers_
    original_centroids = pca.inverse_transform(centroids)

    # Plot PCA results
    for n in range(n_pca_components - 1):

        fig = plt.figure(figsize=(figwidth, figheight))
        ax = fig.add_subplot(111)

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

        for c in range(k_pca_clusters):
            # Get datapoints indices for each cluster
            indices = data.loc[data["CLUSTER"] == c].index

            scatter = ax.scatter(data.loc[indices, f"PC{n + 1}"],
                                 data.loc[indices, f"PC{n + 2}"], edgecolors="none",
                                 color=colormap(c + 4), marker=".", alpha=0.3)

            clusters = ax.scatter(centroids[c, n], centroids[c, n+1], edgecolor="black",
                                  color=colormap(c + 4), label=f"cluster {c + 1}",
                                  marker="s", s=100)

            # Calculate mixing lines between cluster centroids
            if k_pca_clusters > 1:
                for i in range(c + 1, k_pca_clusters):
                    m = ((centroids[i, n + 1] - centroids[c, n + 1]) /
                         (centroids[i, n] - centroids[c, n]))
                    b = centroids[c, n + 1] - m * centroids[c, n]

                    x_vals = np.linspace(centroids[c, n], centroids[i, n], res)
                    y_vals = m * x_vals + b

                    ax.plot(x_vals, y_vals, color="black", linestyle="--", linewidth=1)

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, columnspacing=0,
                  handletextpad=-0.5, fontsize=fontsize * 0.694)
        ax.set_xlabel(f"PC{n + 1}")
        ax.set_ylabel(f"PC{n + 2}")
        plt.title("Earthchem Samples")

        # Save the plot to a file if a filename is provided
        if filename:
            plt.savefig(f"{fig_dir}/{filename}-clusters{n + 1}{n + 2}.png")

        else:
            # Print plot
            plt.show()

        # Close device
        plt.close()

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize mixing arrays !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_mixing_array(mixing_array, fig_dir="figs/other", filename="earthchem",
                           figwidth=6.3, figheight=6.3, fontsize=22):
    """
    """
    # Get mixing array attributes
    oxides = mixing_array.oxides
    n_pca_components = mixing_array.n_pca_components
    k_pca_clusters = mixing_array.k_pca_clusters
    data = mixing_array.earthchem_cluster

    # Read benchmark samples
    df_bench = pd.read_csv("assets/data/benchmark-samples.csv")

    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.loc"] = "upper left"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = "False"
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = "True"
    plt.rcParams["figure.dpi"] = 330
    plt.rcParams["savefig.bbox"] = "tight"

    warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

    # Check for synthetic data
    if not mixing_array.synthetic_data_written:
        raise Exception("No synthetic data found! Call create_mixing_arrays() first ...")

    # Initialize synthetic datasets
    synthetic_datasets = {}

    # Compile all synthetic datasets into a dict
    for i in range(k_pca_clusters):
        for j in range(i + 1, k_pca_clusters):
            fname = (f"assets/data/synthetic-samples-pca{n_pca_components}-"
                     f"clusters{i + 1}{j + 1}.csv")
            synthetic_datasets[f"data_synthetic{i + 1}{j + 1}"] = pd.read_csv(fname)

    # Create a grid of subplots
    num_plots = len(oxides) - 1

    if num_plots == 1:
        num_cols = 1

    elif num_plots > 1 and num_plots <= 4:
        num_cols = 2

    elif num_plots > 4 and num_plots <= 9:
        num_cols = 3

    elif num_plots > 9 and num_plots <= 16:
        num_cols = 4

    else:
        num_cols = 5

    num_rows = (num_plots + 1) // num_cols

    # Total figure size
    fig_width = figwidth / 2 * num_cols
    fig_height = figheight / 2 * num_rows

    # Harker diagrams
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    axes = axes.flatten()

    for k, y in enumerate([oxide for oxide in oxides if oxide != "SIO2"]):
        ax = axes[k]

        for i in range(k_pca_clusters):
            for j in range(i + 1, k_pca_clusters):
                first_element = synthetic_datasets[f"data_synthetic{i + 1}{j + 1}"].iloc[0]
                last_element = synthetic_datasets[f"data_synthetic{i + 1}{j + 1}"].iloc[-1]

                sns.scatterplot(data=synthetic_datasets[f"data_synthetic{i + 1}{j + 1}"],
                                x="SIO2", y=y, linewidth=0, s=15, color="black",
                                legend=False, ax=ax, zorder=3)

        sns.kdeplot(data=data, x="SIO2", y=y, hue="COMPOSITION", fill=False, ax=ax, levels=5,
                    zorder=2)
        sns.scatterplot(data=data, x="SIO2", y=y, hue="COMPOSITION", linewidth=0, s=5,
                        legend=False, ax=ax, zorder=1)

        for name in ["PUM", "NMORB"]:
            ax.annotate(name, xy=(df_bench.loc[df_bench["NAME"] == name, "SIO2"].iloc[0],
                                  df_bench.loc[df_bench["NAME"] == name, y].iloc[0]),
                        xytext=(5, -10), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.1", color="white", alpha=0.8),
                        fontsize=fontsize * 0.579, zorder=6)

        sns.scatterplot(data=df_bench[df_bench["NAME"].isin(["PUM", "NMORB"])], x="SIO2",
                        y=y, color="black", linewidth=1.2, s=75, legend=False, ax=ax,
                        zorder=7)

        ax.set_title(f"{y}")
        ax.set_ylabel("")
        ax.set_xlabel("")

        if k < (num_plots - num_cols):
            ax.set_xticks([])

        if k == (num_plots - 1):
            handles = ax.get_legend().legendHandles
            labels = ["ultramafic", "mafic"]

        for line in ax.get_legend().get_lines():
            line.set_linewidth(5)

        ax.get_legend().remove()

    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=4)
    fig.suptitle("Harker Diagrams vs. SIO2 (wt.%)")

    if num_plots < len(axes):
        for i in range(num_plots, len(axes)):
            fig.delaxes(axes[i])

    # Save the plot to a file if a filename is provided
    if filename:
        plt.savefig(f"{fig_dir}/{filename}-harker-diagram.png")

    else:
        # Print plot
        plt.show()

    # Close device
    plt.close()