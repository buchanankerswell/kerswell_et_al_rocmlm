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
# machine learning !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score, mean_squared_error

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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#######################################################
## .1.              Visualizations               !!! ##
#######################################################

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ .1.1            Helper Functions              !!! ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# get geotherm !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_geotherm(results, target, threshold, Qs=55e-3, Ts=273, crust_thickness=35,
                 litho_thickness=150):
    """
    """
    # Get PT and target values and transform units
    df = pd.DataFrame({"P": results["P"], "T": results["T"],
                       target: results[target]}).sort_values(by="P")

    # Geotherm Parameters
    P = results["P"]
    Z_min = np.min(P) * 35e3
    Z_max = np.max(P) * 35e3
    z = np.linspace(Z_min, Z_max, len(P))
    T_geotherm = np.zeros(len(P))

    # Layer1 (crust)
    A1 = 1e-6 # Radiogenic heat production (W/m^3)
    k1 = 2.3 # Thermal conductivity (W/mK)
    D1 = crust_thickness * 1e3 # Thickness (m)

    # Layer2 (lithospheric mantle)
    A2 = 2.2e-8
    k2 = 3.0
    D2 = litho_thickness * 1e3

    # Calculate heat flow at the top of each layer
    Qt2 = Qs - (A1 * D1)
    Qt1 = Qs

    # Calculate T at the top of each layer
    Tt1 = Ts
    Tt2 = Tt1 + (Qt1 * D1 / k1) - (A1 / 2 / k1 * D1**2)
    Tt3 = Tt2 + (Qt2 * D2 / k2) - (A2 / 2 / k2 * D2**2)

    # Calculate T within each layer
    for j in range(len(P)):
        if z[j] <= D1:
            T_geotherm[j] = Tt1 + (Qt1 / k1 * z[j]) - (A1 / (2 * k1) * z[j]**2)
        elif D1 < z[j] <= D2 + D1:
            T_geotherm[j] = Tt2 + (Qt2 / k2 * (z[j] - D1)) - (A2 / (2 * k2) * (z[j] - D1)**2)
        elif z[j] > D2 + D1:
            T_geotherm[j] = Tt3 + 0.5e-3 * (z[j] - D1 - D2)

    P_geotherm = np.round(z / 35e3, 1)
    T_geotherm = np.round(T_geotherm, 2)

    df["geotherm_P"] = P_geotherm
    df["geotherm_T"] = T_geotherm

    # Subset df along geotherm
    df = df[abs(df["T"] - df["geotherm_T"]) < 10]

    # Extract the three vectors
    P_values = df["P"].values
    T_values = df["T"].values
    targets = df[target].values

    return P_values, T_values, targets

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# get 1d reference models !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_1d_reference_models():
    """
    """
    # Data asset dir
    data_dir = "assets/data"

    # Check for data dir
    if not os.path.exists(data_dir):
        raise Exception(f"Data not found at {data_dir}!")

    # Reference model paths
    ref_paths = {"prem": f"{data_dir}/PREM_1s.csv", "ak135": f"{data_dir}/AK135F_AVG.csv",
                 "stw105": f"{data_dir}/STW105.csv"}

    # Define column headers
    prem_cols = ["radius", "depth", "rho", "Vp", "Vph", "Vs", "Vsh", "eta", "Q_mu",
                 "Q_kappa"]
    ak135_cols = ["depth", "rho", "Vp", "Vs", "Q_kappa", "Q_mu"]
    stw105_cols = ["radius", "rho", "Vp", "Vs", "unk1", "unk2", "Vph", "Vsh", "eta"]

    ref_cols = {"prem": prem_cols, "ak135": ak135_cols, "stw105": stw105_cols}
    columns_to_keep = ["depth", "P", "rho", "Vp", "Vs"]

    # Initialize reference models
    ref_models = {}

    # Load reference models
    for name, path in ref_paths.items():
        if not os.path.exists(path):
            raise Exception(f"Refernce model {name} not found at {path}!")

        # Read reference model
        model = pd.read_csv(path, header=None, names=ref_cols[name])

        # Transform units
        if name == "stw105":
            model["depth"] = (model["radius"].max() - model["radius"]) / 1000
            model["rho"] = model["rho"] / 1000
            model["Vp"] = model["Vp"] / 1000
            model["Vs"] = model["Vs"] / 1000

        model["P"] = model["depth"] / 30

        # Clean up df
        model = model[columns_to_keep]
        model = model.round(3)

        # Save model
        ref_models[name] = model

    return ref_models

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
    # Parse and sort models
    magemin_models = [m if m.program == "magemin" and m.dataset == "train" else
                      None for m in gfem_models]
    magemin_models = sorted(magemin_models, key=lambda m: (m.sample_id if m else ""))

    perplex_models = [m if m.program == "perplex" and m.dataset == "train" else
                      None for m in gfem_models]
    perplex_models = sorted(perplex_models, key=lambda m: (m.sample_id if m else ""))

    # Iterate through all models
    for magemin_model, perplex_model in zip(magemin_models, perplex_models):
        magemin = True if magemin_model is not None else False
        perplex = True if perplex_model is not None else False

        if not magemin and not perplex:
            continue

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
            if magemin_model.model_prefix == perplex_model.model_prefix:
                model_prefix = magemin_model.model_prefix
            else:
                raise ValueError("Model prefix are not the same!")
            if magemin_model.verbose == perplex_model.verbose:
                verbose = magemin_model.verbose
            else:
                verbose = 1

            program = "magemin + perplex"
            fig_dir_mage = magemin_model.fig_dir
            fig_dir_perp = perplex_model.fig_dir
            fig_dir_diff = f"figs/gfem/diff_{sample_id}_{res}"

            fig_dir = f"figs/gfem/{sample_id}_{res}"
            os.makedirs(fig_dir, exist_ok=True)

        elif magemin and not perplex:
            # Get model data
            program = "magemin"
            sample_id = magemin_model.sample_id
            res = magemin_model.res
            dataset = magemin_model.dataset
            targets = magemin_model.targets
            fig_dir = magemin_model.fig_dir
            model_prefix = magemin_model.model_prefix
            verbose = magemin_model.verbose

        elif perplex and not magemin:
            # Get model data
            program = "perplex"
            sample_id = perplex_model.sample_id
            res = perplex_model.res
            dataset = perplex_model.dataset
            targets = perplex_model.targets
            fig_dir = perplex_model.fig_dir
            model_prefix = perplex_model.model_prefix
            verbose = perplex_model.verbose

        # Set geotherm threshold for extracting depth profiles
        if res <= 8:
            geotherm_threshold = 80
        elif res <= 16:
            geotherm_threshold = 40
        elif res <= 32:
            geotherm_threshold = 20
        elif res <= 64:
            geotherm_threshold = 10
        elif res <= 128:
            geotherm_threshold = 5
        else:
            geotherm_threshold = 2.5

        # Rename targets
        targets_rename = [target.replace("_", "-") for target in targets]

        if verbose >= 1:
            print(f"Composing {model_prefix} [{program}] ...")

        # Check for existing plots
        existing_figs = []
        for target in targets_rename:
            fig_1 = f"{fig_dir}/image2-{sample_id}-{dataset}-{target}.png"
            fig_2 = f"{fig_dir}/image3-{sample_id}-{dataset}-{target}.png"
            fig_3 = f"{fig_dir}/image4-{sample_id}-{dataset}-{target}.png"
            fig_4 = f"{fig_dir}/image9-{sample_id}-{dataset}.png"

            check = ((os.path.exists(fig_3) and os.path.exists(fig_4)) |
                     (os.path.exists(fig_1) and os.path.exists(fig_2)) |
                     (os.path.exists(fig_1) and os.path.exists(fig_2) and
                      os.path.exists(fig_4)))

            if check:
                existing_figs.append(check)

        if existing_figs:
            return None

        if magemin and perplex:
            for target in targets_rename:
                if target not in ["assemblage", "variance"]:
                    combine_plots_horizontally(
                        f"{fig_dir_mage}/magemin-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir_perp}/perplex-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/temp1.png",
                        caption1="a)",
                        caption2="b)"
                    )

                    combine_plots_horizontally(
                        f"{fig_dir}/temp1.png",
                        f"{fig_dir_diff}/diff-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/image3-{sample_id}-{dataset}-{target}.png",
                        caption1="",
                        caption2="c)"
                    )

                if target in ["rho", "Vp", "Vs"]:
                    combine_plots_horizontally(
                        f"{fig_dir_mage}/magemin-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir_perp}/perplex-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/temp1.png",
                        caption1="a)",
                        caption2="b)"
                    )

                    combine_plots_horizontally(
                        f"{fig_dir_diff}/diff-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir_diff}/prem-{sample_id}-{dataset}-{target}.png",
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

                for dir in [fig_dir_mage, fig_dir_perp, fig_dir_diff]:
                    shutil.rmtree(dir)

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

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# create dataset movies !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def create_dataset_movies(gfem_models):
    """
    """
    # Parse and sort models
    magemin_models = [m if m.program == "magemin" and m.dataset == "train" else
                      None for m in gfem_models]
    magemin_models = sorted(magemin_models, key=lambda m: (m.sample_id if m else ""))

    perplex_models = [m if m.program == "perplex" and m.dataset == "train" else
                      None for m in gfem_models]
    perplex_models = sorted(perplex_models, key=lambda m: (m.sample_id if m else ""))

    # Iterate through all models
    for magemin_model, perplex_model in zip(magemin_models, perplex_models):
        magemin = True if magemin_model is not None else False
        perplex = True if perplex_model is not None else False

        if not magemin and not perplex:
            continue

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
            if magemin_model.model_prefix == perplex_model.model_prefix:
                model_prefix = magemin_model.model_prefix
            else:
                raise ValueError("Model prefix are not the same!")
            if magemin_model.verbose == perplex_model.verbose:
                verbose = magemin_model.verbose
            else:
                verbose = 1

            program = "magemin + perplex"
            fig_dir_mage = magemin_model.fig_dir
            fig_dir_perp = perplex_model.fig_dir
            fig_dir_diff = f"figs/gfem/diff_{sample_id}_{res}"

            fig_dir = f"figs/gfem/{sample_id}_{res}"
            os.makedirs(fig_dir, exist_ok=True)

        elif magemin and not perplex:
            # Get model data
            program = "magemin"
            sample_id = magemin_model.sample_id
            res = magemin_model.res
            dataset = magemin_model.dataset
            targets = magemin_model.targets
            fig_dir = magemin_model.fig_dir
            model_prefix = magemin_model.model_prefix
            verbose = magemin_model.verbose

        elif perplex and not magemin:
            # Get model data
            program = "perplex"
            sample_id = perplex_model.sample_id
            res = perplex_model.res
            dataset = perplex_model.dataset
            targets = perplex_model.targets
            fig_dir = perplex_model.fig_dir
            model_prefix = perplex_model.model_prefix
            verbose = perplex_model.verbose

        # Rename targets
        targets_rename = [target.replace("_", "-") for target in targets]

        # Check for existing movies
        if sample_id not in ["PUM", "DMM"]:
            if "sb" in sample_id:
                pattern = "sb?????"
                prefix = "sb"
            if "sm" in sample_id:
                pattern = "sm?????"
                prefix = "sm"
            if "st" in sample_id:
                pattern = "st?????"
                prefix = "st"
            if "sr" in sample_id:
                pattern = "sr?????"
                prefix = "sr"

            existing_movs = []
            for target in targets_rename:
                mov_1 = f"figs/movies/image2-{prefix}-{target}.mp4"
                mov_2 = f"figs/movies/image3-{prefix}-{target}.mp4"
                mov_3 = f"figs/movies/image9-{prefix}.mp4"

                check = ((os.path.exists(mov_1) and os.path.exists(mov_3)) |
                         (os.path.exists(mov_2) and os.path.exists(mov_3)))

                if check:
                    existing_movs.append(check)

            if existing_movs:
                return None

            else:
                print(f"Creating movie for {prefix} samples [{program}] ...")

                if not os.path.exists("figs/movies"):
                    os.makedirs("figs/movies", exist_ok=True)

                for target in targets_rename:
                    if perplex and magemin:
                        ffmpeg = (f"ffmpeg -framerate 15 -pattern_type glob -i "
                                  f"'figs/gfem/{pattern}_{res}/image2-{pattern}-{dataset}-"
                                  f"{target}.png' -vf 'scale=3915:1432' -c:v h264 -pix_fmt "
                                  f"yuv420p 'figs/movies/image2-{prefix}-{target}.mp4'")
                    else:
                        ffmpeg = (f"ffmpeg -framerate 15 -pattern_type glob -i "
                                  f"'figs/gfem/{program[:4]}_{pattern}_{res}/image2-"
                                  f"{pattern}-{dataset}-{target}.png' -vf 'scale=3915:1432' "
                                  f"-c:v h264 -pix_fmt yuv420p 'figs/movies/image2-{prefix}-"
                                  f"{target}.mp4'")

                    try:
                        subprocess.run(ffmpeg, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, shell=True)

                    except subprocess.CalledProcessError as e:
                        print(f"Error running FFmpeg command: {e}")

                if all(item in targets_rename for item in ["rho", "Vp", "Vs"]):
                    for target in ["rho", "Vp", "Vs"]:
                        if perplex and magemin:
                            ffmpeg = (f"ffmpeg -framerate 15 -pattern_type glob -i "
                                      f"'figs/gfem/{pattern}_{res}/image3-{pattern}-"
                                      f"{dataset}-{target}.png' -vf 'scale=5832:1432' -c:v "
                                      f"h264 -pix_fmt yuv420p 'figs/movies/image3-{prefix}-"
                                      f"{target}.mp4'")
                        else:
                            ffmpeg = (f"ffmpeg -framerate 15 -pattern_type glob -i "
                                      f"'figs/gfem/{program[:4]}_{pattern}_{res}/image3-"
                                      f"{pattern}-{dataset}-{target}.png' -vf 'scale=5832:"
                                      f"1432' -c:v h264 -pix_fmt yuv420p 'figs/movies/"
                                      f"image3-{prefix}-{target}.mp4'")

                        try:
                            subprocess.run(ffmpeg, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL, shell=True)

                        except subprocess.CalledProcessError as e:
                            print(f"Error running FFmpeg command: {e}")

                    if perplex and magemin:
                        ffmpeg = (f"ffmpeg -framerate 15 -pattern_type glob -i "
                                  f"'figs/gfem/{pattern}_{res}/image9-{pattern}-{dataset}."
                                  f"png' -vf 'scale=5842:4296' -c:v h264 -pix_fmt yuv420p "
                                  f"'figs/movies/image9-{prefix}.mp4'")
                    else:
                        ffmpeg = (f"ffmpeg -framerate 15 -pattern_type glob -i "
                                  f"'figs/gfem/{program[:4]}_{pattern}_{res}/image9-"
                                  f"{pattern}-{dataset}.png' -vf 'scale=5842:4296' -c:v "
                                  f"h264 -pix_fmt yuv420p 'figs/movies/image9-{prefix}.mp4'")

                    try:
                        subprocess.run(ffmpeg, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, shell=True)

                    except subprocess.CalledProcessError as e:
                        print(f"Error running FFmpeg command: {e}")

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# compose rocmlm plots !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compose_rocmlm_plots(rocmlm):
    """
    """
    # Get ml model attributes
    program = rocmlm.program
    model_prefix = rocmlm.model_prefix
    ml_model_label = rocmlm.ml_model_label
    sample_ids = rocmlm.sample_ids
    res = rocmlm.res
    targets = rocmlm.targets
    fig_dir = rocmlm.fig_dir
    fig_dir_perf = "figs/rocmlm"
    verbose = rocmlm.verbose

    # Rename targets
    targets_rename = [target.replace("_", "-") for target in targets]

    # Check for existing plots
    existing_figs = []
    for target in targets_rename:
        for sample_id in rocmlm.sample_ids:
            fig_1 = f"{fig_dir}/prem-{sample_id}-{ml_model_label}-{target}.png"
            fig_2 = f"{fig_dir}/surf-{sample_id}-{ml_model_label}-{target}.png"
            fig_3 = f"{fig_dir}/image-{sample_id}-{ml_model_label}-{target}.png"
            fig_4 = f"{fig_dir}/image9-{sample_id}-{ml_model_label}-{target}.png"

            check = ((os.path.exists(fig_1) and os.path.exists(fig_2) and
                      os.path.exists(fig_3) and os.path.exists(fig_4)) |
                     (os.path.exists(fig_2) and os.path.exists(fig_3) and
                      os.path.exists(fig_4)))

            if check:
                existing_figs.append(check)

    if existing_figs:
        return None

    for target in targets_rename:
        for sample_id in rocmlm.sample_ids:
            if verbose >= 1:
                print(f"Composing {model_prefix}-{sample_id}-{target} [{program}] ...")

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
                    f"{fig_dir}/prem-{sample_id}-{ml_model_label}-{target}.png",
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
                f"{fig_dir}/surf-{sample_id}-{ml_model_label}-{target}.png",
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
                f"{fig_dir}/image-{sample_id}-{ml_model_label}-{target}.png",
                caption1="",
                caption2="c)"
            )

        if all(item in targets_rename for item in ["rho", "Vp", "Vs"]):
            captions = [("a", "b", "c"), ("d", "e", "f"), ("g", "h", "i")]
            targets = ["rho", "Vp", "Vs"]

            for i, target in enumerate(targets):
                combine_plots_horizontally(
                    f"{fig_dir}/{model_prefix}-{sample_id}-{target}-targets.png",
                    f"{fig_dir}/{model_prefix}-{sample_id}-{target}-predictions.png",
                    f"{fig_dir}/temp1.png",
                    caption1=captions[i][0],
                    caption2=captions[i][1]
                )

                combine_plots_horizontally(
                    f"{fig_dir}/temp1.png",
                    f"{fig_dir}/{model_prefix}-{sample_id}-{target}-prem.png",
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
                f"{fig_dir}/image9-{sample_id}-{ml_model_label}-{target}.png",
                caption1="",
                caption2=""
            )

    # Clean up directory
    rocmlm_files = glob.glob(f"{fig_dir}/rocmlm*.png")
    tmp_files = glob.glob(f"{fig_dir}/temp*.png")
    program_files = glob.glob(f"{fig_dir}/{program[:4]}*.png")

    for file in rocmlm_files + tmp_files + program_files:
        os.remove(file)

    return None

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ .1.2          Plotting Functions              !!! ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize gfem design !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_gfem_design(P_min=1, P_max=28, T_min=773, T_max=2273, fig_dir="figs/other",
                          T_mantle1=273, T_mantle2=1773, grad_mantle1=1, grad_mantle2=0.5,
                          fontsize=12, figwidth=6.3, figheight=3.54):
    """
    """
    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # T range
    T = np.arange(0, T_max + 728)

    # Olivine --> Ringwoodite Clapeyron slopes
    references_410 = {"[410 km] Akaogi89": [0.001, 0.002],
                      "[410 km] Katsura89": [0.0025],
                      "[410 km] Morishima94": [0.0034, 0.0038]}

    # Ringwoodite --> Bridgmanite + Ferropericlase Clapeyron slopes
    references_660 = {"[660 km] Ito82": [-0.002],
                      "[660 km] Ito89 & Hirose02": [-0.0028],
                      "[660 km] Ito90": [-0.002, -0.006],
                      "[660 km] Katsura03": [-0.0004, -0.002],
                      "[660 km] Akaogi07": [-0.0024, -0.0028]}

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

    db_data_handle = mpatches.Patch(color="gray", alpha=0.2, label="Dataset PT Range")

    labels_660.add("Dataset PT Range")
    label_color_mapping["Dataset PT Range"] = "gray"

    training_data_handle = mpatches.Patch(facecolor="blue", edgecolor="black", alpha=0.1,
                                          label="Mantle Conditions")

    labels_660.add("Mantle Conditions")
    label_color_mapping["Mantle Conditions"] = "gray"

    # Define the desired order of the legend items
    desired_order = ["Dataset PT Range", "Mantle Conditions", "[410 km] Akaogi89",
                     "[410 km] Katsura89", "[410 km] Morishima94", "[660 km] Ito82",
                     "[660 km] Ito89 & Hirose02", "[660 km] Ito90", "[660 km] Katsura03",
                     "[660 km] Akaogi07", "Geotherm 1", "Geotherm 2"]

    # Sort the legend handles based on the desired order
    legend_handles = sorted(ref_line_handles + [db_data_handle, training_data_handle],
                            key=lambda x: desired_order.index(x.get_label()))

    plt.xlabel("Temperature (K)")
    plt.ylabel("Pressure (GPa)")
    plt.title("Traning Dataset PT Range")
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
                              figwidth=6.3, figheight=3.54, fontsize=12):
    """
    """
    # Data assets dir
    data_dir = "assets/data"

    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Read data
    data = pd.read_csv(f"{data_dir}/gfem-efficiency.csv")
    data_lut = pd.read_csv(f"{data_dir}/lut-efficiency.csv")

    # Combine data
    data = pd.concat([data, data_lut], axis=0, ignore_index=True)

    # Filter out validation dataset
    data = data[data["dataset"] == "train"]

    # Filter out non-benchmark samples
    data = data[data["sample"].isin(["DMM", "PUM", "SMA128", "SMA64"])]

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
    sample_colors = {"DMM": "black", "PUM": "black", "SMA128": "black", "SMA64": "black"}
    sample_linetypes = {"DMM": ":", "PUM": "-", "SMA128": "--", "SMA64": "-."}

    # Group the data by sample
    grouped_data = data.groupby(["sample"])

    # Create a list to store the legend labels and handles
    legend_labels = []
    legend_handles = []

    # Plot the data for MAGEMin lines
    for group, group_data in grouped_data:
        sample_id = group[0]
        color_val = sample_colors[sample_id]
        line_val = sample_linetypes[sample_id]

        # mgm data
        mgm_data = group_data[group_data["program"] == "magemin"]
        mgm_x = mgm_data["size"]
        mgm_y = mgm_data["time"] / mgm_x * 1000

        # ppx data
        ppx_data = group_data[group_data["program"] == "perplex"]
        ppx_x = ppx_data["size"]
        ppx_y = ppx_data["time"] / ppx_x * 1000

        # lut data
        lut_data = group_data[group_data["program"] == "LUT"]
        lut_x = lut_data["size"]
        lut_y = lut_data["time"] * 1000

        # Plot mgm data points and connect them with lines
        line_mgm, = plt.plot(mgm_x, mgm_y, marker="o", color=color_val,
                             linestyle=line_val, label=f"[{sample_id}] MAGEMin")

        if sample_id in ["DMM", "PUM"]:
            legend_handles.append(line_mgm)
            legend_labels.append(f"[{sample_id}] MAGEMin")

        # Plot ppx data points and connect them with lines
        line_ppx, = plt.plot(ppx_x, ppx_y, marker="s", color=color_val,
                             linestyle=line_val, label=f"[{sample_id}] Perple_X")

        if sample_id in ["DMM", "PUM"]:
            legend_handles.append(line_ppx)
            legend_labels.append(f"[{sample_id}] Perple_X")

        # Plot lut data points and connect them with lines
        line_lut, = plt.plot(lut_x, lut_y, marker="^", color=color_val,
                             linestyle=line_val, label=f"[{sample_id}] LUT")

        if sample_id in ["SMA128", "SMA64"]:
            legend_handles.append(line_lut)
            legend_labels.append(f"[{sample_id}] LUT")

    # Set labels and title
    plt.xlabel("Number of PT Points")
    plt.ylabel("Elapsed Time (ms)")
    plt.title("GFEM & Lookup Table Efficiency")
    plt.xscale("log")
    plt.yscale("log")

    # Create the legend with the desired order
    plt.legend(title="", handles=legend_handles, loc="center left",
               bbox_to_anchor=(1.02, 0.5))

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
                   title=None, fig_dir="figs", filename=None, figwidth=6.3, figheight=4.725,
                   fontsize=22):
    """
    """
    # Data asset dir
    data_dir = "assets/data"

    # Check for data dir
    if not os.path.exists(data_dir):
        raise Exception(f"Data not found at {data_dir}!")

    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Get 1D reference models
    ref_models = get_1d_reference_models()

    # Get 1D refernce model profiles
    P_prem, target_prem = ref_models["prem"]["P"], ref_models["prem"][target]
    P_ak135, target_ak135 = ref_models["ak135"]["P"], ref_models["ak135"][target]
    P_stw105, target_stw105 = ref_models["stw105"]["P"], ref_models["stw105"][target]

    # Initialize geotherms
    P_mgm, P_ppx, P_ml, P_pum = None, None, None, None
    target_mgm, target_ppx, target_ml, target_pum = None, None, None, None

    # Get benchmark models
    pum_path = f"gfems/{program[:4]}_PUM_{dataset[0]}{res}/results.csv"
    source = "assets/data/benchmark-samples-pca.csv"
    targets = ["rho", "Vp", "Vs"]

    if os.path.exists(pum_path) and sample_id != "PUM":
        pum_model = GFEMModel(program, dataset, "PUM", source, res, 1, 28, 773, 2273, "all",
                              targets, False, 0, False)
        results_pum = pum_model.results
    else:
        results_pum = None

    # Extract target values along a geotherm
    if results_mgm:
        P_mgm, _, target_mgm = get_geotherm(results_mgm, target, geotherm_threshold)
    if results_ppx:
        P_ppx, _, target_ppx = get_geotherm(results_ppx, target, geotherm_threshold)
    if results_ml:
        P_ml, _, target_ml = get_geotherm(results_ml, target, geotherm_threshold)
    if results_pum:
        P_pum, _, target_pum = get_geotherm(results_pum, target, geotherm_threshold)

    # Get min and max P from geotherms
    P_min = min(np.nanmin(P) for P in [P_mgm, P_ppx, P_ml, P_pum] if P is not None)
    P_max = max(np.nanmax(P) for P in [P_mgm, P_ppx, P_ml, P_pum] if P is not None)

    # Create cropping mask for prem
    mask_prem = (P_prem >= P_min) & (P_prem <= P_max)
    mask_ak135 = (P_ak135 >= P_min) & (P_ak135 <= P_max)
    mask_stw105 = (P_stw105 >= P_min) & (P_stw105 <= P_max)

    # Crop pressure and target values
    P_prem, target_prem = P_prem[mask_prem], target_prem[mask_prem]
    P_ak135, target_ak135 = P_ak135[mask_ak135], target_ak135[mask_ak135]
    P_stw105, target_stw105 = P_stw105[mask_stw105], target_stw105[mask_stw105]

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

    # Get min max
    target_min = min(min(np.nanmin(lst) for lst in
                         [target_mgm, target_ppx, target_ml, target_pum] if lst is not None),
                     min(np.nanmin(lst) for lst in
                         [target_prem, target_ak135, target_stw105]))
    target_max = max(max(np.nanmax(lst) for lst in
                         [target_mgm, target_ppx, target_ml, target_pum] if lst is not None),
                     max(np.nanmin(lst) for lst in
                         [target_prem, target_ak135, target_stw105]))

    # Interpolate reference models to match GFEM and RocMLM profiles
    if results_ppx:
        xnew = np.linspace(target_min, target_max, len(target_ppx))
        P_prem, target_prem = np.interp(xnew, target_prem, P_prem), xnew
        P_ak135, target_ak135 = np.interp(xnew, target_ak135, P_ak135), xnew
        P_stw105, target_stw105 = np.interp(xnew, target_stw105, P_stw105), xnew
    if results_ml:
        xnew = np.linspace(target_min, target_max, len(target_ml))
        P_prem, target_prem = np.interp(xnew, target_prem, P_prem), xnew
        P_ak135, target_ak135 = np.interp(xnew, target_ak135, P_ak135), xnew
        P_stw105, target_stw105 = np.interp(xnew, target_stw105, P_stw105), xnew

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

    # Plot GFEM and RocMLM profiles
    if results_ppx:
        ax1.plot(target_ppx, P_ppx, "-", linewidth=3, color=colormap(0), label=sample_id)
    if results_ml:
        ax1.plot(target_ml, P_ml, "-.", linewidth=3, color=colormap(1), label=model)

    # Plot reference models
    ax1.plot(target_prem, P_prem, "-", linewidth=2, color="black", label="PREM")
    ax1.plot(target_ak135, P_ak135, "--", linewidth=2, color="black", label="AK135")
    ax1.plot(target_stw105, P_stw105, ":", linewidth=2, color="black", label="STW105")

    if target == "rho":
        target_label = "Density"
    else:
        target_label = target

    ax1.set_xlabel(f"{target_label} ({target_unit})")
    ax1.set_ylabel("P (GPa)")
    ax1.set_xticks(np.linspace(target_min, target_max, num=4))

    if target in ["Vp", "Vs", "rho"]:
        ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    # Vertical text spacing
    text_margin_x = 0.04
    text_margin_y = 0.15
    text_spacing_y = 0.1

    # Compute metrics
    if results_mgm:
        nan_mask = np.isnan(target_mgm)
        P_mgm, target_mgm = P_mgm[~nan_mask], target_mgm[~nan_mask]
        P_prem, target_prem = P_prem[~nan_mask], target_prem[~nan_mask]
        rmse = np.sqrt(mean_squared_error(target_prem, target_mgm))
        r2 = r2_score(target_prem, target_mgm)
    if results_ppx:
        nan_mask = np.isnan(target_ppx)
        P_ppx, target_ppx = P_ppx[~nan_mask], target_ppx[~nan_mask]
        P_prem, target_prem = P_prem[~nan_mask], target_prem[~nan_mask]
        rmse = np.sqrt(mean_squared_error(target_prem, target_ppx))
        r2 = r2_score(target_prem, target_ppx)
    if results_ml:
        nan_mask = np.isnan(target_ml)
        P_ml, target_ml = P_ml[~nan_mask], target_ml[~nan_mask]
        P_prem, target_prem = P_prem[~nan_mask], target_prem[~nan_mask]
        rmse = np.sqrt(mean_squared_error(target_prem, target_ml))
        r2 = r2_score(target_prem, target_ml)

    # Add R-squared and RMSE values as text annotations in the plot
    plt.text(1 - text_margin_x, text_margin_y - (text_spacing_y * 0), f"R$^2$: {r2:.3f}",
             transform=plt.gca().transAxes, fontsize=fontsize * 0.833,
             horizontalalignment="right", verticalalignment="bottom")
    plt.text(1 - text_margin_x, text_margin_y - (text_spacing_y * 1), f"RMSE: {rmse:.3f}",
             transform=plt.gca().transAxes, fontsize=fontsize * 0.833,
             horizontalalignment="right", verticalalignment="bottom")

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

    # Set geotherm threshold for extracting depth profiles
    res = target_array.shape[0]
    if res <= 8:
        geotherm_threshold = 80
    elif res <= 16:
        geotherm_threshold = 40
    elif res <= 32:
        geotherm_threshold = 20
    elif res <= 64:
        geotherm_threshold = 10
    elif res <= 128:
        geotherm_threshold = 5
    else:
        geotherm_threshold = 2.5

    # Get geotherm
    results = pd.DataFrame({"P": P, "T": T, target: target_array.flatten()})
    P_geotherm, T_geotherm, _ = get_geotherm(results, target, geotherm_threshold)

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
        ax.plot(T_geotherm, P_geotherm, linestyle="-", color="white", linewidth=3)
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
        ax.plot(T_geotherm, P_geotherm, linestyle="-", color="white", linewidth=3)
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
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
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
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
            ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
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
            continue

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

        if verbose >= 1:
            print(f"Visualizing {model_prefix} [{program}] ...")

        # Check for existing plots
        existing_figs = []
        for i, target in enumerate(targets):
            fig_1 = f"{fig_dir}/image3-{sample_id}-{dataset}-{target}.png"
            fig_2 = f"{fig_dir}/image4-{sample_id}-{dataset}-{target}.png"
            fig_3 = f"{fig_dir}/image9-{sample_id}-{dataset}.png"

            check = ((os.path.exists(fig_1) and os.path.exists(fig_3)) |
                     (os.path.exists(fig_1) and os.path.exists(fig_3)))

            if check:
                existing_figs.append(check)

        if existing_figs:
            return None

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
            filename = f"{program}-{sample_id}-{dataset}-{target_rename}.png"
            if verbose >= 2:
                print(f"Saving figure: {filename}")

            # Plot targets
            visualize_target_array(P, T, square_target, target, program_title, palette,
                                   color_discrete, color_reverse, vmin, vmax, fig_dir,
                                   filename)
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
                                       fig_dir, f"grad-{filename}")

            # Set geotherm threshold for extracting depth profiles
            if res <= 8:
                geotherm_threshold = 80
            elif res <= 16:
                geotherm_threshold = 40
            elif res <= 32:
                geotherm_threshold = 20
            elif res <= 64:
                geotherm_threshold = 10
            elif res <= 128:
                geotherm_threshold = 5
            else:
                geotherm_threshold = 2.5

            filename = f"prem-{sample_id}-{dataset}-{target_rename}.png"

            # Plot PREM comparisons
            if target == "rho":
                # Print filepath
                if verbose >= 2:
                    print(f"Saving figure: {filename}")

                if program == "magemin":
                    results_mgm = results
                    results_ppx = None
                    visualize_prem(program, sample_id, dataset, res, target, "g/cm$^3$",
                                   results_mgm, results_ppx,
                                   geotherm_threshold=geotherm_threshold,
                                   title="PREM Comparison", fig_dir=fig_dir,
                                   filename=filename)

                elif program == "perplex":
                    results_mgm = None
                    results_ppx = results
                    visualize_prem(program, sample_id, dataset, res, target, "g/cm$^3$",
                                   results_mgm, results_ppx,
                                   geotherm_threshold=geotherm_threshold,
                                   title="PREM Comparison", fig_dir=fig_dir,
                                   filename=filename)

            if target in ["Vp", "Vs"]:
                # Print filepath
                if verbose >= 2:
                    print(f"Saving figure: {filename}")

                if program == "magemin":
                    results_mgm = results
                    results_ppx = None
                    visualize_prem(program, sample_id, dataset, res, target, "km/s",
                                   results_mgm, results_ppx,
                                   geotherm_threshold=geotherm_threshold,
                                   title="PREM Comparison", fig_dir=fig_dir,
                                   filename=filename)

                elif program == "perplex":
                    results_mgm = None
                    results_ppx = results
                    visualize_prem(program, sample_id, dataset, res, target, "km/s",
                                   results_mgm, results_ppx,
                                   geotherm_threshold=geotherm_threshold,
                                   title="PREM Comparison", fig_dir=fig_dir,
                                   filename=filename)

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
    magemin_models = sorted(magemin_models, key=lambda m: (m.sample_id if m else ""))

    perplex_models = [m if m.program == "perplex" and m.dataset == "train" else
                      None for m in gfem_models]
    perplex_models = sorted(perplex_models, key=lambda m: (m.sample_id if m else ""))

    # Iterate through models
    for magemin_model, perplex_model in zip(magemin_models, perplex_models):
        # Check for model
        if magemin_model is None or perplex_model is None:
            continue

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
        if magemin_model.verbose == perplex_model.verbose:
            verbose = magemin_model.verbose
        else:
            verbose = 1

        fig_dir = f"figs/gfem/diff_{sample_id}_{res}"

        results_mgm, results_ppx = magemin_model.results, perplex_model.results
        P_mgm, T_mgm = results_mgm["P"], results_mgm["T"]
        P_ppx, T_ppx = results_ppx["P"], results_ppx["T"]
        target_array_mgm = magemin_model.target_array
        target_array_ppx = perplex_model.target_array

        for i, target in enumerate(targets):
            # Check for existing figures
            fig_1 = f"{fig_dir}/image3-{sample_id}-{dataset}-{target}.png"
            fig_2 = f"{fig_dir}/image4-{sample_id}-{dataset}-{target}.png"
            fig_3 = f"{fig_dir}/image9-{sample_id}-{dataset}.png"

            if os.path.exists(fig_1) and os.path.exists(fig_2) and os.path.exists(fig_3):
                print(f"Found composed plots at {fig_1}!")
                print(f"Found composed plots at {fig_2}!")
                print(f"Found composed plots at {fig_3}!")

            else:
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
                    vmin_mgm=np.min(
                        square_array_mgm[np.logical_not(np.isnan(square_array_mgm))])
                    vmax_mgm=np.max(
                        square_array_mgm[np.logical_not(np.isnan(square_array_mgm))])
                    vmin_ppx=np.min(
                        square_array_ppx[np.logical_not(np.isnan(square_array_ppx))])
                    vmax_ppx=np.max(
                        square_array_ppx[np.logical_not(np.isnan(square_array_ppx))])

                    vmin = min(vmin_mgm, vmin_ppx)
                    vmax = max(vmax_mgm, vmax_ppx)

                else:
                    num_colors_mgm = len(np.unique(square_array_mgm))
                    num_colors_ppx = len(np.unique(square_array_ppx))

                    vmin = 1
                    vmax = max(num_colors_mgm, num_colors_ppx) + 1

                if not color_discrete:
                    # Define a filter to ignore the specific warning
                    warnings.filterwarnings("ignore",
                                            message="invalid value encountered in divide")

                    # Create nan mask
                    mask = ~np.isnan(square_array_mgm) & ~np.isnan(square_array_ppx)

                    # Compute normalized diff
                    diff = square_array_mgm - square_array_ppx

                    # Add nans to match original target arrays
                    diff[~mask] = np.nan

                    # Rename target
                    target_rename = target.replace("_", "-")

                    # Print filepath
                    filename = f"diff-{sample_id}-{dataset}-{target_rename}.png"
                    if verbose >= 2:
                        print(f"Saving figure: {filename}")

                    # Plot target array normalized diff mgm-ppx
                    visualize_target_array(P_ppx, T_ppx, diff, target, "Residuals",
                                           "seismic", color_discrete, False, vmin, vmax,
                                           fig_dir, filename)

                    # Set geotherm threshold for extracting depth profiles
                    if res <= 8:
                        geotherm_threshold = 80
                    elif res <= 16:
                        geotherm_threshold = 40
                    elif res <= 32:
                        geotherm_threshold = 20
                    elif res <= 64:
                        geotherm_threshold = 10
                    elif res <= 128:
                        geotherm_threshold = 5
                    else:
                        geotherm_threshold = 2.5

                    filename = f"prem-{sample_id}-{dataset}-{target_rename}.png"

                    # Plot PREM comparisons
                    if target == "rho":
                        # Print filepath
                        if verbose >= 2:
                            print(f"Saving figure: {filename}")

                        visualize_prem("perplex", sample_id, dataset, res, target,
                                       "g/cm$^3$", results_mgm, results_ppx,
                                       geotherm_threshold=geotherm_threshold,
                                       title="PREM Comparison", fig_dir=fig_dir,
                                       filename=filename)

                    if target in ["Vp", "Vs"]:
                        # Print filepath
                        if verbose >= 2:
                            print(f"Saving figure: {filename}")

                        visualize_prem("perplex", sample_id, dataset, res, target, "km/s",
                                       results_mgm, results_ppx,
                                       geotherm_threshold=geotherm_threshold,
                                       title="PREM Comparison", fig_dir=fig_dir,
                                       filename=filename)

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize rocmlm !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_rocmlm(rocmlm, figwidth=6.3, figheight=4.725, fontsize=22):
    """
    """
    # Get ml model attributes
    program = rocmlm.program
    if program == "perplex":
        program_label = "Perple_X"
    elif program == "magemin":
        program_label = "MAGEMin"
    model_label_full = rocmlm.ml_model_label_full
    model_label = rocmlm.ml_model_label
    model_prefix = rocmlm.model_prefix
    sample_ids = rocmlm.sample_ids
    feature_arrays = rocmlm.feature_square
    target_arrays = rocmlm.target_square
    pred_arrays = rocmlm.prediction_square
    cv_info = rocmlm.cv_info
    targets = rocmlm.targets
    fig_dir = rocmlm.fig_dir
    palette = rocmlm.palette
    n_feats = feature_arrays.shape[-1] - 2
    n_models = feature_arrays.shape[0]
    w = feature_arrays.shape[1]
    verbose = rocmlm.verbose

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

    # Rename targets
    targets_rename = [target.replace("_", "-") for target in targets]

    # Check for existing plots
    existing_figs = []
    for target in targets_rename:
        for s in sample_ids:
            fig_1 = f"{fig_dir}/prem-{s}-{model_label}-{target}.png"
            fig_2 = f"{fig_dir}/surf-{s}-{model_label}-{target}.png"
            fig_3 = f"{fig_dir}/image-{s}-{model_label}-{target}.png"

            check = (os.path.exists(fig_1) and os.path.exists(fig_2) and
                     os.path.exists(fig_3))

            if check:
                existing_figs.append(check)

    if existing_figs:
        return None

    for s, sample_id in enumerate(sample_ids):
        if verbose >= 1:
            print(f"Visualizing {model_prefix}-{sample_id} [{program}] ...")

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
            diff = target_array[: ,: , i] - pred_array[: , :, i]

            # Make nans consistent
            diff[mask] = np.nan

            # Plot training data distribution and ML model predictions
            colormap = plt.cm.get_cmap("tab10")

            # Reverse color scale
            if palette in ["grey"]:
                color_reverse = False

            else:
                color_reverse = True

            # Filename
            filename = f"{model_prefix}-{sample_id}-{target_rename}"

            # Plot target array 2d
            P = feature_array[:, :, 0 + n_feats]
            T = feature_array[:, :, 1 + n_feats]
            t = target_array[:, :, i]
            p = pred_array[:, :, i]

            visualize_target_array(P.flatten(), T.flatten(), t, target, program_label,
                                   palette, False, color_reverse, vmin[i], vmax[i], fig_dir,
                                   f"{filename}-targets.png")
            # Plot target array 3d
            visualize_target_surf(P, T, t, target, program_label, palette, False,
                                  color_reverse, vmin[i], vmax[i], fig_dir,
                                  f"{filename}-targets-surf.png")

            # Plot ML model predictions array 2d
            visualize_target_array(P.flatten(), T.flatten(), p, target, model_label_full,
                                   palette, False, color_reverse, vmin[i], vmax[i], fig_dir,
                                   f"{filename}-predictions.png")

            # Plot ML model predictions array 3d
            visualize_target_surf(P, T, p, target, model_label_full, palette, False,
                                  color_reverse, vmin[i], vmax[i], fig_dir,
                                  f"{filename}-surf.png")

            # Plot PT normalized diff targets vs. ML model predictions 2d
            visualize_target_array(P.flatten(), T.flatten(), diff, target, "Residuals",
                                   "seismic", False, False, vmin[i], vmax[i], fig_dir,
                                   f"{filename}-diff.png")

            # Plot PT normalized diff targets vs. ML model predictions 3d
            visualize_target_surf(P, T, diff, target, "Residuals", "seismic", False, False,
                                  vmin[i], vmax[i], fig_dir, f"{filename}-diff-surf.png")

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
            results_rocmlm = {"P": P.flatten().tolist(), "T": T.flatten().tolist(),
                              target: p.flatten().tolist()}

            # Get relevant metrics for PREM plot
            rmse = cv_info[f"rmse_val_mean_{target}"]
            r2 = cv_info[f"r2_val_mean_{target}"]

            metrics = [rmse[0], r2[0]]

            # Set geotherm threshold for extracting depth profiles
            res = w - 1

            if res <= 8:
                geotherm_threshold = 80
            elif res <= 16:
                geotherm_threshold = 40
            elif res <= 32:
                geotherm_threshold = 20
            elif res <= 64:
                geotherm_threshold = 10
            elif res <= 128:
                geotherm_threshold = 5
            else:
                geotherm_threshold = 2.5

            # Plot PREM comparisons
            if target == "rho":
                visualize_prem(program, sample_id, "train", res, target, "g/cm$^3$",
                               results_mgm, results_ppx, results_rocmlm, model_label,
                               geotherm_threshold, title=model_label_full, fig_dir=fig_dir,
                               filename=f"{filename}-prem.png")

            if target in ["Vp", "Vs"]:
                visualize_prem(program, sample_id, "train", res, target, "km/s",
                               results_mgm, results_ppx, results_rocmlm, model_label,
                               geotherm_threshold, title=model_label_full, fig_dir=fig_dir,
                               filename=f"{filename}-prem.png")
    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize rocmlm performance !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_rocmlm_performance(targets, res, fig_dir="figs/rocmlm", filename="rocmlm",
                                 fontsize=22, figwidth=6.3, figheight=5):
    """
    """
    # Data assets dir
    data_dir = "assets/data"

    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Read regression data
    data = pd.read_csv(f"{data_dir}/rocmlm-performance.csv")

    # Summarize data
    data = data[data["size"] == data["size"].min()]
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

    # Get LUT time
    with open(f"{data_dir}/lut-efficiency.txt", "r") as f:
        content = f.read()
        time_lut = float(content)

    # Define the metrics to plot
    metrics = ["training_time_mean", "inference_time_mean", "rmse_test_mean",
               "rmse_val_mean"]
    metric_names = ["Training Efficiency", "Prediction Efficiency", "Training Error",
                    "Validation Error"]

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
    color_mapping = {"DT": colormap(0), "KN": colormap(2), "NN1": colormap(1),
                     "NN2": colormap(4), "NN3": colormap(3), "RF": colormap(5)}

    for i, metric in enumerate(metrics):
        plt.figure(figsize=(figwidth, figheight))

        # Define the offset for side-by-side bars
        bar_width = 0.1

        if metric == "training_time_mean":
            order = summary_df[metric].sort_values().index
            models_order = summary_df.loc[order]["model"].tolist()
            x_pos = np.arange(len(summary_df[metric]))

            bars = plt.bar(x_pos * bar_width, summary_df.loc[order][metric],
                           edgecolor="black", width=bar_width / 1.5,
                           color=[color_mapping[model] for model in models_order],
                           label=models_order if i == 1 else "")

            plt.title(f"{metric_names[i]}")
            plt.ylabel("Elapsed Time (ms)")
            plt.yscale("log")
            plt.gca().set_xticks([])
            plt.gca().set_xticklabels([])

        elif metric == "inference_time_mean":
            lut_line = plt.axhline(time_lut, color="black", linestyle=":",
                                   label="Lookup Table", alpha=0.5)
            mgm_line = plt.axhline(time_mgm, color="black", linestyle="-", label="MAGEMin",
                                   alpha=0.5)
            ppx_line = plt.axhline(time_ppx, color="black", linestyle="--", label="Perple_X",
                                   alpha=0.5)

            order = summary_df[metric].sort_values().index
            models_order = summary_df.loc[order]["model"].tolist()
            x_pos = np.arange(len(summary_df[metric]))

            bars = plt.bar(x_pos * bar_width, summary_df.loc[order][metric],
                           edgecolor="black", width=bar_width / 1.5,
                           color=[color_mapping[model] for model in models_order],
                           label=models_order if i == 1 else "")

            plt.title(f"{metric_names[i]}")
            plt.ylabel("Elapsed Time (ms)")
            plt.yscale("log")
            handles = [lut_line, mgm_line, ppx_line]
            labels = [handle.get_label() for handle in handles]
            legend = plt.legend(fontsize="x-small")
            plt.gca().set_xticks([])
            plt.gca().set_xticklabels([])

        elif metric == "rmse_test_mean":
            mult = 0
            y_label_pos = []
            for j, target in enumerate(targets):
                var = f"rmse_test_mean_{target}"

                order = summary_df[var].sort_values().index
                models_order = summary_df.loc[order]["model"].tolist()

                x_pos = np.arange(len(models_order)) + (len(models_order) * j) + mult

                # Calculate limits
                max_error = np.max(np.concatenate([
                    summary_df.loc[order][f"rmse_test_std_{target}"].values * 2,
                    summary_df.loc[order][f"rmse_val_std_{target}"].values * 2
                ]))

                max_mean = np.max(np.concatenate([
                    summary_df.loc[order][f"rmse_test_mean_{target}"].values,
                    summary_df.loc[order][f"rmse_val_mean_{target}"].values
                ]))

                vmax = max_mean + max_error + ((max_mean + max_error) * 0.01)
                y_label_pos.append(max_mean)

                bars = plt.bar(x_pos * bar_width,
                               summary_df.loc[order][f"rmse_test_mean_{target}"],
                               edgecolor="black", width=bar_width,
                               color=[color_mapping[model] for model in models_order],
                               label=models_order)

#                plt.errorbar(x_pos * bar_width,
#                             summary_df.loc[order][f"rmse_test_mean_{target}"],
#                             yerr=summary_df.loc[order][f"rmse_test_std_{target}"] * 2,
#                             fmt="none", capsize=5, color="black", linewidth=2)
                mult += 1

            x_tick_pos = [(p * bar_width) + (len(models_order) * o * bar_width) +
                          (len(models_order) / 2 * bar_width) - (bar_width / 2) for p, o in
                          zip(np.arange(len(targets)), np.arange(len(targets)))]

            for x_pos, y_pos, label in zip(x_tick_pos, y_label_pos, targets):
                if label == "rho":
                    unit = "g/cm$^3$"
                elif label in ["Vp", "Vs"]:
                    unit = "km/s"

                plt.text(x_pos, y_pos + (y_pos * 0.05), f"{label}\n({unit})", ha="center",
                         va="bottom")

            plt.title(f"{metric_names[i]}")
            plt.ylabel("RMSE")
            plt.yscale("log")
            plt.gca().set_xticks([])

        elif metric == "rmse_val_mean":
            mult = 0
            y_label_pos = []
            for j, target in enumerate(targets):
                var = f"rmse_test_mean_{target}"

                order = summary_df[var].sort_values().index
                models_order = summary_df.loc[order]["model"].tolist()

                x_pos = np.arange(len(models_order)) + (len(models_order) * j) + mult

                # Calculate limits
                max_error = np.max(np.concatenate([
                    summary_df.loc[order][f"rmse_test_std_{target}"].values * 2,
                    summary_df.loc[order][f"rmse_val_std_{target}"].values * 2
                ]))

                max_mean = np.max(np.concatenate([
                    summary_df.loc[order][f"rmse_test_mean_{target}"].values,
                    summary_df.loc[order][f"rmse_val_mean_{target}"].values
                ]))

                vmax = max_mean + max_error + ((max_mean + max_error) * 1)
                y_label_pos.append(max_mean)

                bars = plt.bar(x_pos * bar_width,
                               summary_df.loc[order][f"rmse_val_mean_{target}"],
                               edgecolor="black", width=bar_width,
                               color=[color_mapping[model] for model in models_order],
                               label=models_order if i == 1 else "")

#                plt.errorbar(x_pos * bar_width,
#                             summary_df.loc[order][f"rmse_val_mean_{target}"],
#                             yerr=summary_df.loc[order][f"rmse_val_std_{target}"] * 2,
#                             fmt="none", capsize=5, color="black", linewidth=2)
                mult += 1

            x_tick_pos = [(p * bar_width) + (len(models_order) * o * bar_width) +
                          (len(models_order) / 2 * bar_width) - (bar_width / 2) for p, o in
                          zip(np.arange(len(targets)), np.arange(len(targets)))]

            for x_pos, y_pos, label in zip(x_tick_pos, y_label_pos, targets):
                if label == "rho":
                    unit = "g/cm$^3$"
                elif label in ["Vp", "Vs"]:
                    unit = "km/s"

                plt.text(x_pos, y_pos + (y_pos * 0.05), f"{label}\n({unit})", ha="center",
                         va="bottom")

            plt.title(f"{metric_names[i]}")
            plt.ylabel("RMSE")
            plt.yscale("log")
            plt.ylim(5e-3, vmax)
            plt.gca().set_xticks([])

        # Save the plot to a file if a filename is provided
        if filename:
            plt.savefig(f"{fig_dir}/{filename}-{metric.replace('_', '-')}.png")

        else:
            # Print plot
            plt.show()

        # Close device
        plt.close()

    # Compose plots
    combine_plots_horizontally(
        f"{fig_dir}/rocmlm-inference-time-mean.png",
        f"{fig_dir}/rocmlm-training-time-mean.png",
        f"{fig_dir}/temp1.png",
        caption1="a)",
        caption2="b)"
    )

    os.remove(f"{fig_dir}/rocmlm-inference-time-mean.png")
    os.remove(f"{fig_dir}/rocmlm-training-time-mean.png")

    combine_plots_horizontally(
        f"{fig_dir}/temp1.png",
        f"{fig_dir}/rocmlm-rmse-val-mean.png",
        f"{fig_dir}/rocmlm-performance.png",
        caption1="",
        caption2="c)"
    )

    os.remove(f"{fig_dir}/rocmlm-rmse-test-mean.png")
    os.remove(f"{fig_dir}/rocmlm-rmse-val-mean.png")
    os.remove(f"{fig_dir}/temp1.png")

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize pca loadings !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_pca_loadings(mixing_array, fig_dir="figs/mixing_array", filename="earthchem",
                           batch=False, figwidth=6.3, figheight=5, fontsize=22):
    """
    """
    # Get mixing array attributes
    res = mixing_array.res
    pca = mixing_array.pca_model
    oxides = mixing_array.oxides_system
    n_pca_components = mixing_array.n_pca_components
    pca_model = mixing_array.pca_model
    pca_scaler = mixing_array.scaler
    mixing_array_endpoints = mixing_array.mixing_array_endpoints
    mixing_array_tops = mixing_array.mixing_array_tops
    mixing_array_bottoms = mixing_array.mixing_array_bottoms
    data = mixing_array.earthchem_pca
    D_tio2 = mixing_array.D_tio2

    # Get correct Depletion column
    if batch:
        D_col = "D_BATCH"
    else:
        D_col = "D_FRAC"

    # Check for benchmark samples
    df_bench_path = "assets/data/benchmark-samples.csv"
    df_bench_pca_path = "assets/data/benchmark-samples-pca.csv"
    df_synth_bench_path = "assets/data/synthetic-samples-benchmarks.csv"

    # Read benchmark samples
    if os.path.exists(df_bench_path) and os.path.exists(df_synth_bench_path):
        if os.path.exists(df_bench_pca_path):
            df_bench = pd.read_csv(df_bench_pca_path)
            df_synth_bench = pd.read_csv(df_synth_bench_path)

        else:
            df_bench = pd.read_csv(df_bench_path)
            df_synth_bench = pd.read_csv(df_synth_bench_path)

            # Fit PCA to benchmark samples
            df_bench[["PC1", "PC2"]] = pca_model.transform(
                pca_scaler.fit_transform(df_bench[oxides]))
            df_bench[["PC1", "PC2"]] = df_bench[["PC1", "PC2"]].round(3)

            # Calculate F melt
            ti_init = 0.199
            df_bench["R_TIO2"] = round(df_bench["TIO2"] / ti_init, 3)
            df_bench["F_MELT_BATCH"] = round(
                ((D_tio2 / df_bench["R_TIO2"]) - D_tio2) / (1 - D_tio2), 3)
            df_bench["D_BATCH"] = round(1 - df_bench["F_MELT_BATCH"], 3)
            df_bench["F_MELT_FRAC"] = round(
                1 - df_bench["R_TIO2"]**(1 / ((1 / D_tio2) - 1)), 3)
            df_bench["D_FRAC"] = round(1 - df_bench["F_MELT_FRAC"], 3)

            # Save to csv
            df_bench.to_csv("assets/data/benchmark-samples-pca.csv", index=False)

    # Filter Depletion < 1
    data = data[(data[D_col] <= 1) & (data[D_col] >= 0)]

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

    # Legend order
    legend_order = ["lherzolite", "harzburgite"]

    fig = plt.figure(figsize=(figwidth * 2, figheight * 1.2))

    ax = fig.add_subplot(121)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    legend_handles = []
    for i, comp in enumerate(legend_order):
        marker = mlines.Line2D([0], [0], marker="o", color="w", label=comp, markersize=4,
                               markerfacecolor=colormap(i), markeredgewidth=0,
                               linestyle="None", alpha=1)
        legend_handles.append(marker)

        indices = data.loc[data["ROCKNAME"] == comp].index

        scatter = ax.scatter(data.loc[indices, "PC1"],
                             data.loc[indices, "PC2"], edgecolors="none",
                             color=colormap(i), marker=".", s=55, label=comp, alpha=0.1)

    sns.kdeplot(data=data, x="PC1", y="PC2", hue="ROCKNAME",
                hue_order=legend_order, ax=ax, levels=5, zorder=1)

    x_offset, y_offset, text_fac, arrow_fac = 3.0, 6.0, 3.5, 1.8
    for oxide in ["SIO2", "MGO", "FEO", "AL2O3", "CR2O3", "TIO2"]:
        ax.arrow(x_offset, y_offset, loadings.at[0, oxide] * arrow_fac,
                 loadings.at[1, oxide] * 1.8, width=0.1, head_width=0.4,
                 color="black")
        ax.text(x_offset + (loadings.at[0, oxide] * text_fac),
                y_offset + (loadings.at[1, oxide] * text_fac), oxide,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, pad=0.1),
                fontsize=fontsize * 0.833, color="black", ha="center", va="center")

    edge_colors = ["white", "black"]
    face_colors = ["black", "white"]
    for l, name in enumerate(["PUM", "DMM"]):
        sns.scatterplot(data=df_bench[df_bench["SAMPLEID"] == name], x="PC1",
                        y="PC2", marker="s", facecolor=face_colors[l],
                        edgecolor=edge_colors[l], linewidth=2, s=150, legend=False, ax=ax,
                        zorder=7)
        ax.annotate(
            name, xy=(df_bench.loc[df_bench["SAMPLEID"] == name, "PC1"].iloc[0],
                      df_bench.loc[df_bench["SAMPLEID"] == name, "PC2"].iloc[0]),
            xytext=(-35, 15), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                      edgecolor=edge_colors[l], linewidth=1.5, alpha=0.8),
            fontsize=fontsize * 0.833, zorder=8
        )

    legend = ax.legend(handles=legend_handles, loc="upper center", frameon=False,
                       bbox_to_anchor=(0.5, 0.12), ncol=4, columnspacing=0,
                       handletextpad=-0.5, markerscale=3, fontsize=fontsize * 0.833)
    # Legend order
    for i, label in enumerate(legend_order):
        legend.get_texts()[i].set_text(label)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    plt.title("Earthchem Data")

    ax2 = fig.add_subplot(122)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    sns.scatterplot(data=data, x="PC1", y="PC2", hue=D_col,
                    palette=sns.color_palette("magma", as_cmap=True).reversed(),
                    edgecolor="None", linewidth=2, s=12, legend=False, ax=ax2, zorder=0)

    # Create colorbar
    norm = plt.Normalize(data[D_col].min(), data[D_col].max())
    sm = plt.cm.ScalarMappable(cmap="magma_r", norm=norm)
    sm.set_array([])

    for c in range(len(mixing_array_endpoints)):
        # Calculate mixing lines between mixing array endpoints
        if len(mixing_array_endpoints) > 1:
            for i in range(c + 1, len(mixing_array_endpoints)):
                if (((c == 0) & (i == 1)) | ((c == 1) & (i == 2)) |
                    ((c == 2) & (i == 3)) | ((c == 3) & (i == 4)) |
                    ((c == 4) & (i == 5))):
                    m = ((mixing_array_endpoints[i, 1] -
                          mixing_array_endpoints[c, 1]) /
                         (mixing_array_endpoints[i, 0] - mixing_array_endpoints[c, 0]))
                    m_tops = ((mixing_array_tops[i, 1] -
                               mixing_array_tops[c, 1]) /
                              (mixing_array_tops[i, 0] - mixing_array_tops[c, 0]))
                    m_bottoms = ((mixing_array_bottoms[i, 1] -
                                  mixing_array_bottoms[c, 1]) /
                                 (mixing_array_bottoms[i, 0] -
                                  mixing_array_bottoms[c, 0]))
                    b = (mixing_array_endpoints[c, 1] - m *
                         mixing_array_endpoints[c, 0])
                    b_tops = (mixing_array_tops[c, 1] - m_tops *
                              mixing_array_tops[c, 0])
                    b_bottoms = (mixing_array_bottoms[c, 1] - m_bottoms *
                                 mixing_array_bottoms[c, 0])

                    x_vals = np.linspace(mixing_array_endpoints[c, 0],
                                         mixing_array_endpoints[i, 0], res)
                    x_vals_tops = np.linspace(mixing_array_tops[c, 0],
                                              mixing_array_tops[i, 0], res)
                    x_vals_bottoms = np.linspace(mixing_array_bottoms[c, 0],
                                                 mixing_array_bottoms[i, 0], res)
                    y_vals = m * x_vals + b
                    y_vals_tops = m * x_vals_tops + b_tops
                    y_vals_bottoms = m * x_vals_bottoms + b_bottoms

                    ax2.plot(x_vals, y_vals, color="black", linestyle="--",
                             linewidth=4)
                    ax2.plot(x_vals_tops, y_vals_tops, color="black", linestyle="-",
                            linewidth=4)
                    ax2.plot(x_vals_bottoms, y_vals_bottoms, color="black",
                             linestyle="-", linewidth=4)

    sns.scatterplot(data=df_synth_bench[df_synth_bench["SAMPLEID"] == "sm12000"],
                    x="PC1", y="PC2", facecolor="white", edgecolor="black",
                    linewidth=2, s=150, legend=False, ax=ax2, zorder=6)
    sns.scatterplot(data=df_synth_bench[df_synth_bench["SAMPLEID"] == "sm12127"],
                    x="PC1", y="PC2", facecolor="black", edgecolor="white",
                    linewidth=2, s=150, legend=False, ax=ax2, zorder=6)
    ax2.annotate("DSUM", xy=(df_synth_bench.loc[df_synth_bench["SAMPLEID"] == "sm12000",
                               "PC1"].iloc[0],
                            df_synth_bench.loc[df_synth_bench["SAMPLEID"] == "sm12000",
                               "PC2"].iloc[0]),
                xytext=(-45, 15), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="black",
                          linewidth=1.5, alpha=0.8),
                fontsize=fontsize * 0.833, zorder=8)
    ax2.annotate("PSUM", xy=(df_synth_bench.loc[df_synth_bench["SAMPLEID"] == "sm12127",
                               "PC1"].iloc[0],
                            df_synth_bench.loc[df_synth_bench["SAMPLEID"] == "sm12127",
                               "PC2"].iloc[0]),
                xytext=(-45, 15), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="white",
                          linewidth=1.5, alpha=0.8),
                fontsize=fontsize * 0.833, zorder=8)

    plt.title("Mixing Arrays")
    plt.xlim(ax.get_xlim())
    plt.ylim(ax.get_ylim())

    # Add colorbar
    cbaxes = inset_axes(ax2, width="50%", height="3%", loc=1)
    colorbar = plt.colorbar(sm, ax=ax2, cax=cbaxes, label="Fertility Index",
                            orientation="horizontal")
    colorbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1g"))

    ax2.set_xlabel("PC1")
    ax2.set_ylabel("")
    ax2.set_yticks([])
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    # Save the plot to a file if a filename is provided
    if filename:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            plt.savefig(f"{fig_dir}/{filename}-mixing-arrays.png")

    else:
        # Print plot
        plt.show()

    # Close device
    plt.close()

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize harker diagrams !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_harker_diagrams(mixing_array, fig_dir="figs/mixing_array", filename="earthchem",
                              figwidth=6.3, figheight=6.3, fontsize=22):
    """
    """
    # Get mixing array attributes
    oxides = [ox for ox in mixing_array.oxides_system if ox not in ["SIO2", "FE2O3", "H2O"]]
    n_pca_components = mixing_array.n_pca_components
    mixing_array_endpoints = mixing_array.mixing_array_endpoints
    mixing_array_tops = mixing_array.mixing_array_tops
    mixing_array_bottoms = mixing_array.mixing_array_bottoms
    data = mixing_array.earthchem_filtered

    # Check for benchmark samples
    df_bench_path = "assets/data/benchmark-samples.csv"
    df_synth_bench_path = "assets/data/synthetic-samples-benchmarks.csv"

    if os.path.exists(df_bench_path) and os.path.exists(df_synth_bench_path):
        # Read benchmark samples
        df_bench = pd.read_csv(df_bench_path)
        df_synth_bench = pd.read_csv(df_synth_bench_path)

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
    synthetic_samples = pd.read_csv(f"assets/data/synthetic-samples-mixing-random.csv")

    # Create a grid of subplots
    num_plots = len(oxides) + 1

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

    xmin, xmax = data["SIO2"].min(), data["SIO2"].max()

    # Harker diagrams
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    axes = axes.flatten()

    # Legend order
    legend_order = ["lherzolite", "harzburgite"]

    for k, y in enumerate(oxides + ["pie"]):
        ax = axes[k]

        if y != "pie":
            sns.scatterplot(data=synthetic_samples, x="SIO2", y=y, linewidth=0, s=8,
                            color="black", alpha=1, legend=False, ax=ax, zorder=3)

            sns.scatterplot(data=data, x="SIO2", y=y, hue="ROCKNAME", hue_order=legend_order,
                            linewidth=0, s=8, alpha=0.1, ax=ax, zorder=1, legend=False)
            sns.kdeplot(data=data, x="SIO2", y=y, hue="ROCKNAME", hue_order=legend_order,
                        ax=ax, levels=5, zorder=1, legend=False)

            edge_colors = ["white", "black"]
            face_colors = ["black", "white"]
            for l, name in enumerate(["PUM", "DMM"]):
                sns.scatterplot(data=df_bench[df_bench["SAMPLEID"] == name],
                                x="SIO2", y=y, marker="s", facecolor=face_colors[l],
                                edgecolor=edge_colors[l], linewidth=2, s=75, legend=False,
                                ax=ax, zorder=7)

            sns.scatterplot(data=df_synth_bench[df_synth_bench["SAMPLEID"] == "sm12000"],
                            x="SIO2", y=y, facecolor="white", edgecolor="black",
                            linewidth=2, s=75, legend=False, ax=ax, zorder=6)
            sns.scatterplot(data=df_synth_bench[df_synth_bench["SAMPLEID"] == "sm12127"],
                            x="SIO2", y=y, facecolor="black", edgecolor="white",
                            linewidth=2, s=75, legend=False, ax=ax, zorder=6)

        if k == 5:
            for l, name in enumerate(["PUM", "DMM"]):
                ax.annotate(
                    name, xy=(df_bench.loc[df_bench["SAMPLEID"] == name, "SIO2"].iloc[0],
                              df_bench.loc[df_bench["SAMPLEID"] == name, y].iloc[0]),
                    xytext=(-35, 10), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                              edgecolor=edge_colors[l], linewidth=1.5, alpha=0.8),
                    fontsize=fontsize * 0.579, zorder=6
                )
            ax.annotate(
                "DSUM", xy=(df_synth_bench.loc[df_synth_bench["SAMPLEID"] == "sm12000",
                                               "SIO2"].iloc[0],
                            df_synth_bench.loc[df_synth_bench["SAMPLEID"] == "sm12000",
                                               y].iloc[0]),
                xytext=(5, -10), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="black",
                          linewidth=1.5, alpha=0.8),
                fontsize=fontsize * 0.579, zorder=8)
            ax.annotate(
                "PSUM", xy=(df_synth_bench.loc[df_synth_bench["SAMPLEID"] == "sm12127",
                                               "SIO2"].iloc[0],
                            df_synth_bench.loc[df_synth_bench["SAMPLEID"] == "sm12127",
                                               y].iloc[0]),
                xytext=(5, -10), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="white",
                          linewidth=1.5, alpha=0.8),
                fontsize=fontsize * 0.579, zorder=8)

        if k < (num_plots - num_cols - 1):
            ax.set_xticks([])

        ax.set_xlim(xmin - (xmin * 0.02), xmax + (xmax * 0.02))
        ax.set_ylabel("")
        ax.set_xlabel("")

        if y in ["NA2O", "TIO", "CR2O3", "K2O", "CAO", "AL2O3"]:
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
        else:
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2g}"))

        if y != "pie":
            ax.set_title(f"{y}")

        if y == "pie":
            colormap = plt.cm.get_cmap("tab10")
            colors = {"lherzolite": colormap(1), "harzburgite": colormap(0)}
            legend_colors = [colors[label] for label in legend_order]
            rock_counts = data["ROCKNAME"].value_counts()
            labels, counts = zip(*rock_counts.items())
            plt.pie(counts, labels=labels, autopct="%1.0f%%", startangle=0,
                    pctdistance=0.3, labeldistance=0.6, radius=1.3, colors=legend_colors,
                    textprops={"fontsize": fontsize * 0.694})

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

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize gfem analysis !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_gfem_analysis(batch=False, fig_dir="figs/mixing_array", filename="fmelt",
                            figwidth=6.3, figheight=5.5, fontsize=22):
    """
    """
    # Check for analysis data
    analysis_path = "assets/data/gfem-analysis.csv"

    if os.path.exists(analysis_path):
        df_analysis = pd.read_csv(analysis_path)
    else:
        raise Exception(f"GFEM analysis not found at {analysis_path}!")

    # Get correct Depletion column
    if batch:
        D_col = "D_BATCH"
    else:
        D_col = "D_FRAC"

    # Samples data paths
    df_paths = ["assets/data/benchmark-samples-pca.csv",
                "assets/data/synthetic-samples-mixing-tops.csv",
                "assets/data/synthetic-samples-mixing-bottoms.csv",
                "assets/data/synthetic-samples-mixing-random.csv"]
    df_names = ["benchmark", "top", "bottom", "random"]
    dfs = {}

    # Read data
    for name, path in zip(df_names, df_paths):
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df[(df[D_col] <= 1) & (df[D_col] >= 0)]
            dfs[name] = df.merge(df_analysis, on="SAMPLEID", how="inner")
        else:
            raise Exception(f"Missing sample data: {path}!")

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

    # Legend order
    legend_order = df_names[1:]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(figwidth * 3, figheight))
    for j, target in enumerate(["rho", "Vp", "Vs"]):
        # Define units
        if target == "rho":
            units = "g/cm$^3$"
            target_label = "Density"
        elif target in ["Vp", "Vs"]:
            units = "km/s"
            target_label = target

        ax = axes[j]
        legend_handles = []
        for i, comp in enumerate(legend_order):
            data = dfs[comp]
            data = data[data["TARGET"] == target]

            marker = mlines.Line2D([0], [0], marker="o", color="w", label=comp, markersize=4,
                                   markerfacecolor=colormap(i), markeredgewidth=0,
                                   linestyle="None", alpha=1)
            legend_handles.append(marker)

            scatter = ax.scatter(data[D_col], data["RMSE_PREM_PROFILE"],
                                 edgecolors="none", color=colormap(i), marker=".", s=150,
                                 label=comp)

        edge_colors = ["white", "black"]
        face_colors = ["black", "white"]
        df_bench = dfs["benchmark"]
        df_bench = df_bench[df_bench["TARGET"] == target]
        for l, name in enumerate(["PUM", "DMM"]):
            sns.scatterplot(data=df_bench[df_bench["SAMPLEID"] == name], x=D_col,
                            y="RMSE_PREM_PROFILE", facecolor=face_colors[l],
                            edgecolor=edge_colors[l], linewidth=2, s=75, legend=False,
                            ax=ax, zorder=7)
            ax.annotate(
                name, xy=(df_bench.loc[df_bench["SAMPLEID"] == name, D_col].iloc[0],
                          df_bench.loc[df_bench["SAMPLEID"] == name,
                                       "RMSE_PREM_PROFILE"].iloc[0]),
                xytext=(5, 10), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                          edgecolor=edge_colors[l], linewidth=1.5, alpha=0.8),
                fontsize=fontsize * 0.694, zorder=8
            )

        if j == 0:
            legend = ax.legend(handles=legend_handles, loc="upper center", frameon=False,
                               title="Mixing Array", bbox_to_anchor=(0.5, 0.28), ncol=4,
                               columnspacing=0, handletextpad=-0.5, markerscale=3,
                               fontsize=fontsize * 0.694)

            # Legend order
            for i, label in enumerate(legend_order):
                legend.get_texts()[i].set_text(label)

        ax.set_title(f"{target_label} vs. PREM")
        ax.set_xlabel("Fertility Index")
        ax.set_ylabel(f"RMSE ({units})")
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # Save the plot to a file if a filename is provided
    if filename:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            plt.savefig(f"{fig_dir}/{filename}-gfem-analysis.png")

    else:
        # Print plot
        plt.show()

    # Close device
    plt.close()

    return None