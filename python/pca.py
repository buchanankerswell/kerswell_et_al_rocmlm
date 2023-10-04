#######################################################
## .0.              Load Libraries               !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# utilities !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import math
import warnings

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# dataframes and arrays !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# plotting !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# machine learning !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#######################################################
## .0.          PCA and Synthetic Sampling       !!! ##
#######################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# read earthchem data !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_earthchem_data(oxides, verbose):
    """
    """
    # Data assets dir
    data_dir = "assets/data"

    # Find earthchem data files
    filepaths = [
        file for file in os.listdir(data_dir) if file.startswith("earthchem-igneous")
    ]

    # Filter criteria
    metadata = ["SAMPLE ID", "LATITUDE", "LONGITUDE", "COMPOSITION"]

    # Read all filepaths into dataframes
    dataframes = {}
    df_name = []

    for file in filepaths:
        df_name.append(file.split("-")[-1].split(".")[0])

        idx = file.split("-")[-1].split(".")[0]

        dataframes[f"df_{idx}"] = pd.read_csv(f"{data_dir}/{file}", delimiter="\t")
        dataframes[f"df_{idx}"] = dataframes[f"df_{idx}"][metadata + oxides]

    data = pd.concat(dataframes, ignore_index=True)

    if "SIO2" in oxides:
        data = data[data["SIO2"] >= 25]
        data = data[data["SIO2"] <= 90]

    if "CAO" in oxides:
        data = data[data["CAO"] <= 25]

    if "FE2O3" in oxides:
        data = data[data["FE2O3"] <= 20]

    if "TIO2" in oxides:
        data = data[data["TIO2"] <= 10]

    if verbose >= 2:
        # Print info
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print("Eartchem search portal critera:")
        print("    material: bulk")
        print("    normalization: oxides as reported")
        print("    sample type:")
        for name in df_name:
            print(f"        igneos > {name}")
        print("    oxides: (and/or)")
        for oxide in oxides:
            print(f"        {oxide}")
        print("Dataset filtering:")
        if "SIO2" in oxides:
            print("    SIO2 >= 25 wt.%")
            print("    SIO2 <= 90 wt.%")
        if "CAO" in oxides:
            print("    CAO <= 25 wt.%")
        if "FE2O3" in oxides:
            print("    FE2O3 <= 20 wt.%")
        if "TIO2" in oxides:
            print("    TIO2 <= 10 wt.%")
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print(f"Combined and filtered samples summary:")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(data[oxides].describe().map("{:.4g}".format))

    return data

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# pca mixing arrays !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def pca_mixing_arrays(res, oxides, n_pca_components, k_pca_clusters, seed, verbose,
                      palette="tab10", figwidth=6.3, figheight=6.3, fontsize=22,
                      filename="earthchem-samples", fig_dir="figs"):
    """
    """
    # Data assets dir
    data_dir = "assets/data"

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

    # Ignore warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

    # SIO2 required to be in oxides list
    if "SIO2" not in oxides:
        oxides = ["SIO2"] + oxides

    # Read geochemical data
    print("Reading Earthchem samples ...")
    data = read_earthchem_data(oxides, verbose)

    # Sort by composition
    if "MGO" in oxides:
        data.sort_values(by=["SIO2", "MGO"], ascending=[True, False], inplace=True,
                         ignore_index=True)

    else:
        data.sort_values(by="SIO2", ascending=True, inplace=True, ignore_index=True)

    # Impute missing measurement values by K-nearest algorithm
    imputer = KNNImputer(n_neighbors=4, weights="distance")
    imputer.fit(data[oxides])

    # Add missing values back to data
    data[oxides] = imputer.transform(data[oxides])

    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[oxides])

    # PCA modeling
    pca = PCA(n_components=n_pca_components)
    pca.fit(data_scaled)

    if verbose >= 1:
        # Print summary
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print("PCA summary:")
        print(f"    number of samples: {pca.n_samples_}")
        print(f"    PCA components: {n_pca_components}")
        print(f"    K-means clusters: {k_pca_clusters}")
        print(f"    features ({len(oxides)}): {oxides}")
        print("    explained variance:")
        for i, value in enumerate(pca.explained_variance_ratio_):
            print(f"        PC{i+1}: {round(value, 3)}")
        print("    cumulative explained variance:")
        cumulative_variance = pca.explained_variance_ratio_.cumsum()
        for i, value in enumerate(cumulative_variance):
            print(f"        PC{i+1}: {round(value, 3)}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Transform the data to obtain the principal components
    principal_components = pca.transform(data_scaled)

    # Create a DataFrame to store the results
    pca_columns = [f"PC{i+1}" for i in range(n_pca_components)]
    data[pca_columns] = principal_components

    # Round numerical data
    data[oxides + pca_columns] = data[oxides + pca_columns].round(3)

    # Write csv file
    data.to_csv(f"{data_dir}/earthchem-samples-pca.csv", index=False)

    # Plot PCA loadings
    loadings = pd.DataFrame((pca.components_.T * np.sqrt(pca.explained_variance_)).T,
                            columns=oxides)

    # Colormap
    colormap = plt.cm.get_cmap(palette)

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

    # Plot PCA results
    for n in range(n_pca_components-1):

        fig = plt.figure(figsize=(figwidth, figheight))
        ax = fig.add_subplot(111)

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

        for i, comp in enumerate(["ultramafic", "mafic", "intermediate", "felsic"]):
            indices = data.loc[data["COMPOSITION"] == comp].index

            scatter = ax.scatter(data.loc[indices, f"PC{n+1}"],
                                 data.loc[indices, f"PC{n+2}"], edgecolors="none",
                                 color=colormap(i), marker=".", label=comp)

        for oxide in oxides:
            ax.arrow(0, 0, loadings.at[n, oxide] * 3, loadings.at[n+1, oxide] * 3,
                     width=0.02, head_width=0.14, color="black")
            ax.text((loadings.at[n, oxide] * 3) + (loadings.at[n, oxide] * 1),
                    (loadings.at[n+1, oxide] * 3) + (loadings.at[n+1, oxide] * 1),
                    oxide, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, pad=0.1),
                    fontsize=fontsize * 0.579, color="black", ha = "center", va = "center")

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, columnspacing=0,
                  markerscale=3, handletextpad=-0.5, fontsize=fontsize * 0.694)
        ax.set_xlabel(f"PC{n+1}")
        ax.set_ylabel(f"PC{n+2}")
        plt.title("Earthchem Samples")

        # Save the plot to a file if a filename is provided
        if filename:
            plt.savefig(f"{fig_dir}/{filename}-pca{n+1}{n+2}.png")

        else:
            # Print plot
            plt.show()

        # Close device
        plt.close()

    # Kmeans clustering in PCA space
    kmeans = KMeans(n_clusters=k_pca_clusters, n_init="auto", random_state=seed)
    kmeans.fit(principal_components)

    # Add cluster labels to data
    data["CLUSTER"] = kmeans.labels_

    # Get centroids
    centroids = kmeans.cluster_centers_
    original_centroids = pca.inverse_transform(centroids)

    # Plot PCA results and extract mixing lines among cluster centroids
    for n in range(n_pca_components-1):

        fig = plt.figure(figsize=(figwidth, figheight))
        ax = fig.add_subplot(111)

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

        for c in range(k_pca_clusters):
            # Get datapoints indices for each cluster
            indices = data.loc[data["CLUSTER"] == c].index

            scatter = ax.scatter(data.loc[indices, f"PC{n+1}"],
                                 data.loc[indices, f"PC{n+2}"], edgecolors="none",
                                 color=colormap(c+4), marker=".", alpha=0.3)

            clusters = ax.scatter(centroids[c, n], centroids[c, n+1], edgecolor="black",
                                  color=colormap(c+4), label=f"cluster {c+1}",
                                  marker="s", s=100)

            # Calculate mixing lines between cluster centroids
            if k_pca_clusters > 1:
                for i in range(c+1, k_pca_clusters):
                    m = ((centroids[i, n+1] - centroids[c, n+1]) /
                         (centroids[i, n] - centroids[c, n]))
                    b = centroids[c, n+1] - m * centroids[c, n]

                    x_vals = np.linspace(centroids[c, n], centroids[i, n], res)
                    y_vals = m * x_vals + b

                    ax.plot(x_vals, y_vals, color="black", linestyle="--", linewidth=1)

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, columnspacing=0,
                  handletextpad=-0.5, fontsize=fontsize * 0.694)
        ax.set_xlabel(f"PC{n+1}")
        ax.set_ylabel(f"PC{n+2}")
        plt.title("Earthchem Samples")

        # Save the plot to a file if a filename is provided
        if filename:
            plt.savefig(f"{fig_dir}/{filename}-clusters{n+1}{n+2}.png")

        else:
            # Print plot
            plt.show()

        # Close device
        plt.close()

    # Initialize a dictionary for mixing lines
    mixing_lines = {}

    # Loop through PCA components
    print("Calculating mixing lines between cluster centroids ...")

    for n in range(n_pca_components):
        for c in range(k_pca_clusters):
            # Calculate mixing lines between cluster centroids
            if k_pca_clusters > 1:
                for i in range(c+1, k_pca_clusters):
                    if verbose >= 2:
                        print(f"    PC{n+1}", f"cluster{c+1}", f"cluster{i+1}")

                    if n == 0:
                        mixing_lines[f"cluster{c+1}{i+1}"] = (
                            np.linspace(centroids[c, n], centroids[i, n], res)
                        )

                    else:
                        mixing_lines[f"cluster{c+1}{i+1}"] = np.vstack((
                            mixing_lines[f"cluster{c+1}{i+1}"],
                            np.linspace(centroids[c, n], centroids[i, n], res)
                        ))

    # Write mixing lines to csv
    print(f"Saving mixing lines to {data_dir} ...")

    for i in range(k_pca_clusters):
        for j in range(i+1, k_pca_clusters):
            data_synthetic = pd.DataFrame(
                np.hstack((
                    scaler.inverse_transform(
                        pca.inverse_transform(
                            mixing_lines[f"cluster{i+1}{j+1}"].T
                        )
                    ),
                    mixing_lines[f"cluster{i+1}{j+1}"].T
                )),
                columns=oxides + [f"PC{n+1}" for n in range(n_pca_components)]
            ).round(3)

            # Add sample id column
            data_synthetic.insert(
                0, "NAME",
                [f"c{i+1}{j+1}-{n}" for n in range(len(data_synthetic))]
            )

            # Write to csv
            data_synthetic.to_csv(
                f"{data_dir}/synthetic-samples-pca{n_pca_components}-clusters{i+1}{j+1}.csv",
                index=False
            )

    # Compile all synthetic datasets into a dict
    synthetic_datasets = {}

    for i in range(k_pca_clusters):
        for j in range(i+1, k_pca_clusters):
            synthetic_datasets[f"data_synthetic{i+1}{j+1}"] = pd.read_csv(
                f"{data_dir}/synthetic-samples-pca{n_pca_components}-clusters{i+1}{j+1}.csv"
            )

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
            for j in range(i+1, k_pca_clusters):
                first_element = synthetic_datasets[f"data_synthetic{i+1}{j+1}"].iloc[0]
                last_element = synthetic_datasets[f"data_synthetic{i+1}{j+1}"].iloc[-1]

                sns.scatterplot(data=synthetic_datasets[f"data_synthetic{i+1}{j+1}"],
                                x="SIO2", y=y, linewidth=0, s=15, color=".2", legend=False,
                                ax=ax, zorder=3)
                ax.annotate(f"{i+1}", xy=(first_element["SIO2"], first_element[y]),
                            xytext=(0, 0), textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.1",
                                      edgecolor="black", facecolor="white", alpha=0.8),
                            fontsize=fontsize * 0.579, zorder=4)
                ax.annotate(f"{j+1}", xy=(last_element["SIO2"], last_element[y]),
                            xytext=(0, 0), textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.1",
                                      edgecolor="black", facecolor="white", alpha=0.8),
                            fontsize=fontsize * 0.579, zorder=5)

        sns.kdeplot(data=data, x="SIO2", y=y, hue="COMPOSITION",
                    hue_order=["ultramafic", "mafic", "intermediate", "felsic"], fill=False,
                    ax=ax, levels=5, zorder=2)
        sns.scatterplot(data=data, x="SIO2", y=y, hue="COMPOSITION",
                        hue_order=["ultramafic", "mafic", "intermediate", "felsic"],
                        linewidth=0, s=5, legend=False, ax=ax, zorder=1)

        ax.set_title(f"{y}")
        ax.set_ylabel("")
        ax.set_xlabel("")

        if k < (num_plots - num_cols):
            ax.set_xticks([])

        if k == (num_plots - 1):
            handles = ax.get_legend().legendHandles
            labels = ["ultramafic", "mafic", "intermediate", "felsic"]

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

