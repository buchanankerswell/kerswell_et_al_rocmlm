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
# machine learning !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#######################################################
## .0.                 PCA Class                 !!! ##
#######################################################
class MixingArray:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # init !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, res, n_pca_components=2, k_pca_clusters=3, seed=42, verbose=1):
        # Input
        self.res = res
        self.n_pca_components = n_pca_components
        self.k_pca_clusters = k_pca_clusters
        self.seed = seed
        self.verbose = verbose

        # Earthchem data
        self.earthchem_filepaths = [file for file in os.listdir("assets/data") if
                                    file.startswith("earthchem-igneous")]
        self.metadata = ["SAMPLE ID", "LATITUDE", "LONGITUDE", "COMPOSITION"]
        self.oxides = ["SIO2", "AL2O3", "CAO", "MGO", "FEO", "K2O", "NA2O", "TIO2", "FE2O3",
                       "CR2O3"]
        self.earthchem_raw = pd.DataFrame()
        self.earthchem_filtered = pd.DataFrame()
        self.earthchem_imputed = pd.DataFrame()
        self.earthchem_pca = pd.DataFrame()
        self.earthchem_cluster = pd.DataFrame()

        # PCA results
        self.scaler = None
        self.pca_model = None
        self.pca_results = np.array([])

        # KMeans clustering results
        self.kmeans_model = None

        # Mixing array results
        self.mixing_arrays = None
        self.synthetic_data_written = False

        # Errors
        self.mixing_array_error = False
        self.error = None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # read earthchem data !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _read_earthchem_data(self):
        """
        """
        # Get self attributes
        filepaths = self.earthchem_filepaths
        metadata = self.metadata
        oxides = self.oxides
        verbose = self.verbose

        # Check for earthchem data
        if not filepaths:
            raise Exception("No Earthchem data found!")

        # Initialize dataframes
        dataframes = {}
        df_name = []

        print("Reading Earthchem data ...")

        # Read all filepaths into dataframes
        for file in filepaths:
            df_name.append(file.split("-")[-1].split(".")[0])

            idx = file.split("-")[-1].split(".")[0]

            dataframes[f"df_{idx}"] = pd.read_csv(f"assets/data/{file}", delimiter="\t")
            dataframes[f"df_{idx}"] = dataframes[f"df_{idx}"][metadata + oxides]

        # Combine dataframes
        data = pd.concat(dataframes, ignore_index=True)

        # Update self attribute
        self.earthchem_raw = data.copy()

        # Filter data
        if "SIO2" in oxides:
            data = data[data["SIO2"] >= 25]
            data = data[data["SIO2"] <= 90]

        if "CAO" in oxides:
            data = data[data["CAO"] <= 25]

        if "FE2O3" in oxides:
            data = data[data["FE2O3"] <= 20]

        if "TIO2" in oxides:
            data = data[data["TIO2"] <= 10]

        if verbose >= 1:
            # Print info
            print("+++++++++++++++++++++++++++++++++++++++++++++")
            print("Eartchem search portal critera:")
            print("    material: bulk")
            print("    normalization: oxides as reported")
            print("    sample type:")
            for name in df_name:
                print(f"        igneos > {name}")
            print(f"Oxides: {oxides}")
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

        # Update self attribute
        self.earthchem_filtered = data.copy()

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # run pca !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _run_pca(self):
        """
        """
        # Get self attributes
        oxides = self.oxides
        n_pca_components = self.n_pca_components
        data = self.earthchem_filtered.copy()
        verbose = self.verbose

        # Check for earthchem data
        if data.empty:
            raise Exception("No Earthchem data found! Call _read_earthchem_data() first ...")

        # SIO2 required to be in oxides list
        if "SIO2" not in oxides:
            oxides = ["SIO2"] + oxides

        # Sort by composition
        if "MGO" in oxides:
            data.sort_values(by=["SIO2", "MGO"], ascending=[True, False], inplace=True,
                             ignore_index=True)

        else:
            data.sort_values(by="SIO2", ascending=True, inplace=True, ignore_index=True)

        # Initialize KNN imputer
        imputer = KNNImputer(weights="distance")

        print("Imputing missing oxides ...")

        # Impute missing measurement values by K-nearest algorithm
        imputer.fit(data[oxides])

        # Add missing values back to data
        data[oxides] = imputer.transform(data[oxides])

        # Initialize scaler
        scaler = StandardScaler()

        # Standardize data
        data_scaled = scaler.fit_transform(data[oxides])

        # Update attribute
        self.scaler = scaler

        # Initialize PCA
        pca = PCA(n_components=n_pca_components)

        print(f"Running PCA to reduce to {n_pca_components} dimensions ...")

        # PCA modeling
        pca.fit(data_scaled)

        # Update self attribute
        self.pca_model = pca

        if verbose >= 1:
            # Print summary
            print("+++++++++++++++++++++++++++++++++++++++++++++")
            print("PCA summary:")
            print(f"    number of samples: {pca.n_samples_}")
            print(f"    PCA components: {n_pca_components}")
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

        # Update self attribute
        self.pca_results = principal_components

        # Update dataframe
        pca_columns = [f"PC{i+1}" for i in range(n_pca_components)]
        data[pca_columns] = principal_components

        # Round numerical data
        data[oxides + pca_columns] = data[oxides + pca_columns].round(3)

        # Update self attribute
        self.earthchem_pca = data.copy()

        # Write csv file
        data.to_csv(f"assets/data/earthchem-samples-pca.csv", index=False)

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # create mixing arrays !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def create_mixing_array(self):
        """
        """
        # Get earthchem data
        self._read_earthchem_data()

        # Run PCA
        self._run_pca()

        # Get self attributes
        res = self.res
        oxides = self.oxides
        n_pca_components = self.n_pca_components
        k_pca_clusters = self.k_pca_clusters
        scaler = self.scaler
        pca = self.pca_model
        principal_components = self.pca_results
        data = self.earthchem_pca.copy()
        seed = self.seed
        verbose = self.verbose

        try:
            # Initialize KMeans
            kmeans = KMeans(n_clusters=k_pca_clusters, n_init="auto", random_state=seed)

            print(f"Clustering data (k={k_pca_clusters}) in reduced PCA space ...")

            # Kmeans clustering in PCA space
            kmeans.fit(principal_components)

            # Update self attribute
            self.kmeans_model = kmeans

            # Add cluster labels to data
            data["CLUSTER"] = kmeans.labels_

            # Update self attribute
            self.earthchem_cluster = data.copy()

            # Get centroids
            centroids = kmeans.cluster_centers_
            original_centroids = pca.inverse_transform(centroids)

            # Initialize mixing lines
            mixing_lines = {}

            # Loop through PCA components
            for n in range(n_pca_components):
                for c in range(k_pca_clusters):
                    # Calculate mixing lines between cluster centroids
                    if k_pca_clusters > 1:
                        for i in range(c + 1, k_pca_clusters):
                            if n == 0:
                                mixing_lines[f"cluster{c + 1}{i + 1}"] = (
                                    np.linspace(centroids[c, n], centroids[i, n], res)
                                )

                            else:
                                mixing_lines[f"cluster{c + 1}{i + 1}"] = np.vstack((
                                    mixing_lines[f"cluster{c + 1}{i + 1}"],
                                    np.linspace(centroids[c, n], centroids[i, n], res)
                                ))

            # Update self attribute
            self.mixing_arrays = mixing_lines

            # Create dataframes for mixing lines
            for i in range(k_pca_clusters):
                for j in range(i + 1, k_pca_clusters):
                    data_synthetic = pd.DataFrame(
                        np.hstack((
                            scaler.inverse_transform(
                                pca.inverse_transform(
                                    mixing_lines[f"cluster{i + 1}{j + 1}"].T
                                )
                            ),
                            mixing_lines[f"cluster{i + 1}{j + 1}"].T
                        )),
                        columns=oxides + [f"PC{n + 1}" for n in range(n_pca_components)]
                    ).round(3)

                    # Add sample id column
                    data_synthetic.insert(0, "NAME", [f"s{i + 1}{j + 1}{str(n).zfill(3)}" for
                                                      n in range(len(data_synthetic))])


                    # Write to csv
                    fname = (f"assets/data/synthetic-samples-pca{n_pca_components}-"
                             f"clusters{i + 1}{j + 1}.csv")
                    data_synthetic.to_csv(fname, index=False)

            # Update attribute
            self.synthetic_data_written = True

            return None

        except Exception as e:
            print("Error occurred when computing mixing arrays!")
            traceback.print_exc()

            self.mixing_array_error = True
            self.error = e

            return None
