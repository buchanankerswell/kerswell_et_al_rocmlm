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
import matplotlib.pyplot as plt

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# machine learning !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#######################################################
## .0.             Helper Functions              !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# convert to nio !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def convert_to_nio(df, digits=3):
    """
    """
    # Copy df
    data = df.copy()

    # Check for misreported units
    condition = data["NIO"] > data["SIO2"]
    data.loc[condition, "NI"] = data.loc[condition]["NIO"]
    data.loc[condition, "NIO"] = np.nan

    # If NIO exists
    condition = data["NIO"].notna()
    data.loc[condition, "NI"] = np.nan

    # If NI exists but not NIO
    condition = data["NIO"].isna() & data["NI"].notna()
    data.loc[condition, "NIO"] = round(data.loc[condition]["NI"] / 1e4 * 1.2725, digits)
    data.loc[condition, "NI"] = np.nan

    return data

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# convert to cr2o3 !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def convert_to_cr2o3(df, digits=3):
    """
    """
    # Copy df
    data = df.copy()

    # Check for misreported units
    condition = data["CR2O3"] > data["SIO2"]
    data.loc[condition, "CR"] = data.loc[condition]["CR2O3"]
    data.loc[condition, "CR2O3"] = np.nan

    # If CR2O3 exists
    condition = data["CR2O3"].notna()
    data.loc[condition, "CR"] = np.nan

    # If CR exists but not CR2O3
    condition = data["CR2O3"].isna() & data["CR"].notna()
    data.loc[condition, "CR2O3"] = round(data.loc[condition]["CR"] / 1e4 * 1.4615, digits)
    data.loc[condition, "CR"] = np.nan

    return data

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# convert to fe2o3t !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def convert_to_fe2o3t(df, digits=3):
    """
    """
    # Copy df
    data = df.copy()

    # If FE2O3T exists set all Fe to nan except FE2O3T
    condition = data["FE2O3T"].notna()
    data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

    # If FE2O3 exists but not FE2O3T, FEO, or FEOT
    condition = (
        data["FE2O3"].notna() &
        data["FE2O3T"].isna() &
        data["FEO"].isna() &
        data["FEOT"].isna()
    )
    data.loc[condition, ["FE2O3T"]] = data.loc[condition]["FE2O3"]
    data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

    # If FEO exists but not FE2O3, FE2O3T, or FEOT
    condition = (
        data["FE2O3"].isna() &
        data["FE2O3T"].isna() &
        data["FEO"].notna() &
        data["FEOT"].isna()
    )
    data.loc[condition, ["FE2O3T"]] = round(data.loc[condition]["FEO"] / 0.89998, digits)
    data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

   # If FEOT exists but not FE2O3, FE2O3T, or FEO
    condition = (
        data["FE2O3"].isna() &
        data["FE2O3T"].isna() &
        data["FEO"].isna() &
        data["FEOT"].notna()
    )
    data.loc[condition, ["FE2O3T"]] = round(data.loc[condition]["FEOT"] / 0.89998, digits)
    data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

    # If FEO and FEOT exists but not FE2O3 or FE2O3T
    condition = (
        data["FE2O3"].isna() &
        data["FE2O3T"].isna() &
        data["FEO"].notna() &
        data["FEOT"].notna()
    )
    data.loc[condition, ["FE2O3T"]] = round(data.loc[condition]["FEOT"] / 0.89998, digits)
    data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

    # If FE2O3 and FEO exist but not FE2O3T or FEOT
    condition = (
        data["FE2O3"].notna() &
        data["FE2O3T"].isna() &
        data["FEO"].notna() &
        data["FEOT"].isna()
    )
    data.loc[condition, ["FE2O3T"]] = round(data.loc[condition]["FE2O3"] +
                                            data.loc[condition]["FEO"] / 0.89998, digits)
    data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

    # If FE2O3 and FEOT exist but not FE2O3T or FEO
    condition = (
        data["FE2O3"].notna() &
        data["FE2O3T"].isna() &
        data["FEO"].isna() &
        data["FEOT"].notna()
    )
    data.loc[condition, "FE2O3T"] = data.loc[condition, "FE2O3"]
    data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

    ## If FE2O3, FEO and FEOT exist but not FE2O3T
    condition = (
        data["FE2O3"].notna() &
        data["FE2O3T"].isna() &
        data["FEO"].notna() &
        data["FEOT"].notna()
    )
    data.loc[condition, ["FE2O3T"]] = round(data.loc[condition]["FEOT"] / 0.89998, digits)
    data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

    return data

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# convert to feot !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def convert_to_feot(df, digits=3):
    """
    """
    # Copy df
    data = df.copy()

    # If FEOT exists set all Fe to nan except FEOT
    condition = data["FEOT"].notna()
    data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

    # If FEO exists but not FEOT, FE2O3, or FE2O3T
    condition = (
        data["FEO"].notna() &
        data["FEOT"].isna() &
        data["FE2O3"].isna() &
        data["FE2O3T"].isna()
    )
    data.loc[condition, ["FEOT"]] = data.loc[condition]["FEO"]
    data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

    # If FE2O3 exists but not FEO, FEOT, or FE2O3T
    condition = (
        data["FEO"].isna() &
        data["FEOT"].isna() &
        data["FE2O3"].notna() &
        data["FE2O3T"].isna()
    )
    data.loc[condition, ["FEOT"]] = round(data.loc[condition]["FE2O3"] * 0.89998, digits)
    data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

   # If FE2O3T exists but not FEO, FEOT, or FE2O3
    condition = (
        data["FEO"].isna() &
        data["FEOT"].isna() &
        data["FE2O3"].isna() &
        data["FE2O3T"].notna()
    )
    data.loc[condition, ["FEOT"]] = round(data.loc[condition]["FE2O3T"] * 0.89998, digits)
    data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

    # If FE2O3 and FE2O3T exists but not FEO or FEOT
    condition = (
        data["FEO"].isna() &
        data["FEOT"].isna() &
        data["FE2O3"].notna() &
        data["FE2O3T"].notna()
    )
    data.loc[condition, ["FEOT"]] = round(data.loc[condition]["FE2O3T"] * 0.89998, digits)
    data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

    # If FEO and FE2O3 exist but not FEOT or FE2O3T
    condition = (
        data["FEO"].notna() &
        data["FEOT"].isna() &
        data["FE2O3"].notna() &
        data["FE2O3T"].isna()
    )
    data.loc[condition, ["FEOT"]] = round(data.loc[condition]["FEO"] +
                                          data.loc[condition]["FE2O3"] * 0.89998, digits)
    data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

    # If FEO and FE2O3T exist but not FEOT or FE2O3
    condition = (
        data["FEO"].notna() &
        data["FEOT"].isna() &
        data["FE2O3"].isna() &
        data["FE2O3T"].notna()
    )
    data.loc[condition, "FEOT"] = data.loc[condition, "FEO"]
    data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

    ## If FEO, FE2O3 and FE2O3T exist but not FEOT
    condition = (
        data["FEO"].notna() &
        data["FEOT"].isna() &
        data["FE2O3"].notna() &
        data["FE2O3T"].notna()
    )
    data.loc[condition, ["FEOT"]] = round(data.loc[condition]["FE2O3T"] * 0.89998, digits)
    data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

    return data

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# normalize volatile free !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def normalize_volatile_free(df, oxides, volatiles, loi, digits=3):
    """
    """
    # Copy df
    data = df.copy()

    # Normalization components
    ox = oxides
    vol = volatiles + loi

    # Sum oxides with and without volatiles
    data["total_ox"] = data[ox].sum(axis=1).round(digits)
    data["total_loi"] = data[ox + vol].sum(axis=1).round(digits)

    # Set total threshold
    total_threshold = [97, 103]

    # Check for samples within of threshold
    condition = (((data["total_ox"] >= total_threshold[0]) &
                  (data["total_ox"] <= total_threshold[1])) |
                 ((data["total_loi"] >= total_threshold[0]) &
                  (data["total_loi"] <= total_threshold[1])))

    # Filter data
    data = data.loc[condition]

    # Normalize to volatile free basis
    data[ox] = round(data[ox].div(data["total_ox"], axis=0) * 100, digits)

    # Re-sum oxides with and without volatiles
    data["total_ox"] = data[ox].sum(axis=1).round(digits)
    data["total_loi"] = data[ox + vol].sum(axis=1).round(digits)

    return data

#######################################################
## .0.                 PCA Class                 !!! ##
#######################################################
class MixingArray:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # init !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, res=128, n_pca_components=2, k_pca_clusters=3, seed=42, verbose=1):
        # Input
        self.res = res
        self.n_pca_components = n_pca_components
        self.k_pca_clusters = k_pca_clusters
        self.seed = seed
        self.verbose = verbose

        # Earthchem data
        self.earthchem_filename = "earthchem-igneous-plutonic-ultramafic.txt"
        self.metadata = ["SAMPLEID", "SOURCE", "REFERENCE", "LATITUDE", "LONGITUDE",
                         "COMPOSITION", "ROCKNAME"]
        self.oxides_system = ["SIO2", "AL2O3", "CAO", "MGO", "FEO", "K2O", "NA2O", "TIO2",
                              "FE2O3", "CR2O3"]
        self.oxides = ["SIO2", "AL2O3", "CAO", "MGO", "FEOT", "K2O", "NA2O", "TIO2", "FE2O3",
                       "CR2O3", "FE2O3T", "FEO", "NIO", "MNO", "P2O5"]
        self.loi = ["LOI"]
        self.volatiles = ["H2O", "CO2"]
        self.trace = ["CR", "NI"]
        self.digits = 3
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
        filename = self.earthchem_filename
        metadata = self.metadata
        oxides_system = self.oxides_system
        oxides = self.oxides
        loi = self.loi
        volatiles = self.volatiles
        trace = self.trace
        oxides_methods = [string + "METH" for string in oxides]
        loi_methods = [string + "METH" for string in loi]
        volatiles_methods = [string + "METH" for string in volatiles]
        trace_methods = [string + "METH" for string in trace]
        digits = self.digits

        # Check for earthchem data
        if not filename:
            raise Exception("No Earthchem data found!")

        # Initialize dataframes
        dataframes = {}
        df_name = []

        print("Reading Earthchem data ...")

        # Read data
        data = pd.read_csv(f"assets/data/{filename}", delimiter="\t")

        # Rename columns
        data.columns = [col.replace(" ", "") for col in data.columns]

        # Select columns
        data = data[metadata + oxides + oxides_methods + loi + loi_methods + volatiles +
                    volatiles_methods + trace + trace_methods]

        # Round values
        data[oxides] = data[oxides].round(digits)

        # Update self attribute
        self.earthchem_raw = data.copy()

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # process data !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _process_earthchem_data(self):
        """
        """
        # Get self attributes
        data = self.earthchem_raw.copy()
        oxides = self.oxides
        loi = self.loi
        volatiles = self.volatiles
        digits = self.digits
        verbose = self.verbose

        # Keep only samples with required oxides
        condition = data[["SIO2", "MGO", "AL2O3", "CAO"]].notna().all(axis=1)
        data = data.loc[condition]

        # Convert all Cr to CR2O3
        data = convert_to_cr2o3(data, digits)

        # Convert all Ni to NIO
        data = convert_to_nio(data, digits)

        # Convert all Fe oxides to FE2O3T
        data = convert_to_fe2o3t(data, digits)

        # Normalize to volatile free basis
        data = normalize_volatile_free(data, oxides, volatiles, loi, digits)

        # Convert all Fe oxides to FEOT
        data = convert_to_feot(data)

        # Set FE2O3 to zero
        data["FE2O3"] = float(0)

        # Drop totals
        data = data.drop(["total_ox", "total_loi"], axis=1)

        # Drop all NA columns
        data = data.dropna(axis=1, how="all")

        # Rename FEOT
        data = data.rename(columns={"FEOT": "FEO"})

        # Consolidate unique methods
        cols = [col for col in data.columns if "METH" in col]
        unique_methods = data[cols].apply(lambda x: ", ".join(x.dropna().unique()), axis=1)
        data["METHODS"] = unique_methods.str.upper()

        # Drop methods
        data = data.drop(cols, axis=1)

        # Arrange columns by dtype
        cols = (data.select_dtypes(include=["int", "float"]).columns.tolist() +
                data.select_dtypes(exclude=["int", "float"]).columns.tolist())
        data = data[cols]

        # Arrange rows by SIO2
        data = data.sort_values(by=["SIO2", "MGO"], ascending=[True, False])

        # Update self attribute
        self.earthchem_filtered = data.copy()

        if verbose >= 1:
            # Print info
            print(f"Combined and filtered samples summary:")
            print(data.info())
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Sample sources:")
            print("+++++++++++++++++++++++++++++++++++++++++++++")
            for source, count in data["SOURCE"].value_counts().items():
                print(f"[{count}] {source}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Rock types:")
            print("+++++++++++++++++++++++++++++++++++++++++++++")
            for name, count in data["ROCKNAME"].value_counts().items():
                print(f"[{count}] {name}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Top 5 most used methods:")
            print("+++++++++++++++++++++++++++++++++++++++++++++")
            for method, count in data["METHODS"].value_counts().head(5).items():
                print(f"[{count}] {method}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Top 5 references:")
            print("+++++++++++++++++++++++++++++++++++++++++++++")
            for ref, count in data["REFERENCE"].value_counts().head(5).items():
                print(f"[{count}] {ref}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # run pca !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _run_pca(self):
        """
        """
        # Get self attributes
        oxides = self.oxides_system
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

        # Process earthchem data
        self._process_earthchem_data()

        # Run PCA
        self._run_pca()

        # Get self attributes
        res = self.res
        oxides = self.oxides_system
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
