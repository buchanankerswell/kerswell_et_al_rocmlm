#######################################################
## .0.              Load Libraries               !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# utilities !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import math
import warnings
from scipy import stats
import traceback

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
    condition = (data["FE2O3"].notna() & data["FE2O3T"].isna() & data["FEO"].isna() &
                 data["FEOT"].isna())
    data.loc[condition, ["FE2O3T"]] = data.loc[condition]["FE2O3"]
    data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

    # If FEO exists but not FE2O3, FE2O3T, or FEOT
    condition = (data["FE2O3"].isna() & data["FE2O3T"].isna() & data["FEO"].notna() &
                 data["FEOT"].isna())
    data.loc[condition, ["FE2O3T"]] = round(data.loc[condition]["FEO"] / 0.89998, digits)
    data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

   # If FEOT exists but not FE2O3, FE2O3T, or FEO
    condition = (data["FE2O3"].isna() & data["FE2O3T"].isna() & data["FEO"].isna() &
                 data["FEOT"].notna())
    data.loc[condition, ["FE2O3T"]] = round(data.loc[condition]["FEOT"] / 0.89998, digits)
    data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

    # If FEO and FEOT exists but not FE2O3 or FE2O3T
    condition = (data["FE2O3"].isna() & data["FE2O3T"].isna() & data["FEO"].notna() &
                 data["FEOT"].notna())
    data.loc[condition, ["FE2O3T"]] = round(data.loc[condition]["FEOT"] / 0.89998, digits)
    data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

    # If FE2O3 and FEO exist but not FE2O3T or FEOT
    condition = (data["FE2O3"].notna() & data["FE2O3T"].isna() & data["FEO"].notna() &
                 data["FEOT"].isna())
    data.loc[condition, ["FE2O3T"]] = round(data.loc[condition]["FE2O3"] +
                                            data.loc[condition]["FEO"] / 0.89998, digits)
    data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

    # If FE2O3 and FEOT exist but not FE2O3T or FEO
    condition = (data["FE2O3"].notna() & data["FE2O3T"].isna() & data["FEO"].isna() &
                 data["FEOT"].notna())
    data.loc[condition, "FE2O3T"] = data.loc[condition, "FE2O3"]
    data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

    ## If FE2O3, FEO and FEOT exist but not FE2O3T
    condition = (data["FE2O3"].notna() & data["FE2O3T"].isna() & data["FEO"].notna() &
                 data["FEOT"].notna())
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
    condition = (data["FEO"].notna() & data["FEOT"].isna() & data["FE2O3"].isna() &
                 data["FE2O3T"].isna())
    data.loc[condition, ["FEOT"]] = data.loc[condition]["FEO"]
    data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

    # If FE2O3 exists but not FEO, FEOT, or FE2O3T
    condition = (data["FEO"].isna() & data["FEOT"].isna() & data["FE2O3"].notna() &
                 data["FE2O3T"].isna())
    data.loc[condition, ["FEOT"]] = round(data.loc[condition]["FE2O3"] * 0.89998, digits)
    data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

   # If FE2O3T exists but not FEO, FEOT, or FE2O3
    condition = (data["FEO"].isna() & data["FEOT"].isna() & data["FE2O3"].isna() &
                 data["FE2O3T"].notna())
    data.loc[condition, ["FEOT"]] = round(data.loc[condition]["FE2O3T"] * 0.89998, digits)
    data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

    # If FE2O3 and FE2O3T exists but not FEO or FEOT
    condition = (data["FEO"].isna() & data["FEOT"].isna() & data["FE2O3"].notna() &
                 data["FE2O3T"].notna())
    data.loc[condition, ["FEOT"]] = round(data.loc[condition]["FE2O3T"] * 0.89998, digits)
    data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

    # If FEO and FE2O3 exist but not FEOT or FE2O3T
    condition = (data["FEO"].notna() & data["FEOT"].isna() & data["FE2O3"].notna() &
                 data["FE2O3T"].isna())
    data.loc[condition, ["FEOT"]] = round(data.loc[condition]["FEO"] +
                                          data.loc[condition]["FE2O3"] * 0.89998, digits)
    data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

    # If FEO and FE2O3T exist but not FEOT or FE2O3
    condition = (data["FEO"].notna() & data["FEOT"].isna() & data["FE2O3"].isna() &
                 data["FE2O3T"].notna())
    data.loc[condition, "FEOT"] = data.loc[condition, "FEO"]
    data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

    ## If FEO, FE2O3 and FE2O3T exist but not FEOT
    condition = (data["FEO"].notna() & data["FEOT"].isna() & data["FE2O3"].notna() &
                 data["FE2O3T"].notna())
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# samples to csv !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def samples_to_csv(sampleids, source, filename):
    """
    """
    # Check for file
    if not os.path.exists(source):
        raise Exception("Sample data source does not exist!")

    # Read data
    df = pd.read_csv(source)

    # Subset samples
    samples = df[df["SAMPLEID"].isin(sampleids)]

    # Write csv
    if os.path.exists(filename):
        df_bench = pd.read_csv(filename)
        df_bench = df_bench[~df_bench["SAMPLEID"].isin(sampleids)]
        df_bench = pd.concat([df_bench, samples])
        df_bench.to_csv(filename, index=False)
    else:
        samples.to_csv(filename, index=False)

    return None

#######################################################
## .0.            MixingArray Class              !!! ##
#######################################################
class MixingArray:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # init !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, res=128, mc_sample=1, weighted_random=True, k=3, D_tio2=5e-2,
                 seed=42, verbose=1):
        # Input
        self.res = res
        self.mc_sample = mc_sample
        self.weighted_random = weighted_random
        self.k = k
        self.D_tio2 = D_tio2
        self.seed = seed
        self.verbose = verbose

        # Earthchem data
        self.earthchem_filename = "earthchem-igneous-plutonic-ultramafic.txt"
        self.metadata = ["SAMPLEID", "SOURCE", "REFERENCE", "LATITUDE", "LONGITUDE",
                         "COMPOSITION", "ROCKNAME"]
        self.oxides_system = ["SIO2", "AL2O3", "CAO", "MGO", "FEO", "K2O", "NA2O", "TIO2",
                              "FE2O3", "CR2O3", "H2O"]
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

        # PCA results
        self.n_pca_components = 2
        self.scaler = None
        self.pca_model = None
        self.pca_results = np.array([])

        # Mixing array results
        self.mixing_array_endpoints = np.array([])
        self.mixing_array_tops = np.array([])
        self.mixing_array_bottoms = np.array([])
        self.mixing_arrays = None
        self.top_arrays = None
        self.bottom_arrays = None
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

        # Rock names meta groups
        peridotite = ["peridotite", "harzburgite", "lherzolite", "dunite", "wehrlite"]
        pyroxenite = ["pyroxenite", "websterite", "hornblendite", "orthopyroxenite",
                      "clinopyroxenite"]

        # Keep only samples with required oxides
        condition = data[["SIO2", "MGO", "AL2O3", "CAO"]].notna().all(axis=1)
        data = data.loc[condition]

        # Drop unknown rocks
        data = data[~data["ROCKNAME"].isin(["unknown", "chromitite", "limburgite",
                                            "wehrlite", "peridotite"])]

        # Drop pyroxenite samples
        data = data[~data["ROCKNAME"].isin(pyroxenite)]

        # Add new rock type column
        conditions = [data["ROCKNAME"].isin(peridotite), data["ROCKNAME"].isin(pyroxenite)]
        values = ["peridotite", "pyroxenite"]
        data["ROCKTYPE"] = np.select(conditions, values, default="other")

        # Function to remove outliers based on IQR
        def remove_outliers(group, threshold):
            Q1, Q3 = group[oxides].quantile(0.25), group[oxides].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound, upper_bound = Q1 - threshold * IQR, Q3 + threshold * IQR
            outlier_rows = ((group[oxides] < lower_bound) |
                            (group[oxides] > upper_bound)).any(axis=1)
            return group[~outlier_rows]

        # Remove outliers for each rock type
        data = data.groupby("ROCKNAME").apply(remove_outliers, 1.5)
        data.reset_index(drop=True, inplace=True)

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

        # Set H2O to zero
        data["H2O"] = float(0)

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

        # Drop individual methods
        data = data.drop(cols, axis=1)

        # Arrange columns by dtype
        cols = (data.select_dtypes(include=["int", "float"]).columns.tolist() +
                data.select_dtypes(exclude=["int", "float"]).columns.tolist())
        data = data[cols]

        # Arrange rows by SIO2
        data = data.sort_values(by=["SIO2", "MGO"], ascending=[True, False], ignore_index=True)

        # Update self attribute
        self.earthchem_filtered = data.copy()

        if verbose >= 1:
            # Print info
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"Combined and filtered samples summary:")
            print("+++++++++++++++++++++++++++++++++++++++++++++")
            print(data.info())
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Sample sources:")
            print("+++++++++++++++++++++++++++++++++++++++++++++")
            for source, count in data["SOURCE"].value_counts().items():
                print(f"[{count}] {source}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Rock names:")
            print("+++++++++++++++++++++++++++++++++++++++++++++")
            for name, count in data["ROCKNAME"].value_counts().items():
                print(f"[{count}] {name}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Rock types:")
            print("+++++++++++++++++++++++++++++++++++++++++++++")
            for type, count in data["ROCKTYPE"].value_counts().items():
                print(f"[{count}] {type}")
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
        digits = self.digits
        verbose = self.verbose

        # Check for earthchem data
        if data.empty:
            raise Exception("No Earthchem data found! Call _read_earthchem_data() first ...")

        # Sort by composition
        data = data.sort_values(by=["SIO2", "MGO"], ascending=[True, False],
                                ignore_index=True)

        # Initialize KNN imputer
        imputer = KNNImputer(weights="distance")

        print("Imputing missing oxides ...")

        # Impute missing values for each oxide
        for col in oxides:
            column_to_impute = data[[col]]
            imputer.fit(column_to_impute)
            imputed_values = imputer.transform(column_to_impute).round(digits)
            data[col] = imputed_values

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
        scaler = self.scaler
        pca = self.pca_model
        principal_components = self.pca_results
        data = self.earthchem_pca.copy()
        mc_sample = self.mc_sample
        weighted_random = self.weighted_random
        k = self.k
        D_tio2 = self.D_tio2
        seed = self.seed
        verbose = self.verbose

        try:
            # Define sample centroids
            centroids = data.groupby("ROCKNAME")[["PC1", "PC2"]].median()

            # Initialize endpoints
            mixing_array_endpoints = []
            mixing_array_tops = []
            mixing_array_bottoms = []

            for x, y, rockname in zip(centroids["PC1"].tolist(), centroids["PC2"].tolist(),
                                      centroids.index.tolist()):
                # Identify centroid quadrant
                if x > 0 and y > 0:
                    quadrant = "Q1"
                elif x < 0 and y > 0:
                    quadrant = "Q2"
                elif x < 0 and y < 0:
                    quadrant = "Q3"
                elif x > 0 and y < 0:
                    quadrant = "Q4"
                else:
                    raise Exception("Invalid quadrant!")

                # Subset cluster datapoints
                condition = data["ROCKNAME"] == rockname

                # Get IQR for PC1
                q1_pc1 = np.percentile(data.loc[condition, "PC1"], 25)
                q3_pc1 = np.percentile(data.loc[condition, "PC1"], 75)
                iqr_pc1 = q3_pc1 - q1_pc1

                # Get median for PC1
                median_x = np.median(data.loc[condition, "PC1"])

                # Define adjustment factor
                median_adjustment = 0
                top_adjustment = 1.2
                bottom_adjustment = 1.2

                # Adjust endpoint for PC1
                if quadrant == "Q1":
                    endpoint_x = median_x + median_adjustment * iqr_pc1
                elif quadrant == "Q2":
                    endpoint_x = median_x - median_adjustment * iqr_pc1
                elif quadrant == "Q3":
                    endpoint_x = median_x - median_adjustment * iqr_pc1
                elif quadrant == "Q4":
                    endpoint_x = median_x + median_adjustment * iqr_pc1

                # Get IQR for PC2
                q1_pc2 = np.percentile(data.loc[condition, "PC2"], 25)
                q3_pc2 = np.percentile(data.loc[condition, "PC2"], 75)
                iqr_pc2 = q3_pc2 - q1_pc2

                # Get median for PC2
                median_y = np.median(data.loc[condition, "PC2"])

                # Adjust endpoint for PC2
                if quadrant == "Q1":
                    endpoint_y = median_y + median_adjustment * iqr_pc2
                elif quadrant == "Q2":
                    endpoint_y = median_y + median_adjustment * iqr_pc2
                elif quadrant == "Q3":
                    endpoint_y = median_y - median_adjustment * iqr_pc2
                elif quadrant == "Q4":
                    endpoint_y = median_y - median_adjustment * iqr_pc2

                mixing_array_endpoints.append([endpoint_x, endpoint_y])

                # Identify highest SIO2 sample in Q1
                if quadrant == "Q1":
                    endpoint_x2 = median_x + 2.0 * iqr_pc1
                    endpoint_y2 = median_y + 1.7 * iqr_pc2

                    mixing_array_endpoints[-1] = [endpoint_x2, endpoint_y2]

                # Identify lowest SIO2 sample in Q2
                if quadrant == "Q2":
                    endpoint_x2 = median_x - 1.2 * iqr_pc1
                    endpoint_y2 = median_y + 0.2 * iqr_pc2

                    mixing_array_endpoints[-1] = [endpoint_x2, endpoint_y2]

            endpoints_sorted = sorted(mixing_array_endpoints, key=lambda x: x[0])

            mixing_array_endpoints = np.array(endpoints_sorted)

            mixing_array_tops = mixing_array_endpoints.copy()
            mixing_array_bottoms = mixing_array_endpoints.copy()

            mixing_array_tops[:,-1] += top_adjustment * iqr_pc2
            mixing_array_bottoms[:,-1] -= bottom_adjustment * iqr_pc2

            self.mixing_array_endpoints = mixing_array_endpoints
            self.mixing_array_tops = mixing_array_tops
            self.mixing_array_bottoms = mixing_array_bottoms

            # Initialize mixing lines
            mixing_lines = {}
            top_lines = {}
            bottom_lines = {}

            # Loop through PCA components
            for n in range(n_pca_components):
                for i in range(len(mixing_array_endpoints)):
                    # Calculate mixing lines between endpoints
                    if len(mixing_array_endpoints) > 1:
                        for j in range(i + 1, len(mixing_array_endpoints)):
                            if n == 0:
                                mixing_lines[f"{i + 1}{j + 1}"] = (
                                    np.linspace(mixing_array_endpoints[i, n],
                                                mixing_array_endpoints[j, n], res))
                                top_lines[f"{i + 1}{j + 1}"] = (
                                    np.linspace(mixing_array_tops[i, n],
                                                mixing_array_tops[j, n], res))
                                bottom_lines[f"{i + 1}{j + 1}"] = (
                                    np.linspace(mixing_array_bottoms[i, n],
                                                mixing_array_bottoms[j, n], res))

                            else:
                                if (((i == 0) & (j == 1)) | ((i == 1) & (j == 2)) |
                                    ((i == 2) & (j == 3)) | ((i == 3) & (j == 4)) |
                                    ((i == 4) & (j == 5))):
                                    mixing_lines[f"{i + 1}{j + 1}"] = np.vstack((
                                        mixing_lines[f"{i + 1}{j + 1}"],
                                        np.linspace(mixing_array_endpoints[i, n],
                                                    mixing_array_endpoints[j, n], res)))
                                    top_lines[f"{i + 1}{j + 1}"] = np.vstack((
                                        top_lines[f"{i + 1}{j + 1}"],
                                        np.linspace(mixing_array_tops[i, n],
                                                    mixing_array_tops[j, n], res)))
                                    bottom_lines[f"{i + 1}{j + 1}"] = np.vstack((
                                        bottom_lines[f"{i + 1}{j + 1}"],
                                        np.linspace(mixing_array_bottoms[i, n],
                                                    mixing_array_bottoms[j, n], res)))

            # Update self attribute
            self.mixing_arrays = mixing_lines
            self.top_arrays = top_lines
            self.bottom_arrays = bottom_lines

            # Initialize dataframes
            mixing_list = []
            top_list = []
            bottom_list = []

            # Write mixing lines to csv
            for i in range(len(mixing_array_endpoints)):
                for j in range(i + 1, len(mixing_array_endpoints)):
                    if (((i == 0) & (j == 1)) | ((i == 1) & (j == 2)) |
                        ((i == 2) & (j == 3)) | ((i == 3) & (j == 4)) |
                        ((i == 4) & (j == 5))):
                        # Create dataframe
                        mixing_synthetic = pd.DataFrame(
                            np.hstack((scaler.inverse_transform(pca.inverse_transform(
                                mixing_lines[f"{i + 1}{j + 1}"].T)),
                                mixing_lines[f"{i + 1}{j + 1}"].T)),
                            columns=oxides + [f"PC{n + 1}" for n in range(n_pca_components)]
                        ).round(3)
                        tops_synthetic = pd.DataFrame(
                            np.hstack((scaler.inverse_transform(pca.inverse_transform(
                                top_lines[f"{i + 1}{j + 1}"].T)),
                                top_lines[f"{i + 1}{j + 1}"].T)),
                            columns=oxides + [f"PC{n + 1}" for n in range(n_pca_components)]
                        ).round(3)
                        bottoms_synthetic = pd.DataFrame(
                            np.hstack((scaler.inverse_transform(pca.inverse_transform(
                                bottom_lines[f"{i + 1}{j + 1}"].T)),
                                bottom_lines[f"{i + 1}{j + 1}"].T)),
                            columns=oxides + [f"PC{n + 1}" for n in range(n_pca_components)]
                        ).round(3)

                        # Add sample id column
                        mixing_synthetic.insert(
                            0, "SAMPLEID", [f"sm{i + 1}{j + 1}{str(n).zfill(3)}" for
                                            n in range(len(mixing_synthetic))])
                        tops_synthetic.insert(
                            0, "SAMPLEID", [f"st{i + 1}{j + 1}{str(n).zfill(3)}" for
                                            n in range(len(tops_synthetic))])
                        bottoms_synthetic.insert(
                            0, "SAMPLEID", [f"sb{i + 1}{j + 1}{str(n).zfill(3)}" for
                                            n in range(len(bottoms_synthetic))])

                        # Append to list
                        mixing_list.append(mixing_synthetic)
                        top_list.append(tops_synthetic)
                        bottom_list.append(bottoms_synthetic)

            # Combine mixing arrays
            all_mixing = pd.concat(mixing_list, ignore_index=True)
            all_tops = pd.concat(top_list, ignore_index=True)
            all_bottoms = pd.concat(bottom_list, ignore_index=True)

            # No negative oxides
            data[oxides] = data[oxides].apply(lambda x: x.apply(lambda y: max(0.0, y)))
            all_mixing[oxides] = all_mixing[
                oxides].apply(lambda x: x.apply(lambda y: max(0.0, y)))
            all_tops[oxides] = all_tops[
                oxides].apply(lambda x: x.apply(lambda y: max(0.0, y)))
            all_bottoms[oxides] = all_bottoms[
                oxides].apply(lambda x: x.apply(lambda y: max(0.0, y)))

            # Calculate F melt
            ti_init = all_mixing.iloc[-1]["TIO2"]
            data["R_TIO2"] = round(data["TIO2"] / ti_init, 3)
            data["F_MELT_BATCH"] = round(
                ((D_tio2 / data["R_TIO2"]) - D_tio2) / (1 - D_tio2), 3)
            data["F_MELT_FRAC"] = round(1 - data["R_TIO2"]**(1 / ((1 / D_tio2) - 1)), 3)

            self.earthchem_pca = data.copy()

            # Write csv file
            data.to_csv(f"assets/data/earthchem-samples-pca.csv", index=False)

            all_mixing["R_TIO2"] = round(all_mixing["TIO2"] / ti_init, 3)
            all_mixing["F_MELT_BATCH"] = round(
                ((D_tio2 / all_mixing["R_TIO2"]) - D_tio2) / (1 - D_tio2), 3)
            all_mixing["F_MELT_FRAC"] = round(
                1 - all_mixing["R_TIO2"]**(1 / ((1 / D_tio2) - 1)), 3)

            all_tops["R_TIO2"] = round(all_tops["TIO2"] / ti_init, 3)
            all_tops["F_MELT_BATCH"] = round(
                ((D_tio2 / all_tops["R_TIO2"]) - D_tio2) / (1 - D_tio2), 3)
            all_tops["F_MELT_FRAC"] = round(
                1 - all_tops["R_TIO2"]**(1 / ((1 / D_tio2) - 1)), 3)

            all_bottoms["R_TIO2"] = round(all_bottoms["TIO2"] / ti_init, 3)
            all_bottoms["F_MELT_BATCH"] = round(
                ((D_tio2 / all_bottoms["R_TIO2"]) - D_tio2) / (1 - D_tio2), 3)
            all_bottoms["F_MELT_FRAC"] = round(
                1 - all_bottoms["R_TIO2"]**(1 / ((1 / D_tio2) - 1)), 3)

            # Write to csv
            fname = (f"assets/data/synthetic-samples-mixing-middle.csv")
            all_mixing.to_csv(fname, index=False)
            fname = (f"assets/data/synthetic-samples-mixing-tops.csv")
            all_tops.to_csv(fname, index=False)
            fname = (f"assets/data/synthetic-samples-mixing-bottoms.csv")
            all_bottoms.to_csv(fname, index=False)

            # Define bounding box around top and bottom mixing arrays
            min_x = min(mixing_array_tops[:, 0].min(), mixing_array_bottoms[:, 0].min())
            max_x = max(mixing_array_tops[:, 0].max(), mixing_array_bottoms[:, 0].max())

            # Define the sampling interval
            interval_x = (max_x - min_x) / res

            randomly_sampled_points = []
            sample_ids = []

            # Monte carlo sampling of synthetic samples
            for j in range(mc_sample):
                # Set seed
                np.random.seed(seed + j)

                # Create an array to store sampled points
                sampled_points = []
                sampled_weights = []

                # Iterate over x positions
                for x in np.linspace(min_x, max_x, res):
                    # Calculate the range of y values for the given x position
                    y_min = np.interp(x, mixing_array_tops[:, 0], mixing_array_tops[:, 1])
                    y_max = np.interp(x, mixing_array_bottoms[:, 0],
                                      mixing_array_bottoms[:, 1])
                    y_mid = (y_max + y_min) / 2

                    # Create a grid of y values for the current x position
                    y_values = np.linspace(y_min, y_max, res)

                    # Calculate exponential distance weights
                    point_weights = np.exp(-k * np.abs(y_values - y_mid))

                    # Combine x and y values to create points
                    points = np.column_stack((x * np.ones_like(y_values), y_values))

                    # Append points to the sampled_points array
                    sampled_points.extend(points)
                    sampled_weights.extend(point_weights)

                # Convert to np array
                sampled_points = np.array(sampled_points)
                sampled_weights = np.array(sampled_weights)

                # Define probability distribution for selecting random points
                if weighted_random:
                    prob_dist = sampled_weights / np.sum(sampled_weights)
                else:
                    prob_dist = None

                # Randomly select from sampled points
                sample_idx = np.random.choice(len(sampled_points), res, replace=False,
                                              p=prob_dist)
                randomly_sampled_points.append([sampled_points[i] for i in sample_idx])

                # Save random points
                sample_ids.extend([f"sr{j}{str(n).zfill(4)}" for n in range(len(sample_idx))])

            # Combine randomly sampled points
            randomly_sampled_points = np.vstack(randomly_sampled_points)

            # Create dataframe
            random_synthetic = pd.DataFrame(
                np.hstack((scaler.inverse_transform(pca.inverse_transform(
                    randomly_sampled_points)), randomly_sampled_points)),
                columns=oxides + [f"PC{n + 1}" for n in range(n_pca_components)]
            ).round(3)

            # Add sample id column
            random_synthetic.insert(0, "SAMPLEID", sample_ids)

            # Calculate F melt
            random_synthetic["R_TIO2"] = round(random_synthetic["TIO2"] / ti_init, 3)
            random_synthetic["F_MELT_BATCH"] = round(
                ((D_tio2 / random_synthetic["R_TIO2"]) - D_tio2) / (1 - D_tio2), 3)
            random_synthetic["F_MELT_FRAC"] = round(
                1 - random_synthetic["R_TIO2"]**(1 / ((1 / D_tio2) - 1)), 3)

            # Write to csv
            fname = (f"assets/data/synthetic-samples-mixing-random.csv")
            random_synthetic.to_csv(fname, index=False)

            # Update attribute
            self.synthetic_data_written = True

            return None

        except Exception as e:
            print("Error occurred when computing mixing arrays!")
            traceback.print_exc()

            self.mixing_array_error = True
            self.error = e

            return None
