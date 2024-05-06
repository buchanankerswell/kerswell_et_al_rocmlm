#######################################################
## .0.              Load Libraries               !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# utilities !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import re
import time
import random
import shutil
import itertools
import subprocess
from datetime import datetime

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# parallel computing !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import multiprocessing as mp
mp.set_start_method("fork")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# dataframes and arrays !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# machine learning !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn.impute import KNNImputer
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score, mean_squared_error

#######################################################
## .1.             Helper Functions              !!! ##
#######################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# get unique value !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_unique_value(input_list):
    """
    """
    unique_value = input_list[0]

    for item in input_list[1:]:
        if item != unique_value:
            raise ValueError("Not all values are the same!")

    return unique_value

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# get sampleids !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_sampleids(filepath, batch, n_batches=8):
    """
    """
    # Check for file
    if not os.path.exists(filepath):
        raise Exception("Sample data source does not exist!")

    # Read data
    df = pd.read_csv(filepath)

    if "benchmark" in filepath:
        return df["SAMPLEID"].values

    if batch == "all":
        return df["SAMPLEID"].values

    # Calculate the batch size
    total_samples = len(df)
    batch_size = int(total_samples // n_batches)

    # Check if batch number is within valid range
    if batch < 0 or batch >= n_batches:
        print("Invalid batch number! Sampling from the first 0th batch ...")

        batch = 0

    # Calculate the start and end index for the specified batch
    start = batch * batch_size
    end = min((batch + 1) * batch_size, total_samples)

    return df[start:end]["SAMPLEID"].values

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# get geotherm !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_geotherm(results, target, threshold, Qs=55e-3, Ts=273, A1=1e-6, A2=2.2e-8, k1=2.3,
                 k2=3.0, crust_thickness=35, litho_thickness=150, mantle_potential=1573):
    """
    """
    # Get P results
    P = results["P"]

    # Get PT and target values and transform units
    df = pd.DataFrame({"P": results["P"], "T": results["T"],
                       target: results[target]}).sort_values(by="P")

    # Geotherm Parameters
    Z_min = np.min(P) * 35e3
    Z_max = np.max(P) * 35e3
    z = np.linspace(Z_min, Z_max, len(P))
    T_geotherm = np.zeros(len(P))

    # Layer1 (crust)
    # A1 Radiogenic heat production (W/m^3)
    # k1 Thermal conductivity (W/mK)
    D1 = crust_thickness * 1e3 # Thickness (m)

    # Layer2 (lithospheric mantle)
    # A2 Radiogenic heat production (W/m^3)
    # k2 Thermal conductivity (W/mK)
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
        potential_temp = mantle_potential + 0.5e-3 * z[j]
        if z[j] <= D1:
            T_geotherm[j] = Tt1 + (Qt1 / k1 * z[j]) - (A1 / (2 * k1) * z[j]**2)
            if T_geotherm[j] >= potential_temp:
                T_geotherm[j] = potential_temp
        elif D1 < z[j] <= D2 + D1:
            T_geotherm[j] = Tt2 + (Qt2 / k2 * (z[j] - D1)) - (A2 / (2 * k2) *
                                                              (z[j] - D1)**2)
            if T_geotherm[j] >= potential_temp:
                T_geotherm[j] = potential_temp
        elif z[j] > D2 + D1:
            T_geotherm[j] = Tt3 + 0.5e-3 * (z[j] - D1 - D2)
            if T_geotherm[j] >= potential_temp:
                T_geotherm[j] = potential_temp

    P_geotherm = np.round(z / 35e3, 1)
    T_geotherm = np.round(T_geotherm, 2)

    df["geotherm_P"] = P_geotherm
    df["geotherm_T"] = T_geotherm

    # Subset df along geotherm
    df = df[abs(df["T"] - df["geotherm_T"]) < threshold]

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
    ref_paths = {"prem": f"{data_dir}/PREM_1s.csv", "stw105": f"{data_dir}/STW105.csv"}

    # Define column headers
    prem_cols = ["radius", "depth", "rho", "Vp", "Vph", "Vs", "Vsh", "eta", "Q_mu",
                 "Q_kappa"]
    stw105_cols = ["radius", "rho", "Vp", "Vs", "unk1", "unk2", "Vph", "Vsh", "eta"]

    ref_cols = {"prem": prem_cols, "stw105": stw105_cols}
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
            model["depth"] = (model["radius"].max() - model["radius"]) / 1e3
            model["rho"] = model["rho"] / 1e3
            model["Vp"] = model["Vp"] / 1e3
            model["Vs"] = model["Vs"] / 1e3
            model.sort_values(by=["depth"], inplace=True)

        def calculate_pressure(row):
            z = row["depth"]
            depths = model[model["depth"] <= z]["depth"] * 1e3
            rhos = model[model["depth"] <= z]["rho"] * 1e3
            rho_integral = np.trapz(rhos, x=depths)
            pressure = 9.81 * rho_integral / 1e9
            return pressure

        model["P"] = model.apply(calculate_pressure, axis=1)

        # Clean up df
        model = model[columns_to_keep]
        model = model.round(3)

        # Save model
        ref_models[name] = model

    # Invert STW105
    ref_models["stw105"] = ref_models["stw105"].sort_values(by="depth", ascending=True)

    return ref_models

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# measure gfem model accuracy vs prem !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def measure_gfem_model_accuracy_vs_prem(gfem_model, filename):
    """
    """
    # Get model attributes
    sample_id = gfem_model.sample_id
    program = gfem_model.program
    dataset = gfem_model.dataset
    res = gfem_model.res
    results_model = gfem_model.results
    targets = ["rho", "Vp", "Vs"]
    verbose = gfem_model.verbose

    # Data asset dir
    data_dir = "assets/data"

    # Check for data dir
    if not os.path.exists(data_dir):
        raise Exception(f"Data not found at {data_dir}!")

    # Get 1D reference models
    ref_models = get_1d_reference_models()

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

    # Initialize metrics lists
    smpid, pgrm, trgt, profile_section = [], [], [], []
    rmse_prem_profile, r2_prem_profile = [], []
    rmse_stw105_profile, r2_stw105_profile = [], []

    crop_profile = {"1–28 GPa": (1., 28.), "10–16 GPa": (10., 16.), "20–26 GPa": (20., 26.)}

    for target in targets:
        for name, bounds in crop_profile.items():
            # Get 1D refernce model profiles
            P_prem, target_prem = ref_models["prem"]["P"], ref_models["prem"][target]
            P_stw105, target_stw105 = ref_models["stw105"]["P"], ref_models["stw105"][target]

            # Get model profile
            P_model, _, target_model = get_geotherm(results_model, target,
                                                    geotherm_threshold)

            # Create cropping mask
            mask_prem = (P_prem >= bounds[0]) & (P_prem <= bounds[1])
            mask_stw105 = (P_stw105 >= bounds[0]) & (P_stw105 <= bounds[1])
            mask_model = (P_model >= bounds[0]) & (P_model <= bounds[1])

            # Crop profiles
            P_prem, target_prem = P_prem[mask_prem], target_prem[mask_prem]
            P_stw105, target_stw105 = P_stw105[mask_stw105], target_stw105[mask_stw105]
            P_model, target_model = P_model[mask_model], target_model[mask_model]

            # Initialize interpolators
            interp_prem = interp1d(P_prem, target_prem, fill_value="extrapolate")
            interp_stw105 = interp1d(P_stw105, target_stw105, fill_value="extrapolate")

            # New x values for interpolation
            x_new = np.linspace(bounds[0], bounds[1], len(P_model))

            # Interpolate profiles
            P_prem, target_prem = x_new, interp_prem(x_new)
            P_stw105, target_stw105 = x_new, interp_stw105(x_new)

            # Create nan mask
            nan_mask_model = np.isnan(target_model)
            nan_mask_prem = np.isnan(target_prem)
            nan_mask_stw105 = np.isnan(target_stw105)
            nan_mask = np.logical_or(nan_mask_model,
                                     np.logical_or(nan_mask_prem, nan_mask_stw105))

            # Remove nans
            P_model, target_model = P_model[~nan_mask], target_model[~nan_mask]
            P_prem, target_prem = P_prem[~nan_mask], target_prem[~nan_mask]
            P_stw105, target_stw105 = P_stw105[~nan_mask], target_stw105[~nan_mask]

            # Calculate rmse and r2 along profiles
            rmse_prem = np.sqrt(mean_squared_error(target_prem, target_model))
            rmse_prem_profile.append(np.round(rmse_prem, 3))
            r2_prem_profile.append(np.round(r2_score(target_prem, target_model), 3))

            rmse_stw105 = np.sqrt(mean_squared_error(target_stw105, target_model))
            rmse_stw105_profile.append(np.round(rmse_stw105, 3))
            r2_stw105_profile.append(np.round(r2_score(target_stw105, target_model), 3))

            # Save other results info
            smpid.append(sample_id)
            pgrm.append(program)
            trgt.append(target)
            profile_section.append(name)

    # Save results
    results = {"SAMPLEID": smpid, "PROGRAM": pgrm, "TARGET": trgt,
               "SECTION": profile_section,
               "RMSE_PREM": rmse_prem_profile, "R2_PREM": r2_prem_profile,
               "RMSE_STW105": rmse_stw105_profile, "R2_STW105": r2_stw105_profile}

    # Create dataframe
    df = pd.DataFrame(results)

    # Write csv
    if os.path.exists(filename) and os.stat(filename).st_size > 0:
        try:
            df_existing = pd.read_csv(filename)

            if df_existing.empty:
                if verbose >= 1:
                    print(f"Saving {sample_id} RMSE vs. PREM to {filename}!")

                df.to_csv(filename, index=False)

            else:
                # Check existing samples
                new_samples = df["SAMPLEID"].values
                existing_samples = df_existing["SAMPLEID"].values
                overlap = set(existing_samples).intersection(new_samples)

                if overlap:
                    if verbose >= 1:
                        print(f"Sample {sample_id} already saved to {filename}!")

                else:
                    if verbose >= 1:
                        print(f"Saving {sample_id} RMSE vs. PREM to {filename}!")

                    df_existing = pd.concat([df_existing, df], ignore_index=True)
                    df_existing = df_existing.sort_values(by=["SAMPLEID", "TARGET"],
                                                          ignore_index=True)
                    df_existing.to_csv(filename, index=False)

        except pd.errors.EmptyDataError:
            if verbose >= 1:
                print(f"Saving {sample_id} RMSE vs. PREM to {filename}!")

            df.to_csv(filename, index=False)

    else:
        if verbose >= 1:
            print(f"Saving {sample_id} RMSE vs. PREM to {filename}!")

        df.to_csv(filename, index=False)

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# gfem_iteration !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def gfem_iteration(args):
    """
    """

    # Unpack arguments
    program, perplex_db, dataset, sampleid, source, res, Pmin, Pmax, Tmin, Tmax, \
        oxides_exclude, targets, maskgeotherm, verbose, debug = args

    # Initiate GFEM model
    iteration = GFEMModel(program, perplex_db, dataset, sampleid, source, res, Pmin, Pmax,
                          Tmin, Tmax, oxides_exclude, targets, maskgeotherm, verbose, debug)

    if iteration.model_built:
        return iteration

    else:
        iteration.build_model()

        if not iteration.model_build_error:
            iteration.get_results()
            iteration.get_feature_array()
            iteration.get_target_array()
            return iteration

        else:
            return iteration

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# build gfem models !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def build_gfem_models(source, sampleids=None, programs=["perplex"], perplex_db="stx21",
                      datasets=["train", "valid"], batch="all", nbatches=8, res=128, Pmin=1,
                      Pmax=28, Tmin=773, Tmax=2273, oxides_exclude=["H2O", "FE2O3"],
                      targets=["rho", "Vp", "Vs"], maskgeotherm=False, parallel=True,
                      nprocs=os.cpu_count(), verbose=1, debug=True):
    """
    """
    # Check sampleids
    if os.path.exists(source) and sampleids is None:
        sampleids = sorted(get_sampleids(source, batch, nbatches))

    elif os.path.exists(source) and sampleids is not None:
        sids = get_sampleids(source, batch, nbatches)

        if not set(sampleids).issubset(sids):
            raise Exception(f"Sampleids {sampleids} not in source: {source}!")

    else:
        raise Exception(f"Source {source} does not exist!")

    # CSV path
    filename = "assets/data/gfem-accuracy-vs-prem.csv"

    # Write blank CSV
    if not os.path.exists(filename):
        columns = ["SAMPLEID", "PROGRAM", "TARGET", "RMSE_PREM_PROFILE", "R2_PREM_PROFILE",
                   "RMSE_STW105_PROFILE", "R2_STW105_PROFILE"]
        df = pd.DataFrame(columns=columns)
        df.to_csv(filename, index=False, header=True)

    print("Building GFEM models for samples:")
    print(sampleids)

    # Create combinations of samples and datasets
    combinations = list(itertools.product(programs, datasets, sampleids))

    # Define number of processors
    if not parallel:
        nprocs = 1

    elif parallel:
        if nprocs is None or nprocs > os.cpu_count():
            nprocs = os.cpu_count()

        else:
            nprocs = nprocs

    # Make sure nprocs is not greater than sample combinations
    if nprocs > len(combinations):
        nprocs = len(combinations)

    # Create list of args for mp pooling
    run_args = [(program, perplex_db, dataset, sampleid, source, res, Pmin, Pmax, Tmin, Tmax,
                 oxides_exclude, targets, maskgeotherm, verbose, debug) for
                program, dataset, sampleid in combinations]

    # Create a multiprocessing pool
    with mp.Pool(processes=nprocs) as pool:
        models = pool.map(gfem_iteration, run_args)

        # Wait for all processes
        pool.close()
        pool.join()

    # Get successful models
    successful_models = [model for model in models if not model.model_build_error]

    # Measure GFEM model accuracy vs. PREM
    for model in successful_models:
        if model.dataset == "train":
            measure_gfem_model_accuracy_vs_prem(model, filename)

    # Check for errors in the models
    error_count = 0

    for model in models:
        if model.model_build_error:
            error_count += 1

    if error_count > 0:
        print(f"Total models with errors: {error_count}")

    else:
        print("All GFEM models built successfully!")

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    return successful_models

#######################################################
## .2.              GFEMModel class              !!! ##
#######################################################
class GFEMModel:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # init !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, program, perplex_db, dataset, sample_id, source, res, P_min=1,
                 P_max=28, T_min=773, T_max=2273, oxides_exclude=["H2O", "FE2O3"],
                 targets=["rho", "Vp", "Vs"], maskgeotherm=False, seed=42, verbose=1,
                 debug=True):
        """
        """
        # Input
        self.program = program
        self.P_min = P_min
        self.P_max = P_max
        self.T_min = T_min
        self.T_max = T_max
        self.res = res
        self.source = source
        self.sample_id = sample_id
        self.oxides_exclude = oxides_exclude
        self.dataset = dataset
        self.targets = targets
        self.mask_geotherm = maskgeotherm
        self.seed = seed
        self.verbose = verbose
        self.debug = debug

        # Decimals
        self.digits = 3

        # Check perplex db
        if perplex_db not in ["hp02", "hp633", "stx21"]:
            self.perplex_db = "hp633"
        else:
            self.perplex_db = perplex_db

        # Adjust oxides for HP02 dataset
        if perplex_db == "hp02":
            self.oxides_exclude = ["CR2O3", "H2O", "FE2O3"]

        # Adjust oxides for STX21 dataset
        if perplex_db == "stx21":
            self.oxides_exclude = ["K2O", "TIO2", "CR2O3", "H2O", "FE2O3"]

        # System oxide components
        self.oxides_system = ["SIO2", "AL2O3", "CAO", "MGO", "FEO", "K2O", "NA2O", "TIO2",
                              "FE2O3", "CR2O3", "H2O"]

        # Features
        self.features = ["PC1", "PC2", "R_TIO2", "F_MELT_BATCH", "D_BATCH", "F_MELT_FRAC",
                         "D_FRAC"]

        # Program and config paths
        if self.program == "magemin":
            # Magemin dirs and filepaths
            self.model_out_dir = (f"gfems/{self.program[:4]}_{self.sample_id}_"
                                  f"{self.dataset[0]}{self.res}")
            self.magemin_in_path = f"{self.model_out_dir}/in.dat"
            self.magemin_out_path = f"{self.model_out_dir}/output/_pseudosection_output.txt"

        elif self.program == "perplex":
            # Perplex dirs and filepaths
            self.model_out_dir = (f"gfems/{self.program[:4]}_{self.perplex_db}_"
                                  f"{self.sample_id}_{self.dataset[0]}{self.res}")
            self.perplex_targets = f"{self.model_out_dir}/target-array.tab"
            self.perplex_assemblages = f"{self.model_out_dir}/assemblages.txt"

        else:
            raise ValueError("Unrecognized GFEM program! Use 'magemin' or 'perplex' ...")

        # Output file paths
        self.model_prefix = f"{self.sample_id}-{self.dataset[0]}{self.res}"
        self.fig_dir = f"figs/gfem/{self.program[:4]}_{self.sample_id}_{self.res}"
        self.log_file = f"log/log-{self.program[:4]}-{self.model_prefix}"

        # Results
        self.sample_composition = []
        self.fertility_index = None
        self.sample_features = []
        self.norm_sample_composition = []
        self.comp_time = None
        self.model_built = False
        self.results = {}
        self.feature_array = np.array([])
        self.feature_array_unmasked = np.array([])
        self.target_array = np.array([])
        self.target_array_unmasked = np.array([])

        # Errors
        self.timeout = (res**2) * 3
        self.model_build_error = False
        self.model_error = None

        # Check for existing model build
        if os.path.exists(self.model_out_dir):
            if (os.path.exists(f"{self.model_out_dir}/results.csv") and
                os.path.exists(f"{self.model_out_dir}/assemblages.csv")):
                try:
                    self.model_built = True
                    self._get_sample_composition()
                    self._normalize_sample_composition()
                    self._get_sample_features()
                    self.get_results()
                    self.get_feature_array()
                    self.get_target_array()

                except Exception as e:
                    print(f"!!! {e} !!!")

                    return None

                if self.verbose >= 1:
                    print(f"Found [{self.program}] model for sample {self.model_prefix}!")

            else:
                # Make new model if results not found
                shutil.rmtree(self.model_out_dir)
                os.makedirs(self.model_out_dir, exist_ok=True)
        else:
            os.makedirs(self.model_out_dir, exist_ok=True)

        # Set np array printing option
        np.set_printoptions(precision=3, suppress=True)

        return None

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .0.1.          Helper Functions              !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get sample composition !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_sample_composition(self):
        """
        """
        # Get self attributes
        program = self.program
        source = self.source
        sample_id = self.sample_id
        oxides = self.oxides_system
        oxide_zero = 0.0001 if program == "magemin" else float(0)

        # Read the data file
        df = pd.read_csv(source)

        # Subset the DataFrame based on the sample name
        subset_df = df[df["SAMPLEID"] == sample_id]

        if subset_df.empty:
            raise ValueError("Sample name not found in the dataset!")

        # Get Fertility Index
        self.fertility_index = float(subset_df["D_FRAC"].values)

        # Get the oxide compositions for the selected sample
        composition = []

        for oxide in oxides:
            if oxide in subset_df.columns and pd.notnull(subset_df[oxide].iloc[0]):
                composition.append(float(subset_df[oxide].iloc[0]))

            else:
                composition.append(oxide_zero)

        self.sample_composition = composition

        return composition

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # normalize sample composition !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _normalize_sample_composition(self):
        """
        """
        # Get self attributes
        program = self.program
        sample_composition = self.sample_composition
        oxides_exclude = self.oxides_exclude
        oxides = self.oxides_system
        oxide_zero = 0.0001 if program == "magemin" else float(0)
        subset_oxides = [oxide for oxide in oxides if oxide not in oxides_exclude]
        digits = self.digits

        # Check for sample composition
        if not sample_composition:
            raise Exception("No sample found! Call _get_sample_composition() first ...")

        # No normalizing for all components
        if not oxides_exclude:
            self.norm_sample_composition = sample_composition

            return sample_composition

        # Check input
        if len(sample_composition) != len(oxides):
            error_message = (f"The input sample list must have exactly {len(oxides)} "
                             f"components!\n{oxides}")

            raise ValueError(error_message)

        # Filter components
        if program == "magemin":
            subset_sample = [comp if oxide in subset_oxides else oxide_zero for
                             comp, oxide in zip(sample_composition, oxides)]
        else:
            subset_sample = [comp for comp, oxide in zip(sample_composition, oxides) if
                            oxide in subset_oxides]

        # Set negative compositions to zero
        subset_sample = [comp if comp >= oxide_zero else oxide_zero for comp in
                         subset_sample]

        # Get total oxides
        total_subset_concentration = sum([comp for comp in subset_sample if
                                          comp != oxide_zero])

        # Normalize
        if program == "magemin":
            normalized_concentrations = [
                round(((comp / total_subset_concentration) * 100 if comp != oxide_zero else
                       oxide_zero), digits) for comp, oxide in zip(subset_sample, oxides)
            ]
        else:
            normalized_concentrations = [
                round(((comp / total_subset_concentration) * 100 if comp != oxide_zero else
                       oxide_zero), digits) for comp, oxide in
                zip(subset_sample, subset_oxides)
            ]

        self.norm_sample_composition = normalized_concentrations

        return normalized_concentrations

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get sample features !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_sample_features(self):
        """
        """
        # Get self attributes
        program = self.program
        source = self.source
        sample_id = self.sample_id
        features = self.features
        norm_sample_composition = self.norm_sample_composition

        # Check for sample composition
        if not norm_sample_composition:
            raise Exception("Sample composition not normalized yet! "
                            "Call _normalize_sample_composition() first ...")

        # Read the data file
        df = pd.read_csv(source)

        # Subset the DataFrame based on the sample name
        subset_df = df[df["SAMPLEID"] == sample_id]

        if subset_df.empty:
            raise ValueError("Sample name not found in the dataset!")

        # Get features for selected sample
        feature_values = subset_df[features].values.flatten().tolist()
        sample_features = norm_sample_composition + feature_values

        self.sample_features = sample_features

        return sample_features

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # count lines !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _count_lines(self):
        """
        """
        line_count = 0

        with open(self.magemin_in_path, "r") as file:
            for line in file:
                line_count += 1

        return line_count

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # replace in file !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _replace_in_file(self, filepath, replacements):
        """
        """
        with open(filepath, "r") as file:
            file_data = file.read()

            for key, value in replacements.items():
                file_data = file_data.replace(key, value)

        with open(filepath, "w") as file:
            file.write(file_data)

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get comp time !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_comp_time(self):
        """
        """
        # Get self attributes
        log_file = self.log_file
        program = self.program
        sample_id = self.sample_id
        dataset = self.dataset
        res = self.res

        if os.path.exists(log_file) and os.path.exists("assets/data"):
            # Define a list to store the time values
            time_values_mgm = []
            time_values_ppx = []

            # Open the log file and read its lines
            with open(log_file, "r") as log:
                lines = log.readlines()

            # Iterate over the lines in reverse order
            for line in reversed(lines):
                if "MAGEMin comp time:" in line:
                    match = re.search(r"\+([\d.]+) ms", line)

                    if match:
                        time_ms = float(match.group(1))
                        time_s = time_ms / 1e3

                        time_values_mgm.append(time_s)

                    break

            for line in reversed(lines):
                if "Total elapsed time" in line:
                    match = re.search(r"\s+([\d.]+)", line)

                    if match:
                        time_m = float(match.group(1))
                        time_s = time_m * 60

                        time_values_ppx.append(time_s)

                    break

            if program == "magemin":
                last_value = time_values_mgm[-1]

            elif program == "perplex":
                last_value = time_values_ppx[-1]

            # Create the line to append to the CSV file
            line_to_append = (f"{sample_id},{program},{dataset},"
                              f"{res**2},{last_value:.1f}")

            date_formatted = datetime.now().strftime("%d-%m-%Y")
            csv_filepath = f"assets/data/gfem-efficiency-{date_formatted}.csv"

            # Check if the CSV file already exists
            if not os.path.exists(csv_filepath):
                header_line = "sample,program,dataset,size,time"

                # If the file does not exist, write the header line first
                with open(csv_filepath, "w") as csv_file:
                    csv_file.write(header_line + "\n")

            # Append the line to the CSV file
            with open(csv_filepath, "a") as csv_file:
                csv_file.write(line_to_append + "\n")

            self.comp_time = round(last_value, 3)

            return round(last_value, 3)

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # encode assemblages !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _encode_assemblages(self, assemblages):
        """
        """
        unique_assemblages = {}
        encoded_assemblages = []

        # Encoding unique phase assemblages
        for assemblage in assemblages:
            assemblage_tuple = tuple(sorted(assemblage))

            if assemblage_tuple not in unique_assemblages:
                unique_assemblages[assemblage_tuple] = len(unique_assemblages) + 1

        # Create dataframe
        df = pd.DataFrame(list(unique_assemblages.items()), columns=["assemblage", "index"])

        # Put spaces between phases
        df["assemblage"] = df["assemblage"].apply(" ".join)

        # Reorder columns
        df = df[["index", "assemblage"]]

        # Save to csv
        assemblages_csv = f"{self.model_out_dir}/assemblages.csv"
        df.to_csv(assemblages_csv, index=False)

        # Encoding phase assemblage numbers
        for assemblage in assemblages:
            if assemblage == "":
                encoded_assemblages.append(np.nan)

            else:
                encoded_assemblage = unique_assemblages[tuple(sorted(assemblage))]
                encoded_assemblages.append(encoded_assemblage)

        return encoded_assemblages

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .0.2.           MAGEMin Functions            !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # configure magemin model !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _configure_magemin_model(self):
        """
        """
        # Get self attributes
        dataset = self.dataset
        res = self.res
        magemin_in_path = self.magemin_in_path

        # Transform units to kbar C
        P_min, P_max = self.P_min * 10, self.P_max * 10
        T_min, T_max = self.T_min - 273, self.T_max - 273

        # Get sample composition
        sample_comp = self._get_sample_composition()
        norm_comp = self._normalize_sample_composition()

        # Shift validation dataset
        if dataset != "train":
            P_step, T_step = 1, 25

            P_min += P_step
            P_max -= P_step
            T_min += T_step
            T_max -= T_step

        # Define PT arrays
        P_range = [P_min, P_max, (P_max - P_min) / res]
        T_range = [T_min, T_max, (T_max - T_min) / res]

        P_array = np.arange(float(P_range[0]),
                            float(P_range[1]) + float(P_range[2]),
                            float(P_range[2])).round(3)

        T_array = np.arange(float(T_range[0]),
                            float(T_range[1]) + float(T_range[2]),
                            float(T_range[2])).round(3)

        # Setup PT vectors
        magemin_input = ""

        # Expand PT vectors into grid
        combinations = list(itertools.product(P_array, T_array))

        for p, t in combinations:
            magemin_input += (
                f"0 {p} {t} "
                f"{norm_comp[0]} {norm_comp[1]} {norm_comp[2]} "
                f"{norm_comp[3]} {norm_comp[4]} {norm_comp[5]} "
                f"{norm_comp[6]} {norm_comp[7]} {norm_comp[8]} "
                f"{norm_comp[9]} {norm_comp[10]}\n"
            )

        # Write input file
        with open(magemin_in_path, "w") as f:
            f.write(magemin_input)

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # run magemin !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _run_magemin(self):
        """
        """
        # Get self attributes
        norm_sample_composition = self.norm_sample_composition
        oxides = self.oxides_system
        oxides_exclude = self.oxides_exclude
        subset_oxides = [oxide for oxide in oxides if oxide not in oxides_exclude]
        magemin_in_path = self.magemin_in_path
        model_out_dir = self.model_out_dir
        model_prefix = self.model_prefix
        log_file = self.log_file
        timeout = self.timeout
        verbose = self.verbose

        # Check for input MAGEMin input files
        if not os.path.exists(magemin_in_path):
            raise Exception("No MAGEMin input! Call _configure_magemin_model() first ...")

        if verbose >= 1:
            print(f"Building MAGEMin model: {model_prefix}...")
            max_oxide_width = max(len(oxide) for oxide in oxides)
            max_comp_width = max(len(str(comp)) for comp in norm_sample_composition)
            max_width = max(max_oxide_width, max_comp_width)
            print(" ".join([f"  {oxide:<{max_width}}" for oxide in oxides]))
            print(" ".join([f"  {comp:<{max_width}}" for comp in norm_sample_composition]))

        # Get number of pt points
        n_points = self._count_lines()

        # Execute MAGEMin
        exec = (f"../../MAGEMin/MAGEMin --File=../../{magemin_in_path} "
                f"--n_points={n_points} --sys_in=wt --db=ig")

        try:
            # Run MAGEMin
            process = subprocess.Popen([exec], stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, shell=True,
                                       cwd=model_out_dir)

            # Wait for the process to complete and capture its output
            stdout, stderr = process.communicate(timeout=timeout)

            if verbose >= 2:
                print(f"{stdout.decode()}")

            # Write to logfile
            with open(log_file, "a") as log:
                log.write(stdout.decode())
                log.write(stderr.decode())

            if process.returncode != 0:
                raise RuntimeError(f"Error executing magemin {model_out_dir}!")

            if verbose >= 2:
                print(f"MAGEMin output:")
                print(f"{stdout.decode()}")

        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # read magemin output !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _read_magemin_output(self):
        """
        """
        # Get self attributes
        magemin_out_path = self.magemin_out_path

        # Open file
        with open(magemin_out_path, "r") as file:
            lines = file.readlines()

        # Skip the comment line
        lines = lines[1:]

        # Initialize results
        results = []

        # Read lines
        while lines:
            # Get line with PT, Gamma, etc.
            line = lines.pop(0)

            # Split line at whitespace and transform into floats
            a = list(map(float, line.split()))

            # Get results
            num_point = int(a[0]) # PT point
            status = int(a[1]) # solution status
            p = a[2] # P kbar
            t = a[3] # T celcius
            gibbs = a[4] # gibbs free energy J
            br_norm = a[5] # normalized mass residual
            gamma = a[6:15] # chemical potential of pure components (gibbs hyperplane)
            unknown = a[16] # unknown parameter following gamma in output ???
            vp = a[17] # P-wave velocity km/s
            vs = a[18] # S-wave velocity km/s
            entropy = a[19] # entropy J/K

            # Initialize empty lists for stable solutions and endmember info
            assemblage = []
            assemblage_mode = []
            assemblage_rho = []
            compositional_vars = []
            em_fractions = []
            em_list = []

            # Get line with stable solutions and endmember info
            line = lines.pop(0)

            # Read line
            while line.strip():
                # Split line at whitespace
                out = line.split()

                # Initialize empty list to store stable solutions info
                data = []

                # Store stable solutions info as floats or strings (if not numeric)
                for value in out:
                    try:
                        data.append(float(value))

                    except ValueError:
                        data.append(value)

                # Store stable solutions info
                assemblage.append(data[0]) # phase assemblage
                assemblage_mode.append(data[1]) # assemblage mode
                assemblage_rho.append(data[2]) # assemblage density

                # Initialize empty lists for endmember info
                comp_var = []
                em_frac = []
                em = []

                if len(data) > 4:
                    n_xeos = int(data[3]) # number of compositional variables
                    comp_var = data[4:4 + n_xeos] # compositional variables
                    em = out[4 + n_xeos::2] # endmembers
                    em_frac = data[5 + n_xeos::2] # endmember fractions

                # Store endmember info
                compositional_vars.append(comp_var)
                em_fractions.append(em_frac)
                em_list.append(em)

                line = lines.pop(0)

            # Get indices of melt
            ind_liq = [idx for idx, sol in enumerate(assemblage) if sol == "liq"]

            # Get melt fraction
            if ind_liq:
                liq = assemblage_mode[ind_liq[0]]

                if liq <= 0.01:
                    liq = np.nan

            else:
                liq = np.nan

            # Compute average density of full assemblage
            rho_total = sum(mode * rho for mode, rho in zip(assemblage_mode, assemblage_rho))

            # Append results dictionary
            results.append({"point": num_point, # point
                            "T": t, # temperature celcius
                            "P": p, # pressure kbar
                            "rho": rho_total, # density of full assemblage kg/m3
                            "Vp": vp, # pressure wave velocity km/s
                            "Vs": vs, # shear wave velocity km/s
                            "melt": liq, # melt fraction
                            "assemblage": assemblage, # stable assemblage
                            })

        # Merge lists within dictionary
        combined_results = {key: [d[key] for d in results] for key in results[0]}

        return combined_results

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # process magemin results !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _process_magemin_results(self):
        """
        """
        # Get self attributes
        model_out_dir = self.model_out_dir
        magemin_in_path = self.magemin_in_path
        magemin_out_path = self.magemin_out_path
        model_prefix = self.model_prefix
        verbose = self.verbose
        debug = self.debug

        # Check for MAGEMin output files
        if not os.path.exists(model_out_dir):
            raise Exception("No MAGEMin files to process! Call _run_magemin() first ...")

        if not os.path.exists(magemin_out_path):
            raise Exception("No MAGEMin files to process! Call _run_magemin() first ...")

        if verbose >= 2:
            print(f"Reading MAGEMin output: {model_prefix} ...")

        # Read results
        results = self._read_magemin_output()

        # Remove point index
        results.pop("point")

        # Compute assemblage variance (number of phases)
        assemblages = results.get("assemblage")

        variance = []

        for assemblage in assemblages:
            unique_phases = set(assemblage)
            count = len(unique_phases)

            variance.append(count)

        # Add assemblage variance to merged results
        results["variance"] = variance

        # Encode assemblage
        encoded_assemblages = self._encode_assemblages(assemblages)

        # Replace assemblage with encoded assemblages
        results["assemblage"] = encoded_assemblages

        # Point results that can be converted to numpy arrays
        point_params = ["T", "P", "rho", "Vp", "Vs", "melt", "assemblage", "variance"]

        # Convert numeric point results into numpy arrays
        for key, value in results.items():
            if key in point_params:
                if key == "P":
                    # Convert from kbar to GPa
                    results[key] = np.round(np.array(value) / 10, 3)

                elif key == "T":
                    # Convert from C to K
                    results[key] = np.round(np.array(value) + 273, 3)

                elif key == "rho":
                    # Convert from kg/m3 to g/cm3
                    results[key] = np.round(np.array(value) / 1e3, 3)

                elif key == "melt":
                    # Convert from fraction to percent
                    results[key] = np.round(np.array(value) * 100, 3)

                else:
                    results[key] = np.array(value)

        # Save as pandas df
        df = pd.DataFrame.from_dict(results)

        if verbose >= 2:
            print(f"Writing MAGEMin results: {model_out_dir} ...")

        # Write to csv file
        df.to_csv(f"{model_out_dir}/results.csv", index=False)

        if debug:
            # Create dir to store model files
            model_out_files_dir = f"{model_out_dir}/model"
            os.makedirs(model_out_files_dir, exist_ok=True)

            # Clean up output directory
            shutil.move(magemin_in_path, model_out_files_dir)
            shutil.move(f"{model_out_dir}/output", model_out_files_dir)

        else:
            # Clean up output directory
            os.remove(magemin_in_path)
            shutil.rmtree(f"{model_out_dir}/output")

        return None

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .0.3.          Perple_X Functions            !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # configure perplex model !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _configure_perplex_model(self):
        """
        """
        # Get self attributes
        perplex_db = self.perplex_db
        dataset = self.dataset
        model_out_dir = self.model_out_dir
        model_prefix = self.model_prefix
        digits = self.digits
        res = self.res

        # Transform units to bar
        P_min, P_max = self.P_min * 1e4, self.P_max * 1e4
        T_min, T_max = self.T_min, self.T_max

        # Get sample composition
        sample_comp = self._get_sample_composition()
        norm_comp = self._normalize_sample_composition()

        # Shift validation dataset
        if dataset != "train":
            P_step, T_step = 1e3, 25
            comp_adj = 0.01

            P_min += P_step
            P_max -= P_step
            T_min += T_step
            T_max -= T_step

            random.seed(42)
            norm_comp = [round(x * 1 + comp_adj, digits) for x in norm_comp]

        # Config dir
        config_dir = f"assets/config_{perplex_db}"

        # Configuration files
        build = "build-config"
        thermodb = "td-data"
        solutions = "solution-models"
        minimize = "vertex-minimize"
        targets = "werami-targets"
        phase = "werami-phase"
        options = "build-options"
        draw = "pssect-draw"
        plot = "plot-options"

        # Copy original configuration files to the perplex directory
        shutil.copy(f"{config_dir}/{build}", f"{model_out_dir}/{build}")
        shutil.copy(f"{config_dir}/{thermodb}", f"{model_out_dir}/{thermodb}")
        shutil.copy(f"{config_dir}/{solutions}", f"{model_out_dir}/{solutions}")
        shutil.copy(f"{config_dir}/{minimize}", f"{model_out_dir}/{minimize}")
        shutil.copy(f"{config_dir}/{targets}", f"{model_out_dir}/{targets}")
        shutil.copy(f"{config_dir}/{phase}", f"{model_out_dir}/{phase}")
        shutil.copy(f"{config_dir}/{options}", f"{model_out_dir}/{options}")
        shutil.copy(f"{config_dir}/{draw}", f"{model_out_dir}/{draw}")
        shutil.copy(f"{config_dir}/{plot}", f"{model_out_dir}/perplex_plot_option.dat")

        # Modify the copied configuration files within the perplex directory
        self._replace_in_file(f"{model_out_dir}/{build}",
                              {"{SAMPLEID}": f"{model_prefix}",
                               "{OUTDIR}/": "",
                               "{TMIN}": str(T_min), "{TMAX}": str(T_max),
                               "{PMIN}": str(P_min), "{PMAX}": str(P_max),
                               "{SAMPLECOMP}": " ".join(map(str, norm_comp))})
        self._replace_in_file(f"{model_out_dir}/{minimize}",
                              {"{SAMPLEID}": f"{model_prefix}"})
        self._replace_in_file(f"{model_out_dir}/{targets}",
                              {"{SAMPLEID}": f"{model_prefix}"})
        self._replace_in_file(f"{model_out_dir}/{phase}",
                              {"{SAMPLEID}": f"{model_prefix}"})
        self._replace_in_file(f"{model_out_dir}/{options}",
                              {"{XNODES}": f"{int(res / 4)} {res + 1}",
                               "{YNODES}": f"{int(res / 4)} {res + 1}"})
        self._replace_in_file(f"{model_out_dir}/{draw}",
                              {"{SAMPLEID}": f"{model_prefix}"})

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # run perplex !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _run_perplex(self):
        """
        """
        # Get self attributes
        norm_sample_composition = self.norm_sample_composition
        oxides = self.oxides_system
        oxides_exclude = self.oxides_exclude
        subset_oxides = [oxide for oxide in oxides if oxide not in oxides_exclude]
        log_file = self.log_file
        model_out_dir = self.model_out_dir
        model_prefix = self.model_prefix
        timeout = self.timeout
        verbose = self.verbose

        # Check for input MAGEMin input files
        if not os.path.exists(f"{model_out_dir}/build-config"):
            raise Exception("No Perple_X input! Call _configure_perplex_model() first ...")

        if verbose >= 1:
            print(f"Building Perple_X model: {model_prefix} ...")
            max_oxide_width = max(len(oxide) for oxide in subset_oxides)
            max_comp_width = max(len(str(comp)) for comp in norm_sample_composition)
            max_width = max(max_oxide_width, max_comp_width)
            print(" ".join([f"  {oxide:<{max_width}}" for oxide in subset_oxides]))
            print(" ".join([f"  {comp:<{max_width}}" for comp in norm_sample_composition]))

        # Run programs with corresponding configuration files
        for program in ["build", "vertex", "werami", "pssect"]:
            # Get config files
            config_files = []

            if program == "build":
                config_files.append(f"{model_out_dir}/build-config")

            elif program == "vertex":
                config_files.append(f"{model_out_dir}/vertex-minimize")

            elif program == "werami":
                config_files.append(f"{model_out_dir}/werami-targets")
                config_files.append(f"{model_out_dir}/werami-phase")

                self._replace_in_file(f"{model_out_dir}/build-options",
                                      {"Anderson-Gruneisen     F":
                                       "Anderson-Gruneisen     T"})

            elif program == "pssect":
                config_files.append(f"{model_out_dir}/pssect-draw")

            # Get program path
            program_path = f"Perple_X/{program}"
            relative_program_path = f"../../{program_path}"

            for i, config in enumerate(config_files):
                try:
                    # Set permissions
                    os.chmod(program_path, 0o755)

                    # Open the subprocess and redirect input from the input file
                    with open(config, "rb") as input_stream:
                        process = subprocess.Popen([relative_program_path],
                                                   stdin=input_stream,
                                                   stdout=subprocess.PIPE,
                                                   stderr=subprocess.PIPE,
                                                   shell=True, cwd=model_out_dir)

                    # Wait for the process to complete and capture its output
                    stdout, stderr = process.communicate(timeout=timeout)

                    if verbose >= 2:
                        print(f"{stdout.decode()}")

                    # Write to logfile
                    with open(log_file, "a") as log:
                        log.write(stdout.decode())
                        log.write(stderr.decode())

                    if process.returncode != 0:
                        raise RuntimeError(f"Error executing perplex program '{program}'!")

                    elif verbose >= 2:
                        print(f"{program} output:")
                        print(f"{stdout.decode()}")

                    if program == "werami" and i == 0:
                        # Copy werami pseudosection output
                        shutil.copy(f"{model_out_dir}/{model_prefix}_1.tab",
                                    f"{model_out_dir}/target-array.tab")

                        # Remove old output
                        os.remove(f"{model_out_dir}/{model_prefix}_1.tab")

                    elif program == "werami" and i == 1:
                        # Copy werami mineral assemblage output
                        shutil.copy(f"{model_out_dir}/{model_prefix}_1.tab",
                                    f"{model_out_dir}/phases.tab")

                        # Remove old output
                        os.remove(f"{model_out_dir}/{model_prefix}_1.tab")

                    elif program == "pssect":
                        # Copy pssect assemblages output
                        shutil.copy(f"{model_out_dir}/"
                                    f"{model_prefix}_assemblages.txt",
                                    f"{model_out_dir}/assemblages.txt")

                        # Copy pssect auto refine output
                        shutil.copy(f"{model_out_dir}/"
                                    f"{model_prefix}_auto_refine.txt",
                                    f"{model_out_dir}/auto_refine.txt")

                        # Copy pssect seismic data output
                        shutil.copy(f"{model_out_dir}/"
                                    f"{model_prefix}_seismic_data.txt",
                                    f"{model_out_dir}/seismic_data.txt")

                        # Remove old output
                        os.remove(f"{model_out_dir}/{model_prefix}_assemblages.txt")
                        os.remove(f"{model_out_dir}/{model_prefix}_auto_refine.txt")
                        os.remove(f"{model_out_dir}/{model_prefix}_seismic_data.txt")

                except subprocess.CalledProcessError as e:
                    print(f"Error: {e}")

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # read perplex targets !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _read_perplex_targets(self):
        """
        """
        # Get self attributes
        perplex_db = self.perplex_db
        perplex_targets = self.perplex_targets

        # Initialize results
        if perplex_db == "stx21":
            results = {"T": [], "P": [], "rho": [], "Vp": [], "Vs": [], "entropy": [],
                       "assemblage_index": [], "assemblage": [], "variance": []}
        else:
            results = {"T": [], "P": [], "rho": [], "Vp": [], "Vs": [], "entropy": [],
                       "assemblage_index": [], "melt": [], "assemblage": [], "variance": []}

        # Open file
        with open(perplex_targets, "r") as file:
            # Skip lines until column headers are found
            for line in file:
                if line.strip().startswith("T(K)"):
                    break

            # Read the data
            for line in file:
                # Split line on whitespace
                values = line.split()

                # Read the table of P, T, rho etc.
                if len(values) == 7:
                    try:
                        for i in range(7):
                            # Make values floats or assign nan
                            value = (float(values[i])
                                     if not np.isnan(float(values[i]))
                                     else np.nan)

                            # Convert from bar to GPa
                            if i == 1: # P column
                                value /= 1e4

                            # Convert assemblage index to an integer
                            if i == 6: # assemblage index column
                                value = int(value) if not np.isnan(value) else np.nan

                            # Append results
                            results[list(results.keys())[i]].append(value)

                    except ValueError:
                        continue

                elif len(values) >= 8:
                    try:
                        for i in range(8):
                            # Make values floats or assign nan
                            value = (float(values[i])
                                     if not np.isnan(float(values[i]))
                                     else np.nan)

                            # Convert from bar to GPa
                            if i == 1: # P column
                                value /= 1e4

                            # Convert assemblage index to an integer
                            if i == 6: # assemblage index column
                                value = int(value) if not np.isnan(value) else np.nan

                            # Convert from % to fraction
                            if i == 7: # melt fraction column
                                value /= 100

                            # Append results
                            results[list(results.keys())[i]].append(value)

                    except ValueError:
                        continue

        return results

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # read perplex assemblages !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _read_perplex_assemblages(self):
        """
        """
        # Initialize dictionary to store assemblage info
        assemblage_dict = {}

        # Open assemblage file
        with open(self.perplex_assemblages, "r") as file:
            for i, line in enumerate(file, start=1):
                assemblages = line.split("-")[1].strip().split()

                # Make string formatting consistent
                cleaned_assemblages = [assemblage.split("(")[0].lower()
                                       for assemblage in assemblages]

                # Add assemblage to dict
                assemblage_dict[i] = cleaned_assemblages

        return assemblage_dict

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # process perplex results !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _process_perplex_results(self):
        """
        """
        # Get self attributes
        perplex_db = self.perplex_db
        perplex_targets = self.perplex_targets
        perplex_assemblages = self.perplex_assemblages
        model_out_dir = self.model_out_dir
        model_prefix = self.model_prefix
        verbose = self.verbose
        debug = self.debug

        # Check for targets
        if not os.path.exists(perplex_targets):
            raise Exception("No Perple_X files to process! Call _run_perplex() first ...")

        # Check for assemblages
        if not os.path.exists(perplex_assemblages):
            raise Exception("No Perple_X files to process! Call _run_perplex() first ...")

        if verbose >= 2:
            print(f"Reading Perple_X output: {model_out_dir} ...")

        # Read results
        results = self._read_perplex_targets()

        # Remove entropy
        results.pop("entropy")

        # Get assemblages from file
        assemblages = self._read_perplex_assemblages()

        # Parse assemblages by index
        for index in results.get("assemblage_index"):
            if np.isnan(index):
                results["assemblage"].append("")

            else:
                phases = assemblages[index]
                results["assemblage"].append(phases)

        # Count unique phases (assemblage variance)
        for assemblage in results.get("assemblage"):
            if assemblage is None:
                results["variance"].append(np.nan)

            else:
                unique_phases = set(assemblage)
                count = len(unique_phases)

                results["variance"].append(count)

        # Remove assemblage index
        results.pop("assemblage_index")

        # Encode assemblage
        encoded_assemblages = self._encode_assemblages(results["assemblage"])

        # Replace assemblage with encoded assemblages
        results["assemblage"] = encoded_assemblages

        # Point results that can be converted to numpy arrays
        if perplex_db == "stx21":
            point_params = ["T", "P", "rho", "Vp", "Vs", "assemblage", "variance"]
        else:
            point_params = ["T", "P", "rho", "Vp", "Vs", "melt", "assemblage", "variance"]

        # Convert numeric point results into numpy arrays
        for key, value in results.items():
            if key in point_params:
                if key == "rho":
                    # Convert from kg/m3 to g/cm3
                    results[key] = np.round(np.array(value) / 1e3, 3)

                elif key == "melt":
                    # Convert from kg/m3 to g/cm3
                    results[key] = np.round(np.array(value) * 100, 3)

                else:
                    results[key] = np.array(value)

        # Save as pandas df
        df = pd.DataFrame.from_dict(results)

        if verbose >= 2:
            print(f"Writing Perple_X results: {model_prefix} ...")

        # Write to csv file
        df.to_csv(f"{model_out_dir}/results.csv", index=False)

        if debug:
            # Clean up output directory
            files_to_keep = ["assemblages.csv", "results.csv"]

            # Create dir to store model files
            model_out_files_dir = f"{model_out_dir}/model"
            os.makedirs(model_out_files_dir, exist_ok=True)

            try:
                # List all files in the directory
                all_files = os.listdir(model_out_dir)

                # Iterate through the files and delete those not in the exclusion list
                for filename in all_files:
                    file_path = os.path.join(model_out_dir, filename)
                    destination_path = os.path.join(model_out_files_dir, filename)

                    if os.path.isfile(file_path) and filename not in files_to_keep:
                        shutil.move(file_path, destination_path)

            except Exception as e:
                print(f"Error: {e}")

        else:
            # Clean up output directory
            files_to_keep = ["assemblages.csv", "results.csv"]

            try:
                # List all files in the directory
                all_files = os.listdir(model_out_dir)

                # Iterate through the files and delete those not in the exclusion list
                for filename in all_files:
                    file_path = os.path.join(model_out_dir, filename)

                    if os.path.isfile(file_path) and filename not in files_to_keep:
                        os.remove(file_path)

            except Exception as e:
                print(f"Error: {e}")

        return None

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .0.4.        Post Process GFEM Models        !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get results !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_results(self):
        """
        """
        # Get self attributes
        program = self.program
        model_prefix = self.model_prefix
        model_built = self.model_built
        model_out_dir = self.model_out_dir
        verbose = self.verbose

        # Check for model
        if not model_built:
            raise Exception("No GFEM model! Call build_model() first ...")

        # Get filepaths for magemin output
        filepath = f"{model_out_dir}/results.csv"

        if not os.path.exists(filepath):
            raise Exception("No results to read!")

        if verbose >= 2:
            print(f"Reading results: {filepath} ...")

        # Read results
        df = pd.read_csv(filepath)

        # Convert to dict of np arrays
        for column in df.columns:
            self.results[column] = df[column].values

        # Check for all nans
        any_array_all_nans = False

        for key, array in self.results.items():
            if key not in ["melt"]:
                if np.all(np.isnan(array)):
                    any_array_all_nans = True

        if any_array_all_nans:
            self.results = {}
            self.model_build_error = True
            self.model_error = f"Model {model_prefix} [{program}] produced all nans!"

            raise Exception(self.model_error)

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # create geotherm mask !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _create_geotherm_mask(self, T_mantle1=673, T_mantle2=1773, grad_mantle1=0.5,
                              grad_mantle2=0.5):
        """
        """
        # Get self attributes
        results = self.results

        # Check for results
        if not results:
            raise Exception("No GFEM model results! Call get_results() first ...")

        # Get PT values
        P, T = results["P"].copy(), results["T"].copy()

        # Get max PT
        P_min, P_max, T_min, T_max = np.min(P), np.max(P), np.min(T), np.max(T)

        # Find geotherm boundaries
        T1_Pmax = (P_max * grad_mantle1 * 35) + T_mantle1
        P1_Tmin = (T_min - T_mantle1) / (grad_mantle1 * 35)
        T2_Pmin = (P_min * grad_mantle2 * 35) + T_mantle2
        T2_Pmax = (P_max * grad_mantle2 * 35) + T_mantle2

        # Iterate through PT array and set nan where PT is out of geotherm bounds
        PT_array = np.stack((P, T), axis=-1)

        for i in range(PT_array.shape[0]):
            p = PT_array[i, 0]
            t = PT_array[i, 1]

            # Calculate mantle geotherms
            geotherm1 = (t - T_mantle1) / (grad_mantle1 * 35)
            geotherm2 = (t - T_mantle2) / (grad_mantle2 * 35)

            # Set PT array to nan if outside of geotherm bounds
            if (
                   ((t <= T1_Pmax) and (p >= geotherm1)) or
                   ((t >= T2_Pmin) and (p <= geotherm2))
            ):
                PT_array[i] = [np.nan, np.nan]

        # Create nan mask
        mask = np.isnan(PT_array[:,0])

        return mask

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get feature array !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_feature_array(self):
        """
        """
        # Get self attributes
        results = self.results
        targets = self.targets
        mask_geotherm = self.mask_geotherm
        model_built = self.model_built
        verbose = self.verbose

        # Check for model
        if not model_built:
            raise Exception("No GFEM model! Call build_model() first ...")

        # Check for results
        if not results:
            raise Exception("No GFEM model results! Call get_results() first ...")

        # Get P T arrays
        P, T = results["P"].copy(), results["T"].copy()

        # Stack PT arrays
        self.feature_array_unmasked = np.stack((P, T), axis=-1)

        # Mask geotherm
        if mask_geotherm:
            if verbose >= 2:
                print("Masking geotherm!")

            # Get geotherm mask
            mask = self._create_geotherm_mask()

            self.feature_array = self.feature_array_unmasked.copy()

            # Apply mask to all target arrays
            for j in range(self.feature_array.shape[1]):
                self.feature_array[:, j][mask] = np.nan

        else:
            self.feature_array = self.feature_array_unmasked.copy()

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # process array !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _process_array(self, array, n_neighbors=1, threshold=5):
        """
        """
        # Create a copy of the input array to avoid modifying the original array
        result_array = array.copy()

        # Iterate through each element of the array
        for i in range(len(result_array)):
            for j in range(len(result_array[i])):
                # Define the neighborhood indices
                neighbors = []

                for x in range(i - n_neighbors, i + n_neighbors + 1):
                    for y in range(j - n_neighbors, j + n_neighbors + 1):
                        if (0 <= x < len(result_array) and
                            0 <= y < len(result_array[i]) and
                            (x != i or y != j)):
                            neighbors.append((x, y))

                # Get neighborhood values
                surrounding_values = [result_array[x, y] for x, y in neighbors if not
                                      np.isnan(result_array[x, y])]

                # Define anomalies
                if surrounding_values:
                    mean_neighbors = np.mean(surrounding_values)
                    std_neighbors = np.std(surrounding_values)
                    anom_threshold = threshold * std_neighbors

                    # Impute anomalies
                    if abs(result_array[i, j] - mean_neighbors) > anom_threshold:
                        if surrounding_values:
                            result_array[i, j] = np.mean(surrounding_values)
                        else:
                            result_array[i, j] = 0

                    # Impute nans
                    elif np.isnan(result_array[i, j]):
                        nan_count = sum(1 for x, y in neighbors if
                                        0 <= x < len(result_array) and
                                        0 <= y < len(result_array[i]) and
                                        np.isnan(result_array[x, y]))

                        if nan_count >= int((((((n_neighbors * 2) + 1)**2) - 1) / 3)):
                            result_array[i, j] = 0

                        else:
                            result_array[i, j] = np.mean(surrounding_values)

        return result_array.flatten()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get target array !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_target_array(self):
        """
        """
        # Get self attributes
        perplex_db = self.perplex_db
        results = self.results
        targets = self.targets
        res = self.res
        mask_geotherm = self.mask_geotherm
        model_built = self.model_built
        verbose = self.verbose

        # Check for model
        if not model_built:
            raise Exception("No GFEM model! Call build_model() first ...")

        # Check for results
        if not results:
            raise Exception("No GFEM model results! Call get_results() first ...")

        # Set n_neighbors for KNN imputer
        if res <= 8:
            n_neighbors = 1
        elif res <= 16:
            n_neighbors = 2
        elif res <= 32:
            n_neighbors = 3
        elif res <= 64:
            n_neighbors = 4
        elif res <= 128:
            n_neighbors = 5

        # Remove melt if perplex uses stx21 dataset
        if "melt" in targets:
            if perplex_db == "stx21":
                targets.remove("melt")

        # Initialize empty list for target arrays
        target_array_list = []

        # Rearrange results to match targets
        results_rearranged = {key: results[key] for key in targets}

        # Impute missing values
        for key, value in results_rearranged.items():
            if key in targets:
                if key == "melt":
                    # Process array
                    target_array_list.append(
                        self._process_array(value.reshape(res + 1, res + 1)).flatten()
                    )

                else:
                    # Impute target array with KNN
                    imputer = KNNImputer(n_neighbors = n_neighbors, weights="distance")
                    imputed_array = imputer.fit_transform(value.reshape(res + 1, res + 1))

                    # Process imputed array
                    target_array_list.append(self._process_array(imputed_array).flatten())

        # Stack target arrays
        self.target_array_unmasked = np.stack(target_array_list, axis=-1)

        if mask_geotherm:
            if verbose >= 2:
                print("Masking geotherm!")

            # Get geotherm mask
            mask = self._create_geotherm_mask()

            self.target_array = self.target_array_unmasked.copy()

            # Apply mask to all target arrays
            for j in range(self.target_array.shape[1]):
                self.target_array[:, j][mask] = np.nan

        else:
            self.target_array = self.target_array_unmasked.copy()

        return None

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .0.5.           Build GFEM Models            !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # build model !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def build_model(self):
        """
        """
        # Set retries
        max_retries = 3

        for retry in range(max_retries):
            try:
                if self.program == "magemin":
                    # Build model and write results to csv
                    self._configure_magemin_model()
                    self._run_magemin()
                    self._get_comp_time()
                    self._process_magemin_results()

                elif self.program == "perplex":
                    # Build model and write results to csv
                    self._configure_perplex_model()
                    self._run_perplex()
                    self._get_comp_time()
                    self._process_perplex_results()

                self.model_built = True

                return None

            except Exception as e:
                print(f"!!! {e} !!!")

                if retry < max_retries - 1:
                    print(f"Retrying in 5 seconds ...")
                    time.sleep(5)

                else:
                    self.model_build_error = True
                    self.model_error = e

                    return None

        return None
