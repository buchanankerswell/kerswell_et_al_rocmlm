#######################################################
## .0.              Load Libraries               !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# utilities !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import re
import time
import shutil
import itertools
import subprocess

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

#######################################################
## .1.             Helper Functions              !!! ##
#######################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# get sampleids !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_sampleids(filepath, n, seed=42):
    """
    """
    # Check for file
    if not os.path.exists(filepath):
        raise Exception("Sample data source does not exist!")

    # Read data
    df = pd.read_csv(filepath)

    # Sample data
    if n == len(df):
        sampleids = df["NAME"].values
    else:
        sampleids = df.sample(n, random_state=seed)["NAME"].values

    return sampleids

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# gfem_iteration !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def gfem_iteration(args):
    """
    """
    # Unpack arguments
    program, dataset, sampleid, source, res, Pmin, Pmax, Tmin, Tmax, normox, targets, \
        maskgeotherm, verbose, debug = args

    # Initiate GFEM model
    iteration = GFEMModel(program, dataset, sampleid, source, res, Pmin, Pmax, Tmin, Tmax,
                          normox, targets, maskgeotherm, verbose, debug)

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# build gfem models !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def build_gfem_models(programs=["perplex", "magemin"], datasets=["train", "valid"],
                      source="assets/data/benchmark-samples", nsamples=3, res=128, Pmin=1,
                      Pmax=28, Tmin=773, Tmax=2273, normox="all",
                      targets=["rho", "Vp", "Vs", "melt_fraction"], maskgeotherm=False,
                      parallel=True, nprocs=os.cpu_count()-2, verbose=1, debug=False):
    """
    """
    # Get samples
    sampleids = get_sampleids(source, nsamples)

    # Create combinations of samples and datasets
    combinations = list(itertools.product(programs, datasets, sampleids))

    # Define number of processors
    if not parallel:
        nprocs = 1

    elif parallel:
        if nprocs is None or nprocs > os.cpu_count():
            nprocs = os.cpu_count() - 2

        else:
            nprocs = nprocs

    # Make sure nprocs is not greater than sample combinations
    if nprocs > len(combinations):
        nprocs = len(combinations)

    # Create list of args for mp pooling
    run_args = [(program, dataset, sampleid, source, res, Pmin, Pmax, Tmin, Tmax, normox,
                 targets, maskgeotherm, verbose, debug)
                for program, dataset, sampleid in combinations]

    # Create a multiprocessing pool
    with mp.Pool(processes=nprocs) as pool:
        models = pool.map(gfem_iteration, run_args)

        # Wait for all processes
        pool.close()
        pool.join()

    # Check for errors in the models
    error_count = 0

    for model in models:
        if model.model_build_error:
            error_count += 1

    if error_count > 0:
        print(f"Total {program} models with errors: {error_count}")

    else:
        print("All GFEM models built successfully!")

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    successful_models = [model for model in models if not model.model_build_error]

    return successful_models

#######################################################
## .2.              GFEMModel class              !!! ##
#######################################################
class GFEMModel:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # init !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, program, dataset, sample_id, source, res,
                 P_min=1, P_max=28, T_min=773, T_max=2273, normox="all",
                 targets=["rho", "Vp", "Vs", "melt_fraction"], maskgeotherm=False,
                 verbose=1, debug=False):
        # Input
        self.program = program
        self.P_min = P_min
        self.P_max = P_max
        self.T_min = T_min
        self.T_max = T_max
        self.res = res
        self.source = source
        self.sample_id = sample_id
        self.normox = normox
        self.dataset = dataset
        self.targets = targets
        self.mask_geotherm = maskgeotherm
        self.verbose = verbose
        self.debug = debug

        if self.program == "magemin":
            # Magemin dirs and filepaths
            self.model_out_dir = (f"runs/{self.program[:4]}_{self.sample_id}_"
                                  f"{self.dataset[0]}{self.res}")
            self.magemin_in_path = f"{self.model_out_dir}/in.dat"
            self.magemin_out_path = f"{self.model_out_dir}/output/_pseudosection_output.txt"

        elif self.program == "perplex":
            # Perplex dirs and filepaths
            cwd = os.getcwd()
            self.perplex_dir = f"{cwd}/assets/perplex"
            self.model_out_dir = (f"{cwd}/runs/{self.program[:4]}_{self.sample_id}_"
                                  f"{self.dataset[0]}{self.res}")
            self.perplex_targets = f"{self.model_out_dir}/target-array.tab"
            self.perplex_assemblages = f"{self.model_out_dir}/assemblages.txt"

        else:
            raise ValueError("Unrecognized GFEM program! Use 'magemin' or 'perplex' ...")

        self.model_prefix = f"{self.sample_id}-{self.dataset}-{self.res}"
        self.fig_dir = f"figs/{self.sample_id}_{self.res}"
        self.log_file = f"log/log-{self.program}-{self.model_prefix}"

        # Results
        self.sample_composition = []
        self.norm_sample_composition = []
        self.comp_time = None
        self.model_built = False
        self.results = {}
        self.feature_array = np.array([])
        self.feature_array_unmasked = np.array([])
        self.target_array = np.array([])
        self.target_array_unmasked = np.array([])

        # Errors
        self.model_build_error = False
        self.model_error = None

        # Check for existing model build
        if os.path.exists(self.model_out_dir):
            if (os.path.exists(f"{self.model_out_dir}/results.csv") and
                os.path.exists(f"{self.model_out_dir}/assemblages.csv")):
                try:
                    self.model_built = True
                    self.get_results()

                    if self.targets:
                        self.get_feature_array()
                        self.get_target_array()

                except Exception as e:
                    print(f"!!! {e} !!!")

                    return None

                print(f"Found results for model {self.model_prefix} [{self.program}]!")

            else:
                # Make new model if results not found
                shutil.rmtree(self.model_out_dir)
                os.makedirs(self.model_out_dir, exist_ok=True)
        else:
            os.makedirs(self.model_out_dir, exist_ok=True)

        # Set np array printing option
        np.set_printoptions(precision=3, suppress=True)

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
        source = self.source
        sample_id = self.sample_id

        # All oxides needed for MAGEMin
        oxides = ["SIO2", "AL2O3", "CAO", "MGO", "FEO", "K2O", "NA2O", "TIO2", "FE2O3",
                  "CR2O3", "H2O"]

        # Read the data file
        df = pd.read_csv(source)

        # Subset the DataFrame based on the sample name
        subset_df = df[df["NAME"] == sample_id]

        if subset_df.empty:
            raise ValueError("Sample name not found in the dataset!")

        # Get the oxide compositions for the selected sample
        composition = []

        for oxide in oxides:
            if oxide in subset_df.columns and pd.notnull(subset_df[oxide].iloc[0]):
                composition.append(float(subset_df[oxide].iloc[0]))

            else:
                if oxide != "H2O":
                    composition.append(0.01)

                else:
                    composition.append(0.00)

        self.sample_composition = composition

        return composition

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # normalize sample composition !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _normalize_sample_composition(self):
        """
        """
        # Get self attributes
        sample_composition = self.sample_composition
        normox = self.normox

        # Check for sample composition
        if not sample_composition:
            raise Exception("No sample composition! Call _get_sample_composition() first ...")

        # No normalizing for all components
        if normox == "all":
            return sample_composition

        # MAGEMin req components
        oxides = ["SIO2", "AL2O3", "CAO", "MGO", "FEO", "K2O", "NA2O", "TIO2", "FE2O3",
                  "CR2O3", "H2O"]

        # Check input
        if len(sample_composition) != 11:
            error_message = ("The input sample list must have exactly 11 components!\n"
                             f"{oxides}")

            raise ValueError(error_message)

        # Filter components
        subset_sample = [comp if component in components else 0.01
                         for comp, component in zip(sample, oxides)]

        # Normalize
        total_subset_concentration = sum([comp for comp in subset_sample if comp != 0.01])

        normalized_concentrations = []

        for comp, component in zip(sample, oxides):
            if component in components:
                normalized_concentration = (
                    (comp / total_subset_concentration) * 100 if comp != 0.01 else 0.01
                )

            else:
                normalized_concentration = 0.01

            normalized_concentrations.append(normalized_concentration)

        self.norm_sample_composition = normalized_concentrations

        return normalized_concentrations

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
                        time_s = time_ms / 1000

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

            csv_filepath = "assets/data/gfem-efficiency.csv"

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

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # run magemin !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _run_magemin(self):
        """
        """
        # Get self attributes
        magemin_in_path = self.magemin_in_path
        model_out_dir = self.model_out_dir
        model_prefix = self.model_prefix
        log_file = self.log_file
        verbose = self.verbose

        # Check for input MAGEMin input files
        if not os.path.exists(magemin_in_path):
            raise Exception("No MAGEMin input! Call _configure_magemin_model() first ...")

        print(f"Building MAGEMin model: {model_prefix}...")

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
            stdout, stderr = process.communicate()

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

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # read magemin output !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _read_magemin_output(self):
        """
        """
        # Open file
        with open(self.magemin_out_path, "r") as file:
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
                            "melt_fraction": liq, # melt fraction
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

        assemblage_variance = []

        for assemblage in assemblages:
            unique_phases = set(assemblage)
            count = len(unique_phases)

            assemblage_variance.append(count)

        # Add assemblage variance to merged results
        results["assemblage_variance"] = assemblage_variance

        # Encode assemblage
        encoded_assemblages = self._encode_assemblages(assemblages)

        # Replace assemblage with encoded assemblages
        results["assemblage"] = encoded_assemblages

        # Point results that can be converted to numpy arrays
        point_params = ["T", "P", "rho", "Vp", "Vs", "melt_fraction", "assemblage",
                        "assemblage_variance"]

        # Convert numeric point results into numpy arrays
        for key, value in results.items():
            if key in point_params:
                if key == "P":
                    # Convert from kbar to GPa
                    results[key] = np.array(value) / 10

                elif key == "T":
                    # Convert from C to K
                    results[key] = np.array(value) + 273

                elif key == "rho":
                    # Convert from kg/m3 to g/cm3
                    results[key] = np.array(value) / 1000

                elif key == "melt_fraction":
                    # Convert from kg/m3 to g/cm3
                    results[key] = np.array(value) * 100

                else:
                    results[key] = np.array(value)

        # Print results
        if verbose >= 2:
            units = {"T": "K", "P": "GPa", "rho": "g/cm3", "Vp": "km/s", "Vs": "km/s",
                     "melt_fraction": "%", "assemblage": "", "assemblage_variance": ""}

            print("+++++++++++++++++++++++++++++++++++++++++++++")
            for key, value in results.items():
                if isinstance(value, list):
                    print(f"    ({len(value)},) list:      : {key}")

                elif isinstance(value, np.ndarray):
                    min, max = np.nanmin(value), np.nanmax(value)

                    print(f"    {value.shape} np array: {key} "
                          f"({min:.1f}, {max:.1f}) {units[key]}")

            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # Save as pandas df
        df = pd.DataFrame.from_dict(results)

        if verbose >= 2:
            print(f"Writing MAGEMin results: {model_out_dir} ...")

        # Write to csv file
        df.to_csv(f"{model_out_dir}/results.csv", index=False)

        if not debug:
            # Clean up output directory
            os.remove(magemin_in_path)
            shutil.rmtree(f"{model_out_dir}/output")

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
        dataset = self.dataset
        model_out_dir = self.model_out_dir
        model_prefix = self.model_prefix
        perplex_dir = self.perplex_dir
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

            P_min += P_step
            P_max -= P_step
            T_min += T_step
            T_max -= T_step

        # Configuration files
        build = "perplex-build-config"
        minimize = "perplex-vertex-minimize"
        targets = "perplex-werami-targets"
        phase = "perplex-werami-phase"
        options = "perplex-build-options"
        draw = "perplex-pssect-draw"
        plot = "perplex-plot-options"

        # Copy original configuration files to the perplex directory
        shutil.copy(f"assets/config/{build}", f"{model_out_dir}/{build}")
        shutil.copy(f"assets/config/{minimize}", f"{model_out_dir}/{minimize}")
        shutil.copy(f"assets/config/{targets}", f"{model_out_dir}/{targets}")
        shutil.copy(f"assets/config/{phase}", f"{model_out_dir}/{phase}")
        shutil.copy(f"assets/config/{options}", f"{model_out_dir}/{options}")
        shutil.copy(f"assets/config/{draw}", f"{model_out_dir}/{draw}")
        shutil.copy(f"assets/config/{plot}", f"{model_out_dir}/perplex_plot_option.dat")

        # Modify the copied configuration files within the perplex directory
        self._replace_in_file(f"{model_out_dir}/{build}",
                              {"{SAMPLEID}": f"{model_prefix}",
                               "{PERPLEX}": f"{perplex_dir}",
                               "{OUTDIR}": f"{model_out_dir}",
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

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # run perplex !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _run_perplex(self):
        """
        """
        # Get self attributes
        log_file = self.log_file
        model_out_dir = self.model_out_dir
        model_prefix = self.model_prefix
        perplex_dir = self.perplex_dir
        verbose = self.verbose

        # Check for input MAGEMin input files
        if not os.path.exists(f"{model_out_dir}/perplex-build-config"):
            raise Exception("No Perple_X input! Call _configure_perplex_model() first ...")

        print(f"Building Perple_X model: {model_prefix} ...")

        # Run programs with corresponding configuration files
        for program in ["build", "vertex", "werami", "pssect"]:
            # Get config files
            config_files = []

            if program == "build":
                config_files.append(f"{model_out_dir}/perplex-build-config")

            elif program == "vertex":
                config_files.append(f"{model_out_dir}/perplex-vertex-minimize")

            elif program == "werami":
                config_files.append(f"{model_out_dir}/perplex-werami-targets")
                config_files.append(f"{model_out_dir}/perplex-werami-phase")

#                self._replace_in_file(f"{model_out_dir}/perplex-build-options",
#                                    {"Anderson-Gruneisen  F": "Anderson-Gruneisen  T"})

            elif program == "pssect":
                config_files.append(f"{model_out_dir}/perplex-pssect-draw")

            # Get program path
            program_path = f"{perplex_dir}/{program}"

            for i, config in enumerate(config_files):
                try:
                    # Set permissions
                    os.chmod(program_path, 0o755)

                    # Open the subprocess and redirect input from the input file
                    with open(config, "rb") as input_stream:
                        process = subprocess.Popen([program_path], stdin=input_stream,
                                                   stdout=subprocess.PIPE,
                                                   stderr=subprocess.PIPE,
                                                   shell=True, cwd=model_out_dir)

                    # Wait for the process to complete and capture its output
                    stdout, stderr = process.communicate()

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

                        # Convert postscript file to pdf
                        ps = f"{model_out_dir}/{model_prefix}.ps"
                        pdf = f"{model_out_dir}/{model_prefix}.pdf"

                        subprocess.run(f"ps2pdf {ps} {pdf}", shell=True)

                except subprocess.CalledProcessError as e:
                    print(f"Error: {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # read perplex targets !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _read_perplex_targets(self):
        """
        """
        # Initialize results
        results = {"T": [], "P": [], "rho": [], "Vp": [], "Vs": [], "entropy": [],
                   "assemblage_index": [], "melt_fraction": [], "assemblage": [],
                   "assemblage_variance": []}

        # Open file
        with open(self.perplex_targets, "r") as file:
            # Skip lines until column headers are found
            for line in file:
                if line.strip().startswith("T(K)"):
                    break

            # Read the data
            for line in file:
                # Split line on whitespace
                values = line.split()

                # Read the table of P, T, rho etc.
                if len(values) >= 8:
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
                results["assemblage_variance"].append(np.nan)

            else:
                unique_phases = set(assemblage)
                count = len(unique_phases)

                results["assemblage_variance"].append(count)

        # Remove assemblage index
        results.pop("assemblage_index")

        # Encode assemblage
        encoded_assemblages = self._encode_assemblages(results["assemblage"])

        # Replace assemblage with encoded assemblages
        results["assemblage"] = encoded_assemblages

        # Point results that can be converted to numpy arrays
        point_params = ["T", "P", "rho", "Vp", "Vs", "melt_fraction", "assemblage",
                        "assemblage_variance"]

        # Convert numeric point results into numpy arrays
        for key, value in results.items():
            if key in point_params:
                if key == "rho":
                    # Convert from kg/m3 to g/cm3
                    results[key] = np.array(value) / 1000

                elif key == "melt_fraction":
                    # Convert from kg/m3 to g/cm3
                    results[key] = np.array(value) * 100

                else:
                    results[key] = np.array(value)

        # Print results
        if verbose >= 2:
            units = {"T": "K", "P": "GPa", "rho": "g/cm3", "Vp": "km/s", "Vs": "km/s",
                     "melt_fraction": "%", "assemblage": "", "assemblage_variance": ""}

            print("+++++++++++++++++++++++++++++++++++++++++++++")
            for key, value in results.items():
                if isinstance(value, list):
                    print(f"    ({len(value)},) list       : {key}")

                elif isinstance(value, np.ndarray):
                    min, max = np.nanmin(value), np.nanmax(value)

                    print(f"    {value.shape} np array: {key} "
                          f"({min:.1f}, {max:.1f}) {units[key]}")

            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # Save as pandas df
        df = pd.DataFrame.from_dict(results)

        if verbose >= 2:
            print(f"Writing Perple_X results: {model_prefix} ...")

        # Write to csv file
        df.to_csv(f"{model_out_dir}/results.csv", index=False)

        if not debug:
            # Clean up output directory
            files_to_keep = ["assemblages.csv", "results.csv", f"{model_prefix}.pdf"]

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
            if key not in ["melt_fraction"]:
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
    def _create_geotherm_mask(self, T_mantle1=273, T_mantle2=1773, grad_mantle1=1,
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
    # impute array with nans !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _impute_array_with_nans(self, array, n_neighbors=1):
        """
        """
        # Create a copy of the input array to avoid modifying the original array
        result_array = array.copy()

        # Reshape into square
        result_array = result_array.reshape(self.res + 1, self.res + 1)

        # Iterate through each element of the array
        for i in range(len(result_array)):
            for j in range(len(result_array[i])):
                if np.isnan(result_array[i, j]):
                    # Define the neighborhood indices
                    neighbors = []

                    for x in range(i - n_neighbors, i + n_neighbors + 1):
                        for y in range(j - n_neighbors, j + n_neighbors + 1):
                            if (0 <= x < len(result_array) and 0 <= y < len(result_array[i])
                                    and (x != i or y != j)):
                                neighbors.append((x, y))

                    # Check if all of the surrounding values are also nan
                    nan_count = sum(1 for x, y in neighbors if 0 <= x < len(result_array) and
                                    0 <= y < len(result_array[i]) and
                                    np.isnan(result_array[x, y]))

                    if nan_count >= int((((((n_neighbors * 2) + 1)**2) - 1) / 3)):
                        # If one-third of the square neighborhood is nan
                        result_array[i, j] = 0

                    else:
                        # Impute the mean value of neighbors
                        surrounding_values = [result_array[x, y] for x, y in neighbors
                                              if not np.isnan(result_array[x, y])]

                        if surrounding_values:
                            result_array[i, j] = np.mean(surrounding_values)

                        else:
                            # If there are no surrounding numerical values, set to 0
                            result_array[i, j] = 0

        return result_array.flatten()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get target array !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_target_array(self):
        """
        """
        # Get self attributes
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

        # Initialize empty list for target arrays
        target_array_list = []

        # Rearrange results to match targets
        results_rearranged = {key: results[key] for key in targets}

        # Impute missing values
        for key, value in results_rearranged.items():
            if key in targets:
                if key == "melt_fraction":
                    target_array_list.append(self._impute_array_with_nans(value))
                else:
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

                    else:
                        geotherm_threshold = 0.125

                    # Initialize KNN imputer
                    imputer = KNNImputer(n_neighbors = n_neighbors, weights="distance")

                    # Impute target array
                    target_array_list.append(imputer.fit_transform(
                        value.reshape(res + 1, res + 1)).flatten())

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

