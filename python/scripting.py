#######################################################
## .0.              Load Libraries               !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# utilities !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import sys
import ast
import yaml
import shutil
import zipfile
import argparse
import platform
import subprocess
import pkg_resources
from git import Repo
import urllib.request

#######################################################
## .1.  General Helper Functions for Scripting   !!! ##
#######################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# print session info !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def print_session_info(condafile=None, makefile=None):
    """
    """
    # Print session info
    print("Session info:")

    # Print Python version
    version_string = ".".join(map(str, sys.version_info))

    print(f"    Python Version: {version_string}")

    # Print package versions
    print("    Loaded packages:")

    if condafile:
        conda_packages = get_conda_packages(condafile)

        for package in conda_packages:
            if isinstance(package, str) and package != "python":
                package_name = package.split("=")[0]

                try:
                    version = pkg_resources.get_distribution(package_name).version

                    print(f"        {package_name} Version: {version}")

                except pkg_resources.DistributionNotFound:
                    print(f"        {package_name} not found ...")
    else:
        print("    No Conda file provided ...")

    # Print operating system information
    os_info = platform.platform()

    print(f"    Operating System: {os_info}")

    if makefile:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Makefile variables:")

        # Makefile assets
        makefile_vars_assets = ["DATADIR", "CONFIGDIR", "PERPLEXDIR"]

        # Get Makefile variables
        makefile_dict = {}

        for variable in makefile_vars_assets:
            makefile_dict[f"{variable}"] = read_makefile_variable(makefile, variable)

        # Print variables
        print("    Assets:")
        for key, value in makefile_dict.items():
            print(f"        {key}: {value}")

        # Makefile dataset build options
        makefile_vars_dataset_build_options = [
            "SAMPLEID", "PMIN", "PMAX", "TMIN", "TMAX", "RES", "EMSONLY", "DATASET", "NORMOX",
            "SEED", "PARALLEL", "NPROCS", "KFOLDS", "VERBOSE"
        ]

        # Get Makefile variables
        makefile_dict = {}

        for variable in makefile_vars_dataset_build_options:
            makefile_dict[f"{variable}"] = read_makefile_variable(makefile, variable)

        # Print variables
        print("    Dataset build options:")
        for key, value in makefile_dict.items():
            print(f"        {key}: {value}")

        # Makefile rocml options
        makefile_vars_rocml_options = ["TARGETS", "MLMODS", "MLTUNE"]

        # Get Makefile variables
        makefile_dict = {}

        for variable in makefile_vars_rocml_options:
            makefile_dict[f"{variable}"] = read_makefile_variable(makefile, variable)

        # Print variables
        print("    RocML options:")
        for key, value in makefile_dict.items():
            print(f"        {key}: {value}")

        # Makefile pca sampling options
        makefile_vars_pca_options = ["NPCA", "KCLUSTER"]

        # Get Makefile variables
        makefile_dict = {}

        for variable in makefile_vars_pca_options:
            makefile_dict[f"{variable}"] = read_makefile_variable(makefile, variable)

        # Print variables
        print("    PCA options:")
        for key, value in makefile_dict.items():
            print(f"        {key}: {value}")

        # Makefile visualization options
        makefile_vars_visualization_options = ["FIGDIR", "VISTARGETS", "COLORMAP"]

        # Get Makefile variables
        makefile_dict = {}

        for variable in makefile_vars_visualization_options:
            makefile_dict[f"{variable}"] = read_makefile_variable(makefile, variable)

        # Print variables
        print("    Visualization options:")
        for key, value in makefile_dict.items():
            print(f"        {key}: {value}")

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    else:
        print("No Makefile provided.")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# read makefile variable !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_makefile_variable(makefile, variable):
    """
    """
    try:
        with open(makefile, "r") as file:
            lines = file.readlines()

            for line in lines:
                if line.strip().startswith(variable):
                    return line.split("=")[1].strip()

    except IOError as e:
        print(f"Error reading Makefile: {e}")

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# read conda packages !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_conda_packages(condafile):
    """
    """
    try:
        with open(condafile, "r") as file:
            conda_data = yaml.safe_load(file)

        return conda_data.get("dependencies", [])

    except (IOError, yaml.YAMLError) as e:
        print(f"Error reading Conda file: {e}")

        return []

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# download and unzip !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def download_and_unzip(url, destination):
    """
    """
    try:
        # Download the file
        urllib.request.urlretrieve(url, "assets.zip")

        # Extract the contents of the zip file
        with zipfile.ZipFile("assets.zip", "r") as zip_ref:
            zip_ref.extractall(destination)

        # Remove the zip file
        os.remove("assets.zip")

    except urllib.error.URLError as e:
        raise Exception(f"Unable to download from {url}!")

    except zipfile.BadZipFile as e:
        raise Exception(f"The downloaded file is not a valid zip file!")

    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}!")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# download github submodule !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def download_github_submodule(repository_url, submodule_dir, commit_hash):
    """
    """
    # Check if submodule directory already exists and delete it
    if os.path.exists(submodule_dir):
        shutil.rmtree(submodule_dir)

    # Clone submodule and recurse its contents
    try:
        repo = Repo.clone_from(repository_url, submodule_dir, recursive=True)
        repo.git.checkout(commit_hash)

    except Exception as e:
        print(f"An error occurred while cloning the GitHub repository: {e} ...")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# compile magemin !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compile_magemin(emsonly, verbose):
    """
    """
    # Config dir
    config_dir = "assets/config"

    # Check for MAGEMin repo
    if os.path.exists("MAGEMin"):
        if emsonly:
            print("Compiling MAGEMin with HP endmembers ...")

            # Move modified MAGEMin config file with HP mantle endmembers
            config = f"{config_dir}/magemin-init-hp-endmembers"
            old_config = "MAGEMIN/src/initialize.h"

            if os.path.exists(config):
                # Replace MAGEMin config file with modified (endmembers only) file
                subprocess.run(f"cp {config} {old_config}", shell=True)
        else:
            print("Compiling MAGEMin ...")

        # Compile MAGEMin
        if verbose >= 2:
            subprocess.run("(cd MAGEMin && make)", shell=True, text=True)

        else:
            with open(os.devnull, "w") as null:
                subprocess.run("(cd MAGEMin && make)", shell=True, stdout=null, stderr=null)

    else:
        # MAGEMin repo not found
        sys.exit("MAGEMin does not exist!")

    print("Compiling successful!")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# check non-matching strings !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def check_non_matching_strings(list1, list2):
    """
    """
    set1 = set(list1)
    set2 = set(list2)

    non_matching_strings = set1 - set2

    return bool(non_matching_strings)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# parse list of strings !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def parse_list_of_strings(arg):
    """
    """
    if arg == "all":
        return arg

    try:
        str_list = ast.literal_eval(arg)

        if (
            isinstance(str_list, list) and
            all(isinstance(item, str) for item in str_list)
        ):
            return str_list

        else:
            raise argparse.ArgumentTypeError(
                f"Invalid list: {arg} ...\nIt must contain a valid list of strings ..."
            )

    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid list: {arg} ...\nIt must contain a valid list of strings ..."
        )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# parse arguments !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def parse_arguments():
    """
    """
    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add the command-line arguments
    parser.add_argument("--Pmin", type=int, required=False)
    parser.add_argument("--Pmax", type=int, required=False)
    parser.add_argument("--Tmin", type=int, required=False)
    parser.add_argument("--Tmax", type=int, required=False)
    parser.add_argument("--sampleid", type=str, required=False)
    parser.add_argument("--normox", type=parse_list_of_strings, required=False)
    parser.add_argument("--dataset", type=str, required=False)
    parser.add_argument("--res", type=int, required=False)
    parser.add_argument("--benchmarks", type=str, required=False)
    parser.add_argument("--nsamples", type=int, required=False)
    parser.add_argument("--emsonly", type=str, required=False)
    parser.add_argument("--maskgeotherm", type=str, required=False)
    parser.add_argument("--targets", type=parse_list_of_strings, required=False)
    parser.add_argument("--mlmodels", type=parse_list_of_strings, required=False)
    parser.add_argument("--tune", type=str, required=False)
    parser.add_argument("--epochs", type=int, required=False)
    parser.add_argument("--batchprop", type=float, required=False)
    parser.add_argument("--kfolds", type=int, required=False)
    parser.add_argument("--npca", type=int, required=False)
    parser.add_argument("--kcluster", type=int, required=False)
    parser.add_argument("--parallel", type=str, required=False)
    parser.add_argument("--nprocs", type=int, required=False)
    parser.add_argument("--seed", type=int, required=False)
    parser.add_argument("--palette", type=str, required=False)
    parser.add_argument("--figdir", type=str, required=False)
    parser.add_argument("--verbose", type=int, required=False)

    # Parse the command-line arguments
    args = parser.parse_args()

    return args

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# check arguments !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def check_arguments(args, script):
    """
    """
    # Arguments
    Pmin = args.Pmin
    Pmax = args.Pmax
    Tmin = args.Tmin
    Tmax = args.Tmax
    sampleid = args.sampleid
    normox = args.normox
    dataset = args.dataset
    res = args.res
    benchmarks = args.benchmarks
    nsamples = args.nsamples
    emsonly = args.emsonly
    maskgeotherm = args.maskgeotherm
    targets = args.targets
    mlmodels = args.mlmodels
    tune = args.tune
    epochs = args.epochs
    batchprop = args.batchprop
    kfolds = args.kfolds
    npca = args.npca
    kcluster = args.kcluster
    parallel = args.parallel
    nprocs = args.nprocs
    seed = args.seed
    palette = args.palette
    figdir = args.figdir
    verbose = args.verbose

    # MAGEMin oxide options
    oxide_list_magemin = ["SIO2", "AL2O3", "CAO", "MGO", "FEO", "K2O", "NA2O",
                          "TIO2", "FE2O3", "CR2O3", "H2O" ]

    valid_args = {}

    # Check arguments and print
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"Running {script} with:")

    if Pmin is not None:
        print(f"    P min: {Pmin} GPa")

        valid_args["Pmin"] = Pmin

    if Pmax is not None:
        print(f"    P max: {Pmax} GPa")

        valid_args["Pmax"] = Pmax

    if Tmin is not None:
        print(f"    T min: {Tmin} K")

        valid_args["Tmin"] = Tmin

    if Tmax is not None:
        print(f"    T max: {Tmax} K")

        valid_args["Tmax"] = Tmax

    if sampleid is not None:
        print(f"    sample id: {sampleid}")

        valid_args["sampleid"] = sampleid

    if normox is not None:
        if normox != "all":
            if check_non_matching_strings(normox, oxide_list_magemin):
                print("Warning: invalid --normox argument!")
                print(f"    Can only normalize to oxides {oxide_list_magemin}")
                print("Using normox = 'all'")

                normox = "all"

            else:
                print(f"    Normalizing composition to: {oxide}")

        valid_args["normox"] = normox

    if dataset is not None:
        print(f"    dataset: {dataset}")

        valid_args["dataset"] = dataset

    if res is not None:
        if res > 128:
            print("Warning: invalid --res argument!")
            print("    --res must be <= 128")
            print("Using res = 128")

            res = 128

        print(f"    resolution: {res} pts")

        if Tmin is not None and Tmax is not None:
            print(f"    T range: [{Tmin}, {Tmax}, {res}] K")

        if Pmin is not None and Pmax is not None:
            print(f"    P range: [{Pmin}, {Pmax}, {res}] GPa")

        valid_args["res"] = res

    if benchmarks is not None:
        benchmarks = benchmarks.lower() == "true" if benchmarks else False

        if not isinstance(benchmarks, bool):
            print("Warning: invalid --benchmarks argument!")
            print("    --benchmarks must be True or False")
            print("Using benchmarks = True")

            benchmarks = True

        print(f"    benchmarks: {benchmarks}")

        valid_args["benchmarks"] = benchmarks

    if nsamples is not None:
        print(f"    n synthetic samples: {nsamples}")

        valid_args["nsamples"] = nsamples

    if emsonly is not None:
        emsonly = emsonly.lower() == "true" if emsonly else False

        if not isinstance(emsonly, bool):
            print("Warning: invalid --emsonly argument!")
            print("    --emsonly must be True or False")
            print("Using emsonly = False")

            emsonly = False

        print(f"    endmembers only: {emsonly}")

        valid_args["emsonly"] = emsonly

    if maskgeotherm is not None:
        maskgeotherm = maskgeotherm.lower() == "true" if maskgeotherm else False

        if not isinstance(maskgeotherm, bool):
            print("Warning: invalid --maskgeotherm argument!")
            print("    --maskgeotherm must be True or False")
            print("Using maskgeotherm = True")

            maskgeotherm = True

        print(f"    mask geotherm: {maskgeotherm}")

        valid_args["maskgeotherm"] = maskgeotherm

    if targets is not None:
        print(f"    targets: {targets}")

        valid_args["targets"] = targets

    if mlmodels is not None:
        print(f"    RocML models: {mlmodels}")

        valid_args["mlmodels"] = mlmodels

    if tune is not None:
        tune = tune.lower() == "true" if tune else False

        if not isinstance(tune, bool):
            print("Warning: invalid --tune argument!")
            print("    --tune must be True or False")
            print("Using tune = False")

            tune = False

        print(f"    hyperparameter tuning: {tune}")

        valid_args["tune"] = tune

    if epochs is not None:
        print(f"    NN epochs: {epochs}")

        valid_args["epochs"] = epochs

    if batchprop is not None:
        print(f"    NN batch proportion: {batchprop}")

        valid_args["batchprop"] = batchprop

    if kfolds is not None:
        print(f"    kfolds: {kfolds}")

        valid_args["kfolds"] = kfolds

    if npca is not None:
        print(f"    pca components: {npca}")

        valid_args["npca"] = npca

    if kcluster is not None:
        print(f"    k-means clusters: {kcluster}")

        valid_args["kcluster"] = kcluster

    if parallel is not None:
        parallel = parallel.lower() == "true" if parallel else False

        if not isinstance(parallel, bool):
            print("Warning: invalid --parallel argument!")
            print("    --parallel must be True or False")
            print("Using parallel = True")

            parallel = True

        print(f"    parallel: {parallel}")

        valid_args["parallel"] = parallel

    if nprocs is not None:
        if nprocs > os.cpu_count():
            print(f"Warning: {nprocs} is greater than {os.cpu_count()} available processors!")
            print(f"Setting number of processors to: {os.cpu_count() - 2} ...")

            nprocs = os.cpu_count() - 2

        print(f"    processors: {nprocs}")

        valid_args["nprocs"] = nprocs

    if seed is not None:
        print(f"    seed: {seed}")

        valid_args["seed"] = seed

    if palette is not None:
        if palette not in ["viridis", "bone", "pink", "seismic", "grey", "blues"]:
            print(f"Warning: invalid --palette argument ({palette})!")
            print("    Palettes: viridis, bone, pink, seismic, grey, blues")
            print("Using palette = 'bone'")

            palette = "bone"

        print(f"    palette: {palette}")

        valid_args["palette"] = palette

    if figdir is not None:
        print(f"    fig directory: {figdir}")

        valid_args["figdir"] = figdir

    if verbose is not None:
        if not isinstance(verbose, int):
            print("Warning: invalid --verbose argument!")
            print("    --verbose must be an integer or 0")
            print("Using verbose = 1")

            verbose = 1

        print(f"    verbose: {verbose}")

        valid_args["verbose"] = verbose

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    return valid_args