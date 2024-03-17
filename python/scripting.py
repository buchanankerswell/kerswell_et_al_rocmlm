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
import datetime
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
# check os !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def check_os():
    """
    """
    system = platform.system()

    if system == "Darwin":
        os = "macos"

    elif system == "Linux":
        os = "linux"

    else:
        print("Operating system is unrecognized ...")
        os = None

    return os

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# print session info !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def print_session_info(condafile):
    """
    """
    # Print session info
    print("Session info:")

    print(f"    Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Print Python version
    version_string = ".".join(map(str, sys.version_info))

    print(f"    Python Version: {version_string}")

    # Print package versions
    print("    Loaded packages:")

    conda_packages = get_conda_packages(condafile)

    for package in conda_packages:
        if isinstance(package, str) and package != "python":
            package_name = package.split("=")[0]

            try:
                version = pkg_resources.get_distribution(package_name).version

                print(f"        {package_name} Version: {version}")

            except pkg_resources.DistributionNotFound:
                print(f"        {package_name} not found ...")

    # Print operating system information
    os_info = platform.platform()

    print(f"    Operating System: {os_info}")

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

        return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# download and unzip !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def download_and_unzip(url, filename, destination):
    try:
        # Download the file
        response = urllib.request.urlopen(url)

        with open("temp.zip", "wb") as zip_file:
            zip_file.write(response.read())

        # Extract the contents of the zip file
        with zipfile.ZipFile("temp.zip", "r") as zip_ref:
            if filename == "all":
                zip_ref.extractall(destination)

            else:
                target_found = False

                # Check if the target ZIP file is present in the archive
                for file_info in zip_ref.infolist():
                    if file_info.filename == filename:
                        zip_ref.extract(file_info, destination)

                        # Check extension
                        _, file_ext = os.path.splitext(filename)

                        if file_ext == ".zip":
                            with zipfile.ZipFile(f"{destination}/{filename}", "r") as in_zip:
                                in_zip.extractall(destination)

                            # Remove zip file
                            os.remove(f"{destination}/{filename}")

                        target_found = True
                        break

                if not target_found:
                    raise Exception(f"{filename} not found in zip archive!")

        # Remove the temporary zip file
        os.remove("temp.zip")

    except urllib.error.URLError as e:
        raise Exception(f"Unable to download from {url}!")

    except zipfile.BadZipFile as e:
        raise Exception(f"The downloaded file is not a valid zip file!")

    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}!")

    return None

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
        print(f"Cloning {repository_url} repo [commit: {commit_hash}] ...")

        repo = Repo.clone_from(repository_url, submodule_dir, recursive=True)
        repo.git.checkout(commit_hash)

    except Exception as e:
        print(f"An error occurred while cloning the GitHub repository: {e} ...")

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# compile magemin !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compile_magemin(commit_hash="7293cbc", verbose=1):
    """
    """
    # Config directory
    config_dir = "assets/config"

    # Get source
    download_github_submodule("https://github.com/ComputationalThermodynamics/MAGEMin.git",
                              "tmp", commit_hash)

    try:
        # Configure
        print(f"Compiling MAGEMin ...")

        # Compile
        if verbose >= 2:
            subprocess.run("(cd tmp && make)", shell=True, text=True)

        else:
            with open(os.devnull, "w") as null:
                subprocess.run("(cd tmp && make)", shell=True, stdout=null, stderr=null)

        # Move magemin program into directory
        os.makedirs("MAGEMin")
        shutil.move("tmp/MAGEMin", "MAGEMin")
        shutil.rmtree("tmp")

        print("MAGEMin installation successful!")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    except Exception as e:
        print(f"Compilation error: !!! {e} !!!")

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# compile perplex !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compile_perplex():
    """
    """
    # Config directory
    config_dir = "assets/config"

    try:
        url = ("https://www.perplex.ethz.ch/perplex/ibm_and_mac_archives/OSX/"
               "previous_version/Perple_X_7.0.9_OSX_ARM_SP_Apr_16_2023.zip")

        print("Installing Perple_X from:")
        print(f"{url}")

        download_and_unzip(url, "dynamic.zip", "Perple_X")

        print("Perple_X install successful!")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    except Exception as e:
        print(f"Compilation error: !!! {e} !!!")

    return None

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

        if (isinstance(str_list, list) and all(isinstance(item, str) for item in str_list)):
            return str_list

        else:
            raise argparse.ArgumentTypeError(f"Invalid list: {arg} ...\n"
                                             "It must contain a valid list of strings ...")

    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid list: {arg} ...\n"
                                         "It must contain a valid list of strings ...")

    return None

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
    parser.add_argument("--programs", type=parse_list_of_strings, required=False)
    parser.add_argument("--source", type=str, required=False)
    parser.add_argument("--sampleid", type=str, required=False)
    parser.add_argument("--batch", type=int, required=False)
    parser.add_argument("--nbatches", type=int, required=False)
    parser.add_argument("--normox", type=parse_list_of_strings, required=False)
    parser.add_argument("--dataset", type=str, required=False)
    parser.add_argument("--res", type=int, required=False)
    parser.add_argument("--nsamples", type=int, required=False)
    parser.add_argument("--hpendmembers", type=str, required=False)
    parser.add_argument("--hpc", type=str, required=False)
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
    parser.add_argument("--debug", type=str, required=False)
    parser.add_argument("--visualize", type=str, required=False)

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
    programs = args.programs
    source = args.source
    sampleid = args.sampleid
    batch = args.batch
    nbatches = args.nbatches
    normox = args.normox
    dataset = args.dataset
    res = args.res
    nsamples = args.nsamples
    hpendmembers = args.hpendmembers
    hpc = args.hpc
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
    debug = args.debug
    visualize = args.visualize

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

    if programs is not None:
        print(f"    GFEM programs: {programs}")

        valid_args["programs"] = programs

    if source is not None:
        print(f"    sample source: {source}")

        valid_args["source"] = source

    if sampleid is not None:
        print(f"    sample id: {sampleid}")

        valid_args["sampleid"] = sampleid

    if batch is not None:
        print(f"    sample batch: {batch}")

        valid_args["batch"] = batch

    if nbatches is not None:
        print(f"    n batches: {nbatches}")

        valid_args["nbatches"] = nbatches

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

    if nsamples is not None:
        print(f"    n synthetic samples: {nsamples}")

        valid_args["nsamples"] = nsamples

    if hpendmembers is not None:
        hpendmembers = hpendmembers.lower() == "true" if hpendmembers else False

        if not isinstance(hpendmembers, bool):
            print("Warning: invalid --hpendmembers argument!")
            print("    --hpendmembers must be True or False")
            print("Using hpendmembers = False")

            hpendmembers = False

        print(f"    HP endmembers: {hpendmembers}")

        valid_args["hpendmembers"] = hpendmembers

    if hpc is not None:
        hpc = hpc.lower() == "true" if hpc else False

        if not isinstance(hpc, bool):
            print("Warning: invalid --hpc argument!")
            print("    --hpc must be True or False")
            print("Using hpc = False")

            hpc = False

        print(f"    HPC Setup: {hpc}")

        valid_args["hpc"] = hpc

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

    if debug is not None:
        debug = debug.lower() == "true" if debug else False

        if not isinstance(debug, bool):
            print("Warning: invalid --debug argument!")
            print("    --debug must be True or False")
            print("Using debug = False")

            debug = False

        print(f"    debug: {debug}")

        valid_args["debug"] = debug

    if visualize is not None:
        visualize = visualize.lower() == "true" if visualize else False

        if not isinstance(visualize, bool):
            print("Warning: invalid --visualize argument!")
            print("    --visualize must be True or False")
            print("Using visualize = True")

            visualize = True

        print(f"    visualize: {visualize}")

        valid_args["visualize"] = visualize

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    return valid_args