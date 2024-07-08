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