from scripting import (parse_arguments, check_arguments, download_github_submodule,
                       compile_magemin)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "clone-magemin.py")

# Load valid arguments
locals().update(valid_args)

# MAGEMin repo url
repository_url = "https://github.com/ComputationalThermodynamics/MAGEMin.git"

 # Version 1.3.2
commit_hash = "69017cb"

# Submodule directory to clone reop into
submodule_dir = "MAGEMin"

# Download and recurse submodule
print(f"Cloning MAGEMin from {repository_url} ...")
print(f"Checking out commit {commit_hash} (v.1.3.2) ...")

# Clone and unpack submodule
download_github_submodule(repository_url, submodule_dir, commit_hash)

# Compile magemin with end members only (or solution models)
compile_magemin(hpc=hpc)

print("clone-magemin.py done!")
