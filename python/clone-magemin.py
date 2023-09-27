from rocml import download_github_submodule

# MAGEMin repo url
repository_url = "https://github.com/ComputationalThermodynamics/MAGEMin.git"

 # Version 1.3.2
commit_hash = "69017cb"

# Submodule directory to clone reop into
submodule_dir = "MAGEMin"

# Download and recurse submodule
print(f"Cloning MAGEMin from: {repository_url}")
print(f"Checking out commit: {commit_hash}")

# Clone and unpack submodule
download_github_submodule(repository_url, submodule_dir, commit_hash)

print("clone-magemin.py done!")
