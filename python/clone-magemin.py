from rocml import download_github_submodule

# MAGEMin repo url
repository_url = "https://github.com/ComputationalThermodynamics/MAGEMin.git"

# Submodule directory to clone reop into
submodule_dir = "MAGEMin"

# Download and recurse submodule
print(f"Cloning MAGEMin from: {repository_url}")

# Clone and unpack submodule
download_github_submodule(repository_url, submodule_dir)

print("clone-magemin.py done!")
