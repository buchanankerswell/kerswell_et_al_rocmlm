from rocml import download_github_submodule

# MAGEMin repo
repository_url = "https://github.com/ComputationalThermodynamics/MAGEMin.git"
submodule_dir = "MAGEMin"

# Download and recurse submodule
print("Cloning MAGEMin from:")
print(f"    {repository_url}")

download_github_submodule(repository_url, submodule_dir)
