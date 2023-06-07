from magemin import download_github_submodule

# MAGEMin repo
repository_url = "https://github.com/ComputationalThermodynamics/MAGEMin.git"
submodule_dir = "MAGEMin"

# Download and recurse submodule
download_github_submodule(repository_url, submodule_dir)
