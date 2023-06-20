from magemin import print_session_info

# Provide the paths to the Conda YAML file and Makefile
conda_file = 'python/conda-environment.yaml'
makefile = 'Makefile'

# Call the function
print_session_info(conda_file, makefile)