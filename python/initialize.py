from scripting import download_and_unzip, compile_magemin, compile_perplex, print_session_info

# Download assets from OSF
download_and_unzip(("https://files.osf.io/v1/resources/k23tb/providers/osfstorage/"
                   "649149796513ba03733a3536/?zip="), "all", "assets")

# Get MAGEMin
compile_magemin()

# Get Perple_X
compile_perplex()

# Print session info
print_session_info("python/conda-environment.yaml")

print("Initializing done!")