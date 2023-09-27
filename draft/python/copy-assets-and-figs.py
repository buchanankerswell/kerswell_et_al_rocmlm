from draft import parse_arguments, check_arguments, copy_assets, copy_figs

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "copy-assets-and-figs.py")

# Load valid arguments
locals().update(valid_args)

# Copy assets
copy_assets()

# Copy figs
copy_figs(ms)

print("copy-assets-and-figs.py done!")