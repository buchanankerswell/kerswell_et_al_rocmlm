#!/bin/zsh

# Clock time
SECONDS=0

# Exit if any command fails
set -e
#
# Make MAD with MAGEMin
python/mad-create.py

# Print clock time
t=$SECONDS
printf '\ntime taken: %d days, %d minutes, %d seconds\n' \
  "$(( t/86400 ))" "$(( t/60 - 1440*(t/86400) ))" "$(( t ))"
exit 0