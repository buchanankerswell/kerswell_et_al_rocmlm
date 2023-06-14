from magemin import (
    random_sample_for_MAGEMin,
    normalize_sample,
    create_MAGEMin_input,
    run_MAGEMin
)

# PT range
P_range = [10, 110, 10]
T_range = [500, 2500, 200]

# Compositions: [SiO2, Al2O3, CaO, MgO, FeO, K2O, Na2O, TiO2, Fe2O3, Cr2O3, H2O]

# Compositions from the literature

# Depleted MORB Mantle
dmm_workman_hart_2005 = [44.71, 3.98, 3.17, 38.73, 8.18, 0.006, 0.13, 0.13, 0.01, 0.57, 0.0]
# Primitive Upper Mantle
pum_mcdonough_sun_1996 = [44.90, 4.44, 3.54, 37.71, 8.03, 0.029, 0.36, 0.20, 0.01, 0.38, 0.0]

# MAGEMin Test compositions

# Bulk rock composition of Peridotite from Holland et al., 2018
peridotite_KLB1 = [38.49, 1.776, 2.824, 50.566, 5.886, 0.01, 0.25, 0.10, 0.096, 0.109, 0.0]
# Icelandic basalt Yang et al., 1996
basalt_RE46 = [50.72, 9.16, 15.21, 16.25, 7.06, 0.01, 1.47, 0.39, 0.35, 0.01, 0.0]
# NMORB Gale et al., 2013
basalt_NMORB = [53.21, 9.41, 12.21, 12.21, 8.65, 0.09, 2.90, 1.21, 0.69, 0.02, 0.0]
# MIX1-G, Hirschmann et al., 2003
pyroxenite_MIXG1 = [45.25, 8.89, 12.22, 24.68, 6.45, 0.03, 1.39, 0.67, 0.11, 0.02, 0.0]
# High Al basalt Baker 1983
basalt_high_Al = [50.40, 12.96, 11.31, 7.68, 8.63, 0.54, 3.93, 0.79, 0.41, 0.01, 0.0]
# Tonalite 101
tonalite_101 = [66.01, 11.98, 7.06, 4.16, 5.30, 1.57, 4.12, 0.66, 0.97, 0.01, 50.0]
# Wet basalt
basalt_wet = [50.0810, 8.6901, 11.6698, 12.1438, 7.7832, 0.215, 1.0059, 0.467, 0.01, 5.4364]
# Test unidentified from MAGEMin doc
test = [41.49, 1.57, 4.824, 52.56, 5.88, 0.01, 0.25, 0.1, 0.1, 0.0, 0.0]

# Write MAGEMin input
run_name = "test"
sample = test
sample_norm = normalize_sample(sample)
create_MAGEMin_input(P_range,T_range, sample_norm, run_name=run_name)

# Run MAGEMin
run_MAGEMin("MAGEMin/", run_name=run_name)

## Select sample randomly from Earthchem dataset and run MAGEMin n-times
#
## Random samples
#n = 3
#
#for i in range(n):
#    # Sample Earthchem
#    sample = random_sample_for_MAGEMin(datafile="data/earthchem-ig.csv")
#    sample_norm = normalize_sample(sample)
#
#    # Write MAGEMin input
#    run_name = f"earthchem-sample-{i}"
#    create_MAGEMin_input(P_range,T_range, sample_norm, run_name=run_name)
#
#    # Run MAGEMin
#    run_MAGEMin("MAGEMin/", run_name=run_name, comp_type="wt")