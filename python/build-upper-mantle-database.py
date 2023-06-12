from magemin import normalize_concentrations, create_MAGEMin_input, run_MAGEMin

# PT range
P_range = [10, 100, 1]
T_range = [500, 2500, 20]

# Composition [SiO2, Al2O3, CaO, MgO, FeOt, K2O, Na2O, TiO2, O(Fe2O3), Cr2O3, H2O]
dmm_workman_hart_2005 = [44.71, 3.98, 3.17, 38.73, 8.18, 0.006, 0.13, 0.13, 0.01, 0.57, 0.0]
pum_mcdonough_sun_1996 = [44.90, 4.44, 3.54, 37.71, 8.03, 0.029, 0.36, 0.20, 0.01, 0.38, 0.0]
peridotite_KLB1 = [38.49, 1.77, 2.824, 50.56, 5.88, 0.01, 0.25, 0.10, 0.1, 0.11, 0.0]
pyroxenite_MIXG1 = [45.25, 8.89, 12.22, 24.68, 6.45, 0.03, 1.39, 0.67, 0.11, 0.02, 0.0]
basalt_RE46 = [50.72, 9.16, 15.21, 16.25, 7.06, 0.01, 1.47, 0.39, 0.35, 0.01, 0.0]
basalt_NMORB = [53.21, 9.41, 12.21, 12.21, 8.65, 0.09, 2.90, 1.21, 0.69, 0.02, 0.0]
tonalite_101 = [66.01, 11.98, 7.06, 4.16, 5.30, 1.57, 4.12, 0.66, 0.97, 0.01, 50.0]
test = [41.49, 1.57, 4.824, 52.56, 5.88, 0.01, 0.25, 0.1, 0.1, 0.0, 0.0]

# Write MAGEMin input
run_name = "test"
create_MAGEMin_input(
    P_range=P_range,
    T_range=T_range,
    composition=normalize_concentrations(
        concentrations=test,
        #components=["CaO", "FeOt", "MgO", "SiO2"]
        components="all"
    ),
    mode=0,
    run_name=run_name
)

# Run MAGEMin
run_MAGEMin(
    program_path="MAGEMin/",
    run_name=run_name,
    comp_type="mol",
    database="ig",
    parallel=True,
    nprocs=8,
    verbose=False
)