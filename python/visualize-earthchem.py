from magemin import read_geochemical_data, plot_harker_diagram

data = read_geochemical_data("data/earthchem-ig.csv")
plot_harker_diagram(
    data,
    x_oxide="SiO2",
    y_oxide=["CaO", "FeO", "MgO", "Al2O3"],
    filename="earthchem-ig-harker.png"
)