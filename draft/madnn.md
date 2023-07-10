# Introduction

The dominant mineral phases in Earth's upper mantle are olivine, ringwoodite, bridgmanite, and ferropericlase [@ringwood1975; @ringwood1991]. These four phases alone may comprise up to 60–90% of the mantle, depending on the assumed mantle composition [e.g., @stixrude2012]. Work since the 1950's has established that these Mg-Fe-rich minerals form by a series of discrete phase changes, defining critical mantle transition zones (MTZs, @eq:mantle-transitions) near 410 km and 660 km depths---inducing sharp contrasts in the physical properties of the mantle [e.g, density and elasticity, @dziewonski1981; @ita1992].

\begin{align}
	\text{olivine} \xrightarrow{\text{410 km}} \text{ringwoodite} &\xrightarrow{\text{660 km}} \text{bridgmanite} + \text{ferropericlase} \label{eq:mantle-transitions} \\
	\text{(Mg,Fe)}_{2}\text{SiO}_{4} \xrightarrow{\text{410 km}} \text{(Mg,Fe)}_{2}\text{SiO}_{4} &\xrightarrow{\text{660 km}} \text{(Mg,Fe)}\text{SiO}_{3} + \text{(Mg,Fe)}\text{O} \nonumber
\end{align}

Previous work demonstrates that mantle convection, melting, and plate tectonics are strongly impacted by the sharp contrast in physical properties associated with MTZs [@fukao2001; @ringwood1991; @schubert1975; @kuritani2019; @karato2001; @wang2015; @jenkins2016], although the physio-chemical nature of MTZs remains under vigorous investigation [@goes2022; @pearson2014; @yoshino2008; @waszek2021; @kiseeva2018; @fei2017; @zhou2022]. Notwithstanding, modeling the interplay between plate tectonics and MTZs is possible with numerical geodynamic experiments that approximate pressure-temperature (PT)-dependent phase changes in the upper mantle. This approach has generated many important models and hypotheses regarding how plate motions, viscosity contrasts, and Clapeyron slopes of phase changes (e.g., @eq:mantle-transitions) affect plate velocities and slab stagnation at MTZs [@agrusta2017; @li2019; @torii2007]---with implications towards understanding deep water cycling and flux melting in the deep Earth [e.g., @yang2020]. However, these experiments assume fixed mantle compositions and neglect the effects of compositional changes due to fluid-rock interactions and partial melting. Advancing towards more comprehensive models of plate motions, deep water cycling, and mantle melting at MTZs requires a leap to fully implementing iterative PT-composition-time (PTXt)-dependent phase changes in numerical geodynamic simulations.

Even with simple parameterizations, however, implementing fully iterative PTXt-dependent phase changes in large-scale geodynamic simulations is intractable because Gibbs free energy minimization (GFEMs) programs [e.g., @connolly2009; @riel2022] remain too slow to converge on stable solutions for mineral assemblage diagrams (MADs). For example, assuming a large-scale 2D geodynamic simulation has 900 x 300 nodes with a 100-second MAD compute time at each node (the best-case scenario for current GFEM programs, see [@fig:benchmark-times] and [@tbl:benchmark-times-table]), it would take 7500 hours to predict stable mineral assemblages across the model domain for a single timestep. While recent parallelized GFEM programs [@riel2022] have increased efficiency dramatically (up to 11.2x improvement; [@fig:benchmark-times] and [@tbl:benchmark-times-table]), computing MADs iteratively during geodynamic simulations requires GFEM efficiency on the order of milliseconds to be feasible. A 10$^5$ rate of improvement (1$^2$–10$^{-3}$ s) seems unlikely within the current GFEM paradigm and applying parallelization across thousands of CPU/GPU cores is inaccessible in many cases.

![Benchmark results for GFEM programs MAGEMin [dashed lines with square, @riel2022] and Perple_X [solid lines with circles, @connolly2009]. Note that MAGEMin was ran in parallel on 6 CPU cores, while Perple_X has no parallel capabilities. In the best case for a 128x128 PT grid, MADs take 170 seconds to compute ([@tbl:benchmark-times-table]).](assets/figs/benchmark-times.png){#fig:benchmark-times width=100%}

To overcome the intractability of iteratively GFEM programs with large-scale geodynamic simulations, we propose a novel approach for inferring MADs using pre-trained neural networks (referred to here as MADNNs). We hypothesize that MADNNs can improve efficiency by up to 5 orders of magnitude versus incumbent GFEM programs for computing PTXt-dependent phase changes in the mantle. If true, real-time inference of PTXt-dependent phase changes at the individual node-scale in geodynamic simulations will be feasible---enabling new models of tectonic plate behavior, deep water cycling, and mantle melting at MTZs. Moreover, the approach detailed here is generalizable and has potential for adoption to models of other Earth systems. If false, we will demonstrate the practical limitations of applying neural networks to petrological datasets---a critical step for discovering alternative approaches for implementing PTXt-dependent phase changes in numerical geodynamic simulations.

# Methods

## Computing Datasets for MADNN Training

### PT Conditions in the Upper Mantle

High-pressure experiments constrain the reaction $\text{olivine} \xrightarrow{\text{410 km}} \text{ringwoodite}$ between 14.0 ± 1.0 GPa and 1600 ± 400 K with Clapeyron slopes between 2.4x10$^{-3}$ ± 1.4x10$^{-3}$ GPa/K  [@akaogi1989; @katsura1989; @morishima1994; @li2019] and the reaction $\text{ringwoodite} \xrightarrow{\text{660 km}} \text{bridgmanite} + \text{ferropericlase}$ between 24.0 ± 1.5 GPa and 1600 ± 400 K with negative Clapeyron slopes between -2.0x10$^{-3}$ ± 1.6x10$^{-3}$ GPa/K  [@akaogi2007; @bina1994; @litasov2005; @katsura2003; @ito1990; @ito1982; @ito1989a; @ito1989b; @hirose2002; @ishii2018; @li2019]. Our training dataset is therefore computed across 1.0–28.0 GPa and 773–2273 K ([@fig:madnn-training-pt-range]). This PT range encompasses expected conditions from the Moho to the 660 km MTZ---including cold subduction zone conditions---ensuring applicability of our MADNN to the entire upper mantle.

![PT diagram showing experimentally-derived phase boundaries for the 410 and 660 km MTZs and the range of conditions for computing MADNN training data. Phase boundaries are calculated after @li2019. Geotherm 1 represents a cold subduction gradient at 0.5 ˚C/km, while geotherms 2 and 3 assume a 0.5 ˚C/km mantle adiabat with a 1473 ˚C and 1773 ˚C mantle potential temperature, respectively.](assets/figs/madnn-training-pt-range.png){#fig:madnn-training-pt-range width=100%}

### Chemical Compositions of the Upper Mantle

Existing estimates for bulk compositions of the upper mantle are based on analyses of high-pressure-high-temperature melting experiments and mantle-derived xenoliths, kimberlites, and alkali basalts [e.g., @allegre1984; @green1979; @ringwood1962; @jagoutz1979; @sun1982; @ringwood1991; @palme2003; @stracke2021]. [@tbl:benchmark-comps] provides some well-referenced examples, including hypothetical mantle compositions with varying degrees of differentiation by partial melting [Primitive Upper Mantle: PUM, and Depleted MORB Mantle: DMM, @sun1989; @workman2005], as well as real and hypothetical products of mantle melting [Iclandic Basalt: RE46 and Normal MORB: NMORB, @gale2013; @yang1996]. Our training dataset includes these compositions because they approximate the range of expected (average) mantle compositions from the seafloor to the lower upper mantle---tailoring our MADNN for the entire upper mantle.

| Name   |   SiO$_2$ |   Al$_2$O$_3$ |   CaO |   MgO |   FeO |   K$_2$O |   Na$_2$O |   TiO$_2$ |   O$_2$ |   Cr$_2$O$_3$ |   H$_2$O |
|:-------|----------:|--------------:|------:|------:|------:|---------:|----------:|----------:|--------:|--------------:|---------:|
| DMM    |     44.71 |          3.98 |  3.17 | 38.73 |  8.18 |    0.006 |      0.13 |      0.13 |    0.01 |          0.57 |        0 |
| NMORB  |     53.21 |          9.41 | 12.21 | 12.21 |  8.65 |    0.090 |      2.90 |      1.21 |    0.69 |          0.02 |        0 |
| PUM    |     44.90 |          4.44 |  3.54 | 37.71 |  8.03 |    0.029 |      0.36 |      0.20 |    0.01 |          0.38 |        0 |
| RE46   |     50.72 |          9.16 | 15.21 | 16.25 |  7.06 |    0.010 |      1.47 |      0.39 |    0.35 |          0.01 |        0 |

: Estimated bulk compositions (in wt. % oxides) for the mantle. {#tbl:benchmark-comps}

# Results

## GFEM Benchmarking

Benchmarking GFEM programs was a necessary first step for estimating the time required for building MADNN training datasets and quantifying the efficiency of incumbent GFEM programs (@fig:benchmark-times), which our MADNNs will need to beat to be considered an advancement beyond the status-quo. Estimated bulk compositions for primitive and depleted mantle-derived rocks ([@tbl:benchmark-comps]) were used for benchmarking [MAGEMin](https://github.com/ComputationalThermodynamics/MAGEMin) and [Perple_X](https://github.com/ondrolexa/Perple_X) [@riel2022; @connolly2009]. [@tbl:benchmark-times-table] shows the computation times with respect to various PT grid resolutions (8x8, 16x16, 32x32, 64x64, 128x128). All computations were made on a Macbook Pro (2022; M2 chip) with macOS 13.4 and Python 3.11.4. Note that MAGEMin was ran on 6 CPU cores in parallel, while Perple_X does not have parallel capabilities.

| Name   | P (GPa)   | T (K)    |   Grid Size |   MAGEMin (s) |   Perple_X (s) |   PPX/MGM |
|:-------|----------:|---------:|------------:|--------------:|---------------:|----------:|
| DMM    | 1.0-24.0  | 773-2273 |          64 |           2.0 |            9.4 |       4.7 |
| NMORB  | 1.0-24.0  | 773-2273 |          64 |           1.3 |           13.6 |      10.5 |
| PUM    | 1.0-24.0  | 773-2273 |          64 |           0.7 |           10.1 |      14.4 |
| RE46   | 1.0-24.0  | 773-2273 |          64 |          11.6 |           14.7 |       1.3 |
| DMM    | 1.0-24.0  | 773-2273 |         256 |           5.2 |           34.4 |       6.6 |
| NMORB  | 1.0-24.0  | 773-2273 |         256 |           6.3 |           50.3 |       8.0 |
| PUM    | 1.0-24.0  | 773-2273 |         256 |           3.4 |           36.2 |      10.6 |
| RE46   | 1.0-24.0  | 773-2273 |         256 |          43.6 |           52.1 |       1.2 |
| DMM    | 1.0-24.0  | 773-2273 |        1024 |          11.9 |          118.7 |      10.0 |
| NMORB  | 1.0-24.0  | 773-2273 |        1024 |          21.5 |          180.6 |       8.4 |
| PUM    | 1.0-24.0  | 773-2273 |        1024 |          12.3 |          126.4 |      10.3 |
| RE46   | 1.0-24.0  | 773-2273 |        1024 |         154.7 |          190.7 |       1.2 |
| DMM    | 1.0-24.0  | 773-2273 |        4096 |          42.6 |          428.2 |      10.1 |
| NMORB  | 1.0-24.0  | 773-2273 |        4096 |          82.4 |          642.8 |       7.8 |
| PUM    | 1.0-24.0  | 773-2273 |        4096 |          41.3 |          461.6 |      11.2 |
| RE46   | 1.0-24.0  | 773-2273 |        4096 |         604.3 |          674.2 |       1.1 |
| DMM    | 1.0-24.0  | 773-2273 |       16384 |         170.2 |         1595.3 |       9.4 |
| NMORB  | 1.0-24.0  | 773-2273 |       16384 |         304.0 |         2263.2 |       7.4 |
| PUM    | 1.0-24.0  | 773-2273 |       16384 |         147.2 |         1654.6 |      11.2 |
| RE46   | 1.0-24.0  | 773-2273 |       16384 |        2265.4 |         2404.0 |       1.1 |

: Computation times for various bulk mantle compositions. {#tbl:benchmark-times-table}

## GFEM Comparisons

### MADs

![caption](assets/figs/benchmark/PUM-128x128/comp-PUM-128x128-StableSolutions.png){#fig:benchmark-PUM-solutions}

### Density

![caption](assets/figs/benchmark/PUM-128x128/comp-PUM-128x128-DensityOfFullAssemblage.png){#fig:benchmark-PUM-density}

### Liquid Fraction

![caption](assets/figs/benchmark/PUM-128x128/comp-PUM-128x128-LiquidFraction.png){#fig:benchmark-PUM-liquid}

### Seismic Wave Velocities

![caption](assets/figs/benchmark/PUM-128x128/comp-PUM-128x128-Vp.png){#fig:benchmark-PUM-vp}

\clearpage

# References

<div id="refs"></div>

\clearpage

# Appendix

## GFEM Benchmarking Results

![caption](assets/figs/benchmark/DMM-128x128/comp-DMM-128x128-StableSolutions.png){#fig:benchmark-DMM-solutions}

![caption](assets/figs/benchmark/DMM-128x128/comp-DMM-128x128-DensityOfFullAssemblage.png){#fig:benchmark-DMM-density}

![caption](assets/figs/benchmark/DMM-128x128/comp-DMM-128x128-LiquidFraction.png){#fig:benchmark-DMM-liquid}

![caption](assets/figs/benchmark/DMM-128x128/comp-DMM-128x128-Vp.png){#fig:benchmark-DMM-vp}

![caption](assets/figs/benchmark/NMORB-128x128/comp-NMORB-128x128-StableSolutions.png){#fig:benchmark-NMORB-solutions}

![caption](assets/figs/benchmark/NMORB-128x128/comp-NMORB-128x128-DensityOfFullAssemblage.png){#fig:benchmark-NMORB-density}

![caption](assets/figs/benchmark/NMORB-128x128/comp-NMORB-128x128-LiquidFraction.png){#fig:benchmark-NMORB-liquid}

![caption](assets/figs/benchmark/NMORB-128x128/comp-NMORB-128x128-Vp.png){#fig:benchmark-NMORB-vp}

![caption](assets/figs/benchmark/RE46-128x128/comp-RE46-128x128-StableSolutions.png){#fig:benchmark-RE46-solutions}

![caption](assets/figs/benchmark/RE46-128x128/comp-RE46-128x128-DensityOfFullAssemblage.png){#fig:benchmark-RE46-density}

![caption](assets/figs/benchmark/RE46-128x128/comp-RE46-128x128-LiquidFraction.png){#fig:benchmark-RE46-liquid}

![caption](assets/figs/benchmark/RE46-128x128/comp-RE46-128x128-Vp.png){#fig:benchmark-RE46-vp}
