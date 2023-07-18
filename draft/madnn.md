# Introduction

The dominant mineral phases in Earth's upper mantle are olivine, ringwoodite, bridgmanite, and ferropericlase [@ringwood1975; @ringwood1991], comprising up to 60–90% of the mantle's volume [e.g., @stixrude2012]. These Mg-Fe-rich phases form by a series of discrete reactions (@eq:mantle-transitions) that define mantle transition zones (MTZs) near 410 km and 660 km depths beneath Earth's surface. MTZs are characterized by relatively sharp boundaries with contrasting physical properties [e.g, density and elasticity, @dziewonski1981; @ita1992] that strongly impact mantle convection, melting, and plate tectonics [@fukao2001; @ringwood1991; @schubert1975; @kuritani2019; @karato2001; @wang2015; @jenkins2016].

\begin{align}
	\text{olivine} \xrightarrow{\text{410 km}} \text{ringwoodite} &\xrightarrow{\text{660 km}} \text{bridgmanite} + \text{ferropericlase} \label{eq:mantle-transitions} \\
	\text{(Mg,Fe)}_{2}\text{SiO}_{4} \xrightarrow{\text{410 km}} \text{(Mg,Fe)}_{2}\text{SiO}_{4} &\xrightarrow{\text{660 km}} \text{(Mg,Fe)}\text{SiO}_{3} + \text{(Mg,Fe)}\text{O} \nonumber
\end{align}

Although the physio-chemical nature of MTZs remains under vigorous investigation [@goes2022; @pearson2014; @yoshino2008; @waszek2021; @kiseeva2018; @fei2017; @zhou2022], modelling the interplay between plate tectonics and MTZs is possible with numerical geodynamic simulations of mantle flow that implement pressure-temperature (PT)-dependent phase changes (e.g., @eq:mantle-transitions). This approach has generated many important hypotheses implicating MTZs as critical features controlling global plate tectonics and water cycling in the deep Earth [e.g., @agrusta2017; @li2019; @torii2007; @yang2020]. However, the tendency to assume fixed mantle compositions that neglect chemical fractionation from fluid-rock interactions and partial melting limit such numerical experiments to rough first-order approximations of true mantle flow.

Advancing towards more comprehensive models of plate interactions at MTZs requires a leap from modelling PT- to PT-composition-time (PTXt)-dependent phase changes in the mantle. This is currently intractable, however, because Gibbs Free Energy minimization (GFEMs) programs [e.g., @connolly2009; @riel2022] used to calculate PTX-dependent phase relations---referred to as mineral assemblage diagrams, or MADs---remain slow (10$^2$–10$^4$ seconds; [@fig:benchmark-times]). While recent parallelized GFEM programs [@riel2022] have increased efficiency dramatically ([@tbl:benchmark-times-table]), computing MADs iteratively during geodynamic simulations requires GFEM efficiency on the order of $\leq$ 10 milliseconds to be feasible. A rate of improvement from 10$^2$ to 10$^{-2}$ seconds seems unlikely within the current GFEM paradigm, however, and applying parallelization across thousands of CPU/GPU cores is inaccessible in many cases.

![Computational efficieincy for Gibbs Free Energy minimization (GFEM) programs MAGEMin [dashed lines with squares, @riel2022] and Perple_X [solid lines with circles, @connolly2009]. Note that MAGEMin was ran in parallel on 6 CPU cores, while Perple_X has no parallel capabilities. In the best case for a 128x128 resolution PT grid, stable phase relations (i.e., mineral assemblage diagrams, or MADs) take 171.2 seconds to compute ([@tbl:benchmark-times-table]).](assets/figs/benchmark-times.png){#fig:benchmark-times width=100%}

Here we propose an alternative approach for inferring MADs using pre-trained neural networks (referred to as MADNNs). We hypothesize that MADNNs can improve efficiency by up to 4 orders of magnitude versus incumbent GFEM programs for computing PTXt-dependent phase changes in the mantle. If true, real-time inference of PTXt-dependent phase changes at the individual node-scale in geodynamic simulations will be feasible---enabling new models of global tectonic plate behavior, deep water cycling, and mantle melting at MTZs. If false, we will demonstrate the practical limitations of applying neural networks to petrological datasets---a critical step for discovering alternative approaches for implementing PTXt-dependent phase changes in numerical geodynamic simulations.

# Methods

## Computing Datasets for MADNN Training

The following sections describe our design decisions for building MADNN training datasets. Our goal is to compute minimal training data for a broad range of upper mantle PT conditions and chemical compositions---ensuring widespread applicability of our MADNN to geodynamic problems. Here we use GFEM programs [MAGEMin](https://github.com/ComputationalThermodynamics/MAGEMin) and [Perple_X](https://github.com/ondrolexa/Perple_X) [@riel2022; @connolly2009] to compute stable mineral assemblages and rock properties for the upper mantle (density, melt fraction, and seismic wave velocities), which we then use for MADNN training.

### Bulk Chemical Compositions

Existing estimates for bulk chemical compositions of the upper mantle are based on analyses of high-pressure-high-temperature (HP-HT) melting experiments and mantle-derived xenoliths, kimberlites, and basalts [e.g., @allegre1984; @green1979; @ringwood1962; @jagoutz1979; @sun1982; @ringwood1991; @palme2003; @stracke2021]. [@tbl:benchmark-comps] provides some well-referenced examples, including hypothetical mantle compositions with varying degrees of differentiation by partial melting [Primitive Upper Mantle: PUM, and Depleted MORB Mantle: DMM, @sun1989; @workman2005], as well as real and hypothetical products of mantle melting [Iclandic Basalt: RE46, and Normal MORB: NMORB, @yang1996; @gale2013]. MADNN training data currently includes PUM only, which represents the average bulk (pyrolitic) composition of the upper mantle. Training data will eventually be expanded to all compositions in @tbl:benchmark-comps to approximate the range of expected compositions for the entire upper mantle.

| Name   |   SiO$_2$ |   Al$_2$O$_3$ |   CaO |   MgO |   FeO |   K$_2$O |   Na$_2$O |   TiO$_2$ |   O$_2$ |   Cr$_2$O$_3$ |   H$_2$O |
|:-------|----------:|--------------:|------:|------:|------:|---------:|----------:|----------:|--------:|--------------:|---------:|
| DMM    |     44.71 |          3.98 |  3.17 | 38.73 |  8.18 |    0.006 |      0.13 |      0.13 |    0.01 |          0.57 |        0 |
| NMORB  |     53.21 |          9.41 | 12.21 | 12.21 |  8.65 |    0.090 |      2.90 |      1.21 |    0.69 |          0.02 |        0 |
| PUM    |     44.90 |          4.44 |  3.54 | 37.71 |  8.03 |    0.029 |      0.36 |      0.20 |    0.01 |          0.38 |        0 |
| RE46   |     50.72 |          9.16 | 15.21 | 16.25 |  7.06 |    0.010 |      1.47 |      0.39 |    0.35 |          0.01 |        0 |

: Estimated bulk chemical compositions (in wt. % oxides) for the mantle. {#tbl:benchmark-comps}

### PT Conditions

HP experiments constrain the reaction $\text{olivine} \xrightarrow{\text{410 km}} \text{ringwoodite}$ between 14.0 ± 1.0 GPa and 1600 ± 400 K with Clapeyron slopes between 2.4x10$^{-3}$ ± 1.4x10$^{-3}$ GPa/K  [@akaogi1989; @katsura1989; @morishima1994; @li2019]. Likewise, the reaction $\text{ringwoodite} \xrightarrow{\text{660 km}} \text{bridgmanite} + \text{ferropericlase}$ is constrained between 24.0 ± 1.5 GPa and 1600 ± 400 K with negative Clapeyron slopes between -2.0x10$^{-3}$ ± 1.6x10$^{-3}$ GPa/K  [@akaogi2007; @bina1994; @litasov2005; @katsura2003; @ito1990; @ito1982; @ito1989a; @ito1989b; @hirose2002; @ishii2018]. We therefore compute MADs within a rectangular PT region bound between 1.0–28.0 GPa and 773–2273 K to encompass expected conditions for the entire upper mantle---including colder subduction zone conditions ([@fig:madnn-training-pt-range]).

On the other hand, some conditions between 1.0–28.0 GPa and 773–2273 K are not expected to exist in the upper mantle, nor typically modelled during geodynamic simulations. Training MADNNs on the entire MAD datasets would therefore be impractical. For example, MAD results can be spurious and noisy at LT-HP and include high degrees of partial melt at LP-HT. Training data are therefore subset from MADs between two geotherms that represent first-order bounds for expected upper mantle PT conditions ([@fig:madnn-training-pt-range]). The bounding geotherms assume a 1 K/km thermal gradient with a 273 K mantle potential temperature (cold conditions), and a 0.5 K/km thermal gradient with a 1773 K mantle potential temperature (warm conditions). This approach increases MADNN training efficiency while decreasing the probability of erroneous predictions.

![PT diagram showing experimentally-derived phase boundaries for the 410 and 660 km MTZs (colored lines), the range of PT conditions for computing MADs with GFEM programs (shaded grey box), and the subset of MAD data used for MADNN training (shaded blue box with hatches). Geotherm 1 (solid black line) assumes a mantle potential temperature of 273 K with a 1 K/km thermal gradient, representing a lower bound for cold subduction zone settings. Geotherm 2 (dashed black line) assumes a mantle potential temperature of 1773 K with a  0.5 K/km thermal gradient, representing an upper bound for ambient mantle conditions---including divergent plate boundaries. MADNNs are trained on expected PT conditions for the entire upper mantle from 1–28 GPa, while ignoring unexpected cold and warm conditions. Phase boundaries are calculated after @li2019.](assets/figs/madnn-training-pt-range.png){#fig:madnn-training-pt-range width=100%}

### Phase Solutions and Thermodynamic Data

Thermodynamic data for computing MADNN training datasets are based on end-member thermodynamic properties from @holland2018, with updates from @tomlinson2021 and @holland2022. The database (tc-ds634.txt from [hpxeosandthermocalc.org](https://hpxeosandthermocalc.org)) is specifically formulated for calculating phase relations for a wide array of igneous rocks and melt compositions. @holland2018 itself is an extension of the foundational database from @holland2011, which is calibrated up to 300 GPa and 2000 ˚C. Thus, the dataset tc-ds634.txt is appropriate for building MADNN training datasets for the entire upper mantle ([@fig:madnn-training-pt-range]).

All GFEM calculations are computed with equations of state for pure phases: quartz, coesite, stishovite, kyanite, corundum, and rutile, and solution phases: feldspar, spinel, garnet, clinopyroxene, orthopyroxene, olivine, ilmenite, and silicate melt. The same solution models from @holland2018 are used for MAGEMin and Perple_X calculations. The one notable exception is ternary feldspar models, which differ for MAGEMin [after @holland2022] and Perple_X [after @fuhrman1988].

More importantly, Perple_X includes solution models for wadsleyite, ringwoodite, wuestite, perovskite, ferropericlase, and HP clinopyroxene that are not included in the current release of MAGEMin (version 1.3.2, June 6, 2023). To make MAGEMin calculations approximately identical to Perple_X, the pure end-member phases for wadsleyite, ringwoodite, wuestite, perovskite, ferropericlase, and HP clinopyroxene are used without solution models ([@fig:benchmark-PUM-solutions]). This issue will be addressed in future releases of MAGEMin software, which will include solution models for deep mantle phases (Riel, [personal communications](https://github.com/ComputationalThermodynamics/MAGEMin/issues/61), July 11, 2023).

![Mineral assemblage diagrams (MADs) showing stable phases predicted by GFEM programs (a) MAGEMin and (b) Perple_X for a Primitive Upper Mantle (PUM) bulk composition from @sun1989. The colors represent integer indices of different mineral assemblages. Straight lines are the bounding geotherms for subsetting MADNN training data from @fig:madnn-training-pt-range.](assets/figs/benchmark/PUM-128x128/comp-PUM-128x128-StableSolutions.png){#fig:benchmark-PUM-solutions}

# Results

## Comparing GFEM Programs

![PT diagrams showing the density (greyscale in g/cm$^3$) predicted by GFEM programs (a) MAGEMin and (b) Perple_X for a Primitive Upper Mantle (PUM) bulk composition from @sun1989. (c) Normalized differences and (d) maximum difference gradients at each pixel highlight discrepancies between the GFEM programs. For example, notice the inconsistent phase transitions above 10 GPa.](assets/figs/benchmark/PUM-128x128/comp-PUM-128x128-DensityOfFullAssemblage.png){#fig:benchmark-PUM-density}

![PT diagrams showing the volume fraction of melt (greyscale) predicted by GFEM programs (a) MAGEMin and (b) Perple_X for a Primitive Upper Mantle (PUM) bulk composition from @sun1989. (c) Normalized differences and (d) maximum difference gradients at each pixel highlight discrepancies between the GFEM programs. For example, notice the inconsistent phase transitions above 10 GPa.](assets/figs/benchmark/PUM-128x128/comp-PUM-128x128-LiquidFraction.png){#fig:benchmark-PUM-liquid}

![PT diagrams showing the pressure wave velocity (greyscale, in km/s) predicted by GFEM programs (a) MAGEMin and (b) Perple_X for a Primitive Upper Mantle (PUM) bulk composition from @sun1989. (c) Normalized differences and (d) maximum difference gradients at each pixel highlight discrepancies between the GFEM programs. For example, notice the inconsistent phase transitions above 10 GPa.](assets/figs/benchmark/PUM-128x128/comp-PUM-128x128-Vp.png){#fig:benchmark-PUM-vp}

![PT diagrams showing the pressure wave velocity (greyscale, in km/s) predicted by GFEM programs (a) MAGEMin and (b) Perple_X for a Primitive Upper Mantle (PUM) bulk composition from @sun1989. (c) Normalized differences and (d) maximum difference gradients at each pixel highlight discrepancies between the GFEM programs. For example, notice the inconsistent phase transitions above 10 GPa.](assets/figs/benchmark/PUM-128x128/comp-PUM-128x128-Vs.png){#fig:benchmark-PUM-vs}

\clearpage

# References

<div id="refs"></div>

\clearpage

# Appendix

## GFEM Benchmarking

Benchmarking GFEM programs was a necessary first step for estimating the time required for building MADNN training datasets and quantifying the efficiency of incumbent GFEM programs (@fig:benchmark-times), which our MADNNs will need to beat to be considered an advancement beyond the status-quo. Estimated bulk compositions for primitive and depleted mantle-derived rocks ([@tbl:benchmark-comps]) were used for benchmarking MAGEMin and Perple_X. [@tbl:benchmark-times-table] shows the computation times with respect to various PT grid resolutions (8x8, 16x16, 32x32, 64x64, 128x128). All computations were made on a Macbook Pro (2022; M2 chip) with macOS 13.4 and Python 3.11.4. Note that MAGEMin was ran on 6 CPU cores in parallel, while Perple_X does not have parallel capabilities.

| Name   | P (GPa)   | T (K)    |   Grid Size |   MAGEMin (s) |   Perple_X (s) |   PPX/MGM |
|:-------|----------:|---------:|------------:|--------------:|---------------:|----------:|
| DMM    | 1-28      | 773-2273 |          64 |           2.6 |            9.3 |       3.6 |
| NMORB  | 1-28      | 773-2273 |          64 |           3.6 |           12.6 |       3.5 |
| PUM    | 1-28      | 773-2273 |          64 |           1.9 |           10.1 |       5.3 |
| RE46   | 1-28      | 773-2273 |          64 |           8.2 |           13.8 |       1.7 |
| DMM    | 1-28      | 773-2273 |         256 |           5.2 |           31.7 |       6.1 |
| NMORB  | 1-28      | 773-2273 |         256 |           7.7 |           43.4 |       5.6 |
| PUM    | 1-28      | 773-2273 |         256 |           5.8 |           33.4 |       5.8 |
| RE46   | 1-28      | 773-2273 |         256 |          34.4 |           47.1 |       1.4 |
| DMM    | 1-28      | 773-2273 |        1024 |          21.2 |          107.6 |       5.1 |
| NMORB  | 1-28      | 773-2273 |        1024 |          25.3 |          160.8 |       6.4 |
| PUM    | 1-28      | 773-2273 |        1024 |          19.8 |          116.1 |       5.9 |
| RE46   | 1-28      | 773-2273 |        1024 |         117.8 |          171.5 |       1.5 |
| DMM    | 1-28      | 773-2273 |        4096 |          70.5 |          397.2 |       5.6 |
| NMORB  | 1-28      | 773-2273 |        4096 |          94.6 |          580.0 |       6.1 |
| PUM    | 1-28      | 773-2273 |        4096 |          68.4 |          429.5 |       6.3 |
| RE46   | 1-28      | 773-2273 |        4096 |         459.0 |          601.6 |       1.3 |
| DMM    | 1-28      | 773-2273 |       16384 |         260.3 |         1438.3 |       5.5 |
| NMORB  | 1-28      | 773-2273 |       16384 |         364.9 |         2030.3 |       5.6 |
| PUM    | 1-28      | 773-2273 |       16384 |         244.5 |         1587.5 |       6.5 |
| RE46   | 1-28      | 773-2273 |       16384 |        1826.0 |         2210.3 |       1.2 |

: Computation times for various bulk mantle compositions. {#tbl:benchmark-times-table}

## Comparing GFEM Programs

![Mineral assemblage diagrams (MADs) showing stable phases predicted by GFEM programs (a) MAGEMin and (b) Perple_X for a Depleted MORB Mantle (DMM) bulk composition from @workman2005. Greyscale represents integer indices of different mineral assemblages.](assets/figs/benchmark/DMM-128x128/comp-DMM-128x128-StableSolutions.png){#fig:benchmark-DMM-solutions}

![PT diagrams showing the density (greyscale in g/cm$^3$) predicted by GFEM programs (a) MAGEMin and (b) Perple_X for a Depleted MORB Mantle (DMM) bulk composition from @workman2005. (c) Normalized differences and (d) maximum difference gradients at each pixel highlight discrepancies between the GFEM programs. For example, notice the inconsistent phase transitions above 10 GPa.](assets/figs/benchmark/DMM-128x128/comp-DMM-128x128-DensityOfFullAssemblage.png){#fig:benchmark-DMM-density}

![PT diagrams showing the volume fraction of melt (greyscale) predicted by GFEM programs (a) MAGEMin and (b) Perple_X for a Depleted MORB Mantle (DMM) bulk composition from @workman2005. (c) Normalized differences and (d) maximum difference gradients at each pixel highlight discrepancies between the GFEM programs. For example, notice the inconsistent phase transitions above 10 GPa.](assets/figs/benchmark/DMM-128x128/comp-DMM-128x128-LiquidFraction.png){#fig:benchmark-DMM-liquid}

![PT diagrams showing the pressure wave velocity (greyscale, in km/s) predicted by GFEM programs (a) MAGEMin and (b) Perple_X for a Depleted MORB Mantle (DMM) bulk composition from @workman2005. (c) Normalized differences and (d) maximum difference gradients at each pixel highlight discrepancies between the GFEM programs. For example, notice the inconsistent phase transitions above 10 GPa.](assets/figs/benchmark/DMM-128x128/comp-DMM-128x128-Vp.png){#fig:benchmark-DMM-vp}

![PT diagrams showing the pressure wave velocity (greyscale, in km/s) predicted by GFEM programs (a) MAGEMin and (b) Perple_X for a Depleted MORB Mantle (DMM) bulk composition from @workman2005. (c) Normalized differences and (d) maximum difference gradients at each pixel highlight discrepancies between the GFEM programs. For example, notice the inconsistent phase transitions above 10 GPa.](assets/figs/benchmark/DMM-128x128/comp-DMM-128x128-Vs.png){#fig:benchmark-DMM-vs}

![Mineral assemblage diagrams (MADs) showing stable phases predicted by GFEM programs (a) MAGEMin and (b) Perple_X for a Normal MORB (NMORB) bulk composition from @gale2013. Greyscale represents integer indices of different mineral assemblages.](assets/figs/benchmark/NMORB-128x128/comp-NMORB-128x128-StableSolutions.png){#fig:benchmark-NMORB-solutions}

![PT diagrams showing the density (greyscale in g/cm$^3$) predicted by GFEM programs (a) MAGEMin and (b) Perple_X for a Normal MORB (NMORB) bulk composition from @gale2013. (c) Normalized differences and (d) maximum difference gradients at each pixel highlight discrepancies between the GFEM programs. For example, notice the inconsistent phase transitions above 10 GPa.](assets/figs/benchmark/NMORB-128x128/comp-NMORB-128x128-DensityOfFullAssemblage.png){#fig:benchmark-NMORB-density}

![PT diagrams showing the volume fraction of melt (greyscale) predicted by GFEM programs (a) MAGEMin and (b) Perple_X for a Normal MORB (NMORB) bulk composition from @gale2013. (c) Normalized differences and (d) maximum difference gradients at each pixel highlight discrepancies between the GFEM programs. For example, notice the inconsistent phase transitions above 10 GPa.](assets/figs/benchmark/NMORB-128x128/comp-NMORB-128x128-LiquidFraction.png){#fig:benchmark-NMORB-liquid}

![PT diagrams showing the pressure wave velocity (greyscale, in km/s) predicted by GFEM programs (a) MAGEMin and (b) Perple_X for a Normal MORB (NMORB) bulk composition from @gale2013. (c) Normalized differences and (d) maximum difference gradients at each pixel highlight discrepancies between the GFEM programs. For example, notice the inconsistent phase transitions above 10 GPa.](assets/figs/benchmark/NMORB-128x128/comp-NMORB-128x128-Vp.png){#fig:benchmark-NMORB-vp}

![PT diagrams showing the pressure wave velocity (greyscale, in km/s) predicted by GFEM programs (a) MAGEMin and (b) Perple_X for a Normal MORB (NMORB) bulk composition from @gale2013. (c) Normalized differences and (d) maximum difference gradients at each pixel highlight discrepancies between the GFEM programs. For example, notice the inconsistent phase transitions above 10 GPa.](assets/figs/benchmark/NMORB-128x128/comp-NMORB-128x128-Vs.png){#fig:benchmark-NMORB-vs}

![Mineral assemblage diagrams (MADs) showing stable phases predicted by GFEM programs (a) MAGEMin and (b) Perple_X for an Icelandic basalt sample (RE46) bulk composition from @yang1996. Greyscale represents integer indices of different mineral assemblages.](assets/figs/benchmark/RE46-128x128/comp-RE46-128x128-StableSolutions.png){#fig:benchmark-RE46-solutions}

![PT diagrams showing the density (greyscale in g/cm$^3$) predicted by GFEM programs (a) MAGEMin and (b) Perple_X for an Icelandic basalt sample (RE46) bulk composition from @yang1996. (c) Normalized differences and (d) maximum difference gradients at each pixel highlight discrepancies between the GFEM programs. For example, notice the inconsistent phase transitions above 10 GPa.](assets/figs/benchmark/RE46-128x128/comp-RE46-128x128-DensityOfFullAssemblage.png){#fig:benchmark-RE46-density}

![PT diagrams showing the volume fraction of melt (greyscale) predicted by GFEM programs (a) MAGEMin and (b) Perple_X for an Icelandic basalt sample (RE46) bulk composition from @yang1996. (c) Normalized differences and (d) maximum difference gradients at each pixel highlight discrepancies between the GFEM programs. For example, notice the inconsistent phase transitions above 10 GPa.](assets/figs/benchmark/RE46-128x128/comp-RE46-128x128-LiquidFraction.png){#fig:benchmark-RE46-liquid}

![PT diagrams showing the pressure wave velocity (greyscale, in km/s) predicted by GFEM programs (a) MAGEMin and (b) Perple_X for an Icelandic basalt sample (RE46) bulk composition from @yang1996. (c) Normalized differences and (d) maximum difference gradients at each pixel highlight discrepancies between the GFEM programs. For example, notice the inconsistent phase transitions above 10 GPa.](assets/figs/benchmark/RE46-128x128/comp-RE46-128x128-Vp.png){#fig:benchmark-RE46-vp}

![PT diagrams showing the pressure wave velocity (greyscale, in km/s) predicted by GFEM programs (a) MAGEMin and (b) Perple_X for an Icelandic basalt sample (RE46) bulk composition from @yang1996. (c) Normalized differences and (d) maximum difference gradients at each pixel highlight discrepancies between the GFEM programs. For example, notice the inconsistent phase transitions above 10 GPa.](assets/figs/benchmark/RE46-128x128/comp-RE46-128x128-Vs.png){#fig:benchmark-RE46-vs}
