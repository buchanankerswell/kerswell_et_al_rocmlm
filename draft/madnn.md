# Introduction

## The Importance of Modeling Phase Changes at Mantle Transition Zones

The dominant mineral phases in Earth's mantle are olivine, ringwoodite, bridgmanite, and ferropericlase [@ringwood1975; @ringwood1991]. Depending on the assumed mantle composition, these four phases alone may comprise up to 60–90% of the mantle [@stixrude2012]. Work since the 1950's has established that a series of discrete phase changes near 410 km and 660 km depth ([@eq:mantle-transitions]) define critical mantle transition zones (MTZs) where physical properties of the mantle abruptly change [e.g, density, elasticity, etc., @ishii2018; @ita1992; @ito1989a]. Although the physio-chemical nature of MTZs remains under vigorous investigation [@goes2022; @pearson2014; @yoshino2008; @waszek2021; @kiseeva2018; @fei2017; @zhou2022], it is widely accepted that mantle convection, melting, and plate tectonics are strongly impacted by the phase changes associated with MTZs [@fukao2001; @ringwood1991; @schubert1975; @li2019; @kuritani2019; @karato2001; @wang2015; @yang2020]. Thus, numerical geodynamic models that fail to implement pressure-temperature-composition-time (PTXt)-dependent phase changes are merely first-order approximations of mantle flows with limited applicability to fundamental questions of mantle dynamics.

\begin{align}\label{eq:mantle-transitions}
	\text{olivine} \xrightarrow{\text{410 transition}} \text{ringwoodite} &\xrightarrow{\text{660 transition}} \text{bridgmanite} + \text{ferropericlase} \\
	\text{(Mg,Fe)}_{2}\text{SiO}_{4} \xrightarrow{\text{410 transition}} \text{(Mg,Fe)}_{2}\text{SiO}_{4} &\xrightarrow{\text{660 transition}} \text{(Mg,Fe)}\text{SiO}_{3} + \text{(Mg,Fe)}\text{O}
\end{align}

## Current Limitations

Even with simple parameterizations, full coupling of PTXt-dependent phase changes with large-scale geodynamic models is currently intractable because Gibbs free energy minimization (GFEMs) programs [e.g., @connolly2009; @riel2022] remain too slow to converge ($\geq$ 100 seconds, see [@fig:benchmark-comp-times] and [@tbl:benchmark-comp-times-table]) on stable solutions for mineral assemblage diagrams (MADs). For example, assuming a large-scale 2D geodynamic simulation has 900 x 300 nodes with a 100-second MAD compute time at each node, it would take 7500 hours to predict stable mineral assemblages for each single timestep. While recent parallelized GFEM programs [@riel2022] have increased efficiency dramatically (up to 30x improvement; [@fig:benchmark-comp-times] and [@tbl:benchmark-comp-times-table]), computing MADs iteratively during geodynamic simulations requires GFEM efficiency on the order of milliseconds to be feasible. A 10$^5$ rate of improvement seems unlikely within the current GFEM paradigm and applying parallelization across thousands of GPU cores is inaccessible in many cases.

![Benchmark results for GFEM programs MAGEMin [dashed lines with square, @riel2022] and Perple_X [solid lines with circles, @connolly2009]. Note that MAGEMin was ran in parallel on 6 CPU cores, while Perple_X has no parallel capabilities. In the best case for a 128 x 128 PT grid, MADs take 124 seconds to compute ([@tbl:benchmark-comp-times-table]).](assets/figs/benchmark-comp-times.png){#fig:benchmark-comp-times width=100%}

## A New Approach: MADNNs

To overcome the intractability of iteratively coupling MADs with large-scale geodynamic simulations, we propose a novel approach for inferring MADs using pre-trained neural networks (referred to here as MADNNs). We hypothesize that MADNNs can improve efficiency by up to 5 orders of magnitude versus incumbent GFEM programs for computing PTXt-dependent phase changes in the mantle. If true, real-time inference of PTXt-dependent phase changes at the individual node-scale in geodynamic simulations will be feasible---enabling new models of tectonic plate behavior and mantle plume interactions at MTZs. Moreover, the approach detailed here is generalizable and therefore has potential for adoption to models of other Earth systems. If false, we discover practical limitations and demonstrate the technical challenges of applying neural networks to petrological datasets---which is a critical step for guiding exploration of alternative approaches.

# Computing a Training Dataset for MADNNs

## PT Conditions and Bulk Compositions of the Upper Mantle

High-pressure experiments conducted since the 1990s demonstrate the reaction $\text{ringwoodite} \leftrightarrow \text{bridgmanite} + \text{ferropericlase}$ occurs at pressures coinciding with the 660 km discontinuity in the upper mantle at 23.4 ± 0.05 GPa and 1950 ± 50 K [@ito1989a; @ito1989b; @ishii2018]. Our training datasets are therefore computed across 1.0–24.0 GPa (10–240 kbar) and 773–2273 K (500–2000 ˚C). This PT range encompasses expected PTs at MTZ---practically ensuring applicability of the neural network in a large range of use-cases.

Many estimates for bulk compositions of the mantle exist [e.g., @allegre1984; @green1979; @ringwood1962; @jagoutz1979; @sun1982; @ringwood1991; @palme2003; @stracke2021]. [@tbl:benchmark-samples] provides some well-referenced examples. The compositions include hypothetical mantles with very low [Primitive Upper Mantle, PUM, @sun1989] and moderate [Depleted MORB Mantle, DMM, @workman2005] degrees of differentiation by melt extraction, as well as real [Iclandic Basalt, RE46, @yang1996] and hypothetical [Normal MORB, NMORB, @gale2013] products of mantle melting.


**!!! Add @ringwood1991 samples and justify the selection for training MADNNs !!!**

| Sample | SiO$_2$ | Al$_2$O$_3$ |   CaO |   MgO |  FeO | K$_2$O | Na$_2$O | TiO$_2$ |   O$_2$ | Cr$_2$O$_3$ |    Ref |
|:-------|--------:|------------:|------:|------:|-----:|-------:|--------:|--------:|--------:|------------:|-------:|
| PUM    |   44.90 |        4.44 |  3.54 | 37.71 | 8.03 |  0.029 |    0.36 |    0.20 |    0.01 |        0.38 | Sun89  |
| DMM    |   44.71 |        3.98 |  3.17 | 38.73 | 8.18 |  0.006 |    0.13 |    0.13 |    0.01 |        0.57 | Wrk05  |
| RE46   |   50.72 |        9.16 | 15.21 | 16.25 | 7.06 |  0.010 |    1.47 |    0.39 |    0.35 |        0.01 | Yng96  |
| NMORB  |   53.21 |        9.41 | 12.21 | 12.21 | 8.65 |  0.090 |    2.90 |    1.21 |    0.69 |        0.02 | Gle13  |

: Estimated bulk compositions (in wt. % oxides) for the mantle. {#tbl:benchmark-samples}

# References

<div id="refs"></div>

# Appendix

## Gibbs Free Energy Minimization Benchmarking Results {#sec:appendix-benchmark-comp-times}

Estimated bulk compositions for primitive and depleted mantle rocks ([@tbl:benchmark-samples]) were used for benchmarking GFEM programs [MAGEMin](https://github.com/ComputationalThermodynamics/MAGEMin) and [Perple_X](https://github.com/ondrolexa/Perple_X) [@riel2022; @connolly2009]. [@tbl:benchmark-comp-times-table] shows the computation times with respect to various PT grid resolutions (8x8, 16x16, 32x32, 64x64, 128x128). All computations were made on a Macbook Pro (2022; M2 chip) with macOS 13.4 and Python 3.11.4. Note that MAGEMin was ran on 6 CPU cores in parallel, while Perple_X does not have parallel capabilities.

| Sample | Grid Size |  MAGEMin (s) | Perple_X (s)  | PPX/MGM |
|:-------|----------:|-------------:|--------------:|--------:|
| DMM    |        64 |          1.9 |          17.1 |     9.0 |
| NMORB  |        64 |          1.3 |          21.2 |    16.3 |
| PUM    |        64 |          0.7 |          18.8 |    26.8 |
| RE46   |        64 |          5.0 |          22.1 |     4.4 |
| DMM    |       256 |          3.4 |          63.0 |    18.5 |
| NMORB  |       256 |          3.9 |          79.1 |    20.2 |
| PUM    |       256 |          2.0 |          65.3 |    32.6 |
| RE46   |       256 |         15.0 |          82.2 |     5.4 |
| DMM    |      1024 |          9.1 |          229.8|    25.2 |
| NMORB  |      1024 |         13.8 |          264.7|    19.1 |
| PUM    |      1024 |          7.7 |          245.2|    31.8 |
| RE46   |      1024 |         55.4 |          283.4|     5.1 |
| DMM    |      4096 |         33.0 |         847.4 |    25.7 |
| NMORB  |      4096 |         54.9 |         995.9 |    18.1 |
| PUM    |      4096 |         29.7 |         922.1 |    31.0 |
| RE46   |      4096 |        201.8 |        1036.1 |     5.1 |
| DMM    |     16384 |        128.9 |        3381.2 |    26.2 |
| NMORB  |     16384 |        211.7 |        3827.6 |    18.1 |
| PUM    |     16384 |        115.4 |        3672.7 |    31.8 |
| RE46   |     16384 |        774.4 |        4102.7 |     5.3 |

: Computation times for various bulk mantle compositions. {#tbl:benchmark-comp-times-table}

![caption](assets/figs/benchmark/PUM-128x128/comp-PUM-128x128-StableSolutions.png){#fig:benchmark-PUM-solutions}

![caption](assets/figs/benchmark/DMM-128x128/comp-DMM-128x128-StableSolutions.png){#fig:benchmark-DMM-solutions}

![caption](assets/figs/benchmark/NMORB-128x128/comp-NMORB-128x128-StableSolutions.png){#fig:benchmark-NMORB-solutions}

![caption](assets/figs/benchmark/RE46-128x128/comp-RE46-128x128-StableSolutions.png){#fig:benchmark-RE46-solutions}

![caption](assets/figs/benchmark/PUM-128x128/comp-PUM-128x128-StableVariance.png){#fig:benchmark-PUM-variance}

![caption](assets/figs/benchmark/DMM-128x128/comp-DMM-128x128-StableVariance.png){#fig:benchmark-DMM-variance}

![caption](assets/figs/benchmark/NMORB-128x128/comp-NMORB-128x128-StableVariance.png){#fig:benchmark-NMORB-variance}

![caption](assets/figs/benchmark/RE46-128x128/comp-RE46-128x128-StableVariance.png){#fig:benchmark-RE46-variance}

![caption](assets/figs/benchmark/PUM-128x128/comp-PUM-128x128-DensityOfFullAssemblage.png){#fig:benchmark-PUM-density}

![caption](assets/figs/benchmark/DMM-128x128/comp-DMM-128x128-DensityOfFullAssemblage.png){#fig:benchmark-DMM-density}

![caption](assets/figs/benchmark/NMORB-128x128/comp-NMORB-128x128-DensityOfFullAssemblage.png){#fig:benchmark-NMORB-density}

![caption](assets/figs/benchmark/RE46-128x128/comp-RE46-128x128-DensityOfFullAssemblage.png){#fig:benchmark-RE46-density}

![caption](assets/figs/benchmark/PUM-128x128/comp-PUM-128x128-LiquidFraction.png){#fig:benchmark-PUM-liquid}

![caption](assets/figs/benchmark/DMM-128x128/comp-DMM-128x128-LiquidFraction.png){#fig:benchmark-DMM-liquid}

![caption](assets/figs/benchmark/NMORB-128x128/comp-NMORB-128x128-LiquidFraction.png){#fig:benchmark-NMORB-liquid}

![caption](assets/figs/benchmark/RE46-128x128/comp-RE46-128x128-LiquidFraction.png){#fig:benchmark-RE46-liquid}

![caption](assets/figs/benchmark/PUM-128x128/comp-PUM-128x128-Vp.png){#fig:benchmark-PUM-vp}

![caption](assets/figs/benchmark/DMM-128x128/comp-DMM-128x128-Vp.png){#fig:benchmark-DMM-vp}

![caption](assets/figs/benchmark/NMORB-128x128/comp-NMORB-128x128-Vp.png){#fig:benchmark-NMORB-vp}

![caption](assets/figs/benchmark/RE46-128x128/comp-RE46-128x128-Vp.png){#fig:benchmark-RE46-vp}
