# Introduction

The dominant mineral phases in Earth's upper mantle are olivine, ringwoodite, bridgmanite, and ferropericlase [@ringwood1975; @ringwood1991], comprising up to 60–90% of the mantle's volume [e.g., @stixrude2012]. These Mg-Fe-rich phases form by a series of discrete reactions (@eq:mantle-transitions) that define mantle transition zones (MTZs) near 410 km and 660 km depths beneath Earth's surface. MTZs are characterized by relatively sharp boundaries with contrasting physical properties [e.g, density and elasticity, @dziewonski1981; @ita1992] that strongly impact mantle convection, melting, and plate tectonics [@fukao2001; @ringwood1991; @schubert1975; @kuritani2019; @karato2001; @wang2015; @jenkins2016].

\begin{align}
	\text{olivine} \xrightarrow{\text{410 km}} \text{ringwoodite} &\xrightarrow{\text{660 km}} \text{bridgmanite} + \text{ferropericlase} \label{eq:mantle-transitions} \\
	\text{(Mg,Fe)}_{2}\text{SiO}_{4} \xrightarrow{\text{410 km}} \text{(Mg,Fe)}_{2}\text{SiO}_{4} &\xrightarrow{\text{660 km}} \text{(Mg,Fe)}\text{SiO}_{3} + \text{(Mg,Fe)}\text{O} \nonumber
\end{align}

Although the physio-chemical nature of MTZs remains under vigorous investigation [@goes2022; @pearson2014; @yoshino2008; @waszek2021; @kiseeva2018; @fei2017; @zhou2022], modelling mantle convection within and across the MTZs is possible with numerical geodynamic simulations that implement pressure-temperature (PT)-dependent phase changes (e.g., @eq:mantle-transitions). This approach has generated many important hypotheses implicating MTZs as critical features impacting tectonic plate motions and water cycling in the deep Earth [e.g., @agrusta2017; @li2019; @torii2007; @yang2020]. However, the tendency to assume fixed mantle compositions---i.e., neglecting chemical fractionation from fluid-rock interactions and partial melting---limit such numerical experiments to a rough zeroth-order approximation of true mantle flow.

More comprehensive numerical models of mantle convection would account for chemical fractionation by implementing PT-composition-time (PTXt)-dependent phase changes. This is currently intractable, however, because Gibbs Free Energy Minimization (GFEMs) programs [e.g., @connolly2009; @riel2022] used to calculate PTX-dependent phase relations remain too slow to apply at each node, at each timestep, in a geodynamic simulation. While recent parallelized GFEM programs have increased efficiency dramatically [10$^4$--10$^2$ ms at a fixed PTX, @riel2022], computing phase relations iteratively during geodynamic simulations requires GFEM efficiency on the order of $\leq$ 10$^1$--10$^{-1}$ ms to be feasible. A rate of improvement from 10$^4$--10$^2$ to 10$^1$--10$^{-1}$ ms seems unlikely within the current GFEM paradigm, however, and applying parallelization across many CPU/GPU cores can be inaccessible.

This study demonstrates an alternative approach to inferring phase relations and rock properties using pre-trained machine learning models (referred to as RocMLs). We begin by detailing the size (PT range), scope (compositional range), and methods for building RocML training datasets. We then train and evaluate performance metrics for each RocML by k-fold cross-validation---demonstrating up to 10$^4$ faster RocML performance vs. GFEM programs. Our results imply that real-time inference of PTXt-dependent phase changes in geodynamic simulations is feasible with RocMLs. Finally, we discuss the validity and limitations of RocMLs through comparisons with empirically-derived geophysical datasets, and how implementing RocMLs in geodynamic simulations can enable new models of global plate tectonics, deep water cycling, and mantle melting at MTZs.

# Methods {#sec:methods}

The following sections describe our methods for building RocML training datasets and evaluating RocML performance on unknown data. Our objectives are threefold. First, design the size (PT range) and scope (compositional range) of RocML training data to ensure widespread applicability of RocMLs to geodynamic problems within the upper mantle (@sec:design-training-data). Second, compute minimal RocML training data and directly compare the GFEM programs MAGEMin and Perple_X (@sec:build-training-data). Third, train various RocMLs and determine the best models by k-fold cross-validation (@sec:training-RocMLs). Comparisons among MAGEMin, Perple_X, and RocMLs results are then detailed in @sec:results.

## Designing RocML Training Datasets {#sec:design-training-data}

### PT Conditions

High-pressure experiments constrain the reaction $\text{olivine} \xrightarrow{\text{410 km}} \text{ringwoodite}$ between 14.0 ± 1.0 GPa and 1600 ± 400 K with Clapeyron slopes between 2.4x10$^{-3}$ ± 1.4x10$^{-3}$ GPa/K  [@akaogi1989; @katsura1989; @morishima1994; @li2019]. Likewise, the reaction $\text{ringwoodite} \xrightarrow{\text{660 km}} \text{bridgmanite} + \text{ferropericlase}$ is constrained between 24.0 ± 1.5 GPa and 1600 ± 400 K with negative Clapeyron slopes between -2.0x10$^{-3}$ ± 1.6x10$^{-3}$ GPa/K  [@akaogi2007; @bina1994; @litasov2005; @katsura2003; @ito1990; @ito1982; @ito1989a; @ito1989b; @hirose2002; @ishii2018]. We therefore compute RocML training data within a rectangular PT region bound between 1.0–28.0 GPa and 773–2273 K to encompass expected conditions for the entire upper mantle---from the base of the Moho at 35 km to the 660 km MTZ ([@fig:training-dataset-design]).

![PT diagram showing the ranges of PT conditions for RocML training data (grey region) and the upper mantle (blue hatched region). Geotherm 1 (solid black line) and geotherm 2 (dashed black line) assume mantle potential temperatures of 273 K and 1773 K with 1 K/km and 0.5 K/km thermal gradients, respectively. Phase boundaries for the 410 and 660 km MTZs (colored lines) are calculated after @li2019.](assets/figs/training-dataset-design.png){#fig:training-dataset-design}

@fig:training-dataset-design shows that our training dataset includes PT conditions that are not expected to exist in the upper mantle, nor typically modelled during geodynamic simulations [e.g., very cold conditions below thermal gradients of $\leq$ 5 K/km, @maruyama1996; @syracuse2010]. Thus, training RocMLs on the entire dataset can be considered impractical with respect to efficiency (unnecessarily large amount of training data) and accuracy (outside the bounds of calibrated thermodynamic data). For example, phase relations can be spurious and noisy at very low-temperature and high-pressure (e.g., at 20 GPa and 1000 K) and include anomalously-high degrees of partial melt at very low-pressure and high-temperature (e.g., at 5 GPa and 2000 K).

On the other hand, a regular rectangular training dataset design is more straightforward to compute,  validate, and benchmark. Moreover, size inefficiencies and inaccuracies at extreme PT conditions will only affect RocMLs at training time, not during inference. For example, unknown PTs input into RocMLs for inference (e.g., from a geodynamic simulation) should be within the PT range where RocML training data are most accurate (blue hatched region in @fig:training-dataset-design). Thus, RocML predictions should be no less accurate than MAGEMin or Perple_X models---assuming RocMLs fit the training data well. Thus, we conveniently train and validate RocMLs on the entire rectangular range of training data shown in @fig:training-dataset-design.

### Solution Phase Models and Thermodynamic Data {#sec:thermodynamic-data}

Thermodynamic data for computing RocML training datasets are based on end-member thermodynamic properties from @holland2018, with updates from @tomlinson2021 and @holland2022. The database (tc-ds634.txt from [hpxeosandthermocalc.org](https://hpxeosandthermocalc.org)) is specifically formulated for calculating phase relations for a wide array of igneous rocks and melt compositions. @holland2018 itself is an extension of the foundational database from @holland2011, which is calibrated up to 300 GPa and 2000 ˚C. Thus, the dataset tc-ds634.txt is appropriate for building RocML training datasets for the entire upper mantle ([@fig:training-dataset-design]).

All MAGEMin and Perple_X models are computed with equations of state for pure phases: quartz, coesite, stishovite, kyanite, corundum, and rutile, and solution phases: feldspar, spinel, garnet, clinopyroxene, orthopyroxene, olivine, ilmenite, and silicate melt. The same solution models from @holland2018 are applied to MAGEMin and Perple_X. The one notable exception is ternary feldspar models, which differ for MAGEMin [after @holland2022] and Perple_X [after @fuhrman1988].

More importantly, Perple_X includes solution models for wadsleyite, ringwoodite, wuestite, perovskite, ferropericlase, and high-pressure clinopyroxene that are not included in the current release of MAGEMin (version 1.3.2, June 6, 2023). To make Perple_X models approximately identical to MAGEMin, the pure end-member phases for wadsleyite, ringwoodite, wuestite, perovskite, ferropericlase, and high-pressure clinopyroxene are used without solution models. This issue will be addressed in future releases of MAGEMin software, which will include solution models for deep mantle phases (Riel, [personal communications](https://github.com/ComputationalThermodynamics/MAGEMin/issues/61), July 11, 2023).

### Bulk Chemical Compositions

Existing estimates for the bulk chemical composition of the upper mantle are based on analyses of high-pressure-high-temperature melting experiments and mantle-derived xenoliths, kimberlites, and basalts [e.g., @allegre1984; @green1979; @ringwood1962; @jagoutz1979; @sun1982; @ringwood1991; @palme2003; @stracke2021]. [@tbl:benchmark-samples] provides some well-referenced examples, including hypothetical mantle compositions with varying degrees of differentiation by partial melting [Primitive Upper Mantle: PUM, and Depleted MORB Mantle: DMM, @sun1989; @workman2005], as well as real and hypothetical products of mantle melting [Iclandic Basalt: RE46, and Normal MORB: NMORB, @gale2013; @yang1996]. RocML training data are currently fixed at PUM, which represents the average bulk (pyrolitic) composition of the upper mantle. Eventually, training data will include all compositions in @tbl:benchmark-samples to approximate a more complete range of expected mantle compositions.

{{ benchmark-samples.md }}

## Computing RocML Training Datasets {#sec:build-training-data}

### Gibbs Free Energy Minimization

We use the GFEM programs [MAGEMin](https://github.com/ComputationalThermodynamics/MAGEMin) and [Perple_X](https://github.com/ondrolexa/Perple_X) [@riel2022; @connolly2009] to compute RocML training data for a broad range of upper mantle PT conditions. The two programs use slightly different computational approaches to minimize the total GFE of a multicomponent multiphase thermodynamic system. At a fixed PT, the GFE for such a system is defined by the following equation [@gibbs1878; @spear1993]:

\begin{equation}
	\text{GFE} = \sum_{\lambda=1}^{\Lambda} p_{\lambda} \sum_{n=1}^{N} p_n \mu_n + \sum_{\omega=1}^{\Omega} p_{\omega} \mu_{\omega} \label{eq:gfe}
\end{equation}

\noindent where $\Lambda$ is the number solution phases, $N$ is the number of end-member compounds that mix to form solution phases, and $\Omega$ is the number of pure (stoichiometric) phases. Thus, @eq:gfe states that the total GFE of a thermodynamic system (at a fixed PT) is the weighted sum of the molar fractions $p_{\lambda}$ of solution phases $\lambda$ and the molar fractions $p_n$ and chemical potentials $\mu_n$ of end-member compounds $n$ that mix to form solution phases $\lambda$, plus the weighted sum of the molar fractions $p_{\omega}$ and chemical potentials $\mu_{\omega}$ of pure phases $\omega$.

For pure phases, the chemical potential is a constant [@spear1993]:

\begin{equation}
	\mu_{\omega} = \text{GFE}_{\omega}^{\text{standard}}
\end{equation}

\noindent where $\text{GFE}_{\omega}^{\text{standard}}$ is the Gibbs Free Energy of formation at standard PT (1 bar, 273 K). For a solution phase, however, the chemical potential is described by a non-ideal mixture of end-member compounds:

\begin{equation}
	\mu_n = \text{GFE}_n^{\text{standard}} + RTln(a_n^{\text{ideal}}) + \text{GFE}_n^{\text{excess}}
\end{equation}

\noindent where $\mu_n$ is the chemical potential of end-member compound $n$, $R$ is the gas constant, $T$ is temperature, $a_n^{\text{ideal}}$ is the activity of an end-member compound defined by ideal mixing: $a_n^{\text{ideal}} = x_n$, where $x_n$ is the molar fraction of the end-member compound $n$. The $\text{GFE}_n^{\text{excess}}$ term models non-ideal behavior by defining symmetric (site-independent) and asymmetric (site-dependent) mixing of end-member compounds on different crystallographic sites for a particular solution phase $\lambda$ [mixing-on-site formalism, @holland2003; @powell1993].

Additional compositional constraints are imposed on @eq:gfe by the Gibbs-Duhem equation [@spear1993]:

\begin{equation}
	\mu_{(\omega,n)} = \sum_{c=1}^{C} \mu_c a_{(\omega,n) c} \label{eq:gibbs-duhem}
\end{equation}

\noindent where $C$ is the number of chemical components (oxides) considered in the thermodynamic system. The Gibbs-Duhem equation states that the total chemical potential of a pure phase or end-member compound $\mu_{(\omega,n)}$ is equal to the weighted sum of the chemical potentials $\mu_c$ and activities $a_{(\omega,n) c}$ of the chemical components dissolved in the pure phase or end-member compound.

@eq:gibbs-duhem implies that the total GFE of the thermodynamic system is dependent on its bulk chemical composition. Consequently, for a fixed bulk composition at equilibrium, the stable mineral assemblage must satisfy the Gibbs phase rule:

\begin{equation}
	F = C - \Phi + 2
\end{equation}

\noindent where $F$ is the number of degrees of freedom, $C$ is the number of chemical components (oxides), and $\Phi$ is the number of stable mineral phases in the rock. In this case, the "degrees of freedom" $F$ refers to the number of independent mineral phases that can vary their chemical potentials while the system remains in equilibrium.

Lastly, conservation of mass is maintained by equating the sum total of the chemical potentials in the system to the bulk rock composition:

\begin{equation}
	\sum_{c=1}^{C} \sum_{\lambda=1}^{\Lambda} p_{\lambda} \sum_{n=1}^{N} a_{nc} p_n + \sum_{c=1}^{C} \sum_{\omega=1}^{\Omega} p_{\omega} a_{\omega c} = \sum_{c=1}^{C} \text{bulk-rock}_c \label{eq:mass-balance}
\end{equation}

In principle, applying identical sets of solution phase models, thermodynamic data, and bulk compositions to the above equations will define identical GFE hyperplanes (i.e. define the same G-X "surfaces" in multidimensional space). This implies that GFEM programs should converge on identical phase relations irrespective of the minimization algorithm. We can therefore expect similar results between MAGEMin and Perple_X when comparing the two programs with the same bulk compositions and thermodynamic data.

### Comparing MAGEMin and Perple_X Models

In practice, small differences between MAGEMin and Perple_X models arise while minimizing @eq:gfe because different GFEM algorithms can converge on different local minima within the same region of the GFE hyperplane. Proximate local minima will produce the same stable phase assemblages with small differences in the chemical potentials among the phases---hence small differences in rock properties. However, MAGEMin and Perple_X models generally show density differences of $\leq$ 5% that indicate high degrees of correlation and internal consistency between the GFEM algorithms (@fig:image4-PUM-train-rho).

![PT-density diagrams showing density (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples). (c) PT-error diagram showing the percent difference between (a) and (b). (d) Density profiles comparing MAGEMin and Perple_X models with the empirically-derived Preliminary Reference Earth Model (PREM) from @dziewonski1981.](assets/figs/image4-PUM-train-rho.png){#fig:image4-PUM-train-rho}

In addition to the internal consistency observed between GFEM algorithms, density profiles extracted from MAGEMin and Perple_X models generally indicate external consistency with empirical-derived density profiles from the Preliminary Reference Earth Model [PREM, @dziewonski1981]. In contrast to the PREM, however, MAGEMin and Perple_X models show two discrete density jumps between 410 and 660 km that are offset (relative to the PREM) towards lower densities at higher Ps ([@fig:image4-PUM-train-rho]d). The P offset is partially explained by the exclusion of certain solution models described in @sec:thermodynamic-data. For example, Perple_X models that include solid solutions for high-pressure mantle minerals are more consistent with the PREM between 410–660 km than equivalent models that only consider end-members. The density offset, on the other hand, can partially be explained by an estimated 3-4% uncertainty for Perple_X density profiles [approximately 0.1 g/cm$^3$, @connolly2016].

## Training RocMLs {#sec:training-rocmls}

Rock properties produced by MAGEMin and Perple_X were preprocessed before training using the following procedure. First, the square 2D rock properties datasets $Z = (z_{1,1}, z_{1,2}, \ldots, z_{1,W}, z_{2,1}, z_{2,2}, \ldots, z_{2,W}, z_{3,1}, z_{3,2}, \ldots, z_{W,W})$ are separated into a flat 2D feature array of PTs $X = (x_{1,1}, x_{1,2}, x_{2,1}, x_{2,2}, \ldots, x_{V,1}, x_{V,2})$ and 1D target array of a single rock property $y = (y_1, y_2, \ldots, y_V)$, where $V = W^2$ is the total number of training data points (i.e. the PT grid size). Next, the feature array $X$ and target array $y$ are standardized by shifting their values by their means and dividing by their standard deviations, respectively:

\begin{align}
	X_{V,1}^{\text{standardized}} &= \frac{X_{V,1} - \mu_{X_{V,1}}}{\sigma_{X_{V,1}}} \label{eq:standard-scaler} \\
	X_{V,2}^{\text{standardized}} &= \frac{X_{V,2} - \mu_{X_{V,2}}}{\sigma_{X_{V,2}}} \nonumber \\
	y_{V}^{\text{standardized}} &= \frac{y_{V} - \mu_{y_{V}}}{\sigma_{y_{V}}} \nonumber
\end{align}

\noindent where $X_{V,1}^{\text{standardized}}$ are the standardized Ps, $X_{V,2}^{\text{standardized}}$ are the standardized Ts, $y_{V}^{\text{standardized}}$ are the standardized rock property (density in this case), $\mu$ is the mean and $\sigma$ is the standard deviation of the appropriate array. This so-called "z-score normalization" is a necessary step before RocML training because the difference in magnitude of the feature values (1-28 GPa vs. 773-2273 K) results in poor performance for RocMLs that use distance-metrics for fitting.

The preprocessed training data were then fit with eight different non-linear ML models (@tbl:mlm-pro-con). Each model used standard parameters from the scikit-learn python library [@scikit2011] except for K Nearest, which was fit using a distance-weighting scheme instead of uniform weighting of the k-nearest data points. For Neural Networks, dense (fully-connected) layers were used with constant layer sizes equal to the total number of training data $V$ divided by 100 and rounded down to the nearest integer. The reader is referred to the scikit-learn [documentation](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning) on regression models for more specifics.

{{ mlm-pro-con.md }}

Finally, average performance metrics for each model were evaluated using a k-fold cross-validation technique as follows. First, the training data are partitioned into k = 30 non-overlapping folds of $V$/k samples, where $V$ is the total number of training data points (i.e., the PT grid size). To reduce the impact of inherent ordering in the data, the data are shuffled before splitting into folds. Cross-validation then proceeds with k iterations, where in each iteration, models are trained on samples from k-1 folds and performance is evaluated on the remaining fold. Performance metrics evaluated during each iteration included the correlation coefficient (R$^2$), root mean squared error (RMSE), and elapsed time during training and inference. After all iterations completed, means and standard deviations of performance metrics were computed to provide a measurement of each model's generalizability. Performance metrics are detailed in @sec:comparing-rocmls.

# Results {#sec:results}

## Comparing RocMLs with MAGEMin and Perple_X {#sec:comparing-rocmls}

### Qualitative Comparison

Model performance can be assessed qualitatively by comparing the density surfaces computed by MAGEMin and Perple_X with the density surfaces inferred by RocMLs (@fig:all-surf-PUM-rho).

![PT-density diagrams showing density (in g/cm$^3$) calculated by (a) MAGEMin and (i) Perple_X for a PUM bulk composition (@tbl:benchmark-samples). PT-density diagrams (b–h) and (j–p) show density (in g/cm$^3$) inferred by various RocMLs trained on MAGEMin and Perple_X, respectively.](assets/figs/all-surf-PUM-rho.png){#fig:all-surf-PUM-rho}

### Quantitative Comparison

{{ rocml-performance.md }}

![caption](assets/figs/rocml-performance-rho.png)

## Best RocML Selection

![Density profiles comparing (a) MAGEMin and (b) Perple_X models with the empirically-derived Preliminary Reference Earth Model (PREM) from @dziewonski1981. Performance metrics R$^2$ and RMSE show the correlations and accuracies of the Decision Tree RocML compared to MAGEMin and Perple_X. MAGEMin and Perple_X models assume a PUM bulk composition (@tbl:benchmark-samples).](assets/figs/prem-PUM-DT-rho.png)

# Discussion

## Validating RocML Density Models

\clearpage

# References

<div id="refs"></div>

\cleardoublepage

# Appendix

<!--
## GFEM Benchmarking

Benchmarking GFEM programs was a necessary first step for estimating the time required for building RocML training datasets and quantifying the efficiency of incumbent GFEM programs (@fig:benchmark-times), which our RocMLs will need to beat to be considered an advancement beyond the status-quo. Estimated bulk compositions for primitive and depleted mantle-derived rocks ([@tbl:benchmark-samples]) were used for benchmarking MAGEMin and Perple_X. [@tbl:benchmark-times-table] shows the computation times with respect to various PT grid resolutions (8x8, 16x16, 32x32, 64x64, 128x128). All computations were made on a Macbook Pro (2022; M2 chip) with macOS 13.4 and Python 3.11.4. Note that MAGEMin was ran on 6 CPU cores in parallel, while Perple_X does not have parallel capabilities.

{{ benchmark-efficiency.md }}
-->

\clearpage

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Decision Tree models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/surf-PUM-DT-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Random Forest models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/surf-PUM-RF-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by K Nearest models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/surf-PUM-KN-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Neural Network 1L models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/surf-PUM-NN1-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Neural Network 2L models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/surf-PUM-NN2-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Neural Network 3L models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/surf-PUM-NN3-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Decision Tree models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/image6-PUM-DT-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Random Forest models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/image6-PUM-RF-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by K Nearest models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/image6-PUM-KN-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Neural Network 1L models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/image6-PUM-NN1-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Neural Network 2L models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/image6-PUM-NN2-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Neural Network 3L models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/image6-PUM-NN3-rho.png)
