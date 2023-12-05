# Introduction

The dominant mineral phases in Earth's upper mantle are olivine, ringwoodite, bridgmanite, and ferropericlase [@ringwood1975; @ringwood1991], comprising up to 60–90% of the mantle's volume [e.g., @stixrude2012]. These Mg-Fe-rich phases form by a series of discrete reactions (@eq:mantle-transitions) that define mantle transition zones (MTZs) near 410 km and 660 km depths beneath Earth's surface. MTZs are characterized by relatively sharp boundaries with contrasting physical properties [e.g, density and elasticity, @dziewonski1981; @ita1992] that strongly impact mantle convection, melting, and plate tectonics [@fukao2001; @ringwood1991; @schubert1975; @kuritani2019; @karato2001; @wang2015; @jenkins2016].

\begin{align}
	\text{olivine} \xrightarrow{\text{410 km}} \text{ringwoodite} &\xrightarrow{\text{660 km}} \text{bridgmanite} + \text{ferropericlase} \label{eq:mantle-transitions} \\
	\text{(Mg,Fe)}_{2}\text{SiO}_{4} \xrightarrow{\text{410 km}} \text{(Mg,Fe)}_{2}\text{SiO}_{4} &\xrightarrow{\text{660 km}} \text{(Mg,Fe)}\text{SiO}_{3} + \text{(Mg,Fe)}\text{O} \nonumber
\end{align}

Although the physio-chemical nature of MTZs remains under vigorous investigation [@goes2022; @pearson2014; @yoshino2008; @waszek2021; @kiseeva2018; @fei2017; @zhou2022], modeling mantle convection within and across the MTZs is possible with numerical geodynamic simulations that implement pressure-temperature (PT)-dependent phase changes (e.g., @eq:mantle-transitions). This approach has generated many important hypotheses implicating MTZs as critical features impacting tectonic plate motions and water cycling in the deep Earth [e.g., @agrusta2017; @li2019; @torii2007; @yang2020]. However, the tendency to assume fixed mantle compositions---i.e., neglecting chemical fractionation from fluid-rock interactions and partial melting---limit such numerical experiments to a rough zeroth-order approximation of true mantle flow.

More comprehensive numerical models of mantle convection would account for chemical fractionation by implementing PT-composition-time (PTXt)-dependent phase changes. This is currently intractable, however, because Gibbs Free Energy Minimization (GFEMs) programs [e.g., @connolly2009; @riel2022] used to calculate PTX-dependent phase relations remain too slow to apply recursively during a geodynamic simulation (10$^2$--10$^4$ ms per node). While parallelization of GFEM programs has increased efficiency dramatically [@riel2022], computing phase relations recursively during geodynamic simulations requires GFEM efficiency on the order of $\leq$ 10$^0$--10$^{-1}$ ms to be feasible. A rate of improvement from 10$^4$--10$^2$ to 10$^0$--10$^{-1}$ ms seems unlikely within the current GFEM paradigm, however, and applying parallelization across a sufficient number of CPU/GPU cores can be inaccessible in many cases.

This study demonstrates an alternative approach to inferring phase relations and rock properties using pre-trained machine learning models (referred to as RocMLMs). We begin by detailing our methods for building, training, and evaluating RocMLMs. We then demonstrate that RocMLMs can predict rock properties up to 10$^3$ faster than commonly-used GFEM programs with equivalent accuracies. Our results imply that real-time recursive inference of rock properties during geodynamic simulations is feasible with RocMLMs. Finally, we compare RocMLM predictions with empirically-derived geophysical datasets and discuss the validity and limitations of RocMLMs with respect to their implementation in geodynamic simulations.

# Methods {#sec:methods}

The following sections describe our methods for building, training, and evaluating RocMLMs with three main objectives in mind. First, define the size and scope of RocMLM training data to ensure widespread applicability of RocMLMs to the upper mantle (@sec:training-data-design). Second, define a generalized approach to generating RocMLM training data to ensure widespread reproducibility with any GFEM program (e.g., MAGEMin, Perple_X, and others) (@sec:generate-training-data). Third, define a set of RocMLMs input features that are routinely computed during geodynamic simulations to ensure widespread applicability of RocMLMs to various geodynamic codes. We also describe our methods for determining the best RocMLM algorithm by cross-validation (@sec:train-evaluate-rocmlms).

## RocMLM Training Dataset Design {#sec:training-data-design}

### PT Conditions {#sec:pt-conditions}

High-pressure experiments constrain the reaction $\text{olivine} \xrightarrow{\text{410 km}} \text{ringwoodite}$ between 14.0 ± 1.0 GPa and 1600 ± 400 K with Clapeyron slopes between 2.4x10$^{-3}$ ± 1.4x10$^{-3}$ GPa/K  [@akaogi1989; @katsura1989; @morishima1994; @li2019]. Likewise, the reaction $\text{ringwoodite} \xrightarrow{\text{660 km}} \text{bridgmanite} + \text{ferropericlase}$ is constrained between 24.0 ± 1.5 GPa and 1600 ± 400 K with negative Clapeyron slopes between -2.0x10$^{-3}$ ± 1.6x10$^{-3}$ GPa/K  [@akaogi2007; @bina1994; @litasov2005; @katsura2003; @ito1990; @ito1982; @ito1989a; @ito1989b; @hirose2002; @ishii2018]. We therefore compute RocMLM training data within a rectangular PT region bound between 1.0–28.0 GPa and 773–2273 K to encompass expected conditions for the entire upper mantle---from the Moho at 35 km to just beyond the base of the MTZ at 660 km ([@fig:training-dataset-design]).

@fig:training-dataset-design shows that our training dataset includes PT conditions that are not expected to exist in the upper mantle, nor typically modeled during geodynamic simulations [e.g., very cold conditions below thermal gradients of $\leq$ 5 K/km, @maruyama1996; @syracuse2010]. In this case, a rectangular training dataset might be considered impractical with respect to training efficiency (unnecessarily large amount of training data) and accuracy (outside the bounds of calibrated thermodynamic data). However, initial sensitivity tests showed comparable RocMLM performance irrespective of the range of PT conditions used to generate RocMLM training data (grey vs. blue shaded regions in @fig:training-dataset-design). Thus, we adopted a regular rectangular training dataset design because it is computationally convenient and does not deteriorate RocMLM accuracy.

![PT diagram showing the range of conditions for generating RocMLM training data (grey region) compared to range of possible upper mantle conditions (blue shaded region). Geotherm 1 (solid black line) and geotherm 2 (dashed black line) assume mantle potential temperatures of 273 K and 1773 K with 1 K/km and 0.5 K/km thermal gradients, respectively. Phase boundaries for the 410 km and 660 km discontinuities (colored lines) are after @li2019.](assets/figs/training-dataset-design.png){#fig:training-dataset-design}

### Bulk Mantle Compositions {#sec:bulk-rock-compositions}

In principle, the bulk composition of the upper mantle can be estimated from chemical analyses of high-pressure-high-temperature melting experiments and/or mantle-derived intrusive and extrusive igneous rocks [e.g., @allegre1984; @green1979; @ringwood1962; @jagoutz1979; @sun1982; @ringwood1991; @palme2003; @stracke2021]. @tbl:benchmark-samples gives four examples of bulk mantle compositions with varying degrees of depletion by partial melting. Two examples are commonly-referenced in the literature, Primitive Upper Mantle [PUM, @sun1989] and Depleted MORB Mantle [DMM, @workman2005], while the other two examples, Primitive Synthetic Upper Mantle (PSUM) and Depleted Synthetic Upper Mantle (DSUM), were derived in this study. Below we describe our statistical approach to deriving PSUM and DSUM from chemical analyses of thousands of peridotite samples.

{{ benchmark-samples.md }}

Bulk chemical analyses of peridotite samples were downloaded using the [Earthchem.org](https://earthchem.org) Search Portal with a single search criterion: *"set sample type > igneous rocks > names from Earthchem categories > igneous-plutonic-ultramafic"*, which queried 19791 samples. Samples lacking analyses for SiO$_2$, MgO, Al$_2$O$_3$, or CaO were excluded from the dataset. All unknown rock types, chromitites, limburgites, wehrites, undifferentiated peridotites, dunites, and pyroxenite samples were also excluded from the dataset to focus on samples that are most likely the products or sources of fractional partial melting, rather than the products of metasomatism or melt refertilization [@elthon1992], although the samples' petrogeneses are ambiguous based on the chemical analyses alone. The data were grouped according to the remaining rock types (lherzolite and harzburgite) and outliers were removed from each group using a 1.5 inter-quartile range threshold. Cr and Ni measured as trace elements (ppm) were converted to Cr$_2$O$_3$ and NiO (wt.%) and all Fe oxides were converted to Fe$_2$O$_3$T. Total oxides were then checked against H$_2$O, CO$_2$, and LOI to determine if chemical analyses were performed before or after ignition. Analyses with total oxides summing to $\leq$ 97 % or $\geq$ 103 % were considered erroneous, or otherwise low-quality, and excluded from the dataset. Finally, all analyses were normalized to a volatile-free basis before converting Fe$_2$O$_3$T to FeOT. The final dataset contains 3707 chemical analyses of classified peridotite samples (@tbl:earthchem-counts).

PCA was applied to the standardized peridotite dataset to reduce the chemical system to two principal components (PC1 and PC2). PCA requires non-missing data, however, so samples were first arranged by MgO and SiO2 and a k-nearest neighbor algorithm was applied to impute missing oxide analyses (see @tbl:earthchem-counts for missing analyses counts). After PCA was applied, two mixing lines were defined by connecting the centroids of lehrzolite, harzburgite, and dunite samples (i.e., the median values for PC1 and PC2 for each group). The lehrzolite-harzburgite and harzburgite-dunite mixing lines were then extended until reaching the approximate location of the most fertile (Al$_2$O$_3$-CaO-TiO$_2$-rich) and most refractory (MgO-rich, SiO2-poor) peridotite samples (@fig:earthchem-mixing-arrays). The mixing lines shown in @fig:earthchem-mixing-arrays approximate a wide array of hypothetical mantle compositions derived from the natural rock record---providing a basis for sampling synthetic bulk mantle compositions used to generate RocMLM training data (@fig:earthchem-harker-diagram).

![PCA diagrams showing the distribution of peridotite data from Earthchem.org (left panel) and mixing array between endmember mantle compositions PSUM and DSUM (right panel). Thick black arrows (left panel) indicate the strongest PCA loading vectors. The bold dashed line (right panel) indicates the mixing array between PSUM and DSUM, while the bold solid lines indicate the upper and lower boundaries for random sampling. Melt fraction is calculated with a modal fractional melting model (@eq:melt-fraction).](assets/figs/earthchem-mixing-arrays.png){#fig:earthchem-mixing-arrays}

{{ earthchem-counts.md }}

![Harker Diagrams vs. SIO2 (in wt.%) showing the distribution of peridotite samples from Earthchem.org (colored contours and pie chart, @tbl:earthchem-counts). PUM (black square) and DMM (white square) are commonly-referenced bulk mantle compositions (@tbl:benchmark-samples), while PSUM (black circle) and DSUM (white circle) define a mixing array (see @fig:earthchem-mixing-arrays) for sampling synthetic bulk mantle compositions used to generate RocMLM training data (black datapoints).](assets/figs/earthchem-harker-diagram.png){#fig:earthchem-harker-diagram}

### Defining a Depletion Index for RocMLM Training {#sec:melt-fractions}

In order to reduce the dimensionality of the training dataset from eleven dimensions (9 oxide components + PT), to three dimensions (1 compositional dimension + PT), we estimated the amount of melt extraction (depletion) that might have produced the synthetic bulk mantle compositions used for generating RocMLM training data. Here we assumed that all synthetic samples were derived from a PUM source (@tbl:benchmark-samples) by modal fractional melting [after @shaw1970]:

\begin{equation}
	\frac{C_{\text{TiO}_2}^s}{C_{\text{TiO}_2}^0} = R = (1 - F)^{\frac{1}{D_0} - 1} \label{eq:shaw-melting}
\end{equation}

\noindent where $R$ is the ratio of the TiO$_2$ concentration of the sample to the initial PSUM source (@tbl:benchmark-samples), $F$ is the melt fraction, and $D_0$ = 0.05 is the bulk distribution coefficient for TiO$_2$ in peridotite [after @brown2016]. Depletion is calculated by rearranging @eq:shaw-melting for $F$ and subtracting it from 1:

\begin{equation}
	D = 1 - F = R^{\frac{1}{(\frac{1}{D_0}) - 1}} \label{eq:melt-fraction}
\end{equation}

Training RocMLMs on $D$ instead of nine oxide components is ideal for two reasons: 1) it greatly increases computational efficiency and 2) unlike oxide components, $F$ is routinely implemented in numerical geodynamic simulations [e.g., @cerpa2019; @gerya2003; @sizova2010; @li2019; @yang2020; @kelley2010]. Although we chose $D$ instead of $F$ for RocMLM training to distinguish $F$ used for RocMLM training (a model input) from $F$ predicted by RocMLM inference (a model output), $D$ and $F$ are otherwise interchangeable and thus, @eq:shaw-melting offers a generalized solution for coupling RocMLMs to various geodynamic codes. Three critical assumptions implicit to @eq:shaw-melting are important to note, however, because they impact $F$, and thus ultimately impact RocMLM training and predictions: 1) melt is instantaneously removed from the source region, 2) $D_0$ is constant, and 3) minerals melt in the same proportions that they exist in the source rock. While these assumptions are unlikely to occur in nature, @eq:melt-fraction predicts that DMM is produced by 2.2 % melt extraction from PUM (@tbl:benchmark-samples), which is consistent with the degree of melt extraction estimated by others [2-3 % melt removal from PUM, @workman2005].

## Generating RocMLM Training Data {#sec:generate-training-data}

We use the GFEM program Perple_X [version 7.0.9, @connolly2009] to generate RocMLM training data using PT conditions as described in @sec:pt-conditions (@fig:training-dataset-design) and synthetic bulk mantle compositions as described in @sec:bulk-rock-compositions (@fig:earthchem-harker-diagram). All Perple_X models include equations of state for pure phases: quartz, coesite, stishovite, kyanite, corundum, and rutile, and solution phases: feldspar, spinel, garnet, clinopyroxene, orthopyroxene, olivine, ilmenite, wadsleyite, ringwoodite, wuestite, perovskite, ferropericlase, high-pressure clinopyroxene, and silicate melt. Thermodynamic data (tc-ds634.txt) are from @holland2018, with updates from @tomlinson2021 and @holland2022, which are specifically formulated for a wide array of igneous rocks and melt compositions. @holland2018 itself is an extension of the foundational database from @holland2011, which is calibrated up to 300 GPa and 2000 ˚C, making tc-ds634.txt the most appropriate choice for the upper mantle. Once configured, Perple_X generates RocMLM training data (rock properties) by minimizing the total GFE of a multicomponent multiphase thermodynamic system at a fixed PT point, defined by the following equation [@gibbs1878; @spear1993]:

\begin{equation}
	\text{GFE} = \sum_{\lambda=1}^{\Lambda} p_{\lambda} \sum_{n=1}^{N} p_n \mu_n + \sum_{\omega=1}^{\Omega} p_{\omega} \mu_{\omega} \label{eq:gfe}
\end{equation}

\noindent where $\Lambda$ is the number solution phases, $N$ is the number of end-member compounds that mix to form solution phases, and $\Omega$ is the number of pure (stoichiometric) phases. @eq:gfe states that the total GFE of a thermodynamic system (at a fixed PT) is the weighted sum of the molar fractions $p_{\lambda}$ of solution phases $\lambda$ and the molar fractions $p_n$ and chemical potentials $\mu_n$ of end-member compounds $n$ that mix to form solution phases $\lambda$, plus the weighted sum of the molar fractions $p_{\omega}$ and chemical potentials $\mu_{\omega}$ of pure phases $\omega$.

For pure phases, the chemical potential is a constant [@spear1993]:

\begin{equation}
	\mu_{\omega} = \text{GFE}_{\omega}^{\text{standard}}
\end{equation}

\noindent where $\text{GFE}_{\omega}^{\text{standard}}$ is the Gibbs Free Energy of formation at standard PT (1 bar, 273 K). For a solution phase, however, the chemical potential is described by a non-ideal mixture of end-member compounds:

\begin{equation}
	\mu_n = \text{GFE}_n^{\text{standard}} + RTln(a_n^{\text{ideal}}) + \text{GFE}_n^{\text{excess}}
\end{equation}

\noindent where $\mu_n$ is the chemical potential of end-member compound $n$, $R$ is the gas constant, $T$ is temperature, $a_n^{\text{ideal}}$ is the activity of an end-member compound defined by ideal mixing: $a_n^{\text{ideal}} = x_n$, where $x_n$ is the molar fraction of the end-member compound $n$. The $\text{GFE}_n^{\text{excess}}$ term models non-ideal behavior by defining symmetric (site-independent) and asymmetric (site-dependent) mixing of end-member compounds on different crystallographic sites for a particular solution phase $\lambda$ [@holland2003; @powell1993].

Additional constraints are imposed on @eq:gfe by the Gibbs-Duhem equation [@spear1993]:

\begin{equation}
	\mu_{(\omega,n)} = \sum_{c=1}^{C} \mu_c a_{(\omega,n) c} \label{eq:gibbs-duhem}
\end{equation}

\noindent where $C$ is the number of chemical components (oxides) considered in the thermodynamic system. @eq:gibbs-duhem states that the total chemical potential of a pure phase or end-member compound $\mu_{(\omega,n)}$ is equal to the weighted sum of the chemical potentials $\mu_c$ and activities $a_{(\omega,n) c}$ of the chemical components dissolved in the pure phase or end-member compound. @eq:gibbs-duhem implies that the total GFE of the thermodynamic system is dependent on its bulk chemical composition. Consequently, for a fixed bulk composition at equilibrium, the stable mineral assemblage must satisfy the Gibbs phase rule:

\begin{equation}
	F = C - \Phi + 2
\end{equation}

\noindent where $F$ is the number of degrees of freedom, $C$ is the number of chemical components (oxides), and $\Phi$ is the number of stable mineral phases in the rock. In this case, the "degrees of freedom" $F$ refers to the number of independent mineral phases that can vary their chemical potentials while the system remains in equilibrium.

Lastly, conservation of mass is maintained by equating the sum total of the chemical potentials in the system to the bulk rock composition:

\begin{equation}
	\sum_{c=1}^{C} \sum_{\lambda=1}^{\Lambda} p_{\lambda} \sum_{n=1}^{N} a_{nc} p_n + \sum_{c=1}^{C} \sum_{\omega=1}^{\Omega} p_{\omega} a_{\omega c} = \sum_{c=1}^{C} \text{bulk-rock}_c \label{eq:mass-balance}
\end{equation}

In principle, applying identical sets of solution phase models, thermodynamic data, and bulk compositions to @eq:gfe will define identical GFE hyperplanes (i.e. define the same G-X "surfaces" in multidimensional space). This implies that GFEM programs should converge on identical phase relations irrespective of the minimization algorithm. Similar results are therefore expected for other GFEM programs, although this study uses Perple_X exclusively for generating RocMLM training data.

## Training and Evaluating RocMLMs {#sec:train-evaluate-rocmlms}

RocMLM training data were preprocessed using the following procedure. First, square PT grids of rock properties (density, Vp, and Vs) produced by Perple_X for $n$ random synthetic bulk mantle compositions were combined into cubic three-dimensional arrays, $Z = (z_{1,1,1}, \ldots, z_{w,w,w})$, where $w$ = $n$ = 128 is the resolution of the PT grid. $Z$ was reshaped into flat arrays of training features (PT and $F$), $X = (x_{1,1,1}, \ldots, x_{v,v,v})$, and targets (density, Vp, Vs), $y = (y_{1,1,1}, \ldots, y_{v,v,v})$, where $v = w^3$ is the total number of training examples. $X$ and $y$ were then standardized by removing their mean and scaling to unit variance:

\begin{align}
	\overline{X} &= \frac{X - \mu_X}{\sigma_X} \label{eq:standard-scaler} \\
	\overline{y} &= \frac{y - \mu_y}{\sigma_y} \nonumber
\end{align}

\noindent where $\overline{X}$ and $\overline{y}$ are the standardized feature and target arrays, respectively, and $\mu$ and $\sigma$ are the mean and standard deviation of the training examples. Applying @eq:standard-scaler ("z-score normalization") to RocMLM training data allows RocMLMs to fit training features and targets with different units (GPa, K, g/cm$^3$, km/s). The preprocessed training data were then fit with eight different non-linear RocMLMs from the scikit-learn python library [@scikit2011]. Each RocMLM was tuned with a grid-search algorithm where a performance score (RMSE) was evaluated over all combinations of relevant parameters (@tbl:rocml-config). The set of parameters that produced the best score was used to train the RocMLM.

During RocMLM training, performance metrics were evaluated using a k-fold cross-validation technique to determine how well each RocMLM fit the training data. The training data were shuffled to reduce the impact of inherent ordering in the data and partitioned into $k$ = 8 non-overlapping folds of training examples. Cross-validation then proceeded with $k$ iterations, where in each iteration, models were fit to training examples from $k$-1 folds and performance (RMSE) was evaluated on the remaining fold. After all iterations completed, performance was evaluated on a separate set of input features (validation set) not seen by the RocMLM during training to provide a measurement of generalizability to new data.

{{ rocmlm-config.md }}

# Results {#sec:results}

## Comparing Rock Properties Inferred by Perple_X, RocMLMs, and PREM {#sec:comparing-rocmlms}

RocMLM performance can be assessed qualitatively by comparing depth profiles of rock properties generated by Perple_X with the profiles predicted by RocMLMs and inferred by PREM ([@fig:prem-PUM-DT-rho]). Profiles extracted from RocMLMs are nearly identical to Perple_X profiles and generally indicate external consistency with empirically-derived profiles from the Preliminary Reference Earth Model [PREM, @dziewonski1981].

![PT-density diagrams (in g/cm$^3$) calculated by (a) Perple_X and (b) a Decision Tree RocMLM for a PUM bulk composition (@tbl:benchmark-samples). (c) PT-error diagram (in g/cm$^3$) showing the difference between (a) and (b).](assets/figs/image-PUM-DT-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) Perple_X and (b) a Decision Tree RocMLM for a PUM bulk composition (@tbl:benchmark-samples). (c) PT-error diagram (in g/cm$^3$) showing the difference between (a) and (b).](assets/figs/surf-PUM-DT-rho.png)

![Density profiles comparing (a) MAGEMin and (b) Perple_X models with the empirically-derived Preliminary Reference Earth Model (PREM) from @dziewonski1981. Performance metrics R$^2$ and RMSE show the correlations and accuracies of the Decision Tree RocMLM compared to MAGEMin and Perple_X. MAGEMin and Perple_X models assume a PUM bulk composition (@tbl:benchmark-samples).](assets/figs/prem-PUM-DT-rho.png){#fig:prem-PUM-DT-rho}

![PT diagrams and depth profiles showing an unprocessed endmember training example [PUM, @sun1989]. Density is shown in (a, b, c), Vp in (d, e, f), and Vs in (g, h, i). The bold white line in is the geotherm used to define the depth profiles. PREM is the Preliminary Reference Earth Model from @dziewonski1981.](assets/figs/image9-PUM-train.png){#fig:image9-PUM-train}

## RocMLM Efficiency and Performance

{{ rocmlm-performance.md }}

![caption](assets/figs/rocmlm-performance.png)

# Discussion

\clearpage

# Acknowledgements

# Open Research

All data, code, and relevant information for reproducing this work can be found at [https://github.com/buchanankerswell/kerswell_et_al_rocmlm](https://github.com/buchanankerswell/kerswell_et_al_rocmlm), and at [https://doi.org/10.17605/OSF.IO/K23TB](https://doi.org/10.17605/OSF.IO/K23TB), the official Open Science Framework data repository (Kerswell et al., 2024). All code is MIT Licensed and free for use and distribution (see license details).

# References

<div id="refs"></div>

\cleardoublepage

# Appendix

## GFEM Benchmarking

PUM and DMM mantle compositions were used for benchmarking MAGEMin and Perple_X efficiency (@tbl:benchmark-samples). [@fig:gfem-efficiency] shows the computation times with respect to various PT grid resolutions (8x8, 16x16, 32x32, 64x64, 128x128). All computations were made on a Macbook Pro (2022; M2 chip) with macOS 13.4 and Python 3.11.4.

![caption](assets/figs/gfem-efficiency.png){#fig:gfem-efficiency}

\clearpage

## RocMLM Comparisons

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Decision Tree models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/surf-PUM-DT-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Random Forest models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/surf-PUM-RF-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by K Nearest models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/surf-PUM-KN-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Neural Network 1L models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/surf-PUM-NN1-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Neural Network 2L models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/surf-PUM-NN2-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Neural Network 3L models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/surf-PUM-NN3-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Decision Tree models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/image-PUM-DT-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Random Forest models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/image-PUM-RF-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by K Nearest models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/image-PUM-KN-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Neural Network 1L models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/image-PUM-NN1-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Neural Network 2L models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/image-PUM-NN2-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Neural Network 3L models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/image-PUM-NN3-rho.png)
