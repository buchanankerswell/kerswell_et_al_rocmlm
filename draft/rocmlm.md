# Introduction

The dominant mineral phases in Earth's upper mantle are olivine, ringwoodite, bridgmanite, and ferropericlase [@ringwood1975; @ringwood1991], comprising up to 60–90% of the mantle's volume [e.g., @stixrude2012]. These Mg-Fe-rich phases form by a series of discrete reactions (@eq:mantle-transitions) that define mantle transition zones (MTZs) near 410 km and 660 km depths beneath Earth's surface. MTZs are characterized by relatively sharp boundaries with contrasting elastic properties [e.g, density\, Vp\, and Vs\, @dziewonski1981; @ita1992] that strongly impact mantle convection, melting, and plate tectonics [@fukao2001; @ringwood1991; @schubert1975; @kuritani2019; @karato2001; @wang2015; @jenkins2016].

\begin{align}
	\text{olivine} \xrightarrow{\text{410 km}} \text{ringwoodite} &\xrightarrow{\text{660 km}} \text{bridgmanite} + \text{ferropericlase} \label{eq:mantle-transitions} \\
	\text{(Mg,Fe)}_{2}\text{SiO}_{4} \xrightarrow{\text{410 km}} \text{(Mg,Fe)}_{2}\text{SiO}_{4} &\xrightarrow{\text{660 km}} \text{(Mg,Fe)}\text{SiO}_{3} + \text{(Mg,Fe)}\text{O} \nonumber
\end{align}

Although the physio-chemical nature of the upper mantle remains under vigorous investigation [@goes2022; @pearson2014; @yoshino2008; @waszek2021; @kiseeva2018; @fei2017; @zhou2022], important phase changes (e.g., @eq:mantle-transitions) are often implemented in numerical geodynamic simulations with simple pressure-temperature (PT)-dependent reaction boundaries based on high-pressure experiments. This approach has generated many important hypotheses implicating MTZs as critical features impacting tectonic plate motions and water cycling in the deep Earth [e.g., @agrusta2017; @li2019; @torii2007; @yang2020]. However, the tendency to assume fixed mantle compositions and approximate phase transitions with simple functions limit such numerical experiments to a rough zeroth-order approximation that is not self-consistent with respect to thermodynamic variables, most importantly the Gibbs Free Energy (GFE), which is a function of bulk mantle composition.

More comprehensive numerical models of mantle convection would account for GFE by implementing PT-composition (PTX)-dependent phase changes. This is currently intractable, however, because GFE minimization (GFEM) programs [e.g., @connolly2009; @riel2022] used to calculate PTX-dependent phase relations remain too slow [≥ 10$^2$ ms per node, @riel2022] to apply recursively during a geodynamic simulation. While parallelization of GFEM programs has increased efficiency dramatically [@riel2022], computing phase relations during geodynamic simulations requires GFEM efficiency on the order of $\leq$ 10$^0$--10$^{-1}$ ms to be feasible. A rate of improvement from 10$^4$--10$^2$ to 10$^0$--10$^{-1}$ ms seems unlikely within the current GFEM paradigm, however, and applying parallelization across a sufficient number of CPU/GPU cores can be inaccessible in many cases.

This study demonstrates an alternative approach to predicting rock properties using pre-trained machine learning models (referred to as RocMLMs). We begin by detailing our methods for building, training, and evaluating RocMLMs. We then demonstrate that RocMLMs can predict rock properties up to 10$^3$ faster than commonly-used GFEM programs with equivalent accuracies. Finally, we compare RocMLM predictions with reference Earth models derived from seismological datasets [@dziewonski1981; @kustowski2008] and discuss the accuracy and performance of RocMLMs with respect to their implementation in numerical geodynamic models.

# Methods {#sec:methods}

The following sections describe our methods for building, training, and evaluating RocMLMs with four main objectives in mind. First, define the size and scope of RocMLM training data to ensure widespread applicability of RocMLMs to the upper mantle (@sec:training-data-design). Second, define a generalized approach for generating RocMLM training data to ensure widespread reproducibility with any GFEM program (e.g., MAGEMin, Perple_X, and others, @sec:generate-training-data). Third, train RocMLMs on a set of input features that are routinely computed during geodynamic simulations to ensure widespread applicability of RocMLMs to various geodynamic codes (@sec:train-rocmlms). Fourth rank the overall performance of RocMLMs in terms of accuracy and efficiency (@sec:evaluate-rocmlms).

## RocMLM Training Dataset Design {#sec:training-data-design}

### Pressure-Temperature Conditions {#sec:pt-conditions}

High-pressure experiments constrain the reaction $\text{olivine} \xrightarrow{\text{410 km}} \text{ringwoodite}$ between 14.0 ± 1.0 GPa and 1600 ± 400 K with Clapeyron slopes between 2.4x10$^{-3}$ ± 1.4x10$^{-3}$ GPa/K  [@akaogi1989; @katsura1989; @morishima1994; @li2019]. Likewise, the reaction $\text{ringwoodite} \xrightarrow{\text{660 km}} \text{bridgmanite} + \text{ferropericlase}$ is constrained between 24.0 ± 1.5 GPa and 1600 ± 400 K with negative Clapeyron slopes between -2.0x10$^{-3}$ ± 1.6x10$^{-3}$ GPa/K  [@akaogi2007; @bina1994; @litasov2005; @katsura2003; @ito1990; @ito1982; @ito1989a; @ito1989b; @hirose2002; @ishii2018]. We therefore compute RocMLM training data within a rectangular PT region bound between 1.0–28.0 GPa and 773–2273 K to encompass expected conditions for the entire upper mantle---from the Moho at 35 km to just beyond the base of the MTZ at 660 km ([@fig:training-dataset-design]).

@fig:training-dataset-design shows that our training dataset includes PT conditions that are not expected to exist in the upper mantle, nor typically modeled during geodynamic simulations [e.g., very cold conditions below thermal gradients of $\leq$ 5 K/km, @maruyama1996; @syracuse2010]. In this case, a rectangular training dataset might be considered impractical with respect to training efficiency (unnecessarily large amounts of training data) and accuracy (outside the bounds of calibrated thermodynamic data). However, initial sensitivity tests showed comparable RocMLM performance irrespective of the range of PT conditions used to generate RocMLM training data (gray vs. blue shaded regions in @fig:training-dataset-design). Thus, we adopted a regular rectangular training dataset design because it is computationally convenient and does not deteriorate RocMLM accuracy.

![PT diagram showing the range of conditions for generating RocMLM training data (gray region) compared to the range of possible upper mantle conditions beneath continents (blue shaded region). The solid and dashed black lines are average geothermal gradients for Proterozoic and Archean continents, respectively. The dotted black lines are 0.5 K/km mantle adiabats placed at arbitrary mantle potential temperatures of 673 K and 1773 K, representing hypothetical lower and upper bounds for expected mantle PT conditions. Phase boundaries for the 410 km and 660 km discontinuities (colored lines) are after @li2019.](assets/figs/training-dataset-design.png){#fig:training-dataset-design}

### Bulk Mantle Compositions {#sec:bulk-rock-compositions}

In principle, the bulk composition of the upper mantle can be estimated from chemical analyses of high-pressure-high-temperature melting experiments and/or mantle-derived intrusive and extrusive igneous rocks [e.g., @allegre1984; @green1979; @ringwood1962; @jagoutz1979; @sun1982; @ringwood1991; @palme2003; @stracke2021]. @tbl:benchmark-samples gives four examples of bulk mantle compositions with varying degrees of depletion by partial melting. Two examples are commonly-referenced in the literature, Primitive Upper Mantle [PUM, @sun1989] and Depleted MORB Mantle [DMM, @workman2005], while the other two examples, Primitive Synthetic Upper Mantle (PSUM) and Depleted Synthetic Upper Mantle (DSUM), were derived in this study. Below we describe our statistical approach to deriving PSUM and DSUM from chemical analyses of thousands of peridotite samples.

Bulk chemical analyses of peridotite samples were downloaded using the [Earthchem.org](https://earthchem.org) Search Portal with a single search criterion: *"set sample type > igneous rocks > names from Earthchem categories > igneous-plutonic-ultramafic"*, which queried 19791 samples. Samples lacking analyses for SiO$_2$, MgO, Al$_2$O$_3$, or CaO were excluded from the dataset to focus on bulk rock compositions. All unknown rock types, chromitites, limburgites, wehrites, undifferentiated peridotites, dunites, and pyroxenite samples were also excluded from the dataset to focus on samples that were most likely the products of partial melting, rather than the products of fractional crystallization [@bowen1915], fluid-rock interactions [@elthon1992; @kelemen1995], or excess olivine crystallization [@niu1997a]. The data were grouped according to the remaining rock types (lherzolite and harzburgite) and outliers were removed from each group using a 1.5 interquartile range threshold applied to each chemical component. Cr and Ni measured as trace elements (ppm) were converted to Cr$_2$O$_3$ and NiO (wt.%) and all Fe oxides were converted to Fe$_2$O$_3$T. Total oxides were then checked against H$_2$O, CO$_2$, and LOI to determine if chemical analyses were performed before or after ignition. Analyses with total oxides summing to $\leq$ 97% or $\geq$ 103% were considered erroneous, or otherwise low-quality, and excluded from the dataset. Finally, all analyses were normalized to a volatile-free basis before converting Fe$_2$O$_3$T to FeOT. The final dataset contains 3111 chemical analyses of classified peridotite samples (@tbl:earthchem-counts).

PCA was applied to the standardized peridotite dataset to reduce the chemical system from 10 oxide components to two principal components (PC1 and PC2). PCA requires non-missing data, however, so samples were first arranged by MgO and SiO$_2$ and a k-neighbors algorithm was applied to impute missing oxide analyses, which mainly affected the K$_2$O and Na$_2$O components (see @tbl:earthchem-counts for missing analyses counts). After PCA was applied, a mixing line was defined by connecting the lherzolite and harzburgite sample centroids (i.e., the median values for PC1 and PC2 for each group). The lherzolite-harzburgite mixing line was then extended until reaching the approximate location of the most fertile (Al$_2$O$_3$-CaO-TiO$_2$-rich) and most refractory (MgO-rich, SiO$_2$-poor) peridotite samples (@fig:earthchem-mixing-arrays). The mixing line shown in [@fig:earthchem-mixing-arrays]b approximates a wide array of hypothetical mantle compositions derived from the natural rock record---providing a basis for sampling synthetic bulk mantle compositions used to generate RocMLM training data.

{{ earthchem-counts.md }}

### Reducing Bulk Mantle Compositions to a Single Fertility Index Value {#sec:melt-fractions}

In order to reduce the dimensionality of the training dataset from twelve dimensions (10 oxide components + PT), to three dimensions (1 compositional dimension + PT), we estimated the amount of melt extraction (depletion) that might have produced the synthetic bulk mantle compositions used for generating RocMLM training data. Here we assumed that all synthetic samples were derived from a PUM source by modal fractional melting [after @shaw1970]:

\begin{equation}
	\frac{C_{\text{TiO}_2}^s}{C_{\text{TiO}_2}^0} = R = (1 - F)^{\frac{1}{D_0} - 1} \label{eq:shaw-melting}
\end{equation}

\noindent where $R$ is the ratio of the TiO$_2$ concentration of the sample to the initial PUM source (@tbl:benchmark-samples), $F$ is the melt fraction, and $D_0$ = 0.05 is the bulk distribution coefficient for TiO$_2$ in peridotite [after @brown2016]. A Fertility Index (FI) is calculated by rearranging @eq:shaw-melting for $F$ and subtracting $F$ from 1:

\begin{equation}
	\text{FI} = 1 - F = R^{\frac{1}{(\frac{1}{D_0}) - 1}} \label{eq:melt-fraction}
\end{equation}

Training RocMLMs on FI instead of nine oxide components is beneficial for two reasons: 1) it greatly increases RocMLM efficiency and 2) unlike oxide components, melt fraction $F$ is routinely implemented in numerical geodynamic simulations [e.g., @cerpa2019; @gerya2003; @sizova2010; @li2019; @yang2020; @kelley2010]. Although we chose FI for RocMLM training to avoid confusion between $F$ used as a RocMLM input (training feature) and output (prediction), FI and $F$ are otherwise interchangeable. Thus, @eq:shaw-melting offers a generalized solution for coupling RocMLMs to various geodynamic codes.

![PCA diagrams showing (a) the distribution of standardized peridotite data queried from Earthchem.org and (b) mixing array between end-member mantle compositions PSUM and DSUM. Thick black arrows in (a) indicate the strongest PCA loading vectors. The bold dashed line in (b) indicates the mixing array between PSUM and DSUM, while the bold solid lines indicate the upper and lower boundaries for random sampling. Fertility Index is calculated with a modal fractional melting model (@eq:melt-fraction).](assets/figs/earthchem-mixing-arrays.png){#fig:earthchem-mixing-arrays}

Three implicit assumptions in @eq:shaw-melting are important to note, however, because they impact FI, and thus ultimately impact RocMLM predictions: 1) melt is instantaneously removed from the source region, 2) $D_0$ is constant, and 3) minerals melt in the same proportions that they exist in the source rock. While these assumptions strongly depend on the progression of melting reactions and mantle conditions [@langmuir1992; @niu1997b; @brown2016], @eq:melt-fraction predicts that DMM is produced by 2.2% melt extraction from PUM (@tbl:benchmark-samples), which is consistent with the degree of depletion indicated by trace element patterns and mass balance constraints [2-3% melt removal from PUM, @workman2005]. We therefore consider FI an adequate first-order proxy for bulk mantle composition.

{{ benchmark-samples.md }}

## Generating RocMLM Training Data {#sec:generate-training-data}

We use the GFEM program Perple_X [version 7.0.9, @connolly2009] to generate RocMLM training data using PT conditions as described in @sec:pt-conditions (@fig:training-dataset-design) and synthetic bulk mantle compositions as described in @sec:bulk-rock-compositions. All Perple_X models include equations of state for pure phases: quartz, coesite, stishovite, kyanite, corundum, and rutile, and solution phases: feldspar, spinel, garnet, clinopyroxene, orthopyroxene, olivine, ilmenite, wadsleyite, ringwoodite, wüstite, perovskite, ferropericlase, high-pressure clinopyroxene, and silicate melt. Thermodynamic data (tc-ds634.txt) are from @holland2018, with updates from @tomlinson2021 and @holland2022, which are specifically formulated for a wide array of igneous rocks and melt compositions. @holland2018 itself is an extension of the foundational database from @holland2011, which is calibrated up to 300 GPa and 2000 ˚C, making tc-ds634.txt the most appropriate choice for the upper mantle. Once configured, Perple_X generates RocMLM training data (rock properties) by minimizing the total GFE of a multicomponent multiphase thermodynamic system at a fixed PT point, defined by the following equation [@gibbs1878; @spear1993]:

\begin{equation}
	\text{GFE} = \sum_{\lambda=1}^{\Lambda} p_{\lambda} \sum_{n=1}^{N} p_n \mu_n + \sum_{\omega=1}^{\Omega} p_{\omega} \mu_{\omega} \label{eq:gfe}
\end{equation}

\noindent where $\Lambda$ is the number of solution phases, $N$ is the number of end-member compounds that mix to form solution phases, and $\Omega$ is the number of pure (stoichiometric) phases. @eq:gfe states that the total GFE of a thermodynamic system (at a fixed PT) is the weighted sum of the molar fractions $p_{\lambda}$ of solution phases $\lambda$ and the molar fractions $p_n$ and chemical potentials $\mu_n$ of end-member compounds $n$ that mix to form solution phases $\lambda$, plus the weighted sum of the molar fractions $p_{\omega}$ and chemical potentials $\mu_{\omega}$ of pure phases $\omega$.

For pure phases, the chemical potential is a constant [@spear1993]:

\begin{equation}
	\mu_{\omega} = \text{GFE}_{\omega}^{\text{standard}}
\end{equation}

\noindent where $\text{GFE}_{\omega}^{\text{standard}}$ is the GFE of formation at standard PT (1 bar, 273 K). For a solution phase, however, the chemical potential is described by a non-ideal mixture of end-member compounds:

\begin{equation}
	\mu_n = \text{GFE}_n^{\text{standard}} + RTln(a_n^{\text{ideal}}) + \text{GFE}_n^{\text{excess}}
\end{equation}

\noindent where $\mu_n$ is the chemical potential of the end-member compound $n$, $R$ is the gas constant, $T$ is temperature, $a_n^{\text{ideal}}$ is the activity of an end-member compound defined by ideal mixing: $a_n^{\text{ideal}} = x_n$, where $x_n$ is the molar fraction of the end-member compound $n$. The $\text{GFE}_n^{\text{excess}}$ term models non-ideal behavior by defining symmetric (site-independent) and asymmetric (site-dependent) mixing of end-member compounds on different crystallographic sites for a particular solution phase $\lambda$ [@holland2003; @powell1993].

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

## Training RocMLMs {#sec:train-rocmlms}

RocMLM training data were preprocessed using the following procedure. First, two-dimensional grids of rock properties (density, Vp, and Vs) calculated by Perple_X were stacked into three-dimensional arrays, $Z$ = $(z_{1,1,1}, \ldots, z_{n,w,w})$, where $w$ = 128 is the resolution of the PT grid and $n$ = 128 is the number of random synthetic bulk mantle compositions represented by a FI value. $Z$ was flattened and separated into arrays of training features (PT and FI), $X$ = $(x_{1,1,1}, \ldots, x_{v,v,v})$, and training targets (density, Vp, Vs), $y$ = $(y_{1,1,1}, \ldots, y_{v,v,v})$, where $v$ = $n \cdot w^2$ = 128$^3$ is the total number of training examples. $X$ and $y$ were then scaled using "z-score normalization" to avoid strong biasing of RocMLMs by training data with mixed units (GPa, K, g/cm$^3$, and km/s):

\begin{align}
	\overline{X} &= \frac{X - \mu_X}{\sigma_X} \label{eq:standard-scaler} \\[12pt]
	\overline{y} &= \frac{y - \mu_y}{\sigma_y} \nonumber
\end{align}

\noindent where $\overline{X}$ and $\overline{y}$ are the scaled feature and target arrays, respectively, and $\mu_X$, $\mu_y$, $\sigma_X$, and $\sigma_y$ are the means and standard deviations of the feature and training examples, respectively. The preprocessed training data were then fit with eight different non-linear regression algorithms from the scikit-learn python library [@scikit2011]. Each regression algorithm was tuned with a grid search approach, where a performance score (RMSE) was evaluated over all hyperparameter combinations relevant to the particular regression algorithm (@tbl:rocmlm-config). The set of hyperparameters that produced the best score was used to train the RocMLM.

{{ rocmlm-config.md }}

## Evaluating RocMLM Accuracy and Performance {#sec:evaluate-rocmlms}

RocMLM accuracy (in terms of R$^2$ and RMSE) was evaluated by: 1) testing RocMLMs on separate training data to determine the generalizability of RocMLMs to unknown mantle conditions (validation accuracy), and 2) comparing RocMLMs with reference Earth Models [PREM and STW105, @dziewonski1981; @kustowski2008] to test RocMLM predictions against inversions of seismological data (external accuracy). RocMLM performance, on the other hand, was evaluated by: 1) measuring single-point prediction times (execution speed), and 2) scaling execution speed by RocMLM file size to account for information compression (efficiency).

The number of PT points and synthetic bulk mantle compositions used for generating training data were varied from 8 to 128 to test the sensitivity of RocMLM accuracy and performance with respect to the size and composition of the training dataset (2$^{11}$–2$^{21}$ total training examples). The same sets of training data were used to evaluate single-point execution speed using a lookup table (LUT) approach, where a cubic spline interpolation was applied to the training dataset and rock properties were evaluated at an arbitrary PT-FI point. Prediction accuracy and performance were measured in a consistent manner so that direct comparisons could be made between RocMLMs, LUTs, and GFEM programs.

# Results {#sec:results}

## RocMLM Accuracy {#sec:rocmlms-accuracy}

The following examples of Decision Tree (DT, @fig:image12-PUM-DT) and single-layer Neural Network (NN1, @fig:image12-PUM-NN1) RocMLMs demonstrate how different regression algorithms ultimately influence the accuracy of RocMLM predictions (see @sec:rocmlm-regression-algorithms for all regression algorithms).

DT predictions are indistinguishable from Perple_X, with low error rates (RMSE density: 0.003 g/cm$^3$, Vp: 0.019 km/s, Vs: 0.033 km/s) that indicate a nearly-perfect mapping of the training dataset by the DT algorithm (@fig:image12-PUM-DT). The few non-zero residuals between DT and Perple_X predictions tend to concentrate at phase transitions, especially the onset of partial melting at HT-LP conditions ([@fig:image12-PUM-DT]g–i). In contrast, NN1 predictions are notably more diffuse than Perple_X (@fig:image12-PUM-NN1), with higher error rates (RMSE density: 0.042 g/cm$^3$, Vp: 0.132 km/s, Vs: 0.266 km/s) that indicate an inability to resolve sharp gradients in physical properties ([@fig:image12-PUM-NN1]g–i).

![PT diagrams showing density (left column, g/cm3), Vp (middle column, km/s), and Vs (right column, km/s) predictions from a Perple_X model with a PUM bulk composition (a–c), a Decision Tree RocMLM (d–f), and absolute differences between Perple_X and DT (g–i). Depth profiles (j–l) compare Perple_X and DT predictions extracted along average geotherms for a Proterozoic craton and mid-ocean ridge (solid and dashed white lines, respectively) with reference Earth models PREM [solid black line, @dziewonski1981] and STW105 [dotted black line, @kustowski2008]. CONT = Proterozoic craton, MOR = mid-ocean ridge. The RMSE indicates the measured differences between DT-CONT and PREM. Colored ribbons indicate 3% uncertainty in RocMLM predictions.](assets/figs/image12-PUM-DT.png){#fig:image12-PUM-DT}

![PT diagrams showing density (left column, g/cm3), Vp (middle column, km/s), and Vs (right column, km/s) predictions from a Perple_X model with a PUM bulk composition (a–c), a single-layer Neural Network RocMLM (d–f), and differences between Perple_X and NN1 (g–i). Depth profiles (j–l) compare Perple_X and NN1 predictions extracted along average geotherms for a Proterozoic craton and mid-ocean ridge (solid and dashed white lines, respectively) with reference Earth models PREM [solid black line, @dziewonski1981] and STW105 [dotted black line, @kustowski2008]. Other legend details are the same as @fig:image12-PUM-DT.](assets/figs/image12-PUM-NN1.png){#fig:image12-PUM-NN1}

Comparing Perple_X and RocMLM depth profiles with PREM and STW105 reveals minor deviations with respect to density ([@fig:image12-PUM-DT; @fig:image12-PUM-NN1]j), but poor correlations, high error rates, and a systematic shift towards lower seismic velocities with respect to elastic properties ([@fig:image12-PUM-DT; @fig:image12-PUM-NN1]k–l). Irrespective of this discrepancy, the influence of different regression algorithms on external accuracy can be evaluated by directly comparing DT and NN1 depth profiles. For example, the smooth NN1 profiles do not resolve the sharp gradients and discontinuities indicated by PREM and STW105 ([@fig:image12-PUM-NN1]j–l), and even exhibit inverted Vs gradients near the 410 km discontinuity ([@fig:image12-PUM-NN1]l). Meanwhile, the DT profiles resolve all sharp gradients and discontinuities, including the $\text{olivine} \xrightarrow{\text{410 km}} \text{ringwoodite}$ and $\text{ringwoodite} \xrightarrow{\text{660 km}} \text{bridgmanite} + \text{ferropericlase}$ transitions. We note, however, that deeper NN architectures with three hidden-layers (NN3) also resolve discontinuities with roughly equivalent external accuracies as DT and KN algorithms (@tbl:rocmlm-performance, compare profiles in @sec:rocmlm-regression-algorithms).

{{ rocmlm-performance.md }}

## RocMLM Performance {#sec:rocmlm-performance-and-efficiency}

@fig:rocmlm-tradeoffs shows how execution speed, efficiency, and accuracy scale with dataset size. Dataset size refers to the amount of training examples (or retrievable petrological information) stored by LUTs, or "learned" by RocMLMs. In the case of GFEM programs, the dataset size is irrelevant because GFEM programs do not depend on any stored information other than the required thermodynamic datasets, phase solutions models, and executable programs. Thus, GFEM performance is indicated by the minimum and maximum point-by-point execution speeds (33.5–228.1 ms) and efficiencies (460.1–3137.8 ms$\cdot$Mb) measured while generating RocMLM training datasets.

![Computational efficiency of various programs in terms prediction time (a) and prediction time scaled by file size (b). Some programs (LUT, DT, KN) become cumbersome and/or slow and/or lose accuracy (RMSE) as the amount of training examples increases, while other programs (NN1 and NN3) maintain high-speed, good compression, and increase accuracy. All computations were made on a Macbook Pro (2022; M2 chip) with macOS 13.4 and Python 3.11.4.](assets/figs/rocmlm-tradeoffs.png){#fig:rocmlm-tradeoffs}

For LUTs, execution speed and efficiency scale roughly linearly with dataset size on a logarithmic scale---indicating an inverse power-law relationship between LUT size and performance ([@fig:rocmlm-tradeoffs]). For RocMLMs, performance scales differently depending on the performance metric and regression algorithm. For example, RocMLM execution speed remains roughly constant, or increasing slightly, and shows relatively small variance among all regression algorithms (0.206 ± 0.833 ms, 2$\sigma$, [@fig:rocmlm-tradeoffs]a). Yet RocMLM efficiency is markedly different for DT and KN algorithms compared to NN algorithms ([@fig:rocmlm-tradeoffs]b). Despite the fast execution times of KN and DT algorithms ([@fig:rocmlm-tradeoffs]a), their efficiency scales roughly linearly with dataset size on a logarithmic scale---indicating an inverse power-law relationship between efficiency and dataset size similar to LUTs ([@fig:rocmlm-tradeoffs]b). NN algorithms, on the other hand, show roughly constant efficiencies that indicate a high degree of information compression without sacrificing execution speed ([@fig:rocmlm-tradeoffs]b). We note that training times for NN algorithms are many orders of magnitude larger than DT and KN algorithms (@tbl:rocmlm-performance). However, training times are neither a limiting factor, nor critical for geological applications as training is independent from, and precedes numerical geodynamic simulation.

Since accuracy is measured relative to the rock properties generated by GFEM programs, GFEM programs and LUTs have perfect accuracy by definition. With respect to RocMLMs, validation accuracies are observed to be roughly constant for regression algorithms that tend to overfit the training data (DT and KN, @tbl:rocmlm-performance), while NNs show a positive correlation between accuracy and dataset size (@fig:rocmlm-tradeoffs). In addition to improving accuracy with increasing amounts of training examples, NN accuracy increases with the number of hidden-layers (@fig:rocmlm-tradeoffs) because deeper networks are more capable of fitting sharp gradients in the training data (see @sec:rocmlm-regression-algorithms for examples of NN1, NN2, and NN3 RocMLMs).

# Discussion

## RocMLM Performance Tradeoffs {#sec:rocmlms-tradeoffs}

RocMLM performance and accuracy are both critical for geological applications and crucial for determining if RocMLMs are an improvement over incumbent methods used to predict rock properties in numerical geodynamic simulations. In terms of pure execution speed, our testing demonstrates that RocMLMs can make predictions up to 10$^3$ times faster than GFEM programs and LUTs (@fig:rocmlm-tradeoffs). We acknowledge that the difference in execution speed between LUTs and RocMLMs is small for low-resolution models (@fig:rocmlm-tradeoffs), which are limited to ≤ 16 mantle compositions at PT resolutions ≥ 17 kbar and 100 ˚C. Although the execution speed of low-resolution LUTs and RocMLMs may be feasible for coupling with numerical geodynamic models, they are not an obvious improvement over simple PT-dependent parameterizations of elastic properties bounded by linear or polynomial approximations of a selective number of important phase transformations. At higher-resolutions, however, our testing demonstrates that RocMLMs can accurately resolve phase transformations for a wide array of bulk compositions ([@fig:earthchem-mixing-arrays; @fig:earthchem-harker-diagram]), at much finer scales (PT resolution ≤ 2 kbar and 12 ˚C, [@fig:image12-PUM-DT; @fig:image12-PUM-NN3]), and at speeds that are feasible for executing at the node-scale during numerical geodynamic simulations (10$^0$–10$^{-1}$, @fig:rocmlm-tradeoffs). We therefore argue that high-resolution RocMLMs overcome all practical limitations for implementing thermodynamically self-consistent elastic properties in numerical geodynamic models.

Execution speed and accuracy alone suggest that all RocMLMs will perform with roughly equivalent outcomes if given enough training data ([@fig:rocmlm-tradeoffs]a). However, our testing reveals an obvious tradeoff between RocMLM performance and accuracy when accounting for compression ratio (i.e., the amount of "learned" information relative to the RocMLM file size). [@fig:rocmlm-tradeoffs]b shows DT and KN algorithms becoming rapidly inefficient compared to NNs as the size of the training dataset increases. This is because NN RocMLMs require relatively little information to make predictions after training (weights and biases for each neuron) compared to DT (tree structure: nodes, splits, and predictions) and KN (entire training dataset with distances). Moreover, accuracy tends to improve monotonically with dataset size for NN, but not for DT or KN algorithms. We therefore argue that NN RocMLMs are the most practical choice for geological applications for three reasons: 1) modeling more rock types only requires adding more training data, 2) adding more training data improves prediction accuracy and does not diminish performance, and 3) further improvements and adaptations to different geological applications are possible by exploring different architectures than the simple NN models we have tested here.

The main limitations of NN RocMLMs are twofold: 1) training is computationally expensive compared to other regression algorithms (@tbl:rocmlm-performance) and 2) simple NN architectures imply smoother gradients in elastic properties than is typically inferred by seismological constraints [@dziewonski1981; @kustowski2008]. We do not consider these limitations critical because training time is independent from RocMLM performance and accuracy and deeper NN architectures can fit large discontinuities in elastic properties with sufficient accuracy (@tbl:rocmlm-performance, [@fig:image12-PUM-NN3]). We note, however, that our testing has been limited to three elastic properties that are mostly P-dependent and are relatively continuous despite a few large discontinuities. In principle, RocMLMs can be trained on any thermodynamic variable output by GFEM programs. However, we have not yet trained RocMLMs on more discrete, discontinuous, and/or highly T-dependent variables, such as modal proportions of minerals, volatile contents, or melt fraction. In cases of highly discontinuous training data, DT or KN algorithms may yield more accurate regressions.

## Geophysical and Thermodynamic Estimates of Elastic Properties {#sec:rocmlm-elastic-properties}

The notable difference in the amount of overlap between RocMLM profiles and PREM with respect to density, Vp, and Vs suggests a considerable disparity between thermodynamic and geophysical estimates of elastic properties of rocks. Uncertainties are on the order of 1–3% for thermodynamic estimates of Vp and Vs [@connolly2002], which we also assume for RocMLM predictions, as RocMLM training data are generated by similar thermodynamic calculations. Uncertainties are not explicitly reported for PREM or STW105 [@dziewonski1981; @kustowski2008], so we conservatively assume that they are on the order of 2–6% (double the thermodynamic uncertainties). While RocMLM and PREM density profiles are indistinguishable within uncertainties of 3% for RocMLMs and 6% for PREM ([@fig:image12-PUM-DT; @fig:image12-PUM-NN1]j), Vp profiles only slightly overlap ([@fig:image12-PUM-DT; @fig:image12-PUM-NN1]k), and Vs profiles are completely non-overlapping ([@fig:image12-PUM-DT; @fig:image12-PUM-NN1]l).

PREM and STW105 are constructed by fitting low-order polynomial models to various seismological observations to constrain density, Vp, and Vs simultaneously [@dziewonski1981; @kustowski2008]. Our thermodynamic calculations, on the other hand, treat Vs independently from Vp and density. In our case, Vp and Vs are computed after @connolly2002:

\begin{align}
	V_p &= \sqrt{\frac{K_b + \frac{4}{3} K_s}{\rho}} \label{eq:seismic-velocities} \\[12pt]
	V_s &= \sqrt{\frac{K_s}{\rho}} \nonumber
\end{align}

\noindent where $K_b$ is the adiabatic bulk modulus, $K_s$ is the shear modulus, and $\rho$ is density. $K_b$ and $\rho$ can be expressed as a function of the GFE for an isostatic system:

\begin{align}
	K_b &= \frac{1}{\beta} = - V \left( \frac{\partial P}{\partial V} \right) = - \frac{\partial G}{\partial P} \left[ \frac{\partial^2 G}{\partial P^2} + \left( \frac{\partial}{\partial P} \frac{\partial G}{\partial T} \right)^2 \middle/ \frac{\partial^2 G}{\partial T^2} \right]^{-1} \\[12pt]
	\rho &= \frac{m}{V} = N \Bigg/ \frac{\partial G}{\partial P} \nonumber
\end{align}

\noindent where $\beta$ is the adiabatic compressibility coefficient, $V$ is volume, $PT$ are pressure and temperature, $m$ is mass, $N$ is the molar mass, $G$ is the GFE of the system, $\partial G / \partial P$ is the molar volume, and  $- \partial G / \partial T$ is the molar entropy. While $K_b$ can be computed directly from thermodynamic data at fixed PT, shear modulus cannot, so an empirical relationship is used [@connolly2002; @riel2022]:

\begin{equation}
	K_s = K_s^0 + T \frac{\partial K_s}{\partial T} + P \frac{\partial K_s}{\partial P} \label{eq:shear-modulus}
\end{equation}

\noindent where $K_s^0$ is the shear modulus at ambient conditions. $K_s^0$ and its PT derivatives are assumed to be constant and compiled from different experimental and theoretical sources [@bailey2000; @bass1995; @helffrich1996; @karki2001; @sinogeikin2000]. The shear modulus $K_s$, especially its P dependence, is the least constrained elastic parameter due to a lack of experimental data [@bina1992]. This implies that the uncertainty in RocMLM Vs predictions is much larger than for Vp and density, which may partially explain why RocMLM Vs profiles do not overlap with PREM.

Athermal (T-independent) elastic parameters of olivine polymorphs calculated from first-principle quantum mechanics find Vs discrepancies on the order of 10–20 % [@karki2001] in the upper mantle (up to 30 GPa) with respect to PREM. Using similar first-principle methods, @karki2001 completely correct large Vs discrepancies with respect to PREM at lower mantle pressures (up to 140 GPa) by introducing thermal dependence to elastic moduli and replacing 10% Mg with Fe in MgSiO$_3$ perovskite and magnesiowüstite. These theoretical models demonstrate that geothermal gradients and compositions of major Mg-rich mantle minerals can have a strong effect on elastic moduli, although the thermal dependence of $K_s$ is still poorly constrained.

@fig:fmelt-analysis shows how the average misfit between RocMLM profiles and PREM changes with bulk mantle composition. As FI decreases by partial melting and the bulk mantle composition becomes more Mg-rich and less dense (@fig:earthchem-mixing-arrays), Vp, and Vs increase in accordance with @eq:seismic-velocities. A change in FI from 1.00–0.76, corresponding to an increase in Mg# from 79-87 (about 8% more Mg-rich), improves the overlap between RocMLM profiles and PREM by 14% and 4% for Vp and Vs, respectively, but deteriorates the overlap for density by 43% (@fig:fmelt-analysis). In other words, changing the mantle bulk composition cannot simultaneously improve the misfit of all RocMLM profiles with respect to PREM. Rather, a correction to the shear modulus (@eq:shear-modulus) seems more appropriate, especially the PT-dependence of $K_s$, since it is the least constrained parameter.

![RMSE vs. Fertility Index diagrams showing the quality of fit between RocMLM depth profiles and PREM. Data points represent synthetic bulk mantle compositions sampled from the top and bottom mixing arrays from @fig:earthchem-mixing-arrays. PUM and DMM compositions are from @tbl:benchmark-samples. A change in FI from 1.00–0.76 in the upper mantle corresponds to an increase in Mg# from 79-87 (about 8% more Mg-rich), which drives elastic properties in opposite directions according to @eq:seismic-velocities.](assets/figs/fmelt-gfem-analysis.png){#fig:fmelt-analysis}

# Conclusions

The structure and dynamics of Earth's upper mantle are largely driven by density contrasts that arise from changes in PT and phase transformations in mantle rocks. As such, numerical geodynamic simulations of mantle convection must account for thermodynamics, which is typically implemented with simple PT-dependent parameterizations of rock properties that are not self-consistent with other thermodynamic variables, namely the GFE. Coupling GFEM programs to numerical geodynamic simulations would allow for self-consistent mantle convection models, but GFEM programs are too slow for this to be feasible. We introduce RocMLMs as an alternative to GFEM programs, and evaluate RocMLM performance and accuracy by testing RocMLM predictions of elastic properties (density, Vp, and Vs) against PREM. Our main findings are as follows:

1. RocMLMs make predictions of elastic properties up to 10$^3$ faster than incumbent methods, making self-consistent mantle convection practical for high-resolution numerical geodynamic models
2. RocMLMs trained with NNs are more robust and efficient compared to RocMLMs trained with other regression algorithms
3. RocMLM predictions are indistinguishable from PREM with respect to density, but show significant discrepancies with respect to Vp and Vs due to large uncertainties in elastic moduli

Based on our results, we argue that RocMLMs can replace GFEM programs for geological applications that require computationally efficient predictions of rock properties. RocMLMs perform remarkably well for dry mantle rocks ranging from lherzolites to harzburgites. However, we have not yet tested RocMLMs on other thermodynamic variables of interest, including modal proportions of minerals, volatile contents, and melt fractions. Future iterations will test RocMLM performance and accuracy on such variables, and extend the training data to include hydrous systems and additional end-member mantle compositions (e.g., pyroxenites and dunites).

# Acknowledgements

This work was supported by the Tremplin-ERC grant LEARNING awarded to Nestor Cerpa by the I-SITE excellence program at the Université de Montpellier. We thank Maurine Montagnat, Fernando Carazo, Nicolas Berlie, and many researchers and students at Géosciences Montpellier for their thoughtful feedback during the development of this work. We gratefully acknowledge additional support from the European Research Council (ERC) under the European Union Horizon 2020 Research and Innovation program grant agreement No. 882450 (ERC RhEoVOLUTION) awarded to Andréa Tommasi.

# Open Research

All data, code, and relevant information for reproducing this work can be found at [https://github.com/buchanankerswell/kerswell_et_al_rocmlm](https://github.com/buchanankerswell/kerswell_et_al_rocmlm), and at [https://doi.org/10.17605/OSF.IO/K23TB](https://doi.org/10.17605/OSF.IO/K23TB), the official Open Science Framework data repository (Kerswell et al., 2024). All code is MIT Licensed and free for use and distribution (see license details). Reference Earth models PREM and STW105 are freely available from the Incorporated Research Institutions for Seismology Earth Model Collaboration [IRIS EMC\, doi: 10.17611/DP/EMC.1, @trabant2012].

# References

<div id="refs"></div>

\cleardoublepage

# Appendix

## RocMLM Training Data {#sec:rocmlm-training-data}

![Harker Diagrams vs. SIO2 (in wt.%) showing the distribution of peridotite samples from Earthchem.org (colored contours and pie chart). PUM (black square) and DMM (white square) are commonly-referenced bulk mantle compositions (@tbl:benchmark-samples), while PSUM (black circle) and DSUM (white circle) define a mixing array used to generate RocMLM training data (black datapoints).](assets/figs/earthchem-harker-diagram.png){#fig:earthchem-harker-diagram}

## RocMLM Regression Algorithms {#sec:rocmlm-regression-algorithms}

![PT diagrams showing density (left column, g/cm3), Vp (middle column, km/s), and Vs (right column, km/s) predictions from a Perple_X model with a PUM bulk composition (a–c), a Decision Tree RocMLM (d–f), and differences between Perple_X and DT (g–i). Depth profiles (j–l) compare Perple_X and DT predictions extracted along average geotherms for a Proterozoic craton and mid-ocean ridge (solid and dashed white lines, respectively) with reference Earth models PREM [solid black line, @dziewonski1981] and STW105 [dotted black line, @kustowski2008]. Other legend details are the same as @fig:image12-PUM-DT.](assets/figs/image12-PUM-DT.png)

![PT diagrams showing density (left column, g/cm3), Vp (middle column, km/s), and Vs (right column, km/s) predictions from a Perple_X model with a PUM bulk composition (a–c), a k-Neighbors RocMLM (d–f), and differences between Perple_X and KN (g–i). Depth profiles (j–l) compare Perple_X and KN predictions extracted along average geotherms for a Proterozoic craton and mid-ocean ridge (solid and dashed white lines, respectively) with reference Earth models PREM [solid black line, @dziewonski1981] and STW105 [dotted black line, @kustowski2008]. Other legend details are the same as @fig:image12-PUM-DT.](assets/figs/image12-PUM-KN.png){#fig:image12-PUM-KN}

![PT diagrams showing density (left column, g/cm3), Vp (middle column, km/s), and Vs (right column, km/s) predictions from a Perple_X model with a PUM bulk composition (a–c), a single-layer Neural Network RocMLM (d–f), and differences between Perple_X and NN1 (g–i). Depth profiles (j–l) compare Perple_X and NN1 predictions extracted along average geotherms for a Proterozoic craton and mid-ocean ridge (solid and dashed white lines, respectively) with reference Earth models PREM [solid black line, @dziewonski1981] and STW105 [dotted black line, @kustowski2008]. Other legend details are the same as @fig:image12-PUM-DT.](assets/figs/image12-PUM-NN1.png)

![PT diagrams showing density (left column, g/cm3), Vp (middle column, km/s), and Vs (right column, km/s) predictions from a Perple_X model with a PUM bulk composition (a–c), a two-layer Neural Network RocMLM (d–f), and differences between Perple_X and NN2 (g–i). Depth profiles (j–l) compare Perple_X and NN2 predictions extracted along average geotherms for a Proterozoic craton and mid-ocean ridge (solid and dashed white lines, respectively) with reference Earth models PREM [solid black line, @dziewonski1981] and STW105 [dotted black line, @kustowski2008]. Other legend details are the same as @fig:image12-PUM-DT.](assets/figs/image12-PUM-NN2.png){#fig:image12-PUM-NN2}

![PT diagrams showing density (left column, g/cm3), Vp (middle column, km/s), and Vs (right column, km/s) predictions from a Perple_X model with a PUM bulk composition (a–c), a three-layer Neural Network RocMLM (d–f), and differences between Perple_X and NN3 (g–i). Depth profiles (j–l) compare Perple_X and NN3 predictions extracted along average geotherms for a Proterozoic craton and mid-ocean ridge (solid and dashed white lines, respectively) with reference Earth models PREM [solid black line, @dziewonski1981] and STW105 [dotted black line, @kustowski2008]. Other legend details are the same as @fig:image12-PUM-DT.](assets/figs/image12-PUM-NN3.png){#fig:image12-PUM-NN3}
