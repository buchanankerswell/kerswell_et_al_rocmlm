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

The following sections describe our methods for building, training, and evaluating RocMLMs with three main objectives in mind. First, define the size and scope of RocMLM training data to ensure widespread applicability of RocMLMs to the upper mantle (@sec:training-data-design). Second, define a generalized approach to generating RocMLM training data to ensure widespread reproducibility with any GFEM program (e.g., MAGEMin, Perple_X, and others, @sec:generate-training-data). Third, define a set of RocMLMs input features that are routinely computed during geodynamic simulations to ensure widespread applicability of RocMLMs to various geodynamic codes. We also describe our methods for determining the best RocMLM algorithm by cross-validation (@sec:train-evaluate-rocmlms).

## RocMLM Training Dataset Design {#sec:training-data-design}

### Pressure-Temperature Conditions {#sec:pt-conditions}

High-pressure experiments constrain the reaction $\text{olivine} \xrightarrow{\text{410 km}} \text{ringwoodite}$ between 14.0 ± 1.0 GPa and 1600 ± 400 K with Clapeyron slopes between 2.4x10$^{-3}$ ± 1.4x10$^{-3}$ GPa/K  [@akaogi1989; @katsura1989; @morishima1994; @li2019]. Likewise, the reaction $\text{ringwoodite} \xrightarrow{\text{660 km}} \text{bridgmanite} + \text{ferropericlase}$ is constrained between 24.0 ± 1.5 GPa and 1600 ± 400 K with negative Clapeyron slopes between -2.0x10$^{-3}$ ± 1.6x10$^{-3}$ GPa/K  [@akaogi2007; @bina1994; @litasov2005; @katsura2003; @ito1990; @ito1982; @ito1989a; @ito1989b; @hirose2002; @ishii2018]. We therefore compute RocMLM training data within a rectangular PT region bound between 1.0–28.0 GPa and 773–2273 K to encompass expected conditions for the entire upper mantle---from the Moho at 35 km to just beyond the base of the MTZ at 660 km ([@fig:training-dataset-design]).

@fig:training-dataset-design shows that our training dataset includes PT conditions that are not expected to exist in the upper mantle, nor typically modeled during geodynamic simulations [e.g., very cold conditions below thermal gradients of $\leq$ 5 K/km, @maruyama1996; @syracuse2010]. In this case, a rectangular training dataset might be considered impractical with respect to training efficiency (unnecessarily large amount of training data) and accuracy (outside the bounds of calibrated thermodynamic data). However, initial sensitivity tests showed comparable RocMLM performance irrespective of the range of PT conditions used to generate RocMLM training data (grey vs. blue shaded regions in @fig:training-dataset-design). Thus, we adopted a regular rectangular training dataset design because it is computationally convenient and does not deteriorate RocMLM accuracy.

![PT diagram showing the range of conditions for generating RocMLM training data (grey region) compared to the range of possible upper mantle conditions beneath continents (blue shaded region). The solid and dashed black lines are average geothermal gradients for Proterozoic and Archean continents, respectively. The dotted black lines are 0.5 K/km mantle adiabats placed at arbitrary mantle potential temperatures of 673 K and 1773 K, representing hypothetical lower and upper bounds for expected mantle PT conditions. Phase boundaries for the 410 km and 660 km discontinuities (colored lines) are after @li2019.](assets/figs/training-dataset-design.png){#fig:training-dataset-design}

### Bulk Mantle Compositions {#sec:bulk-rock-compositions}

In principle, the bulk composition of the upper mantle can be estimated from chemical analyses of high-pressure-high-temperature melting experiments and/or mantle-derived intrusive and extrusive igneous rocks [e.g., @allegre1984; @green1979; @ringwood1962; @jagoutz1979; @sun1982; @ringwood1991; @palme2003; @stracke2021]. @tbl:benchmark-samples gives four examples of bulk mantle compositions with varying degrees of depletion by partial melting. Two examples are commonly-referenced in the literature, Primitive Upper Mantle [PUM, @sun1989] and Depleted MORB Mantle [DMM, @workman2005], while the other two examples, Primitive Synthetic Upper Mantle (PSUM) and Depleted Synthetic Upper Mantle (DSUM), were derived in this study. Below we describe our statistical approach to deriving PSUM and DSUM from chemical analyses of thousands of peridotite samples.

{{ benchmark-samples.md }}

Bulk chemical analyses of peridotite samples were downloaded using the [Earthchem.org](https://earthchem.org) Search Portal with a single search criterion: *"set sample type > igneous rocks > names from Earthchem categories > igneous-plutonic-ultramafic"*, which queried 19791 samples. Samples lacking analyses for SiO$_2$, MgO, Al$_2$O$_3$, or CaO were excluded from the dataset to focus on bulk rock compositions. All unknown rock types, chromitites, limburgites, wehrites, undifferentiated peridotites, dunites, and pyroxenite samples were also excluded from the dataset to focus on samples that were most likely the products of partial melting, rather than the products of fractional crystallization [@bowen1915], fluid-rock interactions [@elthon1992; @kelemen1995], or excess olivine crystallization [@niu1997a]. The data were grouped according to the remaining rock types (lherzolite and harzburgite) and outliers were removed from each group using a 1.5 inter-quartile range threshold applied to each chemical component. Cr and Ni measured as trace elements (ppm) were converted to Cr$_2$O$_3$ and NiO (wt.%) and all Fe oxides were converted to Fe$_2$O$_3$T. Total oxides were then checked against H$_2$O, CO$_2$, and LOI to determine if chemical analyses were performed before or after ignition. Analyses with total oxides summing to $\leq$ 97 % or $\geq$ 103 % were considered erroneous, or otherwise low-quality, and excluded from the dataset. Finally, all analyses were normalized to a volatile-free basis before converting Fe$_2$O$_3$T to FeOT. The final dataset contains 3111 chemical analyses of classified peridotite samples (@tbl:earthchem-counts).

PCA was applied to the standardized peridotite dataset to reduce the chemical system from 10 oxide components to two principal components (PC1 and PC2). PCA requires non-missing data, however, so samples were first arranged by MgO and SiO$_2$ and a k-nearest neighbor algorithm was applied to impute missing oxide analyses, which mainly affected the K$_2$O and Na$_2$O components (see @tbl:earthchem-counts for missing analyses counts). After PCA was applied, a mixing line was defined by connecting the lherzolite and harzburgite sample centroids (i.e., the median values for PC1 and PC2 for each group). The lherzolite-harzburgite mixing line was then extended until reaching the approximate location of the most fertile (Al$_2$O$_3$-CaO-TiO$_2$-rich) and most refractory (MgO-rich, SiO$_2$-poor) peridotite samples (@fig:earthchem-mixing-arrays). The mixing line shown in [@fig:earthchem-mixing-arrays]b approximates a wide array of hypothetical mantle compositions derived from the natural rock record---providing a basis for sampling synthetic bulk mantle compositions used to generate RocMLM training data.

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

![PCA diagrams showing (a) the distribution of standardized peridotite data queried from Earthchem.org and (b) mixing array between endmember mantle compositions PSUM and DSUM. Thick black arrows in (a) indicate the strongest PCA loading vectors. The bold dashed line in (b) indicates the mixing array between PSUM and DSUM, while the bold solid lines indicate the upper and lower boundaries for random sampling. Fertility Index is calculated with a modal fractional melting model (@eq:melt-fraction).](assets/figs/earthchem-mixing-arrays.png){#fig:earthchem-mixing-arrays}

Three implicit assumptions in @eq:shaw-melting are important to note, however, because they impact FI, and thus ultimately impact RocMLM predictions: 1) melt is instantaneously removed from the source region, 2) $D_0$ is constant, and 3) minerals melt in the same proportions that they exist in the source rock. While these assumptions strongly depend on the progression of melting reactions and mantle conditions [@langmuir1992; @niu1997b; @brown2016], @eq:melt-fraction predicts that DMM is produced by 2.2 % melt extraction from PUM (@tbl:benchmark-samples), which is consistent with the degree of depletion indicated by trace element patterns and mass balance constraints [2-3 % melt removal from PUM, @workman2005]. We therefore consider FI an adequate first-order proxy for bulk mantle composition.

## Generating RocMLM Training Data {#sec:generate-training-data}

We use the GFEM program Perple_X [version 7.0.9, @connolly2009] to generate RocMLM training data using PT conditions as described in @sec:pt-conditions (@fig:training-dataset-design) and synthetic bulk mantle compositions as described in @sec:bulk-rock-compositions. All Perple_X models include equations of state for pure phases: quartz, coesite, stishovite, kyanite, corundum, and rutile, and solution phases: feldspar, spinel, garnet, clinopyroxene, orthopyroxene, olivine, ilmenite, wadsleyite, ringwoodite, wuestite, perovskite, ferropericlase, high-pressure clinopyroxene, and silicate melt. Thermodynamic data (tc-ds634.txt) are from @holland2018, with updates from @tomlinson2021 and @holland2022, which are specifically formulated for a wide array of igneous rocks and melt compositions. @holland2018 itself is an extension of the foundational database from @holland2011, which is calibrated up to 300 GPa and 2000 ˚C, making tc-ds634.txt the most appropriate choice for the upper mantle. Once configured, Perple_X generates RocMLM training data (rock properties) by minimizing the total GFE of a multicomponent multiphase thermodynamic system at a fixed PT point, defined by the following equation [@gibbs1878; @spear1993]:

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

## Training RocMLMs and Evaluating Performance {#sec:train-evaluate-rocmlms}

RocMLM training data were preprocessed using the following procedure. First, two-dimensional grids of rock properties (density, Vp, and Vs) calculated by Perple_X were stacked into three-dimensional arrays, $Z$ = $(z_{1,1,1}, \ldots, z_{n,w,w})$, where $w$ = 128 is the resolution of the PT grid and $n$ = 128 is the number of random synthetic bulk mantle compositions represented by a FI value. $Z$ was flattened and separated into arrays of training features (PT and FI), $X$ = $(x_{1,1,1}, \ldots, x_{v,v,v})$, and training targets (density, Vp, Vs), $y$ = $(y_{1,1,1}, \ldots, y_{v,v,v})$, where $v$ = $n \cdot w^2$ = 128$^3$ is the total number of training examples. $X$ and $y$ were then scaled using "z-score normalization" to avoid strong biasing of RocMLMs by training data with mixed units (GPa, K, g/cm$^3$, and km/s):

\begin{align}
	\overline{X} &= \frac{X - \mu_X}{\sigma_X} \label{eq:standard-scaler} \\
	\overline{y} &= \frac{y - \mu_y}{\sigma_y} \nonumber
\end{align}

\noindent where $\overline{X}$ and $\overline{y}$ are the scaled feature and target arrays, respectively, and $\mu_X$, $\mu_y$, $\sigma_X$, and $\sigma_y$ are the means and standard deviations of the feature and training examples, respectively. The preprocessed training data were then fit with eight different non-linear regression algorithms from the scikit-learn python library [@scikit2011]. Each regression algorithm was tuned with a grid search approach, where a performance score (RMSE) was evaluated over all hyperparameter combinations relevant to the particular regression algorithm (@tbl:rocmlm-config). The set of hyperparameters that produced the best score was used to train the RocMLM.

{{ rocmlm-config.md }}

During RocMLM training, performance metrics were evaluated using a k-fold cross-validation technique to determine how well each RocMLM fit the training data. The training data were shuffled to reduce the impact of inherent ordering in the data and partitioned into $k$ = 8 non-overlapping folds of training examples. Cross-validation then proceeded with $k$ iterations, where in each iteration, models were fit to training examples from $k$-1 folds and performance (RMSE) was evaluated on the remaining fold. After all iterations completed, performance was evaluated independently on a separate set of input features (validation set) not seen by the RocMLM during training to provide a measure of generalizability to new data.

# Results {#sec:results}

## RocMLM Precision and Accuracy {#sec:rocmlms-accuracy}

RocMLM precision is practically limitless, since the model inputs and outputs are double precision (float64) numbers that are accurate up to 16 decimal places (i.e., much higher precision than is required to define P, T, and FI), and is therefore not a main concern. RocMLM accuracy, on the other hand, is both critical for geological applications (e.g., simulating mantle convection) and dependent on the regression algorithm used for RocMLM training. Here we define RocMLM accuracy in two ways: by the 1) residuals between RocMLM predictions and the training dataset, and 2) residuals between RocMLM predictions and reference Earth models derived from seismological data. The former definition quantifies how well training data were mapped to RocMLMs by regression (internal accuracy), while the latter quantifies how RocMLM predictions compare to the inferred physical properties of Earth's upper mantle (external accuracy). Two examples of RocMLMs are given below that demonstrate the relative accuracies of different regression algorithms (see all regression algorithms in @sec:training-data-mappings). The relative performance in terms of accuracy, prediction times, and efficiency are presented in @sec:rocmlm-performance-and-efficiency.

@fig:image12-PUM-DT compares PT distributions of density, Vp, and Vs predicted by GFEM (Perple_X) and a RocMLM trained with a Decision Tree (DT) algorithm. DT RocMLM predictions are indistinguishable from Perple_X, indicating a nearly-perfect regression of the training data (high internal accuracy). The largest residuals between Perple_X and the DT RocMLM correspond to sharp gradients in material properties at important phase transitions, including the onset of partial melting at HT-LP, the $\text{olivine} \xrightarrow{\text{410 km}} \text{ringwoodite}$ transition, and the $\text{ringwoodite} \xrightarrow{\text{660 km}} \text{bridgmanite} + \text{ferropericlase}$ transition.

![PT-surface diagrams showing the distributions of density (top row, in g/cm$^3$), Vp (middle row, in km/s), and Vs (bottom row, in km/s) predicted by Perple_X (left column) and a RocMLM trained on the Perple_X surface using a three-layer Neural Network algorithm. Residuals (right column) show the point-by-point differences between Perple_X and RocMLM surfaces (units are the same across rows). All models are based on a PUM bulk composition (@tbl:benchmark-samples).](assets/figs/image12-PUM-DT.png){#fig:image12-PUM-DT}

@fig:image12-PUM-NN3 shows an example of a RocMLM trained with a three-layer Neural Network algorithm (NN3, see other regression algorithms in @sec:training-data-mappings). Density, Vp, and Vs predictions are nearly indistinguishable between Perple_X and the NN3 RocMLM, indicating a nearly-perfect regression of the training data (i.e., high internal accuracy). RocMLM density profiles extracted along an average continental geotherm are in good agreement with reference Earth models [@dziewonski1981; @kustowski2008], showing high correlation (R$^2$: 0.96) and low error rates (RMSE: 0.08 g/m$^3$), indicating good consistency with inversions of seismological observations (i.e., high external accuracy; [@fig:image12-PUM-NN3]a–c). On the other hand, RocMLM Vp and Vs profiles are inconsistent with reference Earth models, showing poor correlation (R$^2$: ≤ 0.47), high error rates (RMSE: ≥ 0.82 g/m$^3$), and a systematic shift towards lower velocities ([@fig:image12-PUM-NN3]d–i).

![PT diagrams showing the distributions of density, Vp, and Vs predicted by Perple_X (left column) and a RocMLM trained with a three-layer Neural Network algorithm (middle column). Depth profiles (right column) comparing Perple_X and RocMLM predictions with reference Earth models derived from seismological data. Perple_X and RocMLM profiles were extracted along an average continental geotherm (white line) and assume a PUM bulk mantle composition (@tbl:benchmark-samples). Reference Earth models are the Preliminary Reference Earth Model (PREM, solid black line) from @dziewonski1981 and STW105 (dotted black line) from @kustowski2008. Performance metrics R$^2$ and RMSE show the correlations and accuracies of the RocMLM with respect to PREM.](assets/figs/image12-PUM-NN3.png){#fig:image12-PUM-NN3}

## RocMLM Performance and Efficiency {#sec:rocmlm-performance-and-efficiency}

PUM and DMM mantle compositions were used for benchmarking MAGEMin and Perple_X efficiency (@tbl:benchmark-samples). [@fig:rocmlm-tradeoffs] shows the computation times with respect to various PT grid resolutions (8x8, 16x16, 32x32, 64x64, 128x128). All computations were made on a Macbook Pro (2022; M2 chip) with macOS 13.4 and Python 3.11.4.

![caption](assets/figs/rocmlm-tradeoffs.png){#fig:rocmlm-tradeoffs}

# Discussion

\clearpage

# Acknowledgements

# Open Research

All data, code, and relevant information for reproducing this work can be found at [https://github.com/buchanankerswell/kerswell_et_al_rocmlm](https://github.com/buchanankerswell/kerswell_et_al_rocmlm), and at [https://doi.org/10.17605/OSF.IO/K23TB](https://doi.org/10.17605/OSF.IO/K23TB), the official Open Science Framework data repository (Kerswell et al., 2024). All code is MIT Licensed and free for use and distribution (see license details). Reference Earth models PREM and STW105 are freely available from the Incorporated Research Institutions for Seismology Earth Model Collaboration [IRIS EMC\, doi: 10.17611/DP/EMC.1, @trabant2012].

# References

<div id="refs"></div>

\cleardoublepage

# Appendix

## RocMLM Training

![Harker Diagrams vs. SIO2 (in wt.%) showing the distribution of peridotite samples from Earthchem.org (colored contours and pie chart). PUM (black square) and DMM (white square) are commonly-referenced bulk mantle compositions (@tbl:benchmark-samples), while PSUM (black circle) and DSUM (white circle) define a mixing array used to generate RocMLM training data (black datapoints).](assets/figs/earthchem-harker-diagram.png){#fig:earthchem-harker-diagram}

## RocMLM Performance and Efficiency Tradeoffs

{{ rocmlm-performance.md }}

## Mapping Training Data to RocMLMs by Different Regression Algorithms {#sec:training-data-mappings}

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Decision Tree models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/surf-PUM-DT-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by K Nearest models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/surf-PUM-KN-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Neural Network 1L models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/surf-PUM-NN1-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Neural Network 3L models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/surf-PUM-NN3-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Decision Tree models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/image-PUM-DT-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by K Nearest models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/image-PUM-KN-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Neural Network 1L models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/image-PUM-NN1-rho.png)

![PT-density diagrams (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples) and (c, d) inferred by Neural Network 3L models trained on (a, b), respectively. (e, f) PT-error diagrams (in percent) showing the normalized difference between (a, b) and (c, d), respectively.](assets/figs/image-PUM-NN3-rho.png)
