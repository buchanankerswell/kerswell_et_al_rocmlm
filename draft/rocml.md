# Introduction

The dominant mineral phases in Earth's upper mantle are olivine, ringwoodite, bridgmanite, and ferropericlase [@ringwood1975; @ringwood1991], comprising up to 60–90% of the mantle's volume [e.g., @stixrude2012]. These Mg-Fe-rich phases form by a series of discrete reactions (@eq:mantle-transitions) that define mantle transition zones (MTZs) near 410 km and 660 km depths beneath Earth's surface. MTZs are characterized by relatively sharp boundaries with contrasting physical properties [e.g, density and elasticity, @dziewonski1981; @ita1992] that strongly impact mantle convection, melting, and plate tectonics [@fukao2001; @ringwood1991; @schubert1975; @kuritani2019; @karato2001; @wang2015; @jenkins2016].

\begin{align}
	\text{olivine} \xrightarrow{\text{410 km}} \text{ringwoodite} &\xrightarrow{\text{660 km}} \text{bridgmanite} + \text{ferropericlase} \label{eq:mantle-transitions} \\
	\text{(Mg,Fe)}_{2}\text{SiO}_{4} \xrightarrow{\text{410 km}} \text{(Mg,Fe)}_{2}\text{SiO}_{4} &\xrightarrow{\text{660 km}} \text{(Mg,Fe)}\text{SiO}_{3} + \text{(Mg,Fe)}\text{O} \nonumber
\end{align}

Although the physio-chemical nature of MTZs remains under vigorous investigation [@goes2022; @pearson2014; @yoshino2008; @waszek2021; @kiseeva2018; @fei2017; @zhou2022], modeling mantle convection within and across the MTZs is possible with numerical geodynamic simulations that implement pressure-temperature (PT)-dependent phase changes (e.g., @eq:mantle-transitions). This approach has generated many important hypotheses implicating MTZs as critical features impacting tectonic plate motions and water cycling in the deep Earth [e.g., @agrusta2017; @li2019; @torii2007; @yang2020]. However, the tendency to assume fixed mantle compositions---i.e., neglecting chemical fractionation from fluid-rock interactions and partial melting---limit such numerical experiments to a rough zeroth-order approximation of true mantle flow.

More comprehensive numerical models of mantle convection would account for chemical fractionation by implementing PT-composition-time (PTXt)-dependent phase changes. This is currently intractable, however, because Gibbs Free Energy Minimization (GFEMs) programs [e.g., @connolly2009; @riel2022] used to calculate PTX-dependent phase relations remain too slow to apply at each node, at each timestep, in a geodynamic simulation. While recent parallelized GFEM programs have increased efficiency dramatically [10$^4$--10$^2$ ms at a fixed PTX, @riel2022], computing phase relations iteratively during geodynamic simulations requires GFEM efficiency on the order of $\leq$ 10$^1$--10$^{-1}$ ms to be feasible. A rate of improvement from 10$^4$--10$^2$ to 10$^1$--10$^{-1}$ ms seems unlikely within the current GFEM paradigm, however, and applying parallelization across many CPU/GPU cores can be inaccessible.

This study demonstrates an alternative approach to inferring phase relations and rock properties using pre-trained machine learning models (referred to as RocMLMs). We begin by detailing the size (PT range), scope (compositional range), and methods for building RocMLM training datasets. We then train and evaluate performance metrics for each RocMLM by k-fold cross-validation---demonstrating up to 10$^4$ faster RocMLM performance vs. GFEM programs. Our results imply that real-time inference of PTXt-dependent phase changes in geodynamic simulations is feasible with RocMLMs. Finally, we discuss the validity and limitations of RocMLMs through comparisons with empirically-derived geophysical datasets, and how implementing RocMLMs in geodynamic simulations can enable new models of global plate tectonics, deep water cycling, and mantle melting at MTZs.

# Methods {#sec:methods}

The following sections describe our methods for building RocMLM training datasets, training RocMLMs, and evaluating RocMLM performance. Our objectives are threefold. First, design the size and scope (i.e., the PT range and compositional range) of RocMLM training data to ensure widespread applicability of RocMLMs to geodynamic problems within the upper mantle (@sec:design-training-data). Second, compute RocMLM training data and compare the performance and accuracy of GFEM programs MAGEMin and Perple_X (@sec:build-training-data). Third, train various RocMLMs and determine the best models through cross-validation (@sec:training-rocmls). Comparisons among MAGEMin, Perple_X, and RocMLMs results are then detailed in @sec:results.

## RocMLM Training Dataset Design {#sec:design-training-data}

### PT Conditions

High-pressure experiments constrain the reaction $\text{olivine} \xrightarrow{\text{410 km}} \text{ringwoodite}$ between 14.0 ± 1.0 GPa and 1600 ± 400 K with Clapeyron slopes between 2.4x10$^{-3}$ ± 1.4x10$^{-3}$ GPa/K  [@akaogi1989; @katsura1989; @morishima1994; @li2019]. Likewise, the reaction $\text{ringwoodite} \xrightarrow{\text{660 km}} \text{bridgmanite} + \text{ferropericlase}$ is constrained between 24.0 ± 1.5 GPa and 1600 ± 400 K with negative Clapeyron slopes between -2.0x10$^{-3}$ ± 1.6x10$^{-3}$ GPa/K  [@akaogi2007; @bina1994; @litasov2005; @katsura2003; @ito1990; @ito1982; @ito1989a; @ito1989b; @hirose2002; @ishii2018]. We therefore compute RocMLM training data within a rectangular PT region bound between 1.0–28.0 GPa and 773–2273 K to encompass expected conditions for the entire upper mantle---from the Moho at 35 km to just beyond the base of the MTZ at 660 km ([@fig:training-dataset-design]).

@fig:training-dataset-design shows that our training dataset (grey shaded region in @fig:training-dataset-design) includes PT conditions that are not expected to exist in the upper mantle, nor typically modeled during geodynamic simulations [e.g., very cold conditions below thermal gradients of $\leq$ 5 K/km, @maruyama1996; @syracuse2010]. Thus, training RocMLMs on a regular rectangular dataset can be considered impractical with respect to efficiency (unnecessarily large amount of training data) and accuracy (outside the bounds of calibrated thermodynamic data). For example, phase relations can be spurious at very low-temperature and high-pressure (e.g., at 20 GPa and 1000 K) or have high degrees of partial melt at very low-pressure and high-temperature (e.g., at 5 GPa and 2000 K).

![PT diagram showing the ranges of PT conditions for RocMLM training data (grey region) and the upper mantle (blue shaded region). Geotherm 1 (solid black line) and geotherm 2 (dashed black line) assume mantle potential temperatures of 273 K and 1773 K with 1 K/km and 0.5 K/km thermal gradients, respectively. Phase boundaries for the MTZs (colored lines) are after @li2019.](assets/figs/training-dataset-design.png){#fig:training-dataset-design}

On the other hand, a regular rectangular training dataset design is computationally more convenient to build, validate, and benchmark. Our initial testing showed that RocMLMs trained on a regular rectangular PT array have comparable accuracy to RocMLMs trained on an irregular PT array (blue shaded region in @fig:training-dataset-design). This implies that RocMLM predictions should be no less accurate than the underlying GFEM models they are derived from---irrespective of the range of PT conditions. For example, if the GFEM models are spurious at 20 GPa and 1000 K, the RocMLM will also be spurious at similar conditions. In contrast, wherever the GFEM models are accurate, the RocMLM is comparably accurate. Thus, we adopted a regular rectangular training dataset design because it is computationally convenient and does not deteriorate RocMLM accuracy.

### Solution Phase Models and Thermodynamic Data {#sec:thermodynamic-data}

Thermodynamic data for computing RocMLM training datasets are based on end-member thermodynamic properties from @holland2018, with updates from @tomlinson2021 and @holland2022. The database (tc-ds634.txt from [hpxeosandthermocalc.org](https://hpxeosandthermocalc.org)) is specifically formulated for calculating phase relations for a wide array of igneous rocks and melt compositions. @holland2018 itself is an extension of the foundational database from @holland2011, which is calibrated up to 300 GPa and 2000 ˚C. Thus, the dataset tc-ds634.txt is the most appropriate database available for estimating phase relations within the entire upper mantle ([@fig:training-dataset-design]).

All MAGEMin and Perple_X models are computed with equations of state for pure phases: quartz, coesite, stishovite, kyanite, corundum, and rutile, and solution phases: feldspar, spinel, garnet, clinopyroxene, orthopyroxene, olivine, ilmenite, and silicate melt. The same solution models from @holland2018 are applied to MAGEMin and Perple_X. The notable exceptions are the ternary feldspar models, which differ for MAGEMin [after @holland2022] and Perple_X [after @fuhrman1988].

The most fundamental practical difference between MAGEMin and Perple_X is that MAGEMin is designed to use a limited set of predefined options for chemical systems and solution models that are not intended to be modified by the typical user (Riel, [personal communications](https://github.com/ComputationalThermodynamics/MAGEMin/issues/61), July 11, 2023). For example, MAGEMin does not natively include solution models for wadsleyite, ringwoodite, wuestite, perovskite, ferropericlase, and high-pressure clinopyroxene with the @holland2018 thermodynamic database---which are critical for predicting phase relations up to 28 GPa in our case. We therefore use Perple_X to generate RocMLM training data because the analogous MAGEMin models lack key solution models and therefore introduce untenable errors beyond approximately 3-5 GPa.

### Bulk Mantle Compositions {#sec:bulk-rock-compositions}

In principle, the bulk composition of the upper mantle can be estimated from chemical analyses of high-pressure-high-temperature melting experiments and/or mantle-derived intrusive and extrusive igneous rocks [e.g., @allegre1984; @green1979; @ringwood1962; @jagoutz1979; @sun1982; @ringwood1991; @palme2003; @stracke2021]. @tbl:benchmark-samples gives four examples of hypothetical bulk mantle compositions with varying degrees of differentiation by partial melting. Two examples are commonly-referenced in the literature, Primitive Upper Mantle [PUM, @sun1989] and Depleted MORB Mantle [DMM, @workman2005], while the other two examples, Primitive Synthetic Upper Mantle (PSUM) and Depleted Synthetic Upper Mantle (DSUM), were derived in this study. Below we describe our approach for deriving PSUM and DSUM, which involved two steps: 1) downloading and standardizing chemical analyses of peridotite samples from [Earthchem.org](https://earthchem.org), and 2) applying principal component analysis (PCA) and defining endmember mantle compositions in PCA space.

{{ benchmark-samples.md }}

Bulk chemical analyses of peridotite samples were downloaded using the Earthchem Search Portal with a single search criterion: "`set sample type > igneous rocks > names from Earthchem categories > igneous/plutonic/ultramafic`", which queried 19791 samples. Samples lacking analyses for SiO$_2$, MgO, Al$_2$O$_3$, or CaO were excluded from the dataset, as well as all unknown rock types, chromitites, limburgites, wehrites, undifferentiated peridotites, and pyroxenite samples. The data were grouped according to the remaining rock types (lherzolite, harzburgite, and dunite) and outliers were removed from each group using a 1.5 inter-quartile range threshold. Cr and Ni measured as trace elements (ppm) were converted to Cr$_2$O$_3$ and NiO (wt.%) and all Fe oxides were converted to Fe$_2$O$_3$T. Total oxides were checked against H$_2$O, CO$_2$, and LOI (if reported) to determine if chemical analyses were performed before or after ignition. Analyses with total oxides summing to $\leq$ 97 % or $\geq$ 103 % were excluded from the dataset. Finally, all analyses were normalized to a volatile-free basis before converting Fe$_2$O$_3$T to FeOT. The final dataset contains 3707 chemical analyses of peridotite samples (@tbl:earthchem-counts and @fig:earthchem-harker-diagram).

PCA was applied to the standardized Earthchem dataset to reduce the chemical system to two principal components (PC1 and PC2). PCA requires non-missing data, however, so samples were first arranged by MgO and SiO2 and a k-nearest neighbor algorithm was implied to impute missing oxide analyses (see @tbl:earthchem-counts for missing data counts). Two mixing lines were then defined by connecting the centroids of lehrzolite, harzburgite, and dunite samples (i.e., the median values for PC1 and PC2 for each group). The lehrzolite-harzburgite and dunite-harzburgite mixing lines were extended until reaching the approximate location of the most fertile (Al$_2$O$_3$-CaO-TiO$_2$-rich) and most refractory (MgO-rich, SiO2-poor) Earthchem samples, respectively (@fig:earthchem-mixing-arrays). The mixing lines shown in @fig:earthchem-mixing-arrays therefore approximate a wide array of hypothetical mantle compositions derived from the natural rock record and provide a basis for sampling synthetic bulk rock compositions for RocMLM training data.

{{ earthchem-counts.md }}

### Melt Fractions {#sec:melt-fractions}

In order to reduce the dimensionality of the training dataset from eleven dimensions (9 oxide components + PT), to three dimensions (1 compositional dimension + PT), we estimated the melt fraction that might have produced the bulk mantle compositions used for generating RocMLM training data. Here we assumed that all synthetic samples were derived from a PSUM source by modal fractional melting [after @shaw1970]:

\begin{equation}
	\frac{C_{\text{TiO}_2}^s}{C_{\text{TiO}_2}^0} = R = (1 - F)^{\frac{1}{D_0} - 1} \label{eq:shaw-melting}
\end{equation}

\noindent where $R$ is the ratio of the TiO$_2$ concentration of the sample to the initial PSUM source (@tbl:benchmark-samples), $F$ is the fraction of melt, and $D_0$ = 0.05 is the bulk distribution coefficient for TiO$_2$ in peridotite [after @brown2016]. Melt fraction is calculated by rearranging @eq:shaw-melting:

\begin{equation}
	F = 1 - R^{\frac{1}{(\frac{1}{D_0}) - 1}} \label{eq:melt-fraction}
\end{equation}

![Harker Diagrams vs. SIO2 (wt.%) showing the distribution of Earthchem data (colored contours) and synthetic compositions used for RocMLM training data (black datapoints). PSUM (black circle), DSUM (white circle), PUM (black square), and DMM (white square) are hypothetical endmember mantle compositions (see @tbl:benchmark-samples).](assets/figs/earthchem-harker-diagram.png){#fig:earthchem-harker-diagram}

Training RocMLMs on $F$ instead of nine oxide components is ideal for two reasons: 1) it greatly increases computational efficiency and 2) unlike oxide components, $F$ is routinely implemented in numerical geodynamic simulations [e.g., @cerpa2019; @gerya2003; @sizova2010; @li2019; @yang2020; @kelley2010]. Thus, @eq:shaw-melting offers a generalized solution for coupling RocMLMs to geodynamic codes. Three critical assumptions implicit to @eq:shaw-melting are important to note, however, because they impact $F$, and thus ultimately impact RocMLM training and predictions: 1) melt is instantaneously removed from the source region, 2) $D_0$ is constant, and 3) minerals melt in the same proportions that they exist in the source rock. While these assumptions are unlikely to occur in nature [@shaw1970], @eq:melt-fraction predicts that DMM is produced by 2.4 % melt removal from PSUM (@tbl:benchmark-samples), which is consistent with the degree of depletion indicated by others [2-3 % melt removal from PUM, @workman2005].

![PCA diagrams showing the distribution of Earthchem data (left panel) and mixing arrays between endmember mantle compositions PSUM and DSUM (right panel). Thick black arrows (left panel) indicate the strongest PCA loading vectors. Thick dashed line (right panel) indicates the mixing array between PSUM and DSUM, while thick solid lines indicate the upper and lower boundaries for random sampling. Melt fraction is calculated with @eq:melt-fraction.](assets/figs/earthchem-mixing-arrays.png){#fig:earthchem-mixing-arrays}

## Generating RocMLM Training Data {#sec:build-training-data}

### Gibbs Free Energy Minimization

We use the GFEM program Perple_X [version 7.0.9, @connolly2009] to compute RocMLM training data (rock properties) using a broad range of upper mantle PT conditions (@fig:training-dataset-design) and the two sets of bulk chemical compositions described above (@sec:bulk-rock-compositions). Perple_X computes rock properties by minimizing the total GFE of a multicomponent multiphase thermodynamic system defined by the following equation [at a fixed PT point, @gibbs1878; @spear1993]:

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

In principle, applying identical sets of solution phase models, thermodynamic data, and bulk compositions to the above equations will define identical GFE hyperplanes (i.e. define the same G-X "surfaces" in multidimensional space). This implies that GFEM programs should converge on identical phase relations irrespective of the minimization algorithm. Similar results are therefore expected for other GFEM programs, although this study exclusively uses Perple_X to compute RocMLM training examples (@fig:image9-sr00000-train).

## Training RocMLMs {#sec:training-rocmls}

RocMLM training data were preprocessed using the following procedure. First, square PT grids of rock properties produced by Perple_X for $n$ random synthetic bulk mantle compositions were combined into cubic three-dimensional datasets, $Z = (z_{1,1,1}, \ldots, z_{w,w,w})$ for density, Vp, and Vs, where $w$ = $n$ = 128 is the resolution of the PT grid. $Z$ was flattened into arrays of training features (PT and $F$), $X = (x_{1,1,1}, \ldots, x_{v,v,v})$, and targets (density, Vp, Vs), $y = (y_{1,1,1}, \ldots, y_{v,v,v})$, where $v = w^3$ is the total number of training examples. $X$ and $y$ were then standardized by removing their mean and scaling to unit variance:

\begin{align}
	\overline{X} &= \frac{X - \mu_X}{\sigma_X} \label{eq:standard-scaler} \\
	\overline{y} &= \frac{y - \mu_y}{\sigma_y} \nonumber
\end{align}

\noindent where $\overline{X}$ and $\overline{y}$ are the standardized feature and target arrays, respectively, and $\mu$ and $\sigma$ are the mean and standard deviation of the training examples. Applying @eq:standard-scaler ("z-score normalization") to RocMLM training data allows RocMLMs to fit training examples with different magnitude values (i.e., different units: GPa, K, g/cm$^3$, km/s).

![PT diagrams and depth profiles showing an unprocessed training example sr00000 (1 of 128 random bulk mantle compositions) used for RocMLM training. Density is shown in (a, b, c), Vp in (d, e, f), and Vs in (g, h, i). The bold white line in is the geotherm used to define the depth profiles. PUM and PREM are from @sun1989 and @dziewonski1981, respectively.](assets/figs/image9-sr00000-train.png){#fig:image9-sr00000-train}

The preprocessed training data were then fit with eight different non-linear RocMLMs (@tbl:mlm-pro-con). Each RocMLM was tuned by defining arrays of relevant model parameters (e.g., n-neighbors, max features, batch size, learning rate, etc.) and evaluating a performance score (RMSE) for each RocMLM over the combination of parameter arrays. The set of parameters that produce the best score were used for RocMLM training. The reader is referred to the scikit-learn [documentation](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning) on regression models (@tbl:mlm-pro-con) and model selection by CVGridSearch for more specifics.

During training, performance metrics for each RocMLM were evaluated using a k-fold cross-validation technique to determine how well RocMLMs were fitting the training data. The training data were shuffled to reduce the impact of inherent ordering in the data and partitioned into $k$ = 8 non-overlapping folds of training examples. Cross-validation then proceeded with $k$ iterations, where in each iteration, models were trained on training examples from $k$-1 folds and performance (RMSE) was evaluated on the remaining fold. After all iterations completed, performance was evaluated on a separate set of training examples (validation set) not seen by the RocMLM during training to provide a measurement of each RocMLM's generalizability.

{{ mlm-pro-con.md }}

# Results {#sec:results}

### Comparing Perple_X Models with PREM

In practice, small differences between MAGEMin and Perple_X models arise while minimizing @eq:gfe because different GFEM algorithms can converge on different local minima within the same region of the GFE hyperplane. Proximate local minima will produce the same stable phase assemblages with small differences in the chemical potentials among the phases---hence small differences in rock properties. However, MAGEMin and Perple_X models generally show density differences of $\leq$ 5% that indicate high degrees of correlation and internal consistency between the GFEM algorithms (@fig:image4-PUM-train-rho).

![PT-density diagrams showing density (in g/cm$^3$) calculated by (a) MAGEMin and (b) Perple_X for a PUM bulk composition (@tbl:benchmark-samples). (c) PT-error diagram showing the percent difference between (a) and (b). (d) Density profiles comparing MAGEMin and Perple_X models with the empirically-derived Preliminary Reference Earth Model (PREM) from @dziewonski1981.](assets/figs/image4-PUM-train-rho.png){#fig:image4-PUM-train-rho}

In addition to the internal consistency observed between GFEM algorithms, density profiles extracted from MAGEMin and Perple_X models generally indicate external consistency with empirical-derived density profiles from the Preliminary Reference Earth Model [PREM, @dziewonski1981]. In contrast to the PREM, however, MAGEMin and Perple_X models show two discrete density jumps between 410 and 660 km that are offset (relative to the PREM) towards lower densities at higher Ps ([@fig:image4-PUM-train-rho]d). The P offset is partially explained by the exclusion of certain solution models described in @sec:thermodynamic-data. For example, Perple_X models that include solid solutions for high-pressure mantle minerals are more consistent with the PREM between 410–660 km than equivalent models that only consider end-members. The density offset, on the other hand, can partially be explained by an estimated 3-4% uncertainty for Perple_X density profiles [approximately 0.1 g/cm$^3$, @connolly2016].

## Comparing RocMLMs with MAGEMin and Perple_X {#sec:comparing-rocmls}

### Qualitative Comparison

Model performance can be assessed qualitatively by comparing the density surfaces computed by MAGEMin and Perple_X with the density surfaces inferred by RocMLMs (@fig:all-surf-PUM-rho).

### Quantitative Comparison

{{ rocml-performance.md }}

![caption](assets/figs/rocml-performance.png)

## Best RocMLM Selection

![Density profiles comparing (a) MAGEMin and (b) Perple_X models with the empirically-derived Preliminary Reference Earth Model (PREM) from @dziewonski1981. Performance metrics R$^2$ and RMSE show the correlations and accuracies of the Decision Tree RocMLM compared to MAGEMin and Perple_X. MAGEMin and Perple_X models assume a PUM bulk composition (@tbl:benchmark-samples).](assets/figs/prem-PUM-DT-rho.png)

# Discussion

## Validating RocMLM Density Models

\clearpage

# Acknowledgements

# Open Research

All data, code, and relevant information for reproducing this work can be found at [https://github.com/buchanankerswell/kerswell_et_al_madmlm](https://github.com/buchanankerswell/kerswell_et_al_madmlm), and at [https://doi.org/10.17605/OSF.IO/K23TB](https://doi.org/10.17605/OSF.IO/K23TB), the official Open Science Framework data repository (Kerswell et al., 2024). All code is MIT Licensed and free for use and distribution (see license details).

# References

<div id="refs"></div>

\cleardoublepage

# Appendix

## GFEM Benchmarking

Benchmarking GFEM programs was a necessary first step for estimating the time required for building RocMLM training datasets and quantifying the efficiency of incumbent GFEM programs (@fig:benchmark-times), which our RocMLMs will need to beat to be considered an advancement beyond the status-quo. Estimated bulk compositions for primitive and depleted mantle-derived rocks ([@tbl:benchmark-samples]) were used for benchmarking MAGEMin and Perple_X. [@tbl:benchmark-times-table] shows the computation times with respect to various PT grid resolutions (8x8, 16x16, 32x32, 64x64, 128x128). All computations were made on a Macbook Pro (2022; M2 chip) with macOS 13.4 and Python 3.11.4. Note that MAGEMin was ran on 6 CPU cores in parallel, while Perple_X does not have parallel capabilities.

<!--{{ gfem-efficiency.md }}-->

![caption](assets/figs/gfem-efficiency.png)

\clearpage

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
