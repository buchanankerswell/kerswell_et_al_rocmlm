# Makefile config
SHELL = /bin/bash -o pipefail
WORKDIR = $(shell pwd)
UNAMES := $(shell uname -s)
# Logging
LOGFILE := $(WORKDIR)/log/log-$(shell date +"%d-%m-%Y")
LOG := 2>&1 | tee -a $(LOGFILE)
# Requirements for running study
# Conda env
CONDAENVNAME = madnn
HASCONDA := $(shell command -v conda > /dev/null && echo true || echo false)
CONDASPECSFILE = $(WORKDIR)/python/conda-environment.yaml
CONDAPYTHON = $$(conda run -n $(CONDAENVNAME) which python)
# Magemin program
MAGEMIN = $(WORKDIR)/MAGEMin
# Perplex program
PERPLEX = $(WORKDIR)/assets/perplex
# Directories with data and scripts
BENCHMARK = $(WORKDIR)/assets/benchmark
DATA = $(WORKDIR)/assets/data
CONFIG = $(WORKDIR)/assets/config
PYTHON = $(WORKDIR)/python/conda-environment.yaml \
				 $(WORKDIR)/python/magemin.py \
				 $(WORKDIR)/python/clone-magemin.py \
				 $(WORKDIR)/python/build-database.py \
				 $(WORKDIR)/python/benchmark-magemin-perplex.py \
				 $(WORKDIR)/python/submit-jobs.py \
				 $(WORKDIR)/python/visualize-database.py \
				 $(WORKDIR)/python/download-assets.py
# Other variables
GITHUBREPO = https://github.com/buchanankerswell/kerswell_et_al_madnn
MAGEMINREPO = https://github.com/ComputationalThermodynamics/MAGEMin.git
# Database build and benchmarking options
SEED = 32
PMIN ?= 1
PMAX ?= 28
PRES ?= 8
TMIN ?= 773
TMAX ?= 2273
TRES ?= 8
COMP ?= [44.9, 4.44, 3.54, 37.71, 8.03, 0.029, 0.36, 0.2, 0.01, 0.38, 0]
FRAC ?= wt
SAMPLEID ?= PUM
NORMOX ?= all
SOURCE ?= earthchem
STRATEGY ?= random
N ?= 1
K ?= 0
PARALLEL ?= True
NPROCS ?= $(shell expr $(shell nproc) - 2)
OUTDIR ?= runs
# Database visualization options
FIGDIR ?= figs
FIGOX ?= ["MgO", "FeO", "CaO", "Al2O3"]
PARAMS ?= ["Vp", "Vs", "LiquidFraction", "StableSolutions", "StableVariance", "DensityOfFullAssemblage"]
COLORMAP ?= bone
# Make clean
DATAPURGE = python/__pycache__ .job output
DATACLEAN = assets log MAGEMin runs
FIGSPURGE = figs
FIGSCLEAN = figs

all: $(LOGFILE) $(PYTHON) create_conda_env $(DATA) $(CONFIG) $(PERPLEX) $(MAGEMIN)
	@echo "=============================================" $(LOG)
	@$(CONDAPYTHON) python/session-info.py $(LOG)
	@echo "=============================================" $(LOG)
	@echo "Run any of the following:" $(LOG)
	@echo -e \
		"make build_database\n\
		PMIN=     <GPa>\n\
		PMAX=     <GPa>\n\
		PRES=     <number of points>\n\
		TMIN=     <K>\n\
		TMAX=     <K>\n\
		TRES=     <number of points>\n\
		COMP=     <'[SiO2, Al2O3, CaO, MgO, FeO, K2O, Na2O, TiO2, Fe2O3, Cr2O3, H2O]'>\n\
		FRAC=     <mol or wt>\n\
		SAMPLEID= <sample name>\n\
		NORMOX=   <'[\"oxide\", \"oxide\", \"oxide\"]'> or <all>\n\
		SOURCE=   <earthchem or sample>\n\
		STRATEGY= <batch or random>\n\
		N=        <number of samples>\n\
		K=        <batch number>\n\
		PARALLEL= <True or False>\n\
		NPROCS=   <number of processors for parallel>\n\
		SEED=     <number for random state>" $(LOG)
	@echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" $(LOG)
	@echo -e \
		"make benchmark\n\
		PRES=     <number of points>\n\
		TRES=     <number of points>\n\
		SAMPLEID= <sample name>\n\
		PARALLEL= <True or False>\n\
		NPROCS=   <number of processors for parallel>" $(LOG)
	@echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" $(LOG)
	@echo -e \
		"make visualize_database\n\
		PARAMS=   <'[\"param\", \"param\", \"param\"]'>\n\
			Options:\n\
			Point, Status, Gibbs, BrNorm, Vp, Vs, Entropy, StableSolutions\n\
			LiquidFraction, DensityOfFullAssemblage, DensityOfLiquid, DensityOfSolid\n\
			DensityOfMixture\n\
		FIGOX=    <'[\"oxide\", \"oxide\", \"oxide\"]'> for Harker diagrams\n\
			Options:\n\
			SiO2, Al2O3, CaO, MgO, FeO, K2O, Na2O, TiO2, Fe2O3, Cr2O3, H2O\n\
		COLORMAP= <viridis bone pink grey>\n\
		OUTDIR=   <directory of MAGEMin output>\n\
		FIGDIR=   <directory for saving plots>" $(LOG)
	@echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" $(LOG)
	@echo -e \
		"make visualize_benchmark\n\
		PARAMS=   <'[\"param\", \"param\", \"param\"]'>\n\
			Options:\n\
			Point, Status, Gibbs, BrNorm, Vp, Vs, Entropy, StableSolutions\n\
			LiquidFraction, DensityOfFullAssemblage, DensityOfLiquid, DensityOfSolid\n\
			DensityOfMixture\n\
		COLORMAP= <viridis bone pink grey>\n\
		OUTDIR=   <directory of MAGEMin output>\n\
		FIGDIR=   <directory for saving plots>" $(LOG)
	@echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" $(LOG)
	@echo "make submit_jobs" $(LOG)
	@echo "make remove_conda_env" $(LOG)
	@echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" $(LOG)
	@echo "=============================================" $(LOG)

visualize_benchmark_all: $(LOGFILE) $(PYTHON)
	@for run in $$(ls -d $(OUTDIR)/* | sed 's/$(OUTDIR)\/\(.*\)/\1/'); do \
		echo "Visualizing benchmark $$run" $(LOG); \
		$(MAKE) visualize_benchmark SAMPLEID=$$run FIGDIR=figs/benchmark/$$run; \
	done
	@$(MAKE) visualize_other
	@echo "=============================================" $(LOG)

visualize_other: $(LOGFILE) $(PYTHON)
	@echo "Visualizing other figures ..." $(LOG)
	@$(CONDAPYTHON) python/visualize-other.py $(LOG)
	@echo "=============================================" $(LOG)

visualize_benchmark: $(LOGFILE) $(PYTHON)
	@echo "Visualizing benchmark comparisons ..." $(LOG)
	@$(CONDAPYTHON) python/visualize-benchmark.py \
		--sampleid '$(SAMPLEID)' \
		--params '$(PARAMS)' \
		--figox '$(FIGOX)' \
		--colormap $(COLORMAP) \
		--outdir $(OUTDIR) \
		--figdir $(FIGDIR) \
	$(LOG)
	@echo "=============================================" $(LOG)

visualize_database: $(LOGFILE) $(PYTHON)
	@echo "Visualizing MAGEMin database ..." $(LOG)
	@$(CONDAPYTHON) python/visualize-database.py \
		--sampleid '$(SAMPLEID)' \
		--params '$(PARAMS)' \
		--figox '$(FIGOX)' \
		--colormap $(COLORMAP) \
		--outdir $(OUTDIR) \
		--figdir $(FIGDIR) \
	$(LOG)
	@echo "=============================================" $(LOG)

submit_jobs: $(LOGFILE) $(PYTHON) $(DATA)
	@echo "Submitting job to SLURM ..." $(LOG)
	@$(CONDAPYTHON) python/submit-jobs.py $(LOG)
	@echo "=============================================" $(LOG)

benchmark_all: $(LOGFILE) $(PYTHON) $(DATA) $(CONFIG) $(PERPLEX) $(MAGEMIN)
	@$(MAKE) benchmark TRES=$(TRES) PRES=$(PRES) SAMPLEID=DMM
	@$(MAKE) benchmark TRES=$(TRES) PRES=$(PRES) SAMPLEID=NMORB
	@$(MAKE) benchmark TRES=$(TRES) PRES=$(PRES) SAMPLEID=PUM
	@$(MAKE) benchmark TRES=$(TRES) PRES=$(PRES) SAMPLEID=RE46

benchmark: $(LOGFILE) $(PYTHON) $(DATA) $(CONFIG) $(PERPLEX) $(MAGEMIN)
	@if [ -e \
		"$(BENCHMARK)/$(SAMPLEID)-$(TRES)x$(PRES)/$(SAMPLEID)-$(TRES)x$(PRES)_phases.tab" \
	]; then \
		echo "$(SAMPLEID)-$(TRES)x$(PRES) already benchmarked ..."; \
		exit 1; \
	fi
	@if [ ! -d "$(OUTDIR)/$(SAMPLEID)-$(TRES)x$(PRES)" ]; then \
		mkdir -p $(BENCHMARK); \
		mkdir -p $(BENCHMARK)/$(SAMPLEID)-$(TRES)x$(PRES); \
		mkdir -p $(WORKDIR)/$(FIGDIR)/benchmark; \
		if [ "$(UNAMES)" = "Linux" ]; then \
			echo "Configuring MAGEMin for meso ..." $(LOG); \
			cp $(CONFIG)/meso-compile MAGEMin/Makefile; \
		fi; \
		echo "Configuring MAGEMin for $(SAMPLEID) ..." $(LOG); \
		cp $(CONFIG)/magemin-init $(MAGEMIN)/src/initialize.h $(LOG); \
		echo "Compiling MAGEMin ..." $(LOG); \
		echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" $(LOG); \
		(cd MAGEMin && make) $(LOG); \
		echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" $(LOG); \
		echo "Building MAGEMin model ..." $(LOG); \
		$(CONDAPYTHON) python/benchmark-magemin-perplex.py \
			--Pmin $(PMIN) \
			--Pmax $(PMAX) \
			--Pres $(PRES) \
			--Tmin $(TMIN) \
			--Tmax $(TMAX) \
			--Tres $(TRES) \
			--comp '$(COMP)' \
			--frac $(FRAC) \
			--sampleid $(SAMPLEID) \
			--normox '$(NORMOX)' \
			--source $(SOURCE) \
			--strategy $(STRATEGY) \
			--n $(N) \
			--k $(K) \
			--parallel $(PARALLEL) \
			--nprocs $(NPROCS) \
			--seed $(SEED) \
			--outdir $(OUTDIR) \
			$(LOG); \
		mv $(OUTDIR)/$(SAMPLEID) $(OUTDIR)/$(SAMPLEID)-$(TRES)x$(PRES); \
		(cd $(OUTDIR)/$(SAMPLEID)-$(TRES)x$(PRES) && \
			find . -name "*$(SAMPLEID)*" -type f -exec sh -c \
			'mv "$$0" "$${0/$(SAMPLEID)/$(SAMPLEID)-$(TRES)x$(PRES)}"' {} \; \
		); \
		echo -n "$(SAMPLEID),$(PMIN),$(PMAX),$(TMIN),$(TMAX),$$((TRES * PRES)),$$( \
			grep -oE "MAGEMin comp time: \+([0-9.]+) ms }" $(LOGFILE) | \
			tail -n 1 | \
			sed -E 's/MAGEMin comp time: \+([0-9.]+) ms }/\1/' | \
			awk '{printf "%.1f", $$NF/1000}')" >> $(DATA)/benchmark-times.csv; \
		chmod +x $(PERPLEX)/build $(PERPLEX)/vertex $(PERPLEX)/pssect $(PERPLEX)/werami; \
	fi
	@if [ ! -e "$(PERPLEX)/$(SAMPLEID).dat" ]; then \
		echo "Building perplex model ..." $(LOG); \
		(cd $(PERPLEX) && \
			cp $(CONFIG)/perplex-build perplex-build-$(SAMPLEID) && \
			cp $(CONFIG)/perplex-grid perplex-grid-$(SAMPLEID) && \
			cp $(CONFIG)/perplex-phase perplex-phase-$(SAMPLEID) && \
			cp $(CONFIG)/perplex-options perplex-options-$(SAMPLEID) && \
			cp $(CONFIG)/perplex-plot perplex_plot_option.dat && \
			awk '{ \
				gsub("{SAMPLEID}", \
				"$(SAMPLEID)"); print \
			}' perplex-build-$(SAMPLEID) > temp-file && \
			mv temp-file perplex-build-$(SAMPLEID) && \
			awk \
				-v tmin="$(TMIN)" \
				-v tmax="$(TMAX)" \
				-v pmin="$$(echo $(PMIN)*10000 | bc)" \
				-v pmax="$$(echo $(PMAX)*10000 | bc)" \
				'{\
					gsub("{TMIN}", tmin); \
					gsub("{TMAX}", tmax); \
					gsub("{PMIN}", pmin); \
					gsub("{PMAX}", pmax); \
					print \
				}' perplex-build-$(SAMPLEID) > temp-file && \
			mv temp-file perplex-build-$(SAMPLEID) && \
			awk -F',' -v sample_id="$(SAMPLEID)" 'BEGIN { \
				found=0 \
			} $$13 == sample_id { \
				print $$1" "$$2" "$$3" "$$4" "$$5" "$$6" "$$7" "$$8" "$$9" "$$10" "$$11; \
				found=1 \
			} END { \
				if (found==0) \
					print "Sample ID not found" \
			}' $(DATA)/benchmark-comps.csv > sample-data && \
			awk -v sample_comp="$$(cat sample-data)" '/{SAMPLECOMP}/ { \
				sub(/{SAMPLECOMP}/, sample_comp) \
			} { \
				print \
			}' perplex-build-$(SAMPLEID) > temp-file && \
			mv temp-file perplex-build-$(SAMPLEID) && \
			rm sample-data && \
			awk '{ \
				gsub("{SAMPLEID}", \
				"$(SAMPLEID)"); print \
			}' perplex-grid-$(SAMPLEID) > temp-file && \
			mv temp-file perplex-grid-$(SAMPLEID) && \
			awk '{ \
				gsub("{SAMPLEID}", \
				"$(SAMPLEID)"); print \
			}' perplex-phase-$(SAMPLEID) > temp-file && \
			mv temp-file perplex-phase-$(SAMPLEID) && \
			awk -v tres=$(TRES) -v pres=$(PRES) 'BEGIN { \
				div_tres = tres / 4; \
				div_pres = pres / 4 \
			} /x_nodes/ { \
				sub(/default/, div_tres " " (tres + 1)) \
			} /y_nodes/ { \
				sub(/default/, div_pres " " (pres + 1)) \
			} 1' perplex-options-$(SAMPLEID) > temp-file && \
			mv temp-file perplex-options-$(SAMPLEID) && \
			./build < perplex-build-$(SAMPLEID) && \
			echo "$(SAMPLEID)" | ./vertex && \
			./werami < perplex-grid-$(SAMPLEID) && \
			mv $(SAMPLEID)_1.tab $(SAMPLEID)_grid.tab && \
			./werami < perplex-phase-$(SAMPLEID) && \
			mv $(SAMPLEID)_1.tab $(SAMPLEID)_phases.tab && \
			echo "$(SAMPLEID)" > pssect-options-$(SAMPLEID) && \
			echo "N" >> pssect-options-$(SAMPLEID) && \
			./pssect < pssect-options-$(SAMPLEID) && \
			ps2pdf $(SAMPLEID).ps $(SAMPLEID).pdf && \
			rm -rf \
			perplex_plot_option.dat \
			perplex-build-$(SAMPLEID) \
			perplex-grid-$(SAMPLEID) \
			perplex-options-$(SAMPLEID) \
			perplex-phase-$(SAMPLEID) \
			pssect-options-$(SAMPLEID) && \
			mv $(SAMPLEID)* $(BENCHMARK)/$(SAMPLEID)-$(TRES)x$(PRES) \
		) $(LOG); \
		(cd $(BENCHMARK)/$(SAMPLEID)-$(TRES)x$(PRES) && \
			find . -name "*$(SAMPLEID)*" -type f -exec sh -c \
			'mv "$$0" "$${0/$(SAMPLEID)/$(SAMPLEID)-$(TRES)x$(PRES)}"' {} \; \
		); \
		echo ",$$( \
			grep -oE "Total elapsed time\s+([0-9.]+)" $(LOGFILE) | \
			tail -n 1 | \
			sed -E 's/Total elapsed time\s+([0-9.]+)/\1/' | \
			awk '{printf "%.1f", $$NF*60}')" >> $(DATA)/benchmark-times.csv; \
		echo "=============================================" $(LOG); \
		echo "Finished perplex model ..." $(LOG); \
		echo "Finished benchmarking $(SAMPLEID) ..." $(LOG); \
		echo "=============================================" $(LOG); \
	fi
	@echo "To visualize benchmark, run:" $(LOG)
	@echo "make visualize_benchmark \
	SAMPLEID=$(SAMPLEID)-$(TRES)x$(PRES) \
	FIGDIR=$(FIGDIR)/benchmark/$(SAMPLEID)-$(TRES)x$(PRES) \
	COLORMAP=$(COLORMAP)" $(LOG)
	@echo "=============================================" $(LOG)

build_database: $(LOGFILE) $(PYTHON) $(DATA) $(MAGEMIN)
	@echo "Building MAGEMin database ..." $(LOG)
	@$(CONDAPYTHON) python/build-database.py \
		--Pmin $(PMIN) \
		--Pmax $(PMAX) \
		--Pres $(PRES) \
		--Tmin $(TMIN) \
		--Tmax $(TMAX) \
		--Tres $(TRES) \
		--comp '$(COMP)' \
		--frac $(FRAC) \
		--sampleid $(SAMPLEID) \
		--normox '$(NORMOX)' \
		--source $(SOURCE) \
		--strategy $(STRATEGY) \
		--n $(N) \
		--k $(K) \
		--parallel $(PARALLEL) \
		--nprocs $(NPROCS) \
		--seed $(SEED) \
		--outdir $(OUTDIR) \
	$(LOG)
	@echo "=============================================" $(LOG)

$(MAGEMIN): $(LOGFILE) $(PYTHON)
	@if [ ! -e "$(MAGEMIN)" ]; then \
		echo "=============================================" $(LOG); \
		echo "Cloning MAGEMin from $(MAGEMINREPO) ..." $(LOG); \
		chmod +x python/clone-magemin.py; \
		$(CONDAPYTHON) python/clone-magemin.py $(LOG); \
	else \
		echo "MAGEMin found at:" $(LOG); \
		echo "$(MAGEMIN)" $(LOG); \
	fi

remove_conda_env: $(LOGFILE)
	@echo "Removing conda env $(CONDAENVNAME) ..." $(LOG)
	@conda remove --name $(CONDAENVNAME) --all --yes $(LOG)
	@echo "=============================================" $(LOG)

create_conda_env: $(LOGFILE) $(CONDASPECSFILE) find_conda_env
	@if [ "$(HASCONDA)" = "false" ]; then \
		echo "conda not found in PATH, install conda first ..." $(LOG); \
		echo "See $(GITHUBREPO) for details ..." $(LOG); \
		exit 1; \
	fi
	@if [ -d "$(MY_ENV_DIR)" ]; then \
		echo "Conda environment \"$(CONDAENVNAME)\" found at:" $(LOG); \
		echo "$(MY_ENV_DIR)" $(LOG); \
	else \
		echo "Creating env $(CONDAENVNAME) from $(CONDASPECSFILE) ..." $(LOG); \
		conda env create --file $(CONDASPECSFILE) > /dev/null $(LOG); \
		echo "Conda env created ..." $(LOG); \
	fi

find_conda_env:
	@echo "Looking for conda env ..." $(LOG)
	$(eval MY_ENV_DIR := $(shell conda env list | grep $(CONDAENVNAME) | awk '{print $$2}'))

$(PERPLEX): $(LOGFILE) $(PYTHON)
	@if [ ! -d "$(DATA)" ]; then \
		$(CONDAPYTHON) python/download-assets.py $(LOG); \
	else \
		echo "Perplex found at:" $(LOG); \
		echo "$(PERPLEX)" $(LOG); \
	fi

$(DATA): $(LOGFILE) $(PYTHON)
	@if [ ! -d "$(DATA)" ]; then \
		$(CONDAPYTHON) python/download-assets.py $(LOG); \
	else \
		echo "Data files found at:" $(LOG); \
		echo "$(DATA)" $(LOG); \
	fi

$(CONFIG): $(LOGFILE) $(PYTHON)
	@if [ ! -e "$(CONFIG)" ]; then \
		$(CONDAPYTHON) python/download-assets.py $(LOG); \
		echo "=============================================" $(LOG); \
	else \
		echo "Configuration files found at:" $(LOG); \
		echo "$(CONFIG)" $(LOG); \
	fi

$(LOGFILE):
	@if [ ! -e "$(LOGFILE)" ]; then \
		mkdir log; \
		touch $(LOGFILE); \
	fi

purge:
	@rm -rf $(DATAPURGE) $(FIGSPURGE)

clean: purge
	@rm -rf $(DATACLEAN) $(FIGSCLEAN)

.PHONY: find_conda_env remove_conda_env create_conda_env build_database benchmark_all benchmark submit_jobs visualize_benchmark visualize_database visualize_other all purge clean