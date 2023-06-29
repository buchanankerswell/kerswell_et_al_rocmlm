# Makefile config
SHELL = /bin/bash -o pipefail
WORKDIR = $(shell pwd)
UNAME_S := $(shell uname -s)
# Logging
LOGFILE := $(WORKDIR)/log/log-$(shell date +%Y-%m-%d)
LOG := 2>&1 | tee -a $(LOGFILE)
# Requirements for running study
# Conda env
CONDA_ENV_NAME = madnn
HAS_CONDA := $(shell command -v conda > /dev/null && echo true || echo false)
CONDA_SPECS_FILE = $(WORKDIR)/python/conda-environment.yaml
CONDA_PYTHON = $$(conda run -n $(CONDA_ENV_NAME) which python)
# Magemin program
MAGEMIN = $(WORKDIR)/MAGEMin/MAGEMin
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
GITHUB_REPO = https://github.com/buchanankerswell/kerswell_et_al_madnn
MAGEMIN_REPO = https://github.com/ComputationalThermodynamics/MAGEMin.git
# Database build and benchmarking options
SEED = 32
PMIN ?= 10
PMAX ?= 110
PRES ?= 32
TMIN ?= 500
TMAX ?= 2500
TRES ?= 32
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
PARAMS ?= ["Vp", "StableSolutions", "StableVariance", "DensityOfFullAssemblage"]
# Make clean
DATAPURGE = python/__pycache__ .job output
DATACLEAN = assets log MAGEMin runs
FIGSPURGE = figs
FIGSCLEAN = figs

all: $(LOGFILE) $(PYTHON) create_conda_env $(DATA) $(CONFIG) $(PERPLEX) $(MAGEMIN)
	@echo "=============================================" $(LOG)
	@$(CONDA_PYTHON) python/session-info.py $(LOG)
	@echo "=============================================" $(LOG)
	@echo "Run any of the following:" $(LOG)
	@echo -e \
		"make build_database\nmake benchmark_magemin_perplex\n\
		PMIN=     <kbar>\n\
		PMAX=     <kbar>\n\
		PRES=     <number of points>\n\
		TMIN=     <Celcius>\n\
		TMAX=     <Celcius>\n\
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
		SEED=     <number for random state>\n\
		OUTDIR=   <directory of MAGEMin output>" $(LOG)
	@echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" $(LOG)
	@echo -e \
		"make visualize_database\n\
		PARAMS=   <'[\"param\", \"param\", \"param\"]'>\n\
			Options:\n\
			Point,Status, Gibbs, BrNorm, Vp, Vs, Entropy, StableSolutions\n\
			LiquidFraction, DensityOfFullAssemblage, DensityOfLiquid, DensityOfSolid\n\
			DensityOfMixture\n\
		FIGOX=   <'[\"oxide\", \"oxide\", \"oxide\"]'> for Harker diagrams\n\
			Options:\n\
			SiO2, Al2O3, CaO, MgO, FeO, K2O, Na2O, TiO2, Fe2O3, Cr2O3, H2O\n\
		OUTDIR=   <directory of MAGEMin output>\n\
		FIGDIR=   <directory for saving plots>" $(LOG)
	@echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" $(LOG)
	@echo "make submit_jobs" $(LOG)
	@echo "make remove_conda_env" $(LOG)
	@echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" $(LOG)
	@echo "=============================================" $(LOG)

visualize_database: $(LOGFILE) $(PYTHON)
	@echo "Visualizing MAGEMin database ..." $(LOG)
	@$(CONDA_PYTHON) python/visualize-database.py \
		--Pmin $(PMIN) \
		--Pmax $(PMAX) \
		--Pres $(PRES) \
		--Tmin $(TMIN) \
		--Tmax $(TMAX) \
		--Tres $(TRES) \
		--sampleid '$(SAMPLEID)' \
		--params '$(PARAMS)' \
		--figox '$(FIGOX)' \
		--outdir $(OUTDIR) \
		--figdir $(FIGDIR) \
	$(LOG)
	@echo "=============================================" $(LOG)

submit_jobs: $(LOGFILE) $(PYTHON) $(DATA)
	@echo "Submitting job to SLURM ..." $(LOG)
	@$(CONDA_PYTHON) python/submit-jobs.py $(LOG)
	@echo "=============================================" $(LOG)

benchmark_magemin_perplex: $(LOGFILE) $(PYTHON) $(DATA) $(CONFIG) $(PERPLEX) $(MAGEMIN)
	@mkdir -p $(BENCHMARK)
	@mkdir -p $(WORKDIR)/$(FIGDIR)/benchmark
	@mkdir -p $(BENCHMARK)/$(SAMPLEID)-$(TRES)x$(PRES)
	@mkdir -p $(BENCHMARK)/$(SAMPLEID)-$(TRES)x$(PRES)/perplex
	@echo "Building MAGEMin model ..." $(LOG)
	@$(CONDA_PYTHON) python/benchmark-magemin-perplex.py \
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
	@echo "magemin,$(SAMPLEID),$(TRES),$(PRES),$$( \
		grep -oE "MAGEMin comp time: \+([0-9.]+) ms }" $(LOGFILE) | \
		tail -n 1 | \
		sed -E 's/MAGEMin comp time: \+([0-9.]+) ms }/\1/' | \
		awk '{printf "%.1f", $$NF/1000}')" >> $(DATA)/benchmark-comp-times.csv
	@$(MAKE) visualize_database \
		SAMPLEID=$(SAMPLEID) \
		TRES=$(TRES) \
		PRES=$(PRES) \
		FIGDIR=$(FIGDIR)/$(SAMPLEID)-$(TRES)x$(PRES) $(LOG)
	@convert $(WORKDIR)/$(FIGDIR)/$(SAMPLEID)-$(TRES)x$(PRES)/StableVariance.png \
		-pointsize 100 \
		-annotate +25+90 "b" \
		$(WORKDIR)/$(FIGDIR)/$(SAMPLEID)-$(TRES)x$(PRES)/bench.png
	@chmod +x $(PERPLEX)/build $(PERPLEX)/vertex $(PERPLEX)/pssect
	@if [ ! -e "$(PERPLEX)/$(SAMPLEID)-$(TRES)x$(PRES).dat" ]; then \
		echo "Building perplex model ..." $(LOG); \
		(cd $(PERPLEX) && cp $(CONFIG)/perplex-build-config .); \
		(cd $(PERPLEX) && cp $(CONFIG)/perplex-options-config perplex_option.dat); \
		(cd $(PERPLEX) && cp $(CONFIG)/perplex-plot-config perplex_plot_option.dat); \
		(cd $(PERPLEX) && \
			awk '{ \
				gsub("{SAMPLEID}", \
				"$(SAMPLEID)-$(TRES)x$(PRES)"); print \
			}' perplex-build-config > temp_file && \
			mv temp_file perplex-build-config \
		); \
		(cd $(PERPLEX) && \
			awk -F',' -v sample_id="$(SAMPLEID)" 'BEGIN { \
				found=0 \
			} $$13 == sample_id { \
				print $$1" "$$2" "$$3" "$$4" "$$5" "$$6" "$$7" "$$8" "$$9" "$$10" "$$11; \
				found=1 \
			} END { \
				if (found==0) \
					print "Sample ID not found" \
			}' $(DATA)/benchmark-samples.csv > sample-data && \
			awk -v sample_comp="$$(cat sample-data)" '/{SAMPLECOMP}/ { \
				sub(/{SAMPLECOMP}/, sample_comp) \
			} { \
				print \
			}' perplex-build-config > temp-file && \
			mv temp-file perplex-build-config \
		); \
		(cd $(PERPLEX) && \
			awk -v tres=$(TRES) -v pres=$(PRES) 'BEGIN { \
				div_tres = tres / 4; \
				div_pres = pres / 4 \
			} /x_nodes/ { \
				sub(/default/, div_tres " " tres) \
			} /y_nodes/ { \
				sub(/default/, div_pres " " pres) \
			} 1' perplex_option.dat > temp-file && \
			mv temp-file perplex_option.dat \
		); \
		(cd $(PERPLEX) && ./build < perplex-build-config) $(LOG); \
		(cd $(PERPLEX) && echo "$(SAMPLEID)-$(TRES)x$(PRES)" | ./vertex) $(LOG); \
		echo "perplex,$(SAMPLEID),$(TRES),$(PRES),$$( \
			grep -oE "Total elapsed time\s+([0-9.]+)" $(LOGFILE) | \
			tail -n 1 | \
			sed -E 's/Total elapsed time\s+([0-9.]+)/\1/' | \
			awk '{printf "%.1f", $$NF*60}')" >> $(DATA)/benchmark-comp-times.csv; \
		echo "Finished perplex model ..." $(LOG); \
		echo "=============================================" $(LOG); \
	fi
	@if [ ! -e "$(PERPLEX)/$(SAMPLEID)-$(TRES)x$(PRES).pdf" ]; then \
		echo "Plotting perplex model ..." $(LOG); \
		(cd $(PERPLEX) && echo "$(SAMPLEID)-$(TRES)x$(PRES)" > pssect-config); \
		(cd $(PERPLEX) && echo "N" >> pssect-config); \
		(cd $(PERPLEX) && ./pssect < pssect-config) $(LOG); \
		(cd $(PERPLEX) && ps2pdf $(SAMPLEID)-$(TRES)x$(PRES).ps $(SAMPLEID)-$(TRES)x$(PRES).pdf); \
		(cd $(PERPLEX) && \
			convert $(SAMPLEID)-$(TRES)x$(PRES).pdf \
				-background white \
				-flatten \
				-crop 565x495+0+165 \
				-resize 1994x1549 \
				-pointsize 100 \
				-fill black \
				-annotate +25+605 "a" \
				$(SAMPLEID)-$(TRES)x$(PRES).png \
		); \
		(cd $(PERPLEX) && \
			rm perplex-build-config pssect-config perplex_plot_option.dat perplex_option.dat \
		); \
		(cd $(PERPLEX) && \
			mv $(SAMPLEID)-$(TRES)x$(PRES)* $(BENCHMARK)/$(SAMPLEID)-$(TRES)x$(PRES)/perplex \
		); \
		montage \
			$(BENCHMARK)/$(SAMPLEID)-$(TRES)x$(PRES)/perplex/$(SAMPLEID)-$(TRES)x$(PRES).png \
			$(WORKDIR)/$(FIGDIR)/$(SAMPLEID)-$(TRES)x$(PRES)/bench.png -tile 2x1 -geometry +0+0 \
			$(WORKDIR)/$(FIGDIR)/benchmark/$(SAMPLEID)-$(TRES)x$(PRES).png; \
		rm $(WORKDIR)/$(FIGDIR)/$(SAMPLEID)-$(TRES)x$(PRES)/bench.png; \
		echo "Finished plotting perplex model ..." $(LOG); \
	fi
	@echo "=============================================" $(LOG)
	@echo "Finished benchmarking $(SAMPLEID)-$(TRES)x$(PRES) ..." $(LOG)
	@echo "See figure at:" $(LOG)
	@echo "$(WORKDIR)/$(FIGDIR)/benchmark/$(SAMPLEID)-$(TRES)x$(PRES).png" $(LOG)
	@echo "=============================================" $(LOG)

build_database: $(LOGFILE) $(PYTHON) $(DATA) $(MAGEMIN)
	@echo "Building MAGEMin database ..." $(LOG)
	@$(CONDA_PYTHON) python/build-database.py \
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

$(MAGEMIN): $(LOGFILE) $(PYTHON) $(CONFIG)
	@if [ ! -e "$(MAGEMIN)" ]; then \
		echo "=============================================" $(LOG); \
		echo "Cloning MAGEMin from $(MAGEMIN_REPO) ..." $(LOG); \
		chmod +x python/clone-magemin.py; \
		$(CONDA_PYTHON) python/clone-magemin.py $(LOG); \
		if [ "$(UNAME_S)" = "Linux" ]; then \
			echo "Configuring MAGEMin for meso ..." $(LOG); \
			cp $(CONFIG)/meso-config MAGEMin/Makefile; \
		fi; \
		echo "Compiling MAGEMin ..." $(LOG); \
		echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" $(LOG); \
		(cd MAGEMin && make) $(LOG); \
		echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" $(LOG); \
	else \
		echo "MAGEMin found at:" $(LOG); \
		echo "$(MAGEMIN)" $(LOG); \
	fi

remove_conda_env: $(LOGFILE)
	@echo "Removing conda env $(CONDA_ENV_NAME) ..." $(LOG)
	@conda remove --name $(CONDA_ENV_NAME) --all --yes $(LOG)
	@echo "=============================================" $(LOG)

create_conda_env: $(LOGFILE) $(CONDA_SPECS_FILE) find_conda_env
	@if [ "$(HAS_CONDA)" = "false" ]; then \
		echo "conda not found in PATH, install conda first ..." $(LOG); \
		echo "See $(GITHUB_REPO) for details ..." $(LOG); \
		exit 1; \
	fi
	@if [ -d "$(MY_ENV_DIR)" ]; then \
		echo "Conda environment \"$(CONDA_ENV_NAME)\" found at:" $(LOG); \
		echo "$(MY_ENV_DIR) ..." $(LOG); \
	else \
		echo "Creating env $(CONDA_ENV_NAME) from $(CONDA_SPECS_FILE) ..." $(LOG); \
		conda env create --file $(CONDA_SPECS_FILE) > /dev/null $(LOG); \
		echo "Conda env created ..." $(LOG); \
	fi

find_conda_env:
	@echo "Looking for conda env ..." $(LOG)
	$(eval MY_ENV_DIR := $(shell conda env list | grep $(CONDA_ENV_NAME) | awk '{print $$2}'))

$(PERPLEX): $(LOGFILE) $(PYTHON)
	@if [ ! -d "$(DATA)" ]; then \
		$(CONDA_PYTHON) python/download-assets.py $(LOG); \
	else \
		echo "Perplex found at:" $(LOG); \
		echo "$(PERPLEX)" $(LOG); \
	fi

$(DATA): $(LOGFILE) $(PYTHON)
	@if [ ! -d "$(DATA)" ]; then \
		$(CONDA_PYTHON) python/download-assets.py $(LOG); \
	else \
		echo "Data files found at:" $(LOG); \
		echo "$(DATA)" $(LOG); \
	fi

$(CONFIG): $(LOGFILE) $(PYTHON)
	@if [ ! -e "$(CONFIG)" ]; then \
		$(CONDA_PYTHON) python/download-assets.py $(LOG); \
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

.PHONY: find_conda_env remove_conda_env create_conda_env build_database benchmark_magemin_perplex submit_jobs visualize_earthchem visualize_database all purge clean