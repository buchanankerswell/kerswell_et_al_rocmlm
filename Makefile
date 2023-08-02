# Makefile config
SHELL = /bin/bash -o pipefail
WORKDIR = $(shell pwd)
UNAMES := $(shell uname -s)
DATE = $(shell date +"%d-%m-%Y")
DATELONG = $(shell date +"%d-%B-%Y")
# Logging
LOGFILE := $(WORKDIR)/log/log-$(shell date +"%d-%m-%Y")
LOG := 2>&1 | tee -a $(LOGFILE)
# Conda env
CONDAENVNAME = madmlm
HASCONDA := $(shell command -v conda > /dev/null && echo true || echo false)
CONDASPECSFILE = $(WORKDIR)/python/conda-environment.yaml
CONDAPYTHON = $$(conda run -n $(CONDAENVNAME) which python)
# Magemin program
MAGEMIN = $(WORKDIR)/MAGEMin
# Perplex program
PERPLEX = $(WORKDIR)/assets/perplex
# Directories with data and scripts
BENCHMARK = $(WORKDIR)/assets/benchmark
DATADIR = $(WORKDIR)/assets/data
DATAURL = https://files.osf.io/v1/resources/k23tb/providers/osfstorage/649149796513ba03733a3536/?zip=
CONFIG = $(WORKDIR)/assets/config
PYTHON = $(WORKDIR)/python/benchmark.py \
				 $(WORKDIR)/python/build-database.py \
				 $(WORKDIR)/python/clone-magemin.py \
				 $(WORKDIR)/python/download-assets.py \
				 $(WORKDIR)/python/magemin.py \
				 $(WORKDIR)/python/regression.py \
				 $(WORKDIR)/python/session-info.py \
				 $(WORKDIR)/python/submit-jobs.py \
				 $(WORKDIR)/python/visualize-database.py \
				 $(WORKDIR)/python/visualize-other.py \
# Other variables
GITHUBREPO = https://github.com/buchanankerswell/kerswell_et_al_madmlm
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
KFOLDS ?= 30
OUTDIR ?= runs
# Machine Learning Regression Options
MLMODS = ["Support Vector", "K Nearest", "Random Forest", "Gradient Boost", "Neural Network 1L", "Neural Network 2L", "Neural Network 3L", "Decision Tree"]
# Database visualization options
FIGDIR ?= $(WORKDIR)/figs
PARAMS ?= ["StableSolutions", "StableVariance", "DensityOfFullAssemblage"]
PARAMSML ?= ["DensityOfFullAssemblage"]
COLORMAP ?= bone
# Make clean
DATAPURGE = python/__pycache__ \
						.job \
						output \
						$(DATADIR)/*assemblages.csv \
						$(DATADIR)/regression-info.csv
DATACLEAN = assets \
						log \
						MAGEMin \
						runs
FIGSPURGE =
FIGSCLEAN = figs

all: $(LOGFILE) $(PYTHON) create_conda_env $(DATADIR) $(CONFIG) $(PERPLEX) $(MAGEMIN)
	@echo "=============================================" $(LOG)
	@$(CONDAPYTHON) python/session-info.py $(LOG)
	@echo "=============================================" $(LOG)
	@$(MAKE) benchmark SAMPLEID=PUM TRES=128 PRES=128
	@$(MAKE) regression SAMPLEID=PUM-128x128 FIGDIR=figs/regression/PUM-128x128
	@$(MAKE) visualize  SAMPLEID=PUM-128x128 FIGDIR=figs/PUM-128x128

init: $(LOGFILE) $(PYTHON) create_conda_env $(DATADIR) $(CONFIG) $(PERPLEX) $(MAGEMIN)
	@echo "=============================================" $(LOG)
	@$(CONDAPYTHON) python/session-info.py $(LOG)
	@echo "=============================================" $(LOG)
	@echo "Run the following in order:" $(LOG)
	@echo "    make benchmark SAMPLEID=PUM TRES=128 PRES=128" $(LOG)
	@echo "    make regression SAMPLEID=PUM-128x128 FIGDIR=figs/regression/PUM-128x128" $(LOG)
	@echo "    make visualize  SAMPLEID=PUM-128x128 FIGDIR=figs/PUM-128x128" $(LOG)
	@echo "To clean up the directory:" $(LOG)
	@echo "    make purge" $(LOG)
	@echo "    make remove_conda_env" $(LOG)
	@echo "=============================================" $(LOG)

visualize: $(LOGFILE) $(PYTHON)
	@for run in $$(ls -d $(OUTDIR)/* | sed 's/$(OUTDIR)\/\(.*\)/\1/'); do \
		echo "Visualizing benchmark $$run" $(LOG); \
		$(MAKE) visualize_database SAMPLEID=$$run FIGDIR=figs/$$run; \
	done
	@$(MAKE) visualize_other

visualize_other: $(LOGFILE) $(PYTHON)
	@$(CONDAPYTHON) python/visualize-other.py $(LOG)
	@echo "=============================================" $(LOG)

visualize_database: $(LOGFILE) $(PYTHON)
	@$(CONDAPYTHON) python/visualize-database.py \
		--sampleid '$(SAMPLEID)' \
		--params '$(PARAMS)' \
		--colormap $(COLORMAP) \
		--outdir $(OUTDIR) \
		--figdir $(FIGDIR) \
		--datadir $(DATADIR) \
	$(LOG)
	@echo "=============================================" $(LOG)

regression:  $(LOGFILE) $(PYTHON)
	@$(CONDAPYTHON) python/regression.py \
		--sampleid '$(SAMPLEID)' \
		--params '$(PARAMSML)' \
		--models '$(MLMODS)' \
		--seed $(SEED) \
		--kfolds $(KFOLDS) \
		--parallel $(PARALLEL) \
		--nprocs $(NPROCS) \
		--colormap $(COLORMAP) \
		--outdir $(OUTDIR) \
		--figdir $(FIGDIR) \
		--datadir $(DATADIR) \
	$(LOG)
	@echo "=============================================" $(LOG)

benchmark_all: $(LOGFILE) $(PYTHON) $(DATADIR) $(CONFIG) $(PERPLEX) $(MAGEMIN)
	@$(MAKE) benchmark TRES=$(TRES) PRES=$(PRES) SAMPLEID=DMM
	@$(MAKE) benchmark TRES=$(TRES) PRES=$(PRES) SAMPLEID=NMORB
	@$(MAKE) benchmark TRES=$(TRES) PRES=$(PRES) SAMPLEID=PUM
	@$(MAKE) benchmark TRES=$(TRES) PRES=$(PRES) SAMPLEID=RE46

benchmark: $(LOGFILE) $(PYTHON) $(DATADIR) $(CONFIG) $(PERPLEX) $(MAGEMIN)
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
		$(CONDAPYTHON) python/benchmark.py \
			--Pmin $(PMIN) \
			--Pmax $(PMAX) \
			--Pres $(PRES) \
			--Tmin $(TMIN) \
			--Tmax $(TMAX) \
			--Tres $(TRES) \
			--sampleid $(SAMPLEID) \
			--normox '$(NORMOX)' \
			--parallel $(PARALLEL) \
			--nprocs $(NPROCS) \
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
			awk '{printf "%.1f", $$NF/1000}')" >> $(DATADIR)/benchmark-times.csv; \
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
			}' $(DATADIR)/benchmark-comps.csv > sample-data && \
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
			awk '{printf "%.1f", $$NF*60}')" >> $(DATADIR)/benchmark-times.csv; \
		echo "=============================================" $(LOG); \
		echo "Finished perplex model ..." $(LOG); \
		echo "Finished benchmarking $(SAMPLEID) ..." $(LOG); \
		echo "=============================================" $(LOG); \
	fi
	@echo "To visualize benchmark, run:" $(LOG)
	@echo "make visualize_database \
	SAMPLEID=$(SAMPLEID)-$(TRES)x$(PRES) \
	FIGDIR=$(FIGDIR)/$(SAMPLEID)-$(TRES)x$(PRES) \
	COLORMAP=$(COLORMAP)" $(LOG)
	@echo "=============================================" $(LOG)

build_database: $(LOGFILE) $(PYTHON) $(DATADIR) $(MAGEMIN)
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

write_tables: $(LOGFILE)
	@echo "Writing markdown tables ..." $(LOG)
	@$(CONDAPYTHON) python/write-markdown-tables.py

submit_jobs: $(LOGFILE) $(PYTHON) $(DATADIR)
	@echo "Submitting job to SLURM ..." $(LOG)
	@$(CONDAPYTHON) python/submit-jobs.py $(LOG)
	@echo "=============================================" $(LOG)

$(MAGEMIN): $(LOGFILE) $(PYTHON)
	@if [ ! -e "$(MAGEMIN)" ]; then \
		echo "=============================================" $(LOG); \
		echo "Cloning MAGEMin from:" $(LOG); \
		echo "	$(MAGEMINREPO)" $(LOG); \
		chmod +x python/clone-magemin.py; \
		$(CONDAPYTHON) python/clone-magemin.py $(LOG); \
	else \
		echo "MAGEMin found at:" $(LOG); \
		echo "	$(MAGEMIN)" $(LOG); \
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
		echo "	$(MY_ENV_DIR)" $(LOG); \
	else \
		echo "Creating environment $(CONDAENVNAME) from:" $(LOG); \
		echo "	$(CONDASPECSFILE)" $(LOG); \
		conda env create --file $(CONDASPECSFILE) $(LOG); \
		echo "	Conda environment $(CONDAENVNAME) created ..." $(LOG); \
	fi
	@echo "=============================================" $(LOG)

find_conda_env: $(LOGFILE)
	@echo "Looking for conda environment $(CONDAENVNAME)..." $(LOG)
	$(eval MY_ENV_DIR := $(shell conda env list | grep $(CONDAENVNAME) | awk '{print $$2}'))
	@echo "=============================================" $(LOG)

$(PERPLEX): $(LOGFILE) $(PYTHON)
	@if [ ! -d "$(DATADIR)" ]; then \
		$(CONDAPYTHON) python/download-assets.py $(LOG); \
	else \
		echo "Perplex found at:" $(LOG); \
		echo "	$(PERPLEX)" $(LOG); \
	fi

$(DATADIR): $(LOGFILE) $(PYTHON)
	@if [ ! -d "$(DATADIR)" ]; then \
		echo "Downloading assets from:" $(LOG); \
		echo "	$(DATAURL)" $(LOG); \
		$(CONDAPYTHON) python/download-assets.py $(LOG); \
	else \
		echo "Data files found at:" $(LOG); \
		echo "	$(DATADIR)" $(LOG); \
	fi

$(CONFIG): $(LOGFILE) $(PYTHON)
	@if [ ! -e "$(CONFIG)" ]; then \
		$(CONDAPYTHON) python/download-assets.py $(LOG); \
		echo "=============================================" $(LOG); \
	else \
		echo "Configuration files found at:" $(LOG); \
		echo "	$(CONFIG)" $(LOG); \
	fi

$(LOGFILE):
	@if [ ! -e "$(LOGFILE)" ]; then \
		mkdir -p log; \
		touch $(LOGFILE); \
	fi

purge:
	@rm -rf $(DATAPURGE) $(FIGSPURGE)

clean: purge
	@rm -rf $(DATACLEAN) $(FIGSCLEAN)

.PHONY: find_conda_env remove_conda_env create_conda_env write_tables submit_jobs build_database benchmark benchmark_all regression visualize_database visualize_other visualize init all purge clean