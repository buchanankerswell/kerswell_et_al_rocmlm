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
CONDAENVNAME = rocml
HASCONDA := $(shell command -v conda > /dev/null && echo true || echo false)
CONDASPECSFILE = $(WORKDIR)/python/conda-environment.yaml
CONDAPYTHON = $$(conda run -n $(CONDAENVNAME) which python)
# Magemin program
MAGEMIN = $(WORKDIR)/MAGEMin
# Perplex program
PERPLEX = $(WORKDIR)/assets/perplex
# Directories with data and scripts
DATADIR = assets/data
DATAURL = https://files.osf.io/v1/resources/k23tb/providers/osfstorage/649149796513ba03733a3536/?zip=
CONFIG = $(WORKDIR)/assets/config
PYTHON = $(WORKDIR)/python/build-magemin-database.py \
				 $(WORKDIR)/python/build-database.py \
				 $(WORKDIR)/python/clone-magemin.py \
				 $(WORKDIR)/python/download-assets.py \
				 $(WORKDIR)/python/rocml.py \
				 $(WORKDIR)/python/train-benchmark-rocmls.py \
				 $(WORKDIR)/python/session-info.py \
				 $(WORKDIR)/python/submit-jobs.py \
				 $(WORKDIR)/python/visualize-database.py \
				 $(WORKDIR)/python/visualize-other.py \
# Other variables
GITHUBREPO = https://github.com/buchanankerswell/kerswell_et_al_rocml
MAGEMINREPO = https://github.com/ComputationalThermodynamics/MAGEMin.git
# Dataset build options
SAMPLEID ?= PUM
PMIN ?= 1
PMAX ?= 28
TMIN ?= 773
TMAX ?= 2273
RES ?= 128
DATASET ?= train
NORMOX ?= all
SEED = 42
PARALLEL ?= True
NPROCS ?= $(shell expr $(shell nproc) - 2)
KFOLDS ?= $(shell expr $(shell nproc) - 2)
OUTDIR ?= runs
# RocML options
PARAMSML ?= ["DensityOfFullAssemblage"]
MLMODS ?= ["KN", "RF", "DT", "NN1", "NN2", "NN3"]
MLTUNE ?= False
# Bulk rock composition sampling options
COMP ?= [44.9, 4.44, 3.54, 37.71, 8.03, 0.029, 0.36, 0.2, 0.01, 0.38, 0]
FRAC ?= wt
SOURCE ?= earthchem
STRATEGY ?= random
N ?= 1
K ?= 0
# Visualization options
FIGDIR ?= figs
PARAMS ?= ["StableSolutions", "StableVariance", "DensityOfFullAssemblage"]
COLORMAP ?= bone
# Cleanup directories
DATAPURGE = python/__pycache__ \
						.job \
						output \
						$(DATADIR)/*assemblages.csv \
						$(DATADIR)/benchmark-rocmls-performance.csv \
						$(DATADIR)/benchmark-gfem-efficiency-$(DATE).csv \

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
	@$(MAKE) magemin_database DATASET=train
	@$(MAKE) magemin_database DATASET=valid
	@$(MAKE) perplex_database DATASET=train
	@$(MAKE) perplex_database DATASET=valid
	@$(MAKE) train_benchmark_rocmls
	@$(MAKE) visualize_database
	@$(MAKE) visualize_other

init: $(LOGFILE) $(PYTHON) create_conda_env $(DATADIR) $(CONFIG) $(PERPLEX) $(MAGEMIN)
	@echo "=============================================" $(LOG)
	@$(CONDAPYTHON) python/session-info.py $(LOG)
	@echo "=============================================" $(LOG)
	@echo "Run the following in order:" $(LOG)
	@echo "    make magemin_database DATASET=train" $(LOG)
	@echo "    make magemin_database DATASET=valid" $(LOG)
	@echo "    make perplex_database DATASET=train" $(LOG)
	@echo "    make perplex_database DATASET=valid" $(LOG)
	@echo "    make train_benchmark_rocmls" $(LOG)
	@echo "    make visualize_database" $(LOG)
	@echo "    make visualize_other" $(LOG)
	@echo "To clean up the directory:" $(LOG)
	@echo "    make purge" $(LOG)
	@echo "    make remove_conda_env" $(LOG)
	@echo "=============================================" $(LOG)

visualize_other: $(LOGFILE) $(PYTHON)
	@$(CONDAPYTHON) python/visualize-other.py $(LOG)
	@echo "=============================================" $(LOG)

visualize_database: $(LOGFILE) $(PYTHON)
	@$(CONDAPYTHON) python/visualize-database.py \
		--sampleid '$(SAMPLEID)' \
		--res $(RES) \
		--dataset $(DATASET) \
		--params '$(PARAMS)' \
		--colormap $(COLORMAP) \
		--outdir $(OUTDIR) \
		--figdir $(FIGDIR) \
		--datadir $(DATADIR) \
	$(LOG)
	@echo "=============================================" $(LOG)

train_benchmark_rocmls:  $(LOGFILE) $(PYTHON)
	@$(CONDAPYTHON) python/train-benchmark-rocmls.py \
		--sampleid '$(SAMPLEID)' \
		--res $(RES) \
		--params '$(PARAMSML)' \
		--models '$(MLMODS)' \
		--tune $(MLTUNE) \
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

train_all_benchmark_rocmls: $(LOGFILE) $(PYTHON) $(DATADIR) $(CONFIG) $(MAGEMIN) $(PERPLEX)
	@for sample in DMM NMORB PUM RE46; do \
		for res in 8 16 32 64 128; do \
			$(MAKE) train_benchmark_rocmls SAMPLEID=$$sample RES=$$res FIGDIR=figs/$$sample\_$$res; \
		done; \
		$(MAKE) visualize_database SAMPLEID=$$sample RES=$$res FIGDIR=figs/$$sample\_$$res; \
  done

build_all_databases: $(LOGFILE) $(PYTHON) $(DATADIR) $(CONFIG) $(MAGEMIN) $(PERPLEX)
	@for sample in DMM NMORB PUM RE46; do \
		for res in 8 16 32 64 128; do \
			for dataset in train valid; do \
				$(MAKE) magemin_database SAMPLEID=$$sample RES=$$res DATASET=$$dataset; \
				$(MAKE) perplex_database SAMPLEID=$$sample RES=$$res DATASET=$$dataset; \
			done; \
		done; \
  done

magemin_database: $(LOGFILE) $(PYTHON) $(DATADIR) $(CONFIG) $(MAGEMIN)
	@echo "Configuring MAGEMin for $(SAMPLEID) ..." $(LOG)
	@echo "Adding HP endmembers to MAGEMIN ..." $(LOG)
	@cp $(CONFIG)/magemin-init-hp-endmembers $(MAGEMIN)/src/initialize.h $(LOG)
	@echo "Compiling MAGEMin ..." $(LOG)
	@echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" $(LOG)
	@(cd MAGEMin && make) $(LOG)
	@echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" $(LOG)
	@if [ "$(DATASET)" = "valid" ]; then \
		mkdir -p $(OUTDIR)/$(SAMPLEID)/magemin_valid_$(RES); \
		PSTEP=1; \
		TSTEP=25; \
		PMIN_TRANS=$$(echo "$(PMIN) + $$PSTEP" | bc); \
		PMAX_TRANS=$$(echo "$(PMAX) - $$PSTEP" | bc); \
		TMIN_TRANS=$$(echo "$(TMIN) + $$TSTEP" | bc); \
		TMAX_TRANS=$$(echo "$(TMAX) - $$TSTEP" | bc); \
		echo "Shifting training dataset by a small amount:" $(LOG); \
		echo "Pstep: $$PSTEP" $(LOG); \
		echo "Tstep: $$TSTEP" $(LOG); \
		echo "Pmin: $(PMIN) --> Pmin: $$PMIN_TRANS" $(LOG); \
		echo "Pmax: $(PMAX) --> Pmax: $$PMAX_TRANS" $(LOG); \
		echo "Tmin: $(TMIN) --> Tmin: $$TMIN_TRANS" $(LOG); \
		echo "Tmax: $(TMAX) --> Tmax: $$TMAX_TRANS" $(LOG); \
		echo "Building MAGEMin validation dataset ..." $(LOG); \
		$(CONDAPYTHON) python/build-magemin-database.py \
			--Pmin $$PMIN_TRANS \
			--Pmax $$PMAX_TRANS \
			--Tmin $$TMIN_TRANS \
			--Tmax $$TMAX_TRANS \
			--res $(RES) \
			--sampleid $(SAMPLEID) \
			--normox '$(NORMOX)' \
			--parallel $(PARALLEL) \
			--nprocs $(NPROCS) \
			--outdir $(OUTDIR) \
			$(LOG); \
		(cd $(OUTDIR)/$(SAMPLEID) && mv _$(SAMPLEID)* magemin_valid_$(RES)/); \
		(cd $(OUTDIR)/$(SAMPLEID) && mv $(SAMPLEID).dat magemin_valid_$(RES)/); \
	else \
		mkdir -p $(OUTDIR)/$(SAMPLEID)/magemin_train_$(RES); \
		echo "Building MAGEMin train dataset ..." $(LOG); \
		$(CONDAPYTHON) python/build-magemin-database.py \
			--Pmin $(PMIN) \
			--Pmax $(PMAX) \
			--Tmin $(TMIN) \
			--Tmax $(TMAX) \
			--res $(RES) \
			--sampleid $(SAMPLEID) \
			--normox '$(NORMOX)' \
			--parallel $(PARALLEL) \
			--nprocs $(NPROCS) \
			--outdir $(OUTDIR) \
			$(LOG); \
		(cd $(OUTDIR)/$(SAMPLEID) && mv _$(SAMPLEID)* magemin_train_$(RES)/); \
		(cd $(OUTDIR)/$(SAMPLEID) && mv $(SAMPLEID).dat magemin_train_$(RES)/); \
	fi
	@echo "$(SAMPLEID),magemin,$$(($(RES) * $(RES))),$$( \
		grep -oE "MAGEMin comp time: \+([0-9.]+) ms }" $(LOGFILE) | \
		tail -n 1 | \
		sed -E 's/MAGEMin comp time: \+([0-9.]+) ms }/\1/' | \
		awk '{printf "%.1f", $$NF/1000*$(NPROCS)}')" >> \
		$(DATADIR)/benchmark-gfem-efficiency-$(DATE).csv
	@echo "Finished MAGEMin model ..." $(LOG)
	@echo "=============================================" $(LOG)
	@echo "To visualize benchmark, run:" $(LOG)
	@echo "make visualize_database \
	SAMPLEID=$(SAMPLEID) \
	FIGDIR=figs/$(SAMPLEID) \
	COLORMAP=$(COLORMAP)" $(LOG)
	@echo "=============================================" $(LOG)

perplex_database: $(LOGFILE) $(PYTHON) $(DATADIR) $(CONFIG) $(PERPLEX)
	@if [ "$(DATASET)" = "valid" ]; then \
		mkdir -p $(OUTDIR)/$(SAMPLEID)/perplex_valid_$(RES); \
		PSTEP=1; \
		TSTEP=25; \
		PMIN_TRANS=$$(echo "$(PMIN) + $$PSTEP" | bc); \
		PMAX_TRANS=$$(echo "$(PMAX) - $$PSTEP" | bc); \
		TMIN_TRANS=$$(echo "$(TMIN) + $$TSTEP" | bc); \
		TMAX_TRANS=$$(echo "$(TMAX) - $$TSTEP" | bc); \
		echo "Shifting training dataset by a small amount:" $(LOG); \
		echo "Pstep: $$PSTEP" $(LOG); \
		echo "Tstep: $$TSTEP" $(LOG); \
		echo "Pmin: $(PMIN) --> Pmin: $$PMIN_TRANS" $(LOG); \
		echo "Pmax: $(PMAX) --> Pmax: $$PMAX_TRANS" $(LOG); \
		echo "Tmin: $(TMIN) --> Tmin: $$TMIN_TRANS" $(LOG); \
		echo "Tmax: $(TMAX) --> Tmax: $$TMAX_TRANS" $(LOG); \
		echo "Building perplex validation dataset ..." $(LOG); \
		chmod +x $(PERPLEX)/build $(PERPLEX)/vertex $(PERPLEX)/pssect $(PERPLEX)/werami; \
		(cd $(PERPLEX) && \
			cp $(CONFIG)/perplex-build-endmembers perplex-build-$(SAMPLEID) && \
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
				-v tmin="$$TMIN_TRANS" \
				-v tmax="$$TMAX_TRANS" \
				-v pmin="$$(echo $$PMIN_TRANS*10000 | bc)" \
				-v pmax="$$(echo $$PMAX_TRANS*10000 | bc)" \
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
			}' $(WORKDIR)/$(DATADIR)/benchmark-samples.csv > sample-data && \
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
			awk -v tres=$(RES) -v pres=$(RES) 'BEGIN { \
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
			mv $(SAMPLEID)* $(WORKDIR)/$(OUTDIR)/$(SAMPLEID)/perplex_valid_$(RES)/ \
		) $(LOG); \
	else \
		mkdir -p $(OUTDIR)/$(SAMPLEID)/perplex_train_$(RES); \
		echo "Building perplex model ..." $(LOG); \
		chmod +x $(PERPLEX)/build $(PERPLEX)/vertex $(PERPLEX)/pssect $(PERPLEX)/werami; \
		(cd $(PERPLEX) && \
			cp $(CONFIG)/perplex-build-endmembers perplex-build-$(SAMPLEID) && \
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
			}' $(WORKDIR)/$(DATADIR)/benchmark-samples.csv > sample-data && \
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
			awk -v tres=$(RES) -v pres=$(RES) 'BEGIN { \
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
			mv $(SAMPLEID)* $(WORKDIR)/$(OUTDIR)/$(SAMPLEID)/perplex_train_$(RES)/ \
		) $(LOG); \
	fi
	@echo "$(SAMPLEID),perplex,$$(($(RES) * $(RES))),$$( \
		grep -oE "Total elapsed time\s+([0-9.]+)" $(LOGFILE) | \
		tail -n 1 | \
		sed -E 's/Total elapsed time\s+([0-9.]+)/\1/' | \
		awk '{printf "%.1f", $$NF*60}')" >> \
		$(DATADIR)/benchmark-gfem-efficiency-$(DATE).csv
	@echo "Finished perplex model ..." $(LOG)
	@echo "=============================================" $(LOG)
	@echo "To visualize benchmark, run:" $(LOG)
	@echo "make visualize_database \
	SAMPLEID=$(SAMPLEID) \
	FIGDIR=figs/$(SAMPLEID) \
	COLORMAP=$(COLORMAP)" $(LOG)
	@echo "=============================================" $(LOG)

build_database: $(LOGFILE) $(PYTHON) $(DATADIR) $(MAGEMIN)
	@$(CONDAPYTHON) python/build-database.py \
		--Pmin $(PMIN) \
		--Pmax $(PMAX) \
		--Tmin $(TMIN) \
		--Tmax $(TMAX) \
		--res $(RES) \
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

.PHONY: find_conda_env remove_conda_env create_conda_env write_tables submit_jobs build_database train_all_benchmark_rocmls train_benchmark_rocmls visualize_database visualize_other init all purge clean