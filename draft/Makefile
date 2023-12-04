# Logging config
DATE = $(shell date +"%d-%m-%Y")
LOGFILE := log/log
LOG := 2>&1 | tee -a log/log
# Conda config
CONDAENVNAME = rocmlm
HASCONDA := $(shell command -v conda > /dev/null && echo true || echo false)
CONDASPECSFILE = python/conda-environment.yaml
CONDAPYTHON = $$(conda run -n $(CONDAENVNAME) which python)
# Manuscript
MS = rocmlm
# Python scripts
PYTHON = \
				 python/draft.py \
				 python/place-md-tables.py \
				 python/copy-assets-and-figs.py \
# Cleanup directories
DATAPURGE = \
						log \
						tmp.md \
						python/__pycache__ \
						agu-draft-rocmlm.pdf \
						eisvogel-draft-rocmlm.pdf \
						assets/pandoc/mlm-pro-con.md \
						assets/pandoc/rocmlm-performance.md \
						assets/pandoc/gfem-efficiency.md \
						assets/pandoc/benchmark-samples.md \
						assets/pandoc/earthchem-counts.md \
DATACLEAN = assets/data
FIGSPURGE =
FIGSCLEAN = assets/figs

all: $(LOGFILE) $(EIMETA) $(EITEMPLATE) $(MS).md place_md_tables eisvogel

agu: $(LOGFILE) $(AGUMETA) $(AGUTEMPLATE) $(MS).md $(TABLES)
	@pandoc $(MS).md \
		-o agu-draft-$(MS).pdf \
		--quiet \
		--from markdown \
		--pdf-engine pdflatex \
		--metadata-file assets/pandoc/agu.yaml \
		--metadata date=$(DATE) \
		--template assets/tex/template-agu.latex \
		--number-sections \
		--highlight-style tango \
		--filter pandoc-crossref \
		--citeproc \
		$(LOG)
	@echo "=============================================" $(LOG)

eisvogel: $(LOGFILE) $(EIMETA) $(EITEMPLATE) $(MS).md place_md_tables
	@pandoc tmp.md \
		-o eisvogel-draft-$(MS).pdf \
		--quiet \
		--from markdown \
		--pdf-engine pdflatex \
		--metadata-file assets/pandoc/eisvogel.yaml \
		--metadata date=$(shell date +"%d-%B-%Y") \
		--template assets/tex/template-eisvogel.latex \
		--number-sections \
		--highlight-style tango \
		--filter pandoc-crossref \
		--citeproc \
		$(LOG)
	@rm tmp.md
	@echo "=============================================" $(LOG)

place_md_tables: $(LOGFILE) copy_assets_and_figs
	@$(CONDAPYTHON) -u python/place-md-tables.py --ms $(MS) $(LOG)
	@echo "=============================================" $(LOG)

copy_assets_and_figs: $(LOGFILE) $(PYTHON) $(MS).md
	@$(CONDAPYTHON) -u python/copy-assets-and-figs.py --ms $(MS) $(LOG)
	@echo "=============================================" $(LOG)

create_conda_env: $(LOGFILE) $(CONDASPECSFILE) find_conda_env
	@if [ "$(HASCONDA)" = "false" ]; then \
		echo "Install conda first!" $(LOG); \
		echo "See: https://github.com/buchanankerswell/kerswell_et_al_rocml" $(LOG); \
		exit 1; \
	fi
	@if [ -d "$(MYENVDIR)" ]; then \
		echo "Conda environment \"$(CONDAENVNAME)\" found!" $(LOG); \
	else \
		echo "Creating environment $(CONDAENVNAME) ..." $(LOG); \
		conda env create --file $(CONDASPECSFILE) $(LOG) > /dev/null 2>&1; \
		echo "Conda environment $(CONDAENVNAME) created!" $(LOG); \
	fi

find_conda_env: $(LOGFILE)
	$(eval MYENVDIR := $(shell conda env list | grep $(CONDAENVNAME) | awk '{print $$2}'))

$(LOGFILE):
	@if [ ! -e "$(LOGFILE)" ]; then \
		mkdir -p log; \
		touch $(LOGFILE); \
	fi

purge:
	@rm -rf $(DATAPURGE) $(FIGSPURGE)

clean: purge
	@rm -rf $(DATACLEAN) $(FIGSCLEAN)
	@rm -f *.{html,acr,alg,gz,glg,gls,ilg,nls,acn,glo,ist,lof,lot,nlo,aux,dvi,log,bbl,blg,brf,fls,toc,thm,out,fdb_latexmk,tex,pdf}

.PHONY: clean purge find_conda_env create_conda_env copy_assets_and_figs place_md_tables eisvogel agu all