# Makefile config
SHELL = /bin/bash -o pipefail
WORKDIR = $(shell pwd)
MAINDIR = $(shell dirname $(WORKDIR))
ASSETSDIR = $(WORKDIR)/assets
UNAMES := $(shell uname -s)
DATE = $(shell date +"%d-%m-%Y")
DATELONG = $(shell date +"%d-%B-%Y")
# Logging
LOGDIR = $(ASSETSDIR)/log
LOGFILE := $(LOGDIR)/log-$(shell date +"%d-%m-%Y")
LOG := 2>&1 | tee -a $(LOGFILE)
# Manuscript
MS = madnn
# Figures
FIGDIRMAIN = $(MAINDIR)/figs
FIGDIRDRAFT = $(ASSETSDIR)/figs
# Data
DATADIR = $(ASSETSDIR)/data
# Tex
TEXDIR = $(ASSETSDIR)/tex
AGUTEMPLATE = $(TEXDIR)/template-agu.latex
EITEMPLATE = $(TEXDIR)/template-eisvogel.latex
# Bib
BIBDIR = $(ASSETSDIR)/bib
BIB = $(BIBDIR)/main.bib
CSL = $(BIBDIR)/agu.csl
# Pandoc
PANDIR = $(ASSETSDIR)/pandoc
AGUMETA = $(PANDIR)/agu.yaml
EIMETA = $(PANDIR)/eisvogel.yaml
TABLES = \
	$(PANDIR)/benchmark-comps.md \
	$(PANDIR)/benchmark-times.md \
	$(PANDIR)/regression-info.md \
# Purging and cleaning
DATAPURGE = \
	$(LOGDIR) \
	$(PANDIR)/benchmark-comps.md \
	$(PANDIR)/benchmark-times.md \
	$(PANDIR)/regression-info.md \
	tmp.md \
	eisvogel-draft-madnn.pdf
DATACLEAN = $(DATADIR)
FIGSPURGE =
FIGSCLEAN = $(FIGDIRDRAFT)

agu: $(LOGFILE) $(AGUMETA) $(AGUTEMPLATE) $(MS).md copy_assets $(TABLES)
	@pandoc $(MS).md \
		-o agu-draft-$(MS).pdf \
		--verbose \
		--from markdown \
		--pdf-engine pdflatex \
		--metadata-file $(AGUMETA) \
		--metadata date=$(DATE) \
		--template $(AGUTEMPLATE) \
		--number-sections \
		--highlight-style tango \
		--filter pandoc-crossref \
		--citeproc \
		$(LOG)
	@echo "=============================================" $(LOG)

eisvogel: $(LOGFILE) $(EIMETA) $(EITEMPLATE) $(MS).md copy_assets $(TABLES)
	@awk -v pandir="$(PANDIR)" '{ \
		while (match($$0, /{{ ([^}]+) }}/)) { \
			file = substr($$0, RSTART + 3, RLENGTH - 6); \
			replacement = ""; \
			if (file in filenames) { \
				replacement = filenames[file]; \
			} else { \
				cmd = "cat \047" pandir "/" file "\047"; \
				while ((cmd | getline line) > 0) { \
					replacement = replacement line "\\n"; \
				} \
				close(cmd); \
				filenames[file] = replacement; \
			} \
			gsub(/\\n/, "\n", replacement); \
			sub(/{{ [^}]+ }}/, replacement); \
		} \
		print; \
	}' $(MS).md > tmp.md
	@pandoc tmp.md \
		-o eisvogel-draft-$(MS).pdf \
		--verbose \
		--from markdown \
		--pdf-engine pdflatex \
		--metadata-file $(EIMETA) \
		--metadata date=$(DATELONG) \
		--template $(EITEMPLATE) \
		--number-sections \
		--highlight-style tango \
		--filter pandoc-crossref \
		--citeproc \
		$(LOG)
	@echo "=============================================" $(LOG)

copy_assets: $(LOGFILE) copy_data copy_figs

copy_figs: $(LOGFILE) $(MS).md
	@echo "Copying figs:" $(LOG)
	@mkdir -p $(FIGDIRDRAFT)
	@if [ ! -f "$(MS).md" ]; then \
		echo "Error: The markdown file $(MS).md does not exist ..." $(LOG); \
		exit 1; \
	fi
	@for pngfile in $$(grep -o -E '\([^\)]+\.png\)' $(MS).md | tr -d '()'); do \
		basename=$$(basename $$pngfile); \
		fullpath=$$(find $(FIGDIRMAIN) -type f -name $$basename); \
		if [ -z "$$fullpath" ]; then \
			echo "Warning: $$basename not found in $(FIGDIRMAIN)" $(LOG); \
		else \
			cp "$$fullpath" "$(FIGDIRDRAFT)/$$basename"; \
			echo "	$$basename" $(LOG); \
		fi \
	done

copy_data: $(LOGFILE)
	@echo "Copying data ..." $(LOG)
	@cp -r $(MAINDIR)/assets/data $(ASSETSDIR)
	@(cd $(shell dirname $(WORKDIR)) && $(MAKE) write_tables)

$(LOGFILE):
	@if [ ! -e "$(LOGFILE)" ]; then \
		mkdir -p $(LOGDIR); \
		touch $(LOGFILE); \
	fi

purge:
	@rm -rf $(DATAPURGE) $(FIGSPURGE)

clean: purge
	@rm -rf $(DATACLEAN) $(FIGSCLEAN)
	@rm -f *.{html,acr,alg,gz,glg,gls,ilg,nls,acn,glo,ist,lof,lot,nlo,aux,dvi,log,bbl,blg,brf,fls,toc,thm,out,fdb_latexmk,tex,pdf}

.PHONY: create_conda_env find_conda_env copy_assets eisvogel agu all purge clean