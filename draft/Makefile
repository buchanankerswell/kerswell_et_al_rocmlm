# Makefile config
SHELL = /bin/bash -o pipefail
DATE = $(shell date +"%d-%m-%Y")
DATELONG = $(shell date +"%d-%B-%Y")
# Logging
LOGFILE := assets/log/log-$(shell date +"%d-%m-%Y")
LOG := 2>&1 | tee -a $(LOGFILE)
# Manuscript
MS = madnn
# Notes and manuscript draft
BIB = assets/bib/main.bib
CSL = assets/bib/agu.csl
AGUMETA = assets/pandoc/agu.yaml
AGUTEMPLATE = assets/tex/template-agu.latex
EIMETA = assets/pandoc/eisvogel.yaml
EITEMPLATE = assets/tex/template-eisvogel.latex
# Make clean
DATAPURGE = assets/log assets/pandoc/benchmark-comps.md assets/pandoc/benchmark-times.md eisvogel-draft-madnn.pdf
DATACLEAN = 
FIGSPURGE =
FIGSCLEAN = assets/figs

agu: $(LOGFILE) $(AGUMETA) $(AGUTEMPLATE) $(MS).md
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

eisvogel: $(LOGFILE) $(EIMETA) $(EITEMPLATE) $(MS).md
	@pandoc $(MS).md \
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

$(LOGFILE):
	@if [ ! -e "$(LOGFILE)" ]; then \
		mkdir -p assets/log; \
		touch $(LOGFILE); \
	fi

purge:
	@rm -rf $(DATAPURGE) $(FIGSPURGE)

clean: purge
	@rm -rf $(DATACLEAN) $(FIGSCLEAN)
	@rm -f *.{html,acr,alg,gz,glg,gls,ilg,nls,acn,glo,ist,lof,lot,nlo,aux,dvi,log,bbl,blg,brf,fls,toc,thm,out,fdb_latexmk,tex,pdf}

.PHONY: eisvogel agu all purge clean