# Makefile config
SHELL = /bin/bash -o pipefail
# Logging
LOGFILE := assets/log/log-$(shell date +%Y-%m-%d)
LOG := 2>&1 | tee -a $(LOGFILE)
# Notes and manuscript draft
BIB = assets/bib/main.bib
CSL = assets/bib/agu.csl
NOTESTEMPLATE = $(HOME)/.pandoc/templates/eisvogel.latex
NOTESTEMPLATEREPO = https://github.com/Wandmalfarbe/pandoc-latex-template/releases/latest/download/Eisvogel.tar.gz
# Make clean
DATAPURGE = Eisvogel Eisvogel.tar.gz notes.pdf
DATACLEAN = 
FIGSPURGE =
FIGSCLEAN =

notes: $(LOGFILE) $(NOTESTEMPLATE) notes.md
	@pandoc notes.md \
		-o notes.pdf \
		--from markdown \
		--template eisvogel \
		--number-sections \
		--highlight-style tango \
		--filter pandoc-crossref \
		--citeproc \
		$(LOG)

$(NOTESTEMPLATE): $(LOGFILE)
	@if [ ! -d $(HOME)/.pandoc/templates ]; then \
		mkdir -p $(HOME)/.pandoc/templates; \
	fi
	@if [ ! -e $(NOTESTEMPLATE) ]; then \
		curl -LJO $(NOTESTEMPLATEREPO) $(LOG); \
		mkdir Eisvogel; \
		tar -xzf Eisvogel.tar.gz -C Eisvogel $(LOG); \
		cp Eisvogel/eisvogel.latex $(NOTESTEMPLATE); \
		rm -rf Eisvogel.tar.gz Eisvogel; \
	fi

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

.PHONY: notes all purge clean