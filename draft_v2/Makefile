MS = kerswell-et-al-rocmlm
SUPP = kerswell-et-al-rocmlm-supporting
DIFF = ../draft/kerswell-et-al-rocmlm

all: paper supp

paper: $(MS).pdf
	@open $<

supp: $(SUPP).pdf
	@open $<

diff: diff.pdf
	@open $<

$(MS).pdf: $(MS).Rmd
	@R -q -e 'rmarkdown::render("$(MS).Rmd", quiet=T, clean=F)'

$(MS).tex: $(MS).Rmd
	@R -q -e 'rmarkdown::render("$(MS).Rmd", quiet=T, clean=F)'

$(SUPP).pdf: $(SUPP).Rmd
	@R -q -e 'rmarkdown::render("$(SUPP).Rmd", quiet=T, clean=F)'

$(SUPP).tex: $(SUPP).Rmd
	@R -q -e 'rmarkdown::render("$(SUPP).Rmd", quiet=T, clean=F)'

diff.pdf: diff.tex
	@printf "\npdflatex for diff.tex ..."
	@pdflatex diff.tex > /dev/null 2>&1
	@bibtex diff > /dev/null 2>&1
	@pdflatex diff.tex > /dev/null 2>&1
	@pdflatex diff.tex > /dev/null 2>&1

diff.tex: $(MS).tex $(DIFF).tex
	@printf "\nlatexdiff for %s vs. %s ..." "$(DIFF).tex" "$(MS).tex"
	@latexdiff --type=CFONT --disable-citation-markup --graphics-markup=none --allow-spaces \
		$(DIFF).tex $(MS).tex > diff.tex

get_tables:
	@$(MAKE) -C .. write_md_tables

purge:
	@rm -f *.{html,acr,alg,gz,glg,gls,ilg,nls,acn,glo,ist,lof,lot,nlo,aux,dvi,log,bbl,blg,brf,\
		fls,toc,thm,out,fdb_latexmk,pdf} *.knit.md $(MS).tex $(SUPP).tex

clean: purge

.PHONY: clean purge get_tables diff supp paper all