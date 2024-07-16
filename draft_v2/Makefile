MS = kerswell-et-al-rocmlm
SUP = kerswell-et-al-rocmlm-supporting
DIFFMS = ../draft/kerswell-et-al-rocmlm
DIFFSUP = ../draft/kerswell-et-al-rocmlm-supporting

all: ms sup

ms: $(MS).pdf
	@open $<

sup: $(SUP).pdf
	@open $<

diff: diff-ms.pdf diff-sup.pdf
	@open $<

$(MS).pdf: $(MS).Rmd
	@R -q -e 'rmarkdown::render("$(MS).Rmd", quiet=T, clean=F)'

$(MS).tex: $(MS).Rmd
	@R -q -e 'rmarkdown::render("$(MS).Rmd", quiet=T, clean=F)'

$(SUP).pdf: $(SUP).Rmd
	@R -q -e 'rmarkdown::render("$(SUP).Rmd", quiet=T, clean=F)'

$(SUP).tex: $(SUP).Rmd
	@R -q -e 'rmarkdown::render("$(SUP).Rmd", quiet=T, clean=F)'

diff-sup.pdf: diff-sup.tex
	@echo "pdflatex for diff-sup.tex ..."
	@pdflatex diff-sup.tex > /dev/null 2>&1
	@bibtex diff-sup > /dev/null 2>&1
	@pdflatex diff-sup.tex > /dev/null 2>&1
	@pdflatex diff-sup.tex > /dev/null 2>&1

diff-ms.pdf: diff-ms.tex
	@echo "pdflatex for diff-ms.tex ..."
	@pdflatex diff-ms.tex > /dev/null 2>&1
	@bibtex diff-ms > /dev/null 2>&1
	@pdflatex diff-ms.tex > /dev/null 2>&1
	@pdflatex diff-ms.tex > /dev/null 2>&1

diff-ms.tex: $(MS).tex $(DIFFMS).tex
	@echo "latexdiff for $(DIFFMS).tex vs. $(MS).tex ..."
	@latexdiff --type=CFONT --disable-citation-markup --graphics-markup=none --allow-spaces \
		$(DIFFMS).tex $(MS).tex > diff-ms.tex

diff-sup.tex: $(SUP).tex $(DIFFSUP).tex
	@echo "latexdiff for $(DIFFSUP).tex vs. $(MS).tex ..."
	@latexdiff --type=CFONT --disable-citation-markup --graphics-markup=none --allow-spaces \
		$(DIFFSUP).tex $(SUP).tex > diff-sup.tex

get_tables:
	@$(MAKE) -C .. write_md_tables

purge:
	@rm -f *.{html,acr,alg,gz,glg,gls,ilg,nls,acn,glo,ist,lof,lot,nlo,aux,dvi,log,bbl,blg,brf,fls,toc,thm,out,fdb_latexmk,pdf} *.knit.md $(MS).tex $(SUP).tex diff-ms.tex diff-sup.tex

clean: purge

.PHONY: clean purge get_tables diff sup ms all