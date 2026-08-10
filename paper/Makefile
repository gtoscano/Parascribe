# Build the paper. Bibliography is inline (thebibliography), so no bibtex pass.
# pdflatex is run twice so \ref/\cite/\label cross-references resolve.

TEX      = main.tex
PDF      = main.pdf
DOCKIMG  = texlive/texlive

.PHONY: all docker clean distclean

all: $(PDF)

$(PDF): $(TEX)
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX)
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX)

# Build inside the official TeX Live container (no local LaTeX required).
docker:
	docker run --rm -v "$(CURDIR):/w" -w /w $(DOCKIMG) \
		sh -c 'pdflatex -interaction=nonstopmode -halt-on-error $(TEX) && \
		       pdflatex -interaction=nonstopmode -halt-on-error $(TEX)'

# Remove build intermediates but keep the PDF.
clean:
	rm -f main.aux main.log main.out main.toc main.lof main.lot \
	      main.fls main.fdb_latexmk main.synctex.gz main.bbl main.blg \
	      main.auxlock *.aux

# Also remove the PDF.
distclean: clean
	rm -f $(PDF)
