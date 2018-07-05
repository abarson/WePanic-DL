update: $(wildcard *.tex)
	mkdir -p out
	pdflatex --output-directory=out update.tex
	cp out/update.pdf update.pdf

.PHONY: clean
clean:
	rm -f update.pdf
	rm -rf out/
