setup_easier_ssh: $(wildcard *.tex)
	mkdir -p out
	pdflatex --output-directory=out setup_easier_ssh.tex
	cp out/setup_easier_ssh.pdf setup_easier_ssh.pdf
	open setup_easier_ssh.pdf

.PHONY: clean
clean:
	rm -f setup_easier_ssh.pdf
	rm -rf out/
