# Neural Prnu Extractor Presentation

In order to generate the presentation slides, be sure you have [pandoc](https://pandoc.org/) and [LaTeX](https://www.latex-project.org/) installed.

To check the requirements you can run the following lines (the output will depend on the version you have installed):

```sh
pandoc --version

pandoc 2.14.0.3
```

```sh
pdflatex --version

pdfTeX 3.141592653-2.6-1.40.22 (TeX Live 2021)
```

## Generate the slides

To generate the beamer presentation run the following command:

```sh
pandoc main.md --include-in-header=./preamble.tex \
-t beamer -o main.pdf
```

Once the slides have been generated, you can open them with your favourite document viewer (for instance [Zathura](https://pwmt.org/projects/zathura/installation/))

```sh
zathura main.pdf
```
