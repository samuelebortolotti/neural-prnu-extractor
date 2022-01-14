# Neural Prnu Extractor Presentation

In order to generate the slide be sure you have [pandoc](https://pandoc.org/) and [LaTeX](https://www.latex-project.org/) installed.

## Generate the slides

In order to generate the beamer presentation execute the following command.

```sh
pandoc main.md --include-in-header=./preamble.tex \
-t beamer -o main.pdf
```

