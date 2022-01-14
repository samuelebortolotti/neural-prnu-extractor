---
title:
- Neural PRNU extractor
subtitle:
- A denoising neural network
aspectratio:
- 43
institute:
- University of Trento
author:
- Alghisi Simone
- Bortolotti Samuele
- Rizzoli Massimo
date:
- January 14, 2022
lang: 
- en-US
section-titles: 
- false
theme:
- Copenhagen
colortheme:
- default
navigation:
- horizontal
fontsize: 
- 10pt
---

# How to generate it

To generate the presentation, just execute:

```sh
pandoc main.md --include-in-header=./preamble.tex \
-t beamer -o main.pdf
```

Useful links:

* [Beamer variables](https://www.uv.es/wikibase/doc/cas/pandoc_manual_2.7.3.wiki?20)
* [how to](https://github.com/alexeygumirov/pandoc-beamer-how-to)
* [built in themes](https://mpetroff.net/files/beamer-theme-matrix/)

# Introduction

## Noise extraction

1. First
2. Second
3. Third

## PRNU

Here is a bullet list

* Dot
* Dot
* Dot

Blockquote

  > Blockquote

# Goal

**bold text**

_italic_

# FFDNET

::: columns

:::: {.column width=40%}

Left column text.

Another text line.

::::

:::: {.column width=60%}

- Item 1.
- Item 2.
- Item 3.

::::

:::

# Results

::: columns

:::: column

![Lena](https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png){height=50%}

::::

:::: column

| **Item** | **Option** |
|:---------|:----------:|
| Item 1   | Option 1   |
| Item 2   | Option 2   |

::::

:::

# Appendix

## Appendix content
The appendix contains the topics we are not able to discuss during the oral examination

# Images
![Baboon](https://www.researchgate.net/profile/Andreas-Kleefeld/publication/280083777/figure/fig2/AS:613964962619402@1523392062416/Color-image-baboon-and-its-gray-valued-representation-used-as-transparency.png){ width=50% }
