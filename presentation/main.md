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
- \today
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
logo:
- unitn.pdf
fontsize: 
- 10mm
bibliography: bibliography.bib
link-citations: true
nocite: |
  @*
---

# Introduction

## Noise extraction
Noise reduction is the process of removing noise from a signal. Images taken with both digital cameras and conventional film cameras will pick up noise from a variety of sources and can be (partially) removed for practical purposes such as computer vision [@enwiki:1065098521]

## PRNU
Photo response non-uniformity, pixel response non-uniformity, or PRNU, is a form of fixed-pattern noise related to digital image sensors, as used in cameras and optical instruments [@enwiki:1048256617].
In forensics PRNU is extracted from a set of images taken by the same camera device, with the purpose of identifying which device generated an image.

# Objective
Train a model to extract the PRNU noise for device identification.

![PRNU extraction pipeline](./imgs/prnu_extraction_pipeline.pdf)

# Residual FFDNet [@ipol.2019.231]

![FFDNet layers](./imgs/ffdnet.png)

## Reasons
* able to remove spatially variant noise, specifying a non-uniform noise level map
* works with low noise levels ($\sigma \in [0, 75]$)
* uses state of the art CNN structures as residual connections, which allows for more stable PSNR during training

# Residual FFDNet - Changes

::: {.columns align=center}

:::: column

![ ](imgs/learning_rate.pdf)

::::

:::: column

## ReduceLROnPlateau
Instead of manually reducing the learning rate, we introduced this function in order to decrease it automatically after a *patience* threshold.

::::

:::

# Residual FFDNet - Changes [Cont...]

## Resource usage
To work in a limited environment, we introduced a stopping criterion if the requested resources exceeded a user-fixed limit (different from the machine one).

## Different experiments
To handle different training sessions, we have created an argparser to specify several parameters (e.g. experiment name, datasets, epochs, gpu fraction, ...)

# Residual FFDNet - Changes [Cont...]

## Wiener as groundtruth
As suggested by Prof. De Natale, we trained FFDNet by using the Wiener filter to produce groundtruth images (denoised). The generated images are used for computing the loss of the net

$$\mathcal{L}_{res}(\theta) = \frac{1}{2m}\sum_{j=1}^{m}{\parallel \mathcal{F}((\tilde{\mathrm{I}}_j, \mathrm{M}_j); \theta) - \mathrm{N}_j \parallel^2}$$

## Green channel
In order to work with the Wiener filter we had to reduce images to grayscale. The approach that we followed was simply to extract the green channel, without taking into account possible contributions of the others.

# PRNU Extraction

# PRNU Extraction - Changes

# Training

# Results

| **Technique** | **Sigma** | **Cut Dim** | **CC** | **PCE** | **Time** |
|:--------------|:---------:|:-----------:|:------:|:-------:|:--------:|
| Wavelet       | 5         | 512 x 512   | 0.97   | 0.96    | 04m 49s  |
| Neural Wiener | Adaptive  | 512 x 512   | 0.96   | 0.95    | 05m 20s  |
| Neural Wiener | Adaptive  | 256 x 256   | 0.94   | 0.83    | 03m 18s  |
| Neural Wiener | Adaptive  | 1024 x 1024 | 0.98   | 0.96    | 15m 18s  |
| Neural Wiener | 5         | 512 x 512   | 0.95   | 0.92    | 05m 12s  |
| Neural 0-5    | Adaptive  | 512 x 512   | 0.93   | 0.92    | 05m 14s  |
| Neural 0-5    | 5         | 512 x 512   | 0.95   | 0.97    | 05m 17s  |
| FFDNet        | Adaptive  | 512 x 512   | 0.96   | 0.94    | 05m 03s  |
| FFDNet        | 5         | 512 x 512   | 0.96   | 0.96    | 04m 57s  |

# How to generate it

To generate the presentation, just execute:

```sh
pandoc main.md --include-in-header=preamble.tex \ 
--citeproc --bibliography=bibliography.bib -t \
beamer -o main.pdf
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

# References {.allowframebreaks}