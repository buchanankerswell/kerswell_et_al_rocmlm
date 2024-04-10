![](draft/repo-banner.png)

***Figure:*** *Density models of a Primitive Upper Mantle composition (PUM, from Sun & McDonough, 1989) estimated by Perple_X ([Connolly, 2009](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2009GC002540)) and a simple three-layer Neural Network.*

# Kerswell et al. (2023; in prep.)

This work is in progress---stay tuned for updates.

## Repository

This repository provides all materials for the manuscript *Using Artificial Intellegence to Model Phase Changes in the Upper Mantle* (Kerswell et al., 2023; in prep.).

This repository includes:

- All datasets required to compile the study
- python scripts to reproduce all results and figures
- A Makefile to compile the study
- The complete manuscript written in Rmarkdown

This repository is self-contained but requires the following software (all open-source).

## Prerequisite software

### Python

This study is written in [python](https://www.python.org). For most users, I recommend installing the [anaconda](https://www.anaconda.com), [miniconda](https://docs.conda.io/en/latest/miniconda.html), or [miniforge](https://github.com/conda-forge/miniforge) python distributions. These distributions include at least a minimal installation of python (plus some dependencies) and the package manager [conda](https://docs.conda.io/en/latest/), which is required to build the necessary python environment for this study. Any of these distributions will work to compile and run the study, and any can be installed (for macOS users) with homebrew.

### Installation with homebrew

Follow the instructions at [Hombrew's homepage](https://brew.sh) to download and install Homebrew on your machine. Once Homebrew is installed, use any of the following to install python and conda:

```
brew install anaconda
brew install miniconda
brew install miniforge
```

## Running the study

```
# Clone this repository
git clone https://github.com/buchanankerswell/kerswell_et_al_rocmlm.git

# Change into the directory
cd kerswell_et_al_rocmlm

# Use Makefile to compile
make
```

This will build the required python environment and proceed to run the study. The study takes about ??? to run on my MacBook Pro (M2 16GB, 2022).

## Coauthors

## Acknowledgement

## Open Research

All data, code, and relevant information for reproducing this work can be found at [https://github.com/buchanankerswell/kerswell_et_al_rocmlm](https://github.com/buchanankerswell/kerswell_et_al_rocmlm), and at [https://doi.org/10.17605/OSF.IO/K23TB](https://doi.org/10.17605/OSF.IO/K23TB), the official Open Science Framework data repository ([Kerswell et al., 2023](https://doi.org/10.17605/OSF.IO/K23TB)). All code is MIT Licensed and free for use and distribution (see license details).

## Abstract

# License

MIT License

Copyright (c) 2023 Buchanan Kerswell

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.