# line scanning repository

[![Documentation Status](https://readthedocs.org/projects/linescanning/badge/?version=latest)](https://linescanning.readthedocs.io/en/latest/?badge=latest)

![plot](https://github.com/gjheij/linescanning/blob/docs/examples/figures/overview.png)

This repository contains all of the tools used during the acquisition and postprocessing of line scanning data at the Spinoza Centre for Neuroimaging in Amsterdam. The main goal of the package is to create the most accurate segmentations (both volumetric and surface) by combining various software packages such as [Nighres](https://github.com/nighres/nighres), [fMRIprep](https://fmriprep.org/en/stable/usage.html), [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/), [CAT12](http://www.neuro.uni-jena.de/cat/index.html#DOWNLOAD), and [SPM](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/). 

## In active development
This package is still in development and its API might change. Documentation for this package can be found at [readthedocs](https://linescanning.readthedocs.io/en/latest/) (not up to date)

## Installation
To install, clone the repository and run `bash linescanning/shell/spinoza_setup setup`. This will make the executables in the `bin` and `shell` folders available, install additionally required packages (such as `pRFpy`, `Pycortex`, `Nighres`, `Pybest`, `ITK-Snap`, and `Nideconv`). You can either choose to activate the accompanying `environment.yml`-file (`ACTIVATE_CONDA=1` in `spinoza_setup`; = Default!) or install it in your own environment/python installation (set `ACTIVATE_CONDA=0` in `spinoza_setup`). Installations of `FSL`, `SPM` (+`CAT12`-toolbox), `fMRIprep` and `FreeSurfer` are expected to exist on your system.

## Policy & To Do

- [x] install using `python setup.py develop`
- [x] Docstrings in numpy format.
- [x] PEP8 - please set your editor to autopep8 on save!
- [x] Write docs for glm.py
- [x] Sphinx doesn't show source code for python (`FileNotFoundError: [Errno 2] No such file or directory: '/home/docs/.config/pycortex'`)
- [x] Put `nbsphinx` in `requirements` if you want to build html's from notebooks with Sphinx
- [ ] Explore options to streamline code
- [ ] Examples of applications for package (add notebooks to `doc/source/examples`)