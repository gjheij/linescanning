.. linescanning documentation master file
.. fmriprep documentation master file, created by
   sphinx-quickstart on Mon May  9 09:04:25 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. include:: links.rst

Welcome to the linescanning repository
===========================================

.. image:: ../imgs/overview.png

This repository contains all of the tools used during the acquisition and postprocessing of line scanning data at the Spinoza Centre for Neuroimaging in Amsterdam. The main goal of the package is to create the most accurate segmentations (both volumetric and surface) by combining various software packages such as Nighres_, fMRIprep_, FreeSurfer_, CAT12_, and SPM_, and analyze line-scanning data (mainly including GLMs and deconvolutions with Nideconv_). Also has the capability to read in any type of input file (`npy`, `gii`, `nii`, `nii.gz`, `np.ndarray`), and return a dataframe index on subject, run, and voxels so it's compatible with Nideconv_.

Contents
--------

.. toctree::
   :maxdepth: 3
   :caption: Getting started

   installation
   usage

.. toctree::
   :maxdepth: 1
   :caption: Example gallery

   examples/acompcor
   examples/eyetracker
   examples/genericglm
   examples/lazyplot
   examples/nideconv
   examples/prfmodelfitter
   examples/prf_verify
   
.. toctree::
   :maxdepth: 2
   :caption: Reference

   bash
   classes/index
