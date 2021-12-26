#!/usr/bin/env python

from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: GPL-v3 License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "linescanning: a package for line-scanning fMRI"
# Long description will go up on the pypi page
long_description = """

linescanning
========
linescanning is a package specifically built for the analysis of line-scanning
fMRI data. The idea is to acquire line data based on earlier acquired popu-
lation receptive field (pRF) data and minimal curvature in the brain, for
which we acquire functional runs with pRF-mapping and high resolution ana-
tomical scans that are processed with fmriprep, pRFpy, and FreeSurfer.

To get started using these components in your own software, please go to the
repository README_.

.. _README: https://github.com/tknapen/linescanning/master/README.md

License
=======
``linescanning`` is licensed under the terms of the GPL-v3 license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2019--, Tomas Knapen,
Spinoza Centre for Neuroimaging, Amsterdam.
"""

NAME = "linescanning"
MAINTAINER = "Tomas Knapen"
MAINTAINER_EMAIL = "tknapen@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/tknapen/linescanning"
DOWNLOAD_URL = ""
LICENSE = "GPL3"
AUTHOR = "Tomas Knapen"
AUTHOR_EMAIL = "tknapen@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'linescanning': [pjoin('test', 'data', '*')]}
REQUIRES = ["numpy","scipy","pandas", "nilearn", "nibabel"]
