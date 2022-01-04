# line scanning repository

[![Documentation Status](https://readthedocs.org/projects/linescanning/badge/?version=latest)](https://linescanning.readthedocs.io/en/latest/?badge=latest)

![plot](https://github.com/gjheij/linescanning/blob/master/linescanning/examples/figures/20201106_workflow_acquisition_third_attempt.png)

This repository contains all of the tools used during the acquisition and postprocessing of line scanning data at the Spinoza Centre for Neuroimaging in Amsterdam. The script `master` controls the modules prefixed by `spinoza_`, which in turn call upon various scripts in the `utils` and `bin` directory. The scripts in the latter folders are mostly helper scripts to make life a tad easier. The repository contains a mix of languages in bash, python, and matlab.

## In active development - do not use unless otherwise instructed by repo owners

Documentation for this package can be found at [readthedocs](https://linescanning.readthedocs.io/en/latest/) (not up to date)

## Policy & To Do

- [x] install using `python setup.py develop`
- [x] Docstrings in numpy format.
- [ ] PEP8 - please set your editor to autopep8 on save!
- [ ] Write docs for glm.py
- [ ] Sphinx doesn't show source code for python
- [ ] Explore options to streamline code
- [ ] Examples of applications for package (integration of pycortex & pRFpy)

# overview of the pipeline

## how to set up
Clone the repository: `git clone https://github.com/gjheij/linescanning.git`.

To setup the bash environment, edit setup file `linescanning/shell/spinoza_setup`:
- line 76: add the path to your matlab installation if available (should be, for better anatomicall preprocessing)
- line 87: add the path to your SPM installation
- line 92: add your project name
- line 97: add the path to project name as defined in line 92
- line 102: add whether you're using (ME)MP(2)RAGE. This is required because the pipeline allows the usage of the average of an MP2RAGE and MP2RAGEME acquisition
- line 105: add which type of data you're using (generally this will be the same as line 102)

Go to `linescanning/shell` and hit `./spinoza_setup setup setup`. This will print a set of instructions that you need to follow. If all goes well this will make all the script executable, set all the paths, and install the python modules. The repository comes with a conda environment file, which can be activated with: `conda create --name myenv --file environment.yml`.

# How to plan the line

![plot](https://github.com/gjheij/linescanning/blob/master/linescanning/examples/figures/20201215_detailedintermezzo.png)

We currently aim to have two separate sessions: in the first session, we acquire high resolution anatomical scans and perform a population receptive field (pRF-) mapping paradigm (Dumoulin and Wandell, 2008) to delineate the visual field. After this session, we create surfaces of the brain and map the pRFs onto that via fMRIprep and pRFpy. We then select a certain vertex based on the parameters extracted from the pRF-mapping: eccentricity, size, and polar angle. Using these parameters, we can find an optimal vertex. We can obtain the vertex position, while by calculating the normal vector, we obtain the orientation that line should have (parellel to the normal vector and through the vertex point). Combining this information, we know how the line should be positioned in the first session anatomy. In the second session, we first acquire a low-resolution MP2RAGE with the volume coil. This is exported and registered to the first session anatomy during the second session to obtain the translations and rotations needed to map the line from the first session anatomy to the currently active second session by inputting the values in the MR-console. This procedure from registration to calculation of MR-console values is governed by `spinoza_lineplanning` and can be called with `master -m 00 -s <subject> -h <hemisphere>`. 
