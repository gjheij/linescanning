# line scanning repository

![plot](https://github.com/tknapen/linescanning/blob/master/linescanning/examples/figures/20201106_workflow_acquisition_third_attempt.png)

This repository contains all of the tools used during the acquisition and postprocessing of line scanning data at the Spinoza Centre for Neuroimaging in Amsterdam. The script `master` controls the modules prefixed by `spinoza_`, which in turn call upon various scripts in the `utils` and `bin` directory. The scripts in the latter folders are mostly helper scripts to make life a tad easier. The repository contains a mix of languages in bash, python, and matlab, with jupyter notebooks for experimentation. Usually, these notebooks have been converted to script-form in the `bin` directory.

# overview of the pipeline

## how to set up
Clone the repository: `git clone https://github.com/tknapen/linescanning.git`.

To setup the bash environment, edit setup file `linescanning/shell/spinoza_setup`:
- line 76: add the path to your matlab installation if available (should be, for better anatomicall preprocessing)
- line 87: add the path to your SPM installation
- line 92: add your project name
- line 97: add the path to project name as defined in line 92
- line 102: add whether you're using (ME)MP(2)RAGE. This is required because the pipeline allows the usage of the average of an MP2RAGE and MP2RAGEME acquisition
- line 105: add which type of data you're using (generally this will be the same as line 102)

Go to `linescanning/shell` and hit `./spinoza_setup setup setup`. This will print a set of instructions that you need to follow. If all goes well this will make all the script executable, set all the paths, and install the python modules.

Check whether the files are labeled as executables, by for instance typing `master` in the terminal. This should bring up the help-menu of the master script. If it throws some error, press `ll` in the `shell`/`bin` directories to see whether there's an `x` on the left. If not, run `chmod -R 775 $PWD`.

Whenever you change something to the setup file, remember to source the ~/.bashrc or ~/.bash_profile file (depending on your setup). This will instantaneously change all the linked paths/settings as well (e.g., if you want to switch between projects)

# How to plan the line

![plot](https://github.com/tknapen/linescanning/blob/master/linescanning/examples/figures/20201215_detailedintermezzo.png)

We currently aim to have two separate sessions: in the first session, we acquire high resolution anatomical scans and perform a population receptive field (pRF-) mapping paradigm (Dumoulin and Wandell, 2008) to delineate the visual field. After this session, we create surfaces of the brain and map the pRFs onto that via fMRIprep and prfpy. We then select a certain vertex based on the parameters extracted from the pRF-mapping: eccentricity, size, and polar angle. Using these parameters, we can find an optimal vertex. We can obtain the vertex position, while by calculating the normal vector, we obtain the orientation that line should have (parellel to the normal vector and through the vertex point). Combining this information, we know how the line should be positioned in the first session anatomy. In the second session, we first acquire a low-resolution MP2RAGE with the volume coil. This is exported and registered to the first session anatomy during the second session to obtain the translations and rotations needed to map the line from the first session anatomy to the currently active second session by imputting the values in the MR-console. This whole procedure is governed by `spinoza_lineplanning` and can be called with `./master -m 00`.

This script needs the path to the exported data, the path to the first session anatomy, the path to the output file containing information about the orientation of the line in the first session anatomy (created with `call_pycortex.py`), and whether we want to analyze the left or right hemisphere. It starts by converting the data to nifti format, and looks for the `t1008` tag, denoting a UNI MP2RAGE image. The first session anatomy is registered to this image, creating a affine matrix file. The content of this file is stored in a text file called `line_ants.txt` and contains the rotation and translation values that map session 1 to session 2. Combining this with the information about the orientation of the line in session 1, we can derive all the required information for the MR-console: the orientation of the slice, the direction of foldover (saturation slabs), 3 translation values, and 3 rotation values.
