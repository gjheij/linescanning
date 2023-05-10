# line scanning repository

[![Documentation Status](https://readthedocs.org/projects/linescanning/badge/?version=latest)](https://linescanning.readthedocs.io/en/latest/?badge=latest)

![plot](https://github.com/gjheij/linescanning/blob/docs/examples/figures/overview.png)

This repository contains all of the tools used during the acquisition and postprocessing of line scanning data at the Spinoza Centre for Neuroimaging in Amsterdam. The main goal of the package is to create the most accurate segmentations (both volumetric and surface) by combining various software packages such as [Nighres](https://github.com/nighres/nighres), [fMRIprep](https://fmriprep.org/en/stable/usage.html), [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/), [CAT12](http://www.neuro.uni-jena.de/cat/index.html#DOWNLOAD), and [SPM](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/). 

## In active development
This package is still in development and its API might change. Documentation for this package can be found at [readthedocs](https://linescanning.readthedocs.io/en/latest/) (not up to date)

## Installation

### Spinoza

At the Spinoza Centre for Neuroimaging, there is an environment called `preproc` which is fully configured for this repository. Follow these steps to set it up on your local system:

- Make sure `conda` is configured with `/packages/development/anaconda/3-8/condabin/conda`. Do this when you don't see `(base)` in your terminal. This should add a bunch of things to your `~/.bashrc` file:
  ```bash
  /packages/development/anaconda/3-8/condabin/conda init
  ```

  You may be prompted to close the terminal and start a new one for the changes to take effect.

- Activate environment:
  ```bash
  conda activate preproc
  ```

- Copy the `setup` file to a location of your preference. Ideally, you should have a `programs` and a `projects`-folder in your personal processing folder (`your_folder`). I advise you to put the `spinoza_setup`-file at the root of those folders:
  ```bash
  your_folder=/data1/projects/<some_project>/<your_name>
  cp /packages/development/anaconda/3-8/git/linescanning/shell/spinoza_setup $your_folder
  ```

- Change at least the following fields (run e.g., `gedit $your_folder/spinoza_setup`):
  ```bash
  # path to your setup file
  export SETUP_FILE="${SETUP_DIR}/spinoza_setup"

  # PROJECT
  export DIR_PROJECTS="YOUR_PROJECT_FOLDER"
  export PROJECT="YOUR_PROJECT_NAME"
  export TASK_SES1=("YOUR_TASK_NAMES")
  ```

- Copy `freesurfer`-license file or [download your own license](https://surfer.nmr.mgh.harvard.edu/registration.html):
  ```bash
  your_license=$your_folder/license.txt
  cp /packages/development/anaconda/3-8/git/linescanning/misc/license.txt $your_folder/license.txt
  ```

- Add the following to your `~/.bash_profile` (run e.g., `gedit ~/.bash_profile`):
  ```bash
  # path to your setup file
  source full_path_to_your_setup_file
  export SUBJECTS_DIR=$DIR_DATA_DERIV/freesurfer
  export FS_LICENSE=$your_folder/license.txt

  module load itksnap/3.8.0 # 3.6.0-qt4 if you only see black panels
  module load fsl/6.5.0.2
  module load ANTs/20200131
  ```

### VU

On the VU cluster (`node230`), there is also an environment called `preproc` which is fully configured for this repository. Follow these steps to set it up on your local system:

- Make sure `conda` is configured with `/tank/tkn219/software/anaconda3/bin/conda`. Do this when you don't see `(base)` in your terminal. This should add a bunch of things to your `~/.bashrc` file:
  ```bash
  /tank/tkn219/software/anaconda3/bin/conda init
  ```

  You may be prompted to close the terminal and start a new one for the changes to take effect.

- Append the correct environment directory to the configuration:
  ```bash
  conda config --append envs_dir /tank/shared/software/environments
  ```

  This will create the file `~/.condarc`. Run `vi ~/.condarc` and add the following (this will prevent the full conda-path to be shown):
  ```bash
  env_prompt: "({name}) "
  ```

- Activate environment:
  ```bash
  conda activate preproc
  ```

- Copy the `setup` file to a location of your preference. Ideally, you should have a `programs` and a `projects`-folder in your personal processing folder (`your_folder`). I advise you to put the `spinoza_setup`-file at the root of those folders:
  ```bash
  your_folder=/scratch/2023/<some_project>/
  cp /tank/shared/software/git/linescanning/shell/spinoza_setup $your_folder
  ```

- Change at least the following fields (run e.g., `gedit $your_folder/spinoza_setup`):
  ```bash
  # path to your setup file
  export SETUP_FILE="${SETUP_DIR}/spinoza_setup"

  # PROJECT
  export DIR_PROJECTS="YOUR_PROJECT_FOLDER"
  export PROJECT="YOUR_PROJECT_NAME"
  export TASK_SES1=("YOUR_TASK_NAMES") # if you have multiple tasks: ("task1" "task2") NO COMMA!!
  ```

- Copy `freesurfer`-license file or [download your own license](https://surfer.nmr.mgh.harvard.edu/registration.html):
  ```bash
  your_license=$your_folder/license.txt
  cp /tank/shared/software/git/linescanning/misc/license.txt $your_folder/license.txt
  ```

- Add the following to your `~/.bash_profile` (run e.g., `vi ~/.bash_profile`):
  ```bash
  # path to your setup file
  source full_path_to_your_setup_file
  export SUBJECTS_DIR=$DIR_DATA_DERIV/freesurfer
  export FS_LICENSE=$your_folder/license.txt

  # load modules
  module load java/jre-8u181
  module load matlab/R2021b
  module load fsl

  # Freesurfer
  export FREESURFER_HOME=/tank/shared/software/freesurfer
  source $FREESURFER_HOME/FreeSurferEnv.sh

  # append ANTs to PATH
  export PATH=$PATH:/tank/shared/software/ants/install/bin
  ```

- Check that `python` points to the correct path:
  ```bash
  (preproc) [heij@node230 linescanning]$ which python
  /tank/shared/software/environments/preproc/bin/python
  ```

### Stand-alone
To install, clone the repository and run `bash linescanning/shell/spinoza_setup setup`. This will make the executables in the `bin` and `shell` folders available, install additionally required packages (such as `pRFpy`, `Pycortex`, `Nighres`, `Pybest`, `ITK-Snap`, and `Nideconv`). You can either choose to activate the accompanying `environment.yml`-file (`ACTIVATE_CONDA=1` in `spinoza_setup`; = Default!) or install it in your own environment/python installation (set `ACTIVATE_CONDA=0` in `spinoza_setup`). Installations of `FSL`, `SPM` (+`CAT12`-toolbox), `fMRIprep` and `FreeSurfer` are expected to exist on your system.

## Updating
Sometimes I need to add new stuff to the [setup file](https://github.com/gjheij/linescanning/blob/main/shell/spinoza_setup). The fact that each user can/should adapt this script locally can interfere with updates, as you'll get the `git error` that it can't overwrite due to existing changes. Please do the following before running `git pull`
```bash
# copy the spinoza_setup file to a different directory
cd $DIR_SCRIPTS
cp shell/spinoza_setup ..

# pull the latest change
git pull

# open an editor and copy your personalized stuff in again
# remove copy of setup-file
```

## Policy & To Do

- [x] install using `python setup.py develop`
- [x] Docstrings in numpy format.
- [x] PEP8 - please set your editor to autopep8 on save!
- [x] Write docs for glm.py
- [x] Sphinx doesn't show source code for python (`FileNotFoundError: [Errno 2] No such file or directory: '/home/docs/.config/pycortex'`)
- [x] Put `nbsphinx` in `requirements` if you want to build html's from notebooks with Sphinx
- [x] Explore options to streamline code
- [x] Examples of applications for package (add notebooks to `doc/source/examples`)
- [] ..
