.. include:: links.rst

------------
Installation
------------
You can install *LineScanning* within a `Manually Prepared Environment (Python 3.7+)`_. The workflow to install *LineScanning* is as follows. 

Download the repository from ``github`` (see **TIP** below on what ``<some directory>`` could be!):

.. code:: bash

   $ cd <some directory>
   $ git clone https://github.com/gjheij/linescanning.git

Setup the environment for ``bash``:

.. code:: bash

   $ bash <some directory>/linescanning/shell/spinoza_setup setup

This makes sure the setup file is loaded in your ``.bash_profile`` each time you start a new a new terminal and makes the scripts inside the repository executable similar to how *FSL* works. Setup the ``python`` environment. Edit the *environment.yml* file to your liking (e.g., paths and environment name) and execute the following commands:

.. code:: bash

   $ conda create --name <environment name> --file environment.yml
   $ pip install -e .

Test the installation with:

.. code:: bash

   $ python -c "import linescanning"

If no error was given, the installation was successful. To test the ``bash`` environment, enter the following:

.. code:: bash

   $ master

This should give the help menu of the master script. If not, something went wrong. A first check if to make sure the files in ``linescanning/bin`` and ``linescanning/shell`` are indeed executables. If not, hit the following and try again:

.. code:: bash
   
   $ chmod -R 775 linescanning/bin
   $ chmod -R 775 linescanning/shell

Manually Prepared Environment (Python 3.7+)
===========================================
Make sure all of *LineScanning*'s `External Dependencies`_ are installed. These tools must be installed and their binaries available in the system's ``$PATH``.
As an additional installation setting, FreeSurfer requires a license file (see `<https://surfer.nmr.mgh.harvard.edu/fswiki/License>`_).

External Dependencies
---------------------
*LineScanning* is written using Python 3.7 (or above). *LineScanning* requires some other neuroimaging software tools used to deploy the ``LineScanning`` package that are not handled by the Python's packaging system (Pypi) or that require special attention setting up. For each of these packages, evaluate whether your computing cluster has some of these already available. If not, follow their individual installation instructions as per the packages' documentation:

.. tip::
   For python packages that are currently in development (e.g., pRFpy_, Nideconv_, Pycortex_) or require special attention (e.g., Nighres_), it is advisable to create a separate folder like ``packages`` locally, ``git clone`` the packages into this directory, and run ``pip install -e .`` in the individual packages (should the respective packages allow this procedure). That could look something like:

   .. code:: bash
      
         $ tree ${PATH_HOME}/programs/packages
         packages
         ├── prfpy
         ├── nideconv
         ├── pycortex
         ├── nighres
         └── linescanning # this repository

   The nice thing about this is that, by default, we use this directory to store important files for fMRIprep_ and ``MRIqc`` as such:

   .. code:: bash

         ├── ...
         ├── linescanning # this repository
         └── fmriprep
             ├── containers_bids-fmriprep--20.2.5.simg
             ├── <project name 1>
             │   └── fmriprep_wf
             └── <project name 2>
                 └── fmriprep_wf

   This would keep the organization of fMRIprep_'s tidy and increases reproducibility as you'd only have to change the project name in order to access everything from logs, preprocessing, and analyses. In ``spinoza_setup``, this directory is specified as ``PATH_HOME/programs/packages``, but can obviously be changed to the user's desired path. So the steps would be: 
   
   1) Create the directory ``PATH_HOME/programs/packages``
   2) For the packages pRFpy_, Nideconv_, Nighres_, Pycortex_, Pybest_, but also the *linescanning* repo, run ``git clone + pip install -e .`` in **THAT** directory
   3) Let this package organize your fMRIprep_ output for you

- FSL_ (version 5.0.9) > will be on your PATH by default upon installation
- ANTs_ (version 2.3.1) > will be on your PATH by default upon installation
- ITK-Snap_ (version 3.8.0) > has to be added to path manually, e.g., in ``~/.bash_profile``
- `C3D <https://sourceforge.net/projects/c3d/>`_ (version 1.0.0) > will be on your PATH by default upon installation
- Nideconv_ (dev version) > use ``git clone + pip install -e .``
- pRFpy_ (dev version) > use ``git clone + pip install -e .``
- fMRIprep_ (version 20.2.6)
- CAT12_ (version r1113)
- SPM_ (spm12)
- Nighres_ requires specific Java modules!
- Pybest_ (dev version) > use ``git clone + pip install -e .``
- Pycortex_ (dev version) > use ``git clone + pip install -e .``
- FreeSurfer_ (version 7.2.0) > add the following to your ``~/.bash_profile``:
  
.. code:: bash
   
   export FREESURFER_HOME=<path to FreeSurfer installation>
   export FS_LICENSE=$FREESURFER_HOME/license.txt
   source $FREESURFER_HOME/SetUpFreeSurfer.sh

.. attention:: 
   FreeSurfer_ requires a ``SUBJECTS_DIR``, and upon installation this is set to ``FREESURFER_HOME``. However, this variable is overwritten in ``spinoza_setup``! It is possible that in some cases this might lead to undesirable effects. For instance, if you use this package alongside your own scripts. If you wish to overwrite the variable in ``spinoza_setup``, comment out that line and set a ``SUBJECTS_DIR`` in your ``~/.bash_profile``.
      