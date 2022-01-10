.. include:: links.rst

------------
Installation
------------
You can install *LineScanning* within a `Manually Prepared Environment (Python 3.7+)`_. The workflow to install *LineScanning* is as follows. 

Download the repository from ``github`` (see **TIP** below on what ``<some directory>`` could be!):

.. code:: bash

   $ cd <some directory>
   $ git clone https://github.com/gjheij/linescanning.git

Edit the ``ACTIVATE_CONDA`` variable in the ``spinoza_setup``-file to your liking: by default, we'll activate and install several packages into that environment. Set to ``0`` if you wish to use the currently active python installation. Also adjust ``PATH_HOME`` if you want (See **TIP** below), as we'll be putting the installed packages here. By default this is in the same directory as where you ``git clone``'d this repository. Then enter:

.. code:: bash

   $ bash <some directory>/linescanning/shell/spinoza_setup setup

This makes sure the setup file is loaded in your ``.bash_profile`` each time you start a new a new terminal and makes the scripts inside the repository executable similar to how *FSL* works. The file then looks for an installation of ITK-Snap_, and install it in ``PATH_HOME`` if it can't find an installation. It will also attempt to install Nighres_, Pymp2rage_, Pybest_, pRFpy_, Pycortex_, and Nideconv_ if no installations are found.

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
You can either choose to activate the accompanying ``environment.yml``-file (``ACTIVATE_CONDA=1`` in ``spinoza_setup``; = Default!) or install it in your own environment/python installation (set ``ACTIVATE_CONDA=0`` in ``spinoza_setup``). Installations of ANTs_, FSL_, SPM_ (+CAT12_-toolbox), fMRIprep_ and FreeSurfer_ are expected to exist on your system.

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

   This would keep the organization of fMRIprep_'s tidy and increases reproducibility as you'd only have to change the project name in order to access everything from logs, preprocessing, and analyses. In ``spinoza_setup``, this directory is specified as ``PATH_HOME``, but can obviously be changed to the user's desired path. So the steps would be: 
   
   1) Create the directory ``PATH_HOME``
   2) For the packages pRFpy_, Nideconv_, Nighres_, Pycortex_, Pybest_, but also the *linescanning* repo, run ``git clone + pip install -e .`` in **THAT** directory
   3) Let this package organize your fMRIprep_ output for you

   **This is done by default by ``spinoza_setup``**

- FSL_ (version 5.0.9) > should be present/user install
- ANTs_ (version 2.3.1) > should be present/user install
- ITK-Snap_ (version 3.8.0) > installed by ``spinoza_setup``
- `C3D <https://sourceforge.net/projects/c3d/>`_ (version 1.0.0) > will be on your PATH by default upon installation/user install
- Nideconv_ (dev version) > installed by ``spinoza_setup``
- pRFpy_ (dev version) > installed by ``spinoza_setup``
- fMRIprep_ (version 20.2.6) > should be present/user install
- CAT12_ (version r1113) > should be present/user install
- SPM_ (spm12) > should be present/user install
- Nighres_ > installed by ``spinoza_setup`` (Java-modules might have to be installed by user!)
- Pybest_ (dev version) > installed by ``spinoza_setup``
- Pycortex_ (dev version) > installed by ``spinoza_setup``
- FreeSurfer_ (version 7.2.0) > should be present/user install, make sure the following is presnt in your ``~/.bash_profile``:
  
.. code:: bash
   
   export FREESURFER_HOME=<path to FreeSurfer installation>
   export FS_LICENSE=$FREESURFER_HOME/license.txt
   source $FREESURFER_HOME/SetUpFreeSurfer.sh

.. attention:: 
   FreeSurfer_ requires a ``SUBJECTS_DIR``, and upon installation this is set to ``FREESURFER_HOME``. However, this variable is overwritten in ``spinoza_setup``! It is possible that in some cases this might lead to undesirable effects. For instance, if you use this package alongside your own scripts. If you wish to overwrite the variable in ``spinoza_setup``, comment out that line and set a ``SUBJECTS_DIR`` in your ``~/.bash_profile``.
      