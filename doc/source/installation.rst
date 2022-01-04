.. include:: links.rst

------------
Installation
------------
You can install *LineScanning* within a `Manually Prepared Environment (Python 3.7+)`_. The workflow to install *LineScanning* is as follows. 

Download the repository from ``github``:

.. code:: bash

   $ cd <some directory>
   $ git clone https://github.com/gjheij/linescanning.git

Setup the environment for ``bash``:

.. code:: bash

   $ bash <some directory>/linescanning/shell/spinoza_setup setup

This will print out a set of instructions that are executed under the hood; it makes sure the setup file is loaded in your ``.bash_profile`` each time you start a new a new terminal and makes the scripts inside the repository executable similar to how *FSL* works. Setup the ``python`` environment. Edit the *environment.yml* file to your liking (e.g., paths and environment name) and execute the following commands:

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
*LineScanning* is written using Python 3.7 (or above). *LineScanning* requires some other neuroimaging software tools that are not handled by the Python's packaging system (Pypi) used to deploy the ``LineScanning`` package:

- FSL_ (version 5.0.9)
- ANTs_ (version 2.3.1)
- ITK-Snap_ (version 3.8.0)
- `C3D <https://sourceforge.net/projects/c3d/>`_ (version 1.0.0)
- FreeSurfer_ (version 7.2.0)
- Nideconv_
- pRFpy_
- fMRIprep_ (version 20.2.6)
- CAT12_ (version r1113)
- SPM_
- Nighres_
- Pybest_
- Pycortex_ 