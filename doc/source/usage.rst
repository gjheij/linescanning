.. include:: links.rst

------------
Usage notes
------------

On this page, we will elaborate on the following:

1) `Set up your own setup file`_. How to setup your environment to make the most out of the programs.
2) `File naming`_. How to name your files once your setup is completed
3) `Introduction to master`_. This gives an introduction to the master script which will do the bulk of our work.
4) `Typical preprocessing steps`_. Overview of modules typically used to preprocess/analyze functional and anatomical data.
5) `Selecting best vertex`_ to plan the line through and other vertex-related processes. This is handled by the python modules.
6) `Vertex to session 2`_. Translate a coordinate through various coordinates systems.
   
Set up your own setup file
===========================================

The path specifications are mainly controlled by the setup file called ``spinoza_setup``. You'll basically have to set only 2 things and everything will branch out from there. You need to set the `DIR_PROJECTS`-variable, which tells the pipeline where your project folders are stored. Inside `DIR_PROJECTS`, you should have project-specific names denoted with `PROJECT`. Once you've set `DIR_PROJECTS`, you can use `PROJECT` to switch between projects very easily (you'll have to do `source ~/.bash_profile` for the changes to take effect).
:: 

    # If you have access to SGE, leave to SGE; otherwise change to e.g., LOCAL
    export PLACE=SGE

    # MATLAB
    export MRRECON=<path to MR-Recon version>       # OPTIONAL! for reconstruction of line data 
    export MATLAB_DIR=<path to matlab installation> # REQUIRED
    export SPM_PATH=<path to SPM installation>      # REQUIRED
    export SKIP_LINES=30                            # OPTINAL! number of lines to skip during start up

    # fMRIPREP
    export MRIQC_SIMG=<path to MRIQC container>     # OPTIONAL; ALWAYS LOOK AT YOUR DATA!
    export FPREP_SIMG=<path to fMRIprep container>  # REQUIRED (if PLACE=SGE); see https://fmriprep.org/en/1.5.1/singularity.html
    export FPREP_OUT_SPACES="fsnative fsaverage MNI152NLin6Asym:res-1"
    export FPREP_BINDING=<binding path>             # REQUIRED (if PLACE=SGE); binding directory for singularity image
    export CIFTI=170k                               # OPTIONAL; leave empty if you don't want cifti output
    export DO_SYN=1                                 # OPTIONAL; set to zero if you do not want additional syn-distortion correction
    export BOLD_T1W_INIT="register"                 # REQUIRED; tells bbregister how to initialize the registration between bold & T1w

    # PYBEST
    export PYBEST_SPACE="fsnative"                  # REQUIRED; which space to use for denoising with Pybest
    export PYBEST_ENV="ls"                          # OPTIONAL; which environment to use for pybest (can be the same as the one earlier installed)

    # PROJECT
    export DIR_PROJECTS=$(dirname $(dirname ${PATH_HOME}))/projects # REQUIRED; path where PROJECT lives
    export PROJECT=<project name>                                   # REQUIRED; project name
    export TASK_SES1="2R"                                           # OPTIONAL (but advised); task name ses-1
    export TASK_SES2="LR"                                           # OPTIONAL; task name ses-2
    export PREFIX="sub-"                                            # REQUIRED; prefix for subject folders
    export COMBINE_SEGMENTATIONS="weighted"                         # REQUIRED; method for combining segmentations; 'weighted' means combination of all available segmentations

    # DATA TYPE(S)
    declare -a ACQ=("MP2RAGE")                                      # REQUIRED; another example = ("MP2RAGE" "MEMP2RAGE") or ("MPRAGE")
    export DATA=${ACQ[0]}                                           # OPTIONAL; will read first element from ACQ by default

    # For DATA == AVERAGE we'll need multiple acquisitions
    if [[ ${DATA} == "AVERAGE" ]]; then
      if [[ `echo ${#ACQ[@]}` -ne 2 ]]; then
        echo "Average of what..? \"ACQ\" variable in spinoza_setup has ${#ACQ[@]} items"
        exit 1
      fi
    fi

    #===================================================================================================
    # PATHS
    #===================================================================================================

    export DIR_SCRIPTS=${REPO_DIR}
    export DIR_DATA_HOME=${DIR_PROJECTS}/${PROJECT}
    export DIR_LOGS=${DIR_DATA_HOME}/code/logs
    export DIR_DATA_SOURCE=${DIR_DATA_HOME}/sourcedata
    export DIR_DATA_DERIV=${DIR_DATA_HOME}/derivatives
    export DIR_DATA_ATLAS=${PATH_HOME}/atlas/MICCAI2012-Multi-Atlas-Challenge-Data
    export SOURCEDATA=${DIR_DATA_HOME}
    export DERIVATIVES=${DIR_DATA_DERIV}
    export MASKS=${DIR_DATA_DERIV}/manual_masks
    export ANTS=${DIR_DATA_DERIV}/ants
    export FS=${DIR_DATA_DERIV}/freesurfer
    export ANTS=${DIR_DATA_DERIV}/ants
    export AVG=${DIR_DATA_DERIV}/average
    export MASKED_AVG=${DIR_DATA_DERIV}/masked_average
    export PYMP2RAGE=${DIR_DATA_DERIV}/pymp2rage
    export NIGHRES=${DIR_DATA_DERIV}/nighres
    export FSL=${DIR_DATA_DERIV}/fsl
    export SKULLSTRIP=${DIR_DATA_DERIV}/skullstripped
    export CTX=${DIR_DATA_DERIV}/pycortex
    export PRF=${DIR_DATA_DERIV}/prf

You will have to set a few things yourself:

1) If you have access to a sun-grid engine, leave ``PLACE`` to ``SGE``. If not, set it to ``LOCAL``:

  .. code:: bash

    # If you have access to SGE, leave to SGE; otherwise change to e.g., LOCAL
    export PLACE=SGE

2) Set the settings for MATLAB:

  .. code:: bash

    $ export MRRECON=<path to MRecon installation> # only required if you have actual line data
    $ export MATLAB_DIR=<path to matlab>
    $ export SPM_PATH=<path to spm12>
    $ export SKIP_LINES=30 # number of lines to skip during start up

3) Set the settings for fMRIprep_:

  .. code:: bash

    # fMRIPREP
    $ export MRIQC_SIMG=<path to MRIQC container>
    $ export FPREP_SIMG=<path to fMRIprep container> 
    $ export FPREP_OUT_SPACES="fsnative fsaverage MNI152NLin6Asym:res-1 T1w func"
    $ export FPREP_BINDING=<binding path>
    $ export CIFTI=170k
    $ export DO_SYN=1  
    $ export BOLD_T1W_INIT="register"

5) Set the settings for Pybest_:

  .. code:: bash

    # PYBEST
    $ export PYBEST_SPACE="fsnative"
    $ export PYBEST_ENV="ls"

6) Set project-specific settings (basically the most important one with respect to folder structure):

  .. code:: bash

    # PROJECT
    $ export DIR_PROJECTS=$(dirname $(dirname ${PATH_HOME}))/projects 
    $ export PROJECT=hemifield
    $ export TASK_SES1="2R"
    $ export TASK_SES2="LR"
    $ export PREFIX="sub-"
    $ export COMBINE_SEGMENTATIONS="weighted"

.. attention::
    if `COMBINE_SEGMENTATIONS="weighted"`, then a weighted combination will produce the tissue probability maps that will be inserted in CRUISE; this will utilize `call_gdhcombine`. Alternatively, you can set `COMBINE_SEGMENTATIONS="hard"`. This means you trust the segmentation from CAT12_, and `call_combine` will use this in concert with FreeSurfer_'s segmentation to create **binary** 'probabilities' that are directly converted in `spinoza_cortexreconstruction` rather than estimating levelsets using CRUISE. There's benefit to both; *weighted* allows you to be more conservative around GM/CSF interfaces, whereas *hard* can be useful in hard-to-segment areas

7) Specify your data type(s).
   
   * This is to account for a protocol including ``MP2RAGE`` and ``MEMP2RAGE`` where you might want to averge the two acquisitions into ``AVERAGE``. In that case, you'd specify:

    .. code:: bash

        $ declare -a ACQ=("MP2RAGE" "MEMP2RAGE")
        $ export DATA=AVERAGE

   * If you have an ``MP2RAGE`` and ``MEMP2RAGE`` acquisition, but you only want to use one of them, specify:

    .. code:: bash

        $ declare -a ACQ=("MP2RAGE" "MEMP2RAGE")
        $ export DATA="MP2RAGE"

   * In case you have an ``MPRAGE`` acquisition, specify:

    .. code:: bash

        $ declare -a ACQ=("MPRAGE")
        $ export DATA=MPRAGE

   * In case you have an ``MP2RAGE`` acquisition, specify:

    .. code:: bash

        $ declare -a ACQ=("MP2RAGE")
        $ export DATA=MP2RAGE

   * In case you have an ``MEMP2RAGE``, specify:

    .. code:: bash

        $ declare -a ACQ=("MEMP2RAGE")
        $ export DATA=MEMP2RAGE

.. attention:: By default, ``DATA`` will be the first element of ``ACQ``!

.. note::

    **Example**: Let's say I have the following:

    * project name: ``my_first_project``
    * full path to project: ``/home/user/projects/my_first_project``
    * matlab: ``/home/user/matlab``
    * spm12: ``/home/users/programs/spm12``
    * data acquisition: ``MP2RAGE``
    * task name session 1: ``pRF``
    * task name session 2: ``sizeresponse``

  .. code:: bash

    # I would then set the following:
    $ export PLACE=SGE
    $ export MATLAB_DIR=/home/user/matlab
    $ export SPM_PATH=/home/user/programs/spm12
    $ export PROJECT='my_first_project'
    $ export TASK_SES1="pRF"
    $ export TASK_SES2="sizeresponse"
    $ export DIR_PROJECTS=/home/user/projects
    $ declare -a ACQ=("MPRAGE")
    $ export DATA=${ACQ[0]}

File naming
===========================================

The input dataset is required to be in valid `BIDS (Brain Imaging Data Structure)` format. The directory pointing to the project should be specified in the ``spinoza_setup``-file as ``$DIR_PROJECTS``. Then specify the the project name as ``$PROJECT``. It is assumed your converted data lived in:

.. code:: bash
    
    $ $DIR_PROJECTS/$PROJECT/<subjects>

It is also assumed your ``T1w``-files have the ``acq-(ME)MP(2)RAGE`` tag in the filename. This is because the package can deal with either of these, or an *average* of MP2RAGE and MP2RAGEME acquisitions (see e.g., `<https://www.sciencedirect.com/science/article/pii/S105381192031168X?via%3Dihub>`_). So, a typical folder structure would look like this:

.. code:: bash

    $ tree $DIR_PROJECTS/$PROJECT/sub-001
    sub-001
    └── ses-1
        ├── anat
        │   ├── sub-001_ses-1_acq-3DTSE_T2w.nii.gz
        │   ├── sub-001_ses-1_acq-3DTSE_T2w.json
        │   ├── sub-001_ses-1_acq-MP2RAGE_inv-1_part-mag.nii.gz
        │   ├── sub-001_ses-1_acq-MP2RAGE_inv-1_part-phase.nii.gz
        │   ├── sub-001_ses-1_acq-MP2RAGE_inv-2_part-mag.nii.gz
        │   └── sub-001_ses-1_acq-MP2RAGE_inv-2_part-phase.nii.gz
        ├── fmap
        │   ├── sub-001_ses-1_task-2R_run-1_epi.json
        │   └── sub-001_ses-1_task-2R_run-1_epi.nii.gz
        └── func
            ├── sub-001_ses-1_task-2R_run-1_bold.json
            └── sub-001_ses-1_task-2R_run-1_bold.nii.gz

From here on, one can use the ``master``-script. As mentioned before, you can use the script like:

.. code:: bash

    $ master -m <module> 

If you want to search the module number of a particular process, you can try to do:

.. code:: bash

    $ master -l <keyword>

This will return the module number of a script by looking for matches in the help-information. For example:

.. code:: bash

    $ master -l layering

Will return the module number for ``spinoza_layering``, which is ``23``. If you want to return the help information of this module, you can do:

.. code:: bash

    $ master -m 23

Introduction to master
===========================================

The way the pipeline is set up is a remnant from my time in Berlin, where I learnt most of my bash scripting. We have one main script that controls everything, called `master` (see help text below). This script controls all the different *modules* (or steps/programs) that we can call, prefixed by *spinoza_*. These modules will run a particular function from the `bin` folder, prefixed with *call_*, by looping over all subjects it can find. The *call_*-scripts do a particular process, you can think of these like the `FSL`-commands such as `fslmaths`. In short, there's a hierarchy in which things run: `master` controls everything, `spinoza_`-programs loop over subjects to do a certain process, and then the `call_`-scripts doing *that* particular process.

.. code:: bash

    $ master -m <module number>

Below is the ``help``-information from the ``master`` script:
::

  ===================================================================================================
                              MASTER SCRIPT FOR LINE SCANNING ANALYSIS
  ===================================================================================================

  Main script governing all (pre-)processing steps for high resolution anatomical data, as well as 
  whole-brain fMRI data analyses (specifically population receptive field [pRF]) and linescanning
  data. All modules can be called with master -m <module>, which in it's bare form with loop through
  the subjects it can find in the specified input directories. You can also include an -s flag to 
  process a single subject. See below for additional settings. Please read through it's module what
  it's default settings are and what you'd need to do to get it running for you.

  Have fun!

  Args:
    -c                List used for clipping of design matrix. Format must be:
                      [<top>,<bottom>,<left>,<right>]
    -e <n_echoes>     Specify the amount of echoes of the acquisition
    -h <hemisphere>   used for lineplanning, denotes which hemisphere to process
    -i <image path>   specify custom path to PNG's for pRF-fitting. This allows deviation from the 
                      standard path that 'spinoza_fitprfs' will be looking for
    -l <which mod>    look for a module number given a string to search for (e.g., master -l layer); can
                      also be used with:
                      - spinoza_bestvertex to use a particular label-file (default = V1_exvivo.thresh).
                        If label file is being specified, the following applies:
                          - File must live in SUBJECTS_DIR/<subject>/label
                          - File must be preceded by '?h.', representing a file for both hemispheres 
                            separately
                          - Can be either a FreeSurfer-defined label file or gitfi-files (e.g., 'handknob
                            _fsnative.gii' means there's a 'lh.handknob_fsnative.gii' and 'rh.handknob_
                            fsnative.gii' file)
    -m <module>       always required, specifies which module to run. Multiple modules should be com-
                      ma separated
    -n <session>      session nr, used to create file structure according to BIDS. Specify '-n none' if 
                      you don't want to differentiate between sessions
    -o                turn on overwrite mode; removes subject-specific directories or crucial files so 
                      that the process is re-ran.
    -p <prf_model>    switch to specify what type of model to use for the pRF-modeling. Enabled for spi-
                      noza_fitprfs & spinoza_bestvertex
    -q <info mod>     query the usage-information of a given module number: e.g., master -m 00 -q
                      brings up the help-text from module 00 (spinoza_lineplanning)
    -r <roi>          Set label file for 'spinoza_bestvertex' or ROI for 'spinoza_extractregions'
    -s <subject>      Subject ID (e.g., 01). Can also be comma-separated list: 01,02,05
    -t <type|task>    used for;
                        - spinoza_fmriprep: to run either the 'anat' workflow only, or also include func
                          data ('func') or a direct task-identifier such as '${TASK_SES1[0]}'
                        - spinoza_fitprfs: limit pRF-fitting to a given task. You can also specify multi-
                          ple tasks if you want to bypass the setup file completely. In that case, use 
                          the format '-t <task1>,<task2>,<task3>'. If you use this option, please still 
                          use '--multi_design' to select the correct log-directory
                        - spinoza_bestvertex: select pRF-parameters from specific task
    -u <config>       specify configuration file for fMRIprep (see linescanning/misc for examp-
                      les). If not specified, no configuration will be used. If ses-1 is specified,
                      linescanning/misc/fmriprep_config.json is used; above that, fmriprep_con-
                      fig2.json is used. Enter '-u none' if you have multiple sessions and you want
                      to process everything without filtering
    -v <vertices>     manually insert vertices to use as "best_vertex" (module 18). Should be format-
                      ted as -v "lh,rh" (please use two vertices; one for left hemi and one for right
                      hemi > makes life easier)
    -w <wagstyl>      Use Wagstyl's equivolumetric layering for layerification. Use Nighres otherwise
                      accepted options are 'gauss' for Gaussian model, and 'norm' for Normalization
                      model. Should/can be use for module 17, pRF-fitting
    -x <**kwargs>     Can be used for a variety of options by setting it to ANYTHING other than empty. 
                      Usages include:
                        - spinoza_profiling; specify the type of file to use
    --affine          use 'affine'-registration (spinoza_registration)
    --comb_only       skip nighres, only do combination of segmentations. Works with '-o' flag. 
    --full            Do full processing with CAT12, assuming module 8 was skipped. This does itera-
                      tive SANLM-filtering, bias correction, and intensity normalization. Can some-
                      times be overkill, so generally we use CAT12 for the additional segmentations
                      and brain extraction
    --fv              use FreeView to draw sinus mask (spinoza_sagittalsinus) 
    --fsl             use FSLeyes to draw sinus mask (spinoza_sagittalsinus) 
    --gdh             use custom implementation of MGDM (via call_gdhmgdm) that also considers extra
                      filter files. By default we'll use call_nighresmgdm, which includes the T1w/T1map
                      files
    --grid            maps to '-g' in spinoza_fitprfs; runs gridfit only
    --hrf             also fit the HRF during pRF-fitting (spinoza_fitprfs)
    --itk             use ITK-Snap to draw sinus mask (spinoza_sagittalsinus) [default]
    --lines           specifies that we're dealing with a line-scanning session (spinoza_scanner2bids). 
                      Uses a custom pipeline for converting the files in a veeery specific way. If not
                      specified, we'll assume a regular scan-session of whole-brain data.
    --local           maps to '-l' in spinoza_fmriprep & spinoza_fitprfs; runs the process locally.
    --multi_design    specifies that for all runs in the dataset have run-/task-specific screenshot direc-
                      tories. This requires that the directory names you saved must match your naming 
                      scheme of functional files as we'll match on run-/task-ID (spinoza_fitprfs)
    --nighres         use nighres-software for cortex reconstruction (spinoza_cortexrecon) or equivolu-
                      metric layering (spinoza_layering)
    --no_bbr          maps to '--force-no-bbr' in call_fmriprep
    --no_biascorr     maps to '-b' in spinoza_biassanlm; don't perform bias correction after denoising
    --no_clip         spinoza_fitprfs; ensures that the design matrix is NOT clipped, despite the possible
                      presence of screen delimiter files  
    --no_freeview     disable FreeView during vertex selection (spinoza_bestvertex)
    --ups             maps to '-u' in spinoza_qmrimaps; turn on Universal Pulses
    --remove_wf       maps to '-r all' in spinoza_fmriprep; removes full fmriprep_wf folder. Be careful
                      with this one! Mainly used for debugging of a single subject. Use \"--remove_surf_wf\"
                      to specifically remove the surface reconstruction folder when you have new FreeSurfer
                      output that fMRIPrep needs to use
    --remove_surf_wf  Remove surface reconstruction workflow folder; refreshes the surfaces used for regi-
                      stration and transformation
    --rigid           use 'rigid-body' registration (spinoza_registration)
    --sge             passes '--sge' on to:
                        -spinoza_linerecon
                        -spinoza_denoising
                        -spinoza_nordic
                        -spinoza_registration
                      submits jobs to cluster
    --surface         use Wagstyl's equivolumetric layering function
    --syn             use 'syn'-registration (spinoza_registration)
    --verbose         turn on verbose
    --warp_only       spinoza_fmriprep; skip full processing, but make new 'boldref'-images and copy bold-
                      to-T1w registration files

  Usage:
    master -m <MODULES TO RUN>
    master -m 01,02,03,04 (comma-separated for running multiple modules in succession)
    master -m 00 -q   (print help-text of module 00)
    master -l mgdm    (fetch module number mgdm-module)
    master -m 02 -n prf -x 2

  Additional options:
    - Specify a subject to run (for certain modules):   master -m <module> -s <subj-ID>
    - Specify a hemisphere (for certain modules):       master -m <module> -h left
    - Disable overwrite mode (for certain modules):     master -m <module> -o n
    - Specify a particular session to use:              master -m <module> -n 1
    - Specify processing type fMRIprep (anat|func)      master -m <module> -t anat

  Available modules:                                     Script                           Time (pp)
    -00:  Register anat ses-1 to ms ses-1 for planning   (spinoza_lineplanning)
    -01:  Make new session for subject                   (spinoza_bidssession)
    -02:  Convert raw files to nifti                     (spinoza_scanner2bids)
    -03:  Reconstruction of line data                    (spinoza_linerecon)
    -04:  Estimate T1's from mp2rage and memp2rage       (spinoza_qmrimaps)               00:01:30
    -05a: Register T1 from memp2rage to T1 from mp2rage  (spinoza_registration)           00:03:00
    -05b: Register T1 from mp2rage to MNI152             (spinoza_registration)           01:05:00
    -06:  Average T1w/T1map from MP2RAGE & MP2RAGEME     (spinoza_averageanatomies)       00:02:00
    -07:  Make sinus mask from T1w/T2w ratio             (spinoza_sinusfrommni)           00:03:00
    -08:  Bias correction + SANLM denoising              (spinoza_biassanlm)              00:07:00
    -09:  Brain extract (CAT12 || ANTs || FSL)           (spinoza_brainextraction)        00:10:00
    -10:  NORDIC denoising of whole-brain data           (spinoza_nordic)                 00:05:00
    -11:  Dura estimation and skull estimations          (spinoza_dura)                   00:08:00
    -12:  Create mask of sagittal sinus                  (spinoza_sagittalsinus)          00:03:00
    -13:  Combine all masks and apply to average         (spinoza_masking)                00:02:30
    -14:  Surface segmentation with FreeSurfer           (spinoza_freesurfer)             09:30:00
    -15:  Preprocessing with fMRIprep                    (spinoza_fmriprep)               14:00:00
    -16:  Data denoising with pybest                     (spinoza_denoising)              00:05:00
    -17:  pRF-fitting with pRFpy                         (spinoza_fitprfs)                00:07:00
    -18:  Determine best vertex with pycortex            (spinoza_bestvertex)             00:01:30
    -19:  Tissue segmentation with FAST                  (spinoza_segmentfast)            00:17:00
    -20:  Tissue segmentation with MGDM                  (spinoza_mgdm)                   00:11:45
    -21:  Region extraction with nighres                 (spinoza_extractregions)         00:02:00
    -22:  Cortex reconstruction with nighres             (spinoza_cortexrecon)            00:05:30
    -23:  Layering with nighres/surface_tools            (spinoza_layering)               00:07:30
    -24:  Subcortex parcellation with nighres            (spinoza_segmentsubcortex)       00:10:00
    -25:  Project line to surface                        (spinoza_line2surface)           00:02:00
    -26:  Sample a dataset across depth with Nighres     (spinoza_profiling)              00:20:00

  --END--
 
All of these modules loop, by default, through *all* the subjects present in ``$DIR_PROJECTS/$PROJECT`` (which is defined as ``DIR_DATA_HOME``. Also by default, it assumes that we only want to process ``ses-1``. To steer the pipeline towards particular directories, such as individual subjects, you can use additional flags when calling the ``master`` script. For instance:

.. code:: bash

    $ master -m 04 -s 001         # run module 4 for 1 subject
    $ master -m 04 -s 001,004,005 # run module 4 for subject 001, 004, and 005 (note that the subjects are comma-separated without spaces!)

Some modules also use long-flagged arguments to turn on/off a specific function. For instance, we'll use `percent signal-change` by default when fitting pRFs. We can also use the zscore'd output from Pybest_, by specifying `--zscore` when calling the pybest module.

**IMPORTANT**: throughout the entire pipeline, it's assumes we're dealing with `ses-1`. If this is not the case, specify the `-n` flag with all your `master`-calls. Also, if you do *not* have separate sessions at all, use `-n none`.

Typical preprocessing steps
===========================================

First, we need to convert our DICOMs/PARRECs to nifti-files. We can do this by placing the DICOMs/PARRECs in the ``sourcedata``-folder of our project:

.. code:: bash

    $ tree $DIR_PROJECTS/$PROJECT/sourcedata/sub-001
    sub-001
    └── ses-1
        ├── task # put the outputs from Exptools2 here
        │   ├── sub-001_ses-1_task-2R_run-1_Logs
        │   │   ├── sub-001_ses-1_task-2R_run-1_Screenshots
        │   │   │   └── <bunch of png-files>
        │   │   ├── sub-001_ses-1_task-2R_run-1_desc-screen.json
        │   │   ├── sub-001_ses-1_task-2R_run-1_events.tsv
        │   │   ├── sub-001_ses-1_task-2R_run-1_settings.yml
        │   ├── sub-001_ses-1_task-2R_run-2_Logs
        │   └── sub-001_ses-1_task-2R_run-3_Logs
        └── Raw files (DICOMs/PARRECs)

Once you've put the files there, you can run ``module 02``. This will convert your data to nifti's and store them according to BIDS. You can use the ``-n`` flag to specify with which session we're dealing. This creates the folder structure outlined above. If you also have phase data from your BOLD, then it creates an additional `phase`-folder, which can be used for ``NORDIC`` later on. If you have a non-linescanning acquisition (i.e., standard whole-brain-ish data), we'll set the coordinate system to `LPI` for all image. This ensures we can use later outputs from fMRIprep_ on our native data, which will help if you want to combine segmentations from different software packages (affine matrices are always a pain; check [here](https://nipy.org/nibabel/coordinate_systems.html) for a good explanation). If you have extremely large par/rec files, `dcm2niix` [might fail](https://github.com/rordenlab/dcm2niix/issues/659). To work around this, we monitor these error and attempt to re-run the conversion with [call_parrec2nii](https://github.com/gjheij/linescanning/blob/main/bin/call_parrec2nii), which wraps `parrec2nii` that comes with nibabel. To do this, use the `--dcm_fix` flag. Additionally, we also try to read the phase-encoding direction from the PAR-file, but this is practically impossible. So there's two ways to automatically populate the `PhaseEncodingDirection` field in your json files: 1) Set the `export PE_DIR_BOLD=<value>` in the setup file, with one of `AP`, `PA`, `LR`, or `RL`. This sets the phase-encoding direction for the BOLD-data; this value is automatically switched for accompanying fieldmaps; 2) use one of the following flags in your call to `master`: `--ap`, `--pa`, `--lr` or `--rl`. This again sets the phase-encoding for the BOLD-data, and assumes that the direction is opposite for the fieldmaps; 3) Manually change the value in your json files after processing.

.. code:: bash

    $ master -m 02 -n 1                         # spinoza_scanner2bids
    $ master -m 02 -n 1 --dcm_fix               # monitor and fix catastrophic errors
    $ master -m 02 -n 1 --dcm_fix --pa          # set the PhaseEncodingDirection for the BOLD to PA (default = AP)
    $ master -m 02 -n 1 --dcm_fix --no_lpi      # do not reorient your data to LPI coordinates (the default one for fMRIPrep); ill-advised when you want to do NORDIC

After converting our data to nifti, we need to create our T1w/T1map images from the first and second inversion images. We can do this quite easily with Pymp2rage_. If you do not combine **INV1** and **INV2** yourself, - you already have a ``T1w``-/ ``T1map``-file in ``DIR_DATA_HOME`` - you can skip the part below:

.. code:: bash

    $ master -m 04 # spinoza_qmrimaps

If you have multiple acquisition (e.g., `MP2RAGE` and `MP2RAGEME`) that you'd like to average, you can register them together with `module 05a`. Here, we'll assume that the reference space is `MP2RAGE`, and register the `MP2RAGEME` to that image. To then average together, use `module 06`. Note that this will only have an effect if you specified ``DATA=AVERAGE`` in the setup file. If you have only 1 image type, skip this step. 

.. code:: bash

    $ master -m 05a # spinoza_registration (anat-to-anat)    
    $ master -m 06  # spinoza_averageanatomies

If you have a ``T2w``-image, you can create a sagittal sinus mask. We can do this by taking the T1w/T2w-ratio (if you do not, see `module 12`). However, this ratio will also highlight areas in the subcortex, which would be removed if we don't do something. In the ``misc`` folder, we provide a heavily dilated sinus mask in MNI-space that we use to exclude the T2-values from the subcortex. First, we register the subject's anatomy to the MNI-space, use these warp files to transform the dilated sinus mask to subject space, and multiply our sinus mask derived from the T2 with that dilated mask. This way, we exclude any subcortical voxels from being masked out. This procedure will lead to a file called ``<subject>_<ses-?>_acq-(ME)MP(2)RAGE_desc-mni_sinus`` in the ``derivatives/masked_(me)mp(2)rage/<subject>/<ses-?>`` folder. If you do *NOT* have a ``T2w``-file, skip these steps and continue to ``module 08``. For both registration steps (05a/05b), you can specify additional long flags: ``--affine`` (run affine registration), ``--rigid`` (run rigid registration), ``--syn`` (run SyN registration).

.. code:: bash

    $ master -m 05b # spinoza_registration (anat-to-MNI)
    $ master -m 07 # spinoza_sinusfrommni

    # optional
    $ master -m 05b --affine # use affine registration instead of SyN (is faster)

Bias correct (*SPM*) and denoise (*SANLM*) your ``T1w``-image. If you did not use Pymp2rage_, we'll be looking for the ``T1w``-file in ``DIR_DATA_HOME/<subject>/<ses>/anat``. If you did use Pymp2rage_, we'll be looking for the file in ``<DIR_DATA_HOME>/derivatives/pymp2rage/<subject>/<ses>``, otherwise we'll default to ``$DIR_DATA_HOME``. If you do not want additional bias correction after denoising, use ``--no_biascorr``

.. code:: bash

    $ master -m 08 # spinoza_biassanlm
    $ master -m 08 --no_biascorr # skip bias correction after denoising 

Perform brain extraction and white/gray matter + CSF segmentations with CAT12_. If you've skipped the previous step because you want a more thorough denoising of your data, you can specify ``--full`` to do iterative filtering of the data. Beware, though, that this is usually a bit overkill as it makes your images look cartoon'ish.

.. code:: bash

    $ master -m 09 # spinoza_brainextraction

    # optional
    $ master -m 09 --full

As previously mentioned, if we have phase data with our BOLD data, we can apply NORDIC-denoising. This expects a `phase` folder to be present in the BIDS-folder of the subjects. If there isn't one and you still called this module, we'll do NORDIC denoising based on the magnitude files only. With this module, we can specify the ``--sge`` flag to submit individual jobs to the cluster. This module will back up you un-NORDIC'ed data in a `no_nordic` folder, while overwriting the functional file in `func`. Messages about the nature of NORDIC denoising are added to the json-files so we can disentangle them later.

.. code:: bash

    $ master -m 10 # spinoza_brainextraction

    # optional
    $ master -m 10 --sge

If you do not have a T2-image, we need to apply manual labor. Use `module 12` to open ITK-Snap_, and start drawing in the sagittal sinus. Once done, save and close ITK-Snap, which will save the sagittal sinus mask to your `manual_masks`-folder.

.. code:: bash

    $ master -m 12 # spinoza_sagittalsinus

The whole point of this pipeline is to create a very good segmentation of the surface and tissue. To help FreeSurfer_ with that, we can create a mask to get rid of the most obvious garbage in the brain (i.e., sagittal sinus & dura), as this is generally marked as grey matter. By now, we should already have a pretty decent sinus mask created by subtracting a relatively crude brain mask (from SPM_) with a more accurate brain mask from CAT12_. We can further enhance this by adding the sagittal sinus mask we created earlier. With `module 13`, we combine all of these separate mask, enhance them with manual edits, and set everything in the mask to zero in the T1w-image so that FreeSurfer_ is not going to be confused by the sinus (this is basically a *pial edit*-phase before FreeSurfer_). Creates a ``desc-outside.nii.gz`` mask in the ``derivatives/masked_(me)mp(2)rage/<subject>/<ses-?>`` folder that contains the sinus, dura, and tentorium (stuff between the cerebellum and cortex). No special flags are in operation here.

.. code:: bash

    $ master -m 13 # spinoza_masking

That was basically the preprocessing of the anatomicals files. We can now run FreeSurfer_ reconstruction *outside* of fMRIprep_. This is because the newest version of FreeSurfer_ deals better with ``T2w``-images than the version used in fMRIprep_. If you have a cluster available (detected via the ``$PLACE`` variable in the spinoza_setup file), we'll automatically submit the job, running all stages of `recon-all`. Check [call_freesurfer](https://github.com/gjheij/linescanning/blob/main/bin/call_freesurfer) for information about further editing/debugging modes. If you've skipped all of the previous preprocessing, we'll look in an hierarchical manner for the T1w-files: first, we'll look in `$DIR_DATA_DERIV/masked_mp2rage` (assuming you've run the preprocessing), then `$DIR_DATA_DERIV/denoising` (denoised, but un-masked), `$DIR_DATA_DERIV/pymp2rage` (un-denoised, un-masked), `$DIR_DATA_HOME` (raw output from the scanner). It also has the ability to process your edits made on the `brainmask.mgz` or white matter, and you can also specify an expert option file (see [this example](https://github.com/gjheij/linescanning/blob/main/misc/example_experts.opts) for inspiration).

.. code:: bash

    $ master -m 14
    $ master -m 14 -s 00 -r 23 -e {wm,pial,cp,aseg}        # insert your edits (one of wm, pial, cp, or aseg)
    $ master -m 14 -s 00 -x expert.opts     # use an expert option file

Once you're satisfied with the surface reconstruction, you can proceed with fMRIprep_. By default this runs fMRIprep_ with the ``--anat-only`` option, but you can use ``-t`` flag to include functional data in as well. If you have multiple tasks in your session and you'd like to preprocess only a particular task, you can specify ``-t <task ID>`` instead of ``-t func``. Alternatively, you can use ``-u <config file>`` (see `DIR_SCRIPTS/misc/fmriprep_config.json`). I use this because most of my sessions will have line-scanning data that cannot be preprocessed by fMRIprep_. Therefore, I use `DIR_SCRIPTS/misc/fmriprep_config.json` to only include data from `ses-1`. If there's no `$DIR_DATA_DERIV/masked_mp2rage`-folder, we'll default straight to `$DIR_DATA_HOME`. fMRIprep_'s bold-reference images are sometimes a bit wonky, so we're creating those again after fMRIPrep has finished. In addition, we'll copy the `bold-to-T1w`-registration files to the output folder as well, just so we have everything in one place. This avoids that we have to request the MNI152 outputs, as we have all the registration files needed in case we'd still want them (saves some disk space). To skip fMRIPrep and only create new bold-reference images and copy the registration files, use ``--warp-only``. 

.. code:: bash

    $ master -m 15 # spinoza_fmriprep
    $ master -m 15 -t func 
    $ master -m 15 -t $TASK_SES1 # preprocess a specific task (maps to '--task-ID' from fMRIprep) 
    $ master -m 15 -t func -u $DIR_SCRIPTS/misc/fmriprep_config.json # run fMRIPrep with specific configuration

    # optional
    $ master -m 15 --warp-only # only create new bold-reference images and copy the registration files
    $ master -m 15 --local # run fmriprep locally; in case of SGE, do not submit singularity job

We then proceed by denoising the functional data using Pybest_. This takes the ``confound``-file generated by fMRIprep_ and denoised the functional data. By default I do this in ``fsnative``, so if you want a different space to be processed you'll need to adapt ``spinoza_setup`` accordingly (`PYBEST_SPACE`). By default Pybest_ zscores the data, which in some cases is undesirable. If the processing space is `fsnative` or `fsaverage`, we'll un-zscore the output from pybest and save the unzscored data in `unzscored`-folder, rather than `denoising`. You can then use this data to standardize however you see fit (for later pRF-fitting, this data is standardized following the method from Marco Aqil, where the timecourses are shifted such that the median of the timepoints without any stimulation are set to zero). If you do not want this additional unzscoring, use ``--no_raw``. Use ```--sge`` to submit the job to the cluster in case you have one available.

.. code:: bash

    $ master -m 16 # spinoza_denoising

    # optional
    $ master -m 16 --no_raw # turn of unzscoring
    $ master -m 16 --sge # submit to cluster

You can do some pRF-fitting as well. If you did not use Pybest_ for denoising, we'll use fMRIprep_-output as input for the pRF-fitting. All models specified within pRFpy_ are available, as well as several other options: ``--grid``, run a grid fit only, no iterative fitting; ``-c``, list of values to clip the design matrix with (if ```--no_clip`` is NOT specified); ``--zscore``, use the zscore'd output from pybest; ``--multi_design``, used if you have multiple pRF designs within a session so the correct screenshots are used for the design matrix. ``--local``, do not submit the job to the cluster (can be useful for debugging). ``-p``, specifies which model to use (can be one of ['gauss', 'css', 'dog', 'norm']); ``-t``, fit a specific task (if you have mutliple in a session). It's advised to copy the pRF-analysis template from the *linescanning*-repository to ``$DIR_DATA_HOME/code`` and adjust settings there. Do NOT rename the settings file, otherwise [call_prf](https://github.com/gjheij/linescanning/blob/main/bin/call_prf) will fail.

.. code:: bash

    $ master -m 17 # spinoza_fitprfs

    # added options
    $ master -m 17 -p norm # run divisive-normalization (DN-) model
    $ master -m 17 --multi_design # multiple designs in a session

Continuing with the anatomical pipeline, we now enter the Nighres_ stage, to fully optimize the quality of segmentations. These modules can be run pretty much successively, and consist of ``MGDM``, ``region extaction``, ``CRUISE``, and ``volumetric layering``:

.. code:: bash

    $ master -m 20 # spinoza_segmentmgdm
    $ master -m 21 # spinoza_extractregions > also combines ALL previous segmentations into optimized levelsets
    $ master -m 22 # spinoza_cortexreconstruction > currently mimicks CRUISE, rather than running it
    $ master -m 23 # spinoza_layering > by default uses Nighres; use --surface to use Wagstyl's equivolumetric layering based on FreeSurfer output

Which is equivalent to:.

.. code:: bash

    $ master -m 20,21,22,23

Selecting best vertex
===========================================

Selecting the best vertex is performed by ``spinoza_bestvertex``, which internally calls upon ``optimal.call_pycortex2``. This function internalizes the classes within ``optimal.py``. First, deal the input folders. We need the path to the pRF-parameters, FreeSurfer output, and Pycortex.

.. code:: python

    if deriv:
        # This is mainly for displaying purposes
        dirs = {'prf': opj(deriv, 'prf'),
                'fs': opj(deriv, 'freesurfer'),
                'ctx': opj(deriv, 'pycortex')}

        prfdir, fsdir, cxdir = dirs['prf'], dirs['fs'], dirs['ctx']
    else:
        if not prfdir and not fsdir and not cxdir:
            raise ValueError("Need the paths to pRF/pycortex/FreeSurfer output. Either specify them separately or specify a derivatives folder. See doc!")
        else:
            # This is mainly for displaying purposes
            dirs = {'prf': prfdir,
                    'fs': fsdir,
                    'ctx': cxdir}

    print("Using following directories:")
    [print(f" {i}: {dirs[i]}") for i in dirs]

    if not out:
        out = opj(cxdir, subject, 'line_pycortex.csv')

Then we will utilize ``optimal.CalcBestVertex`` to combine anatomical information from the surface (embedded in ``optimal.SurfaceCalc``) and pRF-parameters (``optimal.pRFCalc``). *SurfaceCalc* does a bunch of things using Pycortex_ such as extracting curvature, thickness, sulcal depth, smoothing while *pRFCalc* merely loads in the pRF-parameters for easy access. By default, *roi* is set to ``V1_exvivo.thresh``.

.. code:: python

    # This thing mainly does everything. See the linescanning/optimal.py file for more information
    print("Combining surface and pRF-estimates in one object")
    bv = CalcBestVertex(subject=subject, fs_dir=fsdir, prf_file=prf_params, fs_label=roi)

We then specify the criteria to which our target vertex must conform. We do this for surface and pRF properties separately, in case we do not have pRF parameters but we still want to target a flat patch of cortex:

.. code:: python

    print("Set thresholds (leave empty and press [ENTER] to not use a particular property):")
    # get user input with set_threshold > included the possibility to have only pRF or structure only!
    if hasattr(bv, 'prf'):
        ecc_val     = set_threshold(name="eccentricity", borders=(0,15), set_default=round(min(bv.prf.ecc)))
        r2_val      = set_threshold(name="r2", borders=(0,1), set_default=round(min(bv.prf.r2)))
        pol_val_lh  = set_threshold(name="polar angle lh", borders=(0,np.pi), set_default=round(np.pi))
        pol_val_rh  = set_threshold(name="polar angle rh", borders=(-np.pi,0), set_default=round(-np.pi))
        pol_val     = [pol_val_lh,pol_val_rh]
    else:
        ecc_val     = 0
        r2_val      = 0
        pol_val     = 0

    if hasattr(bv, 'surface'):
        thick_val   = set_threshold(name="thickness", borders=(0,5), set_default=max(bv.surface.thickness.data))
        depth_val   = set_threshold(name="sulcal depth", set_default=round(min(bv.surface.depth.data)))
    else:
        thick_val   = 0
        depth_val   = 0

From this, we can create a boolean numpy-mask with vertices that could be used. This mask is applied to the minimal curvature map, so that we find a vertex that conforms to the criteria while ensuring a minimal curvature.

.. code:: python

    # Create mask using selected criteria
    bv.threshold_prfs(ecc_thresh=ecc_val,
                      r2_thresh=r2_val,
                      polar_thresh=pol_val,
                      depth_thresh=depth_val,
                      thick_thresh=thick_val)

    # Pick out best vertex
    bv.best_vertex()

This gives us a vertex, normal, and coordinate for each hemisphere. We can extract pRF parameters from this vertex with ``call_prfinfo``:

.. code:: bash

    call_prfinfo -s <subject> -v <left hemi vertex> -h lh

The function will then open ``FreeView`` on the location of the selected vertex, so that we can inspect its location. Ideally, the vertex is *NOT* located in a cortical fold, but rather on a relatively straight patch. Additionally, we want a vertex that is located around the calcarine sulcus, not too lateral. This should be ensured by the pRF-parameters themselves, though. If the vertex is fine, we can close ``FreeView``, and answer the question:

.. code:: bash 
    
    $ Happy with the position? (y/n):

in the terminal with **y**. If you're not happy, enter **n**, and the function will take you back to selecting thresholds. If you specified **y**, we will create the ``line_pycortex.csv`` file and store it in ``derivatives/pycortex/<subject>``. This file will be used later on to place the line in *session 2* using ``spinoza_lineplanning``, or ``master -m 00``.

Vertex to session 2
===========================================

For this section to be relevant, you're in the following situation: you have a surface reconstruction from a subject and selected a vertex you want to target using the above outlined strategy. Now you're subject is in the scanner again and you just acquired a fast low-resolution anatomical scan (doesn't have to be super high resolution, just enough for rigid body registration). Now you'll have to do the following:

1) Place the *RAW*-files of the anatomical scan in ``<project>/sourcedata/<subject>/<ses-?>/planning``. The script will look for an *MP2RAGE* file with a delay time of **t1008**. If you have a different file, adjust ``spinoza_lineplanning`` to look for the correct file
2) Run the following command:

.. code:: bash

    $ master -m 00 -s <subject (without 'sub')> -h <hemisphere (left|right)> -n <session (new session ID, e.g., 3 | default = 2)>

This will do the following:

- Register ``<project>/derivatives/freesurfer/<subject>/mri/orig.mgz`` to low-resolution anatomical scan
- Use transformation file to translate the target vertex's coordinate/normal vector to session 2
- Convert the normal vector to Euler angles interpretable by the scanner
- Print information you need to enter in the MR-console

.. note:: 
    This is currently tailored to Phillips systems. I'm currently unaware of the coordinate system in SIEMENS devices. Phillips scanners interpret coordinates as LPS. This is rather important to know beforehand, so if you know how a SIEMENS coordinate system works, please let us know so we can add that in the code.
