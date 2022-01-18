.. include:: links.rst

------------
Usage notes
------------

On this page, we will elaborate on the following:

1) `Set up your own setup file`_. How to setup your environment to make the most out of the programs.
2) `File naming`_. How to name your files once your setup is completed
3) `Preprocessing of anatomical data`_. This mostly consists of command line functions with similar characteristics as e.g., FSL.
4) `Selecting best vertex`_ to plan the line through and other vertex-related processes. This is handled by the python modules.
5) `Vertex to session 2`_. Translate a coordinate through various coordinates systems.
   
Set up your own setup file
===========================================

The path specifications are mainly controlled by the setup file called ``spinoza_setup``, which looks like this:

.. code:: bash

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

    # PYBEST
    export PYBEST_SPACE="fsnative"                  # REQUIRED; which space to use for denoising with Pybest
    export PYBEST_NORM="zscore"                     # OPTIONAL; by default, Pybest z-scores data

    # PROJECT
    export DIR_PROJECTS=$(dirname $(dirname ${PATH_HOME}))/projects # REQUIRED; path where PROJECT lives
    export PROJECT=<project name>                                   # REQUIRED; project name
    export TASK_SES1="2R"                                           # OPTIONAL (but advised); task name ses-1
    export TASK_SES2="LR"                                           # OPTIONAL; task name ses-2
    export PREFIX="sub-"                                            # REQUIRED; prefix for subject folders
    export COMBINE_SEGMENTATIONS="weighted"                         # REQUIRED; method for combining segmentations

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
    $ export FPREP_OUT_SPACES="fsnative fsaverage MNI152NLin6Asym:res-1"
    $ export FPREP_BINDING=<binding path>
    $ export CIFTI=170k
    $ export DO_SYN=1  

5) Set the settings for Pybest_:

.. code:: bash

    # PYBEST
    $ export PYBEST_SPACE="fsnative"
    $ export PYBEST_NORM="zscore"

6) Set project-specific settings:

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

    .. warning::
        This is actually untested. If you use this option, let us know how this goes!
   
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

The input dataset is required to be in valid :abbr:`BIDS (Brain Imaging Data Structure)` format. The directory pointing to the project should be specified in the ``spinoza_setup``-file as ``$DIR_PROJECTS``. Then specify the the project name as ``$PROJECT``. It is assumed your converted data lived in:

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

Preprocessing of anatomical data
===========================================

The ``master`` script executes different (pre-)processing steps via different ``modules`` prefixed by *spinoza_* in the ``<shell>`` folder. The modules can be called like: 

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
    -m <module>       always required, specifies which module to run. Multiple modules should be com-
                        ma separated
    -h <hemisphere>   used for lineplanning, denotes which hemisphere to process
    -o <overwrite>    used for multiple modules to overwrite (i.e., delete) files
    -n <session>      session nr, used to create file structure according to BIDS
    -t <type>         used for fMRIprep, to run either the 'anat' workflow only, or also include func
                        data ('func)
    -l <which mod>    look for a module number given a string to search for (e.g., master -l layer)
    -q <info mod>     query the usage-information of a given module number: e.g., master -m 00 -q
                        brings up the help-text from module 00 (spinoza_lineplanning)
    -v <vertices>     manually insert vertices to use as "best_vertex" (module 18). Should be format-
                        ted as -v "lh,rh" (please use two vertices; one for left hemi and one for right
                        hemi > makes life easier)
    -p <prf_model>    switch to specify what type of model to use for the pRF-modeling. For now, the
                        accepted options are 'gauss' for Gaussian model, and 'norm' for Normalization
                        model. Should/can be use for module 17, pRF-fitting
    -c <cat_run>      Do full processing with CAT12, assuming module 8 was skipped. This does itera-
                        tive SANLM-filtering, bias correction, and intensity normalization. Can some-
                        times be overkill, so generally we use CAT12 for the additional segmentations
                        and brain extraction
    -w <wagstyl>      Use Wagstyl's equivolumetric layering for layerification. Use Nighres otherwise
    -x <**kwargs>     Can be used for a variety of options by setting it to ANYTHING other than empty. 
                        Usages include:
                        - spinoza_scanner2bids; denote the session number when you have a regular 
                            pRF-session, as the "-n" flag is used to specify this. 
                        - spinoza_qmrimaps; turn on Universal Pulses
                        - spinoza_fmriprep; remove surface_recon_wf folder. Handy for re-running with
                            new reconstruction
                        - spinoza_bestvertex; disable FreeView during vertex selection                        
                        - spinoza_profiling; specify the type of file to use
    -u <config>       specify configuration file for fMRIprep (see linescanning/bin/data for examp-
                        les). If not specified, no configuration will be used. If ses-1 is specified,
                        linescanning/bin/data/fmriprep_config.json is used; above that, fmriprep_con-
                        fig2.json is used. Enter '-u none' if you have multiple sessions and you want
                        to process everything without filtering

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

    Available modules:                                     Script                         Time (pp)
    - 00: Register anat ses-1 to ms ses-1 for planning   (spinoza_lineplanning)
    - 01: Make new session for subject                   (spinoza_newBIDSsession)
    - 02: Convert raw files to nifti                     (spinoza_scanner2bids)
    - 03: Reconstruction of line data                    (spinoza_linereconstruction)
    - 04: Estimate T1's from mp2rage and memp2rage       (spinoza_qmrimaps)               00:01:30
    - 05: Register T1 from memp2rage to T1 from mp2rage  (spinoza_registration)           00:03:00
    - 06: Register T1 from mp2rage to MNI152             (spinoza_registration)             -
    - 07: Averaging of (me)mp2rages and other qMRI-maps  (spinoza_averagesanatomies)      00:03:00
    - 08: Perform bias field correction on 2nd inversion (spinoza_biascorrection)         00:07:00
    - 09: Brain extract (CAT12 || ANTs || FSL)           (spinoza_brainextraction)        00:10:00
    - 10: Noise level estimation                         (spinoza_noiselevelestimation)   00:00:30
    - 11: Dura estimation and skull estimations          (spinoza_createduraskullmasks)   00:08:00
    - 12: Create mask of sagittal sinus                  (spinoza_createsagsinusmask)     00:03:00
    - 13: Combine all masks and apply to average         (spinoza_masking)                00:02:30
    - 14: Surface segmentation with FreeSurfer           (spinoza_segmentfreesurfer)      09:30:00
    - 15: Preprocessing with fMRIprep                    (spinoza_fmriprep)               14:00:00
    - 16: Data denoising with pybest                     (spinoza_denoising)              00:05:00
    - 17: pRF-fitting with pRFpy                         (spinoza_fitprfs)                00:07:00
    - 18: Determine best vertex with pycortex            (spinoza_bestvertex)             00:01:30
    - 19: Tissue segmentation with FAST                  (spinoza_segmentfast)            00:17:00
    - 20: Tissue segmentation with MGDM                  (spinoza_segmentmgdm)            00:11:45
    - 21: Region extraction with nighres                 (spinoza_extractregions)         00:02:00
    - 22: Cortex reconstruction with nighres             (spinoza_cortexreconstruction)   00:05:30
    - 23: Layering with nighres/surface_tools            (spinoza_layering)               00:07:30
    - 24: Subcortex parcellation with nighres            (spinoza_segmentsubcortex)       00:10:00
    - 25: Project line to surface                        (spinoza_line2surface)           00:02:00
    - 26: Sample a dataset across depth with Nighres     (spinoza_profiling)              00:20:00

    Notes:
    - Module 5 (register T1 to MNI152) and 13 (brain extract T1 with ANTs) are relatively obsolete
        These commands are therefore commented out.
    - Time = approximate run time per subject > Run time of total pipeline is about 1hr30 on SGE
    - There are two modes we can run this pipeline in, to be specified in the setup:
        a session where we have both MP2RAGE and MP2RAGEME data, and a solo MP2RAGE mode. This can be
        specified by setting the "DATA" variable in the setup script.
        You can see if it was the MP2RAGEME-mode with the "space-average" flag, while the MP2RAGE-mode
        is annotated by "space-mp2rage".
    - To know which mode we're running, type from your programs directory "call_whichmode"
 
All of these modules loop, by default, through *all* the subjects present in ``$DIR_PROJECTS/$PROJECT`` (which is defined as ``DIR_DATA_HOME``. Also by default, it assumes that we only want to process ``ses-1``. To steer the pipeline towards particular directories, such as individual subjects, you can use additional flags when calling the ``master`` script. For instance:

.. code:: bash

    $ master -m 04 -s 001

Would only run module **04** (spinoza_qmrimaps)` for **sub-001**. The help info shows more modules than you'll actually need to run in order to get your data in analyzable state. 

Generally, the following steps are included:

Combine **INV1** and **INV2** into ``T1w`` and ``T1map`` with Pymp2rage_. If you do not combine **INV1** and **INV2** yourself, - you already have a ``T1w``-/ ``T1map``-file in ``DIR_DATA_HOME`` - you can skip the part below and go to ``module 06``:

.. code:: bash

    $ master -m 04 # spinoza_qmrimaps

If you have a ``T2w``-image, you can create a sagittal sinus mask. In the ``misc`` folder, we provide a heavily dilated sinus mask in MNI-space that we use to exclude the T2-values from the subcortex. First, we register the subject's anatomy to the MNI-space, use these warp files to transform the dilated sinus mask to subject space, and multiply our sinus mask derived from the T2 with that dilated mask. This way, we exclude any subcortical voxels from being masked out. This procedure will lead to a file called ``<subject>_<ses-?>_acq-(ME)MP(2)RAGE_desc-sagittalsinus`` in the ``derivatives/masked_(me)mp(2)rage/<subject>/<ses-?>`` folder. If you do *NOT* have a ``T2w``-file, skip these steps and continue to ``module 08`` 

.. code:: bash

    $ master -m 06 # spinoza_registration
    $ master -m 07 # spinoza_sinusfrommni

Bias correct (*SPM*) and denoise (*SANLM*) your ``T1w``-image. If you did not use Pymp2rage_, we'll be looking for the ``T1w``-file in ``DIR_DATA_HOME/<subject>/<ses>/anat``. If you did use Pymp2rage_, we'll be looking for the file in ``<DIR_DATA_HOME>/derivatives/pymp2rage/<subject>/<ses>``.

.. code:: bash

    $ master -m 08 # spinoza_biassanlm

Perform brain extraction and white/gray matter + CSF segmentations with CAT12_.

.. code:: bash

    $ master -m 09 # spinoza_brainextraction

Create a ``desc-outside.nii.gz`` mask in the ``derivatives/masked_(me)mp(2)rage/<subject>/<ses-?>`` folder that contains the sinus, dura, and tentorium (stuff between the cerebellum and cortex) and apply this mask to the ``T1w`` image so FreeSurfer doesn't label non-gray matter voxels as gray matter.

.. code:: bash

    $ master -m 13 # spinoza_masking

Then, run the FreeSurfer_ reconstruction *outside* of fMRIprep_. This is because the newest version of FreeSurfer_ deals better with ``T2w``-images than the version used in fMRIprep_.

.. code:: bash

    $ master -m 14

Once you're satisfied with the surface reconstruction, you can proceed with fMRIprep_. First, we'll run the *anatomical workflow* only. This is to safe some time, because these steps we need to do iteratively; if you're already confident in the reconstruction, you can use ``-t func`` to include functional data in fMRIprep_ as well. 

.. code:: bash

    $ master -m 15 # spinoza_fmriprep
    $ master -m 15 -t func 

We then proceed by denoising the functional data using Pybest_. This takes the ``confound``-file generated by fMRIprep_ and denoised the functional data. By default I do this in ``fsnative``, so if you want a different space to be processed you'll need to adapt ``spinoza_denoising`` accordingly.

.. code:: bash

    $ master -m 16 # spinoza_denoising

You can do some pRF-fitting as well. This is pretty tailored to the specific project with which the package is designed. It encompasses both the grid/iterative fit using *Gaussian* or *Divisive Normalization* models as implemented in pRFpy_. If your project involves relatively straightforward pRF-fitting routines, you can use this module. Otherwise, take the output from Pybest_ as starting point for your project-specific package.

.. code:: bash

    $ master -m 17 # spinoza_fitprfs

Continuing with the anatomical pipeline, we now enter the Nighres_ stage, to fully optimize the quality of segmentations. These modules can be run pretty much successively, and consist of ``MGDM``, ``region extaction``, ``CRUISE``, and ``volumetric layering``:

.. code:: bash

    $ master -m 20 # spinoza_segmentmgdm
    $ master -m 21 # spinoza_extractregions > also combines ALL previous segmentations into optimized levelsets
    $ master -m 22 # spinoza_cortexreconstruction > currently mimicks CRUISE, rather than running it
    $ master -m 23 # spinoza_layering > can be done with Nighres or Wagstyl's surface_tools

Which is equivalent to:

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