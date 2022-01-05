.. include:: links.rst

------------
Bash modules
------------

The bash modules are controlled by the ``master`` script, and perform the big processing steps, from preprocessing to FreeSurfer_ and fMRIprep_. Most of these modules have the ``-s`` and ``-n`` flags, meaning you can limit processing to a particular subject (e.g., ``-s 001``) or session (e.g., ``-n 1``). By default, the ``<session>`` flag is set to ``1``, and the subject flag to ``all``. These flags trickle down from the ``master`` script, meaning this can be controlled by calling:

.. code:: bash

    $ master -m <module> -n <session> -s <subject>

Below the help information for each module, which can also be called with:

.. code:: bash

    $ master -m <module> -q

00: spinoza_lineplanning
===========================================

::

    ---------------------------------------------------------------------------------------------------
    spinoza_lineplanning

    Quick registration of partial anatomy from ses-2 with whole brain anatomy from ses-1 to plan the line
    as accurately as possible.

    Assumes you have run the pycortex notebook to obtain the text file with the values describing the
    orientation of the line in the first session anatomy based on the minimal curvature (and pRF-mapping).
    This notebook should ouput a line_orientation.txt file in the pycortex/sub-xxx directory. It will
    complain if this file's not here, we need it!!

    Steps:
    - Convert data                          (call_dcm2niix)
    - Create initial matrix                 (itksnap)
    - Registration                          (call_antsregistration; ANTs)
    - Store output registration in txt file
    - Fetch output from pycortex notebook or spinoza_bestvertex
    - Calculate values from both files      (call_mrconsole)

    Usage:
    spinoza_lineplanning  -s <subject ID>
                            -n <session ID>
                            -i <path to raw session 2 anatomy>
                            -a <path to session 1 anatomy>
                            -p <path to file containing the orientation of the line>
                            -h <which hemisphere> (left | right)
                            -o <overwrite point files> (omit if we should not overwrite)

    Example:
    spinoza_lineplanning -s sub-001 -n 3 -i /path/to/raw/ses-2 -a /path/to/ses-1_anat.nii.gz -p /path/to
                            /pycortex_file.csv -h "left"

    Notes:
    - You NEED ANTs for this to run
    - It also depends on python3; if something doesn't seem to work, try to update the package
        with python -m pip install <package> --upgrade

    ---------------------------------------------------------------------------------------------------

02: spinoza_scanner2bids
===========================================
 
::

    ---------------------------------------------------------------------------------------------------
    spinoza_scanner2bids

    convert raw data from the scanner to nifti format. Depending on which session we're analyzing, we'll
    use either call_dcm2niix.py (session 1 - which is FromScannerToBIDS.py from M. Aquil) which can deal
    nicely with the anatomical and functional stuff or call_dcm2niix.sh, which is more specific for the
    line scanning stuff.

    Input options:
    <project root>    directory to output BIDSified data to
    <sourcedata>      directory containing to be converted data
    <session number>  session number to be converted (ses-1 and ses-${SES_NR} require different types of con-
                        version, whereas session 1 can be converted with call_dcm2niix.py, we need to
                        use a custom conversion for session 2 to get it somewhat in BIDS-format)

    Example:
    spinoza_scanner2bids /path/to/project_root /path/to/your/project/sourcedata 1
    spinoza_scanner2bids (shows this help text)
    spinoza_scanner2bids $DIR_DATA_HOME $DIR_DATA_SOURCE 1

    Notes:
    Assumes that you ran spinoza_newBIDSsession with the following data structure:
    > PROJECT
        > sourcedata
        > sub-001
            > ses-1
            > .PAR
            > .REC

    Converts to:
    > PROJECT
        > sub-001
        > ses-x
            > anat
            > func
            > fmap

    ---------------------------------------------------------------------------------------------------------

03: spinoza_linereconstruction
===========================================

::

    ---------------------------------------------------------------------------------------------------
    spinoza_linereconstruction

    wrapper for call_linerecon that performs the reconstruction of the line data. Uses MRecon, so we can
    only run it on the spinoza cluster. It calls upon call_linerecon, which internally uses a template
    for the reconstruction with MRecon based on scripts provided by Luisa Raimondo.

    Usage:
    spinoza_linereconstruction <project root directory> <sourcedata>

    Arguments:
    <project root>      base directory containing the derivatives and the subject's folders.
    <sourcedata>        base directory containing the raw data for reconstruction

    Eample:
    spinoza_linereconstruction ${DIR_DATA_HOME} ${DIR_DATA_SOURCE}

    Notes:
    relies on matlab scripts stored in '/data1/projects/MicroFunc/common'. As it relies on MRecon,
    we can only run this on the spinoza server

    ---------------------------------------------------------------------------------------------------

04: spinoza_qmrimaps
===========================================

::

    ---------------------------------------------------------------------------------------------------
    spinoza_qmrimaps

    wrapper for estimation of T1 and other parametric maps from the (ME)MP2RAGE sequences by throwing
    the two inversion and phase images in PYMP2RAGE (https://github.com/Gilles86/pymp2rage).

    Usage:
    spinoza_qmrimaps <-s sub> <-n ses> <project root> <derivatives> <session nr>

    Example:
    spinoza_qmrimaps DIR_DATA_HOME DERIVATIVES/pymp2rage 1
    spinoza_qmrimaps $DIR_DATA_HOME $DIR_DATA_DERIV
    spinoza_qmrimaps -s 999 -n 1 $DIR_DATA_HOME $DIR_DATA_DERIV/pymp2rage

    Notes:
    Has the '-s' and '-n' switches to specify a particular subject and session if present

    ---------------------------------------------------------------------------------------------------

06: spinoza_registration
===========================================

::

    ---------------------------------------------------------------------------------------------------
    spinoza_registration

    wrapper for registration with ANTs. This script should be preceded by spinoza_memp2rage & spinoza-
    mp2rage to create the UNI T1w images that we're registering here. It utilizes the variable ACQ in
    the setup script to derive what should be registered to what. In theory, the first element of ACQ
    is taken as reference, and the second element will be registered to that. If ACQ has only one ele-
    ment and 'MNI' is specified, this first element is registered to the MNI template. This step is
    relatively obsolete given that we don't really need it in MNI space + we can do that step with fMRI-
    prep.

    By default, the data type is set to the first element of the ACQ-variable as specified in the setup
    script and is no initial transformation file required.

    Usage:
    spinoza_registration <-s sub> <-n ses> <root input> <root output> <session> <data type> <initial
                        trafo>

    Example:
    spinoza_registration <project>/derivatives/pymp2rage <project>/derivatives/ants <session> 
                        mp2rage n
    spinoza_registration -s 001 -n 1 <project>/derivatives/pymp2rage <project>/derivatives/ants <ses-
                        sion> 

    Notes:
    Has the '-s' and '-n' switches to specify a particular subject and session if present

    ---------------------------------------------------------------------------------------------------

07: spinoza_sinusfrommni
===========================================

::

    ---------------------------------------------------------------------------------------------------
    spinoza_sinusfrommni

    This script takes the registration matrix from MNI to subject space to warp the sagittal sinus mask
    in MNI-space to the subject space. We then multiply this image with the T1w/T2w ratio to get a de-
    cent initial estimate of the sagittal sinus

    Usage:
    spinoza_sinusfrommni <-s sub> <-n ses> <anat> <registration> <mask directory>

    Arguments:
    <-s sub>            digit subject number following 'sub-'
    <-n ses>            integer session number following 'ses-'
    <anat directory>    directory containing the T1w and T2w files; should generally be pymp2rage-
                        folder
    <registration>      path to directory where registration file is
    <mask directory>    path to output mask directory

    Eample:
    spinoza_sinusfrommni ${DIR_DATA_DERIV}/pymp2rage ${DIR_DATA_DERIV}/ants ${DIR_DATA_DERIV}/manual_
                        maskss
    spinoza_sinusfrommni -s 001 -n 1 ${DIR_DATA_DERIV}/pymp2rage ${DIR_DATA_DERIV}/ants 
                            ${DIR_DATA_DERIV}/manual_masks

    Notes:
    - Has the '-s' and '-n' switches to specify a particular subject and session if present

    ---------------------------------------------------------------------------------------------------

08: spinoza_biassanlm
===========================================

::

    ---------------------------------------------------------------------------------------------------
    spinoza_biassanlm

    Sometimes CAT12 can be a bit of an overkill with smoothing and bias corrections. This script should
    be run prior to "spinoza_brainextraction", and runs a SANLM-filter over the image as well as an bias
    field correction with SPM. The subsequent "spinoza_brainextraction" should be run with the "-m brain"
    flag as to turn off bias correction and denoising with CAT12.

    The input image is expected to reside in the input directory and to contain "acq-${ACQ1}" and end
    with *T1w.nii.gz.

    Usage:
    spinoza_biascorrection <-s sub> <-n ses> <dir_to_input_file> <dir_to_output>

    Outputs:
    <dir_to_output>/<subject>/ses-1/<subject>${SES}_acq-${ACQ1}_<T1w|inv-2>.nii.gz

    Example:
    spinoza_biascorrection $DIR_DATA_DERIV/pymp2rage
    spinoza_biascorrection $DIR_DATA_DERIV/pymp2rage $DIR_DATA_DERIV/bias
    spinoza_biascorrection -s 001 -n 1 $DIR_DATA_HOME

    Notes:
    - Has the '-s' and '-n' switches to specify a particular subject and session if present
    - <dir_to_output> is not a required argument. If not specified, the input file will be overwritten.
        The latter is default.

    ---------------------------------------------------------------------------------------------------

09: spinoza_brainextraction
===========================================

::

    ---------------------------------------------------------------------------------------------------
    spinoza_brainextraction

    wrapper for brain extraction with ANTs, FSL, or CAT12 If you use ANTs, specify a prefix; if you use 
    FSL, specify an output name. Not case sensitive (i.e., you can use ANTs/ants or FSL/fsl). Assumes 
    that if you select FSL, we brain extract the INV2 image and if we select ANTs/CAT12, we brain extract 
    the mp2rage T1w with bias field correction. If you want to brain extract something else, either use
    call_fslbet, call_antsbet, or call_cat12. It performs N4 biasfield correction internally. Make sure 
    you added the location of antsBrainExtraction.sh to your path e.g., in your ~/.bash_profile :
    \"export PATH=PATH:/directory/with/antsBrainExtraction.sh\"

    Usage:
    spinoza_brainextraction <input dir> <skullstrip output> <mask output> <ants/FSL/cat12>

    Example:
    spinoza_brainextraction dir/to/t1w dir/to/skullstrip /dir/to/masks ants
    spinoza_brainextraction dir/to/pymp2rage dir/to/cat12 /dir/to/masks cat12
    spinoza_brainextraction dir/to/inv2 dir/to/skullstrip /dir/to/masks inv2

    Notes:
    Has the '-s' and '-n' switches to specify a particular subject and session if present

    ---------------------------------------------------------------------------------------------------

13: spinoza_masking
===========================================

::

    ---------------------------------------------------------------------------------------------------
    spinoza_masking

    Mask out the dura and skull from the T1-image to reduce noise. It follow Gilles' masking procedure,
    by setting the contents of dura ('outside') and other masks ('inside') to zero. The idea is to run
    this, run fMRIprep, check segmentations, manually edit it as "sub-xxx_ses-1_acq-MP2RAGE_desc-manual
    wmseg" or something alike. These 'manualseg' will be taken as 'inside' to boost areas that were not
    counted as brain.

    By default it will overwite existing files to avoid that you have to contanstly remove them, but for
    debugging reasons this module also has the option to disable overwrite mode. You can do this by spe-
    cifying 'spinoza_maskaverages <dir subject> <derivatives> n' or with the master script itself: 'mas-
    ter -m 13 -n'. As said, if you specify 'y' or leave it empty, it will overwrite existing files.

    Usage:
    spinoza_masking <-s sub> <-n ses> <directory to anats> <output dir> <mask dir> <skullstrip dir>
                    <overwrite mode>

    Example:
    spinoza_masking DIR_DATA_DERIV/pymp2rage DIR_DATA_DERIV/masked_mp2rage DIR_DATA_DERIV/manual_masks 
                    DIR_DATA_DERIV/skullstripped y

    spinoza_masking -s 001 -n 1 DIR_DATA_DERIV/pymp2rage DIR_DATA_DERIV/masked_mp2rage DIR_DATA_DERIV/
                    manual_masks DIR_DATA_DERIV/skullstripped y

    Notes:
    - Will overwrite by default, you can disable overwrite in the master call with -o "n"
    - Has the '-s' and '-n' switches to specify a particular subject and session if present

    ---------------------------------------------------------------------------------------------------

14: spinoza_freesurfer
===========================================

::
    
    ---------------------------------------------------------------------------------------------------
    spinoza_freesurfer

    Surface parcellation with FreeSurfer. We only need to specify where to look for the T1w-image which
    stage we should run (default is 'all'), and where to look for a T2w-image if there is one.

    General approach for segmentation:
    - Run autorecon1:   call_freesurfer -s <subj ID> -t <T1> -p <T2> -r 1)      ~an hour
    - Fix skullstrip:   call_freesurfer -s <subj ID> -o gcut                    ~10 minutes
    - Run autorecon2:   call_freesurfer -s <subj ID> -r 2                       ~few hours
    - Fix errors with:
        - norm; then run call_freesurfer -s sub-001 -r 23 -e cp                 ~few hours
        - pia;  then run call_freesurfer -s sub-001 -r 23 -e pial               ~few hours

    You can specify in which directory to look for anatomical scans in the first argument. Usually,
    this is one of the following options: DIR_DATA_HOME if we should use the T1w in the project/sub-xxx
    /anat directory, or DIR_DATA_DERIV/pymp2rage to use T1w-images derived from pymp2rage, or DIR_DATA_
    DERIV/masked_mp2rage to use T1w-images where the dura and sagittal sinus are masked out (should be
    default!). In any case, it assumes that the file is in YOURINPUT/sub-xxx/ses-1/. If the input is
    equal to the DIR_DATA_HOME variable, this will be recognize and 'anat' will be appended to YOURINPUT
    /sub-xxx/ses-1/anat.

    You can also specify a directory where the T2-weighted image is located. Do this the same way as de-
    scribed above. To you path, sub-xxx/ses-x will be appended if the input path is not equal to DIR_DATA
    _HOME. Again, if it is, sub-xxx/ses-x/anat will be appended as well.

    Usage:
    spinoza_freesurfer <-s sub> <-n ses> <directory with anats> <stage> <T2-directory>

    Example:
    spinoza_freesurfer $DIR_DATA_DERIV/masked_mp2rage all $DIR_DATA_HOME
    spinoza_freesurfer -s 001 -n 1 $DIR_DATA_ANAT all $DIR_DATA_HOME

    Notes:
    - Has the '-s' and '-n' switches to specify a particular subject and session if present

    ---------------------------------------------------------------------------------------------------

15: spinoza_fmriprep
===========================================

::

    ---------------------------------------------------------------------------------------------------
    spinoza_fmriprep

    preprocess structural and functional data with fMRIprep. It uses the singularity container in pro-
    grams/packages/fmriprep/containers_bids-fmriprep--20.2.0.simg (which is symlink'ed to /packages/sin-
    gularity_containers/containers_bids-fmriprep--20.2.0.simg). You can also specify your own singu-
    larity image.

    If you have a T2-weighted image as well, you can specify the root directory to that image. If it
    exists, we will copy it to the directory where the T1-weighted is located (<input directory>) so
    that it is included by fMRIprep.

    Arguments:
    <-s subject>    subject ID (optional)
    <-n session>    session ID (optional)
    <-c config>     configuration file as specified in /data/fmriprep_config?.json
    <-f func dir>   directory containing functional data; used after running FreeSurfer outside of
                    fMRIprep <optional>
    <-r>            remove surface_recon_wf to project functional data to new surface; used after
                    repeatedly enhancing the anatomical segmentation before injecting the functio-
                    nal data <optional>

    <anat dir>      directory containing the anatomical data. Can also be the regular project root
                    folder if you want fMRIprep do the surface reconstruction

    <derivatives>   output folder for fMRIprep; generally this will be <project>/derivatives

    <mode>          run anatomical workflow only with 'anat', or everything with 'func'

    <T2 dir>        if you have a T2w-file, but that is not in <anat dir> (because you preprocessed
                    the T1w-file, but not the T2w-file), you can specify the directory where it lives
                    here. Generally this will be the same as <func dir>

    Usage:
    spinoza_fmriprep <-s sub> <-n ses> <-f func dir> <anat dir> <derivaties folder> <mode> <T2 dir>

    Example:
    spinoza_fmriprep <project>/derivatives/masked_mp2rage <project>/derivatives anat
    spinoza_fmriprep -s 001 -n 1 -f <project> <project>/derivatives/masked_mp2rage <project>/deri-
                    vatives anat

    Notes:
    - Has the '-s' and '-n' switches to specify a particular subject and session if present
    - Has an '-f' flag to spcify a directory containing functional data. If your functional data is
        stored in the same directory as the anatomical data, you can leave the flag empty

    ---------------------------------------------------------------------------------------------------

16: spinoza_denoising
===========================================

::

    ---------------------------------------------------------------------------------------------------
    spinoza_denoising

    wrapper for call_pybest that does the denoising of fMRI-data based on the confound file created
    during the preprocessing with fmriprep.

    Usage:
    spinoza_denoising <-s sub> <-n ses> <fmriprep directory> <pybest output directory>

    Arguments:
    <-s sub>            digit subject number following 'sub-'
    <-n ses>            integer session number following 'ses-'
    <project root>      base directory containing the derivatives and the subject's folders.
    <derivatives>       path to the derivatives folder

    Eample:
    spinoza_denoising ${DIR_DATA_HOME} ${DIR_DATA_DERIV}
    spinoza_denoising -s 001 -n 1 ${DIR_DATA_HOME} ${DIR_DATA_DERIV}

    Notes:
    - Has the '-s' and '-n' switches to specify a particular subject and session if present

    ---------------------------------------------------------------------------------------------------

17: spinoza_fitprfs
===========================================

::

    ---------------------------------------------------------------------------------------------------
    spinoza_fitprfs

    wrapper for call_prf that does the pRF-fitting using the output from pybest and the package pRFpy.
    If there is no design matrix for the pRF-modeling, we need to specify a path to the screenshots as
    outputted by the experiment on the stimulus computer. Not really sure what you should do if you do
    not have these screenshots, as it's important for pRFpy to know what kind of stimulus was presented
    when..

    Usage:
    spinoza_fitprfs <-s sub> <-n ses> <prf dir> <pybest dir> <png dir>

    Arguments:
    <-s sub>       digit subject number following 'sub-'
    <-n ses>       integer session number following 'ses-'
    <-m model>     'gauss' or 'norm' for the type of model to use for the fitting
    <-g grid>      only run grid fit, skip iterative fit
    <prf dir>      base output directory for prf data (derivatives/prf)
    <pybest dir>   base input directory with pybest data (derivatives/pybest)
    <png dir>      base path to where the pRF-experiment's Log-directories with png's is
    <overwrite>    delete existing file and re-run analysis (y=yes|n=no)


    Eample:
    spinoza_fitprfs ${DIR_DATA_DERIV}/prf ${DIR_DATA_DERIV}/pybest ${DIR_DATA_SOURCE}
    spinoza_fitprfs -s 001 -n 1 ${DIR_DATA_DERIV}/prf ${DIR_DATA_DERIV}/pybest ${DIR_DATA_SOURCE}

    Notes:
    - Has the '-s' and '-n' switches to specify a particular subject and session if present
    - Also has overwrite switch
    - Has the '-m' switch to specify type of model

    ---------------------------------------------------------------------------------------------------

18: spinoza_bestvertex
===========================================

::

    ---------------------------------------------------------------------------------------------------
    spinoza_bestvertex

    wrapper for call_pycortex to calculate the best vertex and normal vector based on the minimal curva-
    ture given an ROI.

    this script requires input from FreeSurfer, so it won't do much if that hasn't run yet. Ideally, you
    should perform FreeSurfer with the pRF-mapping in fMRIprep (module before [13]), then run this thing so
    it can also take in the pRF to locate an even better vertex.

    (need to update this so it takes in variable areas, now it's just set to V1)

    Args:
    -o              rename existing file to numerically increasing (e.g., line_pycortex > line_pycor-
                    tex1.csv) so that a new line_pycortex.csv-file is created. Given that the original
                    file is NOT deleted, I consider this a soft-overwrite mode. You can always manual-
                    ly delete unwanted/unused files.
    -f              prevent FreeView from opening while verifying the location. ONLY do this if you
                    already know the position. Generally only used for debugging purposes.
    -s <subject>    subject ID as used throughout the pipeline without prefix (e.g., sub-001 > 001)
    -n <session>    session ID used to extract the correct pRF-parameters. Will combined with <deriva-
                    tives>/prf/ses-<session>
    -v <vertices>   manually specify two vertices to use instead of having the program look for it. 
                    The format is important here and should always be: "<vertex in lh>,<vertex in rh>".
                    Always try to specify two vertices; doesn't matter too much if one is not relevant
    <project root>  Path to where the subject-drectories are located; used to loop through subjects, 
                    unless the -s switch is triggered
    <derivatives>   Derivatives folder containing the output from pycortex and pRF-fitting. Looks for 
                    <derivatives>/freesurfer, <derivatives>/pycortex, and <derivatives>/prf for the 
                    surface reconstruction, pycortex-import, and pRF-data, respectively
    <ROI>           Region-of-interest to use. Should be a FreeSurfer-named label-file or a custom
                    file in the format of FreeSurfer label-file: you could for instance draw an ROI
                    in volume-space in FreeView, convert that to a label ("mri_vol2label") and insert
                    that to look for a vertex (might be useful for finding a mask representing indi-
                    vidual fingers when the motor cortex mask from FreeSurfer is to coarse. UNTESTED)

    Usage:
    spinoza_bestvertex <-s sub> <-v "lh,rh"> <sourcedata> <derivatives> <ROI>

    Example:
    spinoza_bestvertex $DIR_DATA_HOME $DIR_DATA_DERIV V1_exvivo.thresh
    spinoza_bestvertex -s 001 $DIR_DATA_HOME $DIR_DATA_DERIV V1_exvivo.thresh
    spinoza_bestvertex -s 001 -v "1957,8753" $DIR_DATA_HOME $DIR_DATA_DERIV V1_exvivo.thresh

    Notes:
    - Has a '-v' switch to specify your own vertices; mind the format!! "lh,rh"
    - Both these switches are optional and can be controlled with the master-script

    ---------------------------------------------------------------------------------------------------

19: spinoza_segmentfast
===========================================

::
    
    ---------------------------------------------------------------------------------------------------
    spinoza_segmentfast

    tissue segmentation with FAST using skullstripped inputs created during spinoza_maskaverages. It is
    important that the range of these images is set correctly, with T1w having a range of 0-4095, and
    the T1map having a range of (0,5050). This should automatically be the case if you have ran the py-
    mp2rage module in combination with the masking module prior to running this. If not, run call_rescale
    on these images

    By default, it's set to overwrite existing files in order to be able to iterate over the structural
    preprocessing pipeline for optimizing the segmentations. To disable, run this particular module with
    the overwrite switch set to 'n': master -m <module> -o n

    Usage:
    spinoza_segmentfast <-s sub> <-n ses> <skullstripped dir> <output dir> <overwrite>

    Example:
    spinoza_segmentfast $DIR_DATA_DERIV/skullstripped $DIR_DATA_DERIV/fsl n
    spinoza_segmentfast -s 001 -n 1 $DIR_DATA_DERIV/skullstripped $DIR_DATA_DERIV/fsl n

    Notes:
    - Has the '-s' and '-n' switches to specify a particular subject and session if present

    ---------------------------------------------------------------------------------------------------

20: spinoza_segmentmgdm
===========================================

::

    ---------------------------------------------------------------------------------------------------
    spinoza_segmentmgdm

    Tissue segmentation using nighres' MGDM. It assumes that you've run module from this pipeline before,
    so that you're directory structure is like derivatives/<process>/sub-xxx/ses-x. For this script, you
    need to give the path to the skullstripped directory up until sub-xxx, the output mgdm directory,
    and the directory containing masks that are used to filter out stuff in the MGDM segmentation.

    By default, it's set to overwrite existing files in order to be able to iterate over the structural
    preprocessing pipeline for optimizing the segmentations. To disable, run this particular module with
    the overwrite switch set to 'n': master -m <module> -o n

    Arguments:
    flags
        -s <sub>    str| subject ID as specifed in DIR_DATA_HOME
        -n <ses>    int| session number
        -t <type>   str| run regular MGDM [empty] or GdH-pipeline version (call_gdhmgdm) ["gdh"]

    positional
        <skullstripped>   path to skullstripped data
        <mgdm>            path to output directory
        <masks>           path to masks
        <overwrite>       enable ('y') or disable ('n') overwrite mode (deletes existing files)

    Usage:
    spinoza_segmentmgdm <-s sub> <-n ses> <type> <skullstripped> <mgdm> <masks> <overwrite>

    Example:
    spinoza_segmentmgdm $SKULLSTRIP $NIGHRES/mgdm $MASKS n
    spinoza_segmentmgdm -s 001 -n 1 $SKULLSTRIP $NIGHRES/mgdm $MASKS n
    spinoza_segmentmgdm -s 001 -n 1 -t gdh $SKULLSTRIP $NIGHRES/mgdm $MASKS n

    Notes:
    - Has the '-s' and '-n' switches to specify a particular subject and session if present
    - Has a '-t' flag to specify whether you want to run the GdH-version (call_gdhmgdm) or the regu-
        lar version of MGDM segmentation (call_nighresmgdm)

---------------------------------------------------------------------------------------------------

21: spinoza_extractregions
===========================================

::
    
    ---------------------------------------------------------------------------------------------------
    spinoza_extractregions

    region extraction using nighres. Calls on call_nighresextractregions; see that file for more info-
    rmation on the required inputs. This script is by default in overwrite mode, meaning that the files
    created earlier will be overwritten when re-ran. To disable, run this module as master -m <module>
    -o n. The second arguments points to the root directory of where the level set probabilities are
    stored. Normally, for region extraction, you use the output from MGDM. You can, however, create a
    custom levelset (see call_gdhcombine).

    For this, you will need four directories: the REGION directory (with tissue classification from
    MGDM), the FreeSurfer directory (read from you SUBJECTS_DIR), the fMRIprep-directory with tissue
    classification from FAST, and the MASK-directory containing manual edits. The REGION directory is
    the directory that will be created first, the FreeSurfer directory will be read from the SUBJECTS_DIR
    variable, the fMRIprep-directory you'll need to specify with the -f flag BEFORE (!!) the positional
    arguments, and the MASK-directory you will already specify.

    Usage:
    spinoza_extractregions <-s sub> <-n ses> <-f fprep> <nighres root> <custom comb> <region to
                            extract> <ow>


    Example:
    spinoza_extractregions $DIR_DATA_DERIV/nighres $DIR_DATA_DERIV/manual_masks cerebrum n
    spinoza_extractregions -s -001 -n 1 $DIR_DATA_DERIV/nighres $DIR_DATA_DERIV/manual_masks cerebrum n
    spinoza_extractregions -s -001 -n 1 -f $DIR_DATA_DERIV/fmriprep $DIR_DATA_DERIV/nighres
                            $DIR_DATA_DERIV/manual_masks cerebrum n

    Notes:
    - If you want a custom levelset, specify the '-f' flag pointing to the fMRIprep-directory
    - Has the '-s' and '-n' switches to specify a particular subject and session if present
    - Region to be extracted can be one of:
        > left_cerebrum
        > right_cerebrum
        > cerebrum,
        > cerebellum
        > cerebellum_brainstem
        > subcortex
        > tissues(anat)
        > tissues(func)
        > brain_mask

    ---------------------------------------------------------------------------------------------------

22: spinoza_cortexreconstruction
===========================================

::

    ---------------------------------------------------------------------------------------------------
    spinoza_cortexreconstruction

    cortex reconstruction using nighres. Calls on call_nighrescruise; see that file for more information
    on the required inputs. This script is by default in overwrite mode, meaning that the files created
    earlier will be overwritten when re-ran. To disable, run this module as master -m <module> -o n

    Usage:
    spinoza_cortexreconstruction <project root dir> <prob seg dir> <region> <overwrite>

    Arguments:
    <-s sub>    digit subject number following 'sub-'
    <-n ses>    integer session number following 'ses-'
    <-o>        overwrite mode active
    <project>   parent directory containing the sub-xxx folders
    <prob seg>  directory containing probabilities of tissue segmentation. By default it will use the
                MDGM output, but you can specify your own. E.g., in the case of GdH-pipeline
    <region>    region you wish to reconstruct. Should be same as spinoza_extractregions

    Example:
    spinoza_cortexreconstruction $PROBSEGS $CRUISE cerebrum
    spinoza_cortexreconstruction -s sub-001 -n 1 -o $PROBSEGS $CRUISE cerebellum

    Notes:
    - Has overwrite mode to run the pipeline like G. de Hollander's one
    - Has the '-s' and '-n' switches to specify a particular subject and session if present

    ---------------------------------------------------------------------------------------------------

23: spinoza_layering
===========================================

::

    ---------------------------------------------------------------------------------------------------
    spinoza_layering

    Equivolumetric layering with either nighres or Wagstyl's surface_tools, as specified by the third
    argument. Surface_tools is based on FreeSurfer, so make sure that has run before it. This script is
    by default in overwrite mode, meaning that the files created earlier will be overwritten when re-ran.
    This doesn't take too long, so it doesn't really matter and prevents the need for an overwrite switch.
    To disable, you can specify a condition before running this particular script.

    Possible inputs for software are:
    Nighres: "nighres", "Nighres", "nigh", "NIGHRES" for nighres or"
    Suface tools: "surface", "surface_tools", "SURFACE", "SURFACE_TOOLS", or "ST" for Wagstyl's surface
                tools"

    Usage:
    spinoza_layering <input dir (FS|nighres)> <software>

    Example:
    spinoza_layering $FS surface
    spinoza_layering $DIR_DATA_DERIV/nighres nighres

    Notes:
    - The script will recognize any of the software inputs specified above, with these variations in
        capitalization.
    - The script will look for a surface_tools installation on the PATH and if it can't find it there,
        it will look for the first match in the HOME directory. To be sure, place the script either in
        the home directory or place it on the PATH.
    - If the script doesn't give an error before printing the starting time, it means it found the
        script.

    ---------------------------------------------------------------------------------------------------

24: spinoza_segmentsubcortex
===========================================

::

    ---------------------------------------------------------------------------------------------------
    spinoza_segmentsubcortex

    Subcortex segmentation using Nighres' MASSP-algorithm. Calls on call_nighresmassp; see that file
    for more information on the required inputs.

    Usage:
    spinoza_segmentsubcortex <project root dir> <prob seg dir> <region> <overwrite>

    Arguments:
    <-s sub>    digit subject number following 'sub-'
    <-n ses>    integer session number following 'ses-'
    <qsm dir>   parent directory containing the files required for MASSP. Files should end with:
                    - "*_R1.nii.gz"   > 1/T1map file
                    - "*_R2s.nii.gz"  > 1/T2* file
                    - "*_QSM.nii.gz"  > QSM file
    <output>    root output directory (<subject>/[<ses->] will be appended!)

    Example:
    spinoza_segmentsubcortex $DIR_DATA_HOME $DIR_DATA_DERIV/nighres
    spinoza_segmentsubcortex -s sub-001 -n 1 $DIR_DATA_HOME $DIR_DATA_DERIV/nighres

    ---------------------------------------------------------------------------------------------------

25: spinoza_line2surface
===========================================

::

    ---------------------------------------------------------------------------------------------------
    spinoza_line2surface

    This script contains the registration cascade from single slice/line > multi-slice (9 slice) > low
    resolution MP2RAGE from session 2 > high resolution MP2RAGE from session 1 > FreeSurfer anatomy.
    In spinoza_lineplanning, we have created a matrix mapping the low resolution anatomy to the high
    resolution anatomy from session 1. Because the single- and multislice images are basically in the
    same space (header-wise) as the low resolution anatomical scan, we can just apply this matrix to the
    single- and multi-slice. We then have everything in the space of the high resolution anatomy from
    session 1.

    Now we need a matrix mapping the high resolution anatomical scan from session 1 with the FreeSurfer
    anatomy. For this, we can use FreeSurfer's bbregister, which registers an input image to orig.mgz.
    We can transform both matrices to FSL-format, so we can create a composite transformation matrix
    that we can apply to everything from session 2 with flirt. Because we need all these transformed
    files for pycortex, we will try to store all the files in the pycortex directory, but you can speci-
    fy this yourself (default is the pycortex directory).

    Easiest thing to do is run the "segmentation.ipynb" notebook to warp all segmentations to session 
    1, then everything - HOP - to FreeSurfer and Pycortex space using the matrix created with spinoza_
    lineplanning (this matrix you should definitely have..).

    Usage:
    spinoza_line2surface -s <subject number> -y <anat session 2> -o <outputdir> -i <input dir>

    Arguments:
    -s <sub number>         number of subject's FreeSurfer directory
    -y <anat ses 2>         anatomical image from session 2 as outputted by spinoza_lineplanning
    -o <output directory>   default is bids_root/derivatives/pycortex (easiest to set this to default
                            other make it the same as Pycortex' filestore) [<sub> will be appended]
    -i <directory to warp>  input directory that we need to warp to the surface; I'm assuming a struc-
                            ture like "<input dir>/<sub>/ses-2"

    Example:
    spinoza_line2surface -s sub-001 -y anat_session2 -d /output/directory/whatev

    ---------------------------------------------------------------------------------------------------

26: spinoza_profiling
===========================================

::

    ---------------------------------------------------------------------------------------------------
    spinoza_profiling

    Sample the profile values of a particular image using call_nighresprofsamp. Here, we provide the
    boundaries-image from nighres.layering module and have the program sample values from a particular
    dataset (e.g., T1map) across depth. The first argument specifies where the nighres output is, used
    for both the layering and profile sampling. The second argument is the main directory where we will
    find the file the we want to sample. The last argument specifies a tag to look for the to-be-sampled
    file (e.g., T1map)

    Usage:
    spinoza_profiling <nighres output> <head directory of to-be-sampled files> <tag for to-be-sampled
                        <file>

    Arguments:
    <-s sub>    digit subject number following 'sub-'
    <-n ses>    integer session number following 'ses-'
    <nighres>   parent directory containing the output files of Nighres
                    - <nighres>/<subject>/<session>/layering/*boundaries*
                    - <nighres>/<subject>/<session>/profiling/*lps_data.nii.gz
    <tag>       tag to use to look for to-be-sampled dataset (e.g., T1map)

    Example:
    spinoza_profiling $NIGHRES $DIR_DATA_HOME T1map

    ---------------------------------------------------------------------------------------------------
