CHANGELOG
  - 29/06/2022: rename some modules; improve argument parsing
  - 24/06/2022: specify subject/session flags once, add more long flagged arguments; add check for
                bash version
  - 21/06/2022: assigned some of the KWARGS-argument to long flagged arguments
  - 19/04/2022: remove default use of config file for fMRIprep; now specify it explicitly with the
                '-u' flag.
  - 20/12/2021: added '-u' flag for custom configurations file for fMRIprep
  - 29/10/2021: added '-x' flag for various options. Mainly to specify the session number for
                pRF-sessions, as you'd have to specify '-n prf' as session, but you'll probably
                like to specify a session number too. You can do that by specifying '-n prf -x 1'
  - 20/10/2021: specified input directory for URIS-MDD project for module 8 and 9. This project
                is not run through Pymp2rage, so needed adjustment. 
                Set cat12-processing to 'brain' by default, as running spinoza_biassanlm before
                CAT12 (spinoza_brainextraction) is preferred.
                Assign 'spinoza_sinusfrommni' to module 7, rather than 'spinoza_averageanatomies', 
                which is a pretty redundant module by now.
  - 18/10/2021: added the '-w' flag to specify Wagstyl's equivolumetric layering for layering mo-
                dule.
  - 18/08/2021: added the '-p' flag to specify the kind of model to use during pRF-fitting. 'Gauss'
                or 'norm'
  - 17/06/2021: added module 26, spinoza_profiling, to sample the values of a particular dataset
                across depth using call_nighresprofsamp to create profile of this dataset. Currently
                using this for URIS to create profiles of T1-values across depth
  - 06/04/2021: added '-l' and '-q' flags for master-command; with 'master -l <string>' you can
                look for a module number given a string. With 'master -m <module> -q' you can ask
                to print the help-stuff from the specified modules
  - 15/03/2021: added '-t' flag for fMRIprep module to run either 'anat' or 'func' workflows
  - 10/02/2021: added subcortex parcellation module after 'layering' module. With that, i moved the
                spinoza_line2surface module to module 25
  - 21/01/2021: added session-switch to most modules to account for situations where one might
                have multiple sessions or even none. Default is set to '1', so if you want to pro-
                cess a different session, specify so with the '-n' flag when calling "master -m X"
  - 20/01/2021: added subject-switch to more modules: 'qmrimaps', 'registration', 'biascorrection',
                'brainextraction', 'createduraskullmask', 'sagittalsinus', 'denoising', 'fitprfs',
                'bestvertex', 'segmentfast'
                changed 'spinoza_estimateunit1' to 'spinoza_qmrimaps'
  - 14/01/2021: added subject-switches to 'masking', 'fmriprep', 'mgdm', 'region' modules, so that
                in addition to positional arguments a '-s' switch can be specified to loop only over
                the input subject, not the entire 'sub-' list. This '-s' flag HAS to be specified
                before the positional arguments!
  - 12/01/2021: added overwrite mode for the GDH-pipeline, in addition to nicer specification of
                paths in the FreeSurfer and Nighres modules (better path specification instead of
                'just' DIR_DATA_HOME and DIR_DATA_DERIV)
  - 15/12/2020: added overwrite mode for certain modules (00 and 13)
  - 10/12/2020: added module to do recon after nifti-conversion (moved other modules down)
  - 25/10/2020: added module to do pRF-fitting with pRFpy (moved other modules down)
  - 24/10/2020: added module to do denoising with pybest (moved other modules down)
  - 21/10/2020: added module to project line to surface with registration cascade
  - 30/09/2020: added the module to determine the best vertex with pycortex as module 14
  - 23/09/2020: removed bias field correction for UNIT1 module, as we'll do that with fMRIprep
                leaving out freesurfer segmentation from the SEG-module. Included in fMRIprep
