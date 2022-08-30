"""
Orchestrating the BOLD-preprocessing workflow
"""
from fmriprep import config
import os
import nibabel as nb
from nipype.interfaces.fsl import Split as FSLSplit
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from niworkflows.utils.connections import pop_file, listify
from fmriprep.interfaces import DerivativesDataSink
from fmriprep.interfaces.reports import FunctionalSummary

# BOLD workflows
from fmriprep.workflows.bold.confounds import init_carpetplot_wf
from fmriprep.workflows.bold.hmc import init_bold_hmc_wf
from fmriprep.workflows.bold.registration import init_bold_t1_trans_wf, init_bold_reg_wf
from fmriprep.workflows.bold.resampling import (
    init_bold_std_trans_wf,
    init_bold_preproc_trans_wf,
)
# from fmriprep.workflows.bold.outputs import init_func_derivatives_wf
from bids import BIDSLayout
import numpy as np
from fmriprep.config import DEFAULT_MEMORY_MIN_GB
import os

class init_single_subject_wf():

    def __init__(
        self, 
        subject_id, 
        fmriprep_dir=None, 
        bids_dir=None, 
        workdir=None, 
        bids_filters=None, 
        non_standard=['func'], 
        omp_nthreads=8, 
        max_topup_vols=5):

        from niworkflows.engine.workflows import LiterateWorkflow as Workflow
        from niworkflows.utils.bids import collect_data
        import json
        from sdcflows import fieldmaps as fm
        from sdcflows.utils.wrangler import find_estimators
        from sdcflows.workflows.base import init_fmap_preproc_wf

        self.subject_id     = subject_id
        self.fmriprep_dir   = fmriprep_dir
        self.bids_dir       = bids_dir
        self.bids_filters   = bids_filters
        self.non_standard   = non_standard
        self.omp_nthreads   = omp_nthreads
        self.workdir        = workdir
        self.max_topup_vols = max_topup_vols

        # get bids layout
        self.layout = BIDSLayout(self.bids_dir, validate=False)

        if isinstance(self.bids_filters, str):
            if len(self.bids_filters) != 0:
                with open(self.bids_filters) as f:
                    self.bids_filters = json.load(f)

        self.name = "single_subject_%s_wf" % self.subject_id
        self.subject_data = collect_data(self.layout,
                                         self.subject_id,
                                         bids_filters=self.bids_filters)[0]

        self.workflow = Workflow(name=self.name)
        
        if self.workdir:
            self.workflow.base_dir = self.workdir

        # deal with fmaps
        self.fmap_estimators = None

        # SDC Step 1: Run basic heuristics to identify available data for fieldmap estimation
        # For now, no fmapless
        self.fmap_estimators = find_estimators(
            layout=self.layout,
            subject=self.subject_id,
            fmapless=False,
            force_fmapless=False,
        )

        if self.fmap_estimators:
            config.loggers.workflow.info(
                "B0 field inhomogeneity map will be estimated with "
                f" the following {len(self.fmap_estimators)} estimators: "
                f"{[e.method for e in self.fmap_estimators]}."
            )

        # initiate func workflows
        self.func_preproc_wfs = []
        self.has_fieldmap = bool(self.fmap_estimators)

        for bold_file in self.subject_data['bold']:
            print(bold_file)
            func_preproc_wf = init_func_preproc_wf(
                bold_file, 
                has_fieldmap=self.has_fieldmap, 
                fmriprep_dir=self.fmriprep_dir, 
                layout=self.layout,
                non_standard=self.non_standard,
                omp_nthreads=self.omp_nthreads)
            if func_preproc_wf is None:
                continue

            self.func_preproc_wfs.append(func_preproc_wf)

        self.fmap_wf = init_fmap_preproc_wf(
            debug=False,
            estimators=self.fmap_estimators,
            omp_nthreads=self.omp_nthreads,
            output_dir=self.fmriprep_dir,
            subject=self.subject_id,
        )


        for func_preproc_wf in self.func_preproc_wfs:
            # fmt: off
            self.workflow.connect([
                (self.fmap_wf, func_preproc_wf, [
                    ("outputnode.fmap", "inputnode.fmap"),
                    ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                    ("outputnode.fmap_coeff", "inputnode.fmap_coeff"),
                    ("outputnode.fmap_mask", "inputnode.fmap_mask"),
                    ("outputnode.fmap_id", "inputnode.fmap_id"),
                    ("outputnode.method", "inputnode.sdc_method"),
                ]),
            ])
            # fmt: on

        # Overwrite ``out_path_base`` of sdcflows's DataSinks
        for node in self.fmap_wf.list_node_names():
            if node.split(".")[-1].startswith("ds_"):
                self.fmap_wf.get_node(node).interface.out_path_base = ""  

        for estimator in self.fmap_estimators:
            config.loggers.workflow.info(f"""\
    Setting-up fieldmap "{estimator.bids_id}" ({estimator.method}) with \
    <{', '.join(s.path.name for s in estimator.sources)}>""")

            # Mapped and phasediff can be connected internally by SDCFlows
            if estimator.method in (fm.EstimatorType.MAPPED, fm.EstimatorType.PHASEDIFF):
                continue

            suffices = [s.suffix for s in estimator.sources]

            if estimator.method == fm.EstimatorType.PEPOLAR:
                if set(suffices) == {"epi"} or sorted(suffices) == ["bold", "epi"]:
                    wf_inputs = getattr(self.fmap_wf.inputs, f"in_{estimator.bids_id}")
                    wf_inputs.in_data = [str(s.path) for s in estimator.sources]
                    wf_inputs.metadata = [s.metadata for s in estimator.sources]

                    # 21.0.x hack to change the number of volumes used
                    # The default of 50 takes excessively long
                    flatten = self.fmap_wf.get_node(f"wf_{estimator.bids_id}.flatten")
                    flatten.inputs.max_trs = self.max_topup_vols
                else:
                    raise NotImplementedError(
                        "Sophisticated PEPOLAR schemes are unsupported."
                    )

    def run(self):

        config.loggers.workflow.log(25, "fMRIPrep (McFlirt+Topup only) started!")
        try:
            self.workflow.run()
        except Exception as e:
            config.loggers.workflow.critical("fMRIPrep failed: %s", e)
            raise
        else:
            config.loggers.workflow.log(25, "fMRIPrep (McFlirt+Topup only) finished successfully!")

class bold_reg_wf():

    def __init__(
        self, 
        subject_id,
        boldref, 
        workdir=None,
        omp_nthreads=8,
        use_bbr=True,
        bold2t1w_dof=6,
        bold2t1w_init='header'):

        self.subject_id     = subject_id
        self.boldref        = boldref
        self.bold2t1w_init  = bold2t1w_init
        self.bold2t1w_dof   = bold2t1w_dof
        self.omp_nthreads   = omp_nthreads
        self.workdir        = workdir
        self.use_bbr        = use_bbr

        if os.path.isfile(self.boldref):
            self.bold_tlen, self.mem_gb = _create_mem_gb(self.boldref)

        # calculate BOLD registration to T1w
        self.bold_reg_wf = init_bold_reg_wf(
            bold2t1w_dof=self.bold2t1w_dof,
            bold2t1w_init=self.bold2t1w_init,
            freesurfer=True,
            mem_gb=self.mem_gb['filesize'],
            name="bold_reg_wf",
            omp_nthreads=self.omp_nthreads,
            sloppy=False,
            use_bbr=self.use_bbr,
            use_compression=False,
            write_report=False # will crash if True; missing 'source_file' on lta_ras2ras node
        )            

        try:
            self.prefix = os.environ.get("PREFIX")
        except:
            self.prefix = "sub-"

        self.bold_reg_wf.inputs.inputnode.ref_bold_brain = self.boldref
        self.bold_reg_wf.inputs.inputnode.subject_id = f"{self.prefix}{self.subject_id}"
        self.bold_reg_wf.inputs.inputnode.subjects_dir = os.environ.get("SUBJECTS_DIR")
        self.bold_reg_wf.base_dir = self.workdir

    def run(self):
        config.loggers.workflow.log(25, "fMRIPrep (bold_reg_wf only) started!")
        try:
            self.bold_reg_wf.run()
        except Exception as e:
            config.loggers.workflow.critical("fMRIPrep failed: %s", e)
            raise
        else:
            config.loggers.workflow.log(25, "fMRIPrep (bold_reg_wf only) finished successfully!")

def init_func_preproc_wf(bold_file, has_fieldmap=False, fmriprep_dir=None, layout=None, non_standard=['func'], omp_nthreads=8):
    """
    This workflow controls the functional preprocessing stages of *fMRIPrep*.
    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.func.util import init_bold_reference_wf
    from niworkflows.interfaces.nibabel import ApplyMask
    from niworkflows.interfaces.utility import KeySelect, DictMerge
    from niworkflows.interfaces.reportlets.registration import (
        SimpleBeforeAfterRPT as SimpleBeforeAfter,
    )
    from niworkflows.utils.spaces import SpatialReferences
    spaces = SpatialReferences(non_standard)

    if nb.load(bold_file[0] if isinstance(bold_file, (list, tuple)) else bold_file
    ).shape[3:] <= (5 - False,):
        config.loggers.workflow.warning(
            f"Too short BOLD series (<= 5 timepoints). Skipping processing of <{bold_file}>."
        )
        return

    mem_gb = {"filesize": 1, "resampled": 1, "largemem": 1}
    bold_tlen = 10

    # Have some options handy
    freesurfer = True # config.workflow.run_reconall

    # Extract BIDS entities and metadata from BOLD file(s)
    entities = extract_entities(bold_file)

    # Extract metadata
    all_metadata = [layout.get_metadata(fname) for fname in listify(bold_file)]

    # Take first file as reference
    ref_file = pop_file(bold_file)
    metadata = all_metadata[0]

    # get original image orientation
    ref_orientation = get_img_orientation(ref_file)

    echo_idxs = listify(entities.get("echo", []))
    multiecho = len(echo_idxs) > 2
    if len(echo_idxs) == 1:
        config.loggers.workflow.warning(
            f"Running a single echo <{ref_file}> from a seemingly multi-echo dataset."
        )
        bold_file = ref_file  # Just in case - drop the list

    if len(echo_idxs) == 2:
        raise RuntimeError(
            "Multi-echo processing requires at least three different echos (found two)."
        )

    if multiecho:
        # Drop echo entity for future queries, have a boolean shorthand
        entities.pop("echo", None)
        # reorder echoes from shortest to largest
        tes, bold_file = zip(
            *sorted([(layout.get_metadata(bf)["EchoTime"], bf) for bf in bold_file])
        )
        ref_file = bold_file[0]  # Reset reference to be the shortest TE

    if os.path.isfile(ref_file):
        bold_tlen, mem_gb = _create_mem_gb(ref_file)

    wf_name = _get_wf_name(ref_file)
    config.loggers.workflow.debug(
        "Creating bold processing workflow for <%s> (%.2f GB / %d TRs). "
        "Memory resampled/largemem=%.2f/%.2f GB.",
        ref_file,
        mem_gb["filesize"],
        bold_tlen,
        mem_gb["resampled"],
        mem_gb["largemem"],
    )

    # Find associated sbref, if possible
    entities["suffix"] = "sbref"
    entities["extension"] = [".nii", ".nii.gz"]  # Overwrite extensions
    sbref_files = layout.get(return_type="file", **entities)

    # sbref_msg = f"No single-band-reference found for {os.path.basename(ref_file)}."
    # if sbref_files and "sbref" in config.workflow.ignore:
    #     sbref_msg = "Single-band reference file(s) found and ignored."
    #     sbref_files = []
    # elif sbref_files:
    #     sbref_msg = "Using single-band reference file(s) {}.".format(
    #         ",".join([os.path.basename(sbf) for sbf in sbref_files])
    #     )
    # config.loggers.workflow.info(sbref_msg)

    if has_fieldmap:
        # First check if specified via B0FieldSource
        estimator_key = listify(metadata.get("B0FieldSource"))

        if not estimator_key:
            from pathlib import Path
            import re
            from sdcflows.fieldmaps import get_identifier

            # Fallback to IntendedFor
            intended_rel = re.sub(
                r"^sub-[a-zA-Z0-9]*/",
                "",
                str(Path(
                    bold_file if not multiecho else bold_file[0]
                ).relative_to(layout.root))
            )
            estimator_key = get_identifier(intended_rel)

        if not estimator_key:
            has_fieldmap = False
            config.loggers.workflow.critical(
                f"None of the available B0 fieldmaps are associated to <{bold_file}>"
            )
        else:
            config.loggers.workflow.info(
                f"Found usable B0-map (fieldmap) estimator(s) <{', '.join(estimator_key)}> "
                f"to correct <{bold_file}> for susceptibility-derived distortions.")

    # Check whether STC must/can be run
    run_stc = False #(
    #     bool(metadata.get("SliceTiming"))
    #     and "slicetiming" not in config.workflow.ignore
    # )

    # Build workflow
    workflow = Workflow(name=wf_name)
    workflow.__postdesc__ = """\
All resamplings can be performed with *a single interpolation
step* by composing all the pertinent transformations (i.e. head-motion
transform matrices, susceptibility distortion correction when available,
and co-registrations to anatomical and output spaces).
Gridded (volumetric) resamplings were performed using `antsApplyTransforms` (ANTs),
configured with Lanczos interpolation to minimize the smoothing
effects of other kernels [@lanczos].
Non-gridded (surface) resamplings were performed using `mri_vol2surf`
(FreeSurfer).
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold_file",
                "subjects_dir",
                "subject_id",
                "t1w_preproc",
                "t1w_mask",
                "t1w_dseg",
                "t1w_tpms",
                "t1w_aseg",
                "t1w_aparc",
                "anat2std_xfm",
                "std2anat_xfm",
                "template",
                "t1w2fsnative_xfm",
                "fsnative2t1w_xfm",
                "fmap",
                "fmap_ref",
                "fmap_coeff",
                "fmap_mask",
                "fmap_id",
                "sdc_method",
            ]
        ),
        name="inputnode",
    )
    inputnode.inputs.bold_file = bold_file

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold_t1",
                "bold_t1_ref",
                "bold2anat_xfm",
                "anat2bold_xfm",
                "bold_mask_t1",
                "bold_aseg_t1",
                "bold_aparc_t1",
                "bold_std",
                "bold_std_ref",
                "bold_mask_std",
                "bold_aseg_std",
                "bold_aparc_std",
                "bold_native",
                "bold_native_ref",
                "bold_mask_native",
                "bold_echos_native",
                "bold_cifti",
                "cifti_variant",
                "cifti_metadata",
                "cifti_density",
                "surfaces",
                "t2star_bold",
                "t2star_t1",
                "t2star_std",
                # "confounds",
                "aroma_noise_ics",
                "melodic_mix",
                "nonaggr_denoised_file",
                # "confounds_metadata",
            ]
        ),
        name="outputnode",
    )

    # Generate a brain-masked conversion of the t1w
    t1w_brain = pe.Node(ApplyMask(), name="t1w_brain")

    # Track echo index - this allows us to treat multi- and single-echo workflows
    # almost identically
    echo_index = pe.Node(niu.IdentityInterface(fields=["echoidx"]), name="echo_index")
    if multiecho:
        echo_index.iterables = [("echoidx", range(len(bold_file)))]
    else:
        echo_index.inputs.echoidx = 0

    # BOLD source: track original BOLD file(s)
    bold_source = pe.Node(niu.Select(inlist=bold_file), name="bold_source")

    # BOLD buffer: an identity used as a pointer to either the original BOLD
    # or the STC'ed one for further use.
    boldbuffer = pe.Node(niu.IdentityInterface(fields=["bold_file"]), name="boldbuffer")

    summary = pe.Node(
        FunctionalSummary(
            slice_timing=run_stc,
            registration=("FSL", "FreeSurfer")[freesurfer],
            registration_dof=6,
            registration_init=os.environ.get("BOLD_T1W_INIT"),
            pe_direction=metadata.get("PhaseEncodingDirection"),
            echo_idx=echo_idxs,
            tr=metadata["RepetitionTime"],
            orientation=ref_orientation,
        ),
        name="summary",
        mem_gb=mem_gb['filesize'],
        run_without_submitting=True,
    )
    summary.inputs.dummy_scans = 0

    func_derivatives_wf = init_func_derivatives_wf(
        bids_root=layout.root,
        cifti_output=False,
        freesurfer=freesurfer,
        all_metadata=all_metadata,
        multiecho=multiecho,
        output_dir=fmriprep_dir,
        spaces=spaces,
        use_aroma=False,
    )
    func_derivatives_wf.inputs.inputnode.all_source_files = bold_file

    # fmt:off
    workflow.connect([
        (outputnode, func_derivatives_wf, [
            ("bold_t1", "inputnode.bold_t1"),
            ("bold_t1_ref", "inputnode.bold_t1_ref"),
            ("bold2anat_xfm", "inputnode.bold2anat_xfm"),
            ("anat2bold_xfm", "inputnode.anat2bold_xfm"),
            ("bold_aseg_t1", "inputnode.bold_aseg_t1"),
            ("bold_aparc_t1", "inputnode.bold_aparc_t1"),
            ("bold_mask_t1", "inputnode.bold_mask_t1"),
            ("bold_native", "inputnode.bold_native"),
            ("bold_native_ref", "inputnode.bold_native_ref"),
            ("bold_mask_native", "inputnode.bold_mask_native"),
            ("bold_echos_native", "inputnode.bold_echos_native"),
            # ("confounds", "inputnode.confounds"),
            ("surfaces", "inputnode.surf_files"),
            ("aroma_noise_ics", "inputnode.aroma_noise_ics"),
            ("melodic_mix", "inputnode.melodic_mix"),
            ("nonaggr_denoised_file", "inputnode.nonaggr_denoised_file"),
            ("bold_cifti", "inputnode.bold_cifti"),
            ("cifti_variant", "inputnode.cifti_variant"),
            ("cifti_metadata", "inputnode.cifti_metadata"),
            ("cifti_density", "inputnode.cifti_density"),
            # ("t2star_bold", "inputnode.t2star_bold"),
            # ("t2star_t1", "inputnode.t2star_t1"),
            # ("t2star_std", "inputnode.t2star_std"),
            # ("confounds_metadata", "inputnode.confounds_metadata"),
            # ("acompcor_masks", "inputnode.acompcor_masks"),
            # ("tcompcor_mask", "inputnode.tcompcor_mask"),
        ]),
    ])
    # fmt:on

    # Generate a tentative boldref
    initial_boldref_wf = init_bold_reference_wf(
        name="initial_boldref_wf",
        omp_nthreads=omp_nthreads,
        bold_file=bold_file,
        sbref_files=sbref_files,
        multiecho=multiecho,
    )
    initial_boldref_wf.inputs.inputnode.dummy_scans = 0

    # Select validated BOLD files (orientations checked or corrected)
    select_bold = pe.Node(niu.Select(), name="select_bold")

    # Top-level BOLD splitter
    bold_split = pe.Node(
        FSLSplit(dimension="t"), name="bold_split", mem_gb=mem_gb["filesize"] * 3
    )

    # HMC on the BOLD
    bold_hmc_wf = init_bold_hmc_wf(
        name="bold_hmc_wf", mem_gb=mem_gb["filesize"], omp_nthreads=omp_nthreads
    )

    # calculate BOLD registration to T1w
    bold_reg_wf = init_bold_reg_wf(
        bold2t1w_dof=6,
        bold2t1w_init=os.environ.get("BOLD_T1W_INIT"),
        freesurfer=freesurfer,
        mem_gb=mem_gb["resampled"],
        name="bold_reg_wf",
        omp_nthreads=omp_nthreads,
        sloppy=False,
        use_bbr=freesurfer,
        use_compression=False,
    )

    # apply BOLD registration to T1w
    bold_t1_trans_wf = init_bold_t1_trans_wf(
        name="bold_t1_trans_wf",
        freesurfer=freesurfer,
        mem_gb=mem_gb["resampled"],
        omp_nthreads=omp_nthreads,
        use_compression=False,
    )
    bold_t1_trans_wf.inputs.inputnode.fieldwarp = "identity"

    # # get confounds
    # bold_confounds_wf = init_bold_confs_wf(
    #     mem_gb=mem_gb["largemem"],
    #     metadata=metadata,
    #     freesurfer=freesurfer,
    #     regressors_all_comps=config.workflow.regressors_all_comps,
    #     regressors_fd_th=config.workflow.regressors_fd_th,
    #     regressors_dvars_th=config.workflow.regressors_dvars_th,
    #     name="bold_confounds_wf",
    # )
    # bold_confounds_wf.get_node("inputnode").inputs.t1_transform_flags = [False]

    # # SLICE-TIME CORRECTION (or bypass) #############################################
    # if run_stc:
    #     bold_stc_wf = init_bold_stc_wf(name="bold_stc_wf", metadata=metadata)
    #     # fmt:off
    #     workflow.connect([
    #         (initial_boldref_wf, bold_stc_wf, [("outputnode.skip_vols", "inputnode.skip_vols")]),
    #         (select_bold, bold_stc_wf, [("out", "inputnode.bold_file")]),
    #         (bold_stc_wf, boldbuffer, [("outputnode.stc_file", "bold_file")]),
    #     ])
    #     # fmt:on

    # # bypass STC from original BOLD in both SE and ME cases
    # else:
    workflow.connect([(select_bold, boldbuffer, [("out", "bold_file")])])

    # # MULTI-ECHO EPI DATA #############################################
    # if multiecho:  # instantiate relevant interfaces, imports
    #     split_opt_comb = bold_split.clone(name="split_opt_comb")

    #     inputnode.inputs.bold_file = ref_file  # Replace reference w first echo

    #     join_echos = pe.JoinNode(
    #         niu.IdentityInterface(fields=["bold_files"]),
    #         joinsource="echo_index",
    #         joinfield=["bold_files"],
    #         name="join_echos",
    #     )

    #     # create optimal combination, adaptive T2* map
    #     bold_t2s_wf = init_bold_t2s_wf(
    #         echo_times=tes,
    #         mem_gb=mem_gb["resampled"],
    #         omp_nthreads=omp_nthreads,
    #         name="bold_t2smap_wf",
    #     )

    #     t2s_reporting_wf = init_t2s_reporting_wf()

    #     ds_report_t2scomp = pe.Node(
    #         DerivativesDataSink(
    #             desc="t2scomp",
    #             datatype="figures",
    #             dismiss_entities=("echo",),
    #         ),
    #         name="ds_report_t2scomp",
    #         run_without_submitting=True,
    #     )

    #     ds_report_t2star_hist = pe.Node(
    #         DerivativesDataSink(
    #             desc="t2starhist",
    #             datatype="figures",
    #             dismiss_entities=("echo",),
    #         ),
    #         name="ds_report_t2star_hist",
    #         run_without_submitting=True,
    #     )

    bold_final = pe.Node(
        niu.IdentityInterface(fields=["bold", "boldref", "mask", "bold_echos", "t2star"]),
        name="bold_final"
    )

    # Generate a final BOLD reference
    # This BOLD references *does not use* single-band reference images.
    final_boldref_wf = init_bold_reference_wf(
        name="final_boldref_wf",
        omp_nthreads=omp_nthreads,
        multiecho=multiecho,
    )
    final_boldref_wf.__desc__ = None  # Unset description to avoid second appearance

    # MAIN WORKFLOW STRUCTURE #######################################################
    # fmt:off
    workflow.connect([
        # # Prepare masked T1w image
        # (inputnode, t1w_brain, [("t1w_preproc", "in_file"),
        #                         ("t1w_mask", "in_mask")]),
        # Select validated bold files per-echo
        (initial_boldref_wf, select_bold, [("outputnode.all_bold_files", "inlist")]),
        # BOLD buffer has slice-time corrected if it was run, original otherwise
        (boldbuffer, bold_split, [("bold_file", "in_file")]),
        # HMC
        (initial_boldref_wf, bold_hmc_wf, [
            ("outputnode.raw_ref_image", "inputnode.raw_ref_image"),
            ("outputnode.bold_file", "inputnode.bold_file"),
        ]),
        # # EPI-T1w registration workflow
        # (inputnode, bold_reg_wf, [
        #     ("t1w_dseg", "inputnode.t1w_dseg"),
        #     # Undefined if --fs-no-reconall, but this is safe
        #     ("subjects_dir", "inputnode.subjects_dir"),
        #     ("subject_id", "inputnode.subject_id"),
        #     ("fsnative2t1w_xfm", "inputnode.fsnative2t1w_xfm"),
        # ]),
        # (bold_final, bold_reg_wf, [
        #     ("boldref", "inputnode.ref_bold_brain")]),
        # (t1w_brain, bold_reg_wf, [("out_file", "inputnode.t1w_brain")]),
        # (inputnode, bold_t1_trans_wf, [
        #     ("bold_file", "inputnode.name_source"),
        #     ("t1w_mask", "inputnode.t1w_mask"),
        #     ("t1w_aseg", "inputnode.t1w_aseg"),
        #     ("t1w_aparc", "inputnode.t1w_aparc"),
        # ]),
        # (t1w_brain, bold_t1_trans_wf, [("out_file", "inputnode.t1w_brain")]),
        # (bold_reg_wf, outputnode, [
        #     ("outputnode.itk_bold_to_t1", "bold2anat_xfm"),
        #     ("outputnode.itk_t1_to_bold", "anat2bold_xfm"),
        # ]),
        # (bold_reg_wf, bold_t1_trans_wf, [
        #     ("outputnode.itk_bold_to_t1", "inputnode.itk_bold_to_t1"),
        # ]),
        # (bold_final, bold_t1_trans_wf, [
        #     ("mask", "inputnode.ref_bold_mask"),
        #     ("boldref", "inputnode.ref_bold_brain"),
        # ]),
        # (bold_t1_trans_wf, outputnode, [
        #     ("outputnode.bold_t1", "bold_t1"),
        #     ("outputnode.bold_t1_ref", "bold_t1_ref"),
        #     ("outputnode.bold_aseg_t1", "bold_aseg_t1"),
        #     ("outputnode.bold_aparc_t1", "bold_aparc_t1"),
        # ]),
        # # Connect bold_confounds_wf
        # (inputnode, bold_confounds_wf, [
        #     ("t1w_tpms", "inputnode.t1w_tpms"),
        #     ("t1w_mask", "inputnode.t1w_mask"),
        # ]),
        # (bold_hmc_wf, bold_confounds_wf, [
        #     ("outputnode.movpar_file", "inputnode.movpar_file"),
        #     ("outputnode.rmsd_file", "inputnode.rmsd_file"),
        # ]),
        # (bold_reg_wf, bold_confounds_wf, [
        #     ("outputnode.itk_t1_to_bold", "inputnode.t1_bold_xform")
        # ]),
        # (initial_boldref_wf, bold_confounds_wf, [
        #     ("outputnode.skip_vols", "inputnode.skip_vols"),
        # ]),
        (initial_boldref_wf, final_boldref_wf, [
            ("outputnode.skip_vols", "inputnode.dummy_scans"),
        ]),
        (final_boldref_wf, bold_final, [
            ("outputnode.ref_image", "boldref"),
            ("outputnode.bold_mask", "mask"),
        ]),
        # (bold_final, bold_confounds_wf, [
        #     ("bold", "inputnode.bold"),
        #     ("mask", "inputnode.bold_mask"),
        # ]),
        # (bold_confounds_wf, outputnode, [
        #     ("outputnode.confounds_file", "confounds"),
        #     ("outputnode.confounds_metadata", "confounds_metadata"),
        #     ("outputnode.acompcor_masks", "acompcor_masks"),
        #     ("outputnode.tcompcor_mask", "tcompcor_mask"),
        # ]),
        # Native-space BOLD files (if calculated)
        (bold_final, outputnode, [
            ("bold", "bold_native"),
            ("boldref", "bold_native_ref"),
            ("mask", "bold_mask_native"),
            ("bold_echos", "bold_echos_native"),
            ("t2star", "t2star_bold"),
        ]),
        # Summary
        # (initial_boldref_wf, summary, [("outputnode.algo_dummy_scans", "algo_dummy_scans")]),
        # (bold_reg_wf, summary, [("outputnode.fallback", "fallback")]),
        # (outputnode, summary, [("confounds", "confounds_file")]),
        # Select echo indices for original/validated BOLD files
        (echo_index, bold_source, [("echoidx", "index")]),
        (echo_index, select_bold, [("echoidx", "index")]),
    ])
    # fmt:on

    # for standard EPI data, pass along correct file
    if not multiecho:
        # fmt:off
        workflow.connect([
            (inputnode, func_derivatives_wf, [("bold_file", "inputnode.source_file")]),
            # (bold_split, bold_t1_trans_wf, [("out_files", "inputnode.bold_split")]),
            # (bold_hmc_wf, bold_t1_trans_wf, [("outputnode.xforms", "inputnode.hmc_xforms")]),
        ])
        # fmt:on
    # else:  # for meepi, use optimal combination
    #     # fmt:off
    #     workflow.connect([
    #         # update name source for optimal combination
    #         (inputnode, func_derivatives_wf, [
    #             (("bold_file", combine_meepi_source), "inputnode.source_file"),
    #         ]),
    #         (join_echos, bold_t2s_wf, [("bold_files", "inputnode.bold_file")]),
    #         (join_echos, bold_final, [("bold_files", "bold_echos")]),
    #         (bold_t2s_wf, split_opt_comb, [("outputnode.bold", "in_file")]),
    #         (split_opt_comb, bold_t1_trans_wf, [("out_files", "inputnode.bold_split")]),
    #         (bold_t2s_wf, bold_final, [("outputnode.bold", "bold"),
    #                                    ("outputnode.t2star_map", "t2star")]),
    #         (inputnode, t2s_reporting_wf, [("t1w_dseg", "inputnode.label_file")]),
    #         (bold_reg_wf, t2s_reporting_wf, [
    #             ("outputnode.itk_t1_to_bold", "inputnode.label_bold_xform")
    #         ]),
    #         (bold_final, t2s_reporting_wf, [("t2star", "inputnode.t2star_file"),
    #                                         ("boldref", "inputnode.boldref")]),
    #         (t2s_reporting_wf, ds_report_t2scomp, [('outputnode.t2s_comp_report', 'in_file')]),
    #         (t2s_reporting_wf, ds_report_t2star_hist, [("outputnode.t2star_hist", "in_file")]),
    #     ])
    #     # fmt:on

    #     # Already applied in bold_bold_trans_wf, which inputs to bold_t2s_wf
    #     bold_t1_trans_wf.inputs.inputnode.hmc_xforms = "identity"

    # Map final BOLD mask into T1w space (if required)

    nonstd_spaces = set(spaces.get_nonstandard())
    if nonstd_spaces.intersection(("T1w", "anat")):
        from niworkflows.interfaces.fixes import (
            FixHeaderApplyTransforms as ApplyTransforms,
        )

        boldmask_to_t1w = pe.Node(
            ApplyTransforms(interpolation="MultiLabel"),
            name="boldmask_to_t1w",
            mem_gb=0.1,
        )
        # fmt:off
        workflow.connect([
            (bold_reg_wf, boldmask_to_t1w, [("outputnode.itk_bold_to_t1", "transforms")]),
            (bold_t1_trans_wf, boldmask_to_t1w, [("outputnode.bold_mask_t1", "reference_image")]),
            (bold_final, boldmask_to_t1w, [("mask", "input_image")]),
            (boldmask_to_t1w, outputnode, [("output_image", "bold_mask_t1")]),
        ])
        # fmt:on

        # if multiecho:
        #     t2star_to_t1w = pe.Node(
        #         ApplyTransforms(interpolation="LanczosWindowedSinc", float=True),
        #         name="t2star_to_t1w",
        #         mem_gb=0.1,
        #     )
        #     # fmt:off
        #     workflow.connect([
        #         (bold_reg_wf, t2star_to_t1w, [("outputnode.itk_bold_to_t1", "transforms")]),
        #         (bold_t1_trans_wf, t2star_to_t1w, [
        #             ("outputnode.bold_mask_t1", "reference_image")
        #         ]),
        #         (bold_final, t2star_to_t1w, [("t2star", "input_image")]),
        #         (t2star_to_t1w, outputnode, [("output_image", "t2star_t1")]),
        #     ])
        #     # fmt:on

    if spaces.get_spaces(nonstandard=False, dim=(3,)):
        # Apply transforms in 1 shot
        # Only use uncompressed output if AROMA is to be run
        bold_std_trans_wf = init_bold_std_trans_wf(
            freesurfer=freesurfer,
            mem_gb=mem_gb["resampled"],
            omp_nthreads=omp_nthreads,
            spaces=spaces,
            multiecho=multiecho,
            name="bold_std_trans_wf",
            use_compression=False,
        )
        bold_std_trans_wf.inputs.inputnode.fieldwarp = "identity"

        # fmt:off
        workflow.connect([
            (inputnode, bold_std_trans_wf, [
                ("template", "inputnode.templates"),
                ("anat2std_xfm", "inputnode.anat2std_xfm"),
                ("bold_file", "inputnode.name_source"),
                ("t1w_aseg", "inputnode.bold_aseg"),
                ("t1w_aparc", "inputnode.bold_aparc"),
            ]),
            (bold_final, bold_std_trans_wf, [
                ("mask", "inputnode.bold_mask"),
                ("t2star", "inputnode.t2star"),
            ]),
            (bold_reg_wf, bold_std_trans_wf, [
                ("outputnode.itk_bold_to_t1", "inputnode.itk_bold_to_t1"),
            ]),
            (bold_std_trans_wf, outputnode, [
                ("outputnode.bold_std", "bold_std"),
                ("outputnode.bold_std_ref", "bold_std_ref"),
                ("outputnode.bold_mask_std", "bold_mask_std"),
            ]),
        ])
        # fmt:on

        # if freesurfer:
        #     # fmt:off
        #     workflow.connect([
        #         (bold_std_trans_wf, func_derivatives_wf, [
        #             ("outputnode.bold_aseg_std", "inputnode.bold_aseg_std"),
        #             ("outputnode.bold_aparc_std", "inputnode.bold_aparc_std"),
        #         ]),
        #         (bold_std_trans_wf, outputnode, [
        #             ("outputnode.bold_aseg_std", "bold_aseg_std"),
        #             ("outputnode.bold_aparc_std", "bold_aparc_std"),
        #         ]),
        #     ])
        #     # fmt:on

        # if not multiecho:
        #     # fmt:off
        #     workflow.connect([
        #         (bold_split, bold_std_trans_wf, [("out_files", "inputnode.bold_split")]),
        #         (bold_hmc_wf, bold_std_trans_wf, [
        #             ("outputnode.xforms", "inputnode.hmc_xforms"),
        #         ]),
        #     ])
            # fmt:on
        # else:
        #     # fmt:off
        #     workflow.connect([
        #         (split_opt_comb, bold_std_trans_wf, [("out_files", "inputnode.bold_split")]),
        #         (bold_std_trans_wf, outputnode, [("outputnode.t2star_std", "t2star_std")]),
        #     ])
        #     # fmt:on

        #     # Already applied in bold_bold_trans_wf, which inputs to bold_t2s_wf
        #     bold_std_trans_wf.inputs.inputnode.hmc_xforms = "identity"

        # fmt:off
        # func_derivatives_wf internally parametrizes over snapshotted spaces.
        # workflow.connect([
        #     (bold_std_trans_wf, func_derivatives_wf, [
        #         ("outputnode.template", "inputnode.template"),
        #         ("outputnode.spatial_reference", "inputnode.spatial_reference"),
        #         ("outputnode.bold_std_ref", "inputnode.bold_std_ref"),
        #         ("outputnode.bold_std", "inputnode.bold_std"),
        #         ("outputnode.bold_mask_std", "inputnode.bold_mask_std"),
        #     ]),
        # ])
        # fmt:on

    #     if config.workflow.use_aroma:  # ICA-AROMA workflow
    #         from fmriprep.workflows.bold.confounds import init_ica_aroma_wf

    #         ica_aroma_wf = init_ica_aroma_wf(
    #             mem_gb=mem_gb["resampled"],
    #             metadata=metadata,
    #             omp_nthreads=omp_nthreads,
    #             err_on_aroma_warn=config.workflow.aroma_err_on_warn,
    #             aroma_melodic_dim=config.workflow.aroma_melodic_dim,
    #             name="ica_aroma_wf",
    #         )

    #         join = pe.Node(
    #             niu.Function(output_names=["out_file"], function=_to_join),
    #             name="aroma_confounds",
    #         )

    #         mrg_conf_metadata = pe.Node(
    #             niu.Merge(2),
    #             name="merge_confound_metadata",
    #             run_without_submitting=True,
    #         )
    #         mrg_conf_metadata2 = pe.Node(
    #             DictMerge(),
    #             name="merge_confound_metadata2",
    #             run_without_submitting=True,
    #         )
    #         # fmt:off
    #         workflow.disconnect([
    #             (bold_confounds_wf, outputnode, [
    #                 ("outputnode.confounds_file", "confounds"),
    #             ]),
    #             (bold_confounds_wf, outputnode, [
    #                 ("outputnode.confounds_metadata", "confounds_metadata"),
    #             ]),
    #         ])
    #         workflow.connect([
    #             (inputnode, ica_aroma_wf, [("bold_file", "inputnode.name_source")]),
    #             (bold_hmc_wf, ica_aroma_wf, [
    #                 ("outputnode.movpar_file", "inputnode.movpar_file"),
    #             ]),
    #             (initial_boldref_wf, ica_aroma_wf, [
    #                 ("outputnode.skip_vols", "inputnode.skip_vols"),
    #             ]),
    #             (bold_confounds_wf, join, [("outputnode.confounds_file", "in_file")]),
    #             (bold_confounds_wf, mrg_conf_metadata, [
    #                 ("outputnode.confounds_metadata", "in1"),
    #             ]),
    #             (ica_aroma_wf, join, [("outputnode.aroma_confounds", "join_file")]),
    #             (ica_aroma_wf, mrg_conf_metadata, [("outputnode.aroma_metadata", "in2")]),
    #             (mrg_conf_metadata, mrg_conf_metadata2, [("out", "in_dicts")]),
    #             (ica_aroma_wf, outputnode, [
    #                 ("outputnode.aroma_noise_ics", "aroma_noise_ics"),
    #                 ("outputnode.melodic_mix", "melodic_mix"),
    #                 ("outputnode.nonaggr_denoised_file", "nonaggr_denoised_file"),
    #             ]),
    #             (join, outputnode, [("out_file", "confounds")]),
    #             (mrg_conf_metadata2, outputnode, [("out_dict", "confounds_metadata")]),
    #             (bold_std_trans_wf, ica_aroma_wf, [
    #                 ("outputnode.bold_std", "inputnode.bold_std"),
    #                 ("outputnode.bold_mask_std", "inputnode.bold_mask_std"),
    #                 ("outputnode.spatial_reference", "inputnode.spatial_reference"),
    #             ]),
    #         ])
    #         # fmt:on

    # # SURFACES ##################################################################################
    # # Freesurfer
    # freesurfer_spaces = spaces.get_fs_spaces()
    # if freesurfer and freesurfer_spaces:
    #     config.loggers.workflow.debug("Creating BOLD surface-sampling workflow.")
    #     bold_surf_wf = init_bold_surf_wf(
    #         mem_gb=mem_gb["resampled"],
    #         surface_spaces=freesurfer_spaces,
    #         medial_surface_nan=config.workflow.medial_surface_nan,
    #         name="bold_surf_wf",
    #     )
    #     # fmt:off
    #     workflow.connect([
    #         (inputnode, bold_surf_wf, [
    #             ("subjects_dir", "inputnode.subjects_dir"),
    #             ("subject_id", "inputnode.subject_id"),
    #             ("t1w2fsnative_xfm", "inputnode.t1w2fsnative_xfm"),
    #         ]),
    #         (bold_t1_trans_wf, bold_surf_wf, [("outputnode.bold_t1", "inputnode.source_file")]),
    #         (bold_surf_wf, outputnode, [("outputnode.surfaces", "surfaces")]),
    #         (bold_surf_wf, func_derivatives_wf, [("outputnode.target", "inputnode.surf_refs")]),
    #     ])
    #     # fmt:on

    #     # CIFTI output
    #     if config.workflow.cifti_output:
    #         from fmriprep.workflows.bold.resampling import init_bold_grayords_wf

    #         bold_grayords_wf = init_bold_grayords_wf(
    #             grayord_density=config.workflow.cifti_output,
    #             mem_gb=mem_gb["resampled"],
    #             repetition_time=metadata["RepetitionTime"],
    #         )

    #         # fmt:off
    #         workflow.connect([
    #             (inputnode, bold_grayords_wf, [("subjects_dir", "inputnode.subjects_dir")]),
    #             (bold_std_trans_wf, bold_grayords_wf, [
    #                 ("outputnode.bold_std", "inputnode.bold_std"),
    #                 ("outputnode.spatial_reference", "inputnode.spatial_reference"),
    #             ]),
    #             (bold_surf_wf, bold_grayords_wf, [
    #                 ("outputnode.surfaces", "inputnode.surf_files"),
    #                 ("outputnode.target", "inputnode.surf_refs"),
    #             ]),
    #             (bold_grayords_wf, outputnode, [
    #                 ("outputnode.cifti_bold", "bold_cifti"),
    #                 ("outputnode.cifti_variant", "cifti_variant"),
    #                 ("outputnode.cifti_metadata", "cifti_metadata"),
    #                 ("outputnode.cifti_density", "cifti_density"),
    #             ]),
    #         ])
    #         # fmt:on

    if spaces.get_spaces(nonstandard=False, dim=(3,)):
        carpetplot_wf = init_carpetplot_wf(
            mem_gb=mem_gb["resampled"],
            metadata=metadata,
            cifti_output=False,
            name="carpetplot_wf",
        )

        # Xform to "MNI152NLin2009cAsym" is always computed.
        carpetplot_select_std = pe.Node(
            KeySelect(fields=["std2anat_xfm"], key="MNI152NLin2009cAsym"),
            name="carpetplot_select_std",
            run_without_submitting=True,
        )

        # if config.workflow.cifti_output:
        #     workflow.connect(
        #         bold_grayords_wf,
        #         "outputnode.cifti_bold",
        #         carpetplot_wf,
        #         "inputnode.cifti_bold",
        #     )

        # fmt:off
        workflow.connect([
            (initial_boldref_wf, carpetplot_wf, [
                ("outputnode.skip_vols", "inputnode.dummy_scans"),
            ]),
            (inputnode, carpetplot_select_std, [("std2anat_xfm", "std2anat_xfm"),
                                                ("template", "keys")]),
            (carpetplot_select_std, carpetplot_wf, [
                ("std2anat_xfm", "inputnode.std2anat_xfm"),
            ]),
            (bold_final, carpetplot_wf, [
                ("bold", "inputnode.bold"),
                ("mask", "inputnode.bold_mask"),
            ]),
            (bold_reg_wf, carpetplot_wf, [
                ("outputnode.itk_t1_to_bold", "inputnode.t1_bold_xform"),
            ]),
            # (bold_confounds_wf, carpetplot_wf, [
            #     ("outputnode.confounds_file", "inputnode.confounds_file"),
            #     ("outputnode.crown_mask", "inputnode.crown_mask")
            # ]),
        ])
        # fmt:on

    # REPORTING ############################################################
    ds_report_summary = pe.Node(
        DerivativesDataSink(
            desc="summary", datatype="figures", dismiss_entities=("echo",)
        ),
        name="ds_report_summary",
        run_without_submitting=True,
        mem_gb=mem_gb['filesize'],
    )

    ds_report_validation = pe.Node(
        DerivativesDataSink(
            desc="validation", datatype="figures", dismiss_entities=("echo",)
        ),
        name="ds_report_validation",
        run_without_submitting=True,
        mem_gb=mem_gb['filesize'],
    )

    # fmt:off
    workflow.connect([
        # (summary, ds_report_summary, [("out_report", "in_file")]),
        (initial_boldref_wf, ds_report_validation, [("outputnode.validation_report", "in_file")]),
    ])
    # fmt:on

    # Fill-in datasinks of reportlets seen so far
    for node in workflow.list_node_names():
        if node.split(".")[-1].startswith("ds_report"):
            workflow.get_node(node).inputs.base_directory = fmriprep_dir
            workflow.get_node(node).inputs.source_file = ref_file

    if not has_fieldmap:
        # Finalize workflow without SDC connections
        summary.inputs.distortion_correction = "None"

        # Resample in native space in just one shot
        bold_bold_trans_wf = init_bold_preproc_trans_wf(
            mem_gb=mem_gb["resampled"],
            omp_nthreads=omp_nthreads,
            use_compression=False,
            use_fieldwarp=False,
            name="bold_bold_trans_wf",
        )
        bold_bold_trans_wf.inputs.inputnode.fieldwarp = "identity"

        # fmt:off
        workflow.connect([
            # Connect bold_bold_trans_wf
            (bold_source, bold_bold_trans_wf, [("out", "inputnode.name_source")]),
            (bold_split, bold_bold_trans_wf, [("out_files", "inputnode.bold_file")]),
            (bold_hmc_wf, bold_bold_trans_wf, [
                ("outputnode.xforms", "inputnode.hmc_xforms"),
            ]),
        ])

        workflow.connect([
            (bold_bold_trans_wf, bold_final, [("outputnode.bold", "bold")]),
            (bold_bold_trans_wf, final_boldref_wf, [
                ("outputnode.bold", "inputnode.bold_file"),
            ])
        ] 
        # if not multiecho else [
        #     (initial_boldref_wf, bold_t2s_wf, [
        #         ("outputnode.bold_mask", "inputnode.bold_mask"),
        #     ]),
        #     (bold_bold_trans_wf, join_echos, [
        #         ("outputnode.bold", "bold_files"),
        #     ]),
        #     (join_echos, final_boldref_wf, [
        #         ("bold_files", "inputnode.bold_file"),
        #     ])]
        )
        # fmt:on
        return workflow

    from niworkflows.interfaces.utility import KeySelect
    from sdcflows.workflows.apply.registration import init_coeff2epi_wf
    from sdcflows.workflows.apply.correction import init_unwarp_wf

    coeff2epi_wf = init_coeff2epi_wf(
        debug=False,
        omp_nthreads=omp_nthreads,
        write_coeff=True,
    )
    unwarp_wf = init_unwarp_wf(
        debug=False,
        omp_nthreads=omp_nthreads,
    )
    unwarp_wf.inputs.inputnode.metadata = metadata

    output_select = pe.Node(
        KeySelect(fields=["fmap", "fmap_ref", "fmap_coeff", "fmap_mask", "sdc_method"]),
        name="output_select",
        run_without_submitting=True,
    )
    output_select.inputs.key = estimator_key[0]
    if len(estimator_key) > 1:
        config.loggers.workflow.warning(
            f"Several fieldmaps <{', '.join(estimator_key)}> are "
            f"'IntendedFor' <{bold_file}>, using {estimator_key[0]}"
        )

    sdc_report = pe.Node(
        SimpleBeforeAfter(
            before_label="Distorted",
            after_label="Corrected",
            dismiss_affine=True,
        ),
        name="sdc_report",
        mem_gb=0.1,
    )

    ds_report_sdc = pe.Node(
        DerivativesDataSink(
            base_directory=fmriprep_dir,
            desc="sdc",
            suffix="bold",
            datatype="figures",
            dismiss_entities=("echo",),
        ),
        name="ds_report_sdc",
        run_without_submitting=True,
    )

    # fmt:off
    workflow.connect([
        (inputnode, output_select, [("fmap", "fmap"),
                                    ("fmap_ref", "fmap_ref"),
                                    ("fmap_coeff", "fmap_coeff"),
                                    ("fmap_mask", "fmap_mask"),
                                    ("sdc_method", "sdc_method"),
                                    ("fmap_id", "keys")]),
        (output_select, coeff2epi_wf, [
            ("fmap_ref", "inputnode.fmap_ref"),
            ("fmap_coeff", "inputnode.fmap_coeff"),
            ("fmap_mask", "inputnode.fmap_mask")]),
        # (output_select, summary, [("sdc_method", "distortion_correction")]),
        (initial_boldref_wf, coeff2epi_wf, [
            ("outputnode.ref_image", "inputnode.target_ref"),
            ("outputnode.bold_mask", "inputnode.target_mask")]),
        (coeff2epi_wf, unwarp_wf, [
            ("outputnode.fmap_coeff", "inputnode.fmap_coeff")]),
        (bold_hmc_wf, unwarp_wf, [
            ("outputnode.xforms", "inputnode.hmc_xforms")]),
        (initial_boldref_wf, sdc_report, [
            ("outputnode.ref_image", "before")]),
        (bold_split, unwarp_wf, [
            ("out_files", "inputnode.distorted")]),
        (final_boldref_wf, sdc_report, [
            ("outputnode.ref_image", "after"),
            ("outputnode.bold_mask", "wm_seg")]),
        (inputnode, ds_report_sdc, [("bold_file", "source_file")]),
        (sdc_report, ds_report_sdc, [("out_report", "in_file")]),

    ])
    # fmt:on

    if not multiecho:
        # fmt:off
        workflow.connect([
            (unwarp_wf, bold_final, [("outputnode.corrected", "bold")]),
            # remaining workflow connections
            (unwarp_wf, final_boldref_wf, [
                ("outputnode.corrected", "inputnode.bold_file"),
            ])
            # (unwarp_wf, bold_t1_trans_wf, [
            #     # TEMPORARY: For the moment we can't use frame-wise fieldmaps
            #     (("outputnode.fieldwarp", pop_file), "inputnode.fieldwarp"),
            # ]),
            # (unwarp_wf, bold_std_trans_wf, [
            #     # TEMPORARY: For the moment we can't use frame-wise fieldmaps
            #     (("outputnode.fieldwarp", pop_file), "inputnode.fieldwarp"),
            # ]),
        ])
        # fmt:on
        return workflow

    # # Finalize connections if ME-EPI
    # join_sdc_echos = pe.JoinNode(
    #     niu.IdentityInterface(
    #         fields=[
    #             "fieldmap",
    #             "fieldwarp",
    #             "corrected",
    #             "corrected_ref",
    #             "corrected_mask",
    #         ]
    #     ),
    #     joinsource="echo_index",
    #     joinfield=[
    #         "fieldmap",
    #         "fieldwarp",
    #         "corrected",
    #         "corrected_ref",
    #         "corrected_mask",
    #     ],
    #     name="join_sdc_echos",
    # )

    # def _dpop(list_of_lists):
    #     return list_of_lists[0][0]

    # # fmt:off
    # workflow.connect([
    #     (unwarp_wf, join_echos, [
    #         ("outputnode.corrected", "bold_files"),
    #     ]),
    #     (unwarp_wf, join_sdc_echos, [
    #         ("outputnode.fieldmap", "fieldmap"),
    #         ("outputnode.fieldwarp", "fieldwarp"),
    #         ("outputnode.corrected", "corrected"),
    #         ("outputnode.corrected_ref", "corrected_ref"),
    #         ("outputnode.corrected_mask", "corrected_mask"),
    #     ]),
    #     # remaining workflow connections
    #     (join_sdc_echos, final_boldref_wf, [
    #         ("corrected", "inputnode.bold_file"),
    #     ]),
    #     (join_sdc_echos, bold_t2s_wf, [
    #         (("corrected_mask", pop_file), "inputnode.bold_mask"),
    #     ]),
    # ])
    # # fmt:on

    # return workflow


def _create_mem_gb(bold_fname):
    bold_size_gb = os.path.getsize(bold_fname) / (1024 ** 3)
    bold_tlen = nb.load(bold_fname).shape[-1]
    mem_gb = {
        "filesize": bold_size_gb,
        "resampled": bold_size_gb * 4,
        "largemem": bold_size_gb * (max(bold_tlen / 100, 1.0) + 4),
    }

    return bold_tlen, mem_gb


def _get_wf_name(bold_fname):
    """
    Derive the workflow name for supplied BOLD file.
    >>> _get_wf_name("/completely/made/up/path/sub-01_task-nback_bold.nii.gz")
    'func_preproc_task_nback_wf'
    >>> _get_wf_name("/completely/made/up/path/sub-01_task-nback_run-01_echo-1_bold.nii.gz")
    'func_preproc_task_nback_run_01_echo_1_wf'
    """
    from nipype.utils.filemanip import split_filename

    fname = split_filename(bold_fname)[1]
    fname_nosub = "_".join(fname.split("_")[1:])
    name = "func_preproc_" + fname_nosub.replace(".", "_").replace(" ", "").replace(
        "-", "_"
    ).replace("_bold", "_wf")

    return name


def _to_join(in_file, join_file):
    """Join two tsv files if the join_file is not ``None``."""
    from niworkflows.interfaces.utility import JoinTSVColumns

    if join_file is None:
        return in_file
    res = JoinTSVColumns(in_file=in_file, join_file=join_file).run()
    return res.outputs.out_file


def extract_entities(file_list):
    """
    Return a dictionary of common entities given a list of files.
    Examples
    --------
    >>> extract_entities("sub-01/anat/sub-01_T1w.nii.gz")
    {'subject': '01', 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}
    >>> extract_entities(["sub-01/anat/sub-01_T1w.nii.gz"] * 2)
    {'subject': '01', 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}
    >>> extract_entities(["sub-01/anat/sub-01_run-1_T1w.nii.gz",
    ...                   "sub-01/anat/sub-01_run-2_T1w.nii.gz"])
    {'subject': '01', 'run': [1, 2], 'suffix': 'T1w', 'datatype': 'anat',
     'extension': '.nii.gz'}
    """
    from collections import defaultdict
    from bids.layout import parse_file_entities

    entities = defaultdict(list)
    for e, v in [
        ev_pair
        for f in listify(file_list)
        for ev_pair in parse_file_entities(f).items()
    ]:
        entities[e].append(v)

    def _unique(inlist):
        inlist = sorted(set(inlist))
        if len(inlist) == 1:
            return inlist[0]
        return inlist

    return {k: _unique(v) for k, v in entities.items()}


def get_img_orientation(imgf):
    """Return the image orientation as a string"""
    img = nb.load(imgf)
    return "".join(nb.aff2axcodes(img.affine))

def prepare_timing_parameters(metadata):
    """ Convert initial timing metadata to post-realignment timing metadata
    """
    timing_parameters = {
        key: metadata[key]
        for key in ("RepetitionTime", "VolumeTiming", "DelayTime",
                    "AcquisitionDuration", "SliceTiming")
        if key in metadata}
    
    run_stc = "SliceTiming" in metadata
    timing_parameters["SliceTimingCorrected"] = run_stc

    if "SliceTiming" in timing_parameters:
        st = sorted(timing_parameters.pop("SliceTiming"))
        TA = st[-1] + (st[1] - st[0])  # Final slice onset - slice duration
        # For constant TR paradigms, use DelayTime
        if "RepetitionTime" in timing_parameters:
            TR = timing_parameters["RepetitionTime"]
            if not np.isclose(TR, TA) and TA < TR:
                timing_parameters["DelayTime"] = TR - TA
        # For variable TR paradigms, use AcquisitionDuration
        elif "VolumeTiming" in timing_parameters:
            timing_parameters["AcquisitionDuration"] = TA

        if run_stc:
            first, last = st[0], st[-1]
            frac = config.workflow.slice_time_ref
            tzero = np.round(first + frac * (last - first), 3)
            timing_parameters["StartTime"] = tzero

    return timing_parameters

def init_func_derivatives_wf(
    bids_root,
    cifti_output,
    freesurfer,
    all_metadata,
    multiecho,
    output_dir,
    spaces,
    use_aroma,
    name='func_derivatives_wf',
):
    """
    Set up a battery of datasinks to store derivatives in the right location.
    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.utility import KeySelect
    from smriprep.workflows.outputs import _bids_relative

    metadata = all_metadata[0]
    timing_parameters = prepare_timing_parameters(metadata)

    nonstd_spaces = set(spaces.get_nonstandard())
    workflow = Workflow(name=name)

    # BOLD series will generally be unmasked unless multiecho,
    # as the optimal combination is undefined outside a bounded mask
    masked = multiecho
    t2star_meta = {
        'Units': 's',
        'EstimationReference': 'doi:10.1002/mrm.20900',
        'EstimationAlgorithm': 'monoexponential decay model',
    }

    inputnode = pe.Node(niu.IdentityInterface(fields=[
        'aroma_noise_ics', 'bold_aparc_std', 'bold_aparc_t1', 'bold_aseg_std',
        'bold_aseg_t1', 'bold_cifti', 'bold_mask_std', 'bold_mask_t1', 'bold_std',
        'bold_std_ref', 'bold_t1', 'bold_t1_ref', 'bold_native', 'bold_native_ref',
        'bold_mask_native', 'bold_echos_native',
        'cifti_variant', 'cifti_metadata', 'cifti_density',
        'confounds', 'confounds_metadata', 'melodic_mix', 'nonaggr_denoised_file',
        'source_file', 'all_source_files',
        'surf_files', 'surf_refs', 'template', 'spatial_reference',
        't2star_bold', 't2star_t1', 't2star_std',
        'bold2anat_xfm', 'anat2bold_xfm', 'acompcor_masks', 'tcompcor_mask']),
        name='inputnode')

    raw_sources = pe.Node(niu.Function(function=_bids_relative), name='raw_sources')
    raw_sources.inputs.bids_root = bids_root

    workflow.connect([
        (inputnode, raw_sources, [('all_source_files', 'in_files')]),
        # (inputnode, ds_confounds, [('source_file', 'source_file'),
        #                            ('confounds', 'in_file'),
        #                            ('confounds_metadata', 'meta_dict')]),
        # (inputnode, ds_ref_t1w_xfm, [('source_file', 'source_file'),
        #                              ('bold2anat_xfm', 'in_file')]),
        # (inputnode, ds_ref_t1w_inv_xfm, [('source_file', 'source_file'),
        #                                  ('anat2bold_xfm', 'in_file')]),
    ])

    if nonstd_spaces.intersection(('func', 'run', 'bold', 'boldref', 'sbref')):
        ds_bold_native = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir, desc='preproc', compress=True, SkullStripped=masked,
                TaskName=metadata.get('TaskName'), **timing_parameters),
            name='ds_bold_native', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_bold_native_ref = pe.Node(
            DerivativesDataSink(base_directory=output_dir, suffix='boldref', compress=True,
                                dismiss_entities=("echo",)),
            name='ds_bold_native_ref', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_bold_mask_native = pe.Node(
            DerivativesDataSink(base_directory=output_dir, desc='brain', suffix='mask',
                                compress=True, dismiss_entities=("echo",)),
            name='ds_bold_mask_native', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)

        workflow.connect([
            (inputnode, ds_bold_native, [('source_file', 'source_file'),
                                         ('bold_native', 'in_file')]),
            (inputnode, ds_bold_native_ref, [('source_file', 'source_file'),
                                             ('bold_native_ref', 'in_file')]),
            (inputnode, ds_bold_mask_native, [('source_file', 'source_file'),
                                              ('bold_mask_native', 'in_file')]),
            (raw_sources, ds_bold_mask_native, [('out', 'RawSources')]),
        ])

    # Resample to T1w space
    if nonstd_spaces.intersection(('T1w', 'anat')):
        ds_bold_t1 = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir, space='T1w', desc='preproc', compress=True,
                SkullStripped=masked, TaskName=metadata.get('TaskName'), **timing_parameters),
            name='ds_bold_t1', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_bold_t1_ref = pe.Node(
            DerivativesDataSink(base_directory=output_dir, space='T1w', suffix='boldref',
                                compress=True, dismiss_entities=("echo",)),
            name='ds_bold_t1_ref', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_bold_mask_t1 = pe.Node(
            DerivativesDataSink(base_directory=output_dir, space='T1w', desc='brain',
                                suffix='mask', compress=True, dismiss_entities=("echo",)),
            name='ds_bold_mask_t1', run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        workflow.connect([
            (inputnode, ds_bold_t1, [('source_file', 'source_file'),
                                     ('bold_t1', 'in_file')]),
            (inputnode, ds_bold_t1_ref, [('source_file', 'source_file'),
                                         ('bold_t1_ref', 'in_file')]),
            (inputnode, ds_bold_mask_t1, [('source_file', 'source_file'),
                                          ('bold_mask_t1', 'in_file')]),
            (raw_sources, ds_bold_mask_t1, [('out', 'RawSources')]),
        ])
        if freesurfer:
            ds_bold_aseg_t1 = pe.Node(DerivativesDataSink(
                base_directory=output_dir, space='T1w', desc='aseg', suffix='dseg',
                compress=True, dismiss_entities=("echo",)),
                name='ds_bold_aseg_t1', run_without_submitting=True,
                mem_gb=DEFAULT_MEMORY_MIN_GB)
            ds_bold_aparc_t1 = pe.Node(DerivativesDataSink(
                base_directory=output_dir, space='T1w', desc='aparcaseg', suffix='dseg',
                compress=True, dismiss_entities=("echo",)),
                name='ds_bold_aparc_t1', run_without_submitting=True,
                mem_gb=DEFAULT_MEMORY_MIN_GB)
            workflow.connect([
                (inputnode, ds_bold_aseg_t1, [('source_file', 'source_file'),
                                              ('bold_aseg_t1', 'in_file')]),
                (inputnode, ds_bold_aparc_t1, [('source_file', 'source_file'),
                                               ('bold_aparc_t1', 'in_file')]),
            ])

    if getattr(spaces, '_cached') is None:
        return workflow

    # Store resamplings in standard spaces when listed in --output-spaces
    if spaces.cached.references:
        from niworkflows.interfaces.space import SpaceDataSource

        spacesource = pe.Node(SpaceDataSource(),
                              name='spacesource', run_without_submitting=True)
        spacesource.iterables = ('in_tuple', [
            (s.fullname, s.spec) for s in spaces.cached.get_standard(dim=(3,))
        ])

        fields = ['template', 'bold_std', 'bold_std_ref', 'bold_mask_std']
        if multiecho:
            fields.append('t2star_std')
        select_std = pe.Node(KeySelect(fields=fields),
                             name='select_std', run_without_submitting=True,
                             mem_gb=DEFAULT_MEMORY_MIN_GB)

        ds_bold_std = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir, desc='preproc', compress=True, SkullStripped=masked,
                TaskName=metadata.get('TaskName'), **timing_parameters),
            name='ds_bold_std', run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_bold_std_ref = pe.Node(
            DerivativesDataSink(base_directory=output_dir, suffix='boldref', compress=True,
                                dismiss_entities=("echo",)),
            name='ds_bold_std_ref', run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)
        ds_bold_mask_std = pe.Node(
            DerivativesDataSink(base_directory=output_dir, desc='brain', suffix='mask',
                                compress=True, dismiss_entities=("echo",)),
            name='ds_bold_mask_std', run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)

        workflow.connect([
            (inputnode, ds_bold_std, [('source_file', 'source_file')]),
            (inputnode, ds_bold_std_ref, [('source_file', 'source_file')]),
            (inputnode, ds_bold_mask_std, [('source_file', 'source_file')]),
            (inputnode, select_std, [('bold_std', 'bold_std'),
                                     ('bold_std_ref', 'bold_std_ref'),
                                     ('bold_mask_std', 'bold_mask_std'),
                                     ('t2star_std', 't2star_std'),
                                     ('template', 'template'),
                                     ('spatial_reference', 'keys')]),
            (spacesource, select_std, [('uid', 'key')]),
            (select_std, ds_bold_std, [('bold_std', 'in_file')]),
            (spacesource, ds_bold_std, [('space', 'space'),
                                        ('cohort', 'cohort'),
                                        ('resolution', 'resolution'),
                                        ('density', 'density')]),
            (select_std, ds_bold_std_ref, [('bold_std_ref', 'in_file')]),
            (spacesource, ds_bold_std_ref, [('space', 'space'),
                                            ('cohort', 'cohort'),
                                            ('resolution', 'resolution'),
                                            ('density', 'density')]),
            (select_std, ds_bold_mask_std, [('bold_mask_std', 'in_file')]),
            (spacesource, ds_bold_mask_std, [('space', 'space'),
                                             ('cohort', 'cohort'),
                                             ('resolution', 'resolution'),
                                             ('density', 'density')]),
            (raw_sources, ds_bold_mask_std, [('out', 'RawSources')]),
        ])

    return workflow