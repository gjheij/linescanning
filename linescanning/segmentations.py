from . import image, utils, transform
import os
import nibabel as nb
import numpy as np
import pickle
from bids import BIDSLayout
import matplotlib.pyplot as plt
opj = os.path.join

class Segmentations:

    def __init__(self, 
                 subject, 
                 run=None,
                 derivatives=None, 
                 trafo_file=None, 
                 reference_slice=None, 
                 reference_session=1, 
                 target_session=2, 
                 foldover="FH", 
                 pickle_file=None,
                 verbose=False):

        """Segmentations

        Class to project segmentations created using the pipeline described on https://linescanning.readthedocs.io/en/latest/ to a single slice image of a new session (typically a line-scanning session). By default, it will look for files in the *Nighres*-directory, as these segmentations are generally of most interest. The output of the class will be a pickle-file containing the following segmentations: CRUISE-tissue segmentation (as per the output of https://github.com/gjheij/linescanning/blob/main/shell/spinoza_cortexreconstruction), the layer+depth segmentation (https://github.com/gjheij/linescanning/blob/main/shell/spinoza_layering), the brain mask, tissue probability maps (https://github.com/gjheij/linescanning/blob/main/shell/spinoza_extractregions), the reference slice, and the line as acquired in the session.

        To warp the files, you'll need to specify a forward-transformation matrix (e.g., from *reference session* to *target session*), the reference slice, and the foldover direction (e.g., FH or AP) describing the nature of the applied saturation slabs. You can also specify an earlier pickle file, in which case the segmentations embedded in that file are loaded in for later manipulation with e.g., :func:`pRFline.segmentations.plot_segmentations` to create overview figures.

        Parameters
        ----------
        subject: str
            Subject ID as used in `SUBJECTS_DIR` and used throughout the pipeline
        run: int, optional
            run number you'd like to have the segmentations for            
        derivatives: str, optional
            Path to derivatives folder of the project. Generally should be the path specified with `DIR_DATA_DERIV` in the bash environment (if using https://github.com/gjheij/linescanning).
        trafo_file: str, optional
            Forward matrix mapping *reference session* (typically `ses-1`) to *target session* (typically `ses-2`) [ANTs file], by default None as it's not required when specifying an earlier created *pickle_file*
        reference_slice: str, optional
            Path to nifti image of a *acq-1slice* image that is used as reference to project the segmentations into, by default None
        reference_session: int, optional
            Origin of the segmentations, by default 1
        target_session: int, optional
            Target of the segmentations, by default 2
        foldover: str, optional
            Direction of applied saturation slabs, by default "FH". You can find this in the *derivatives/pycortex/<subject>/line_pycortex.csv*-file if using https://github.com/gjheij/linescanning.
        pickle_file: str, optional
            Existing pickle file containing filepaths to segmentations in *target session* space, by default None.

        Raises
        ----------
        ValueError
            If either transformation file or reference file do not exists. These are required for the projection of segmentations into the slice. This error will not be thrown if *pickle_file* was specified.

        Example
        ----------
        >>> # load existing pickle file
        >>> from pRFline import segmentations
        >>> ff = "<some_path>/segmentations.pkl"
        >>> ref = "<some_path>/ref_slice.nii.gz"
        >>> segs = segmentations.Segmentations(<subject>, pickle_file=ff, reference_slice=ref)

        >>> # create pickle file with segmentations for a  single subject
        >>> from linescanning import segmentations
        >>> import os
        >>> ref = "<some_path>/ref_slice.nii.gz"
        >>> to_ses = 3
        >>> sub = "sub-003"
        >>> derivatives = os.environ.get('DIR_DATA_DERIV')
        >>> segs = segmentations.Segmentations(sub, reference_slice=ref, trafo_file=trafo, target_session=to_ses, foldover="FH")

        >>> # loop over a bunch of subjects
        >>> subject_list = ['sub-001','sub-003','sub-004','sub-005','sub-006']
        >>> all_segmentations = {}
        >>> for ii in subject_list:
        >>>     ref = f"{subject}_ref_slice.nii.gz"
        >>>     matrix_file = f"{subject}_from-ses1_to-ses2.mat"
        >>>     segs = segmentations.Segmentations(ii, reference_slice=ref, trafo_file=matrix_file)
        >>>     all_segmentations[ii] = segs.segmentations_df.copy
        >>> # plot all subjects
        >>> segmentations.plot_segmentations(all_segmentations, , max_val_ref=3000, figsize=(15,5*len(subject_list))))

        Notes
        ----------
        Assumes your anatomical segmentation files have the *acq-MP2RAGE*-tag. This tag will be replaced by *1slice* in segmentation-to-slice images
        """

        self.subject            = subject
        self.run                = run
        self.project_home       = derivatives
        self.trafo_file         = trafo_file
        self.reference_slice    = reference_slice
        self.reference_session  = reference_session
        self.target_session     = target_session
        self.foldover           = foldover
        self.pickle_file        = pickle_file
        self.verbose            = verbose

    
        if self.pickle_file == None:

            # try default project_home if none is specified
            if self.project_home == None:
                try:
                    self.project_home = os.environ.get('DIR_DATA_HOME')
                except Exception:
                    print("Please specify the project's root directory (where 'derivatives' lives)")

            # specify nighres directory
            self.nighres_source     = opj(self.project_home, 'derivatives', 'nighres', self.subject, f'ses-{self.reference_session}') 
            self.nighres_target     = opj(os.path.dirname(self.nighres_source), f'ses-{self.target_session}') 
            self.mask_dir           = opj(self.project_home, 'derivatives', 'manual_masks', self.subject, f'ses-{self.reference_session}')
            self.cortex_dir         = opj(self.project_home, 'derivatives', 'pycortex', self.subject)

            if not os.path.exists(self.nighres_target):
                os.makedirs(self.nighres_target, exist_ok=True)

            # fetch segmentations, assuming default directory layout
            nighres_layout          = BIDSLayout(self.nighres_source, validate=False).get(extension=['nii.gz'], return_type='file')
            self.wb_cruise          = utils.get_bids_file(nighres_layout, ["cortex"])
            self.wb_layers          = utils.get_bids_file(nighres_layout, ["layers"])
            self.wb_depth           = utils.get_bids_file(nighres_layout, ["depth"])

            # fetch mask and tissue probabilities
            mask_layout             = BIDSLayout(self.mask_dir, validate=False).get(extension=['nii.gz'], return_type='file')
            self.wb_wm              = utils.get_bids_file(mask_layout, ["label-WM"])
            self.wb_gm              = utils.get_bids_file(mask_layout, ["label-GM"])
            self.wb_csf             = utils.get_bids_file(mask_layout, ["label-CSF"])
            self.wb_brainmask       = utils.get_bids_file(mask_layout, ["brainmask"])

            # check if reference slice and transformation file actually exist
            if not os.path.exists(self.reference_slice):
                raise ValueError(f"Could not find reference slice {self.reference_slice}")

            if self.trafo_file == None:
                # try the default in derivatives/pycortex/<subject>/transforms
                self.trafo_file = utils.get_file_from_substring([f"from-fs_to-ses{self.target_session}", ".mat"], opj(self.cortex_dir, 'transforms'), return_msg="None")
                if self.trafo_file == None:
                    raise ValueError(f"Could not find default trafo-file 'from-fs_to-ses{self.target_session}*.mat' in {opj(self.cortex_dir, 'transforms')}")
            elif not os.path.exists(self.trafo_file):
                raise ValueError(f"Could not find trafo_file {self.trafo_file}")
    
            # start warping (in brackets file suffixes)
            #  0 = wm prob  ("label-WM")
            #  1 = gm prob  ("label-GM")
            #  2 = csf prob ("label-CSF")
            #  3 = pve      ("cruise-cortex")
            #  4 = layers   ("layering-layers")
            #  5 = depth    ("layering-depth")
            #  6 = mask     ("brainmask")

            if self.verbose:
                print(f" Source dir: {self.nighres_source}")
                print(f" Target session: ses-{self.target_session}")
                print(f" Foldover: {self.foldover}")
                print(f" Ref slice: {self.reference_slice}")

            in_type = ['prob', 'prob', 'prob', 'tissue', 'layer', 'prob', 'tissue']
            tag = ['wm', 'gm', 'csf', 'cortex', 'layers', 'depth', 'mask']
            self.resampled = {}
            self.resampled_data ={}
            for nr,file in enumerate([self.wb_wm, self.wb_gm, self.wb_csf, self.wb_cruise, self.wb_layers, self.wb_depth, self.wb_brainmask]):

                # replace acq-MP2RAGE with acq-1slice
                if self.run == None:
                    new_fn = utils.replace_string(file, "acq-MP2RAGE", "acq-1slice")
                else:
                    new_fn = utils.replace_string(file, "acq-MP2RAGE", f"acq-1slice_run-{self.run}")
                new_file = opj(self.nighres_target, os.path.basename(new_fn))
                
                if not os.path.exists(new_file):
                    if in_type[nr] == "tissue":
                        # Use MultiLabel-interpolation for tissue-segmentation
                        transform.ants_applytrafo(self.reference_slice, file, interp="mul", trafo=self.trafo_file, output=new_file)
                    elif in_type[nr] == "layer":
                        # Use GenericLabel-interpolation for layer-segmentation
                        transform.ants_applytrafo(self.reference_slice, file, interp="gen", trafo=self.trafo_file, output=new_file)
                    else:
                        # Use nearest neighbor-interpolation for probability maps
                        transform.ants_applytrafo(self.reference_slice, file, interp="nn", trafo=self.trafo_file, output=new_file)
                
                # collect them in 'resampled' dictionary
                self.resampled[tag[nr]] = new_file
                self.resampled_data[tag[nr]] = nb.load(new_file).get_fdata()

            self.resampled['ref'] = self.reference_slice; self.resampled_data['ref'] = nb.load(self.reference_slice).get_fdata()
            self.resampled['line'] = image.create_line_from_slice(self.reference_slice, fold=self.foldover); self.resampled_data['line'] = nb.load(self.reference_slice).get_fdata()

            if self.run != None:
                self.pickle_file = opj(self.nighres_target, f'{subject}_ses-{self.target_session}_run-{self.run}_desc-segmentations.pkl')
            else:
                self.pickle_file = opj(self.nighres_target, f'{subject}_ses-{self.target_session}_desc-segmentations.pkl')
                
            pickle_file = open(self.pickle_file, "wb")
            pickle.dump(self.resampled, pickle_file)
            pickle_file.close()

        else:

            with open(self.pickle_file, 'rb') as pf:
                self.resampled = pickle.load(pf)

            if 'ref' not in list(self.resampled.keys()):
                self.resampled['ref'] = self.reference_slice
                self.resampled['line'] = image.create_line_from_slice(self.reference_slice, fold=self.foldover)

        self.segmentation_df = {}
        self.segmentation_df[self.subject] = self.resampled.copy()
        
        # get the WM/CSF voxels for regressors
        self.wm_csf_voxels_for_regressors()

        if self.verbose:
            print(f" Found {len(self.acompcor_voxels)} voxels for nuisance regression")
            print(" We're good to go!")

    def get_plottable_segmentations(self, input_seg, return_dimensions=2):
        """get_plottable_segmentations

        Quick function to convert the input data to data that is compatible with *plt.imshow* (e.g., 2D data). Internally calls upon :func:`linescanning.utils.squeeze_generic` which allows you to select which dimensions to keep. In the case you want imshow-compatible data, you'd specify `return_dimensions=2`.

        Parameters
        ----------
        data: nibabel.Nifti1Image, str, numpy.ndarray
            Input data to be conformed to *imshow*-compatible data
        return_dimensions: int, optional
            Number of axes to keep, by default 2

        Returns
        ----------
        numpy.ndarray
            Input data with *return_dimensions* retained

        Example
        ----------
        See [insert reference to readthedocs here after pushing to linescanning repo]
        """

        if isinstance(input_seg, nb.Nifti1Image):
            return_data = input_seg.get_fdata()
        elif isinstance(input_seg, str):
            return_data = nb.load(input_seg).get_fdata()
        elif isinstance(input_seg, np.ndarray):
            return_data = input_seg.copy()
        else:
            raise TypeError(f"Unknown type '{type(input_seg)}' for '{input_seg}'")

        return utils.squeeze_generic(return_data, range(return_dimensions))
        
        
    def plot_segmentations(self, 
                           subj_df=None,
                           include=['ref', 'cortex', 'layers'], 
                           cmaps=['Greys_r', 'Greys_r', 'hot'], 
                           cmap_color_line="#f0ff00", 
                           max_val_ref=2400, 
                           overlay_line=True, 
                           figsize=(15,5), 
                           save_as=None):

        """plot_segmentations

        Function to create grid plots of various segmentations, either for one subject or a number of subjects depending on the nature of `segmentation_df`. 

        Parameters
        ----------
        segmentation_df: dict
            Dictionary as per the output of :class:`pRFline.segmentations.Segmentation`, specifically the attribute `:attr:`pRFline.segmentations.Segmentation.segmentation_df`. This is a nested dictionary with the head key being the subject ID specified in the class, and within that there's a dictionary with keys pointing to the various segmentations: ['wm', 'gm', 'csf', 'cortex', 'layers', 'depth', 'mask', 'ref', 'line']
        include: list, optional
            Filter for segmentations to include, by default ['ref', 'cortex', 'layers']. These should match the keys outlined above.
        cmaps: list, optional
            Color maps to be used for segmentations filtered by *include*, by default ['Greys_r', 'Greys_r', 'hot']. Should match the length of included segmentations (`include`)
        cmap_color_line: str, tuple, optional
            Hex code for line overlay, by default "#f0ff00" (yellow-ish). Can also be a tuple (see :func:`linescanning.utils.make_binary_cm`)
        max_val_ref: int, optional
            Scalar for the reference slice image, by default 2400
        overlay_line: bool, optional
            Overlay the outline of the line on top of the segmentations, by default True
        figsize: tuple, optional
            Figure dimensions as per usual matplotlib conventions, by default (15,5). Multiples of 5 seems to scale nicely when plotting multiple subjects. E.g., 3 subject and 3 segmentation > set figsize to *(15,15)*.
        save_as: str, optional
            Save the plot, by default None. If you want to use figures in Inkscape, save them as PDFs to retain high resolution

        Example
        ----------
            >>> from pRFline import segmentations
            >>> ff = "<some_path>/segmentations.pkl"
            >>> ref = "<some_path>/ref_slice.nii.gz"
            >>> segs = segmentations.Segmentations(<subject>, pickle_file=ff, reference_slice=ref)
            >>> segmentations.plot_segmentations(segs.segmentation_df, max_val_ref=3000, figsize=(15,5))
        """
        
        # because 'segmentation_df' can contain multiple subjects, decide on number of columns & rows for figure
        if subj_df == None:
            use_df = self.segmentation_df.copy()
        else:
            use_df = subj_df.copy()
            
        subject_list = list(use_df.keys())
        nr_subjects         = len(subject_list)
        nr_segmentations    = len(include)

        # if one subject is given, plot segmentations along x-axis (each segmentation a column)
        if len(subject_list) == 1:
            fig, axs = plt.subplots(1, nr_segmentations, figsize=figsize)
        else:
            # if multple segmentations, plot the segmentations as rows with subjects as columns
            fig, axs = plt.subplots(nr_segmentations, nr_subjects, figsize=figsize)

        for ix, sub in enumerate(subject_list):
            for ic, seg_type in enumerate(include):

                if len(subject_list) == 1:
                    ax = axs[ic] 
                else:
                    ax = axs[ix,ic]

                seg = self.get_plottable_segmentations(use_df[sub][seg_type])
                if seg_type == "ref":
                    ax.imshow(np.rot90(seg), vmax=max_val_ref, cmap=cmaps[ic])
                else:
                    ax.imshow(np.rot90(seg), cmap=cmaps[ic])

                if overlay_line:
                    # create binary colormap for line
                    beam_cmap = utils.make_binary_cm(cmap_color_line)

                    # load data
                    line = self.get_plottable_segmentations(
                        use_df[sub]['line'])

                    # plot data
                    ax.imshow(np.rot90(line), cmap=beam_cmap, alpha=0.6)

                ax.axis('off')

        plt.tight_layout()

        if save_as:
            fig.savefig(save_as)

    def plot_line_segmentations(self, 
                                subj_df=None,
                                include=['ref', 'wm', 'gm', 'csf', 'cortex', 'layers', 'mask'], 
                                cmap_color_mask="#08B2F0", 
                                figsize=(8,4), 
                                layout="vertical", 
                                move_factor=None, 
                                save_as=None):

        """plot_line_segmentations

        Plot and return the 16 middle voxel rows representing the content of the line for each of the selected segmentations. These segmentations are indexed with keys as per the attribute :class:`pRFline.segmentations.Segmentations.segmentations_df`. The most complex part of this function is the plotting indexing, but the voxel selection is pretty straightforward: in the output dictionary we have a key *line*. This line is converted to a boolean and multiplied with the segmentations to extract 16 middle voxels. The output of this is stored in *beam[<subject>][<segmentation key>]* and returned to the user after plotting the segmentations. If you have multiple subjects in your input dataframe, make sure to tinker with *move_factor*, which represents a factor of moving the subject-specific plots to the right of the total figure. By default, it's some factor over the number of subjects, but it's good to change this parameter and see what happens to understand it. 

        Parameters
        ----------
        segmentation_df: dict
            Dictionary as per the output of :class:`pRFline.segmentations.Segmentation`, specifically the attribute `:attr:`pRFline.segmentations.Segmentation.segmentation_df`. This is a nested dictionary with the head key being the subject ID specified in the class, and within that there's a dictionary with keys pointing to the various segmentations: ['wm', 'gm', 'csf', 'cortex', 'layers', 'depth', 'mask', 'ref', 'line']
        include: list, optional
            Filter for segmentations to include, by default ['ref', 'cortex', 'layers']. These should match the keys outlined above.
        cmap_color_mask: str, tuple, optional
            Hex code for line overlay, by default "#f0ff00" (yellow-ish). Can also be a tuple (see :func:`linescanning.utils.make_binary_cm`)
        figsize: tuple, optional
            Figure dimensions as per usual matplotlib conventions, by default (15,5). Multiples of 5 seems to scale nicely when plotting multiple subjects. E.g., 3 subject and 3 segmentation > set figsize to *(15,15)*.
        cmap_color_mask: str, optional
            [description], by default "#08B2F0"
        figsize: tuple, optional
            [description], by default (8, 4)
        layout: str, optional
            For a single subject, we can plot the segmentations either as rows below each other (*layout='horizontal'*) or in columns next to one another (*layout='vertical'*). For multiple subjects, the latter option is available, hence "vertical" is the default.
        move_factor: float, optional
            A factor of moving the subject-specific plots to the right of the total figure, by default *nr_subject/7*, but this is arbitrary. Make sure to play with this factor!      
        save_as: str, optional
            Save the plot, by default None. If you want to use figures in Inkscape, save them as PDFs to retain high resolution

        Returns
        ----------
        dict
            Dictionary collecting <subject> keys with segmentation keys nested in it

        matplotlib.pyplot
            Prints a plot to the terminal

        str
            If *save_as* was specified, the string representing the path name will also be returned 

        Example
        ----------
        >>> from pRFline import segmentations
        >>> ff = "<some_path>/segmentations.pkl"
        >>> ref = "<some_path>/ref_slice.nii.gz"
        >>> segs = segmentations.Segmentations(<subject>, pickle_file=ff, reference_slice=ref)
        >>> segmentations.plot_line_segmentations(segs.segmentation_df, figsize=(15, 5), layout="horizontal") # plot horizontal beams
        {'ref': array([[14.288, 14.288, 14.288, ..., 42.863, 57.151, 57.151],
        [ 0.   ,  0.   ,  0.   , ..., 14.288, 14.288, 14.288],
        [28.576, 42.863, 42.863, ..., 14.288, 14.288,  0.   ],
        ...,
        [14.288, 14.288, 14.288, ..., 14.288, 14.288, 14.288],
        [ 0.   ,  0.   ,  0.   , ...,  0.   ,  0.   ,  0.   ],
        [ 0.   ,  0.   ,  0.   , ...,  0.   ,  0.   ,  0.   ]]),
        'wm': array([[0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.],
                ...,
        }
        >>> segmentations.plot_line_segmentations(segs.segmentation_df, layout="vertical", figsize=(15,10)) # plot vertical beams

        Notes
        ----------
        Nested gridspec inspiration from: https://matplotlib.org/3.1.1/tutorials/intermediate/gridspec.html
        """

        # because 'segmentation_df' can contain multiple subjects, decide on number of columns & rows for figure
        if subj_df == None:
            use_df = self.segmentation_df.copy()
        else:
            use_df = subj_df.copy()

        subject_list        = list(use_df.keys())
        nr_subjects         = len(subject_list)
        nr_segmentations    = len(include)

        # set plot defaults
        fig = plt.figure(constrained_layout=False, figsize=figsize)

        if nr_subjects > 1:
            if layout == "horizontal":
                layout = "vertical"
                print("WARNING: 'vertical' layout was specified, but I can't do that with multiple subjects. Changing to 'vertical'")
        
        # add gridspec per subject
        self.beam_data = {}
        grids = []
        start_grid_right = 0.48
        start_grid_left = 0.05
        for ix, sub in enumerate(subject_list):

            if move_factor == None:
                move_factor = nr_subjects/7

            if ix != 0:
                start_grid_right += move_factor
                start_grid_left += move_factor
                
            # print(f"Adding axis for {sub} (grid {ix})")
            
            beam = {}
            if layout == "horizontal":
                cols = 1
                rows = nr_segmentations
                aspect = 3/1
                rot = True
            elif layout == "vertical":
                cols = nr_segmentations
                rows = 1
                aspect = 1/3
                rot = False
                
            grids.append(fig.add_gridspec(nrows=rows, ncols=cols,
                            left=start_grid_left, right=start_grid_right, wspace=0.05))

            # add subplots to gridspec by looping over segmentations
            for idx, ii in enumerate(include):

                # print(f" subplot: {idx}")

                seg         = self.get_plottable_segmentations(use_df[sub][ii])
                line        = self.get_plottable_segmentations(use_df[sub]['line'])
                
                if self.foldover == "FH":
                    beam[ii] = np.multiply(seg, line.astype(bool))[:, 352:368]
                else:
                    beam[ii] = np.multiply(seg, line.astype(bool))[352:368, :]
                
                if ii == "mask":
                    use_cmap = utils.make_binary_cm(cmap_color_mask)
                elif ii == "layers":
                    use_cmap = "hot"
                else:
                    use_cmap = "Greys_r"

                if rot:
                    plot_data = np.rot90(beam[ii])
                else:
                    plot_data = beam[ii]

                if layout == "horizontal":
                    plot = fig.add_subplot(grids[ix][idx, 0])
                else:
                    plot = fig.add_subplot(grids[ix][0, idx])
                plot.imshow(plot_data, aspect=aspect, cmap=use_cmap)

                if layout == "vertical":
                    if ix != 0:
                        plot.set_yticks([])
                    else:
                        if idx != 0:
                            plot.set_yticks([])
                    plot.set_xticks([])
                else:
                    if ix != subject_list.index(subject_list[-1]):
                        plot.set_xticks([])
                    else:
                        if idx != include.index(include[-1]):
                            plot.set_xticks([])
                    plot.set_yticks([])

            self.beam_data[sub] = beam.copy()

        plt.show()

        if save_as:
            fig.savefig(save_as)
                

    def wm_csf_voxels_for_regressors(self):

        """wm_csf_voxels_for_regressors

        Generate a list of voxels that consist entirely of WM/CSF-stuff. As the beam is 16 voxels wide, a voxel is considered WM/CSF if *all* values across the beam are the WM/CSF value in the CRUISE-output from Nighres. This list of voxels can then be used to perform aCompCor in :class:`linescanning.dataset.ParseFuncFile`.

        """
            
        self.cortex     = self.get_plottable_segmentations(self.segmentation_df[self.subject]['cortex'])
        self.line       = self.get_plottable_segmentations(self.segmentation_df[self.subject]['line'])
        self.mask       = self.get_plottable_segmentations(self.segmentation_df[self.subject]['mask'])

        if self.foldover == "FH":
            self.beam_ctx   = np.multiply(self.cortex, self.line.astype(bool))[:, 352:368]
            self.mask_beam  = np.multiply(self.mask, self.line.astype(bool))[:, 352:368]
        else:
            self.beam_ctx   = np.multiply(self.cortex, self.line.astype(bool))[352:368, :]
            self.mask_beam  = np.multiply(self.mask, self.line.astype(bool))[352:368, :]

        self.wm_voxels = []
        self.csf_voxels = []
        self.acompcor_voxels = []
        for vox in range(self.beam_ctx.shape[0]):

            # remove outer stuff from beam
            if all(mas == 1 for mas in self.mask_beam[vox,:]):

                # fetch voxel id's where all 16 voxels across beam are 2
                wm_vox = all(elem == 2 for elem in self.beam_ctx[vox,:])
                if wm_vox == True:
                    self.wm_voxels.append(vox)
                    self.acompcor_voxels.append(vox)

                csf_vox = all(elem == 0 for elem in self.beam_ctx[vox, :])
                if csf_vox == True:
                    self.csf_voxels.append(vox)
                    self.acompcor_voxels.append(vox)

        # list of voxels
        self.acompcor_voxels = sorted(self.acompcor_voxels)

        # create 2D representations of nuisance voxels
        empty_slice = np.zeros_like(self.cortex)

        for t_type in ['wm', 'csf']:
            vox_list = getattr(self, f"{t_type}_voxels")
            template = empty_slice.copy()

            if self.foldover == "FH":
                template[vox_list, 352:368] = 1
                # setattr(self, f"{t_type}_in_slice")
                # self.beam_wm_voxels   = np.multiply(self.cortex, self.line.astype(bool))[:, 352:368]
            else:
                template[352:368, vox_list] = 1

            setattr(self, f"{t_type}_in_slice", template)

    def plot_regressor_voxels(self, figsize=(8,8), cmap_color=["#338EFF", "#FF4F33"], ax=None, title="WM/CSF voxels for nuisance regression", fontsize=16):
        """plot_regressor_voxels

        Make an image of where the white matter/CSF voxels from :func:`linescanning.segmentations.Segmentations.wm_csf_voxels_for_regressors` are located along the line. 

        Parameters
        ----------
        figsize: tuple, optional
            Figure size if default setting is insufficient, by default (8,8)
        cmap_color: list, optional
            Colors for the WM/CSF-voxels, by default ["#338EFF", "#FF4F33"]
        ax: matplotlib-axis, optional
            Use a custom matplotlib axis, by default None
        title: str, optional
            _description_, by default "WM/CSF voxels for nuisance regression"
        fontsize: int, optional
            _description_, by default 16

        Example
        ----------
        >>> 
        """

        
        if not hasattr(self, "wm_in_slice"):
            self.wm_csf_voxels_for_regressors()

        if ax == None:
            fig,ax = plt.subplots(figsize=figsize)
        
        self.regressor_voxel_colors = cmap_color
        cmap_csf = utils.make_binary_cm(self.regressor_voxel_colors[0])
        cmap_wm = utils.make_binary_cm(self.regressor_voxel_colors[1])

        ax.imshow(np.rot90(self.cortex), cmap="Greys_r")
        ax.imshow(np.rot90(self.csf_in_slice), cmap=cmap_csf)
        ax.imshow(np.rot90(self.wm_in_slice), cmap=cmap_wm)
        ax.axis('off')
    
        if title:
            ax.set_title(title, fontsize=fontsize)

    def segmentations_to_beam(self, subj_df=None, subject=None):
        
        custom_input = False
        if subject != None:
            if subj_df != None:
                custom_input = True
                use_data = subj_df[subject]
                use_subj = subject
            else:
                raise ValueError(f"Must provide dictionary containing {subject} for this operation")
        elif subject == None and subj_df != None:
            raise ValueError("Must provide a subject ID for this operation")
        else:
            use_data = self.segmentation_df[self.subject].copy()
            subject = self.subject
            
        segmentations_in_beam  = {}; segmentations_in_beam[subject] = {}
        beam_in_slice          = {}; beam_in_slice[subject] = {}
        line                   = self.get_plottable_segmentations(use_data['line'])

        # this bit extracts the middle 16 voxels along the line, resulting in a (720,16) array for each segmentation
        for seg in list(use_data.keys()):
            seg_in_slice = self.get_plottable_segmentations(use_data[seg])

            if self.foldover == "FH":
                seg_in_beam = np.multiply(seg_in_slice, line.astype(bool))[:, 352:368]
            else:
                segin_beam = np.multiply(seg_in_slice, line.astype(bool))[352:368, :]

            segmentations_in_beam[subject][seg] = seg_in_beam

            # this thing inserts the beam into an empty slice, so we can create nifti's of the beam arrays (720,720) with only middle 16 voxels
            beam_in_slice[subject][seg] = image.create_line_from_slice(seg_in_slice, fold=self.foldover, keep_input=True)

        # custom input can be the case of multiple subjects
        if custom_input:
            return segmentations_in_beam
        else:
            self.segmentations_in_beam = segmentations_in_beam
            self.beam_in_slice = beam_in_slice

    def plot_beam_in_slice(self, include='all', figsize=None, cmap_color_mask="#08B2F0", rot=True, save=False):
        """plot_beam_in_slice

        The format of this data is the segmentations in beam-representation. This means everything but the 16 middle voxels along the line are set to zero, ensuring that the slice has a shape (720,720). This function is primarily aimed at plotting these images, but we can also write them to nifti-files using the `save=True` flag. We take the affine and header from `self.segmentation_df[self.subject]['ref']` to create the nifti file.

        Parameters
        ----------
        include: str, list, optional
            Which segmentations to include in the plotting, by default 'all'. Can also be a list of desired segmentations, but then `include` must consist strings that exist as key in `self.segmentation_df`.
        figsize: type, optional
            Specified figure size if the default isn't sufficient, by default None. Defaults to the number of included specifications (via the `include`-flag)*5 by 5 (segmentations will be plotted as columns)
        cmap_color_mask: str, optional
            Hex code for colormap of binary mask, by default "#08B2F0"
        rot: bool, optional
            Rotate (or not) the input numpy array. This might be necessary for adequate visualization of slices where the foldover direction was set to FH, by default True
        save: bool, optional
            Save the arrays as nifti-files that we could potentially project to the surface, by default False

        Example
        ----------
        >>> segs = segmentations.Segmentations(<subject>, pickle_file=<pickle file>, reference_slice=<reference slice>, target_session=<target session>)
        >>> segs.plot_beam_in_slice(save=True)
        """

        if not hasattr(self, "beam_in_slice"):
            self.segmentations_to_beam()

        if include == "all":
            include_seg = self.beam_in_slice[self.subject].keys()
        elif isinstance(include, list):
            include_seg = include.copy()
        else:
            raise ValueError(f"Unrecognized input for 'include': {include}. Must be a list of keys to include or 'all'")

        if figsize == None:
            figsize = (len(include_seg)*5,5)

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1,len(include_seg))

        gs_ix = 0
        for ix, seg in enumerate(self.beam_in_slice[self.subject].keys()):
            if seg in include_seg:

                if seg == "mask":
                    use_cmap = utils.make_binary_cm(cmap_color_mask)
                elif seg == "layers":
                    use_cmap = "hot"
                else:
                    use_cmap = "Greys_r"
                    
                ax = fig.add_subplot(gs[gs_ix])
                img = self.beam_in_slice[self.subject][seg]

                if rot:
                    plot_img = np.rot90(img)
                else:
                    plot_img = img.copy()

                ax.imshow(plot_img, cmap=use_cmap)
                ax.axis('off')
                gs_ix += 1

            if save:
                # line-img already exists in 'func'
                if seg != "line" and seg != "ref":
                    ref_img = nb.load(self.segmentation_df[self.subject]['ref'])
                    out_fn = opj(os.path.dirname(self.segmentation_df[self.subject][seg]), f"{self.subject}_ses-{self.target_session}_acq-beam_desc-{seg}.nii.gz")
                    print(f"Writing {out_fn}")
                    nb.Nifti1Image(img[..., np.newaxis], affine=ref_img.affine, header=ref_img.header).to_filename(out_fn)

