from . import utils, plotting, fitting, image, transform
from kneed import KneeLocator
import matplotlib.pyplot as plt
import matplotlib as mpl
from nilearn.signal import clean
from nilearn.glm.first_level.design_matrix import _cosine_drift
from nitime.timeseries import TimeSeries
from nitime.analysis import SpectralAnalyzer
import numpy as np
import os
import pandas as pd
from scipy import signal
import seaborn as sns
from sklearn import decomposition
from typing import Union
import warnings
from statsmodels.api import stats
from bids import BIDSLayout
import pickle
import nibabel as nb

opj = os.path.join
pd.options.mode.chained_assignment = None # disable warning thrown by string2float
warnings.filterwarnings("ignore")


class Segmentations():
    """Segmentations

    Class to project segmentations created using the pipeline described on https://linescanning.readthedocs.io/en/latest/ to a single slice image of a new session (typically a line-scanning session). By default, it will look for files in the *Nighres*-directory, as these segmentations are generally of most interest. The output of the class will be a pickle-file containing the following segmentations: CRUISE-tissue segmentation (as per the output of https://github.com/gjheij/linescanning/blob/main/shell/spinoza_cortexreconstruction), the layer+depth segmentation (https://github.com/gjheij/linescanning/blob/main/shell/spinoza_layering), the brain mask, tissue probability maps (https://github.com/gjheij/linescanning/blob/main/shell/spinoza_extractregions), the reference slice, and the line as acquired in the session.

    To warp the files, you'll need to specify a forward-transformation matrix (e.g., from *reference session* to *target session*), the reference slice, and the foldover direction (e.g., FH or AP) describing the nature of the applied saturation slabs. You can also specify an earlier pickle file, in which case the segmentations embedded in that file are loaded in for later manipulation with e.g., :func:`pRFline.segmentations.plot_segmentations` to create overview figures.

    Parameters
    ----------
    subject: str
        Subject ID as used in `SUBJECTS_DIR` and used throughout the pipeline
    run: int, optional
        run number you'd like to have the segmentations for
    task: str, optional
        task identifier
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
    voxel_cutoff: int, optional
        When using surface coils, the signal that you pick up drops off rapidly with increasing distance from the coil. With this flag we can set as of which voxel we should ignore signal, which is relevant for selecting of WM/CSF voxels
    verbose: bool, optional
        Print details to the terminal, default is False        
    overwite: bool, optional
        Overwrite existing segmentations file. Default is False

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

    def __init__(
        self, 
        subject, 
        run=None,
        task=None,
        project_home=None, 
        trafo_file=None, 
        reference_slice=None, 
        reference_session=1, 
        target_session=2, 
        foldover="FH", 
        pickle_file=None,
        voxel_cutoff=300,
        shift=0,
        verbose=False,
        **kwargs):

        self.subject            = subject
        self.run                = run
        self.task               = task
        self.project_home       = project_home
        self.trafo_file         = trafo_file
        self.reference_slice    = reference_slice
        self.reference_session  = reference_session
        self.target_session     = target_session
        self.foldover           = foldover
        self.pickle_file        = pickle_file
        self.voxel_cutoff       = voxel_cutoff
        self.shift              = shift
        self.verbose            = verbose
        self.__dict__.update(kwargs)

        # check overwrite key
        if not hasattr(self, "overwrite"):
            self.overwrite = False

        # try default project_home if none is specified
        if not isinstance(self.project_home, str):
            self.project_home = os.environ.get('DIR_DATA_HOME')
            
            if self.project_home == None:
                raise ValueError("Could not read DIR_DATA_DERIV-variable. Please specify the project's root directory (where 'derivatives' lives) with the 'project_home' argument.")

        # define derivatives folder
        self.deriv_dir = opj(self.project_home, "derivatives")

        # specify nighres directory
        self.nighres_source = opj(self.deriv_dir, 'nighres', self.subject, f'ses-{self.reference_session}') 
        self.nighres_target = opj(os.path.dirname(self.nighres_source), f'ses-{self.target_session}') 
        self.mask_dir = opj(self.deriv_dir, 'manual_masks', self.subject, f'ses-{self.reference_session}')
        self.cortex_dir = opj(self.deriv_dir, 'pycortex', self.subject)

        if not isinstance(self.pickle_file, str):
            # check run
            self.base_name = f'{subject}_ses-{self.target_session}'
            if isinstance(self.task, str):
                self.base_name += f"_task-{self.task}"

            if isinstance(self.run, (int,str)):
                self.base_name += f"_run-{self.run}"

            self.pickle_file = opj(self.nighres_target, f'{self.base_name}_desc-segmentations.pkl')
        else:
            comps = utils.split_bids_components(self.pickle_file)
            subject,self.target_session,self.run = comps["sub"],comps["ses"],comps["run"]
            self.nighres_target = os.path.dirname(self.pickle_file)

        if self.overwrite:
            if os.path.exists(self.pickle_file):
                os.remove(self.pickle_file)

        if not os.path.exists(self.pickle_file) or self.overwrite:

            if not os.path.exists(self.nighres_target):
                os.makedirs(self.nighres_target, exist_ok=True)

            # fetch segmentations, assuming default directory layout
            nighres_layout = BIDSLayout(self.nighres_source, validate=False).get(extension=['nii.gz'], return_type='file')
            self.wb_cruise = utils.get_file_from_substring("cruise_cortex", nighres_layout, exclude="1slice")
            self.wb_layers = utils.get_file_from_substring("layering_layers", nighres_layout, exclude="1slice")
            self.wb_depth = utils.get_file_from_substring("layering_depth", nighres_layout, exclude="1slice")

            for n,ii in zip(["cruise","layers","depth"], [self.wb_cruise, self.wb_layers, self.wb_depth]):
                if isinstance(ii, list):
                    if len(ii) > 1:
                        raise ValueError(f"Found multiple files for '{n}': {ii}")
                    else:
                        setattr(self, f"wb_{n}", ii[0])

            # fetch mask and tissue probabilities
            mask_layout = BIDSLayout(self.mask_dir, validate=False).get(extension=['nii.gz'], return_type='file')
            self.wb_wm = utils.get_file_from_substring("label-WM", mask_layout)
            self.wb_gm = utils.get_file_from_substring("label-GM", mask_layout)
            self.wb_csf = utils.get_file_from_substring("label-CSF", mask_layout)
            self.wb_brainmask = utils.get_file_from_substring("brainmask", mask_layout)

            # check if reference slice and transformation file actually exist
            if not os.path.exists(self.reference_slice):
                raise ValueError(f"Could not find reference slice {self.reference_slice}")

            if self.trafo_file == None:
                # try the default in derivatives/pycortex/<subject>/transforms
                self.trafo_file = utils.get_file_from_substring([f"from-ses1_to-ses{self.target_session}_rec-", ".mat"], opj(self.cortex_dir, 'transforms'))
            
            if isinstance(self.trafo_file, str):
                if not os.path.exists(self.trafo_file):
                    raise ValueError(f"Could not find trafo_file {self.trafo_file}")
                invert = 0
            elif isinstance(self.trafo_file, list):
                invert = list(np.zeros(len(self.trafo_file)).astype(int))
    
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
                print(f" Nighres target: {self.nighres_target}")
                print(f" Foldover: {self.foldover}")
                print(f" Ref slice: {self.reference_slice}")
                print(f" Trafo's: {self.trafo_file}")

            in_type = ['prob', 'prob', 'prob', 'tissue', 'layer', 'prob', 'tissue']
            tag = ['wm', 'gm', 'csf', 'cortex', 'layers', 'depth', 'mask']
            in_files = [self.wb_wm, self.wb_gm, self.wb_csf, self.wb_cruise, self.wb_layers, self.wb_depth, self.wb_brainmask]
            self.resampled = {}
            self.resampled_data = {}
            for file,ft,t in zip(in_files, in_type, tag):

                new_fn = utils.replace_string(self.reference_slice, "T1w", f"desc-{t}.nii.gz")
                new_file = opj(self.nighres_target, os.path.basename(new_fn))

                if os.path.exists(new_file):
                    if self.overwrite:
                        os.remove(new_file)

                if not os.path.exists(new_file):
                    if ft == "tissue":
                        interp = "mul"
                    elif ft == "layer":
                        interp = "gen"
                    else:
                        interp = "nn"
                    
                    transform.ants_applytrafo(
                        self.reference_slice, 
                        file, 
                        interp=interp, 
                        invert=invert,
                        trafo=self.trafo_file, 
                        output=new_file
                    )

                # collect them in 'resampled' dictionary
                self.resampled[t] = new_file
                self.resampled_data[t] = nb.load(new_file).get_fdata()

            # add reference and beam images
            self.resampled['ref'] = self.reference_slice; self.resampled_data['ref'] = nb.load(self.reference_slice).get_fdata()
            self.resampled['line'] = image.create_line_from_slice(
                self.reference_slice, 
                fold=self.foldover,
                width=16,
                shift=self.shift)
            self.resampled_data['line'] = nb.load(self.reference_slice).get_fdata()

            # save pickle
            with open(self.pickle_file, "wb") as pf:
                pickle.dump(self.resampled, pf)

        else:
            
            utils.verbose(f" Reading {self.pickle_file}", self.verbose)
            with open(self.pickle_file, 'rb') as pf:
                self.resampled = pickle.load(pf)

            if 'ref' not in list(self.resampled.keys()):
                self.resampled['ref'] = self.reference_slice
                self.resampled['line'] = image.create_line_from_slice(
                    self.reference_slice, 
                    fold=self.foldover,
                    width=16,
                    shift=self.shift)

        self.segmentation_df = {}
        self.segmentation_df[self.subject] = self.resampled.copy()
        
        # get the WM/CSF voxels for regressors
        self.wm_csf_voxels_for_regressors()
        if self.verbose:
            print(f" Found {len(self.acompcor_voxels)} voxel(s) for nuisance regression; (indices<{self.voxel_cutoff} are ignored due to distance from coil)")

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
        
        
    def plot_segmentations(
        self, 
        subj_df=None,
        include=['ref', 'cortex', 'layers'], 
        cmaps=['Greys_r', 'Greys_r', 'hot'], 
        cmap_color_line="#f0ff00", 
        max_val_ref=2400, 
        overlay_line=True, 
        figsize=None, 
        save_as=None,
        **kwargs):

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
        nr_subjects = len(subject_list)
        nr_segmentations = len(include)

        if not isinstance(figsize, tuple):
            figsize = (len(include)*5, nr_subjects*5)

        # if one subject is given, plot segmentations along x-axis (each segmentation a column)
        if len(subject_list) == 1:
            fig,axs = plt.subplots(
                ncols=nr_segmentations, 
                figsize=figsize
            )
        else:
            # if multple segmentations, plot the segmentations as rows with subjects as columns
            fig,axs = plt.subplots(
                ncols=nr_segmentations, 
                nrows=nr_subjects, 
                figsize=figsize
            )

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
                    line = self.get_plottable_segmentations(use_df[sub]['line'])

                    # plot data
                    ax.imshow(np.rot90(line), cmap=beam_cmap, alpha=0.6)

                for tick in ["x","y"]:
                    kwargs = utils.update_kwargs(
                        kwargs,
                        f"{tick}_ticklabels",
                        []
                    )

                kwargs = utils.update_kwargs(
                    kwargs,
                    "title",
                    seg_type,
                    force=True
                )

                plotting.conform_ax_to_obj(
                    ax=ax,
                    **kwargs
                )

        plt.tight_layout()

        if save_as:
            fig.savefig(save_as)

    def plot_line_segmentations(
        self, 
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

                seg = self.get_plottable_segmentations(use_df[sub][ii])
                line = self.get_plottable_segmentations(use_df[sub]['line'])
                
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
            
        self.cortex = self.get_plottable_segmentations(self.segmentation_df[self.subject]['cortex'])
        self.line   = self.get_plottable_segmentations(self.segmentation_df[self.subject]['line'])
        self.mask   = self.get_plottable_segmentations(self.segmentation_df[self.subject]['mask'])

        beam_loc = np.where(self.line>0)
        if self.verbose:
            print(f" Beam location = [{beam_loc[1][0]},{beam_loc[1][-1]+1}]; shift={self.shift}mm")

        if self.foldover == "FH":
            self.beam_ctx   = np.multiply(self.cortex, self.line.astype(bool))[:, beam_loc[1][0]:beam_loc[1][-1]+1]
            self.mask_beam  = np.multiply(self.mask, self.line.astype(bool))[:, beam_loc[1][0]:beam_loc[1][-1]+1]
        else:
            self.beam_ctx   = np.multiply(self.cortex, self.line.astype(bool))[beam_loc[0][0]:beam_loc[0][-1]+1, :]
            self.mask_beam  = np.multiply(self.mask, self.line.astype(bool))[beam_loc[0][0]:beam_loc[0][-1]+1, :]

        self.wm_voxels  = []
        self.csf_voxels = []
        self.gm_voxels  = []
        self.acompcor_voxels = []
        for vox in range(self.beam_ctx.shape[0]):

            if vox >= self.voxel_cutoff:

                # remove outer stuff from beam
                if all(mas == 1 for mas in self.mask_beam[vox,:]):

                    # fetch voxel id's where all 16 voxels across beam are 2
                    csf_vox = all(elem == 0 for elem in self.beam_ctx[vox, :])
                    if csf_vox == True:
                        self.csf_voxels.append(vox)
                        self.acompcor_voxels.append(vox)

                    gm_vox = all(elem == 1 for elem in self.beam_ctx[vox, :])
                    if gm_vox == True:
                        self.gm_voxels.append(vox)                

                    wm_vox = all(elem == 2 for elem in self.beam_ctx[vox,:])
                    if wm_vox == True:
                        self.wm_voxels.append(vox)
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

    def plot_regressor_voxels(
        self, 
        figsize=(8,8), 
        cmap_color=["#338EFF", "#FF4F33"], 
        ax=None, 
        title="WM/CSF voxels for nuisance regression",
        **kwargs
        ):

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

        kwargs = utils.update_kwargs(
            kwargs,
            "title",
            "WM/CSF voxels for nuisance regression"
        )

        for tick in ["x","y"]:
            kwargs = utils.update_kwargs(
                kwargs,
                f"{tick}_ticklabels",
                []
            )

        plotting.conform_ax_to_obj(
            ax=ax, 
            **kwargs
        )

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

    def plot_beam_in_slice(
        self, 
        include='all', 
        figsize=None, 
        cmap_color_mask="#08B2F0", 
        rot=True, 
        save=False,
        imshow_kw={},
        **kwargs
        ):

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

        fig,axs = plt.subplots(
            ncols=len(include_seg), 
            figsize=figsize, 
            constrained_layout=True
        )

        for ix,key in enumerate(include_seg):

            if key == "mask":
                use_cmap = utils.make_binary_cm(cmap_color_mask)
            elif key == "layers":
                use_cmap = "hot"
            else:
                use_cmap = "Greys_r"
            
            val = self.beam_in_slice[self.subject][key]
            if rot:
                plot_img = np.rot90(val)
            else:
                plot_img = val.copy()

            axs[ix].imshow(plot_img, cmap=use_cmap, **imshow_kw)
            for tick in ["x","y"]:
                kwargs = utils.update_kwargs(
                    kwargs,
                    f"{tick}_ticklabels",
                    []
                )

            plotting.conform_ax_to_obj(
                ax=axs[ix], 
                title={
                    "title": key,
                    "fontweight": "bold"
                },
                **kwargs
            )

        if save:
            # line-img already exists in 'func'
            if key not in ["line","seg"]:
                ref_img = nb.load(self.segmentation_df[self.subject]['ref'])
                out_fn = opj(os.path.dirname(val), f"{self.subject}_ses-{self.target_session}_acq-beam_desc-{seg}.nii.gz")
                print(f"Writing {out_fn}")
                nb.Nifti1Image(val[..., np.newaxis], affine=ref_img.affine, header=ref_img.header).to_filename(out_fn)

class Align():

    def __init__(self, moving_object, target_object, verbose=False, weights=None, save_as=None, axs=None,
    **kwargs):

        self.moving_object  = moving_object
        self.target_object  = target_object
        self.verbose        = verbose
        self.weights        = weights
        self.save_as        = save_as
        self.axs            = axs
        self.__dict__.update(kwargs)

        # fetch the segmentations in beam representation
        for obj in self.moving_object, self.target_object:
            try:
                obj.segmentations_to_beam()
            except:
                allowed_types = ["linescanning.dataset.Dataset", "linescanning.segmentations.Segmentations", "linescanning.preproc.aCompCor"]
                raise TypeError(f"Input must be one of {allowed_types}, or linescanning.segmentations.Segmentations, not '{type(obj)}'")

        # get the CRUISE segmentation
        self.moving_cortex  = self.moving_object.segmentations_in_beam[self.moving_object.subject]['cortex']
        self.target_cortex = self.target_object.segmentations_in_beam[self.target_object.subject]['cortex']

        # average across line
        self.moving_cortex_line = self.moving_cortex.mean(axis=-1)
        self.target_cortex_line = self.target_cortex.mean(axis=-1)
        
        # should we include weights?
        if isinstance(self.weights, str):
            
            # take the brainmask from the 'moving' object
            if self.weights == "moving":
                weights_object = self.moving_object
            # take the brainmask from the 'target' object
            elif self.weights == "target":
                weights_object = self.target_object
            else:
                raise NotImplementedError(f"Unknown option '{self.weights}' for weights. Must be one of None, 'moving', or 'target'")

            self.weights = weights_object.segmentations_in_beam[weights_object.subject]['mask'].mean(axis=-1)

        # get the shift, correlations, and more
        self.aligner = self.get_voxel_shift(
            weights=self.weights,
            plot=self.verbose,
            save_as=self.save_as,
            axs=self.axs,
            **kwargs)

        # get GM voxels
        self.moving_gm_voxels = self.get_gm_voxels(self.moving_cortex_line)
        self.target_gm_voxels = self.get_gm_voxels(self.target_cortex_line)

    def get_gm_voxels(self, cortex_img, threshold=1.75):
        return np.where(cortex_img > threshold)[0]
    
    def get_voxel_shift(self,vox_range=[-15,15], plot=False, weights=None, save_as=None, axs=None, **kwargs):
        """get_voxel_shift

        Obtain shift in line direction between 'real' and 'predicted' slice. If the output is positive, it means `moving` need to be shifted X-voxels to the right in order to maximally match `target`.

        Parameters
        ----------
        moving: np.ndarray
            array to be shifted ('moving')
        target: np.ndarray
            array to move to ('reference')
        vox_range: list, optional
            this list represents the possible voxel shifts that are being used to construct a list of correlations. Can be narrower when the arrays are already pretty well aligned. Default = [-15,15]
        plot: bool, optional
            plot the correlation coefficients
        save_as: str, optional
            save the plot (recommended to save as pdf for retaining resolution)

        Returns:
            voxel shift: int
                shift in voxel dimensions, 
            correlations across range: list
                list of correlations across range
            attempted voxel range: list
                the range of voxels used for correlation
        """
        
        corr_vals = []
        range_vox = range(*vox_range)

        if not hasattr(self, "font_size"):
            self.font_size = 18

        if not hasattr(self, "label_size"):
            self.label_size = 14

        if not hasattr(self, "tick_width"):
            self.tick_width = 0.5
        
        if not hasattr(self, "tick_length"):
            self.tick_length = 7

        if not hasattr(self, "axis_width"):
            self.axis_width = 0.5

        # do some checking first
        target = self.target_cortex_line
        moving = self.moving_cortex_line
        for arr in target, moving:
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"Inputs must be numpy arrays (1D), not {type(target)} and {type(moving)}")
            else:
                if arr.ndim > 1:
                    raise ValueError("Inputs must be 1D-arrays")
        
        if isinstance(weights, np.ndarray):
            print("im here")
            if weights.ndim > 1:
                raise ValueError("Weights must be 1D-array")
            else:
                if weights.shape[0] != moving.shape[0] != target.shape[0]:
                    raise ValueError(f"Shape of weights ({weights.shape[0]}) does not match that of input array 1 ({moving.shape[0]}) and/or input array 2 ({target.shape[0]})")

        # loop through the range of voxels
        for ix,ii in enumerate(range_vox):

            padding = np.zeros(abs(ii))
            
            # shift to the right
            if ii > 0:
                padded_moving = np.append(padding, moving)
                padded_moving = padded_moving[:moving.shape[0]]

                # shift the weights with the profiles
                if isinstance(weights, np.ndarray):
                    padded_weights = np.append(padding, weights)
                    padded_weights = padded_weights[:moving.shape[0]]
            
            # shift to the left
            else:
                padded_moving = np.append(moving, padding)
                padded_moving = padded_moving[abs(ii):]

                # shift the weights with the profiles
                if isinstance(weights, np.ndarray):
                    padded_weights = np.append(weights, padding)
                    padded_weights = padded_weights[abs(ii):]

            if isinstance(weights, np.ndarray):
                # use weighted correlation
                corr_array = np.hstack((target[...,np.newaxis], padded_moving[...,np.newaxis]))
                descr = stats.DescrStatsW(corr_array, weights=padded_weights)
                corr = descr.corrcoef[0,1]
            else:
                corr = np.corrcoef(target,padded_moving)[0,1]
                
            corr_vals.append(corr)

        max_corr = np.amax(corr_vals)
        vox_pad_ix = utils.find_nearest(corr_vals, max_corr)[0]
        print(f"max correlation = {max_corr.round(2)}; padding of {range_vox[vox_pad_ix]} voxels")

        if plot:
            fig = plt.figure(figsize=(20,5))
            gs = fig.add_gridspec(1,4, width_ratios=[0.5,0.5,3,1], wspace=0.3)

            # imshow of line-profile fixed image
            ax1 = fig.add_subplot(gs[0])
            ax1.imshow(
                np.tile(self.target_cortex_line[:,np.newaxis],10), 
                cmap='Greys_r', 
                aspect=1/3)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax1.spines[axis].set_linewidth(self.axis_width)

            ax1.set_title("Target", fontsize=self.font_size)
            ax1.axes.get_xaxis().set_visible(False)
            ax1.tick_params(
                width=self.tick_width, 
                length=self.tick_length,
                labelsize=self.label_size)
            ax1.set_ylabel("position", fontsize=self.font_size)
            
            # imshow of line-profile moving image
            ax2 = fig.add_subplot(gs[1])
            ax2.imshow(
                np.tile(self.moving_cortex_line[:, np.newaxis],10),
                cmap='Greys_r', 
                aspect=1/3)
            ax2.set_title("Moving", fontsize=self.font_size)
            ax2.axis('off')

            if isinstance(weights, np.ndarray):
                inputs = [moving, target, weights]
                labels = ['moving array', 'target array', 'mask']
                colors = ["#1B9E77", "#D95F02", '#cccccc']
            else:
                inputs = [moving, target]
                labels = ['moving array', 'target array']
                colors = ["#1B9E77", "#D95F02"]

            ax3 = fig.add_subplot(gs[2])
            plotting.LazyPlot(
                inputs, 
                axs=ax3, 
                color=colors,
                labels=labels, 
                x_label="voxels along line",
                y_label="amplitude (a.u.)",
                title="Line profiles",
                **dict(
                    kwargs, 
                    font_size=self.font_size,
                    label_size=self.label_size,
                    tick_width=self.tick_width,
                    tick_length=self.tick_length,
                    axis_width=self.axis_width))

            # get the shift in line-direction
            ax4 = fig.add_subplot(gs[3])
            plotting.LazyPlot(
                np.array(corr_vals),
                xx=range_vox,
                axs=ax4, 
                color="#D01B47", 
                x_label="<< L   padding    R >>",
                y_label="correlation coefficient",
                title=f"Correlation across range",
                x_lim=vox_range,
                **dict(
                    kwargs, 
                    font_size=self.font_size,
                    label_size=self.label_size,
                    tick_width=self.tick_width,
                    tick_length=self.tick_length,
                    axis_width=self.axis_width))
            ax4.axvline(range_vox[vox_pad_ix], lw=0.5, ls='--', color='k')

            plt.show()

        if save_as:
            fig.savefig(save_as, bbox_inches="tight")

        return range_vox[vox_pad_ix],corr_vals,range_vox

class aCompCor(Segmentations):
    """aCompCor

    _summary_

    Parameters
    ----------
    data: np.ndarray
        Data with the format (voxels,timepoints) on which we need to perfom aCompCor
    run: int, optional
        Run identifier, by default None. Can be useful for filenames of certain outputs
    task: int, optional
        Task identifier, by default None. Can be useful for filenames of certain outputs        
    subject: str, optional
        Full subject identifier (e.g., 'sub-001'), by default None
    wm_voxels: list, optional
        List of voxel IDs that are classified as white matter, by default None. Can be specified if you don't have a line-scanning session and, therefore, no :class:`linescanning.segmentations.Segmentation` object
    csf_voxels: list, optional
        List of voxel IDs that are classified as CSF, by default None. Can be specified if you don't have a line-scanning session and, therefore, no :class:`linescanning.segmentations.Segmentations` object
    n_components: int, optional
        Number of PCA-components to extract for each of the WM/CSF voxels, by default 5
    select_component: int, optional
        Select one particular component to regress out rather than all extracted components, by default None because high-pass filtering the PCAs is much more effective
    filter_confs: float, optional
        Cut-off frequency for high-pass filter of PCA-components that survived the scree-plot, by default None but ~0.2Hz generally leaves in task-related frequencies while removing garbage
    save_ext: str, optional
        Extension to use for saved figures (default = 'pdf')
    save_as: str, optional
        Basename for several output plots/files, by default None. For example, you can save the summary plot of the aCompCor process ('-desc_acompcor.pdf' is appended). Generally a good idea to save in PDF-format, then edit in Inkscape
    summary_plot: bool, optional
        Make the summary plot describing the effect of aCompCor, by default True. Includes an imshow of selected WM/CSF-voxels, the scree-plot for both PCAs (WM+CSF), the power spectra of the surviving components, and the power spectra for the timecourse of voxel 359 (in line-scanning, the middle of the line) for both un-aCompCor'ed and aCompCor'ed data
    TR: float, optional
        Repetition time, by default 0.105. Required for correctly creating power spectra
    verbose: bool, optional
        Print a bunch of details to the terminal, by default False
    ref_slice: str, optional
        Path representing the reference anatomy slice of a particular run, by default None. Required to transform the segmentations to the correct space
    trg_session: int, optional
        Target session, by default None. Required for output names
    trafo_list: str, list, optional
        List or string representing transformation files that need to be applied, by default None.
    foldover: str, optional
        Foldover direction during the line-scanning acquisition, by default "FH". This is to make sure the line is specified correctly when initializing :class:`linescanning.segmentations.Segmentations`
    shift: float, optional
        how many mm the line needs to be shifted in the `foldover` direction.  This may be required if you had a bad shim and you had to move the slice in a particular direction to target the spot you intended. Default = 0

    Raises
    ----------
    ValueError
        When no PCA-components survived the scree-plot

    Returns
    ----------
    self
        The most interesting attribute will be `self.acomp_data`

    Example
    ----------
    >>> from linescanning import preproc
    >>> # data = (voxels,timepoints)
    >>> acomp = preproc.aCompCor(
    >>>     data,
    >>>     subject="sub-003",
    >>>     run=1,
    >>>     trg_session=4,
    >>>     n_components=5,
    >>>     trafo_list=['ses_to_motion.mat', 'run_to_run.mat'],
    >>>     filter_confs=0.2,
    >>>     TR=0.105,
    >>>     verbose=True)
    """
    
    def __init__(
        self, 
        data, 
        run=None,
        task=None,
        subject=None, 
        wm_voxels=None,
        csf_voxels=None,
        n_components=5, 
        select_component=None, 
        filter_confs=None, 
        save_as=None, 
        save_ext="pdf",
        summary_plot=True, 
        TR=0.105, 
        verbose=False, 
        reference_slice=None,
        trg_session=None, 
        trafo_list=None, 
        foldover="FH", 
        shift=0,
        **kwargs):

        self.data               = data
        self.subject            = subject
        self.run                = run
        self.task               = task
        self.wm_voxels          = wm_voxels
        self.csf_voxels         = csf_voxels
        self.n_components       = n_components
        self.select_component   = select_component
        self.filter_confs       = filter_confs
        self.save_as            = save_as
        self.save_ext           = save_ext
        self.summary_plot       = summary_plot
        self.verbose            = verbose
        self.reference_slice    = reference_slice
        self.trafo_list         = trafo_list
        self.trg_session        = trg_session
        self.TR                 = TR
        self.foldover           = foldover
        self.shift              = shift
        self.__dict__.update(kwargs)

        if self.wm_voxels == None and self.csf_voxels == None:
            super().__init__(
                self.subject,
                run=self.run,
                task=self.task,
                reference_slice=self.reference_slice,
                target_session=self.trg_session,
                foldover=self.foldover,
                shift=self.shift,
                verbose=self.verbose,
                trafo_file=self.trafo_list,
                **kwargs)

        utils.verbose(f" Using {self.n_components} components for aCompCor (WM/CSF separately)", self.verbose)

        # initialize some dictionaries
        self.acompcor_components = []
        self.pcas = {}
        self.elbows = {
            "wm": None, 
            "csf": None
        }

        self.tissue_pca = {
            "wm": True, 
            "csf": True
        }

        self.pca_but_timecourses = {
            "wm": False, 
            "csf": False
        }
        
        # run PCA for csf/wm voxels
        for tissue in ['csf', 'wm']:
            
            # check if we got voxels
            self.tissue_voxels = getattr(self, f"{tissue}_voxels")
            self.tissue_tc = utils.select_from_df(
                self.data, 
                expression="ribbon", 
                indices=self.tissue_voxels
            )

            # do PCA if voxels are found
            if len(self.tissue_voxels)>0:
                
                if len(self.tissue_voxels)>self.n_components:
                    self.pca = decomposition.PCA(n_components=self.n_components)
                    self.components = self.pca.fit_transform(self.tissue_tc)

                    # find elbow with KneeLocator
                    self.xx = np.arange(0, self.n_components)
                    self.kn = KneeLocator(
                        self.xx, 
                        self.pca.explained_variance_, 
                        curve='convex', 
                        direction='decreasing'
                    )

                    # this thing will be an integer representing the component number at which inflexion point is
                    self.elbow_ = self.kn.knee
                    if self.elbow_ != None and self.elbow_>0:

                        # only append PCA object if successful
                        self.pcas[tissue] = self.pca

                        # store elbow data
                        self.elbows[tissue] = self.elbow_
                        utils.verbose(f" Found {self.elbow_} component(s) in '{tissue}'-voxels with total explained variance of {round(sum(self.pca.explained_variance_ratio_[:self.elbow_]), 2)}%", self.verbose)

                        self.pca_desc = f"""
Timecourses from these voxels were extracted and fed into a PCA. These components were used to clean the data from respiration/cardiac frequencies. """

                        self.include_components = self.components[:, :self.elbow_]
                        if self.include_components.ndim == 1:
                            self.include_components = self.include_components[..., np.newaxis]

                        self.acompcor_components.append(self.include_components)
                    else:
                        # no PCA
                        self.tissue_pca[tissue] = False
                        self.pca_but_timecourses[tissue] = True
                        utils.verbose(f" PCA for '{tissue}' was unsuccessful. Using all un-PCA'd timecourses ({len(self.tissue_voxels)})", self.verbose)
                        self.pca_desc = f"""
PCA with {self.n_components} was unsuccessful, so '{tissue}' timecourses were used to clean the data from respiration/cardiac 
frequencies. """
                    self.acompcor_components.append(self.tissue_tc.mean(axis=1)[:,np.newaxis])
                else:
                    # no PCA
                    self.tissue_pca[tissue] = False
                    self.pca_but_timecourses[tissue] = True
                    utils.verbose(f" Number of voxels ({len(self.tissue_voxels)}) < number of requested components ({self.n_components}). Using all un-PCA'd timecourses ({len(self.tissue_voxels)})", self.verbose)
                    self.pca_desc = f"""
PCA with {self.n_components} was unsuccessful, so '{tissue}' timecourses were used to clean the data from respiration/cardiac 
frequencies. """
                    self.acompcor_components.append(self.tissue_tc.mean(axis=1)[:,np.newaxis])

            else:
                # no voxels found..
                utils.verbose(f" PCA for '{tissue}' was unsuccessful because no voxels were found", self.verbose)
                self.pca_desc = f"""
No voxels for '{tissue}' were found, so PCA was skipped. """

                self.tissue_pca[tissue] = False

        if len(self.acompcor_components)>0:
            # concatenate components into an array
            self.acompcor_components = np.concatenate(self.acompcor_components, axis=1)
            
            # get frequency spectra for components
            self.nuisance_spectra, self.nuisance_freqs = [], []
            for ii in range(self.acompcor_components.shape[-1]):
                self.freq_, self.power_ = get_freq(self.acompcor_components[:, ii], TR=TR, spectrum_type="fft")
                self.nuisance_spectra.append(self.power_)
                self.nuisance_freqs.append(self.freq_)

            # regress components out
            if self.select_component == None:
                self.confs = self.acompcor_components
            else:
                utils.verbose(f" Only regressing out component {select_component}", self.verbose)
                self.confs = self.acompcor_components[:, self.select_component-1]

            if self.filter_confs != None:
                utils.verbose(f" DCT high-pass filter on components [removes low frequencies <{filter_confs} Hz]", self.verbose)
                if self.confs.ndim >= 2:
                    self.confs, _ = highpass_dct(
                        self.confs.T, 
                        self.filter_confs, 
                        TR=self.TR
                    )
                    self.confs = self.confs.T
                else:
                    self.confs, _ = highpass_dct(
                        self.confs, 
                        self.filter_confs, 
                        TR=self.TR
                    )

            # outputs (timepoints, voxels) array (RegressOut is also usable, but this is easier in linescanning.dataset.Dataset)
            self.acomp_data = clean(
                self.data.values, 
                standardize=False, 
                confounds=self.confs
            ).T

            self.__desc__ = """
    Nighres segmentations were transformed to the individual slices of each run using antsApplyTransforms with `MultiLabel` interpolation. 
    White matter and CSF voxels were selected based on the CRUISE-segmentation if all voxels across the beam (in *phase-enconding* 
    direction) were assigned to this tissue type. This limited the possibility for partial voluming. """

            self.__desc__ += self.pca_desc
            
            # make summary plot of aCompCor effect
            if self.summary_plot:
                self.summary()

        else:
            self.acomp_data = self.data.values.copy().T
            utils.verbose(f" aCompCor unsuccessful, setting original data as 'acomp_data'", self.verbose)

            self.__desc__ = """No aCompCor was applied because no valid voxels could be found."""

    def summary(self, **kwargs):

        if self.tissue_pca["wm"] or self.tissue_pca["csf"]:
            ncols = 4
        else:
            ncols = 3

        fig,axs = plt.subplots(
            figsize=(24,7),
            ncols=ncols,
            constrained_layout=True
        )
        
        self.plot_regressor_voxels(ax=axs[0])
        if not hasattr(self, 'line_width'):
            line_width = 2

        if hasattr(self, "regressor_voxel_colors"):
            use_colors = self.regressor_voxel_colors
            use_colors = {
                "csf": self.regressor_voxel_colors[0],
                "wm": self.regressor_voxel_colors[1]
            }
        else:
            use_colors = {
                "csf": "#cccccc",
                "wm": "#cccccc"
            }

        if self.tissue_pca["wm"] or self.tissue_pca["csf"]:
            if self.tissue_pca["wm"] and self.tissue_pca["csf"]:
                label = ["csf", "wm"]
                colors = self.regressor_voxel_colors
            elif self.tissue_pca["wm"] and not self.tissue_pca["csf"]:
                label = ["wm"]
                colors = use_colors["wm"]
            elif self.tissue_pca["csf"] and not self.tissue_pca["wm"]:
                label = ["csf"]
                colors = use_colors["csf"]

            # make dashed line for each tissue PCA
            for ix,(key,val) in enumerate(self.elbows.items()):
                if val != None:
                    axs[1].axvline(
                        val, 
                        color=use_colors[key], 
                        ls='dashed', 
                        lw=0.5,
                        alpha=0.5
                    )

            pca_list = [val.explained_variance_ratio_ for key,val in self.pcas.items()]
            plotting.LazyPlot(
                pca_list,
                xx=self.xx,
                color=colors,
                axs=axs[1],
                title=f"Scree-plot run-{self.run}",
                x_label="nr of components",
                y_label="variance explained (%)",
                labels=label,
                line_width=line_width,
                **kwargs)

        # create dashed line on cut-off frequency if specified
        if self.filter_confs != None:
            add_vline = {
                'pos': self.filter_confs, 
                'color': 'k',
                'ls': 'dashed', 
                'lw': 0.5
            }
        else:
            add_vline = None
        
        if ncols>3:
            ax2 = axs[2]
        else:
            ax2 = axs[1]
            
        plotting.LazyPlot(
            self.nuisance_spectra,
            xx=self.nuisance_freqs[0],
            axs=ax2,
            labels=[f"component {ii+1}" for ii in range(self.acompcor_components.shape[-1])],
            title=f"Power spectra",
            x_label="frequency (Hz)",
            y_label="power (a.u.)",
            x_lim=[0, 1.5],
            line_width=line_width,
            add_vline=add_vline,
            **kwargs)

        # plot power spectra from non-aCompCor'ed vs aCompCor'ed data
        tc1 = utils.select_from_df(
            self.data, 
            expression='ribbon', 
            indices=self.gm_voxels
        ).mean(axis=1).values

        tc2 = self.acomp_data[self.gm_voxels,:].mean(axis=0)

        if not hasattr(self, "clip_power"):
            clip_power = 100
        
        # larger than 3 means PCA successful
        if ncols>3:
            ax3 = axs[3]
        else:
            ax3 = axs[2]

        freqs = {}
        for tag,tc in zip(
            ['no aCompCor','aCompCor'],
            [tc1,tc2]):

            freqs[tag] = get_freq(
                tc, 
                TR=self.TR, 
                spectrum_type='fft', 
                clip_power=clip_power
            )

        freq_list = [hz[1] for _,hz in freqs.items()]
        power_axis = freqs[list(freqs.keys())[0]][0]
        plotting.LazyPlot(
            freq_list,
            xx=power_axis,
            color=["#1B9E77", "#D95F02"],
            x_label="frequency (Hz)",
            y_label="power (a.u.)",
            title="Power spectra of average GM-voxels",
            labels=list(freqs.keys()),
            axs=ax3,
            x_lim=[0,1.5],
            line_width=2,
            **kwargs
        )

        # add insets with non-cleaned vs cleaned timeseries of average gm-voxels
        x_axis = np.array(list(np.arange(0,tc1.shape[0])*self.TR))
        left, bottom, width, height = [0.2, 0.5, 0.8, 0.4]
        inset = ax3.inset_axes([left, bottom, width, height])
        plotting.LazyPlot(
            [tc1, tc2],
            xx=x_axis,
            color=["#1B9E77", "#D95F02"],
            y_label="magnitude",
            axs=inset,
            add_hline=0,
            x_lim=[0,x_axis[-1]],
            font_size=14,
            **kwargs
        )
            
        inset.set_xticks([])
        sns.despine(
            offset=None, 
            bottom=True, 
            ax=inset
        )

        if self.save_as != None:
            self.base_name = self.subject
            if isinstance(self.trg_session, (str,float,int)):
                self.base_name += f"_ses-{self.trg_session}"

            if isinstance(self.task, str):
                self.base_name += f"_task-{self.task}"         

            fname = opj(self.save_as, f"{self.base_name}_run-{self.run}_desc-acompcor.{self.save_ext}")
            fig.savefig(
                fname, 
                bbox_inches='tight',
                dpi=300
            )

class RegressOut():

    def __init__(self, data, regressors, **kwargs):
        """RegressOut

        Class to regress out nuisance regressors from data.

        Parameters
        ----------
        data: pandas.DataFrame, numpy.ndarray
            Input data to be regressed
        regressors: pandas.DataFrame, numpy.ndarray
            Data to be regressed out

        Raises
        ----------
        ValueError
            If shapes of input data and regressors are not compatible
        """
        self.data = data
        self.regressors = regressors
        
        # add index back if input is dataframe
        self.add_index = False

        if self.data.shape[0] != self.regressors.shape[0]:
            raise ValueError(f"Shape of data ({self.data.shape}) does not match shape of confound array ({self.regressors.shape})")

        if isinstance(self.data, pd.DataFrame):
            self.add_index = True
            self.data_array = self.data.values
        else:
            self.data_array = self.data.copy()
        
        if isinstance(self.regressors, pd.DataFrame):
            self.regressors_array = self.regressors.values
        else:
            self.regressors_array = self.regressors.copy()
        
        self.clean_array = clean(
            self.data_array, 
            standardize=False, 
            confounds=self.regressors_array,
            **kwargs
        )

        if self.add_index:
            self.clean_df = pd.DataFrame(self.clean_array, index=self.data.index, columns=self.data.columns)


def highpass_dct(
    func, 
    lb=0.01, 
    TR=0.105, 
    modes_to_remove=None,
    remove_constant=False,
    ):
    """highpass_dct

    Discrete cosine transform (DCT) is a basis set of cosine regressors of varying frequencies up to a filter cutoff of a specified number of seconds. Many software use 100s or 128s as a default cutoff, but we encourage caution that the filter cutoff isn't too short for your specific experimental design. Longer trials will require longer filter cutoffs. See this paper for a more technical treatment of using the DCT as a high pass filter in fMRI data analysis (https://canlab.github.io/_pages/tutorials/html/high_pass_filtering.html).

    Parameters
    ----------
    func: np.ndarray
        <n_voxels, n_timepoints> representing the functional data to be fitered
    lb: float, optional
        cutoff-frequency for low-pass (default = 0.01 Hz)
    TR: float, optional
        Repetition time of functional run, by default 0.105
    modes_to_remove: int, optional
        Remove first X cosines
        
    Returns
    ----------
    dct_data: np.ndarray
        array of shape(n_voxels, n_timepoints)
    cosine_drift: np.ndarray 
        Cosine drifts of shape(n_scans, n_drifts) plus a constant regressor at cosine_drift[:, -1]

    Notes
    ----------
    * *High-pass* filters remove low-frequency (slow) noise and pass high-freqency signals. 
    * Low-pass filters remove high-frequency noise and thus smooth the data.  
    * Band-pass filters allow only certain frequencies and filter everything else out
    * Notch filters remove certain frequencies
    """

    # Create high-pass filter and clean
    n_vol = func.shape[-1]
    st_ref = 0  # offset frametimes by st_ref * tr
    ft = np.linspace(st_ref * TR, (n_vol + st_ref) * TR, n_vol, endpoint=False)
    hp_set = _cosine_drift(lb, ft)

    # select modes
    if isinstance(modes_to_remove, int):
        hp_set[:,:modes_to_remove]
    else:
        # remove constant column
        if remove_constant:
            hp_set = hp_set[:,:-1]

    dct_data = clean(func.T, detrend=False, standardize=False, confounds=hp_set).T
    return dct_data, hp_set

def lowpass_savgol(
    func, 
    window_length=7, 
    polyorder=3,
    ax=-1,
    verbose=False,
    **kwargs):

    """lowpass_savgol

    The Savitzky-Golay filter is a low pass filter that allows smoothing data. To use it, you should give as input parameter of the function the original noisy signal (as a one-dimensional array), set the window size, i.e. n° of points used to calculate the fit, and the order of the polynomial function used to fit the signal. We might be interested in using a filter, when we want to smooth our data points; that is to approximate the original function, only keeping the important features and getting rid of the meaningless fluctuations. In order to do this, successive subsets of points are fitted with a polynomial function that minimizes the fitting error.

    The procedure is iterated throughout all the data points, obtaining a new series of data points fitting the original signal. If you are interested in knowing the details of the Savitzky-Golay filter, you can find a comprehensive description [here](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter).

    Parameters
    ----------
    func: np.ndarray
        <n_voxels, n_timepoints> representing the functional data to be fitered
    window_length: int
        Length of window to use for filtering. Must be an uneven number according to the scipy-documentation (default = 7)
    poly_order: int
        Order of polynomial fit to employ within `window_length`. Default = 3

    Returns
    ----------
    np.ndarray:
        <n_voxels, n_timepoints> from which high-frequences have been removed

    Notes
    ----------
    * High-pass filters remove low-frequency (slow) noise and pass high-freqency signals. 
    * *Low-pass* filters remove high-frequency noise and thus smooth the data.  
    * Band-pass filters allow only certain frequencies and filter everything else out
    * Notch filters remove certain frequencies            
    """

    if window_length % 2 == 0:
        raise ValueError(f"Window-length must be uneven; not {window_length}")
    
    utils.verbose(f"Window_length = {window_length} | poly order = {polyorder}", verbose)
    return signal.savgol_filter(
        func, 
        window_length, 
        polyorder, 
        axis=ax,
        **kwargs
    )

class Freq():

    def __init__(self, func, *args, **kwargs) -> None:
        self.func = func
        self.freq = get_freq(self.func, *args, **kwargs)

    def plot_timecourse(self, **kwargs):
        plotting.LazyPlot(
            self.func,
            x_label="volumes",
            y_label="amplitude",
            **kwargs)  
        
    def plot_freq(self, **kwargs):
        plotting.LazyPlot(
            self.freq[1],
            xx=self.freq[0],
            x_label="frequency (Hz)",
            y_label="power (a.u.)",
            **kwargs)  
        
def get_freq(func, TR=0.105, spectrum_type='fft', clip_power=None):
    """get_freq

    Create power spectra of input timeseries with the ability to select implementations from `nitime`. Fourier transform is implemented as per J. Siero's implementation.

    Parameters
    ----------
    func: np.ndarray
        Array of shape(timepoints,) 
    TR: float, optional
        Repetition time, by default 0.105
    spectrum_type: str, optional
        Method for extracting power spectra, by default 'psd'. Must be one of 'mtaper', 'fft', 'psd', or 'periodogram', as per `nitime`'s implementations. 
    clip_power: _type_, optional
        _description_, by default None

    Returns
    ----------
    freq
        numpy.ndarray representing the
    power
        numpy.ndarray representing the power spectra

    Raises
    ----------
    ValueError
        If invalid spectrum_type is given. Must be one of `psd`, `mtaper`, `fft`, or `periodogram`.
    """
    if spectrum_type != "fft":
        TC      = TimeSeries(np.asarray(func), sampling_interval=TR)
        spectra = SpectralAnalyzer(TC)

        if spectrum_type == "psd":
            selected_spectrum = spectra.psd
        elif spectrum_type == "fft":
            selected_spectrum = spectra.spectrum_fourier
        elif spectrum_type == "periodogram":
            selected_spectrum = spectra.periodogram
        elif spectrum_type == "mtaper":
            selected_spectrum = spectra.spectrum_multi_taper
        else:
            raise ValueError(f"Requested spectrum was '{spectrum_type}'; available options are: 'psd', 'fft', 'periodogram', or 'mtaper'")
        
        freq,power = selected_spectrum[0],selected_spectrum[1]
        if spectrum_type == "fft":
            power[power < 0] = 0

        if clip_power != None:
            power[power > clip_power] = clip_power

        return freq,power

    else:

        freq    = np.fft.fftshift(np.fft.fftfreq(func.shape[0], d=TR))
        power   = np.abs(np.fft.fftshift(np.fft.fft(func)))**2/func.shape[0]

        if clip_power != None:
            power[power>clip_power] = clip_power

        return freq, power

class ICA():
        
    """ICA

    Wrapper around :class:`sklearn.decomposition.FastICA`, with a few visualization options. The basic input needs to be a `pandas.DataFrame` or `numpy.ndarray` describing a 2D dataset (e.g., the output of :class:`linescanning.dataset.Dataset` or :class:`linescanning.dataset.ParseFuncFile`). :function:`linescanning.preproc.ICA.summary()` outputs a figure describing the effects of ICA denoising on the input data (similar to :function:`linescanning.preproc.aCompCor.summary()`). 

    Parameters
    ----------
    subject: str, optional
        Subject ID to use when saving figures (e.g., `sub-001`)
    data: Union[pd.DataFrame,np.ndarray]
        Dataset to be ICA'd in the format if `<time,voxels>`
    n_components: int, optional
        Number of components to use, by default 10
    filter_confs: float, optional
        Specify a high-pass frequency cut off to retain task-related frequencies, by default 0.02. If you do not want to high-pass filter the components, set `filter_confs=None` and `keep_comps` to the the components you want to retain (e.g., `keep_comps=[0,1]` to retain the first two components)
    keep_comps: list, optional
        Specify a list of components to keep from the data, rather than all high-pass components. If `filter_confs` == None, but `keep_comps` is given, no high-pass filtering is applied to the components. If `filter_confs` & `keep_comps` == None, an error will be thrown. You must either specify `filter_confs` and/or `keep_comps`
    verbose: bool, optional
        Turn on verbosity; prints some stuff to the terminal, by default False
    TR: float, optional
        Repetition time or sampling rate, by default 0.105
    save_as: str, optional
        Path pointing to the location where to save the figures. `sub-<subject>_run-{self.run}_desc-ica.{self.save_ext}"), by default None
    session: int, optional
        Session ID to use when saving figures (e.g., `1`), by default 1
    run: int, optional
        Run ID to use when saving figures (e.g., `1`), by default 1
    summary_plot: bool, optional
        Make a figure regarding the efficacy of the ICA denoising, by default False
    melodic_plot: bool, optional
        Make a figure regarding the information about the components themselves, by default False
    ribbon: tuple, optional
        Range of gray matter voxels. If `None`, we'll check the efficacy of ICA denoising over the average across the data, by default None
    save_ext: str, optional
        Extension to use when saving figures, by default "svg"

    Example
    ----------
    >>> from linescanning.preproc import ICA
    >>> ica_obj = ICA(
    >>>     data_obj.hp_zscore_df,
    >>>     subject=f"sub-{sub}",
    >>>     session=ses,
    >>>     run=3,
    >>>     n_components=10,
    >>>     TR=data_obj.TR,
    >>>     filter_confs=0.18,
    >>>     keep_comps=1,
    >>>     verbose=True,
    >>>     ribbon=None
    >>> )
    >>> # regress components from data. if `filter_confs` & `keep_comps` != None, we'll take the high-passed `keep_comps`components
    >>> # if `filter_confs` == None, but `keep_comps` is given, no high-pass filtering is applied to the components
    >>> # 
    >>> ica_obj.regress()
    """

    def __init__(
        self,
        data:Union[pd.DataFrame,np.ndarray],
        subject:str=None,
        ses:int=None,
        task:str=None,
        n_components:int=10,
        filter_confs:float=0.02,
        keep_comps:Union[int,list,tuple]=[0,1],
        verbose:bool=False,
        TR:float=0.105,
        save_as:str=None,
        session:int=1,
        run:int=1,
        summary_plot:bool=False,
        melodic_plot:bool=False,
        ribbon:Union[list,tuple]=None,
        save_ext:str="svg",
        **kwargs):
        
        self.subject        = subject
        self.ses            = ses
        self.task           = task
        self.data           = data
        self.n_components   = n_components
        self.filter_confs   = filter_confs
        self.verbose        = verbose
        self.TR             = TR
        self.save_as        = save_as
        self.run            = run
        self.summary_plot   = summary_plot
        self.melodic_plot   = melodic_plot
        self.gm_voxels      = ribbon
        self.save_ext       = save_ext
        self.keep_comps     = keep_comps
        self.__dict__.update(kwargs)
        
        self.__desc__ = f"""Sklearn's implementation of 'FastICA' was used to decompose the signal into {self.n_components} components."""

        # sort out gm_voxels format
        if isinstance(self.gm_voxels, tuple):
            self.gm_voxels = list(np.arange(*self.gm_voxels))

        # check data format
        if isinstance(self.data, pd.DataFrame):
            self.add_index = True
            self.data_array = self.data.values
        else:
            self.data_array = self.data.copy()

        # initiate ica
        self.ica = decomposition.FastICA(n_components=self.n_components)
        self.S_ = self.ica.fit_transform(self.data)
        self.A_ = self.ica.mixing_
        
        # transform the sources back to the mixed data (apply mixing matrix)
        self.I_ = np.dot(self.S_, self.A_.T)

        if self.filter_confs != None:
            utils.verbose(f" DCT high-pass filter on components [removes low frequencies <{self.filter_confs} Hz]", self.verbose)

            if self.S_.ndim >= 2:
                self.S_filt, _ = highpass_dct(self.S_.T, self.filter_confs, TR=self.TR)
                self.S_filt = self.S_filt.T
            else:
                self.S_filt, _ = highpass_dct(self.S_, self.filter_confs, TR=self.TR)
        
        # results from ICA
        if self.melodic_plot:
            self.melodic()

    def regress(self):

        if isinstance(self.keep_comps, int):
            self.keep_comps = [self.keep_comps]
        elif isinstance(self.keep_comps, tuple):
            self.keep_comps = list(np.arange(*self.keep_comps))
            
        if isinstance(self.keep_comps, list):

            if len(self.keep_comps) > self.S_.shape[-1]:
                raise ValueError(f"Length of 'keep_comps' is larger ({len(self.keep_comps)}) than number of components ({self.S_.shape[-1]})")

            utils.verbose(f" Keeping components: {self.keep_comps}", self.verbose)

            if self.filter_confs != None:
                use_data = self.S_filt.copy()
            else:
                use_data = self.S_.copy()

            self.confounds = use_data[:,[i for i in range(use_data.shape[-1]) if i not in self.keep_comps]]
        else:
            if self.filter_confs == None:
                raise ValueError("Not sure what to do. Please specify either list of components to keep (e.g., 'keep_comps=[1,2]' or specify a high-pass cut off frequency (e.g., 'filter_confs=0.18')")

            self.__desc__ += f"""Resulting components from the ICA were high-pass filtered using discrete cosine sets (DCT) with a cut off frequency of {self.filter_confs} Hz."""
            # this is pretty hard core: regress out all high-passed components
            utils.verbose(f" Regressing out all high-passed components [>{self.filter_confs} Hz]", self.verbose)
            self.confounds = self.S_filt.copy()

        # outputs (timepoints, voxels) array (RegressOut is also usable, but this is easier in linescanning.dataset.Dataset)
        self.ica_data = clean(self.data.values, standardize=False, confounds=self.confounds).T

        # make summary plot of aCompCor effect
        if self.summary_plot:
            self.summary()

    def summary(self, **kwargs):

        """Create a plot containing the power spectra of all components, the power spectra of the average GM-voxels (or all voxels, depending on the presence of `gm_voxels` before and after ICA, as well as the averaged timecourses before and after ICA"""
        
        if not hasattr(self, 'line_width'):
            self.line_width = 2

        # initiate figure
        fig = plt.figure(figsize=(24, 6))
        gs = fig.add_gridspec(ncols=3, width_ratios=[30,30,100])
        ax1 = fig.add_subplot(gs[0])

        # collect power spectra
        self.freqs = []
        for ii in range(self.n_components):

            # freq
            tc = self.S_[:,ii]
            tc_freq = get_freq(tc, TR=self.TR, spectrum_type='fft')

            # append
            self.freqs.append(tc_freq)

        # create dashed line on cut-off frequency if specified
        if self.filter_confs != None:
            add_vline = {'pos': self.filter_confs}
        else:
            add_vline = None            

        plotting.LazyPlot(
            [self.freqs[ii][1] for ii in range(self.n_components)],
            xx=self.freqs[ii][0],
            x_label="frequency (Hz)",
            y_label="power (a.u.)",
            cmap="inferno",
            title="ICA components",
            axs=ax1,
            x_lim=[0,1.5],
            add_vline=add_vline,
            line_width=self.line_width)  
        
        # plot power spectra from non-aCompCor'ed vs aCompCor'ed data
        if isinstance(self.gm_voxels, (tuple,list)):
            tc1 = utils.select_from_df(self.data, expression='ribbon', indices=self.gm_voxels).mean(axis=1).values
            tc2 = self.ica_data[self.gm_voxels,:].mean(axis=0)
            txt = "GM-voxels"
        else:
            tc1 = self.data.values.mean(axis=-1)
            tc2 = self.ica_data.mean(axis=0)
            txt = "all voxels"
        
        ax2 = fig.add_subplot(gs[1])
        # add insets with power spectra
        tc1_freq = get_freq(tc1, TR=self.TR, spectrum_type='fft')
        tc2_freq = get_freq(tc2, TR=self.TR, spectrum_type='fft')

        plotting.LazyPlot(
            [tc1_freq[1],tc2_freq[1]],
            xx=tc1_freq[0],
            color=["#1B9E77", "#D95F02"],
            x_label="frequency (Hz)",
            title="Average ribbon-voxels",
            labels=['no ICA', 'ICA'],
            axs=ax2,
            line_width=2,
            x_lim=[0,1.5],
            **kwargs)

        x_axis = np.array(list(np.arange(0,tc1.shape[0])*self.TR))
        ax3 = fig.add_subplot(gs[2])
        plotting.LazyPlot(
            [tc1,tc2],
            xx=x_axis,
            color=["#1B9E77", "#D95F02"],
            x_label="time (s)",
            y_label="magnitude",
            title=f"Timeseries of average {txt} ICA'd vs non-ICA'd",
            labels=['no ICA', 'ICA'],
            axs=ax3,
            line_width=2,
            x_lim=[0,x_axis[-1]],
            **kwargs)

        if self.save_as != None:
            self.base_name = self.subject
            if isinstance(self.ses, (str,float,int)):
                self.base_name += f"_ses-{self.ses}"

            if isinstance(self.task, str):
                self.base_name += f"_task-{self.task}" 

            fname = opj(self.save_as, f"{self.base_name}_run-{self.run}_desc-ica.{self.save_ext}")
            utils.verbose(f" Writing {fname}", self.verbose)
            fig.savefig(
                fname, 
                bbox_inches="tight", 
                dpi=300, 
                facecolor="white")

    def melodic(
        self, 
        color:Union[str,tuple]="#6495ED", 
        zoom_freq:bool=False,
        task_freq:float=0.05,
        zoom_lim:list=[0,0.5],
        plot_comps:int=10,
        **kwargs):

        """melodic

        Plot information about the components from the ICA. For each component until `plot_comps`, plot the 2D spatial profile of the component, its timecourse, and its power spectrum. If `zoom_freq=True`, we'll add an extra subplot next to the power spectrum which contains a zoomed in version of the power spectrum with `zoom_lim` as limits.

        Parameters
        ----------
        color: Union[str,tuple], optional
            Color for all subplots, by default "#6495ED"
        zoom_freq: bool, optional
            Add a zoomed in version of the power spectrum, by default False
        task_freq: float, optional
            If `zoom_freq=True`, add a vertical line where the *task-frequency* (`task_freq`) should be, by default 0.05
        zoom_lim: list, optional
            Limits for the zoomed in power spectrum, by default [0,0.5]
        plot_comps: int, optional
            Limit the number of plots being produced in case you have a lot of components, by default 10

        Example
        ----------
        >>> ica_obj.melodic(
        >>>     # color="r",
        >>>     zoom_freq=True, 
        >>>     zoom_lim=[0,0.25])
        """

        # check how many components to plot
        if plot_comps >= self.n_components:
            plot_comps = self.n_components
            
        # initiate figure
        fig = plt.figure(figsize=(24, plot_comps*6), constrained_layout=True)
        subfigs = fig.subfigures(nrows=plot_comps, hspace=0.4, wspace=0)    

        # get plotting defaults
        self.defaults = plotting.Defaults()

        for comp in range(plot_comps):
            
            # make subfigure for each component
            if zoom_freq:
                axs = subfigs[comp].subplots(ncols=4, gridspec_kw={'width_ratios': [0.3,1,0.3,0.2], "wspace": 0.3})
            else:
                axs = subfigs[comp].subplots(ncols=3, gridspec_kw={'width_ratios': [0.3,1,0.3], 'wspace': 0.2})

            # axis for spatial profile
            ax_spatial = axs[0]

            vox_ticks = [0,self.A_.shape[0]//2,self.A_.shape[0]]
            plotting.LazyPlot(
                self.A_[:,comp],
                color=color,
                x_label="voxels",
                y_label="magnitude",
                title="spatial profile",
                axs=ax_spatial,
                line_width=2,
                add_hline=0,
                x_ticks=vox_ticks,
                **kwargs)

            # axis for timecourse of component
            ax_tc = axs[1]

            tc = self.S_[:,comp]
            x_axis = np.array(list(np.arange(0,tc.shape[0])*self.TR))
            plotting.LazyPlot(
                tc,
                xx=x_axis,
                color=color,
                x_label="time (s)",
                y_label="magnitude",
                title="timecourse",
                axs=ax_tc,
                line_width=2,
                x_lim=[0,x_axis[-1]],
                add_hline=0,
                **kwargs)

            # axis for power spectra of component
            ax_freq = axs[2]
            
            # get frequency/power
            freq = get_freq(tc, TR=self.TR, spectrum_type="fft")

            plotting.LazyPlot(
                freq[1],
                xx=freq[0],
                color=color,
                x_label="frequency (Hz)",
                y_label="power (a.u.)",
                title="power spectra",
                axs=ax_freq,
                line_width=2,
                x_lim=[0,1/(2*self.TR)],
                **kwargs)

            if zoom_freq:
                # axis for power spectra of component
                ax_zoom = axs[3]
                plotting.LazyPlot(
                    freq[1],
                    xx=freq[0],
                    color=color,
                    x_label="frequency (Hz)",
                    title="zoomed in",
                    axs=ax_zoom,
                    line_width=2,
                    x_lim=zoom_lim,
                    add_vline={
                        "pos": task_freq,
                        "color": "r",
                        "lw": 2},
                    x_ticks=zoom_lim,
                    sns_left=True,
                    **kwargs)

            subfigs[comp].suptitle(f"component {comp+1}", fontsize=self.defaults.font_size*1.4, y=1.02)

        fig.suptitle("Independent component analysis (ICA)", fontsize=self.defaults.font_size*1.8, y=1.02)

        plt.tight_layout()

        if self.save_as != None:
            self.base_name = self.subject
            if isinstance(self.ses, (str,float,int)):
                self.base_name += f"_ses-{self.ses}"

            if isinstance(self.task, str):
                self.base_name += f"_task-{self.task}" 

            fname = opj(self.save_as, f"{self.base_name}_run-{self.run}_desc-melodic.{self.save_ext}")
            utils.verbose(f" Writing {fname}", self.verbose)
            fig.savefig(
                fname, 
                bbox_inches="tight", 
                dpi=300, 
                facecolor="white")

class DataFilter():

    def __init__(
        self, 
        func,
        **kwargs
        ):

        # filter data based on present identifiers (e.g., task/run)
        self.func = func
        self.filter_input(**kwargs)

    def filter_runs(
        self,
        df_func,
        **kwargs
        ):

        # loop through runs
        self.run_ids = utils.get_unique_ids(df_func, id="run")
        # print(f"task-{task}\t| runs = {run_ids}")
        run_df = []
        for run in self.run_ids:
            
            expr = f"run = {run}"
            run_func = utils.select_from_df(df_func, expression=expr)

            # get regresss
            df = self.single_filter(
                run_func,
                **kwargs
            )

            run_df.append(df)

        run_df = pd.concat(run_df)

        return run_df
    
    def filter_tasks(
        self,
        df_func,
        **kwargs
        ):

        # read task IDs
        self.task_ids = utils.get_unique_ids(df_func, id="task")

        # loop through task IDs
        task_df = [] 
        for task in self.task_ids:

            # extract task-specific dataframes
            expr = f"task = {task}"
            task_func = utils.select_from_df(df_func, expression=expr)

            df = self.filter_runs(
                task_func,
                **kwargs
            )

            task_df.append(df)


        return pd.concat(task_df)
    
    def filter_subjects(
        self,
        df_func,
        **kwargs
        ):

        self.sub_ids = utils.get_unique_ids(df_func, id="subject")
        
        # loop through subject IDs
        sub_df = [] 
        for sub in self.sub_ids:
                
            # extract task-specific dataframes
            expr = f"subject = {sub}"
            self.sub_func = utils.select_from_df(df_func, expression=expr)

            try:
                self.task_ids = utils.get_unique_ids(self.sub_func, id="task")
            except:
                self.task_ids = None

            if isinstance(self.task_ids, list):
                ffunc = self.filter_tasks
            else:
                ffunc = self.filter_runs

            sub_filt = ffunc(
                self.sub_func, 
                **kwargs
            )

            sub_df.append(sub_filt)

        sub_df = pd.concat(sub_df)

        return sub_df
    
    def filter_input(self, **kwargs):

        self.df_filt = self.filter_subjects(
            self.func,
            **kwargs
        )

    @classmethod
    def single_filter(
        self, 
        func, 
        filter_strategy="hp", 
        hp_kw={},
        lp_kw={},
        **kwargs
        ):
        
        allowed_lp = ["lp","lowpass","low-pass","low_pass"]
        allowed_hp = ["hp","highpass","high-pass","high_pass"]

        if isinstance(filter_strategy, str):
            filter_strategy = [filter_strategy]
        
        use_kws = {
            "hp": hp_kw,
            "lp": lp_kw
        }

        use_df = func
        for ix,strat in enumerate(filter_strategy):

            if strat in allowed_hp:
                ffunc = highpass_dct
                kws = use_kws["hp"]
            elif strat in allowed_lp:
                ffunc = lowpass_savgol
                kws = use_kws["lp"]
            else:
                raise ValueError(f"Unknown option '{strat}'. Must be one of {allowed_hp} for high-pass filtering or one of {allowed_lp} for low-pass filtering")
            
            # input dataframe will be <time,voxels>; for filter functions, this should be transposed
            filt_data = ffunc(
                use_df.T.values,
                **kws
            )

            if strat in allowed_hp:
                filt_data = filt_data[0]

            use_df = pd.DataFrame(filt_data.T, index=func.index)
            use_df.columns = func.columns

        return use_df
    
    def get_result(self):
        return self.df_filt

    @classmethod
    def power_spectrum(
        self,
        tc1,
        tc2,
        axs=None,
        TR=0.105,
        figsize=(5,5),
        **kwargs
        ):

        if not isinstance(axs, mpl.axes._axes.Axes):
            fig,axs = plt.subplots(figsize=figsize)

        if not "clip_power" in list(kwargs.keys()):
            clip_power = 25
        else:
            clip_power = kwargs["clip_power"]
            kwargs.pop("clip_power")
        
        pw = []
        for tc in [tc1,tc2]:
            tc_freq = get_freq(
                tc.values.squeeze(), 
                TR=TR, 
                spectrum_type='fft', 
                clip_power=clip_power
            )
            pw.append(tc_freq)

        kwargs = utils.update_kwargs(
            kwargs,
            "x_lim",
            [0,5]
        )

        pl = plotting.LazyPlot(
            [i[1] for i in pw],
            xx=pw[0][0],
            axs=axs,
            markers=[".",None],
            line_width=[0.5,2],
            x_label="frequency (Hz)",
            y_label="power (a.u.)",
            **kwargs
        )

        return pl


    def plot_task_avg(
        self, 
        orig=None,
        filt=None,
        t_col="t", 
        avg=True, 
        plot_title=None,
        incl_task=None,
        sf=None,
        use_cols=["#cccccc","r"],
        power_kws={},
        **kwargs
        ):

        if not isinstance(orig, pd.DataFrame):
            orig = self.func

        if not isinstance(filt, pd.DataFrame):
            filt = self.df_filt
        
        task_ids = utils.get_unique_ids(orig, id="task")
        if isinstance(incl_task, (str,list)):
            if isinstance(incl_task, str):
                incl_task = [incl_task]

            task_ids = [i for i in task_ids if i in incl_task]
        
        if isinstance(sf, (mpl.figure.SubFigure, list)):
            if isinstance(sf, mpl.figure.SubFigure):
                sf = [sf]

            if len(sf) != len(task_ids):
                raise ValueError(f"Number of specified SubFigures ({len(sf)}) does not match number of plots ({len(task_ids)})")
        else:
            fig = plt.figure(
                figsize=(14,len(task_ids)*3), 
                constrained_layout=True,
            )

            sf = fig.subfigures(nrows=len(task_ids))

            if not isinstance(sf, (list,np.ndarray)):
                sf = [sf]

        avg_df = []
        for ix,task in enumerate(task_ids):

            sff = sf[ix]
            print(sff)
            axs = sff.subplots(
                ncols=2,
                width_ratios=[0.1,0.9]
            )

            for df,col,ms,lw,lbl in zip(
                [orig, filt],
                use_cols,
                [".",None],
                [0.5,3],
                ["original","filtered"]):

                task_avg = df.groupby(["subject","task",t_col]).mean()
                task_tcs = utils.select_from_df(task_avg, expression=f"task = {task}")
                t_axs = utils.get_unique_ids(df, id=t_col)

                if avg:
                    task_tcs = pd.DataFrame(task_tcs.mean(axis=1))

                if (ix+1) == len(task_ids):
                    x_lbl = "time (s)"
                else:
                    x_lbl = None

                kwargs = utils.update_kwargs(
                    kwargs,
                    "add_hline",
                    0
                )

                title = None
                if "title" in list(kwargs.keys()):
                    title = kwargs["title"]
                    kwargs.pop("title")

                pl = plotting.LazyPlot(
                    task_tcs.values,
                    axs=axs[1],
                    color=col,
                    markers=ms,
                    line_width=lw,
                    label=[lbl],
                    x_label=x_lbl,
                    y_label="magnitude",
                    **kwargs
                )


                avg_df.append(task_tcs.copy())
            
                if isinstance(title, (str, dict)):
                    if isinstance(title, str):
                        set_title = {}
                        set_title["t"] = title
                    else:
                        set_title = title
                        
                    sff.suptitle(**set_title)

            ps = self.power_spectrum(
                avg_df[0],
                avg_df[1],
                color=use_cols,
                axs=axs[0],
                **power_kws
            )

        if isinstance(plot_title, (str,dict)):
            if isinstance(plot_title, str):
                plot_txt = plot_title
                plot_title = {}
            else:
                plot_txt = plot_title["title"]
                plot_title.pop("title")

            try:
                fig.suptitle(
                    plot_txt, 
                    fontsize=pl.title_size*1.1,
                    **plot_title
                )
                ret_fig = True
            except:
                ret_fig = False

        avg_df = pd.concat(avg_df, axis=1)

        if ret_fig:
            return fig,avg_df
        else:
            return avg_df

class EventRegression(fitting.InitFitter):

    def __init__(
        self, 
        func,
        onsets,
        TR=0.105,
        merge=False,
        evs=None,
        ses=None,
        prediction_plot:bool=False,
        result_plot:bool=False,
        save_ext:str="svg",
        reg_kw:dict={},
        **kwargs
        ):

        self.func = func
        self.onsets = onsets
        self.evs = evs
        self.TR = TR
        self.merge = merge
        self.ses = ses
        self.prediction_plot = prediction_plot
        self.result_plot = result_plot
        self.save_ext = save_ext
        self.reg_kw = reg_kw

        # prepare data
        super().__init__(
            self.func, 
            self.onsets, 
            self.TR,
            merge=self.merge
        )

        # epoch data based on present identifiers (e.g., task/run)
        self.regress_input(**kwargs)

    def regress_runs(
        self,
        df_func,
        df_onsets, 
        basename=None,
        final_ev=True,
        make_figure=False,
        plot_kw={},
        reg_kw={},
        **kwargs
        ):

        # loop through runs
        self.run_ids = utils.get_unique_ids(df_func, id="run")
        print(f"  run_ids: {self.run_ids}")
        # print(f"task-{task}\t| runs = {run_ids}")
        run_df = []
        for run in self.run_ids:
            
            expr = f"run = {run}"
            run_func = utils.select_from_df(df_func, expression=expr)
            run_stims = utils.select_from_df(df_onsets, expression=expr)

            # get regresss
            df,model = self.single_regression(
                run_func,
                run_stims,
                reg_kw=reg_kw,
                **kwargs
            )

            if isinstance(basename, str):
                run_name = f"{basename}_run-{run}"

            if make_figure:
                if final_ev:
                    self.plot_result(
                        run_func,
                        df,
                        basename=run_name,
                        TR=model.TR,
                        **plot_kw
                    )

                    self.plot_model_fits(
                        model,
                        basename=run_name,
                        TR=model.TR,
                        **plot_kw
                    )

            run_df.append(df)

        run_df = pd.concat(run_df)

        return run_df
    
    def regress_tasks(
        self,
        df_func,
        df_onsets,
        basename=None,
        reg_kw={},
        **kwargs
        ):

        # read task IDs
        self.task_ids = utils.get_unique_ids(df_func, id="task")

        # loop through task IDs
        task_df = [] 
        for task in self.task_ids:

            # extract task-specific dataframes
            utils.verbose(f"  task_id: {task}", True)
            expr = f"task = {task}"
            task_func = utils.select_from_df(df_func, expression=expr)
            task_stims = utils.select_from_df(df_onsets, expression=expr)

            if isinstance(basename, str):
                task_name = f"{basename}_task-{task}"
            
            df = self.regress_runs(
                task_func,
                task_stims,
                basename=task_name,
                reg_kw=reg_kw,
                **kwargs
            )

            task_df.append(df)


        return pd.concat(task_df)
    
    def regress_subjects(
        self,
        df_func,
        df_onsets,
        evs=None,
        ses=None,
        reg_kw={},
        **kwargs
        ):

        self.sub_ids = utils.get_unique_ids(df_func, id="subject")
        
        # loop through subject IDs
        sub_df = [] 
        for sub in self.sub_ids:
            
            utils.verbose(f"sub_id: {sub}", True)
            # fetch evs to regress out
            if not isinstance(evs, (str,list)):
                evs = utils.get_unique_ids(df_onsets, id="event_type")
            else:
                if isinstance(evs, str):
                    evs = [evs]

            use_func = df_func.copy()
            for ix,ev in enumerate(evs):
                utils.verbose(f" event: {ev}", True)
                # extract task-specific dataframes
                expr = f"subject = {sub}"
                self.sub_func = utils.select_from_df(use_func, expression=expr)
                self.sub_stims = utils.select_from_df(df_onsets, expression=(expr,"&",f"event_type = {ev}"))

                try:
                    self.task_ids = utils.get_unique_ids(self.sub_func, id="task")
                except:
                    self.task_ids = None

                if isinstance(self.task_ids, list):
                    ffunc = self.regress_tasks
                else:
                    ffunc = self.regress_runs
                
                basename = f"sub-{sub}"
                if isinstance(ses, (int,str)):
                    basename += f"_ses-{ses}"

                # only start plotting after the last event has been regressed
                if (ix+1)==len(evs):
                    final_ev = True
                else:
                    final_ev = False

                ev_regress = ffunc(
                    self.sub_func, 
                    self.sub_stims,
                    basename=basename,
                    final_ev=final_ev,
                    reg_kw=reg_kw,
                    **kwargs
                )

                # set func as regressed output of previous ev
                use_func = ev_regress.copy()

                # append last regressed ev
                if ix+1 == len(evs):
                    sub_df.append(ev_regress)

        sub_df = pd.concat(sub_df)

        return sub_df
    
    def regress_input(self, **kwargs):

        self.df_regress = self.regress_subjects(
            self.func,
            self.onsets,
            evs=self.evs,
            ses=self.ses,
            reg_kw=self.reg_kw,
            **kwargs
        )

    @classmethod
    def single_regression(
        self, 
        func, 
        onsets, 
        reg_kw={},
        **kwargs
        ):
        
        # fit FIR model
        model = fitting.NideconvFitter(
            func,
            onsets,
            **kwargs
        )

        model.timecourses_condition()
        
        # regress out
        cleaned = RegressOut(
            model.func,
            model.sub_pred_full,
            **reg_kw
        )

        return cleaned.clean_df, model

    def plot_timecourse_prediction(
        tc1,
        tc2,
        axs=None,
        figsize=(16,4),
        time_col="t",
        t_axis=None,
        TR=0.105,
        **kwargs
        ):

        if not isinstance(axs, mpl.axes._axes.Axes):
            fig,axs = plt.subplots(figsize=figsize)

        data_list = []
        for i in [tc1,tc2]:
            if isinstance(i, pd.DataFrame):
                data_list.append(i.values.squeeze())
            elif isinstance(i, np.ndarray):
                data_list.append(i)
            else:
                raise TypeError(f"Unrecognized input type {type(i)}.. Must be numpy array or dataframe")

        if not isinstance(t_axis, (list,np.ndarray)):
            if isinstance(tc1, pd.DataFrame):
                t_axis = utils.get_unique_ids(i, id=time_col)
            else:
                t_axis = list(np.arange(0,data_list[0].shape[0])*TR)
        
        pl = plotting.LazyPlot(
            data_list,
            xx=t_axis,
            axs=axs,
            markers=[".",None],
            line_width=[0.5,2],
            x_label="time (s)",
            y_label="magnitude (%)",
            **kwargs
        )

        return pl
    
    def plot_power_spectrum(
        tc1,
        tc2,
        axs=None,
        TR=0.105,
        figsize=(5,5),
        **kwargs
        ):

        if not isinstance(axs, mpl.axes._axes.Axes):
            fig,axs = plt.subplots(figsize=figsize)

        if not "clip_power" in list(kwargs.keys()):
            clip_power = 100
        else:
            clip_power = kwargs["clip_power"]
            kwargs.pop("clip_power")
        
        pw = []
        for tc in [tc1,tc2]:
            tc_freq = get_freq(
                tc.values.squeeze(), 
                TR=TR, 
                spectrum_type='fft', 
                clip_power=clip_power
            )
            pw.append(tc_freq)

        pl = plotting.LazyPlot(
            [i[1] for i in pw],
            xx=pw[0][0],
            axs=axs,
            markers=[".",None],
            line_width=[0.5,2],
            x_label="frequency (Hz)",
            y_label="power (a.u.)",
            **kwargs
        )

        return pl

    @classmethod
    def plot_model_fits(
        self,
        model,
        save=False,
        fig_dir=None,
        basename=None,
        TR=0.105,
        cm="inferno",
        ext="svg",
        time_col="time",
        w_ratio=[0.8,0.2],
        evs=None,
        loc=[0,1],
        **kwargs
        ):

        # parse to list
        func_list = list(model.func.T.values)
        pred_list = list(model.sub_pred_full.T.values)
        prof_list = list(model.tc_condition.T.values)
        sem_list = list(model.sem_condition.T.values)

        n_plots = model.func.shape[-1]
        if n_plots > 20:
            raise ValueError(f"Max number of plots = 20, you requested {n_plots}..")
        
        fig = plt.figure(figsize=(16,n_plots*4), constrained_layout=True)
        sf = fig.subfigures(nrows=n_plots)
        cms = sns.color_palette(cm, n_plots)
        for i in range(n_plots):

            if n_plots == 1:
                sf_ix = sf
            else:
                sf_ix = sf[i]

            axs = sf_ix.subplots(
                ncols=2, 
                width_ratios=w_ratio
            )

            # plot timecourse+prediction
            tc_plot = self.plot_timecourse_prediction(
                func_list[i],
                pred_list[i],
                axs=axs[0],
                color=["#cccccc",cms[i]],
                labels=["data","prediction"],
                **kwargs
            )

            # plot response profile
            resp_plot = plotting.LazyPlot(
                prof_list[i],
                xx=utils.get_unique_ids(model.tc_condition, id=time_col),
                axs=axs[1],
                add_hline=0,
                color=cms[i],
                error=sem_list[i],
                line_width=tc_plot.line_width[-1],
                x_label="time",
                TR=TR,
                **kwargs
            )

            axs[1].axvspan(
                *loc, 
                ymin=0,
                ymax=1, 
                alpha=0.3, 
                color="#cccccc",
            )

            sf_ix.suptitle(
                f"vox-{i+1}", 
                fontsize=resp_plot.title_size,
                fontweight="bold"
            )

        if save:
            if not isinstance(fig_dir, str):
                raise ValueError("Please specify output directory for figure")

            if not isinstance(basename, str):
                raise ValueError("Please specify basename for figure filename")

            fname = opj(fig_dir, f"{basename}_desc-modelfit.{ext}")
            utils.verbose(f" Writing {fname}", True)
            fig.savefig(
                fname, 
                bbox_inches="tight", 
                dpi=300, 
                facecolor="white"
            )

            plt.close()

    @classmethod
    def plot_result(
        self,
        raw,
        regr,
        avg=True,
        save=False,
        fig_dir=None,
        basename=None,
        TR=0.105,
        ext="svg",
        w_ratio=[0.8,0.2],
        cols=["#cccccc","r"],
        evs=None,
        **kwargs
        ):

        if avg:
            n_plots = 1
        else:
            n_plots = raw.shape[-1]

        fig = plt.figure(figsize=(16,n_plots*4), constrained_layout=True)
        sf = fig.subfigures(nrows=n_plots)
        for i in range(n_plots):

            if n_plots == 1:
                sf_ix = sf
            else:
                sf_ix = sf[i]

            axs = sf_ix.subplots(
                ncols=2, 
                width_ratios=w_ratio
            )

            if avg:
                tc1 = pd.DataFrame(raw.mean(axis=1), columns=["avg"])
                tc2 = pd.DataFrame(regr.mean(axis=1), columns=["avg"])
                set_title = "average"
            else:
                tc1 = utils.select_from_df(raw, expression="ribbon", indices=[i])
                tc2 = utils.select_from_df(regr, expression="ribbon", indices=[i])
                set_title = f"vox-{i+1}"
            
            # plot timecourse+prediction
            tc_plot = self.plot_timecourse_prediction(
                tc1,
                tc2,
                axs=axs[0],
                color=cols,
                labels=["pre","post"],
                **kwargs
            )

            # plot power spectrum
            freq_plot = self.plot_power_spectrum(
                tc1,
                tc2,
                axs=axs[1],
                TR=TR,
                color=cols,
                x_lim=[0,1.5],
                **kwargs
            )

            sf_ix.suptitle(set_title, fontsize=freq_plot.title_size)
            
        if isinstance(evs, (str,list)):
            add_txt = f": {evs}"
        else:
            add_txt = ""

        fig.suptitle(
            f"effect of regressing out event{add_txt}", 
            fontsize=freq_plot.title_size*1.1, 
            fontweight="bold"
        )

        if save:
            if not isinstance(fig_dir, str):
                raise ValueError("Please specify output directory for figure")

            if not isinstance(basename, str):
                raise ValueError("Please specify basename for figure filename")

            fname = opj(fig_dir, f"{basename}_desc-regression.{ext}")
            utils.verbose(f" Writing {fname}", True)
            fig.savefig(
                fname, 
                bbox_inches="tight", 
                dpi=300, 
                facecolor="white"
            )

            plt.close()
