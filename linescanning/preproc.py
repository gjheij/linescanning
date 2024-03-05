from . import utils, plotting, fitting
from .segmentations import Segmentations
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
from scipy import signal, fft
import seaborn as sns
from sklearn import decomposition
from typing import Union
import warnings

opj = os.path.join
pd.options.mode.chained_assignment = None # disable warning thrown by string2float
warnings.filterwarnings("ignore")

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

        self.acompcor_components    = []
        self.elbows                 = []
        self.pcas                   = []
        self.tissue_pca             = {"wm": True, "csf": True}
        self.pca_but_timecourses    = {"wm": False, "csf": False}
        for tissue in ['csf', 'wm']:
            
            self.tissue_voxels  = getattr(self, f"{tissue}_voxels")
            self.tissue_tc      = utils.select_from_df(self.data, expression="ribbon", indices=self.tissue_voxels)

            if len(self.tissue_voxels) != 0:
                try:
                    self.pca        = decomposition.PCA(n_components=self.n_components)
                    self.components = self.pca.fit_transform(self.tissue_tc)

                    self.pcas.append(self.pca)
                    # find elbow with KneeLocator
                    self.xx     = np.arange(0, self.n_components)
                    self.kn     = KneeLocator(self.xx, self.pca.explained_variance_, curve='convex', direction='decreasing')
                    self.elbow_ = self.kn.knee
                    
                    utils.verbose(f" Found {self.elbow_} component(s) in '{tissue}'-voxels with total explained variance of {round(sum(self.pca.explained_variance_ratio_[:self.elbow_]), 2)}%", self.verbose)

                    self.pca_desc = f"""
Timecourses from these voxels were extracted and fed into a PCA. These components were used to clean the data from respiration/cardiac frequencies. """

                except:
                    self.elbow_ = None
                    self.tissue_pca[tissue] = False
                    self.pca_but_timecourses[tissue] = True
                    utils.verbose(f" PCA for '{tissue}' was unsuccessful. Using all un-PCA'd timecourses ({len(self.tissue_voxels)})", self.verbose)
                    self.pca_desc = f"""
PCA with {self.n_components} was unsuccessful, so '{tissue}' timecourses were used to clean the data from respiration/cardiac 
frequencies. """

            else:

                utils.verbose(f" PCA for '{tissue}' was unsuccessful because no voxels were found", self.verbose)
                self.pca_desc = f"""
No voxels for '{tissue}' were found, so PCA was skipped. """

                self.elbow_ = None
                self.tissue_pca[tissue] = False

            self.elbows.append(self.elbow_)

            # extract components before elbow of plot
            if self.elbow_ != None:
                self.do_pca = True
                self.info = "components"
                self.include_components = self.components[:, :self.elbow_]
                if self.include_components.ndim == 1:
                    self.include_components = self.include_components[..., np.newaxis]

                self.acompcor_components.append(self.include_components)
            else:
                if self.pca_but_timecourses[tissue]:
                    self.do_pca = False
                    self.info = "timecourses"
                    # raise ValueError("Found 0 components surviving the elbow-plot. Turn on verbose and inspect the plot")
                    self.acompcor_components.append(self.tissue_tc.mean(axis=1)[:,np.newaxis])

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
                self.confs, _ = highpass_dct(self.confs.T, self.filter_confs, TR=self.TR)
                self.confs = self.confs.T
            else:
                self.confs, _ = highpass_dct(self.confs, self.filter_confs, TR=self.TR)

        # outputs (timepoints, voxels) array (RegressOut is also usable, but this is easier in linescanning.dataset.Dataset)
        self.acomp_data = clean(self.data.values, standardize=False, confounds=self.confs).T

        self.__desc__ = """
Nighres segmentations were transformed to the individual slices of each run using antsApplyTransforms with `MultiLabel` interpolation. 
White matter and CSF voxels were selected based on the CRUISE-segmentation if all voxels across the beam (in *phase-enconding* 
direction) were assigned to this tissue type. This limited the possibility for partial voluming. """

        self.__desc__ += self.pca_desc

        # make summary plot of aCompCor effect
        if self.summary_plot:
            self.summary()

    def summary(self, **kwargs):

        if self.tissue_pca["wm"] or self.tissue_pca["csf"]:
            fig = plt.figure(figsize=(30,7))
            gs = fig.add_gridspec(1,4)
        else:
            fig = plt.figure(figsize=(24,7))
            gs = fig.add_gridspec(1, 3)

        ax = fig.add_subplot(gs[0])
        self.plot_regressor_voxels(ax=ax)

        if not hasattr(self, 'line_width'):
            line_width = 2

        if hasattr(self, "regressor_voxel_colors"):
            use_colors = self.regressor_voxel_colors
        else:
            use_colors = "#cccccc"

        if self.tissue_pca["wm"] or self.tissue_pca["csf"]:
            ax1 = fig.add_subplot(gs[1])
                
            if self.tissue_pca["wm"] and self.tissue_pca["csf"]:
                label = ["csf", "wm"]
                colors = use_colors
            elif self.tissue_pca["wm"] and not self.tissue_pca["csf"]:
                label = ["wm"]
                colors = use_colors[1]
            elif self.tissue_pca["csf"] and not self.tissue_pca["wm"]:
                label = ["csf"]
                colors = use_colors[0]

            # make dashed line for each tissue PCA
            for ix, ii in enumerate(self.elbows):
                if ii != None:
                    ax1.axvline(ii, color=use_colors[ix], ls='dashed', lw=0.5, alpha=0.5)

            plotting.LazyPlot(
                [self.pcas[ii].explained_variance_ratio_ for ii in range(len(self.pcas))],
                xx=self.xx,
                color=colors,
                axs=ax1,
                title=f"Scree-plot run-{self.run}",
                x_label="nr of components",
                y_label="variance explained (%)",
                labels=label,
                line_width=line_width,
                **kwargs)

        # create dashed line on cut-off frequency if specified
        if self.filter_confs != None:
            add_vline = {'pos': self.filter_confs, 
                         'color': 'k',
                         'ls': 'dashed', 
                         'lw': 0.5}
        else:
            add_vline = None
        
        if self.do_pca:
            ax2 = fig.add_subplot(gs[2])
        else:
            ax2 = fig.add_subplot(gs[1])
            
        plotting.LazyPlot(
            self.nuisance_spectra,
            xx=self.nuisance_freqs[0],
            axs=ax2,
            labels=[f"component {ii+1}" for ii in range(self.acompcor_components.shape[-1])],
            title=f"Power spectra of {self.info}",
            x_label="frequency (Hz)",
            y_label="power (a.u.)",
            x_lim=[0, 1.5],
            line_width=line_width,
            add_vline=add_vline,
            **kwargs)

        # plot power spectra from non-aCompCor'ed vs aCompCor'ed data
        tc1 = utils.select_from_df(self.data, expression='ribbon', indices=self.gm_voxels).mean(axis=1).values
        tc2 = self.acomp_data[self.gm_voxels,:].mean(axis=0)

        if not hasattr(self, "clip_power"):
            clip_power = 100

        if self.do_pca:
            ax3 = fig.add_subplot(gs[3])
        else:
            ax3 = fig.add_subplot(gs[2])
        tc1_freq = get_freq(tc1, TR=self.TR, spectrum_type='fft', clip_power=clip_power)
        tc2_freq = get_freq(tc2, TR=self.TR, spectrum_type='fft', clip_power=clip_power)

        plotting.LazyPlot(
            [tc1_freq[1], tc2_freq[1]],
            xx=tc1_freq[0],
            color=["#1B9E77", "#D95F02"],
            x_label="frequency (Hz)",
            y_label="power (a.u.)",
            title="Power spectra of average GM-voxels",
            labels=['no aCompCor', 'aCompCor'],
            axs=ax3,
            x_lim=[0, 1.5],
            line_width=2,
            **kwargs)

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
            **kwargs)
            
        inset.set_xticks([])
        sns.despine(
            offset=None, 
            bottom=True, 
            ax=inset)

        if self.save_as != None:
            self.base_name = self.subject
            if isinstance(self.trg_session, (str,float,int)):
                self.base_name += f"_ses-{self.trg_session}"

            if isinstance(self.task, str):
                self.base_name += f"_task-{self.task}"         

            fname = opj(self.save_as, f"{self.base_name}_run-{self.run}_desc-acompcor.{self.save_ext}")
            fig.savefig(fname, bbox_inches='tight', dpi=300)

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

def lowpass_savgol(func, window_length=7, polyorder=3):

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
    
    return signal.savgol_filter(func, window_length, polyorder, axis=-1)

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
    
    def plot_task_avg(
        self, 
        orig=None,
        filt=None,
        t_col="t", 
        avg=True, 
        plot_title=None,
        **kwargs
        ):

        if not isinstance(orig, pd.DataFrame):
            orig = self.func

        if not isinstance(filt, pd.DataFrame):
            filt = self.df_filt
        
        task_ids = utils.get_unique_ids(orig, id="task")
        fig,axs = plt.subplots(
            nrows=len(task_ids), 
            figsize=(14,len(task_ids)*3), 
            constrained_layout=True, 
            sharey=True, 
            sharex=True
        )

        avg_df = []
        for ix,task in enumerate(task_ids):

            if len(task_ids)>1:
                ax = axs[ix]
            else:
                ax = axs

            for df,col,ms,lw,lbl in zip(
                [orig, filt],
                ["#cccccc","r"],
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

                pl = plotting.LazyPlot(
                    task_tcs.values,
                    axs=ax,
                    color=col,
                    markers=ms,
                    line_width=lw,
                    label=[lbl],
                    title={
                        "title": f"task-{task}",
                    },
                    add_hline=0,
                    x_label=x_lbl,
                    y_label="magnitude",
                    **kwargs
                )

                avg_df.append(task_tcs.copy())
        
        if isinstance(plot_title, (str,dict)):
            if isinstance(plot_title, str):
                plot_txt = plot_title
                plot_title = {}
            else:
                plot_txt = plot_title["title"]
                plot_title.pop("title")

            fig.suptitle(
                plot_txt, 
                fontsize=pl.title_size*1.1,
                **plot_title
            )

        avg_df = pd.concat(avg_df)
        return fig,avg_df

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
            