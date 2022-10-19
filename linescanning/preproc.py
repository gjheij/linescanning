from . import utils, plotting
from .segmentations import Segmentations
from kneed import KneeLocator
import matplotlib.pyplot as plt
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
        **kwargs):

        self.data               = data
        self.subject            = subject
        self.run                = run
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
        self.__dict__.update(kwargs)

        if self.wm_voxels == None and self.csf_voxels == None:
            super().__init__(
                self.subject,
                run=self.run,
                reference_slice=self.reference_slice,
                target_session=self.trg_session,
                foldover=self.foldover,
                verbose=self.verbose,
                trafo_file=self.trafo_list,
                **kwargs)

        if self.verbose:
            print(f" Using {self.n_components} components for aCompCor (WM/CSF separately)")

        self.acompcor_components    = []
        self.elbows                 = []
        self.pcas                   = []
        self.tissue_pca             = {"wm": True, "csf": True}
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
                    
                    if self.verbose:
                        print(f" Found {self.elbow_} component(s) in '{tissue}'-voxels with total explained variance of {round(sum(self.pca.explained_variance_ratio_[:self.elbow_]), 2)}%")

                    self.pca_desc = f"""
Timecourses from these voxels were extracted and fed into a PCA. These components were used to clean the data from respiration/cardiac frequencies. """

                except:
                    self.elbow_ = None
                    self.tissue_pca[tissue] = False
                    if self.verbose:
                        print(f" PCA for '{tissue}' was unsuccessful. Using all un-PCA'd timecourses ({len(self.tissue_voxels)})")
                        self.pca_desc = f"""
PCA with {self.n_components} was unsuccessful, so '{tissue}' timecourses were used to clean the data from respiration/cardiac 
frequencies. """

            else:

                if self.verbose:
                    print(f" PCA for '{tissue}' was unsuccessful because no voxels were found")
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
                if self.tissue_pca[tissue]:
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
            if verbose:
                print(f" Only regressing out component {select_component}")
            self.confs = self.acompcor_components[:, self.select_component-1]

        if self.filter_confs != None:
            if self.verbose:
                print(f" DCT high-pass filter on components [removes low frequencies <{filter_confs} Hz]")

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
            if self.trg_session == None:
                self.base_name = self.subject
            else:
                self.base_name = f"{self.subject}_ses-{self.trg_session}"

            fname = opj(self.save_as, f"{self.base_name}_run-{self.run}_desc-acompcor.{self.save_ext}")
            fig.savefig(fname, bbox_inches='tight', dpi=300)

class RegressOut():

    def __init__(self, data, regressors):
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
        
        self.clean_array = clean(self.data_array, standardize=False, confounds=self.regressors_array)

        if self.add_index:
            self.clean_df = pd.DataFrame(self.clean_array, index=self.data.index, columns=self.data.columns)


def highpass_dct(func, lb, TR=0.105):
    """highpass_dct

    Discrete cosine transform (DCT) is a basis set of cosine regressors of varying frequencies up to a filter cutoff of a specified number of seconds. Many software use 100s or 128s as a default cutoff, but we encourage caution that the filter cutoff isn't too short for your specific experimental design. Longer trials will require longer filter cutoffs. See this paper for a more technical treatment of using the DCT as a high pass filter in fMRI data analysis (https://canlab.github.io/_pages/tutorials/html/high_pass_filtering.html).

    Parameters
    ----------
    func: np.ndarray
        <n_voxels, n_timepoints> representing the functional data to be fitered
    lb: float
        cutoff-frequency for low-pass
    TR: float, optional
        Repetition time of functional run, by default 0.105

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
    n_vol           = func.shape[-1]
    st_ref          = 0  # offset frametimes by st_ref * tr
    ft              = np.linspace(st_ref * TR, (n_vol + st_ref) * TR, n_vol, endpoint=False)
    hp_set          = _cosine_drift(lb, ft)
    dct_data        = clean(func.T, detrend=False, standardize=False, confounds=hp_set).T

    return dct_data, hp_set

def lowpass_savgol(func, window_length=None, polyorder=None):

    """lowpass_savgol

    The Savitzky-Golay filter is a low pass filter that allows smoothing data. To use it, you should give as input parameter of the function the original noisy signal (as a one-dimensional array), set the window size, i.e. n° of points used to calculate the fit, and the order of the polynomial function used to fit the signal. We might be interested in using a filter, when we want to smooth our data points; that is to approximate the original function, only keeping the important features and getting rid of the meaningless fluctuations. In order to do this, successive subsets of points are fitted with a polynomial function that minimizes the fitting error.

    The procedure is iterated throughout all the data points, obtaining a new series of data points fitting the original signal. If you are interested in knowing the details of the Savitzky-Golay filter, you can find a comprehensive description [here](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter).

    Parameters
    ----------
    func: np.ndarray
        <n_voxels, n_timepoints> representing the functional data to be fitered
    window_length: int
        Length of window to use for filtering. Must be an uneven number according to the scipy-documentation
    poly_order: int
        Order of polynomial fit to employ within `window_length`.

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

def get_freq(func, TR=0.105, spectrum_type='psd', clip_power=None):
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
        self.data           = data
        self.n_components   = n_components
        self.filter_confs   = filter_confs
        self.verbose        = verbose
        self.TR             = TR
        self.save_as        = save_as
        self.session        = session
        self.run            = run
        self.summary_plot   = summary_plot
        self.melodic_plot   = melodic_plot
        self.gm_voxels      = ribbon
        self.save_ext       = save_ext
        self.keep_comps     = keep_comps
        self.__dict__.update(kwargs)
        
        self.__desc__ = f"""
Sklearn's implementation of 'FastICA' was used to decompose the signal into {self.n_components} components.
 """

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
            if self.verbose:
                print(f" DCT high-pass filter on components [removes low frequencies <{self.filter_confs} Hz]")

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

            if self.verbose:
                print(f" Keeping components: {self.keep_comps}")

            if self.filter_confs != None:
                use_data = self.S_filt.copy()
            else:
                use_data = self.S_.copy()

            self.confounds = use_data[:,[i for i in range(use_data.shape[-1]) if i not in self.keep_comps]]
        else:
            if self.filter_confs == None:
                raise ValueError("Not sure what to do. Please specify either list of components to keep (e.g., 'keep_comps=[1,2]' or specify a high-pass cut off frequency (e.g., 'filter_confs=0.18')")

            self.__desc__ += f"""
Resulting components from the ICA were high-pass filtered using discrete cosine sets (DCT) with a cut off frequency of {self.filter_confs} Hz.
"""
            # this is pretty hard core: regress out all high-passed components
            if self.verbose:
                print(f" Regressing out all high-passed components [>{self.filter_confs} Hz]")
                
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
        gs = fig.add_gridspec(1,3, width_ratios=[30,30,100])
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
            if self.session == None:
                self.base_name = self.subject
            else:
                self.base_name = f"{self.subject}_ses-{self.session}"

            fname = opj(self.save_as, f"{self.base_name}_run-{self.run}_desc-ica.{self.save_ext}")
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
        fig = plt.figure(figsize=(24, plot_comps*6))
        subfigs = fig.subfigures(nrows=plot_comps, hspace=0.4)    

        # get plotting defaults
        self.defaults = plotting.Defaults()

        for comp in range(plot_comps):
            
            # make subfigure for each component
            if zoom_freq:
                axs = subfigs[comp].subplots(ncols=4, gridspec_kw={'width_ratios': [0.3,1,0.3,0.2], "wspace": 0.3})
            else:
                axs = subfigs[comp].subplots(ncols=3, gridspec_kw={'width_ratios': [0.3,1,0.3]})

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
                x_lim=[0,1.5],
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

        fig.suptitle("Independent component analyis (ICA)", fontsize=self.defaults.font_size*1.8, y=1.012)

        plt.tight_layout()

        if self.save_as != None:
            if self.session == None:
                self.base_name = self.subject
            else:
                self.base_name = f"{self.subject}_ses-{self.session}"

            fname = opj(self.save_as, f"{self.base_name}_run-{self.run}_desc-melodic.{self.save_ext}")
            fig.savefig(
                fname, 
                bbox_inches="tight", 
                dpi=300, 
                facecolor="white")