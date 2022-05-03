from . import glm, utils, plotting
from .segmentations import Segmentations
from kneed import KneeLocator
import matplotlib.pyplot as plt
import nibabel as nb
from nilearn.signal import clean
from nilearn.glm.first_level.design_matrix import _cosine_drift
from nitime.timeseries import TimeSeries
from nitime.analysis import SpectralAnalyzer
import numpy as np
import os
import pandas as pd
from scipy import io
from scipy import signal
from sklearn import decomposition
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
    n_pca: int, optional
        Number of PCA-components to extract for each of the WM/CSF voxels, by default 5
    select_component: int, optional
        Select one particular component to regress out rather than all extracted components, by default None because high-pass filtering the PCAs is much more effective
    filter_pca: float, optional
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
    >>> acomp = preproc.aCompCor(data,
    >>>                          subject="sub-003",
    >>>                          run=1,
    >>>                          trg_session=4,
    >>>                          n_pca=5,
    >>>                          trafo_list=['ses_to_motion.mat', 'run_to_run.mat'],
    >>>                          filter_pca=0.2,
    >>>                          TR=0.105,
    >>>                          verbose=True)
    """
    
    def __init__(self, 
                 data, 
                 run=None,
                 subject=None, 
                 wm_voxels=None,
                 csf_voxels=None,
                 n_pca=5, 
                 select_component=None, 
                 filter_pca=None, 
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
        self.n_pca              = n_pca
        self.select_component   = select_component
        self.filter_pca         = filter_pca
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
            super().__init__(self.subject,
                             run=self.run,
                             reference_slice=self.reference_slice,
                             target_session=self.trg_session,
                             foldover=self.foldover,
                             verbose=self.verbose,
                             trafo_file=self.trafo_list,
                             **kwargs)

        if self.verbose:
            print(f" Using {self.n_pca} components for aCompCor (WM/CSF separately)")

        self.acompcor_components    = []
        self.elbows                 = []
        self.pcas                   = []
        for tissue in ['csf', 'wm']:
            
            self.tissue_voxels  = getattr(self, f"{tissue}_voxels")
            self.tissue_tc      = utils.select_from_df(self.data, expression="ribbon", indices=self.tissue_voxels)

            try:
                self.pca        = decomposition.PCA(n_components=self.n_pca)
                self.components = self.pca.fit_transform(self.tissue_tc)

                self.pcas.append(self.pca)
                # find elbow with KneeLocator
                self.xx     = np.arange(0, self.n_pca)
                self.kn     = KneeLocator(self.xx, self.pca.explained_variance_, curve='convex', direction='decreasing')
                self.elbow_ = self.kn.knee
                
                if self.verbose:
                    print(f" Found {self.elbow_} component(s) in '{tissue}'-voxels with total explained variance of {round(sum(self.pca.explained_variance_ratio_[:self.elbow_]), 2)}%")
            except:
                if self.verbose:
                    print(f" PCA with {self.n_pca} was unsuccessful. Using WM/CSF timecourses")

                self.elbow_ = None

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
                self.do_pca = False
                self.info = "timecourses"
                # raise ValueError("Found 0 components surviving the elbow-plot. Turn on verbose and inspect the plot")
                self.acompcor_components.append(self.tissue_tc)

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

        if self.filter_pca != None:
            if self.verbose:
                print(f" DCT high-pass filter on components [removes low frequencies <{filter_pca} Hz]")

            if self.confs.ndim >= 2:
                self.confs, _ = highpass_dct(self.confs.T, self.filter_pca, TR=self.TR)
                self.confs = self.confs.T
            else:
                self.confs, _ = highpass_dct(self.confs, self.filter_pca, TR=self.TR)

        # outputs (timepoints, voxels) array
        self.acomp_data = clean(self.data.values, standardize=False, confounds=self.confs).T

        # make summary plot of aCompCor effect
        if self.summary_plot:
            self.summary()

    def summary(self, **kwargs):
        

        if self.do_pca:
            fig = plt.figure(figsize=(30, 7))
            gs = fig.add_gridspec(1, 4)
        else:
            fig = plt.figure(figsize=(24, 7))
            gs = fig.add_gridspec(1, 3)

        ax = fig.add_subplot(gs[0])
        self.plot_regressor_voxels(ax=ax)

        if not hasattr(self, 'line_width'):
            line_width = 2

        if hasattr(self, "regressor_voxel_colors"):
            use_colors = self.regressor_voxel_colors
        else:
            use_colors = None

        label = ["csf", "wm"]
        if self.do_pca:
            ax1 = fig.add_subplot(gs[1])
            for ix, ii in enumerate(self.elbows):
                if use_colors != None:
                    color = use_colors[ix]
                else:
                    color = "#cccccc"

                if ii != None:
                    ax1.axvline(ii, color=color, ls='dashed', lw=0.5, alpha=0.5)
                    if any(v is None for v in self.elbows):
                        use_colors = use_colors[ii]
                        label = [label[ii]]

            plotting.LazyPlot([self.pcas[ii].explained_variance_ratio_ for ii in range(len(self.pcas))],
                            xx=self.xx,
                            color=use_colors,
                            axs=ax1,
                            title=f"Scree-plot run-{self.run}",
                            x_label="nr of components",
                            y_label="variance explained (%)",
                            labels=label,
                            font_size=16,
                            line_width=line_width,
                            sns_trim=True,
                            **kwargs)

        # create dashed line on cut-off frequency if specified
        if self.filter_pca != None:
            add_vline = {'pos': self.filter_pca, 
                         'color': 'k',
                         'ls': 'dashed', 
                         'lw': 0.5}
        else:
            add_vline = None
        
        if self.do_pca:
            ax2 = fig.add_subplot(gs[2])
        else:
            ax2 = fig.add_subplot(gs[1])
        plotting.LazyPlot(self.nuisance_spectra,
                            xx=self.nuisance_freqs[0],
                            axs=ax2,
                            labels=[f"component {ii+1}" for ii in range(self.acompcor_components.shape[-1])],
                            title=f"Power spectra of {self.info}",
                            x_label="frequency (Hz)",
                            y_label="power (a.u.)",
                            x_lim=[0, 1.5],
                            font_size=16,
                            line_width=line_width,
                            add_vline=add_vline,
                            sns_trim=True,
                            **kwargs)

        # plot power spectra from non-aCompCor'ed vs aCompCor'ed data
        tc1 = self.data['vox 359'].values
        tc2 = self.acomp_data[359,:]

        if not hasattr(self, "clip_power"):
            clip_power = 100

        if self.do_pca:
            ax3 = fig.add_subplot(gs[3])
        else:
            ax3 = fig.add_subplot(gs[2])
        tc1_freq = get_freq(tc1, TR=self.TR, spectrum_type='fft', clip_power=clip_power)
        tc2_freq = get_freq(tc2, TR=self.TR, spectrum_type='fft', clip_power=clip_power)

        plotting.LazyPlot([tc1_freq[1], tc2_freq[1]],
                            xx=tc1_freq[0],
                            color=["#1B9E77", "#D95F02"],
                            x_label="frequency (Hz)",
                            y_label="power (a.u.)",
                            title="Effect aCompCor on timecourses",
                            labels=['no aCompCor', 'aCompCor'],
                            axs=ax3,
                            font_size=16,
                            x_lim=[0, 1.5],
                            line_width=2,
                            sns_trim=True,
                            **kwargs)

        if self.save_as != None:
            fname = self.save_as+f"_run-{self.run}_desc-acompcor.{self.save_ext}"
            
            if self.verbose:
                print(f" Saving {fname}")

            fig.savefig(fname)

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
    _type_
        _description_

    Raises
    ----------
    ValueError
        _description_

    Example
    ----------
    >>> 
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