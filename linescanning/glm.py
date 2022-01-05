from linescanning import utils
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import seaborn as sns
import warnings

class GenericGLM:
    """GenericGLM

    Main class to perform a simple GLM with python. Will do most of the processes internally, and allows you to plot various processes along the way.

    Parameters
    ----------
    onset: pandas.DataFrame
        Dataframe containing the onset times for all events in an experiment. Specifically design to work smoothly with :func:`linescanning.utils.ParseExpToolsFile`. You should insert the output from :func:`linescanning.utils.ParseExpToolsFile.get_onset_df()` as `onset`
    data: numpy.ndarray, pandas.DataFrame
        <time,voxels> numpy array or pandas DataFrame; required for creating the appropriate length of the stimulus vectors
    hrf_pars: dict, optional
        dictionary collecting the parameters required for :func:`linescanning.glm.double_gamma` (generally the defaults are fine though!)

        >>> pars = {'lag': 6,
        >>>         'a2': 12,
        >>>         'b1': 12,
        >>>         'b2': 12,
        >>>         'c': 12,
        >>>         'scale': True}
    TR: float
        repetition time of acquisition
    osf: [type], optional
        Oversampling factor used to account for decimal onset times, by default None    
    type: str, optional
        Use block design of event-related design, by default 'event'. If set to 'block', `block_length` is required.
    block_length: int, optional
        Duration of block in seconds, by default None
    amplitude: int, list, optional
        Amplitude to be used when creating the stimulus vector, by default None. If nothing is specified, the amplitude will be set to '1', like you would in a regular FSL 1-/3-column file. If you want variable amplitudes for different events for in a simulation, you can specify a list with an equal length to the number of events present in `onset_df`.

    Example
    ----------
    >>> # import modules
    >>> from linescanning.glm import GenericGLM
    >>> from linescanning import utils
    >>> 
    >>> # define file with fMRI-data and the output from Exptools2
    >>> func_file = "some_func_file.mat"
    >>> exp_file = "some_exp_file.tsv"
    >>>
    >>> # load in functional data
    >>> func = utils.ParseFuncFile(func_file, 
    >>>                            subject=1, 
    >>>                            run=1, 
    >>>                            deleted_first_timepoints=200, 
    >>>                            deleted_last_timepoints=200,
    >>>                            bp_filter="rolling")
    >>>
    >>> # fetch HP-filtered, percent-signal changed data
    >>> data = func.dct_psc_df.copy()
    >>>
    >>> # load in exptools-file, use attributes from 'func'
    >>> onset = utils.ParseExpToolsFile(exp_file,
    >>>                                 subject=func.subject,
    >>>                                 run=func.run,
    >>>                                 delete_vols=(func.deleted_first_timepoints),
    >>>                                 TR=func.TR)
    >>>
    >>> # fetch the onset times and event names in a dataframe
    >>> onsets = onset.get_onset_df()
    >>>
    >>> # do the fitting
    >>> fitting = GenericGLM(onsets, data.values, TR=func.TR, osf=1000)
    """
    
    def __init__(onsets, data, hrf_pars=None, TR=None, osf=1, exp_type='event', block_length=None, amplitude=None):
        
        # %%
        # instantiate 
        self.onsets         = onsets
        self.data           = data
        self.hrf_pars       = hrf_pars
        self.TR             = TR
        self.osf            = osf
        self.exp_type       = exp_type
        self.block_length   = block_length
        self.amplitude      = amplitude

        # %%
        # make the stimulus vectors
        self.stims = make_stimulus_vector(self.onsets, scan_length=self.data.shape[0], osf=self.osf, type=self.exp_type)

        # %%
        # define HRF
        dt = 1/self.osf
        self.time_points = np.linspace(0, 25, np.rint(float(25)/dt).astype(int))

        if self.hrf_pars:
            self.hrf = double_gamma(self.time_points, lag=self.hrf_pars['lag'], a2=self.hrf_pars['a2'], b1=self.hrf_pars['b1'], b2=self.hrf_pars['b2'], c=self.hrf_pars['c'], scale=self.hrf_pars['scale'])
        else:
            self.hrf = double_gamma(self.time_points, lag=6)
        
        # %%
        # convolve stimulus vectors
        self.convolved_vectors = convolve_hrf(self.hrf, self.stims, osf=self.osf, num_points=self.data.shape[0])
        
        # %%
        # Fit all
        self.betas, self.X_conv = fit_first_level(self.convolved_vectors, self.data.values)


def double_gamma(x, lag=6, a2=12, b1=0.9, b2=0.9, c=0.35, scale=True):
    """double_gamma

    Create a double gamma hemodynamic response function (HRF).

    Parameters
    ----------
    x: numpy.ndarray
        timepoints along the HRF
    lag: int, optional
        duration until peak of HRF is reached, by default 6
    a2: int, optional
        second determinant of the HRF drop, by default 12
    b1: float, optional
        first determinant of HRF rise, by default 0.9
    b2: float, optional
        second determinant of HRF rise, by default 0.9
    c: float, optional
        constant for HRF drop, by default 0.35
    scale: bool, optional
        normalize course of HRF, by default True

    Returns
    ----------
    numpy.ndarray
        HRF across given timepoints with shape (,`x.shape[0]`)

    Example
    ----------
    >>> dt = 1
    >>> time_points = np.linspace(0,36,np.rint(float(36)/dt).astype(int))
    >>> hrf_custom = linescanning.glm.double_gamma(time_points, lag=6)
    >>> hrf_custom = hrf_custom[np.newaxis,...]
    """
    a1 = lag
    d1 = a1 * b1
    d2 = a2 * b2
    hrf = np.array([(t/(d1))**a1 * np.exp(-(t-d1)/b1) - c *
                   (t/(d2))**a2 * np.exp(-(t-d2)/b2) for t in x])

    if scale:
        hrf = (1 - hrf.min()) * (hrf - hrf.min()) / \
            (hrf.max() - hrf.min()) + hrf.min()
    return hrf


def make_stimulus_vector(onset_df, scan_length=None, TR=0.105, osf=None, type='event', block_length=None, amplitude=None):
    """make_stimulus_vector

    Creates a stimulus vector for each of the conditions found in `onset_df`. You can account for onset times being in decimal using the oversampling factor `osf`. This would return an upsampled stimulus vector which should be convolved with an equally upsampled HRF. This can be ensured by using the same `osf` in :func:`linescanning.glm.double_gamma`.

    Parameters
    ----------
    onset_df: pandas.DataFrame
        onset times as read in with :class:`linescanning.utils.ParseExpToolsFile`
    scan_length: float, optional
        length of the , by default None
    TR: float, optional
        Repetition time, by default 0.105. Will be used to calculate the required length of the stimulus vector
    osf: [type], optional
        Oversampling factor used to account for decimal onset times, by default None
    type: str, optional
        Use block design of event-related design, by default 'event'. If set to 'block', `block_length` is required.
    block_length: int, optional
        Duration of block in seconds, by default None
    amplitude: int, list, optional
        Amplitude to be used when creating the stimulus vector, by default None. If nothing is specified, the amplitude will be set to '1', like you would in a regular FSL 1-/3-column file. If you want variable amplitudes for different events for in a simulation, you can specify a list with an equal length to the number of events present in `onset_df`.

    Returns
    ----------
    dict
        Dictionary collecting numpy array stimulus vectors for each event present in `onset_df` under the keys <event name>

    Raises
    ----------
    ValueError
        `onset_df` should contain event names
    ValueError
        if multiple amplitudes are requested but the length of `amplitude` does not match the number of events 
    ValueError
        `block_length` should be an integer

    Example
    ----------
    >>> from linescanning import utils
    >>> from linescanning import glm
    >>> exp_file = 'path/to/exptools2_file.tsv'
    >>> exp_df = utilsParseExpToolsFile(exp_file, subject=1, run=1)
    >>> times = exp_df.get_onset_df()
    >>> # oversample with factor 1000 to get rid of 3 decimals in onset times
    >>> osf = 1000
    >>> # make stimulus vectors
    >>> stims = glm.make_stimulus_vector(times, scan_length=400, osf=osf, type='event')
    >>> stims
    {'left': array([0., 0., 0., ..., 0., 0., 0.]),
    'right': array([0., 0., 0., ..., 0., 0., 0.])}
    """

    # check if we should reset or not
    try:
        onset_df = onset_df.reset_index()
    except:
        onset_df = onset_df    

    # check conditions we have
    try:
        names_cond = onset_df['event_type'].unique()
        names_cond.sort()
    except:
        raise ValueError('Could not extract condition names; are you sure you formatted the dataframe correctly?')

    # check if we got multiple amplitudes
    if isinstance(amplitude, np.ndarray):
        ampl_array = amplitude
    elif isinstance(amplitude, list):
        ampl_array = np.array(amplitude)
    else:
        ampl_array = False

    # loop through unique conditions
    stim_vectors = {}
    for idx,condition in enumerate(names_cond):

        if isinstance(ampl_array, np.ndarray):
            if ampl_array.shape[0] == names_cond.shape[0]:
                ampl = amplitude[idx]
                print(f"Amplitude for event '{names_cond[idx]}' = {round(ampl,2)}")
            else:
                raise ValueError(f"Nr of amplitudes ({ampl_array.shape[0]}) does not match number of conditions ({names_cond.shape[0]})")
        else:
            ampl = 1

        Y = np.zeros(int((scan_length*TR)*osf))

        if type == "event":
            for rr, ii in enumerate(onset_df['onset']):
                if onset_df['event_type'][rr] == condition:
                    try:
                        Y[int(ii*osf)] = ampl
                    except:
                        warnings.warn(f"Warning: could not include event {rr} with t = {ii}. Probably experiment continued after functional acquisition")
                        
        elif type == 'block':

            if not isinstance(block_length, int):
                raise ValueError("Please specify the length of the block in seconds (integer)")

            for rr, ii in enumerate(onset_df['onset']):
                if onset_df['event_type'][rr] == condition:
                    Y[int(ii*osf):int((ii+block_length)*osf)] = ampl

        stim_vectors[condition] = Y

    return stim_vectors


def convolve_hrf(hrf, stim_v, make_figure=False, osf=None, scan_length=None, xkcd=False, add_array1=None, add_array2=None, regressors=None):
    """convolve_hrf

    Convolve :func:`linescanning.glm.double_gamma` with :func:`linescanning.glm.make_stimulus_vector`. There's an option to plot the result in a nice overview figure, though python-wise it's not the prettiest.. 

    Parameters
    ----------
    hrf: numpy.ndarray
        HRF across given timepoints with shape (,`x.shape[0]`)
    stim_v: [type]
        Stimulus vector as per :func:`linescanning.glm.make_stimulus_vector`
    make_figure: bool, optional
        Create overview figure of HRF, stimulus vector, and convolved stimulus vector, by default False
    osf: [type], optional
        Oversampling factor used to account for decimal onset times, by default None
    scan_length: int
        number of volumes in `data` (= `scan_length` in :func:`linescanning.glm.make_stimulus_vector`)        
    xkcd: bool, optional
        Plot the figre in XKCD-style (cartoon), by default False
    add_array1: numpy.ndarray, optional
        additional stimulus vector to be plotted on top of `stim_v`, by default None
    add_array2: numpy.ndarray, optional
        additional **convolved** stimulus vector to be plotted on top of `stim_v`, by default None
    regressors: pandas.DataFrame
        add a bunch of regressors with shape <time,voxels> to the design matrix. Should be in the dimensions of the functional data, not the oversampled..

    Returns
    ----------
    matplotlib.plot
        if `make_figure=True`, a figure will be displayed

    pandas.DataFrame
        if `osf > 1`, then resampled stimulus vector DataFrame is returned. If not, the convolved stimulus vectors are returned in  a dataframe as is

    Example
    ----------
    >>> from linescanning.glm import convolve_hrf
    >>> convolved_stim_vector_left = convolve_hrf(hrf_custom, stims, make_figure=True, xkcd=True) # creates figure too
    >>> convolved_stim_vector_left = convolve_hrf(hrf_custom, stims) # no figure
    """

    def plot(stim_v, hrf, convolved, add_array1=None, add_array2=None):

        plt.figure(figsize=(15, 6))
        plt.subplot2grid((3, 3), (0, 0), colspan=2)
        plt.plot(stim_v, color="#B1BDBD")
        plt.ylim((-.5, 1.5))
        plt.ylabel('Activity (A.U.)', fontsize=12)
        plt.xlabel('Time (ms)', fontsize=12)
        plt.title('Events', fontsize=16)

        if isinstance(add_array1, np.ndarray):
            plt.plot(add_array1, color="#E41A1C")

        plt.subplot2grid((3, 3), (0, 2), rowspan=2)
        plt.axhline(0, color='black', lw=0.5)
        plt.plot(hrf, color="#F4A460")
        plt.title('HRF', fontsize=16)
        # plt.xlim(0, 24)
        plt.xlabel("Time (ms)", fontsize=12)

        plt.subplot2grid((3, 3), (1, 0), colspan=2)
        plt.axhline(0, color='black', lw=0.5)
        plt.plot(convolved, color="#E1C16E")
        plt.title('Convolved stimulus-vector', fontsize=16)
        plt.ylabel('Activity (A.U.)', fontsize=12)
        plt.xlabel('Time (ms)', fontsize=12)

        if isinstance(add_array2, np.ndarray):
            plt.plot(add_array2, "#D2691E")

        plt.tight_layout()
        sns.despine(offset=10)

    # convolve stimulus vectors
    convolved_vectors = {}
    for ii in list(stim_v.keys()):
        convolved_vectors[ii] = np.convolve(stim_v[ii], hrf, 'full')[:stim_v[ii].shape[0]]

        # resample if osf was larger than 1. If 1, we can use the vector as is
        if osf > 1:
            convolved_vectors[ii] = resample_stim_vector(
                convolved_vectors[ii], scan_length)

        if make_figure:
            if xkcd:
                with plt.xkcd():
                    plot(stim_v[ii], hrf, convolved_vectors[ii], add_array1=add_array1, add_array2=add_array2)
            else:
                plot(stim_v[ii], hrf, convolved_vectors[ii])
            plt.show()

    df_stim = pd.DataFrame(convolved_vectors)
    if regresssors:
        return pd.concat([df_stim, regressors], axis=1)
    else:
        return df_stim


def resample_stim_vector(convolved_array, scan_length, interpolate='nearest'):
    """resample_stim_vector

    Resample the oversampled stimulus vector back in to functional time domain

    Parameters
    ----------
    convolved_array: numpy.ndarray
        oversampled convolved stimulus vector as per :func:`linescanning.glm.convolve_hrf`
    scan_length: int
        number of volumes in `data` (= `scan_length` in :func:`linescanning.glm.make_stimulus_vector`)
    interpolate: str, optional
        interpolation method, by default 'nearest'

    Returns
    ----------
    numpy.ndarray
        convolved stimulus vector in time domain that matches the fMRI acquisition

    Example
    ----------
    >>> from linescanning.glm import resample_stim_vector
    >>> convolved_stim_vector_left_ds = resample_stim_vector(convolved_stim_vector_left, <`scan_length`>)
    """

    interpolated = interp1d(np.arange(len(convolved_array)), convolved_array, kind=interpolate, axis=0, fill_value='extrapolate')
    downsampled = interpolated(np.linspace(0, len(convolved_array), scan_length))

    return downsampled


def fit_first_level(stim_vector, voxel_signal, make_figure=False, copes=None, xkcd=False, plot_vox=None):
    """fit_first_level

    First level models are, in essence, linear regression models run at the level of a single session or single subject. The model is applied on a voxel-wise basis, either on the whole brain or within a region of interest. The  timecourse of each voxel is regressed against a predicted BOLD response created by convolving the haemodynamic response function (HRF) with a set of predictors defined within the design matrix (source: https://nilearn.github.io/glm/first_level_model.html)

    Parameters
    ----------
    stim_vector: numpy.ndarray
        output from :func:`linescanning.glm.resample_stim_vector`; convolved stimulus vector in fMRI-acquisition time domain
    voxel_signal: numpy.ndarray
        <time,voxels> numpy array; same input as **data** from :func:`linescanning.glm.make_stimulus_vector`
    make_figure: bool, optional
        Create a figure of best-voxel fit, by default False
    copes: [type], optional
        [description], by default None
    xkcd: bool, optional
        Plot the figre in XKCD-style (cartoon), by default False
    plot_vox: int, optional
        Instead of plotting the best-fitting voxel, specify which voxel to plot the timecourse and fit of, by default None

    Returns
    ----------
    numpy.ndarray
        betas for each voxel for the intercept and the number of stim_vectors used (in case you also add regressors)

    numpy.ndarray
        the design matrix `X_conv`

    Example
    ----------
    >>> from linescanning.glm import fit_first_level
    >>> betas_left,x_conv_left = fit_first_level(convolved_stim_vector_left_ds, data, make_figure=True, xkcd=True)
    """

    if stim_vector.ndim == 1:
        # Add back a singleton axis (which was removed before downsampling)
        # otherwise stacking will give an error
        stim_vector = stim_vector[:, np.newaxis]

    if voxel_signal.ndim == 1:
        # Add back a singleton axis (which was removed before downsampling)
        # otherwise stacking will give an error
        voxel_signal = voxel_signal[:, np.newaxis]

    if stim_vector.shape[0] != voxel_signal.shape[0]:
        stim_vector = stim_vector[:voxel_signal.shape[0],:]

    # create design matrix with intercept
    intercept = np.ones((stim_vector.shape[0], 1))
    intercept_df = pd.DataFrame(intercept, columns=['intercept'])
    X_matrix = pd.concat([intercept_df, stim_vector], axis=1)
    X_conv = X_matrix.values
    print(X_conv.shape)

    if copes is None:
        C = np.identity(X_conv.shape[1])

    # print(voxel_signal.shape)

    # np.linalg.pinv(X) = np.inv(X.T @ X) @ X
    betas_conv  = np.linalg.inv(X_conv.T @ X_conv) @ X_conv.T @ voxel_signal

    # calculate some other relevant parameters
    cope        = C @ betas_conv
    r           = voxel_signal - X_conv@betas_conv
    dof         = X_conv.shape[0] - np.linalg.matrix_rank(X_conv)
    sig2        = np.sum(r**2,axis=0)/dof
    varcope     = np.outer(C@np.diag(np.linalg.inv(X_conv.T@X_conv))@C.T,sig2)

    # calculate t-stats
    tstat = cope / np.sqrt(varcope)

    def best_voxel(betas):
        return np.where(betas == np.amax(betas))[0][0]

    if not plot_vox:
        best_vox = best_voxel(betas_conv[-1])
    else:
        best_vox = plot_vox

    print(f"max tstat (vox {best_vox}) = {round(tstat[-1,best_vox],2)}")
    print(f"max beta (vox {best_vox}) = {round(betas_conv[-1,best_vox],2)}")

    if make_figure:

        if xkcd:
            # with plt.xkcd():
            #     plot(voxel_signal[:,best_vox], X_conv, betas_conv[:,best_vox])
            num_stims
            utils.LazyPlot([voxel_signal[:, best_vox], X_conv@betas_conv[:,best_vox]],
                           y_label="Activity (A.U.)",
                           x_label="volumes",
                           title=f"Model fit vox {best_vox}",
                           labels=['True signal', 'Event signal'],
                           figsize=(20,5),
                           xkcd=True)

        else:
            utils.LazyPlot([voxel_signal[:, best_vox], X_conv@betas_conv[:, best_vox]],
                           y_label="Activity (A.U.)",
                           x_label="volumes",
                           title=f"Model fit vox {best_vox}",
                           labels=['True signal', 'Event signal'],
                           figsize=(20,5))

    return betas_conv,X_conv

