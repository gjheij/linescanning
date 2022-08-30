from linescanning.plotting import LazyPlot
from nilearn.glm.first_level import first_level
from nilearn.glm.first_level import hemodynamic_models 
from nilearn import plotting
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import seaborn as sns
import warnings

class GenericGLM():
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
    osf: int, optional
        Oversampling factor used to account for decimal onset times, by default None. The larger this factor, the more accurate decimal onset times will be processed, but also the bigger your upsampled convolved becomes, which means convolving will take longer.   
    type: str, optional
        Use block design of event-related design, by default 'event'. If set to 'block', `block_length` is required.
    block_length: int, optional
        Duration of block in seconds, by default None
    amplitude: int, list, optional
        Amplitude to be used when creating the stimulus vector, by default None. If nothing is specified, the amplitude will be set to '1', like you would in a regular FSL 1-/3-column file. If you want variable amplitudes for different events for in a simulation, you can specify a list with an equal length to the number of events present in `onset_df`.
    regressors: pandas.DataFrame, numpy.ndarray, optional
        Add a bunch of regressors to the design
    make_figure: bool, optional
        Create overview figure of HRF, stimulus vector, and convolved stimulus vector, by default False
    scan_length: int
        number of volumes in `data` (= `scan_length` in :func:`linescanning.glm.make_stimulus_vector`)        
    xkcd: bool, optional
        Plot the figre in XKCD-style (cartoon), by default False    
    plot_vox: int, optional
        Instead of plotting the best-fitting voxel, specify which voxel to plot the timecourse and fit of, by default None
    plot_event: str, int, list, optional
        If a larger design matrix was inputted with multiple events, you can specify here the name of the event you'd like to plot the betas from. It also accepts a list of indices of events to plot, so you could plot the first to events by specifying `plot_event=[1,2]`. Remember, the 0th index is the intercept! By default we'll plot the event right after the intercept
    contrast_matrix: numpy.ndarray, optional
        contrast array for the event regressors. If none, we'll create a contrast matrix that estimates the effect of each regressor and the baseline
    nilearn: bool, optional
        use nilearn implementation of `FirstLevelModel` (True) or bare python (False). The later gives easier access to betas, while the former allows implementation of AR-noise models.

    Returns
    ----------
    dict
        Dictionary collecting outputs under the following keys

        * betas: <n_regressors (+intercept), n_voxels>  beta values
        * tstats: <n_regressors (+intercept), n_voxels> t-statistics (FSL-way)
        * x_conv: <n_timepoints, n_regressors (+intercept)> design matrix
        * resids: <n_timepoints, n_voxels> residuals>

    matplotlib.pyplot
        plots along the process if `make_figure=True`        

    Example
    ----------
    >>> # import modules
    >>> from linescanning.glm import GenericGLM
    >>> from linescanning import dataset

    >>> # define file with fMRI-data and the output from Exptools2
    >>> func_file = "some_func_file.mat"
    >>> exp_file = "some_exp_file.tsv"

    >>> # load in functional data
    >>> obj = dataset.Dataset(
    >>>     func_file, 
    >>>     exp_file=exp_file,
    >>>     subject=1, 
    >>>     run=1, 
    >>>     deleted_first_timepoints=200, 
    >>>     deleted_last_timepoints=200)
    
    >>> # fetch HP-filtered, percent-signal changed data
    >>> data = obj.fetch_fmri()
    >>> onsets = obj.fetch_onsets()

    >>> # do the fitting
    >>> fitting = GenericGLM(onsets, data.values, TR=func.TR, osf=1000)

    Notes
    ----------
    For `FirstLevelModel` to work with our type of data, I had to add the following to `https://github.com/nilearn/nilearn/blob/main/nilearn/glm/first_level/first_level.py#L683`:

    ```python
    for output_type_ in output_types:
        estimate_ = getattr(contrast, output_type_)()
        
        if return_type == "imgs":
            # Prepare the returned images
            output = self.masker_.inverse_transform(estimate_)
            contrast_name = str(con_vals)
            output.header['descrip'] = (
                '%s of contrast %s' % (output_type_, contrast_name))
            outputs[output_type_] = output
        else:
            output = estimate_
            outputs[output_type_] = output

    ```

    This ensures we're getting an array back, rather than a nifti-image for our statistics
    """
    
    def __init__(
        self, 
        onsets, 
        data, 
        hrf_pars=None, 
        TR=None, 
        osf=1, 
        contrast_matrix=None, 
        exp_type='event', 
        block_length=None, 
        amplitude=None, 
        regressors=None, 
        make_figure=False, 
        xkcd=False, 
        plot_event=[1, 2], 
        plot_vox=None, 
        verbose=False, 
        nilearn=False, 
        derivative=False, 
        dispersion=False, 
        add_intercept=True,
        cmap='inferno'):
        
        # instantiate 
        self.onsets             = onsets
        self.hrf_pars           = hrf_pars
        self.TR                 = TR
        self.osf                = osf
        self.exp_type           = exp_type
        self.block_length       = block_length
        self.amplitude          = amplitude
        self.regressors         = regressors
        self.make_figure        = make_figure
        self.xkcd               = xkcd
        self.plot_event         = plot_event
        self.plot_vox           = plot_vox
        self.verbose            = verbose
        self.contrast_matrix    = contrast_matrix
        self.nilearn_method     = nilearn
        self.dispersion         = dispersion
        self.derivative         = derivative
        self.add_intercept      = add_intercept
        self.cmap               = cmap

        if isinstance(data, np.ndarray):
            self.data = data.copy()
        elif isinstance(data, pd.DataFrame):
            self.data = data.values
        else:
            raise ValueError("Data must be 'np.ndarray' or 'pandas.DataFrame'")
        
        # make the stimulus vectors
        if verbose:
            print("Creating stimulus vector(s)")
        self.stims = make_stimulus_vector(self.onsets, scan_length=self.data.shape[0], osf=self.osf, type=self.exp_type, TR=self.TR)

        # define HRF
        if verbose:
            print("Defining HRF")

        if isinstance(hrf_pars, str):
            if hrf_pars == "glover":
                self.hrf = glover_hrf(osf=osf, TR=self.TR, dispersion=self.dispersion, derivative=self.derivative)
        elif isinstance(hrf_pars, np.ndarray):
            self.hrf = [hrf_pars]
        else:
            dt = 1/self.osf
            self.time_points = np.linspace(0, 25, np.rint(float(25)/dt).astype(int))
            self.hrf = [double_gamma(self.time_points, lag=6)]
            
        
        # convolve stimulus vectors
        if verbose:
            print("Convolve stimulus vectors with HRF")

        self.stims_convolved = convolve_hrf(self.hrf, self.stims, make_figure=self.make_figure, xkcd=self.xkcd)

        if self.osf > 1:
            if verbose:
                print("Resample convolved stimulus vectors")
            self.stims_convolved_resampled = resample_stim_vector(self.stims_convolved, self.data.shape[0])
        else:
            self.stims_convolved_resampled = self.stims_convolved.copy()

        self.condition_names = list(self.stims_convolved_resampled.keys())
        
        # finalize design matrix (with regressors)
        if verbose:
            print("Creating design matrix")

        self.design = first_level_matrix(self.stims_convolved_resampled, regressors=self.regressors, add_intercept=self.add_intercept)

        if self.make_figure:
            self.plot_design_matrix()
        
        # Fit all
        if verbose:
            print("Running fit")
        
        if self.nilearn_method:
            # we're going to hack Nilearn's FirstLevelModel to be compatible with our line-data. First, we specify the model as usual
            self.fmri_glm = first_level.FirstLevelModel(
                t_r=self.TR,
                noise_model='ar1',
                standardize=False,
                hrf_model='spm',
                drift_model='cosine',
                high_pass=.01)

            # Normally, we'd run `fmri_glm = fmri_glm.fit()`, but because this requires nifti-like inputs, we run `run_glm` outside of that function to get the labels:
            if isinstance(data, pd.DataFrame):
                data = data.values
            elif isinstance(data, np.ndarray):
                data = data.copy()
            else:
                raise ValueError(f"Unknown input type {type(data)} for functional data. Must be pd.DataFrame or np.ndarray [time, voxels]")

            self.labels, self.results = first_level.run_glm(data, self.design, noise_model='ar1')

            # Then, we inject this into the `fmri_glm`-class so we can compute contrasts
            self.fmri_glm.labels_    = [self.labels]
            self.fmri_glm.results_   = [self.results]

            # insert the design matrix:
            self.fmri_glm.design_matrices_ = []
            self.fmri_glm.design_matrices_.append(self.design)

            # Then we specify our contrast matrix:
            if self.contrast_matrix == None:
                if self.verbose:
                    print("Defining standard contrast matrix")
                matrix                  = np.eye(len(self.condition_names))
                icept                   = np.zeros((len(self.condition_names), 1))
                matrix                  = np.hstack((icept, matrix)).astype(int)
                self.contrast_matrix    = matrix.copy()

                self.conditions = {}
                for idx, name in enumerate(self.condition_names):
                    self.conditions[name] = self.contrast_matrix[idx, ...]

            if self.verbose:
                print("Computing contrasts")
            self.tstats = []
            for event in self.conditions:
                tstat = self.fmri_glm.compute_contrast(
                    self.conditions[event], 
                    stat_type='t', 
                    output_type='stat', 
                    return_type=None)
                self.tstats.append(tstat)

            self.tstats = np.array(self.tstats)
            
        else:
            self.results = fit_first_level(self.design, self.data, make_figure=self.make_figure, xkcd=self.xkcd, plot_vox=self.plot_vox, plot_event=self.plot_event, cmap=self.cmap)

    def plot_contrast_matrix(self, save_as=None):
        if self.nilearn_method:
            fig,axs = plt.subplots(figsize=(10,10))
            plotting.plot_contrast_matrix(self.contrast_matrix, design_matrix=self.design, ax=axs)

            if save_as:
                fig.savefig(save_as)
        else:
            raise NotImplementedError("Can't use this function without nilearn-fitting. Set 'nilearn=True'")

    def plot_design_matrix(self, save_as=None):
        fig,axs = plt.subplots(figsize=(10,10))
        plotting.plot_design_matrix(self.design, ax=axs)

        if save_as:
            fig.savefig(save_as)

def glover_hrf(osf=1, TR=0.105, dispersion=False, derivative=False, time_length=25):

    # osf factor is different in `hemodynamic_models`
    # osf /= 10

    # set kernel
    hrf_kernel = []
    hrf = hemodynamic_models.glover_hrf(TR, oversampling=osf, time_length=time_length)
    hrf /= hrf.max()

    hrf_kernel.append(hrf)

    if derivative:
        tderiv_hrf = hemodynamic_models.glover_time_derivative(tr=TR, oversampling=osf, time_length=time_length)
        tderiv_hrf /= tderiv_hrf.max()
        hrf_kernel.append(tderiv_hrf)

    if dispersion:
        tdisp_hrf = hemodynamic_models.glover_dispersion_derivative(TR, oversampling=osf, time_length=time_length)
        tdisp_hrf /= tdisp_hrf.max()
        hrf_kernel.append(tdisp_hrf)

    return hrf_kernel

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


def convolve_hrf(hrf, stim_v, make_figure=False, xkcd=False):
    """convolve_hrf

    Convolve :func:`linescanning.glm.double_gamma` with :func:`linescanning.glm.make_stimulus_vector`. There's an option to plot the result in a nice overview figure, though python-wise it's not the prettiest.. 

    Parameters
    ----------
    hrf: numpy.ndarray
        HRF across given timepoints with shape (,`x.shape[0]`)
    stim_v: numpy.ndarray, list
        Stimulus vector as per :func:`linescanning.glm.make_stimulus_vector` or numpy array containing one stimulus vector (e.g., a *key* from :func:`linescanning.glm.make_stimulus_vector`)
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

    def plot(stim_v, hrf, convolved, xkcd=False):

        fig = plt.figure(figsize=(20,6))
        gs = fig.add_gridspec(2, 2, width_ratios=[20, 10], hspace=0.7)

        ax0 = fig.add_subplot(gs[0,0])
        LazyPlot(
            stim_v, 
            color="#B1BDBD", 
            axs=ax0,
            title="Events",
            y_lim=[-.5, 1.5], 
            x_label='Time (*osf)',
            y_label='Activity (A.U.)', 
            xkcd=xkcd,
            font_size=16)
        
        # check if we got derivatives; if so, select first element (= standard HRF)
        if isinstance(convolved, list):
            convolved = np.array(convolved)
        
        if convolved.shape[-1] > 1:
            convolved = convolved[:,0]

        ax1 = fig.add_subplot(gs[1, 0])
        LazyPlot(
            convolved,
            axs=ax1,
            title="Convolved stimulus-vector",
            x_label='Time (*osf)',
            y_label='Activity (A.U.)',
            xkcd=xkcd,
            font_size=16)

        ax2 = fig.add_subplot(gs[:, 1])
        LazyPlot(
            hrf,
            axs=ax2,
            title="HRF", 
            x_label='Time (*osf)',
            xkcd=xkcd,
            font_size=16)

    # check hrf input
    if isinstance(hrf, list):
        hrfs = hrf.copy()
    elif isinstance(hrf, np.ndarray):
        hrfs = [hrf]
    else:
        raise ValueError(f"Unknown input type '{type(hrf)}' for HRF. Must be list or array")

    # convolve stimulus vectors
    if isinstance(stim_v, np.ndarray):

        if len(hrfs) >= 1:
            convolved_stim_vector = np.zeros((stim_v.shape[0], len(hrfs)))
            for ix,rf in enumerate(hrfs):
                convolved_stim_vector[:, ix] = np.convolve(stim_v, rf, 'full')[:stim_v.shape[0]]

        if make_figure:
            plot(stim_v, hrfs[0], convolved_stim_vector, xkcd=xkcd)
            plt.show()

    elif isinstance(stim_v, dict):

        if len(hrfs) >= 1:
            convolved_stim_vector = {}
            for event in list(stim_v.keys()):
                hrf_conv = np.zeros((stim_v[event].shape[0], len(hrf)))
                for ix, rf in enumerate(hrfs):
                    hrf_conv[...,ix] = np.convolve(stim_v[event], rf, 'full')[:stim_v[event].shape[0]]
                
                convolved_stim_vector[event] = hrf_conv
                    
                if make_figure:
                    if xkcd:
                        with plt.xkcd():
                            plot(stim_v[event], hrfs[0], convolved_stim_vector[event])
                    else:
                        plot(stim_v[event], hrfs[0], convolved_stim_vector[event])
                    plt.show()
    else:
        raise ValueError("Data must be 'np.ndarray' or 'dict'")

    return convolved_stim_vector


def resample_stim_vector(convolved_array, scan_length, interpolate='nearest'):
    """resample_stim_vector

    Resample the oversampled stimulus vector back in to functional time domain

    Parameters
    ----------
    convolved_array: dict, numpy.ndarray
        oversampled convolved stimulus vector as per :func:`linescanning.glm.convolve_hrf`
    scan_length: int
        number of volumes in `data` (= `scan_length` in :func:`linescanning.glm.make_stimulus_vector`)
    interpolate: str, optional
        interpolation method, by default 'nearest'

    Returns
    ----------
    dict, numpy.ndarray
        convolved stimulus vector in time domain that matches the fMRI acquisition

    Example
    ----------
    >>> from linescanning.glm import resample_stim_vector
    >>> convolved_stim_vector_left_ds = resample_stim_vector(convolved_stim_vector_left, <`scan_length`>)
    """

    if isinstance(convolved_array, np.ndarray):
        interpolated = interp1d(np.arange(len(convolved_array)), convolved_array, kind=interpolate, axis=0, fill_value='extrapolate')
        downsampled = interpolated(np.linspace(0, len(convolved_array), scan_length))
    elif isinstance(convolved_array, dict):
        downsampled = {}
        for event in list(convolved_array.keys()):
            
            event_arr = convolved_array[event]
            if event_arr.shape[-1] > 1:
                tmp = np.zeros((scan_length, event_arr.shape[-1]))
                for elem in range(event_arr.shape[-1]):
                    data = event_arr[..., elem]
                    interpolated = interp1d(
                        np.arange(len(data)), data, kind=interpolate, axis=0, fill_value='extrapolate')
                    tmp[...,elem] = interpolated(np.linspace(0, len(data), scan_length))
                downsampled[event] = tmp
            else:
                interpolated = interp1d(np.arange(len(convolved_array[event])), convolved_array[event], kind=interpolate, axis=0, fill_value='extrapolate')
                downsampled[event] = interpolated(np.linspace(0, len(convolved_array[event]), scan_length))                
    else:
        raise ValueError("Data must be 'np.ndarray' or 'dict'")

    return downsampled


def first_level_matrix(stims_dict, regressors=None, add_intercept=True, names=None):

    # make dataframe of stimulus vectors
    if isinstance(stims_dict, np.ndarray):
        if names:
            stims = pd.DataFrame(stims_dict, columns=names)
        else:
            stims = pd.DataFrame(stims_dict, columns=[f'event {ii}' for ii in range(stims_dict.shape[-1])])
    elif isinstance(stims_dict, dict):
        # check if we got time/dispersion derivatives
        cols = []
        data = []
        keys = list(stims_dict.keys())
        for key in keys:
            if stims_dict[key].shape[-1] == 1:
                cols.extend([key])
            elif stims_dict[key].shape[-1] == 2:
                cols.extend([key, f'{key}_1st_derivative'])
            elif stims_dict[key].shape[-1] == 3:
                cols.extend([key, f'{key}_1st_derivative', f'{key}_2nd_derivative'])

            data.append(stims_dict[key])

        data = np.concatenate(data, axis=-1)

        stims = pd.DataFrame(data, columns=cols)
    else:
        raise ValueError("Data must be 'np.ndarray' or 'dict'")

    # check if we should add intercept
    if add_intercept:
        intercept = np.ones((stims.shape[0], 1))
        intercept_df = pd.DataFrame(intercept, columns=['intercept'])
        X_matrix = pd.concat([intercept_df, stims], axis=1)
    else:
        X_matrix = stims.copy()

    # check if we should add regressors
    if isinstance(regressors, np.ndarray):
        regressors_df = pd.DataFrame(regressors, columns=[f'regressor {ii}' for ii in range(regressors.shape[-1])])
        return pd.concat([X_matrix, regressors_df], axis=1)
    elif isinstance(regressors, dict):
        regressors_df = pd.DataFrame(regressors)
        return pd.concat([X_matrix, regressors_df], axis=1)
    else:
        return X_matrix


def fit_first_level(stim_vector, data, make_figure=False, copes=None, xkcd=False, plot_vox=None, plot_event=1, verbose=False, cmap='inferno', **kwargs):
    """fit_first_level

    First level models are, in essence, linear regression models run at the level of a single session or single subject. The model is applied on a voxel-wise basis, either on the whole brain or within a region of interest. The  timecourse of each voxel is regressed against a predicted BOLD response created by convolving the haemodynamic response function (HRF) with a set of predictors defined within the design matrix (source: https://nilearn.github.io/glm/first_level_model.html)

    Parameters
    ----------
    stim_vector: pandas.DataFrame, numpy.ndarray
        either the output from :func:`linescanning.glm.resample_stim_vector` (convolved stimulus vector in fMRI-acquisition time domain) or a pandas.DataFrame containing the full design matrix as per the output of :func:`linescanning.glm.first_level_matrix`.s
    data: numpy.ndarray
        <time,voxels> numpy array; same input as **data** from :func:`linescanning.glm.make_stimulus_vector`
    make_figure: bool, optional
        Create a figure of best-voxel fit, by default False
    copes: [type], optional
        [description], by default None
    xkcd: bool, optional
        Plot the figre in XKCD-style (cartoon), by default False
    plot_vox: int, optional
        Instead of plotting the best-fitting voxel, specify which voxel to plot the timecourse and fit of, by default None
    plot_event: str, int, list, optional
        If a larger design matrix was inputted with multiple events, you can specify here the name of the event you'd like to plot the betas from. It also accepts a list of indices of events to plot, so you could plot the first to events by specifying `plot_event=[1,2]`. Remember, the 0th index is the intercept! By default we'll plot the event right after the intercept

    Returns
    ----------
    numpy.ndarray
        betas for each voxel for the intercept and the number of stim_vectors used (in case you also add regressors)

    numpy.ndarray
        the design matrix `X_conv`

    Example
    ----------
    >>> from linescanning.glm import fit_first_level
    >>> betas_left,x_conv_left = fit_first_level(convolved_stim_vector_left_ds, data, make_figure=True, xkcd=True) # plots first event
    >>> betas_left,x_conv_left = fit_first_level(convolved_stim_vector_left_ds, data, make_figure=True, xkcd=True, plot_events=[1,2]) # plots first two events
    """

    # add intercept if input is simple numpy array. 
    if isinstance(stim_vector, np.ndarray):
        if stim_vector.ndim == 1:
            stim_vector = stim_vector[:, np.newaxis]

        if data.ndim == 1:
            data = data[:, np.newaxis]

        if stim_vector.shape[0] != data.shape[0]:
            stim_vector = stim_vector[:data.shape[0],:]

        # create design matrix with intercept
        intercept = np.ones((data.shape[0], 1))
        intercept_df = pd.DataFrame(intercept, columns=['intercept'])
        X_matrix = pd.concat([intercept_df, stim_vector], axis=1)
    else:
        # everything should be ready to go if 'first_level_matrix' was used
        X_matrix = stim_vector.copy()
        intercept = np.ones((data.shape[0], 1))

        if data.ndim == 1:
            data = data[:, np.newaxis]

        if stim_vector.shape[0] != data.shape[0]:
            stim_vector = stim_vector[:data.shape[0], :]

    X_conv = X_matrix.values.copy()

    if copes == None:
        C = np.identity(X_conv.shape[1])

    # np.linalg.pinv(X) = np.inv(X.T @ X) @ X
    betas_conv  = np.linalg.inv(X_conv.T @ X_conv) @ X_conv.T @ data

    # calculate some other relevant parameters
    cope        = C @ betas_conv
    resids      = data - X_conv@betas_conv
    dof         = X_conv.shape[0] - np.linalg.matrix_rank(X_conv)
    sig2        = np.sum(resids**2,axis=0)/dof
    varcope     = np.outer(C@np.diag(np.linalg.inv(X_conv.T@X_conv))@C.T,sig2)

    # calculate t-stats
    tstat = cope / np.sqrt(varcope)

    def best_voxel(betas):
        return np.where(betas == np.amax(betas))[0][0]

    if not plot_vox:
        best_vox = best_voxel(betas_conv[-1])
    else:
        best_vox = plot_vox

    if betas_conv.ndim == 1:
        betas_conv = betas_conv[...,np.newaxis]
    
    if verbose:
        print(f"max tstat (vox {best_vox}) = {round(tstat[-1,best_vox],2)}")
        print(f"max beta (vox {best_vox}) = {round(betas_conv[-1,best_vox],2)}")

    if make_figure:
        
        markers = ['.']
        colors = ["#cccccc"]
        linewidth = [0.5]

        # you can specify to plot multiple events!
        if isinstance(plot_event, int):
            # get the correct stimulus vector and betas for that vector
            use_stim = X_conv[:,plot_event][...,np.newaxis]
            beta = np.array((1, betas_conv[plot_event, best_vox]))

            # to avoid annoying indexing, add intercept here again
            conv = np.hstack((intercept, use_stim))
            signals = [data[:, best_vox], conv@beta]
            labels = ['True signal', 'Event signal']
            markers.append(None)
            colors.append("r")
            linewidth.append(2)
        elif isinstance(plot_event, list):
            signals = [data[:, best_vox]]
            labels = ['True signal']
            for ii in plot_event:
                # get the correct stimulus vector and betas for that vector
                use_stim = X_conv[:,ii][...,np.newaxis]
                beta = np.array((1, betas_conv[ii, best_vox]))

                # to avoid annoying indexing, add intercept here again
                conv = np.hstack((intercept, use_stim))

                signals.append(conv@beta)
                labels.append(f'Event {ii}')
                markers.append(None)
                linewidth.append(2)

            colors = [*colors, *sns.color_palette(cmap, len(plot_event))]

        else:
            raise NotImplementedError("Im lazy.. Please use indexing for now")

        LazyPlot(
            signals,
            y_label="Activity (A.U.)",
            x_label="volumes",
            title=f"Model fit vox {best_vox}",
            labels=labels,
            figsize=(20,5),
            font_size=20,
            xkcd=xkcd,
            markers=markers,
            color=colors,
            line_width=linewidth,
            **kwargs)

    return {'betas': betas_conv,
            'x_conv': X_conv,
            'resids': resids,
            'tstats': tstat}

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