from linescanning.plotting import LazyPlot, Defaults, conform_ax_to_obj
from linescanning import utils
from nilearn.glm.first_level import first_level
from nilearn.glm.first_level import hemodynamic_models 
from nilearn import plotting
import numpy as np
import matplotlib as mpl
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
        plot_design=False,
        plot_convolve=False,
        plot_event=None, 
        plot_vox=None, 
        plot_fit=False,
        verbose=False, 
        nilearn=False, 
        derivative=False, 
        dispersion=False, 
        add_intercept=True,
        fit=True,
        copes=None,
        cmap='inferno',
        kw_conv={}
        ):
        
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
        self.regressors         = regressors
        self.cmap               = cmap
        self.run_fit            = fit
        self.copes              = copes
        self.plot_fit           = plot_fit
        self.plot_design        = plot_design
        self.plot_convolve      = plot_convolve
        self.kw_conv            = kw_conv

        if isinstance(data, np.ndarray):
            self.data = data.copy()
        elif isinstance(data, pd.DataFrame):
            self.orig = data.copy()
            self.data = data.values
        else:
            raise ValueError("Data must be 'np.ndarray' or 'pandas.DataFrame'")
        
        # define HRF
        self.hrf = self.define_hrf()

        # make the stimulus vectors
        self.stims = self.define_stimvector()
        
        # convolve stimulus vectors
        self.stims_convolved = self.convolve_stims(**self.kw_conv)

        # resample
        self.stims_convolved_resampled = self.resample_stimvector()

        # get condition names
        self.condition_names = list(self.stims_convolved_resampled.keys())

    def define_hrf(self):

        utils.verbose(f"Defining HRF with option '{self.hrf_pars}'", self.verbose)
        return define_hrf(
            hrf_pars=self.hrf_pars,
            osf=self.osf,
            TR=self.TR,
            dispersion=self.dispersion,
            derivative=self.derivative
        )

    def define_stimvector(self):
        utils.verbose("Creating stimulus vector(s)", self.verbose)
        return make_stimulus_vector(
            self.onsets, 
            scan_length=self.data.shape[0], 
            osf=self.osf, 
            type=self.exp_type, 
            TR=self.TR,
            block_length=self.block_length,
            amplitude=self.amplitude
        )

    def convolve_stims(self, **kwargs):
        utils.verbose("Convolve stimulus vectors with HRF", self.verbose)
        return convolve_hrf(
            self.hrf, 
            self.stims, 
            TR=self.TR,
            osf=self.osf,
            make_figure=self.plot_convolve, 
            xkcd=self.xkcd,
            **kwargs
        )

    def resample_stimvector(self):
        if self.osf > 1:
            utils.verbose("Resample convolved stimulus vectors", self.verbose)
            return resample_stim_vector(self.stims_convolved, self.data.shape[0])
        else:
            return self.stims_convolved.copy()

    def create_design(self, make_figure=False):

        # finalize design matrix (with regressors)
        utils.verbose("Creating design matrix", self.verbose)

        self.design = first_level_matrix(
            self.stims_convolved_resampled, 
            regressors=self.regressors, 
            add_intercept=self.add_intercept
        )

        if make_figure:
            self.plot_design_matrix()
        
    def fit(
        self,
        nilearn_method=False,
        make_figure=False, 
        xkcd=False, 
        plot_vox=None, 
        plot_event=None, 
        cmap="inferno",
        copes=None,
        plot_full_only=False,
        plot_full=False,
        save_as=None,
        **kwargs
        ):

        self.nilearn_method = nilearn_method
        self.xkcd = xkcd
        self.plot_vox = plot_vox
        self.plot_event = plot_event
        self.cmap = cmap
        self.copes = copes
        self.make_figure = make_figure
        self.plot_full_only = plot_full_only
        self.plot_full = plot_full
        self.save_as = save_as

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
            if isinstance(self.data, pd.DataFrame):
                self.data = self.data.values
            elif isinstance(self.data, np.ndarray):
                self.data = self.data.copy()
            else:
                raise ValueError(f"Unknown input type {type(self.data)} for functional data. Must be pd.DataFrame or np.ndarray [time, voxels]")

            self.labels, self.results = first_level.run_glm(self.data, self.design, noise_model='ar1')

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
            if not isinstance(self.plot_event, (str,list)):
                self.plot_event = self.condition_names

            self.results = fit_first_level(
                self.design, 
                self.data, 
                make_figure=self.make_figure, 
                xkcd=self.xkcd, 
                plot_vox=self.plot_vox, 
                plot_event=self.plot_event, 
                cmap=self.cmap,
                verbose=self.verbose,
                copes=self.copes,
                plot_full_only=self.plot_full_only,
                plot_full=self.plot_full,
                add_intercept=self.add_intercept,
                save_as=self.save_as,
                **kwargs
            )

    def plot_contrast_matrix(self, save_as=None):
        if self.nilearn_method:
            cols = list(self.design.columns)
            fig,axs = plt.subplots(figsize=(len(cols),10))
            plotting.plot_contrast_matrix(self.contrast_matrix, design_matrix=self.design, ax=axs)
            conform_ax_to_obj(axs)

            fig,axs = plt.subplots(figsize=(10,10))
            plotting.plot_contrast_matrix(self.contrast_matrix, design_matrix=self.design, ax=axs)

            if save_as:
                fig.savefig(save_as)
        else:
            raise NotImplementedError("Can't use this function without nilearn-fitting. Set 'nilearn=True'")

    def plot_design_matrix(self, save_as=None):
        cols = list(self.design.columns)
        fig,axs = plt.subplots(figsize=(len(cols),10))
        plotting.plot_design_matrix(self.design, ax=axs)
        conform_ax_to_obj(axs)

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

def spm_hrf(osf=1, TR=0.105, dispersion=False, derivative=False, time_length=25):

    # osf factor is different in `hemodynamic_models`
    # osf /= 10

    # set kernel
    hrf_kernel = []
    hrf = hemodynamic_models.spm_hrf(TR, oversampling=osf, time_length=time_length)
    hrf /= hrf.max()

    hrf_kernel.append(hrf)

    if derivative:
        tderiv_hrf = hemodynamic_models.spm_time_derivative(tr=TR, oversampling=osf, time_length=time_length)
        tderiv_hrf /= tderiv_hrf.max()
        hrf_kernel.append(tderiv_hrf)

    if dispersion:
        tdisp_hrf = hemodynamic_models.spm_dispersion_derivative(TR, oversampling=osf, time_length=time_length)
        tdisp_hrf /= tdisp_hrf.max()
        hrf_kernel.append(tdisp_hrf)

    return hrf_kernel

def make_stimulus_vector(
    onset_df, 
    scan_length=None, 
    cov_as_ampl=None,
    TR=0.105, 
    osf=1, 
    type='event', 
    block_length=None, 
    amplitude=None):

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
        names_cond = utils.get_unique_ids(onset_df, id="event_type")
    except:
        raise ValueError('Could not extract condition names; are you sure you formatted the dataframe correctly? Mostly likely you passed the functional dataframe as onset dataframe. Please switch')

    # check if we should use covariate as amplitude
    if isinstance(cov_as_ampl, (bool,list)):
        if isinstance(cov_as_ampl, bool):
            cov_as_ampl = [cov_as_ampl for _ in names_cond]

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
        
        tmp_onsets = utils.select_from_df(onset_df, expression=f"event_type = {condition}")
        set_ampl = True
        if isinstance(cov_as_ampl, list):
            if cov_as_ampl[idx]:
                if "cov" in list(tmp_onsets.columns):
                    ampl = tmp_onsets["cov"].values
                    set_ampl = False
        
        if set_ampl:
            if isinstance(ampl_array, np.ndarray):
                if ampl_array.shape[0] == names_cond.shape[0]:
                    ampl = amplitude[idx]
                    print(f"Amplitude for event '{names_cond[idx]}' = {round(ampl,2)}")
                else:
                    raise ValueError(f"Nr of amplitudes ({ampl_array.shape[0]}) does not match number of conditions ({names_cond.shape[0]})")
            else:
                ampl = 1

        Y = np.zeros(int((scan_length*osf)))
        if type == "event":
            for rr, ii in enumerate(tmp_onsets['onset']):
                try:
                    if isinstance(ampl, (list,np.ndarray)):
                        use_ampl = ampl[rr]
                    else:
                        use_ampl = ampl

                    Y[int((ii*osf)/TR)] = use_ampl
                except:
                    warnings.warn(f"Warning: could not include event {rr} with t = {ii}. Probably experiment continued after functional acquisition")
                        
        elif type == 'block':

            for rr, ii in enumerate(tmp_onsets['onset']):
                Y[int((ii*osf)/TR):int(((ii+block_length)*osf)/TR)] = ampl

        else:
            raise ValueError(f"Event must be 'event' or 'block', not '{type}'")
        stim_vectors[condition] = Y

    return stim_vectors

def define_hrf(
    hrf_pars="glover", 
    TR=0.105,
    osf=1, 
    dispersion=False,
    derivative=False):

    if isinstance(hrf_pars, str):
        if hrf_pars == "glover":
            hrf = glover_hrf(
                osf=osf, 
                TR=TR, 
                dispersion=dispersion, 
                derivative=derivative)
            
        elif hrf_pars == "spm":
            hrf = spm_hrf(
                osf=osf, 
                TR=TR, 
                dispersion=dispersion, 
                derivative=derivative)

        else:
            raise ValueError(f"Invalid option '{hrf_pars}' specified. Must be either 'spm' or 'glover', a list of 3 parameters, a numpy array describing the HRF, or None (for standard double gamma)")
        
    elif isinstance(hrf_pars, np.ndarray):
        hrf = [hrf_pars]
    elif isinstance(hrf_pars, list):

        hrf = np.array(
            [
                np.ones_like(hrf_pars[1])*hrf_pars[0] *
                hemodynamic_models.spm_hrf(
                    tr=TR,
                    oversampling=osf, 
                    time_length=40)[...,np.newaxis],
                hrf_pars[1] *
                hemodynamic_models.spm_time_derivative(
                    tr=TR,
                    oversampling=osf, 
                    time_length=40)[...,np.newaxis],
                hrf_pars[2] *
                hemodynamic_models.spm_dispersion_derivative(
                    tr=TR,
                    oversampling=osf, 
                    time_length=40)[...,np.newaxis]]).sum(
            axis=0)

        hrf = [np.squeeze(hrf)]

    else:
        osf /= TR
        dt = 1/osf
        time_points = np.linspace(0, 25, np.rint(float(25)/dt).astype(int))
        hrf = [double_gamma(time_points, lag=6)]
    
    return hrf

def convolve_hrf(
    hrf, 
    stim_v, 
    TR=1,
    osf=1,
    make_figure=False, 
    xkcd=False,
    time=None,
    *args,
    **kwargs):

    """convolve_hrf

    Convolve :func:`linescanning.glm.double_gamma` with :func:`linescanning.glm.make_stimulus_vector`. There's an option to plot the result in a nice overview figure, though python-wise it's not the prettiest.. 

    Parameters
    ----------
    hrf: numpy.ndarray
        HRF across given timepoints with shape (,`x.shape[0]`)
    stim_v: numpy.ndarray, list
        Stimulus vector as per :func:`linescanning.glm.make_stimulus_vector` or numpy array containing one stimulus vector (e.g., a *key* from :func:`linescanning.glm.make_stimulus_vector`)
    TR: float
        repetition time of acquisition        
    make_figure: bool, optional
        Create overview figure of HRF, stimulus vector, and convolved stimulus vector, by default False
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

    def plot(
        stim_v, 
        hrf, 
        convolved, 
        osf=1, 
        ev=None,
        xkcd=False, 
        time=time,
        **kwargs):

        fig = plt.figure(figsize=(20,6))
        gs = fig.add_gridspec(2, 2, width_ratios=[20, 10], hspace=0.7)

        if isinstance(hrf, np.ndarray):
            hrf = [hrf]

        if stim_v.min() < 0:
            y_lim = [-1,1]
            y_ticks = [-1,0,1]
        else:
            y_lim = [-0.5,1]
            y_ticks = [0,1]

        ax0 = fig.add_subplot(gs[0,0])
        LazyPlot(
            stim_v, 
            color="#B1BDBD", 
            axs=ax0,
            title=f"Events ('{ev}')",
            y_lim=y_lim, 
            x_label='scan volumes (* osf)',
            y_label='magnitude (a.u.)', 
            xkcd=xkcd,
            font_size=16,
            y_ticks=y_ticks,
            *args,
            **kwargs)
        
        # print(list(convolved.T)[0].shape)
        ax1 = fig.add_subplot(gs[1, 0])
        LazyPlot(
            list(convolved.T),
            axs=ax1,
            title="Convolved stimulus-vector",
            x_label='scan volumes (* osf)',
            y_label='magnitude (a.u.)', 
            *args,
            **kwargs)
        
        ax2 = fig.add_subplot(gs[:, 1])
        if not isinstance(time, (list,np.ndarray)):
            time = list(((np.arange(0,hrf[0].shape[0]))/osf)*TR)
        
        labels = [
            "HRF",
            "1st derivative",
            "2nd derivative"
        ]

        LazyPlot(
            hrf,
            xx=time,
            axs=ax2,
            title="HRF", 
            x_label='Time (s)',
            labels=labels[:len(hrf)],
            *args,
            **kwargs)

    
    # check hrf input
    if isinstance(hrf, np.ndarray):
        hrfs = [hrf]
    elif isinstance(hrf, list):
        hrfs = hrf.copy()
    else:
        raise ValueError(f"Unknown input type '{type(hrf)}' for HRF. Must be list or array")

    # convolve stimulus vectors
    if isinstance(stim_v, np.ndarray):

        if len(hrfs) >= 1:
            convolved_stim_vector = np.zeros((stim_v.shape[0], len(hrfs)))
            for ix,rf in enumerate(hrfs):
                convolved_stim_vector[:, ix] = np.convolve(stim_v, rf, 'full')[:stim_v.shape[0]]

        if make_figure:
            plot(stim_v, hrfs, convolved_stim_vector, xkcd=xkcd)
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
                            plot(
                                stim_v[event], 
                                hrfs, 
                                convolved_stim_vector[event], 
                                osf=osf,
                                ev=event,
                                time=time,
                                *args, 
                                **kwargs)
                    else:
                        plot(
                            stim_v[event], 
                            hrfs, 
                            convolved_stim_vector[event], 
                            ev=event,
                            time=time,
                            osf=osf,
                            *args,
                            **kwargs)
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
        regressors_df = pd.DataFrame(regressors, columns=[f'regressor_{ii+1}' for ii in range(regressors.shape[-1])])
        return pd.concat([X_matrix, regressors_df], axis=1)
    elif isinstance(regressors, dict):
        regressors_df = pd.DataFrame(regressors)
        return pd.concat([X_matrix, regressors_df], axis=1)
    elif isinstance(regressors, pd.DataFrame):
        return pd.concat([X_matrix, regressors], axis=1)    
    else:
        return X_matrix


def fit_first_level(
    stim_vector, 
    data, 
    make_figure=False, 
    copes=None, 
    xkcd=False, 
    plot_vox=None, 
    plot_event=1, 
    verbose=False, 
    cmap='inferno', 
    plot_full_only=False,
    plot_full=False,
    add_intercept=True,
    axs=None,
    figsize=(16,4),
    **kwargs):

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

    utils.verbose("Running GLM", verbose)
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.values

    # add intercept if input is simple numpy array. 
    if isinstance(stim_vector, np.ndarray):
        if stim_vector.ndim == 1:
            stim_vector = stim_vector[:, np.newaxis]

        if data.ndim == 1:
            data = data[:, np.newaxis]

        if stim_vector.shape[0] != data.shape[0]:
            stim_vector = stim_vector[:data.shape[0],:]

        # create design matrix with intercept
        if add_intercept:
            intercept = np.ones((data.shape[0], 1))
            intercept_df = pd.DataFrame(intercept, columns=['intercept'])
            X_matrix = pd.concat([intercept_df, stim_vector], axis=1)
        else:
            X_matrix = pd.DataFrame(stim_vector, axis=1)
    else:
        # everything should be ready to go if 'first_level_matrix' was used
        X_matrix = stim_vector.copy()
        if data.ndim == 1:
            data = data[:, np.newaxis]

        if stim_vector.shape[0] != data.shape[0]:
            raise ValueError(f"Unequal shapes between design {stim_vector.shape} and data {data.shape}. Make sure first elements have the same shape")

    X_conv = X_matrix.values.copy()

    # set default to first predictor
    if not isinstance(copes, (int,list,np.ndarray)):
        copes = 1

    # convert to array
    if isinstance(copes, int):
        C = np.zeros((X_conv.shape[1]))
        C[copes] = 1
    elif isinstance(copes, list):
        C = np.array(copes, dtype=int)
    else:
        C = copes.copy()

    # enforce integer
    C = C.astype(int)

    good_shape = np.zeros((C.shape[0], X_conv.shape[1]))
    utils.paste(good_shape, C)
    C = good_shape.copy()

    # force into 2d
    if C.ndim == 1:
        C = C[np.newaxis,...]

    # np.linalg.pinv(X) = np.inv(X.T @ X) @ X
    betas_conv, sse, rank, s = np.linalg.lstsq(X_conv, data, rcond=-1)
    # betas_conv  = np.linalg.inv(X_conv.T @ X_conv) @ X_conv.T @ data
    
    # get full model predictions
    preds = X_conv@betas_conv

    # calculate r2
    tse = (data.shape[0]-1) * np.var(data, axis=0, ddof=1)
    r2 = 1-(sse/tse)

    # loop through contrasts
    tstat = []
    for co_ix in range(C.shape[0]):
        
        # contrast-specific vector
        cope_c = C[co_ix,:]

        # calculate some other relevant parameters
        cope        = cope_c @ betas_conv
        dof         = X_conv.shape[0] - rank
        sigma_hat   = sse/dof
        varcope     = sigma_hat*design_variance(X_conv, which_predictor=cope_c)

        # calculate t-stats
        t_ = cope / np.sqrt(varcope)
        tstat.append(t_)

    if len(tstat) > 0:
        tstat = np.array(tstat)

    if verbose:
        for co_ix in range(C.shape[0]):
            if tstat[co_ix].shape[0]<10:
                print(f"t-stat {C[co_ix,:]}: {tstat[co_ix]}")

    def best_voxel(r2):
        # get max over voxels and find max
        return np.where(r2 == r2.max())[0][0]

    if not plot_vox:
        best_vox = best_voxel(r2)
    else:
        best_vox = plot_vox

    if betas_conv.ndim == 1:
        betas_conv = betas_conv[...,np.newaxis]

    if make_figure:

        if not isinstance(axs, mpl.axes._axes.Axes):
            fig,axs = plt.subplots(figsize=figsize)

        # set defaults for actual datapoints
        markers = ['.']
        colors = ["#cccccc"]
        linewidth = [0.5]
        col_names = X_matrix.columns.to_list()

        if not plot_full_only:

            # make list so we can loop
            if isinstance(plot_event, str):
                plot_event = [plot_event]

            signals = [data[:, best_vox]]
            labels = ['True signal']
            for ev in plot_event:

                # always include intercept
                beta_idx = [ix for ix,ii in enumerate(col_names) if ev in ii or "regressor" in ii or "intercept" in ii]

                # get betas for all voxels
                betas = betas_conv[beta_idx]
                event = X_conv[:,beta_idx]

                # get predictions
                ev_preds = event@betas

                signals.append(ev_preds[:,best_vox])
                labels.append(f"Event '{ev}'")
                markers.append(None)
                linewidth.append(2)

            colors = [*colors, *sns.color_palette(cmap, len(plot_event))]
        else:
            plot_full = True

        # append full model
        if plot_full:
            signals.append(preds[:,best_vox])
            labels.append(f"full model")
            markers.append(None)
            linewidth.append(1)
            colors.append("k")

        if not "title" in list(kwargs.keys()):
            kwargs = utils.update_kwargs(
                kwargs,
                "title",
                f"model fit vox {best_vox+1}/{data.shape[1]} (r2={round(r2[best_vox],4)})",
            )

        pl = LazyPlot(
            signals,
            y_label="Activity (a.u.)",
            x_label="volumes",
            labels=labels,
            axs=axs,
            markers=markers,
            color=colors,
            line_width=linewidth,
            **kwargs)

    return {
        'betas': betas_conv,
        'x_conv': X_conv,
        'tstats': tstat,
        'r2': r2,
        'copes': C,
        'dm': X_matrix
    }

def design_variance(X, which_predictor=1):
    ''' Returns the design variance of a predictor (or contrast) in X.
    
    Parameters
    ----------
    X : numpy array
        Array of shape (N, P)
    which_predictor : int or list/array
        The index of the predictor you want the design var from.
        Note that 0 refers to the intercept!
        Alternatively, "which_predictor" can be a contrast-vector
        (which will be discussed later this lab).
        
    Returns
    -------
    des_var : float
        Design variance of the specified predictor/contrast from X.
    '''
    
    is_single = isinstance(which_predictor, int)
    if is_single:
        idx = which_predictor
    else:
        idx = np.array(which_predictor) != 0
    
    c = np.zeros(X.shape[1])
    c[idx] = 1 if is_single == 1 else which_predictor[idx]
    des_var = c.dot(np.linalg.inv(X.T.dot(X))).dot(c.T)
    return des_var

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

class Posthoc(Defaults):
    """Posthoc

    Run posthoc analysis on the output from `pingouin` ANOVA/ANCOVA or just straight up as pairwise t-tests. During intialization, the plotting arguments are internalized from :class:`linescanning.plotting.Defaults()`. Posthoc test should then be executed using the :func:`linescanning.glm.Posthoc.run_posthoc()` function, which accepts all arguments that :class:`pingouin.pairwise_tukey()` or :class:`pingouin.pairwise_tests()` accept. You can then choose to have the significance bars plotted on a specified `axs`. Note that this only works for relatively simple tests; there's NO support for complicated (nested) data structures such as repeated-measures.

    Parameters
    ----------
    See: :class:`linescanning.plotting.Defaults()`

    Example
    ----------
    >>> from linescanning import glm
    >>> posth = glm.Posthoc()
    >>> posth.run_posthoc(
    >>>     data=df,
    >>>     dv="dependent_variable",
    >>>     between="grouping_variable",
    >>>     )
    >>> posth.plot_bars(axs=axs)
    """

    def __init__(self, **kwargs):
        # initialize plotting setting
        super().__init__(**kwargs)
        
    def sort_posthoc(self, df):
        """sort the output of posthoc tests based on distance so that the longest significance bar spans the largest distance"""

        distances = []
        for contr in range(df.shape[0]): 
            A = df["A"].iloc[contr]
            B = df["B"].iloc[contr]

            x1 = self.conditions.index(A)
            x2 = self.conditions.index(B)

            distances.append(abs(x2-x1))
        
        df["distances"] = distances
        return df.sort_values("distances", ascending=False)

    def run_posthoc(
        self, 
        test: str="tukey",
        ano: dict=None,
        *args, 
        **kwargs):
        
        """run_posthoc

        Run the posthoc test. By default, we'll run a `tukey`-test. If the argument is something else, the :class:`pingouin.pairwise_tests()` is invoked.

        Parameters
        ----------
        test: str, optional
            Type of test to execute, by default "tukey"

        Raises
        ----------
        ImportError
            If pingouin cannot by imported
        """
        self.test = test

        if "within" in list(kwargs.keys()):
            raise ValueError(f"Drawing significance bars gets too complicated with 'within' variable")
            
        # internalize data
        if "data" in list(kwargs.keys()):
            self.data = kwargs["data"]

        try:
            import pingouin as pg
        except:
            raise ImportError(f"Could not import 'pingouin'")

        # FDR-corrected post hocs with Cohen's D effect size
        if self.test == "tukey":
            self.posthoc = pg.pairwise_tukey(
                *args,
                **kwargs
            )
            self.p_tag = "p-tukey"
        else:
            self.posthoc = pg.pairwise_tests(
                *args,
                **kwargs)
            
            if "p-corr" in list(self.posthoc.columns):
                self.p_tag = "p-corr"
            else:
                self.p_tag = "p-unc"
    
        if self.test == "inherit":
            if len(ano)>0:
                if kwargs['between'] in list(ano.keys()):
                    self.posthoc[self.p_tag] = ano[kwargs['between']]

        # internalize all kwargs
        self.__dict__.update(kwargs)
        self.conditions = utils.get_unique_ids(self.data, id=self.between, sort=False)

    def plot_bars(
        self, 
        axs: mpl.axes._axes.Axes=None,
        alpha: float=0.05,
        y_pos: float=0.95,
        line_separate_factor: float=-0.065,
        ast_frac: float=0.2,
        ns_annot: bool=False,
        ns_frac: bool=5,
        leg_size: float=0.02,
        color: str="black",
        *args,
        **kwargs):

        """plot_bars

        Function that plots the significance bars given a matplotlib axis. Based on the sorted posthoc output, it'll draw the significance bars from top to bottom (longest significance bar up top).

        Parameters
        ----------
        axs: mpl.axes._axes.Axes, optional
            Axis to plot the lines on, by default None
        alpha: float, optional
            Alpha value to consider contrasts significant, by default 0.05
        y_pos: float, optional
            Starting position of top significance line in axis proportions (1 = top of plot), by default 0.95. While looping through significant contrasts, this factor is reduced with `line_separate_factor`
        line_separate_factor: float, optional
            Factor to reduce the y-position of subsequent significance bars with, by default -0.065
        ast_frac: float, optional
            Distance between significance line and annotation (e.g., asterix denoting significance or 'ns' if `ns_annot==True`), by default 0.2
        ns_annot: bool, optional
            Also annotate non-significant contrasts with 'ns', by default False
        ns_frac: bool, optional
            Additional factor to scale the distance between the significance lines and text as `ast_frac` will yield different results for text and asterixes, by default 5
        leg_size: float, optional
            Size of the overhang from the significance bars. Default = 0.02 and is defined as a fraction of the total limit of the y-axis
        color: str, optional
            Define color. Default is black
        """

        # internalize args
        self.axs = axs
        self.alpha = alpha
        self.y_pos = y_pos
        self.line_separate_factor = line_separate_factor
        self.ast_frac = ast_frac
        self.ns_annot = ns_annot
        self.ns_frac = ns_frac
        self.leg_size = leg_size
        self.color = color

        # run posthoc if not present
        if not hasattr(self, "posthoc"):
            self.run_posthoc(*args,**kwargs)
            
            # internalize all kwargs
            self.__dict__.update(kwargs)

        self.minmax = list(self.axs.get_ylim())

        # sort posthoc so that bars furthest away are on top (if significant)
        self.posthoc_sorted = self.sort_posthoc(self.posthoc)

        if self.p_tag in list(self.posthoc_sorted.columns):
            p_meth = self.p_tag
        else:
            p_meth = "p-unc"

        for contr in range(self.posthoc_sorted.shape[0]):
            txt = None
            p_val = self.posthoc_sorted[p_meth].iloc[contr] 
            if p_val<self.alpha:
                
                if 0.01 < p_val < 0.05:
                    txt = "*"
                elif 0.001 < p_val < 0.01:
                    txt = "**"
                elif p_val < 0.001:
                    txt = "***"
                    
                style = None
                f_size = self.font_size
                dist = self.ast_frac

            else:
                if self.ns_annot:
                    txt = "ns"
                    style = "italic"
                    f_size = self.label_size
                    dist = self.ast_frac*self.ns_frac                    

            # read indices from output dataframe and conditions
            if isinstance(txt, str):
                A = self.posthoc_sorted["A"].iloc[contr]
                B = self.posthoc_sorted["B"].iloc[contr]
                
                x1 = self.conditions.index(A)
                x2 = self.conditions.index(B)

                diff = self.minmax[1]-self.minmax[0]
                y,h,col =  (diff*self.y_pos)+self.minmax[0], diff*self.leg_size, self.color
                self.axs.plot(
                    [x1,x1,x2,x2], 
                    [y,y+h,y+h,y], 
                    lw=self.tick_width, 
                    c=col
                )

                x_txt = (x1+x2)*.5
                y_txt = y+(y*dist)
                # print(y,h)
                self.axs.text(
                    x_txt, 
                    y_txt, 
                    txt, 
                    ha='center', 
                    va='bottom', 
                    color=col,
                    fontname=self.fontname,
                    fontsize=f_size,
                    style=style
                )

                # make subsequent bar lower than first
                self.y_pos += self.line_separate_factor

                # reset txt
                txt = None

class ANOVA(Posthoc):

    """ANOVA

    Run an ANOVA with pingouin and subsequently run posthoc test. Allows you to immediately visualize significant results given a matplotlib axis. In contrast to :class:`linescanning.glm.Posthoc()`, arguments for the ANOVA test are immediately passed on during the initialization stage.

    Parameters
    ----------
    alpha: float, optional
        Alpha value to consider contrasts significant, by default 0.05
    axs: mpl.axes._axes.Axes, optional
        Axis to plot the lines on, by default None
    posthoc_kw: dict, optional
        Dictionairy containing arguments that are passed to :func:`linescanning.glm.Posthoc.plot_bars()`, by default {}. See docs for arguments

    Example
    ----------
    >>> from linescanning import glm
    >>> anv = glm.ANOVA(
    >>>     data=df,
    >>>     dv="dependent_variable",
    >>>     between="grouping_variable",
    >>>     posthoc_kw={
    >>>         "ns_annot": True    
    >>>     },
    >>>     axs=axs,
    >>> )
    """

    def __init__(
        self, 
        alpha: float=0.05, 
        axs: mpl.axes._axes.Axes=None,
        posthoc_kw: dict={},
        plot_kw={},
        bar_kw={},
        *args, 
        **kwargs):
        
        self.posthoc_kw = posthoc_kw
        self.bar_kw = bar_kw
        self.plot_kw = plot_kw
        self.alpha = alpha
        self.run_anova(
            alpha=self.alpha,
            posthoc_kw=self.posthoc_kw,
            bar_kw=self.bar_kw,
            axs=axs,
            plot_kw=self.plot_kw,
            *args,
            **kwargs
        )
        
    def _get_results(
        self, 
        df: pd.DataFrame, 
        alpha: float=0.05):

        effects = df["Source"].to_list()
        p_vals = {}
        for ef in effects:
            p_ = df.loc[(df["Source"] == ef)]["p-unc"].values[0]

            if p_ < alpha:
                p_vals[ef] = p_

        return p_vals
    
    def run_anova(
        self, 
        alpha: float=0.05,
        axs: mpl.axes._axes.Axes=None,
        posthoc_kw={},
        plot_kw={},
        bar_kw={},
        *args, 
        **kwargs):
        
        self.alpha = alpha
        try:
            import pingouin as pg
        except:
            raise ImportError(f"Could not import 'pingouin'")
        
        # do stats
        self.ano = pg.anova(
            *args,
            **kwargs
        )

        # check if there's signifcant results
        self.results = self._get_results(self.ano, alpha=self.alpha)

        # found sig results; do posthoc
        self.ph_obj = {}
        super().__init__(**plot_kw)
        self.run_posthoc(ano=self.results, **kwargs, **posthoc_kw)

        if isinstance(axs, mpl.axes._axes.Axes):
            self.plot_bars(
                axs=axs,
                **bar_kw
            )
                
class ANCOVA(Posthoc):

    """ANCOVA

    Run an ANCOVA with pingouin and subsequently run posthoc test. Allows you to immediately visualize significant results given a matplotlib axis. In contrast to :class:`linescanning.glm.Posthoc()`, arguments for the ANCOVA test are immediately passed on during the initialization stage.

    Parameters
    ----------
    alpha: float, optional
        Alpha value to consider contrasts significant, by default 0.05
    axs: mpl.axes._axes.Axes, optional
        Axis to plot the lines on, by default None
    posthoc_kw: dict, optional
        Dictionairy containing arguments that are passed to :func:`linescanning.glm.Posthoc.plot_bars()`, by default {}. See docs for arguments

    Example
    ----------
    >>> from linescanning import glm
    >>> ancv = glm.ANCOVA(
    >>>     data=df,
    >>>     dv="dependent_variable",
    >>>     between="grouping_variable",
    >>>     covar="covariate",
    >>>     posthoc_kw={
    >>>         "ns_annot": True    
    >>>     },
    >>>     axs=axs,
    >>> )
    """

    def __init__(
        self, 
        alpha: float=0.05, 
        axs: mpl.axes._axes.Axes=None,
        posthoc_kw: dict={},
        bar_kw: dict={},
        plot_kw: dict={},
        *args, 
        **kwargs):
        
        self.posthoc_kw = posthoc_kw
        self.plot_kw = plot_kw
        self.bar_kw = bar_kw
        self.alpha = alpha
        self.run_ancova(
            alpha=self.alpha,
            axs=axs,
            posthoc_kw=self.posthoc_kw,
            bar_kw=self.bar_kw,
            plot_kw=self.plot_kw,
            *args,
            **kwargs)
        
    def _get_results(self, df, alpha=0.05):
        effects = df["Source"].to_list()
        p_vals = {}
        for ef in effects:
            p_ = df.loc[(df["Source"] == ef)]["p-unc"].values[0]

            if p_ < alpha:
                p_vals[ef] = p_

        return p_vals
    
    def run_ancova(
        self, 
        alpha: float=0.05,
        axs: mpl.axes._axes.Axes=None,
        posthoc_kw={},
        plot_kw={},
        bar_kw={},
        *args, 
        **kwargs):

        self.alpha = alpha
        try:
            import pingouin as pg
        except:
            raise ImportError(f"Could not import 'pingouin'")
        
        # do stats
        self.ano = pg.ancova(
            *args,
            **kwargs
        )

        # check if there's signifcant results
        self.results = self._get_results(self.ano, alpha=self.alpha)

        # found sig results; do posthoc
        filter_kwargs = [
            "covar",
        ]

        for i in filter_kwargs:
            if i in list(kwargs.keys()):
                kwargs.pop(i)
                
        super().__init__(**plot_kw)
        self.run_posthoc(ano=self.results, **kwargs, **posthoc_kw)

        if isinstance(axs, mpl.axes._axes.Axes):
            self.plot_bars(
                axs=axs,
                **bar_kw
            )
