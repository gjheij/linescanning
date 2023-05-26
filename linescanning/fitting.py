from . import (
    glm, 
    plotting,
    utils)
import lmfit
import matplotlib.pyplot as plt
import nideconv as nd
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Union

class CurveFitter():
    """CurveFitter

    Simple class to perform a quick curve fitting procedure on `y_data`. You can either specify your own function with `func`, or select a polynomial of order `order` (currently up until 3rd-order is included). Internally uses `lmfit.Model` to perform the fitting, allowing for access to confidence intervals.

    Parameters
    ----------
    y_data: np.ndarray
        Data-points to perform fitting on
    x: np.ndarray, optional
        Array describing the x-axis, by default None. If `None`, we'll take `np.arange` of `y_data.shape[0]`. 
    func: <function> object, optional
        Use custom function describing the behavior of the fit, by default None. If `none`, we'll assume either a linear or polynomial fit (up to 3rd order)
    order: str, int, optional
        Order of polynomial fit, by default "3rd". Can either be '1st'|1, '2nd'|2, or '3rd'|3
    verbose: bool, optional
        Print summary of fit, by default True
    interpolate: str, optional
        Method of interpolation for an upsampled version of the predicted data (default = 1000 samples)

    Raises
    ----------
    NotImplementedError
        If `func=None` and no valid polynomial order (see above) was specified

    Example
    ----------
    >>> # imports
    >>> from linescanning import fitting
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> # define data points
    >>> data = np.array([5.436, 5.467, 5.293, 0.99 , 2.603, 1.902, 2.317])
    >>> # create instantiation of CurveFitter
    >>> cf = fitting.CurveFitter(data, order=3, verbose=False)
    >>> # initiate figure with axis to be fed into LazyPlot
    >>> fig, axs = plt.subplots(figsize=(8,8))
    >>> # plot original data points
    >>> axs.plot(cf.x, data, 'o', color="#DE3163", alpha=0.6)
    >>> # plot upsampled fit with 95% confidence intervals as shaded error
    >>> plotting.LazyPlot(
    >>>     cf.y_pred_upsampled,
    >>>     xx=cf.x_pred_upsampled,
    >>>     error=cf.ci_upsampled,
    >>>     axs=axs,
    >>>     color="#cccccc",
    >>>     x_label="x-axis",
    >>>     y_label="y-axis",
    >>>     title="Curve-fitting with polynomial (3rd-order)")
    >>> plt.show()
    """

    def __init__(
        self, 
        y_data, 
        x=None, 
        func=None, 
        order=1, 
        verbose=True, 
        interpolate='linear',
        fix_intercept=False,
        sigma=1):

        self.y_data         = y_data
        self.func           = func
        self.order          = order
        self.x              = x
        self.verbose        = verbose
        self.interpolate    = interpolate
        self.fix_intercept  = fix_intercept
        self.sigma          = sigma
        
        if self.func == None:
            self.guess = True
            if isinstance(self.order, int):
                if self.order == 1:
                    self.pmodel = lmfit.models.LinearModel()
                elif self.order == 2:
                    self.pmodel = lmfit.models.QuadraticModel()
                else:
                    self.pmodel = lmfit.models.PolynomialModel(order=self.order)
            elif isinstance(self.order, str):
                if self.order == 'gauss' or self.order == 'gaussian':
                    self.pmodel = lmfit.models.GaussianModel()
                elif self.order == 'exp' or self.order == 'exponential':
                    self.pmodel = lmfit.models.ExponentialModel()
                else:
                    raise NotImplementedError(f"Model {self.order} is not implemented because I'm lazy..")
        else:
            self.guess = False
            self.pmodel = lmfit.Model(self.func)

        if not isinstance(self.x, list) and not isinstance(self.x, np.ndarray):
            self.x = np.arange(self.y_data.shape[0])

        # self.params = self.pmodel.make_params(a=1, b=1, c=1, d=1)
        if self.guess:
            self.params = self.pmodel.guess(self.y_data, self.x)
        else:
            self.params = self.pmodel.make_params(a=1, b=1, c=1, d=1)

        if self.fix_intercept:
            self.params['intercept'].value = 0
            self.params['intercept'].vary = False
        
        self.result = self.pmodel.fit(self.y_data, self.params, x=self.x)

        if self.verbose:
            print(self.result.fit_report())

        # create predictions & confidence intervals that are compatible with LazyPlot
        self.y_pred             = self.result.best_fit
        self.x_pred_upsampled   = np.linspace(self.x[0], self.x[-1], 1000)
        self.y_pred_upsampled   = self.result.eval(x=self.x_pred_upsampled)
        self.ci                 = self.result.eval_uncertainty(sigma=self.sigma)
        self.ci_upsampled       = glm.resample_stim_vector(self.ci, len(self.x_pred_upsampled), interpolate=self.interpolate)

    def first_order(x, a, b):
        return a * x + b
    
    def second_order(x, a, b, c):
        return a * x + b * x**2 + c
    
    def third_order(x, a, b, c, d):
	    return (a * x) + (b * x**2) + (c * x**3) + d
        
class NideconvFitter():
    """NideconvFitter

    Wrapper class around :class:`nideconv.GroupResponseFitter` to promote reprocudibility, avoid annoyances with pandas indexing, and flexibility when performing multiple deconvolutions in an analysis. Works fluently with :class:`linescanning.dataset.Dataset` and :func:`linescanning.utils.select_from_df`. Because our data format generally involved ~720 voxels, we can specify the range which represents the grey matter ribbon around our target vertex, e.g., `[355,364]`, and select the subset of the main functional dataframe to use as input for this class (see also example).

    Main inputs are the dataframe with fMRI-data, the onset timings, followed by specific settings for the deconvolution. Rigde-regression is not yet available as method, because 2D-dataframes aren't supported yet. This is a work-in-progress.

    Parameters
    ----------
    func: pd.DataFrame
        Dataframe as per the output of :func:`linescanning.dataset.Datasets.fetch_fmri()`, containing the fMRI data indexed on subject, run, and t.
    onsets: pd.DataFrame
        Dataframe as per the output of :func:`linescanning.dataset.Datasets.fetch_onsets()`, containing the onset timings data indexed on subject, run, and event_type.
    TR: float, optional
        Repetition time, by default 0.105. Use to calculate the sampling frequency (1/TR)
    confounds: pd.DataFrame, optional
        Confound dataframe with the same format as `func`, by default None
    basis_sets: str, optional
        Type of basis sets to use, by default "fourier". Should be 'fourier' or 'fir'.
    fit_type: str, optional
        Type of minimization strategy to employ, by default "ols". Should be 'ols' or 'ridge' (though the latter isn't implemented properly yet)
    n_regressors: int, optional
        Number of regressors to use, by default 9
    add_intercept: bool, optional
        Fit the intercept, by default False
    verbose: bool, optional
        _description_, by default False
    lump_events: bool, optional
        If ple are  in the onset dataframe, we can lump the events together and consider all onset times as 1 event, by default False
    interval: list, optional
        Interval to fit the regressors over, by default [0,12]

    Example
    ----------
    >>> from linescanning import utils, dataset, fitting
    >>> func_file
    >>> ['sub-003_ses-3_task-SR_run-3_bold.mat',
    >>> 'sub-003_ses-3_task-SR_run-4_bold.mat',
    >>> 'sub-003_ses-3_task-SR_run-6_bold.mat']
    >>> ribbon = [356,363]
    >>> window = 19
    >>> order = 3
    >>> 
    >>> ## window 5 TR poly 2
    >>> data_obj = dataset.Dataset(
    >>>     func_file,
    >>>     deleted_first_timepoints=50,
    >>>     deleted_last_timepoints=50,
    >>>     tsv_file=exp_file,
    >>>     use_bids=True)
    >>> 
    >>> df_func     = data_obj.fetch_fmri()
    >>> df_onsets   = data_obj.fetch_onsets()
    >>> 
    >>> # pick out the voxels representing the GM-ribbon
    >>> df_ribbon = utils.select_from_df(df_func, expression='ribbon', indices=ribbon)
    >>> nd_fit = fitting.NideconvFitter(
    >>>     df_ribbon,
    >>>     df_onsets,
    >>>     confounds=None,
    >>>     basis_sets='fourier',
    >>>     n_regressors=4,
    >>>     lump_events=False,
    >>>     TR=0.105,
    >>>     interval=[0,20],
    >>>     add_intercept=True,
    >>>     verbose=True)

    Notes
    ---------
    Several plotting options are available:

    * `plot_average_per_event`: for each event, average over the voxels present in the dataframe
    * `plot_average_per_voxel`: for each voxel, plot the response to each event
    * `plot_hrf_across_depth`: for each voxel, fetch the peak HRF response and fit a 3rd-order polynomial to the points (utilizes :class:`linescanning.utils.CurveFitter`)

    See also https://linescanning.readthedocs.io/en/latest/examples/nideconv.html for more details.
    """

    def __init__(
        self, 
        func, 
        onsets, 
        TR=0.105, 
        confounds=None, 
        basis_sets="fourier", 
        fit_type="ols", 
        n_regressors=9, 
        add_intercept=False, 
        concatenate_runs=False, 
        verbose=False, 
        lump_events=False, 
        interval=[0,12], 
        osf=20,
        fit=True,
        **kwargs):

        self.func               = func
        self.onsets             = onsets
        self.confounds          = confounds
        self.basis_sets         = basis_sets 
        self.fit_type           = fit_type
        self.n_regressors       = n_regressors
        self.add_intercept      = add_intercept
        self.verbose            = verbose
        self.lump_events        = lump_events
        self.TR                 = TR
        self.fs                 = 1/self.TR
        self.interval           = interval
        self.concatenate_runs   = concatenate_runs
        self.osf                = osf
        self.do_fit             = fit

        if self.lump_events:
            self.lumped_onsets = self.onsets.copy().reset_index()
            self.lumped_onsets['event_type'] = 'stim'
            self.lumped_onsets = self.lumped_onsets.set_index(['subject', 'run', 'event_type'])
            self.used_onsets = self.lumped_onsets.copy()
        else:
            self.used_onsets = self.onsets.copy()        
        
        # update kwargs
        self.__dict__.update(kwargs)

        # get the model
        self.define_model()

        # specify the events
        self.define_events()

        # # fit
        if self.do_fit:
            self.fit()

        # some plotting defaults
        self.plotting_defaults = plotting.Defaults()
        if not hasattr(self, "font_size"):
            self.font_size = self.plotting_defaults.font_size

        if not hasattr(self, "label_size"):
            self.label_size = self.plotting_defaults.label_size

        if not hasattr(self, "tick_width"):
            self.tick_width = self.plotting_defaults.tick_width

        if not hasattr(self, "tick_length"):
            self.tick_length = self.plotting_defaults.tick_length

        if not hasattr(self, "axis_width"):
            self.axis_width = self.plotting_defaults.axis_width

    def timecourses_condition(self):

        # get the condition-wise timecourses
        if self.fit_type == "ols":
            # averaged runs
            self.tc_condition = self.model.get_conditionwise_timecourses()

            # full timecourses of subjects
            self.tc_subjects = self.model.get_timecourses()
            self.obj_grouped = self.tc_subjects.groupby(level=["subject", "event type", "covariate", "time"])
            
            # get the standard error of mean & standard deviation
            self.tc_sem = self.obj_grouped.sem()
            self.tc_std = self.obj_grouped.std()
            self.tc_mean = self.obj_grouped.mean()

            # rename 'event type' to 'event_type'
            tmp = self.tc_sem.reset_index().rename(columns={"event type": "event_type"})
            self.tc_sem = tmp.set_index(["subject", "event_type", "covariate",  "time"])
            
            tmp = self.tc_std.reset_index().rename(columns={"event type": "event_type"})
            self.tc_std = tmp.set_index(["subject", "event_type", "covariate", "time"])

            tmp = self.tc_mean.reset_index().rename(columns={"event type": "event_type"})
            self.tc_mean = tmp.set_index(["subject", "event_type", "covariate", "time"])            

            self.sem_condition = self.tc_sem.groupby(level=["event_type", "covariate", "time"]).mean()

            self.std_condition = self.tc_std.groupby(level=["event_type", "covariate", "time"]).mean()

            # get r2
            self.rsq_ = self.model.get_rsq()

            # also get the predictions
            self.fitters = self.model._get_response_fitters()

            # loop through runs
            self.predictions = []
            for run in range(self.fitters.shape[0]):
                preds = self.fitters.iloc[run].predict_from_design_matrix()
                
                # overwrite colums
                preds.columns = self.tc_condition.columns

                # append
                self.predictions.append(preds)

            # concatenate and index
            self.predictions = pd.concat(self.predictions)
            self.predictions.index = self.func.index

        elif self.fit_type == "ridge":
            # here we need to stitch stuff back together
            if not hasattr(self, 'ridge_models'):
                raise ValueError("Ridge regression not yet performed")

            tc  = []
            rsq = []
            for vox in list(self.ridge_models.keys()):
                tc.append(self.ridge_models[vox].get_timecourses())
                rsq.append(self.ridge_models[vox].get_rsq())
            self.tc_condition = pd.concat(tc, axis=1)
            self.rsq_ = pd.concat(rsq, axis=1)
            
        # rename 'event type' to 'event_type' so it's compatible with utils.select_from_df
        tmp = self.tc_condition.reset_index().rename(columns={"event type": "event_type"})
        self.tc_condition = tmp.set_index(['event_type', 'covariate', 'time'])

        # set time axis
        self.time = self.tc_condition.groupby(['time']).mean().reset_index()['time'].values

    def define_model(self, **kwargs):

        self.model = nd.GroupResponseFitter(
            self.func,
            self.used_onsets,
            input_sample_rate=self.fs,
            confounds=self.confounds, 
            add_intercept=self.add_intercept,
            concatenate_runs=self.concatenate_runs,
            oversample_design_matrix=self.osf,
            **kwargs)
    
    def add_event(self, *args, **kwargs):
        self.model.add_event(*args, **kwargs)

    def define_events(self):
        
        utils.verbose(f"Selected '{self.basis_sets}'-basis sets", self.verbose)

        # define events
        self.cond = self.used_onsets.reset_index().event_type.unique()
        self.cond = np.array(sorted([event for event in self.cond if event != 'nan']))

        # add events to model
        for event in self.cond:
            utils.verbose(f"Adding event '{event}' to model", self.verbose)
            
            self.model.add_event(
                str(event), 
                basis_set=self.basis_sets,
                n_regressors=self.n_regressors, 
                interval=self.interval)

    def make_onsets_for_ridge(self):
        return self.used_onsets.reset_index().drop(['subject', 'run'], axis=1).set_index('event_type').loc['stim'].onset

    def fit(self):

        utils.verbose(f"Fitting with '{self.fit_type}' minimization", self.verbose)

        if self.fit_type.lower() == "ridge":
            # raise NotImplementedError("Ridge regression doesn't work with 2D-data yet, use 'ols' instead")
            
            # format the onsets properly
            vox_onsets = self.make_onsets_for_ridge()
            
            self.ridge_models = {}
            for ix, signal in enumerate(self.func.columns):
                
                # select single voxel timecourse from main DataFrame
                vox_signal = utils.select_from_df(self.func, expression='ribbon', indices=[ix,ix+1])
                
                # specify voxel-specific model
                vox_model = nd.ResponseFitter(input_signal=vox_signal, sample_rate=self.fs)

                [vox_model.add_event(
                    str(i),
                    onsets=vox_onsets,
                    basis_set=self.basis_sets,
                    n_regressors=self.n_regressors,
                    interval=self.interval) for i in self.cond]

                vox_model.fit(type='ridge')
                self.ridge_models[ix] = vox_model

        elif self.fit_type.lower() == "ols":

            # fitting
            self.model.fit(type=self.fit_type)

        else:
            raise ValueError(f"Unrecognized minimizer '{self.fit_type}'; must be 'ols' or 'ridge'")
        
        utils.verbose("Done", self.verbose)

    def plot_average_per_event(
        self, 
        add_offset: bool=True, 
        axs=None, 
        title: str="Average HRF across events", 
        save_as: str=None, 
        error_type: str="sem", 
        ttp: bool=False, 
        ttp_lines: bool=False, 
        ttp_labels: list=None, 
        events: list=None, 
        fwhm: bool=False, 
        fwhm_lines: bool=False, 
        fwhm_labels: list=None, 
        inset_ttp: list=[0.75, 0.65, 0.3],
        inset_fwhm: list=[0.75, 0.65, 0.3],
        reduction_factor: float=1.3,
        **kwargs):

        """plot_average_per_event

        Plot the average across runs and voxels for each event in your data. Allows the option to have time-to-peak or full-width half max (FWHM) plots as insets. This makes the most sense if you have multiple events, otherwise you have 1 bar..

        Parameters
        ----------
        add_offset: bool, optional
            Shift the HRFs to have the baseline at zero, by default True. Theoretically, this should already happen if your baseline is estimated properly, but for visualization and quantification purposes this is alright
        axs: <AxesSubplot:>, optional
            Matplotlib axis to store the figure on, by default None
        title: str, optional
            Plot title, by default None, by default "Average HRF across events"
        save_as: str, optional
            Save the plot as a file, by default None
        error_type: str, optional
            Which error type to use across runs/voxels, by default "sem"
        ttp: bool, optional
            Plot the time-to-peak on the inset axis, by default False
        ttp_lines: bool, optional
            Plot lines on the original axis with HRFs to indicate the maximum amplitude, by default False
        ttp_labels: list, optional
            Which labels to use for the inset axis; this can be different than your event names (e.g., if you want to round numbers), by default None
        events: list, optional
            List that decides the order of the events to plot, by default None. By default, it takes the event names, but sometimes you want to flip around the order.
        fwhm: bool, optional
            Plot the full-width half-max (FWHM) on the inset axis, by default False
        fwhm_lines: bool, optional
            Plot lines on the original axis with HRFs to indicate the maximum amplitude, by default False        
        fwhm_labels: list, optional
            Which labels to use for the inset axis; this can be different than your event names (e.g., if you want to round numbers), by default None
        inset_ttp: list, optional
            Where to put your TTP-axis, by default [0.75, 0.65, 0.3]. Height will be scaled by the number of events
        inset_fwhm: list, optional
            Where to put your FWHM-axis, by default [0.75, 0.65, 0.3, 0.3]. Width will be scaled by the number of events
        reduction_factor: float, optional
            Reduction factor of the font size in the inset axis, by default 1.3

        Example
        ----------
        >>> # do the fitting
        >>> nd_fit = fitting.NideconvFitter(
        >>>     df_ribbon, # dataframe with functional data
        >>>     df_onsets,  # dataframe with onsets
        >>>     basis_sets='canonical_hrf_with_time_derivative',
        >>>     TR=0.105,
        >>>     interval=[-3,17],
        >>>     add_intercept=True,
        >>>     verbose=True)

        >>> # plot TTP with regular events + box that highlights stimulus onset
        >>> fig,axs = plt.subplots(figsize=(8,8))
        >>> nd_fit.plot_average_per_event(
        >>>     xkcd=plot_xkcd, 
        >>>     x_label="time (s)",
        >>>     y_label="magnitude (%)",
        >>>     add_hline='default',
        >>>     ttp=True,
        >>>     lim=[0,6],
        >>>     ticks=[0,3,6],
        >>>     ttp_lines=True,
        >>>     y_label2="size (°)",
        >>>     x_label2="time-to-peak (s)", 
        >>>     title="regular events",
        >>>     ttp_labels=[f"{round(float(ii),2)}°" for ii in nd_fit.cond],
        >>>     add_labels=True,   
        >>>     fancy=True,                     
        >>>     cmap='inferno')
        >>> # plot simulus onset
        >>> axs.axvspan(0,1, ymax=0.1, color="#cccccc")

        >>> # plot FWHM and flip the events
        >>> nd_fit.plot_average_per_event(
        >>>     x_label="time (s)",
        >>>     y_label="magnitude (%)",
        >>>     add_hline='default',
        >>>     fwhm=True,
        >>>     fwhm_lines=True,
        >>>     lim=[0,5],
        >>>     ticks=[i for i in range(6)],
        >>>     fwhm_labels=[f"{round(float(ii),2)}°" for ii in nd_fit.cond[::-1]],
        >>>     events=nd_fit.cond[::-1],
        >>>     add_labels=True,
        >>>     x_label2="size (°)",
        >>>     y_label2="FWHM (s)",  
        >>>     fancy=True,
        >>>     cmap='inferno')
        """
        
        self.__dict__.update(kwargs)

        if axs == None:
            if not hasattr(self, "figsize"):
                self.figsize = (8,8)
            _,axs = plt.subplots(figsize=self.figsize)

        if not hasattr(self, "tc_condition"):
            self.timecourses_condition()

        # average across runs
        self.avg_across_runs = self.tc_condition.groupby(["event_type", "time"]).mean()

        if not isinstance(events, (list,np.ndarray)):
            events = self.cond
            self.event_indices = None
        else:
            utils.verbose(f"Flipping events to {events}", self.verbose)
            self.avg_across_runs = pd.concat(
                [utils.select_from_df(
                self.avg_across_runs, 
                expression=f"event_type = {ii}") 
            for ii in events])

            # get list of switched indices
            self.event_indices = [list(events).index(ii) for ii in self.cond]
        
        # average across voxels
        self.avg_across_runs_voxels = self.avg_across_runs.mean(axis=1)

        # parse into list so it's compatible with LazyPlot (requires an array of lists)
        self.event_avg = self.avg_across_runs_voxels.groupby("event_type").apply(np.hstack).to_list()
        self.event_sem = self.avg_across_runs.sem(axis=1).groupby("event_type").apply(np.hstack).to_list()
        self.event_std = self.avg_across_runs.std(axis=1).groupby("event_type").apply(np.hstack).to_list()

        # reorder base on indices again
        if isinstance(self.event_indices, list):
            for tt,gg in zip(["avg","sem","std"],[self.event_avg,self.event_sem,self.event_std]):
                reordered = [gg[i] for i in self.event_indices]
                setattr(self, f"event_{tt}", reordered)

        # shift all HRFs to zero
        if add_offset:
            for ev in range(len(self.event_avg)):
                if self.event_avg[ev][0] > 0:
                    self.event_avg[ev] -= self.event_avg[ev][0]
                else:
                    self.event_avg[ev] += abs(self.event_avg[ev][0])

        # decide error type
        if error_type == "sem":
            self.use_error = self.event_sem.copy()
        elif error_type == "std":
            self.use_error = self.event_std.copy()
        else:
            raise ValueError(f"Error type must be 'sem' or 'std', not {error_type}")
        
        # plot
        plotter = plotting.LazyPlot(
            self.event_avg,
            xx=self.time,
            axs=axs,
            error=self.use_error,
            title=title,
            save_as=save_as,
            **kwargs)

        if hasattr(self, "font_size"):
            self.old_font_size = plotter.font_size
            self.old_label_size = plotter.label_size
            self.font_size = plotter.font_size/reduction_factor
            self.label_size = plotter.label_size/reduction_factor

        if ttp:

            # make bar plot, use same color-coding
            if isinstance(ttp_labels, (list,np.ndarray)):
                ttp_labels = ttp_labels
            else:
                ttp_labels = events  

            # scale height by nr of events
            if len(inset_ttp) < 4:
                inset_ttp.append(len(ttp_labels)*0.05)

            left, bottom, width, height = inset_ttp
            ax2 = axs.inset_axes([left, bottom, width, height])
            self.plot_ttp(
                axs=ax2, 
                hrf_axs=axs, 
                ttp_labels=ttp_labels, 
                ttp_lines=ttp_lines,
                **dict(
                    kwargs,
                    font_size=self.font_size,
                    label_size=self.label_size,
                    sns_offset=2))

        if fwhm:

            # make bar plot, use same color-coding
            if isinstance(fwhm_labels, (list,np.ndarray)):
                fwhm_labels = fwhm_labels
            else:
                fwhm_labels = events

            # scale height by nr of events
            if len(inset_fwhm) < 4:
                inset_fwhm.insert(2, len(fwhm_labels)*0.05)
                
            left, bottom, width, height = inset_fwhm
            ax2 = axs.inset_axes([left, bottom, width, height])
            self.plot_fwhm(
                self.event_avg, 
                axs=ax2, 
                hrf_axs=axs, 
                fwhm_labels=fwhm_labels, 
                fwhm_lines=fwhm_lines,
                **dict(
                    kwargs,
                    font_size=self.font_size,
                    label_size=self.label_size,
                    sns_offset=2))

        if hasattr(self, "old_font_size"):
            self.font_size = self.old_font_size

        if hasattr(self, "old_label_size"):
            self.label_size = self.old_label_size

    def plot_ttp(
        self, 
        axs=None, 
        hrf_axs=None, 
        ttp_lines=False, 
        ttp_labels=None, 
        figsize=(8,8), 
        ttp_ori='h', 
        **kwargs):

        if not hasattr(self, "color"):
            if not hasattr(self, "cmap"):
                cmap = "viridis"
            else:
                cmap = self.cmap
            colors = sns.color_palette(cmap, len(self.cond))
        else:
            colors = self.color

        if axs == None:
            _,axs = plt.subplots(figsize=figsize)

        # get time-to-peak and index | parse out idxmax as it returns a tuple by default
        self.df_peaks = self.avg_across_runs_voxels.groupby(level=0, sort=False).agg(['idxmax','max'],axis=1).reset_index()
        self.df_peaks["idxmax"] = [self.df_peaks["idxmax"][ii][1] for ii in range(self.df_peaks.shape[0])]

        if ttp_lines:
            # heights need to be adjusted for by axis length 
            ylim = hrf_axs.get_ylim()
            tot = sum(list(np.abs(ylim)))
            start = (0-ylim[0])/tot
            for ix,ii in enumerate(self.df_peaks["idxmax"].values):
                hrf_axs.axvline(
                    ii, 
                    ymin=start, 
                    ymax=self.df_peaks["max"][ix]/tot+start, 
                    color=colors[ix], 
                    linewidth=0.5)
        
        self.ttp_plot = plotting.LazyBar(
            data=self.df_peaks,
            x="event_type",
            y="idxmax",
            labels=ttp_labels,
            sns_ori=ttp_ori,
            axs=axs,
            error=None,
            **dict(
                kwargs,
                font_size=self.font_size,
                label_size=self.label_size))          


    def plot_fwhm(
        self, 
        hrfs, 
        axs=None, 
        hrf_axs=None, 
        fwhm_lines=False, 
        fwhm_labels=None, 
        figsize=(8,8), 
        fwhm_ori='v', 
        **kwargs):

        if not hasattr(self, "color"):
            if not hasattr(self, "cmap"):
                cmap = "viridis"
            else:
                cmap = self.cmap
            colors = sns.color_palette(cmap, len(self.cond))
        else:
            colors = self.color

        if axs == None:
            fig, axs = plt.subplots(figsize=figsize)

        # get fwhm
        fwhm_objs = []
        for hrf in hrfs:
            fwhm_objs.append(FWHM(self.time, hrf))

        if fwhm_lines:
            # heights need to be adjusted for by axis length 
            xlim = hrf_axs.get_xlim()
            tot = sum(list(np.abs(xlim)))
            for ix, ii in enumerate(fwhm_objs):
                start = (ii.hmx[0]-xlim[0])/tot
                hrf_axs.axhline(
                    ii.half_max, 
                    xmin=start, 
                    xmax=start+ii.fwhm/tot, 
                    color=colors[ix], 
                    linewidth=0.5)

        self.y_fwhm = [i.fwhm for i in fwhm_objs]
        self.fwhm_labels = fwhm_labels
        self.fwhm_plot = plotting.LazyBar(
            x=self.fwhm_labels,
            y=self.y_fwhm,
            palette=colors,
            sns_ori=fwhm_ori,
            axs=axs,
            error=None,
            **dict(
                kwargs,
                font_size=self.font_size,
                label_size=self.label_size))          

    def plot_average_per_voxel(
        self, 
        add_offset: bool=True, 
        axs=None, 
        n_cols: int=4, 
        wspace: float=0, 
        figsize: tuple=None, 
        make_figure: bool=True, 
        labels: list=None, 
        save_as: str=None, 
        sharey: bool=False, 
        **kwargs):

        """plot_average_per_voxel

        Plot the average across runs for each voxel in your dataset. Generally, this plot is used to plot HRFs across depth. If you have multiple events, we'll create a grid of `n_cols` wide (from which the rows are derived), with the HRFs for each event in the subplot. The legend will be put in the first subplot. If you only have 1 event, you can say `n_cols=None` to put the average across events for all voxels in 1 plot

        Parameters
        ----------
        add_offset: bool, optional
            Shift the HRFs to have the baseline at zero, by default True. Theoretically, this should already happen if your baseline is estimated properly, but for visualization and quantification purposes this is alright
        axs: <AxesSubplot:>, optional
            Matplotlib axis to store the figure on, by default None
        n_cols: int, optional
            Decides the number of subplots on the x-axis, by default 4. If you have 1 event, specify `n_cols=None`
        wspace: float, optional
            Decide the width between subplots, by default 0
        figsize: tuple, optional
            Figure size, by default (24,5*nr_of_rows) or (8,8) if `n_cols=None`
        make_figure: bool, optional
            Actually create the plot or just fetch the data across depth, by default True
        labels: list, optional
            Which labels to use for the inset axis; this can be different than your event names (e.g., if you want to round numbers), by default None
        save_as: str, optional
            Save to file, by default None
        sharey: bool, optional
            Save all y-axes the same, by default False. Can be nice if you want to see the progression across depth

        Example
        ----------
        >>> nd_fit.plot_average_per_voxel(
        >>>     labels=[f"{round(float(ii),2)} dva" for ii in nd_fit.cond],
        >>>     wspace=0.2,
        >>>     cmap="inferno",
        >>>     line_width=2,
        >>>     font_size=font_size,
        >>>     label_size=16,
        >>>     sharey=True)
        """
        self.__dict__.update(kwargs)

        if not hasattr(self, "tc_condition"):
            self.timecourses_condition()

        cols = list(self.tc_condition.columns)
        cols_id = np.arange(0, len(cols))
        if n_cols != None:

            # initiate figure
            if len(cols) > 10:
                raise Exception(f"{len(cols)} were requested. Maximum number of plots is set to 30")

            n_rows = int(np.ceil(len(cols) / n_cols))
            if not isinstance(figsize, tuple):
                figsize = (24,5*n_rows)

            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(n_rows, n_cols, wspace=wspace)
        else:
            if not isinstance(figsize, tuple):
                figsize = (8,8)

        self.all_voxels_in_event = []
        self.all_error_in_voxels = []
        for ix, col in enumerate(cols):

            # fetch data from specific voxel for each stimulus size
            self.voxel_in_events = []
            self.error_in_voxels = []
            for idc,stim in enumerate(self.cond):
                col_data = self.tc_condition[col][stim].values
                err_data = self.sem_condition[col][stim].values
                
                if add_offset:
                    if col_data[0] > 0:
                        col_data -= col_data[0]
                    else:
                        col_data += abs(col_data[0])

                self.voxel_in_events.append(col_data[np.newaxis,:])
                self.error_in_voxels.append(err_data[np.newaxis,:])

            # this one is in case we want the voxels in 1 figure
            self.all_voxels_in_event.append(np.concatenate(self.voxel_in_events, axis=0)[np.newaxis,:])
            self.all_error_in_voxels.append(np.concatenate(self.error_in_voxels, axis=0)[np.newaxis,:])

        self.arr_voxels_in_event = np.concatenate(self.all_voxels_in_event, axis=0)
        self.arr_error_in_event = np.concatenate(self.all_error_in_voxels, axis=0)
        
        top = self.arr_voxels_in_event + self.arr_error_in_event
        bottom = self.arr_voxels_in_event - self.arr_error_in_event
        
        if make_figure:
            if n_cols == None:
                if labels:
                    labels = cols.copy()
                else:
                    labels = None

                vox_data = [self.arr_voxels_in_event[ii,0,:] for ii in range(self.arr_voxels_in_event.shape[0])]
                vox_error = [self.arr_error_in_event[ii, 0, :] for ii in range(self.arr_voxels_in_event.shape[0])]
                
                if axs != None:
                    self.pl = plotting.LazyPlot(
                        vox_data,
                        xx=self.time,
                        error=vox_error,
                        axs=axs,
                        labels=labels,
                        add_hline='default',
                        **kwargs)
                else:
                    self.pl = plotting.LazyPlot(
                        vox_data,
                        xx=self.time,
                        error=vox_error,
                        figsize=figsize,
                        labels=labels,
                        add_hline='default',
                        **kwargs)                    
            else:
                for ix, col in enumerate(cols):
                    axs = fig.add_subplot(gs[ix])
                    if ix == 0:
                        label = labels
                    else:
                        label = None

                    vox_data = [self.arr_voxels_in_event[ix,ii,:] for ii in range(len(self.cond))]
                    vox_error = [self.arr_error_in_event[ix,ii,:] for ii in range(len(self.cond))]                    
                    
                    if sharey:
                        ylim = [bottom.min(), top.max()]
                    else:
                        ylim = None

                    self.pl = plotting.LazyPlot(
                        vox_data,
                        xx=self.time,
                        error=vox_error,
                        axs=axs,
                        labels=label,
                        add_hline='default',
                        y_lim=ylim,
                        title=col,
                        **kwargs)

                    if ix in cols_id[::n_cols]:
                        axs.set_ylabel("Magnitude (%change)", fontsize=self.font_size)
                    
                    if ix in np.arange(n_rows*n_cols)[-n_cols:]:
                        axs.set_xlabel("Time (s)", fontsize=self.font_size)
        
                plt.tight_layout()

        if save_as:
            fig.savefig(
                save_as, 
                dpi=300, 
                bbox_inches='tight')


    def plot_hrf_across_depth(
        self,
        axs=None,
        figsize: tuple=(8,8),
        cmap: str='viridis',
        color: Union[str,tuple]=None,
        ci_color: Union[str,tuple]="#cccccc",
        ci_alpha: float=0.6,
        save_as: str=None,
        invert: bool=False,
        **kwargs):

        """plot_hrf_across_depth

        Plot the magnitude of the HRF across depth as points with a seaborn regplot through it. The points can be colored with `color` according to the HRF from :func:`linescanning.fitting.NideconvFitter.plot_average_across_voxels`, or they can be given 1 uniform color. The linear fit can be colored using `ci_color`, for which the default is light gray.

        Parameters
        ----------
        axs: <AxesSubplot:>, optional
            Matplotlib axis to store the figure on, by default None
        figsize: tuple, optional
            Figure size, by default (8,8)
        cmap: str, optional
            Color map for the data points, by default 'viridis'
        color: str, tuple, optional
            Don't use a color map for the data points, but a uniform color instead, by default None. `cmap` takes precedence!
        ci_color: str, tuple, optional
            Color of the linear fit with seaborn's regplot, by default "#cccccc"
        ci_alpha: float, optional
            Alpha of linear fit, by default 0.6
        save_as: str, optional
            Save as file, by default None
        invert: bool, optional
            By default, we'll assume your input data represents voxels from CSF to WM. This flag can flip that around, by default False

        Example
        ----------
        >>> # lump events together
        >>> lumped = fitting.NideconvFitter(
        >>>     df_ribbon,
        >>>     df_onsets,
        >>>     basis_sets='fourier',
        >>>     n_regressors=4,
        >>>     lump_events=True,
        >>>     TR=0.105,
        >>>     interval=[-3,17])

        >>> # plot
        >>> lumped.plot_hrf_across_depth(x_label="depth [%]")

        >>> # make a combined plot of HRFs and magnitude
        >>> fig = plt.figure(figsize=(16, 8))
        >>> gs = fig.add_gridspec(1, 2)
        >>> 
        >>> ax = fig.add_subplot(gs[0])
        >>> lumped.plot_average_per_voxel(
        >>>     n_cols=None, 
        >>>     axs=ax, 
        >>>     labels=True,
        >>>     x_label="time (s)",
        >>>     y_label="magnitude",
        >>>     set_xlim_zero=False)
        >>> ax.set_title("HRF across depth (collapsed stimulus events)", fontsize=lumped.pl.font_size)
        >>> 
        >>> ax = fig.add_subplot(gs[1])
        >>> lumped.plot_hrf_across_depth(
        >>>     axs=ax, 
        >>>     order=1,
        >>>     x_label="depth [%]")
        >>> ax.set_title("Maximum value HRF across depth", fontsize=lumped.pl.font_size)        
        """
        
        if not hasattr(self, "all_voxels_in_event"):
            self.plot_timecourses(make_figure=False)

        self.max_vals = np.array([np.amax(self.all_voxels_in_event[ii]) for ii in range(len(self.all_voxels_in_event))])

        if not axs:
            fig,axs = plt.subplots(figsize=figsize)

        if isinstance(cmap, str):
            color_list = sns.color_palette(cmap, len(self.max_vals))
        else:
            color_list = [color for _ in self.max_vals]

        self.depths = np.linspace(0,100,num=len(self.max_vals))
        if invert:
            self.max_vals = self.max_vals[::-1]

        self.pl = plotting.LazyCorr(
            self.depths, 
            self.max_vals, 
            color=ci_color, 
            axs=axs, 
            x_ticks=[0,50,100],
            points=False,
            scatter_kwargs={"cmap": cmap},
            **kwargs)

        for ix, mark in enumerate(self.max_vals):
            axs.plot(self.depths[ix], mark, 'o', color=color_list[ix], alpha=ci_alpha)

        for pos,tag in zip([(0.02,0.02),(0.85,0.02)],["pial","wm"]):
            axs.annotate(
                tag,
                pos,
                fontsize=self.pl.font_size,
                xycoords="axes fraction"
            )

        if save_as:
            fig.savefig(save_as, dpi=300, bbox_inches='tight')

    def plot_areas_per_event(
        self, 
        colors=None, 
        save_as=None, 
        add_offset=True, 
        error_type="sem", 
        axs=None, 
        events=None,
        **kwargs):

        if not hasattr(self, "tc_condition"):
            self.timecourses_condition()

        n_cols = len(list(self.cond))
        figsize=(n_cols*6,6)

        if not axs:
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(1, n_cols)

        if error_type == "std":
            err = self.std_condition.copy()
        elif error_type == "sem":
            err = self.sem_condition.copy()
        else:
            err = self.std_condition.copy()
            
        if not isinstance(events, list) and not isinstance(events, np.ndarray):
            events = self.cond
        else:
            if self.verbose:
                print(f"Flipping events to {events}")
        
        for ix, event in enumerate(events):
            
            # add axis
            if not axs:
                ax = fig.add_subplot(gs[ix])
            else:
                ax = axs

            for key in list(kwargs.keys()):
                if ix != 0:
                    if key == "y_ticks":
                        kwargs[key] = []
                    elif key == "y_label":
                        kwargs[key] = None

            event_df = utils.select_from_df(self.tc_condition, expression=f"event_type = {event}")
            error_df = utils.select_from_df(err, expression=f"event_type = {event}")            

            self.data_for_plot = []
            self.error_for_plot = []
            for ii, dd in enumerate(list(self.tc_condition.columns)):

                # get the timecourse
                col_data = np.squeeze(utils.select_from_df(event_df, expression='ribbon', indices=[ii]).values)
                col_error = np.squeeze(utils.select_from_df(error_df, expression='ribbon', indices=[ii]).values)

                # shift to zero
                if add_offset:
                    if col_data[0] > 0:
                        col_data -= col_data[0]
                    else:
                        col_data += abs(col_data[0])
                
                self.data_for_plot.append(col_data)
                self.error_for_plot.append(col_error)
            
            if not isinstance(error_type, str):
                self.error_for_plot = None

            plotting.LazyPlot(
                self.data_for_plot,
                xx=self.time,
                error=self.error_for_plot,
                axs=ax,
                **kwargs)

            if not axs:
                if save_as:
                    fig.savefig(save_as, dpi=300, bbox_inches='tight')

class FWHM():

    def __init__(self, x, hrf):
        
        self.x          = x
        self.hrf        = hrf
        self.hmx        = self.half_max_x(self.x,self.hrf)
        self.fwhm       = self.hmx[1] - self.hmx[0]
        self.half_max   = max(hrf)/2
    
    def lin_interp(self, x, y, i, half):
        return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

    def half_max_x(self, x, y):
        half = max(y)/2.0
        signs = np.sign(np.add(y, -half))
        zero_crossings = (signs[0:-2] != signs[1:-1])
        zero_crossings_i = np.where(zero_crossings)[0]
        return [self.lin_interp(x, y, zero_crossings_i[0], half),
                self.lin_interp(x, y, zero_crossings_i[1], half)]
