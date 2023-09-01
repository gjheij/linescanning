from . import (
    glm, 
    plotting,
    utils)
import lmfit
import matplotlib.pyplot as plt
import matplotlib as mpl
import nideconv as nd
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Union
from scipy.optimize import minimize
from joblib import Parallel, delayed
from sklearn import metrics

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

class ParameterFitter():

    def __init__(self, data, onsets, TR=0.105):

        self.data = data
        self.onsets = onsets
        self.TR = TR

        # prepare data
        self.prepare_data()
        self.prepare_onsets()

        # get info about dataframe
        self.run_ids = None            
        self.sub_ids = None
        if isinstance(self.data, pd.DataFrame):
            try:
                self.sub_ids = utils.get_unique_ids(self.data, id="subject")
            except:
                pass

            try:
                self.run_ids = utils.get_unique_ids(self.data, id="run")
            except:
                pass

        # get events
        self.evs = utils.get_unique_ids(self.onsets, id="event_type")
    
    def _get_timepoints(self):
        return list(np.arange(0,self.data.shape[0])*self.TR)

    def prepare_onsets(self):
        
        # store copy of original data
        self.orig_onsets = self.onsets.copy()
        
        # put in dataframe
        if isinstance(self.onsets, np.ndarray):
            self.onsets = pd.DataFrame(self.onsets, columns=["onsets"])
        else:
            try:
                self.onsets = self.onsets.reset_index()
            except:
                pass
            
        # format dataframe with subject, run, and t
        self.onset_index = [
            "subject",
            "run",
            "event_type"
        ]

        for key,val in zip(self.onset_index, [1,1,"stim"]):
            if not key in list(self.onsets.columns):
                self.onsets[key] = val

        self.onsets.set_index(self.onset_index, inplace=True)   

    def prepare_data(self):
        
        # store copy of original data
        self.orig_data = self.data.copy()
        
        # put in dataframe
        self.time = self._get_timepoints()
        if isinstance(self.data, np.ndarray):
            self.data = pd.DataFrame(self.data)
        else:
            try:
                self.data = self.data.reset_index()
            except:
                pass
            
        # format dataframe with subject, run, and t
        self.final_index = [
            "subject",
            "run",
            "t"
        ]

        for key,val in zip(self.final_index, [1,1,self.time]):
            if not key in list(self.data.columns):
                self.data[key] = val

        self.data.set_index(self.final_index, inplace=True)            

    @staticmethod
    def single_response_fitter(
        data,
        onsets,
        TR=0.105,
        **kwargs):

        cl_fit = FitHRFparams(
            data,
            onsets,
            TR=TR,
            **kwargs
        )

        cl_fit.iterative_fit()

        return cl_fit
    
    @staticmethod
    def _set_dict():
        ddict = {}
        for el in ["predictions","profiles","pars"]:
            ddict[el] = []

        return ddict
    
    @staticmethod
    def _concat_dict(ddict):
        new_dict = {}
        for key,val in ddict.items():
            if len(val)>0:
                new_dict[key] = pd.concat(val)

        return new_dict    

    def fit(self, **kwargs):

        self.ddict_sub = self._set_dict()

        # loop through subject
        for sub in self.sub_ids:
            
            print(f"Fitting '{sub}'")
            # get subject specific onsets
            self.sub_onsets = utils.select_from_df(self.onsets, expression=f"subject = {sub}")
            self.ddict_run = self._set_dict()
            for run in self.run_ids:
                
                print(f" run-'{run}'")
                # subject and run specific data
                self.func = utils.select_from_df(
                    self.data, expression=(
                        f"subject = {sub}",
                        "&",
                        f"run = {run}"
                    )
                )                 
                
                # loop through events
                self.ddict_ev = self._set_dict()
                for ev in self.evs:
                    
                    print(f"  ev-'{ev}'")

                    # run/event-specific onsets
                    self.ons = utils.select_from_df(
                        self.sub_onsets, expression=(
                            f"run = {run}",
                            "&",
                            f"event_type = {ev}"
                        )
                    )

                    # do fit
                    self.rf = self.single_response_fitter(
                        self.func.T.values,
                        self.ons,
                        TR=self.TR,
                        **kwargs
                    )

                    # get profiles and full timecourse predictions
                    tmp = self.rf.hrf_profiles.copy().reset_index()
                    pr = self.rf.predictions.copy().reset_index()
                    tmp["run"] = run
                    pr["run"] = run       

                    # finalize dataframe with proper formatting
                    df = tmp.set_index(["run","time"]).groupby(["time"]).mean().reset_index()
                    df["covariate"],df["event_type"] = "intercept",ev
                    pr["event_type"] = ev

                    # get metrics
                    try:
                        pars_ = HRFMetrics(df.set_index(["event_type","covariate","time"])).return_metrics()
                        pars_["event_type"] = ev
                    except Exception as e:
                        raise RuntimeError(f"Caught exception while dealing with '{sub}', 'event = {ev}': {e}")  

                    try:
                        df = df.reset_index(drop=True)
                    except:
                        pass

                    # append
                    self.ddict_ev["predictions"].append(pr)
                    self.ddict_ev["profiles"].append(df)
                    self.ddict_ev["pars"].append(pars_)

                # concatenate all evs
                self.ddict_ev = self._concat_dict(self.ddict_ev)

                # set run id and append to run dict
                for key,val in self.ddict_ev.items():
                    self.ddict_ev[key]['run'] = run
                    self.ddict_run[key].append(val)

            # concatenate all runs
            self.ddict_run = self._concat_dict(self.ddict_run)

            # set run id and append to run dict
            for key,val in self.ddict_run.items():
                self.ddict_run[key]['subject'] = sub
                self.ddict_sub[key].append(val)

        # concatenate all subjects
        self.ddict_sub = self._concat_dict(self.ddict_sub)

        # set final outputs
        self.tc_subjects = self.ddict_sub["profiles"].set_index(["subject","run","event_type","covariate","time"])
        self.tc_condition = self.tc_subjects.groupby(level=['event_type', 'covariate', 'time']).mean()

        self.ev_predictions = self.ddict_sub["predictions"].set_index(["subject","run","event_type","t"])
        self.ev_parameters = self.ddict_sub["pars"].set_index(["subject","run", "event_type"])

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
        Number of regressors to use, by default 9; set `n_regressors="tr"` to synchronize the number of regressors with the TR
    add_intercept: bool, optional
        Fit the intercept, by default False
    verbose: bool, optional
        _description_, by default False
    lump_events: bool, optional
        If ple are  in the onset dataframe, we can lump the events together and consider all onset times as 1 event, by default False
    interval: list, optional
        Interval to fit the regressors over, by default [0,12]
    covariates: dict, optional
        dictionary of covariates for each of the events in onsets. That is, the keys are the names of the covariates, the values are 1D numpy arrays of length identical to onsets; these are the covariate values of each of the events in onsets.

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
        covariates=None,
        conf_intercept=True,
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
        self.covariates         = covariates
        self.conf_icpt          = conf_intercept

        self.allowed_basis_sets = [
            "fir",
            "fourier",
            "dct",
            "legendre",
            "canonical_hrf",
            "canonical_hrf_with_time_derivative",
            "canonical_hrf_with_time_derivative_dispersion"
        ]

        if self.lump_events:
            self.lumped_onsets = self.onsets.copy().reset_index()
            self.lumped_onsets['event_type'] = 'stim'
            self.lumped_onsets = self.lumped_onsets.set_index(['subject', 'run', 'event_type'])
            self.used_onsets = self.lumped_onsets.copy()
        else:
            self.used_onsets = self.onsets.copy()        
        
        if self.basis_sets not in self.allowed_basis_sets:
            raise ValueError(f"Unrecognized basis set '{self.basis_sets}'. Must be one of {self.allowed_basis_sets}")
        else:
            if self.basis_sets == "canonical_hrf":
                self.n_regressors = 1
            elif self.basis_sets == "canonical_hrf_with_time_derivative":
                self.n_regressors = 2
            elif self.basis_sets == "canonical_hrf_with_time_derivative_dispersion":
                self.n_regressors = 3
            else:
                # set 1 regressor per TR
                if isinstance(self.n_regressors, str):
                    self.n_regressors = round(((self.interval[1]+abs(self.interval[0])))/self.TR)        
        
        # update kwargs
        self.__dict__.update(kwargs)

        # get the model
        if self.fit_type == "ols":
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

    def get_event_predictions_from_fitter(self, fitter):

        ev_pred = []
        for ev in self.cond:
            X_stim = fitter.X.xs(ev, axis=1, drop_level=False)
            betas = utils.select_from_df(fitter.betas, expression=f"event type = {ev}")
            pred = X_stim.dot(betas).reset_index()
            pred.rename(columns={"time": "t"}, inplace=True) 

            pred["event_type"] = ev
            ev_pred.append(pred)

        ev_pred = pd.concat(ev_pred, ignore_index=True)    
        return ev_pred
        
    def get_predictions_per_event(self):
        
        # also get the predictions
        if self.fit_type == "ols":
            self.fitters = self.model._get_response_fitters()

        # loop through runs
        self.predictions = []
        self.ev_predictions = []
        for run in range(self.fitters.shape[0]):

            # full model predictions
            run_fitter = self.fitters.iloc[run]
            preds = run_fitter.predict_from_design_matrix()
            
            # overwrite colums
            preds.columns = self.tc_condition.columns

            # append
            self.predictions.append(preds)

            # event-specific predictions
            ev_pred = self.get_event_predictions_from_fitter(run_fitter)
            ev_pred["run"] = run+1

            self.ev_predictions.append(ev_pred)

        # concatenate and index
        self.predictions = pd.concat(self.predictions)
        self.predictions.index = self.func.index
        self.ev_predictions = pd.concat(self.ev_predictions).set_index(["event_type","run","t"])

    def timecourses_condition(self):

        if not isinstance(self.covariates, str):
            self.covariates = "intercept"
            
        # get the condition-wise timecourses
        if self.fit_type == "ols":
            # averaged runs
            self.tc_condition = self.model.get_conditionwise_timecourses()

            # full timecourses of subjects
            self.tc_subjects = self.model.get_timecourses()
        else:
            self.tc_condition = self.tc_subjects.groupby(level=['event type', 'covariate', 'time']).mean()

            self.fitters = self.sub_df.copy()

        # get some more info
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
            
        # rename 'event type' to 'event_type' so it's compatible with utils.select_from_df
        tmp = self.tc_condition.reset_index().rename(columns={"event type": "event_type"})
        self.tc_condition = tmp.set_index(['event_type', 'covariate', 'time'])

        # set time axis
        self.time = self.tc_condition.groupby(['time']).mean().reset_index()['time'].values

        # get r2
        try:
            self.rsq_ = self.model.get_rsq()
        except:
            pass
            
        # get predictions
        if self.fit_type == "ols":
            self.get_predictions_per_event()
        else:
            self.ev_predictions = self.sub_ev_df.copy()
        
    def define_model(self, **kwargs):

        self.model = nd.GroupResponseFitter(
            self.func,
            self.used_onsets,
            input_sample_rate=self.fs,
            confounds=self.confounds, 
            concatenate_runs=self.concatenate_runs,
            oversample_design_matrix=self.osf,
            add_intercept=self.conf_icpt,
            **kwargs)
    
    def add_event(self, *args, **kwargs):
        self.model.add_event(*args, **kwargs)

    def define_events(self):
        
        utils.verbose(f"Selected '{self.basis_sets}'-basis sets (with {self.n_regressors} regressors)", self.verbose)

        # define events
        self.cond = self.used_onsets.reset_index().event_type.unique()
        self.run_ids = self.used_onsets.reset_index().run.unique()
        self.cond = np.array(sorted([event for event in self.cond if event != 'nan']))

        # add events to model
        if self.fit_type.lower() == "ols":
            for ix,event in enumerate(self.cond):
                utils.verbose(f"Adding event '{event}' to model", self.verbose)

                if isinstance(self.covariates, list):
                    cov = self.covariates[ix]
                else:
                    cov = self.covariates

                if isinstance(self.add_intercept, list):
                    icpt = self.add_intercept[ix]
                else:
                    icpt = self.add_intercept

                self.model.add_event(
                    str(event), 
                    basis_set=self.basis_sets,
                    n_regressors=self.n_regressors, 
                    interval=self.interval,
                    add_intercept=icpt,
                    covariates=cov)
                
        else:
            
            utils.verbose(f"Setting up models ridge-regression", self.verbose)

            # get subjects
            try:
                self.sub_ids = utils.get_unique_ids(self.func, id="subject")
            except:
                self.sub_ids = [1]

            self.do_fit = True
            self.sub_df = []
            self.sub_ev_df = []
            self.tc_subjects = []
            for sub in self.sub_ids:

                try:
                    self.run_ids = [int(i) for i in utils.get_unique_ids(self.func, id="run")]
                    self.run_df = []
                    self.run_pred_ev_df = []
                    self.run_prof_ev_df = []
                    set_indices = ["subject","run"]
                    set_ev_idc = ["event_type","run","t"]
                    set_prof_idc = ["subject","run","event type","covariate","time"]
                except:
                    self.run_ids = [None]
                    self.run_df = None
                    self.run_pred_ev_df = None
                    self.run_prof_ev_df = None
                    set_indices = ["subject"]
                    set_ev_idc = ["event_type","t"]
                    set_prof_idc = ["subject","event type","covariate","time"]

                # loop through runs, if available
                for run in self.run_ids:
                    
                    # loop trough voxels (ridge only works for 1d data)
                    self.vox_df = []
                    self.ev_predictions = []
                    self.vox_prof = []
                    for ix, col in enumerate(list(self.func.columns)):
                        
                        # select single voxel timecourse from main DataFrame
                        if isinstance(run, int):
                            self.vox_signal = pd.DataFrame(utils.select_from_df(self.func[col], expression=f"run = {run}")[col])
                        else:
                            self.vox_signal = pd.DataFrame(self.func[col])

                        # make response fitter
                        if isinstance(self.covariates, list):
                            cov = self.covariates[ix]
                        else:
                            cov = self.covariates

                        self.rf = self.make_response_fitter(self.vox_signal, run=run, cov=cov)

                        # fit immediately; makes life easier
                        if self.do_fit:
                            self.rf.fit(type="ridge")

                            # HRF profiles
                            self.vox_prof.append(self.rf.get_timecourses().reset_index())

                            # timecourse predictions
                            self.ev_pred = self.get_event_predictions_from_fitter(self.rf)
                            self.ev_predictions.append(self.ev_pred)

                        self.rf_df = pd.DataFrame({col: self.rf}, index=[0])
                        self.vox_df.append(self.rf_df)
                    
                    self.vox_df = pd.concat(self.vox_df, axis=1)

                    if len(self.ev_predictions)>0:
                        self.ev_predictions = pd.concat(self.ev_predictions, axis=1)

                    if len(self.vox_prof)>0:
                        self.vox_prof = pd.concat(self.vox_prof, axis=1)

                    if isinstance(run, int):
                        self.vox_df["run"] = run
                        self.run_df.append(self.vox_df)

                        if isinstance(self.ev_predictions, pd.DataFrame):
                            self.ev_predictions["run"] = run
                            self.run_pred_ev_df.append(self.ev_predictions)

                        if isinstance(self.vox_prof, pd.DataFrame):
                            self.vox_prof["run"] = run
                            self.run_prof_ev_df.append(self.vox_prof)                            

                    else:
                        self.vox_df["subject"] = sub
                        self.sub_df.append(self.vox_df)
                        
                        if isinstance(self.ev_predictions, pd.DataFrame):
                            self.sub_ev_df.append(self.ev_predictions)

                        if isinstance(self.vox_prof, pd.DataFrame):
                            self.vox_prof["subject"] = sub
                            self.tc_subjects.append(self.vox_prof)

                if isinstance(self.run_df, list):
                    self.run_df = pd.concat(self.run_df)    
                    self.run_df["subject"] = sub
                    self.sub_df.append(self.run_df)

                if isinstance(self.run_pred_ev_df, list):
                    self.run_pred_ev_df = pd.concat(self.run_pred_ev_df)
                    self.sub_ev_df.append(self.run_pred_ev_df)

                if isinstance(self.run_prof_ev_df, list):
                    self.run_prof_ev_df = pd.concat(self.run_prof_ev_df)
                    self.run_prof_ev_df["subject"] = sub
                    self.tc_subjects.append(self.run_prof_ev_df)                    

            self.sub_df = pd.concat(self.sub_df).set_index(set_indices)

            if len(self.sub_ev_df)>0:
                self.sub_ev_df = pd.concat(self.sub_ev_df)
                self.sub_ev_df.set_index(set_ev_idc, inplace=True)

            if len(self.tc_subjects)>0:
                self.tc_subjects = pd.concat(self.tc_subjects)
                self.tc_subjects.set_index(set_prof_idc, inplace=True)                

    def make_response_fitter(self, data, run=None, cov=None):

        # specify voxel-specific model
        model = nd.ResponseFitter(
            input_signal=data, 
            sample_rate=self.fs,
            add_intercept=self.conf_icpt,
            oversample_design_matrix=self.osf
        )

        # get onsets
        [
            model.add_event(
                str(i),
                onsets=self.make_onsets_for_response_fitter(i, run=run),
                basis_set=self.basis_sets,
                n_regressors=self.n_regressors,
                interval=self.interval,
                covariates=cov
            ) for i in self.cond
        ]

        return model
    
    def make_onsets_for_response_fitter(self, i, run=None):
        select_from_onsets = self.used_onsets.copy()
        if isinstance(run, int):
            select_from_onsets = utils.select_from_df(select_from_onsets, expression=f"run = {run}")
            drop_idcs = ["subject","run"]
        else:
            drop_idcs = ["subject"]
        
        return select_from_onsets.reset_index().drop(drop_idcs, axis=1).set_index('event_type').loc[i].onset

    def fit(self):

        # fitting
        utils.verbose(f"Fitting with '{self.fit_type}' minimization", self.verbose)
        if self.fit_type.lower() == "ols":
            self.model.fit(type=self.fit_type)
        
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
        self.tc_df = utils.select_from_df(self.tc_condition, expression=f"covariate = {self.covariates}")

        # average across runs
        self.avg_across_runs = self.tc_df.groupby(["event_type", "time"]).mean()

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
            for tt,gg in zip(
                ["avg","sem","std"],
                [self.event_avg,self.event_sem,self.event_std]):
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
            self.use_error = None
        
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
                df=self.avg_across_runs_voxels,
                axs=ax2, 
                hrf_axs=axs, 
                ttp_labels=ttp_labels, 
                ttp_lines=ttp_lines,
                font_size=self.font_size,
                label_size=self.label_size,
                sns_offset=2
            )

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
                df=self.avg_across_runs_voxels, 
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
        df=None,
        axs=None, 
        hrf_axs=None, 
        ttp_lines=False, 
        ttp_labels=None, 
        figsize=(8,8), 
        ttp_ori='h', 
        split="event_type",
        **kwargs):

        if not isinstance(axs, mpl.axes._axes.Axes):
            _,axs = plt.subplots(figsize=figsize)

        # unstack series | assume split is on event_type
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df).unstack(level=split)
            x_lbl = split
            x_data = self.cond
        else:
            x_lbl = "vox"
            x_data = list(df.columns)

        self.df_pars = HRFMetrics(df).return_metrics()
        self.df_pars[x_lbl] = x_data

        # decide on color based on split
        if not hasattr(self, "color"):
            if not hasattr(self, "cmap"):
                cmap = "viridis"
            else:
                cmap = self.cmap
            colors = sns.color_palette(cmap, self.df_pars.shape[0])
        else:
            colors = self.color

        if ttp_lines:
            # heights need to be adjusted for by axis length 
            if not isinstance(hrf_axs, mpl.axes._axes.Axes):
                raise ValueError(f"Need an axes-object containing HRF profiles to draw lines on")
            
            ylim = hrf_axs.get_ylim()
            tot = sum(list(np.abs(ylim)))
            start = (0-ylim[0])/tot

            for ix,ii in enumerate(self.df_pars["time_to_peak"].values):
                hrf_axs.axvline(
                    ii, 
                    ymin=start, 
                    ymax=self.df_pars["magnitude"].values[ix]/tot+start, 
                    color=colors[ix], 
                    linewidth=0.5)        
        
        self.ttp_plot = plotting.LazyBar(
            data=self.df_pars,
            x=x_lbl,
            y="time_to_peak",
            labels=ttp_labels,
            sns_ori=ttp_ori,
            axs=axs,
            error=None,
            **kwargs
        )   
        
    def plot_fwhm(
        self, 
        df,
        axs=None, 
        hrf_axs=None, 
        fwhm_lines=False, 
        fwhm_labels=None, 
        split="event_type",
        figsize=(8,8), 
        fwhm_ori='v', 
        **kwargs):

        if axs == None:
            fig, axs = plt.subplots(figsize=figsize)

        # unstack series | assume split is on event_type
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df).unstack(level=split)
            x_lbl = split
            x_data = self.cond
        else:
            x_lbl = "vox"
            x_data = list(df.columns)

        self.df_pars = HRFMetrics(df).return_metrics()
        self.df_pars[x_lbl] = x_data

        # decide on color based on split
        if not hasattr(self, "color"):
            if not hasattr(self, "cmap"):
                cmap = "viridis"
            else:
                cmap = self.cmap
            colors = sns.color_palette(cmap, self.df_pars.shape[0])
        else:
            colors = self.color

        if fwhm_lines:
            # heights need to be adjusted for by axis length 
            xlim = hrf_axs.get_xlim()
            tot = sum(list(np.abs(xlim)))
            for ix, ii in enumerate(self.df_pars["fwhm"].values):
                start = (self.df_pars["half_rise_time"].values[ix]-xlim[0])/tot
                half_m = self.df_pars["half_max"].values[ix]
                hrf_axs.axhline(
                    half_m, 
                    xmin=start, 
                    xmax=start+ii/tot, 
                    color=colors[ix], 
                    linewidth=0.5)

        self.fwhm_labels = fwhm_labels
        self.fwhm_plot = plotting.LazyBar(
            data=self.df_pars,
            x=x_lbl,
            y="fwhm",
            palette=colors,
            sns_ori=fwhm_ori,
            axs=axs,
            error=None,
            **kwargs)          

    def plot_average_per_voxel(
        self, 
        add_offset: bool=True, 
        axs=None, 
        n_cols: int=4, 
        fig_kwargs: dict={}, 
        figsize: tuple=None, 
        make_figure: bool=True, 
        labels: list=None, 
        save_as: str=None, 
        sharey: bool=False, 
        skip_x: list=None,
        skip_y: list=None,
        title: list=None,
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

        if not hasattr(self, "tc_condition"):
            self.timecourses_condition()

        cols = list(self.tc_condition.columns)
        cols_id = np.arange(0, len(cols))
        if n_cols != None:

            if isinstance(axs, (list,np.ndarray)):
                if len(axs) != len(cols):
                    raise ValueError(f"For this option {len(cols)} axes are required, {len(axs)} were specified")
            else:
                # initiate figure
                if len(cols) > 10:
                    raise Exception(f"{len(cols)} were requested. Maximum number of plots is set to 30")

                n_rows = int(np.ceil(len(cols) / n_cols))
                if not isinstance(figsize, tuple):
                    figsize = (24,5*n_rows)

                fig = plt.figure(figsize=figsize, constrained_layout=True)
                gs = fig.add_gridspec(n_rows, n_cols, **fig_kwargs)
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

        # try to find min/max across voxels. Use error if there were more than 1 run
        top = self.arr_voxels_in_event
        bottom = self.arr_voxels_in_event
        if len(self.run_ids) > 1:
            top += self.arr_error_in_event
            bottom -= self.arr_error_in_event
        
        if make_figure:
            if n_cols == None:
                if labels:
                    labels = cols.copy()
                else:
                    labels = None

                vox_data = [self.arr_voxels_in_event[ii,0,:] for ii in range(self.arr_voxels_in_event.shape[0])]
                vox_error = [self.arr_error_in_event[ii, 0, :] for ii in range(self.arr_voxels_in_event.shape[0])]
                
                if isinstance(axs, mpl.axes._axes.Axes):
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
                    if isinstance(axs, (np.ndarray,list)):
                        ax = axs[ix]
                        new_axs = True
                    else:
                        new_axs = False
                        ax = fig.add_subplot(gs[ix])

                    if ix == 0:
                        label = labels
                    else:
                        label = None

                    vox_data = [self.arr_voxels_in_event[ix,ii,:] for ii in range(len(self.cond))]
                    vox_error = [self.arr_error_in_event[ix,ii,:] for ii in range(len(self.cond))]                    
                    
                    if sharey:
                        ylim = [np.nanmin(bottom), np.nanmax(top)]
                    else:
                        ylim = None
                    
                    if isinstance(title, (str,list)):
                        if isinstance(title, str):
                            title = [title for _ in cols]
                        
                        add_title = title[ix]
                    else:
                        add_title = col

                    self.pl = plotting.LazyPlot(
                        vox_data,
                        xx=self.time,
                        error=vox_error,
                        axs=ax,
                        labels=label,
                        add_hline='default',
                        y_lim=ylim,
                        title=add_title,
                        **kwargs)

                    if not new_axs:
                        if ix in cols_id[::n_cols]:
                            ax.set_ylabel("Magnitude (%change)", fontsize=self.font_size)
                        
                        if ix in np.arange(n_rows*n_cols)[-n_cols:]:
                            ax.set_xlabel("Time (s)", fontsize=self.font_size)
                    else:
                        if not isinstance(skip_x, list):
                            skip_x = [False for _ in cols]
                        
                        if not isinstance(skip_y, list):
                            skip_y = [False for _ in cols]

                        if not skip_x[ix]:
                            ax.set_xlabel("Time (s)", fontsize=self.font_size)
                        
                        if not skip_y[ix]:
                            ax.set_ylabel("Magnitude (%change)", fontsize=self.font_size)

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

def fwhm_lines(
    fwhm_list,
    axs,
    cmap="viridis",
    color=None,
    **kwargs):

    if not isinstance(fwhm_list, list):
        fwhm_list = [fwhm_list]

    # heights need to be adjusted for by axis length
    if not isinstance(color,(str,tuple)):
        colors = sns.color_palette(cmap, len(fwhm_list))
    else:
        colors = [color for _ in range(len(fwhm_list))]

    xlim = axs.get_xlim()
    tot = sum(list(np.abs(xlim)))
    for ix, ii in enumerate(fwhm_list):
        start = (ii.hmx[0]-xlim[0])/tot
        axs.axhline(
            ii.half_max, 
            xmin=start, 
            xmax=start+ii.fwhm/tot, 
            color=colors[ix], 
            linewidth=0.5,
            **kwargs)

class HRFMetrics():

    def __init__(
        self,
        hrf,
        TR=None,
        force_pos=False,
        force_neg=False):

        self.hrf = hrf
        self.TR = TR
        self.force_pos = force_pos
        self.force_neg = force_neg
        self._get_metrics(
            TR=self.TR, 
            force_pos=self.force_pos,
            force_neg=self.force_neg)

    def return_metrics(self):
        return self.metrics
    
    @staticmethod
    def _check_negative(hrf, force_pos=False, force_neg=False):
        # check if positive or negative is largest
        if abs(hrf.min(axis=0).values) > hrf.max(axis=0).values:
            negative = True
        else:
            negative = False

        if force_pos:
            negative = False
        
        if force_neg:
            negative = True
        return negative
    
    def _get_metrics(
        self, 
        TR=None,
        force_pos=False,
        force_neg=False):

        hrf = self._verify_input(self.hrf, TR=TR)

        if not isinstance(force_neg, list):
            force_neg = [force_neg for _ in list(hrf.columns)]

        if not isinstance(force_pos, list):
            force_pos = [force_pos for _ in list(hrf.columns)]
        
        # print(force_pos)
        self.metrics = []
        self.fwhm_objs = []
        for ix,col in enumerate(list(hrf.columns)):
            pars,fwhm_ = self._get_single_hrf_metrics(
                hrf[col],
                TR=self.TR,
                force_pos=force_pos[ix],
                force_neg=force_neg[ix])
            
            self.metrics.append(pars)
            self.fwhm_objs.append(fwhm_)

        if len(self.metrics) > 0:
            self.metrics = pd.concat(self.metrics)

    @staticmethod
    def _get_time(hrf):
        try:
            tmp_df = hrf.reset_index()
        except:
            tmp_df = hrf.copy()

        try:
            return tmp_df["time"].values
        except:
            try:
                return tmp_df["t"].values
            except:
                raise ValueError("Could not find time dimension. Dataframe should contain 't' or 'time' column..")
    
    @classmethod
    def _verify_input(self, hrf, TR=None):
        if isinstance(hrf, np.ndarray):
            if hrf.ndim > 1:
                hrf = hrf.squeeze()
            
            if not isinstance(TR, (int,float)):
                raise ValueError(f"Please specify repetition time of this acquisition to construct time axis")
            
            time_axis = list(np.arange(0,hrf.shape[0])*TR)
            hrf = pd.DataFrame({"voxel": hrf})
            hrf["time"] = time_axis
            hrf = hrf.set_index(["time"])

        elif isinstance(hrf, pd.Series):
            hrf = pd.DataFrame(hrf)

        return hrf

    @classmethod
    def _get_riseslope(self, hrf, force_pos=False, force_neg=False):

        # fetch time stamps
        time = self._get_time(hrf)

        # check negative:
        negative = self._check_negative(
            hrf, 
            force_neg=force_neg,
            force_pos=force_pos)

        # find slope corresponding to amplitude
        mag = self._get_amplitude(
            hrf, 
            force_pos=force_pos,
            force_neg=force_neg)

        # limit search to where index of highest amplitude
        diff = np.diff(hrf.values.squeeze()[:mag["t_ix"]])/np.diff(time[:mag["t_ix"]])

        # find minimum/maximum depending on whether HRF is negative or not
        if not force_pos:
            if negative:
                val = np.array([np.amin(diff)])
            else:
                val = np.array([np.amax(diff)])
        else:
            val = np.array([np.amax(diff)])

        val_ix = utils.find_nearest(diff,val)[0]
        val_t = time[val_ix]
        return val,val_t
    
    @classmethod
    def _get_amplitude(self, hrf, force_pos=False, force_neg=False):

        # fetch time stamps
        time = self._get_time(hrf)

        # check negative:
        negative = self._check_negative(
            hrf, 
            force_neg=force_neg,
            force_pos=force_pos)

        if not force_pos:
            if negative:
                mag_tmp = hrf.min(axis=0).values
            else:
                mag_tmp = hrf.max(axis=0).values
        else:
            mag_tmp = hrf.max(axis=0).values

        mag_ix = utils.find_nearest(hrf.values.squeeze(), mag_tmp)[0]

        return {
            "amplitude": mag_tmp,
            "t": time[mag_ix],
            "t_ix": mag_ix
        }

    @classmethod    
    def _get_fwhm(
        self,
        hrf,
        force_pos=False, 
        force_neg=False):

        # fetch time stamps
        time = self._get_time(hrf)

        # check negative:
        negative = self._check_negative(
            hrf, 
            force_neg=force_neg,
            force_pos=force_pos)

        # check time stamps
        if time.shape[0] != hrf.values.shape[0]:
            raise ValueError(f"Shape of time dimension ({time.shape[0]}) does not match dimension of HRF ({hrf.values.shape[0]})")
        
        # get amplitude of HRF
        mag = self._get_amplitude(
            hrf, 
            force_pos=force_pos,
            force_neg=force_neg)

        # define index period around magnitude; add 20% to avoid FWHM errors
        end_ix = mag["t_ix"]+mag["t_ix"]
        end_ix += int(end_ix*0.2)

        # get fwhm around max amplitude
        fwhm_val = FWHM(
            time[:end_ix], 
            hrf.values[:end_ix], 
            negative=negative)

        return {
            "fwhm": fwhm_val.fwhm,
            "half_rise": fwhm_val.hmx[0],
            "half_max": fwhm_val.half_max,
            "obj": fwhm_val
        }
            
    def _get_single_hrf_metrics(
        self, 
        hrf, 
        TR=None, 
        force_pos=False,
        force_neg=False):

        # verify input type
        hrf = self._verify_input(hrf, TR=TR)

        # get magnitude and amplitude
        mag = self._get_amplitude(
            hrf, 
            force_pos=force_pos, 
            force_neg=force_neg)

        # fwhm
        fwhm_obj = self._get_fwhm(
            hrf, 
            force_pos=force_pos, 
            force_neg=force_neg)

        # rise slope
        rise_tmp, rise_t = self._get_riseslope(
            hrf, 
            force_pos=force_pos, 
            force_neg=force_neg)
                
        df = pd.DataFrame(
            {
                "magnitude": mag["amplitude"], 
                "magnitude_ix": mag["t_ix"], 
                "fwhm": fwhm_obj["fwhm"], 
                "time_to_peak": mag["t"],
                "half_rise_time": fwhm_obj["half_rise"],
                "half_max": fwhm_obj["half_max"],
                "rise_slope": rise_tmp,
                "rise_slope_t": rise_t
            }
        )

        return df,fwhm_obj

class FWHM():

    def __init__(self, x, hrf, negative=False):
        
        self.x = x
        self.hrf = hrf
        self.negative = negative

        try:
            self.hmx = self.half_max_x(self.x,self.hrf)
            self.fwhm = self.hmx[1] - self.hmx[0]
            
            if self.negative:
                self.half_max = min(hrf)/2
            else:
                self.half_max = max(hrf)/2
        except:
            self.hmx = [np.nan,np.nan]
            self.fwhm = np.nan
    
    def lin_interp(self, x, y, i, half):
        return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

    def half_max_x(self, x, y):

        if not self.negative:
            half = max(y)/2.0
        else:
            half = min(y)/2.0
            
        signs = np.sign(np.add(y, -half))
        zero_crossings = (signs[0:-2] != signs[1:-1])
        zero_crossings_i = np.where(zero_crossings)[0]
        return [self.lin_interp(x, y, zero_crossings_i[0], half),
                self.lin_interp(x, y, zero_crossings_i[1], half)]

def error_function(
    parameters,
    args,
    data,
    objective_function):

    """
    Parameters
    ----------
    parameters : list or ndarray
        A tuple of values representing a model setting.
    args : dictionary
        Extra arguments to `objective_function` beyond those in `parameters`.
    data : ndarray
       The actual, measured time-series against which the model is fit.
    objective_function : callable
        The objective function that takes `parameters` and `args` and
        produces a model time-series.

    Returns
    -------
    error : float
        The residual sum of squared errors between the prediction and data.
    """
    return np.nan_to_num(np.sum((data - objective_function(parameters, **args))**2), nan=1e12)
    #return 1-np.nan_to_num(pearsonr(data,np.nan_to_num(objective_function(*list(parameters), **args)[0]))[0])

def double_gamma_with_d(a1,a2,b1,b2,c,d1,d2,x=None, negative=False):    
    
    # correct for negative onsets
    if x[0]<0:
        x-=x[0]

    if not negative:
        y = (x/(d1))**a1 * np.exp(-(x-d1)/b1) - c*(x/(d2))**a2 * np.exp(-(x-d2)/b2)
    else:
        y = (x/(d1))**a1 * -np.exp(-(x-d1)/b1) - c*(x/(d2))**a2 * -np.exp(-(x-d2)/b2)        

    y[x < 0] = 0
    
    return y

def make_prediction(
    parameters,
    onsets=None,
    scan_length=None,
    TR=1.32,
    osf=100,
    cov_as_ampl=None,
    negative=False,
    interval=[0,25]
    ):

    # this correction is needed to get the HRF in the correct time domain
    dt = 1/(osf/TR)

    time_points = np.linspace(*interval, np.rint(float(interval[-1])/dt).astype(int))
    hrf = [double_gamma_with_d(
        *parameters,
        x=time_points,
        negative=negative)]

    ev_ = glm.make_stimulus_vector(
        onsets, 
        scan_length=scan_length, 
        TR=TR, 
        osf=osf,
        cov_as_ampl=cov_as_ampl
    )

    ev_conv = glm.convolve_hrf(
        hrf, 
        ev_, 
        TR=TR,
        osf=osf,
        time=time_points,
        # make_figure=True
    )
    
    ev_conv_rs = glm.resample_stim_vector(ev_conv, scan_length)
    key = list(ev_conv_rs.keys())[0]
    
    return ev_conv_rs[key].squeeze()    
    
def iterative_search(
    data,
    onsets,
    starting_params=[6,12,0.9,0.9,0.35,5.4,10.8],
    bounds=None,
    constraints=None,
    cov_as_ampl=None,
    TR=1.32,
    osf=100,
    interval=[0,25],
    xtol=1e-4,
    ftol=1e-4):
    """iterative_search

    Im actually using this function..

    Parameters
    ----------
    data: _type_
        _description_
    onsets: _type_
        _description_
    starting_params: list, optional
        _description_, by default [6,12,0.9,0.9,0.35,5.4,10.8]
    bounds: _type_, optional
        _description_, by default None
    constraints: _type_, optional
        _description_, by default None
    cov_as_ampl: _type_, optional
        _description_, by default None
    TR: float, optional
        _description_, by default 1.32
    osf: int, optional
        _description_, by default 100
    interval: list, optional
        _description_, by default [0,25]
    xtol: _type_, optional
        _description_, by default 1e-4
    ftol: _type_, optional
        _description_, by default 1e-4

    Returns
    ----------
    _type_
        _description_

    Example
    ----------
    >>> 
    """
    # data = (voxels,time)
    if data.ndim > 1:
        data = data.squeeze()

    scan_length = data.shape[0]

    # args
    args = {
        "onsets": onsets,
        "scan_length": scan_length,
        "cov_as_ampl": cov_as_ampl,
        "TR": TR,
        "osf": osf,
        "interval": interval
    }
    
    # run for both negative and positive; return parameters of best r2

    # res = {}
    # for lbl,np in zip(["pos","neg"],[False,True]):
    #     res[lbl] = {}
    #     args["negative"] = np
    #     output = minimize(
    #         error_function, 
    #         starting_params, 
    #         args=(
    #             args, 
    #             data,
    #             make_prediction),
    #         method='trust-constr',
    #         bounds=bounds,
    #         tol=ftol,
    #         options=dict(xtol=xtol)
    #     )   

    #     pred = make_prediction(
    #         output['x'],
    #         **args
    #     )

    #     res[lbl]["pars"] = output['x']
    #     res[lbl]["r2"] = metrics.r2_score(data,pred)  

    # r2_pos = res["pos"]["r2"]
    # r2_neg = res["neg"]["r2"]
    # if r2_pos>r2_neg:
    #     return res["pos"]["pars"],"pos"
    # else:
    #     return res["neg"]["pars"],"neg"
    
    return starting_params, "pos"
    
class FitHRFparams():

    def __init__(
        self,
        data,
        onsets,
        verbose=False,
        TR=1.32,
        osf=100,
        starting_params=[6,12,0.9,0.9,0.35,5.4,10.8],
        n_jobs=1,
        bounds=None,
        constraints=None,
        resample_to_shape=None,
        xtol=1e-4,
        ftol=1e-4,
        cov_as_ampl=None,
        interval=[0,25],
        read_index=False,
        **kwargs):

        self.data = data
        self.onsets = onsets
        self.verbose = verbose
        self.TR = TR
        self.osf = osf
        self.interval = interval
        self.starting_params = starting_params
        self.n_jobs = n_jobs
        self.bounds = bounds
        self.constraints = constraints
        self.xtol = xtol
        self.ftol = ftol
        self.cov_as_ampl = cov_as_ampl
        self.resample_to_shape = resample_to_shape
        self.read_index = read_index

        # set default bounds that can be updated with kwargs
        if not isinstance(self.bounds, list):
            self.bounds = [
                (4,8),
                (10,14),
                (0.8,1.2),
                (0.8,1.2),
                (0,0.5),
                (0,10),
                (5,15)
            ]

        self.__dict__.update(kwargs)
    
    def iterative_fit(self):

        # data = (voxels,time)
        if self.data.ndim < 2:
            self.data = self.data[np.newaxis,...]
    
        self.tmp_results = Parallel(self.n_jobs, verbose=self.verbose)(
            delayed(iterative_search)(
                self.data[i,:],
                self.onsets,
                starting_params=self.starting_params,
                TR=self.TR,
                bounds=self.bounds,
                constraints=self.constraints,
                xtol=self.xtol,
                ftol=self.ftol,
                cov_as_ampl=self.cov_as_ampl
            ) for i in range(self.data.shape[0])
        )


        # parse into array
        self.iterative_search_params = np.array([self.tmp_results[i][0] for i in range(self.data.shape[0])])
        self.prof_sign = [self.tmp_results[i][1] for i in range(self.data.shape[0])]

        # self.iterative_search_params = np.array(self.starting_params)[np.newaxis,...]
        # self.prof_sign = None

        self.force_neg = [False for _ in range(self.data.shape[0])]
        self.force_pos = [True for _ in range(self.data.shape[0])]

        if isinstance(self.prof_sign, list):
            for ix,el in enumerate(self.prof_sign):
                if el == "neg":
                    self.force_neg[ix] = True
                    self.force_pos[ix] = False

        # also immediately create profiles
        self.profiles_from_parameters(
            resample_to_shape=self.resample_to_shape, 
            negative=self.prof_sign,
            read_index=self.read_index)

        self.hrf_pars = HRFMetrics(
            self.hrf_profiles,
            force_neg=self.force_neg,
            force_pos=self.force_pos
        ).return_metrics()
        
    def profiles_from_parameters(self, resample_to_shape=None, negative=None, read_index=False):

        assert hasattr(self, "iterative_search_params"), "No parameters found, please run iterative_fit()"

        dt = 1/self.osf
        time_points = np.linspace(*self.interval, np.rint(float(self.interval[-1])/dt).astype(int))
    
        hrfs = []
        preds = []
        pars = []
        for i in range(self.iterative_search_params.shape[0]):
            neg = False
            if isinstance(negative, list):
                if negative[i] in ["negative","neg"]:
                    neg = True

            pars = list(self.iterative_search_params[i,:])
            hrf = double_gamma_with_d(*pars,x=time_points, negative=neg)

            pred = make_prediction(
                pars,
                onsets=self.onsets,
                scan_length=self.data.shape[1],
                TR=self.TR,
                osf=self.osf,
                cov_as_ampl=self.cov_as_ampl,
                interval=self.interval,
                negative=neg
                )

            # resample to specified length
            if isinstance(resample_to_shape, int):
                hrf = glm.resample_stim_vector(hrf, resample_to_shape)

            hrfs.append(hrf)
            preds.append(pred)

        hrf_profiles = np.array(hrfs)
        predictions = np.array(preds)

        if isinstance(resample_to_shape, int):
            time_points = np.linspace(*self.interval, num=resample_to_shape)

        self.hrf_profiles = pd.DataFrame(hrf_profiles.T)
        self.hrf_profiles["time"] = time_points
        self.predictions = pd.DataFrame(predictions.T)
        self.predictions["t"] = list(np.arange(0,self.data.shape[1])*self.TR)

        # read indices from onset dataframe
        self.prof_indices = ["time"]
        self.pred_indices = ["t"]

        self.custom_indices = []
        for el in ["subject","run","event_type"]:
            try:
                el_in_df = utils.get_unique_ids(self.onsets, id=el)
            except:
                el_in_df = None
        
            if isinstance(el_in_df, list):
                self.custom_indices.append(el)

                for df in [self.hrf_profiles,self.predictions]:
                    df[el] = el_in_df[0]

        if len(self.custom_indices)>0:
            self.prof_indices = self.custom_indices+self.prof_indices
            self.pred_indices = self.custom_indices+self.pred_indices

        self.hrf_profiles = self.hrf_profiles.set_index(self.prof_indices)
        self.predictions = self.predictions.set_index(self.pred_indices)        
    
