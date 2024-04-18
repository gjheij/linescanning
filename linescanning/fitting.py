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
from alive_progress import alive_bar

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

class InitFitter():

    def __init__(
        self,
        func,
        onsets,
        TR,
        merge=False,
        *args,
        **kwargs
        ):

        self.func = func
        self.onsets = onsets
        self.TR = TR
        self.merge = merge

        # format inputs into Fitter compatbile input
        self.prepare_data()
        self.prepare_onsets()

        # concatenate runs
        if self.merge:
            self.concatenate()

    def concatenate(self):
        # store originals
        self.og_func = self.func.copy()
        self.og_onsets = self.onsets.copy()

        # create concatenator dictionary
        self.concat_obj = self.concat_runs(self.func,self.onsets)

        # store new dataframes
        self.func = self.concat_obj["func"]
        self.onsets = self.concat_obj["onsets"]

    def _get_timepoints(self):
        return list(np.arange(0,self.func.shape[0])*self.TR)

    def prepare_onsets(self):
        
        # store copy of original data
        self.orig_onsets = self.onsets.copy()
        
        # put in dataframe
        self.old_index = []
        if isinstance(self.onsets, np.ndarray):
            self.onsets = pd.DataFrame(self.onsets)
        else:
            try:
                # check old index
                try:
                    self.old_index = list(self.onsets.index.names)
                except:
                    pass
                self.onsets.reset_index(inplace=True)
            except:
                pass
            
        # format dataframe with subject, run, and t
        try:
            self.task_ids = utils.get_unique_ids(self.onsets, id="task")
            self.final_index = [
                "subject",
                "task",
                "run",
                "event_type"
            ]

            self.final_elements = [1,None,1,"stim"]

        except:
            self.task_ids = None
            self.final_index = [
                "subject",
                "run",
                "event_type"
            ]
            self.final_elements = [1,1,"stim"]
            
        self.onset_index = [
            "subject",
            "run",
            "event_type"
        ]

        for key,val in zip(self.final_index, self.final_elements):
            if not key in list(self.onsets.columns):
                self.onsets[key] = val

        self.drop_cols = []
        if len(self.old_index)>0:
            for ix in self.old_index:
                if ix not in self.final_index:
                    self.drop_cols.append(ix)

        if len(self.drop_cols)>0:
            self.onsets = self.onsets.drop(self.drop_cols, axis=1)

        self.onsets.set_index(self.final_index, inplace=True)   
    
    def prepare_data(self):
        
        # store copy of original data
        self.orig_data = self.func.copy()
        
        # put in dataframe
        self.time = self._get_timepoints()
        self.old_index = []
        if isinstance(self.func, np.ndarray):
            self.func = pd.DataFrame(self.func)
        else:
            try:
                # check old index
                try:
                    self.old_index = list(self.func.index.names)
                except:
                    pass
                self.func = self.func.reset_index()
            except:
                pass
        
        # format dataframe with subject, run, and t
        try:
            self.task_ids = utils.get_unique_ids(self.func, id="task")
            self.final_index = [
                "subject",
                "task",
                "run",
                "t"
            ]

            self.final_elements = [1,None,1,self.time]

        except:
            self.task_ids = None
            self.final_index = [
                "subject",
                "run",
                "t"
            ]
            self.final_elements = [1,1,self.time]

        for key,val in zip(self.final_index, self.final_elements):
            if not key in list(self.func.columns):
                self.func[key] = val

        self.drop_cols = []
        if len(self.old_index)>0:
            for ix in self.old_index:
                if ix not in self.final_index:
                    self.drop_cols.append(ix)

        if len(self.drop_cols)>0:
            self.func = self.func.drop(self.drop_cols, axis=1)

        self.func.set_index(self.final_index, inplace=True)              

    @classmethod
    def concat_func(self, df):

        # check if time matches TR
        t = utils.get_unique_ids(df, id="t")
        tr = np.diff(t)[0]
        tr_list = list(np.arange(0,df.shape[0])*tr)

        # construct new dataframe
        new_func = pd.DataFrame(df.values, columns=list(df.columns))
        new_func["subject"] = utils.get_unique_ids(df, id="subject")[0]
        new_func["t"],new_func["run"] = np.array(tr_list),1
        new_func.set_index(["subject","run","t"], inplace=True)

        return new_func
    
    @classmethod
    def concat_onsets(
        self, 
        onsets, 
        func
        ):

        n_runs = utils.get_unique_ids(onsets, id="run")
        run_dfs = []
        for ix,ii in enumerate(n_runs):
            df = utils.select_from_df(onsets, expression=f"run = {ii}").reset_index()
            ff = utils.select_from_df(func, expression=f"run = {ii}")
            df["onset"] += (ix*ff.shape[0])
            df["run"] = 1
            run_dfs.append(df)

        new_onsets = pd.concat(run_dfs)
        new_onsets.set_index(["subject","run","event_type"], inplace=True)

        return new_onsets    

    @classmethod
    def concat_runs(
        self,
        func,
        onsets
        ):
        
        sub_ids = utils.get_unique_ids(func, id="subject")
        sub_dfs = {}
        sub_dfs["func"] = []
        sub_dfs["onsets"] = []

        # loop through subjects
        for sub in sub_ids:

            # concatenate functional
            sub_func = utils.select_from_df(func, expression=f"subject = {sub}")
            sub_dfs["func"].append(self.concat_func(sub_func))

            # concatenate onsets
            sub_onsets = utils.select_from_df(onsets, expression=f"subject = {sub}")
            sub_dfs["onsets"].append(self.concat_onsets(sub_onsets, sub_func))

        for key,val in sub_dfs.items():
            sub_dfs[key] = pd.concat(val)

        return sub_dfs
    
    def parameters_for_tc_subjects(
        self, 
        *args,
        **kwargs
        ):

        if not hasattr(self, "tc_subjects"):
            raise ValueError(f"{self} does not have 'tc_subjects' attribute, run fitter first")

        utils.verbose(f"Deriving parameters from {self} with 'HRFMetrics'", self.verbose)
        subj_ids = utils.get_unique_ids(self.tc_subjects, id="subject")

        subjs = []
        subjs_avg_run = []
        for sub in subj_ids:
            
            sub_df = utils.select_from_df(self.tc_subjects, expression=f"subject = {sub}")
            ev_ids = utils.get_unique_ids(sub_df, id="event_type")

            # loop through evs
            evs = []
            evs_avg = []
            for ev in ev_ids:
                
                # get event-specific dataframe
                ev_df = utils.select_from_df(sub_df, expression=f"event_type = {ev}")
                run_ids = utils.get_unique_ids(sub_df, id="run")

                # loop through runs
                runs = []
                for run in run_ids:
                    
                    # get run-specific dataframe
                    run_df = utils.select_from_df(ev_df, expression=f"run = {run}")
                    pars = HRFMetrics(
                        run_df, 
                        *args,
                        **kwargs
                    ).return_metrics()

                    pars["event_type"],pars["run"] = ev,run
                    runs.append(pars)
                
                # also get parameters of average across runs
                avg_df = ev_df.groupby(["subject","event_type","time"]).mean()
                avg_pars = HRFMetrics(
                    avg_df, 
                    *args,
                    **kwargs
                ).return_metrics()

                # save average
                avg_pars["event_type"] = ev
                evs_avg.append(avg_pars)

                # save single runs
                runs = pd.concat(runs)
                evs.append(runs)

            # avg
            evs_avg = pd.concat(evs_avg)
            evs_avg["subject"] = sub
            subjs_avg_run.append(evs_avg)

            # single runs
            evs = pd.concat(evs)
            evs["subject"] = sub
            subjs.append(evs)

        df_conc = pd.concat(subjs)
        df_avg = pd.concat(subjs_avg_run)
        avg_idx = ["subject","event_type"]
        idx = avg_idx+["run"]
        if "vox" in list(df_conc.columns):
            idx += ["vox"]
            avg_idx += ["vox"]
        
        self.pars_subjects = df_conc.set_index(idx)
        self.avg_pars_subjects = df_avg.set_index(avg_idx)

class ParameterFitter(InitFitter):

    def __init__(
        self, 
        func, 
        onsets, 
        TR=0.105,
        merge=False,
        *args,
        **kwargs
        ):

        self.func = func
        self.onsets = onsets
        self.TR = TR
        self.merge = merge

        # prepare data
        super().__init__(
            self.func, 
            self.onsets, 
            self.TR,
            merge=self.merge
        )

        # self.func is overwritting by FitHRFParams class; so store original inputs otherwise looping goes awry
        self.orig_func = self.func.copy()
        self.orig_onsets = self.onsets.copy()

        # get info about dataframe
        self.sub_ids = utils.get_unique_ids(self.orig_func, id="subject")
        self.run_ids = utils.get_unique_ids(self.orig_func, id="run")
        self.ev_ids = utils.get_unique_ids(self.orig_onsets, id="event_type")

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

    def fit(
        self, 
        debug=False,
        **kwargs
        ):

        self.ddict_sub = self._set_dict()

        # loop through subject
        for sub in self.sub_ids:
            
            utils.verbose(f"Fitting '{sub}'", debug)

            # get subject specific onsets
            self.sub_onsets = utils.select_from_df(self.orig_onsets, expression=f"subject = {sub}")
            self.ddict_run = self._set_dict()
            for run in self.run_ids:
                
                # subject and run specific data
                utils.verbose(f" run-'{run}'", debug)
                expr = (
                    f"subject = {sub}",
                    "&",
                    f"run = {run}"
                )
                self.run_func = utils.select_from_df(self.orig_func, expression=expr)
                
                # loop through events
                self.ddict_ev = self._set_dict()
                for ev in self.ev_ids:
                    
                    utils.verbose(f"  ev-'{ev}'", debug)

                    # run/event-specific onsets
                    expr = (
                        f"run = {run}",
                        "&",
                        f"event_type = {ev}"
                    )
                    self.ons = utils.select_from_df(self.sub_onsets, expression=expr)
                    # print(f"  {self.run_func.shape}")

                    # do fit
                    self.rf = self.single_response_fitter(
                        self.run_func.T.values,
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

                    try:
                        df = df.reset_index(drop=True)
                    except:
                        pass

                    # append
                    self.ddict_ev["predictions"].append(pr)
                    self.ddict_ev["profiles"].append(df)

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

        self.grouper = self.tc_subjects.groupby(level=['event_type', 'covariate', 'time'])
        self.tc_condition = self.grouper.mean()
        self.sem_condition = self.grouper.sem()
        self.std_condition = self.grouper.std()

        self.ev_predictions = self.ddict_sub["predictions"].set_index(["subject","run","event_type","t"])

class NideconvFitter(InitFitter):
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
        merge=False, 
        verbose=False, 
        lump_events=False, 
        interval=[0,20], 
        osf=20,
        fit=True,
        covariates=None,
        conf_intercept=True,
        **kwargs):

        self.func = func
        self.onsets = onsets
        self.confounds = confounds
        self.basis_sets = basis_sets 
        self.fit_type = fit_type
        self.n_regressors = n_regressors
        self.add_intercept = add_intercept
        self.verbose = verbose
        self.lump_events = lump_events
        self.TR = TR
        self.fs = 1/self.TR
        self.interval = interval
        self.merge = merge
        self.osf = osf
        self.do_fit = fit
        self.covariates = covariates
        self.conf_icpt = conf_intercept

        # check if interval is multiple of TR
        if isinstance(self.interval[-1], str):
            self.interval[-1] = int(self.interval[-1])*self.TR

        # format arrays as dataframe
        super().__init__(
            self.func, 
            self.onsets, 
            self.TR,
            merge=self.merge
        )

        # merge all events?
        if self.lump_events:
            self.used_onsets = self.melt_events()
        else:
            self.used_onsets = self.onsets.copy()
        
        # get n regressors
        self.derive_n_regressors()

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

        # set plotting defaults
        self.set_plotting_defaults()

    def set_plotting_defaults(self):
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

    def melt_events(self):
        self.lumped_onsets = self.onsets.copy().reset_index()
        self.lumped_onsets['event_type'] = 'stim'
        self.lumped_onsets = self.lumped_onsets.set_index(['subject', 'run', 'event_type'])
        return self.lumped_onsets
    
    def derive_n_regressors(self):

        self.allowed_basis_sets = [
            "fir",
            "fourier",
            "dct",
            "legendre",
            "canonical_hrf",
            "canonical_hrf_with_time_derivative",
            "canonical_hrf_with_time_derivative_dispersion"
        ]

        self.n_regressors
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
                    
                    self.tmp_regressors = round(((self.interval[1]+abs(self.interval[0])))/self.TR)
                    
                    if len(self.n_regressors)>2:
                        # assume operation on TR
                        els = self.n_regressors.split(" ")

                        if len(els) != 3:
                            raise TypeError(f"Format of this input must be 'tr <operation> <value>' (WITH SPACES), not '{self.n_regressors}'. Example: 'tr x 2' will use 2 regressors per TR")
                        
                        op = utils.str2operator(els[1])
                        val = float(els[-1])

                        self.n_regressors = int(op(self.tmp_regressors,val))
                    else:
                        self.n_regressors = self.tmp_regressors

    @staticmethod
    def get_event_predictions_from_fitter(fitter, intercept=True):

        ev_pred = []
        for ev in fitter.events.keys():

            if intercept:
                X_stim = pd.concat(
                    [
                        fitter.X.xs("confounds", axis=1, drop_level=False),
                        fitter.X.xs(ev, axis=1, drop_level=False)
                    ],
                    axis=1
                )

                expr = (
                    f"event type = {ev}",
                    "or",
                    "event type = confounds"
                )
            else:
                X_stim = fitter.X.xs(ev, axis=1, drop_level=False)
                expr = f"event type = {ev}"

            betas = utils.select_from_df(fitter.betas, expression=expr)
            pred = X_stim.dot(betas).reset_index()
            pred.rename(columns={"time": "t"}, inplace=True)

            if "index" in list(pred.columns):
                pred.rename(columns={"index": "t"}, inplace=True)

            pred["event_type"] = ev
            ev_pred.append(pred)

        ev_pred = pd.concat(ev_pred, ignore_index=True)
        return ev_pred

    def interpolate_fir_subjects(self):

        # loop through subject IDs        
        subjs = utils.get_unique_ids(self.tc_subjects, id="subject")
        sub_df = []
        for sub in subjs:
            
            # get subject specific dataframes
            sub_tmp = utils.select_from_df(self.tc_subjects, expression=f"subject = {sub}")
            run_ids = utils.get_unique_ids(sub_tmp, id="run")

            # loop through runs
            run_df = []
            for run in run_ids:
                    
                # get run-specific dataframe
                run_tmp = utils.select_from_df(sub_tmp, expression=f"run = {run}")

                # call the interpolate function
                run_interp = self.interpolate_fir_condition(run_tmp)

                # make indexing the same as input
                run_interp.columns = run_tmp.columns
                run_interp = run_interp.reset_index()
                run_interp["run"] = run

                # append interpolated run dataframe
                run_df.append(run_interp)

            # concatenate run dataframe and add subject inedx
            run_df = pd.concat(run_df)
            run_df["subject"] = sub
            sub_df.append(run_df)

        # final indexing
        sub_df = pd.concat(sub_df)
        sub_df.set_index(["subject","run","event_type","covariate","time"], inplace=True)
        return sub_df

    def interpolate_fir_condition(self, obj):

        evs = utils.get_unique_ids(obj, id="event_type")
        evs_interp = []
        for ev in evs:

            ev_df = utils.select_from_df(obj, expression=f"event_type = {ev}")
            interp = []
            for i in range(ev_df.shape[-1]):
                
                # find plateaus
                tc = ev_df.values[:,i].T
                diff = np.diff(tc)
                gradient = np.sign(diff)
                idx = np.where(gradient != 0)[-1]

                # correct so that you take point at the middle of plateau; append 0 and first original index
                new_idx = ((idx[1:] + idx[:-1]) / 2)
                new_idx = np.append(np.array((0,idx[0]//2)), new_idx).astype(int)

                # pull out new values
                new_vals = ev_df.values[new_idx,i]

                # resample to original shape
                # new_vals = glm.resample_stim_vector(new_vals, tc.shape[0], interpolate="linear")
                uni = new_vals[...,np.newaxis]
                interp.append(uni)

            # concatenate into array and extract time column
            interp = np.concatenate(interp, axis=1)
            new_t = self.time[new_idx]
            # new_t = glm.resample_stim_vector(new_t, tc.shape[0], interpolate="linear")

            # format dataframe in the same way as self.tc_condition
            df = pd.DataFrame(interp)
            df["time"], df["event_type"], df["covariate"] = new_t,ev,utils.get_unique_ids(ev_df, id="covariate")[0]

            # append
            evs_interp.append(df)

        # final indexing
        evs_interp = pd.concat(evs_interp)
        evs_interp.set_index(["event_type","covariate","time"], inplace=True)
        
        return evs_interp

    def format_fitters(self):

        # pass
        self.fitters = self.model._get_response_fitters()
        if not isinstance(self.fitters, pd.DataFrame):
            self.fitters = pd.DataFrame(self.fitters)

        self.fitters = self.check_for_run_index(self.fitters)

    def get_predictions_per_event(self):
        
        # also get the predictions
        if self.fit_type == "ols":
            if not hasattr(self, "fitters"):
                self.fitters = self.model._get_response_fitters()
                self.format_fitters()

        sub_ids = utils.get_unique_ids(self.fitters, id="subject")
        sub_pred = []
        sub_pred_full = []

        for sub in sub_ids:
            
            sub_fitters = utils.select_from_df(self.fitters, expression=f"subject = {sub}")

            # loop through runs
            self.predictions = []
            self.sub_predictions = []
            run_ids = utils.get_unique_ids(sub_fitters, id="run")
            for run in run_ids:

                # full model predictions
                run_fitter = utils.select_from_df(self.fitters, expression=f"run = {run}").iloc[0][0]
                preds = run_fitter.predict_from_design_matrix()
                
                # overwrite colums
                preds.columns = self.tc_condition.columns

                # append
                preds = preds.reset_index().rename(columns={"index": "time"})
                preds["run"] = run
                self.predictions.append(preds)

                # event-specific predictions
                ev_pred = self.get_event_predictions_from_fitter(run_fitter, intercept=self.conf_icpt)
                ev_pred["run"] = run
                
                self.sub_predictions.append(ev_pred)

            # concatenate and index
            self.predictions = pd.concat(self.predictions)
            self.sub_predictions = pd.concat(self.sub_predictions)
            self.predictions["subject"] = self.sub_predictions["subject"] = sub

            sub_pred.append(self.sub_predictions)
            sub_pred_full.append(self.predictions)

        # event-specific events
        self.ev_predictions = pd.concat(sub_pred, ignore_index=True)
        self.ev_predictions.set_index(["subject","event_type","run","t"], inplace=True)
        
        # full model
        self.sub_pred_full = pd.concat(sub_pred_full, ignore_index=True)
        self.sub_pred_full.set_index(["subject","run","time"], inplace=True)

    def timecourses_condition(self):

        if not isinstance(self.covariates, str):
            self.covariates = "intercept"
        
        utils.verbose(f"Fetching subject/condition-wise time courses from {self.model}", self.verbose)

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
        self.tc_sem = self.change_name_set_index(self.tc_sem)
        self.tc_std = self.change_name_set_index(self.tc_std)
        self.tc_mean = self.change_name_set_index(self.tc_mean)
        
        self.grouper = ["event_type","covariate","time"]
        self.sem_condition = self.tc_sem.groupby(level=self.grouper).mean()
        self.std_condition = self.tc_std.groupby(level=self.grouper).mean()
            
        # rename 'event type' to 'event_type' so it's compatible with utils.select_from_df
        self.tc_condition = self.change_name_set_index(self.tc_condition, index=['event_type', 'covariate', 'time'])

        # check run ID
        self.tc_subjects = self.check_for_run_index(self.tc_subjects)
        self.tc_subjects = self.change_name_set_index(self.tc_subjects, index=['subject','run','event_type','covariate','time'])

        if "task" in list(self.tc_subjects.columns):
            self.tc_subjects.drop(["task"], inplace=True, axis=1)

        # get time axis
        self.time = self.tc_condition.groupby(['time']).mean().reset_index()['time'].values

        if self.basis_sets == "fir":

            # interpolate FIR @middle of plateaus
            self.tc_condition_interp = self.interpolate_fir_condition(self.tc_condition)
            self.tc_subjects_interp = self.interpolate_fir_subjects()

        # get r2
        try:
            self.rsq_ = self.model.get_rsq()
        except:
            pass
            
        # get predictions
        if self.fit_type == "ols":

            # get fitters
            self.fitters = self.model._get_response_fitters()
            self.format_fitters()

            # get events
            self.get_predictions_per_event()
        else:
            self.ev_predictions = self.sub_ev_df.copy()

    @classmethod
    def check_for_run_index(self, df, loc=1):

        # reset index
        old_idx = list(df.index.names)

        tmp = df.copy()
        if not "run" in old_idx:
            tmp = df.reset_index()
            old_idx.insert(loc, "run")
            tmp["run"] = 1

            tmp.set_index(old_idx, inplace=True)

        # return
        return tmp
        
    def change_name_set_index(self, df, index=["subject","event_type","covariate","time"]):

        # reset index
        tmp = df.reset_index()

        # remove space in event type column
        if "event type" in list(tmp.columns):
            tmp = tmp.rename(columns={"event type": "event_type"})

        # set index
        if isinstance(index, str):
            index = [index]

        tmp.set_index(index, inplace=True)

        # return
        return tmp
    
    def define_model(self, **kwargs):

        self.model = nd.GroupResponseFitter(
            self.func,
            self.used_onsets,
            input_sample_rate=self.fs,
            confounds=self.confounds, 
            oversample_design_matrix=self.osf,
            add_intercept=self.conf_icpt,
            concatenate_runs=None,
            **kwargs
        )
    
    def add_event(self, *args, **kwargs):
        self.model.add_event(*args, **kwargs)

    def define_events(self):
        
        utils.verbose(f"Selected '{self.basis_sets}'-basis sets (with {self.n_regressors} regressors)", self.verbose)

        # define events
        self.cond = utils.get_unique_ids(self.used_onsets , id="event_type")
        self.run_ids = utils.get_unique_ids(self.used_onsets, id="run")

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
                            self.ev_pred = self.get_event_predictions_from_fitter(self.rf, intercept=self.conf_icpt)
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
        for i in self.cond:
            model.add_event(
                str(i),
                onsets=self.make_onsets_for_response_fitter(i, run=run),
                basis_set=self.basis_sets,
                n_regressors=self.n_regressors,
                interval=self.interval,
                covariates=cov
            )

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
        
        utils.verbose("Fitting completed", self.verbose)

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
        lines_only=False,
        par_kw={},
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

        self.df_pars = HRFMetrics(df, **par_kw).return_metrics()
        self.df_pars[x_lbl] = x_data

        # decide on color based on split
        if not hasattr(self, "color"):
            if not hasattr(self, "cmap") and "cmap" not in list(kwargs.keys()):
                cmap = "viridis"
            else:
                if "cmap" in list(kwargs.keys()):
                    cmap = kwargs["cmap"]
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
        
        if not lines_only:
            self.ttp_plot = plotting.LazyBar(
                data=self.df_pars,
                x=x_lbl,
                y="time_to_peak",
                labels=ttp_labels,
                sns_ori=ttp_ori,
                axs=axs,
                # error=None,
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
        par_kw={},
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

        self.df_pars = HRFMetrics(df, **par_kw).return_metrics()
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

class CVDeconv(InitFitter):

    def __init__(
        self,
        func,
        onsets,
        merge=False,
        TR=0.105):

        self.func = func
        self.onsets = onsets
        self.TR = TR
        self.merge = merge

        # format functional data and onset dataframes
        super().__init__(
            self.func, 
            self.onsets, 
            self.TR,
            merge=self.merge
        )

    def return_combinations(self, split=2):

        self.run_ids = utils.get_unique_ids(self.func, id="run")
        if len(self.run_ids) <= 2:
            raise ValueError(f"Crossvalidating is somewhat pointless with with fewer than 3 runs..")

        # get all possible combination of runs
        return utils.unique_combinations(self.run_ids,l=split)

    def single_fitter(self, func, onsets, **kwargs):
        
        if not "fit" in list(kwargs.keys()):
            kwargs["fit"] = True

        # run single fitter and fetch timecourses
        fit_obj = NideconvFitter(
            func, 
            onsets,
            TR=self.TR,
            **kwargs
        )

        if kwargs["fit"]:
            fit_obj.timecourses_condition()

        return fit_obj

    # def out_of_set_prediction(self):
    def predict_out_of_set(self, src, trg):
        
        if isinstance(src, list):
            self.betas = pd.DataFrame(pd.concat([src.fitters.iloc[i].betas for i in range(src.fitters.shape[0])], axis=1).mean(axis=1))
            ref_obj = src[0]
        else:
            ref_obj = src
            self.betas = src.fitters.iloc[0].betas

        if isinstance(trg, (float,int,str)):
            trg = [trg]
            
        cv_r2 = []
        sub_df = []
        sub_ids = utils.get_unique_ids(ref_obj.func, id="subject")
        for sub in sub_ids:

            sub_fitters = utils.select_from_df(src.fitters, expression=f"subject = {sub}")

            if isinstance(src, list):
                self.betas = pd.DataFrame(pd.concat([sub_fitters.iloc[i][0].betas for i in range(sub_fitters.shape[0])], axis=1).mean(axis=1))
            else:
                self.betas = sub_fitters.iloc[0][0].betas

            run_r2 = []
            for run in trg:
                
                # find design matrix of unfitted run
                expr = (f"subject = {sub}","&",f"run = {run}")
                X = utils.select_from_df(self.full_fitters, expression=expr).iloc[0][0].X
                prediction = X.dot(self.betas)

                # get r2
                run_f = utils.select_from_df(self.full_model.func, expression=expr)

                r2 = []
                for i in range(run_f.shape[-1]):
                    r2.append(metrics.r2_score(run_f.values[:,i],prediction.values[:,i]))

                run_r2.append(np.array(r2))

            # average over unfitted runs
            run_r2 = np.array(run_r2).mean(axis=0)
            sub_df.append(np.array(run_r2))
            
        sub_r2 = np.array(sub_df)
        cv_r2 = pd.DataFrame(sub_r2, columns=list(self.func.columns))
        cv_r2["subject"] = sub_ids

        return cv_r2

    def crossvalidate(self, split=2, **kwargs):
        
        # for later reference of design matrices
        self.full_model = self.single_fitter(
            self.func,
            self.onsets,
            fit=False,
            **kwargs # all other parameters for NideconvFitter
        )

        # get fitters
        self.full_fitters = self.full_model.model._get_response_fitters()
        
        self.combos = self.return_combinations(split=split)

        # loop through them
        self.fitters = []
        self.r2_ = []
        for ix,combo in enumerate(self.combos):
            unfitted_runs = [i for i in self.run_ids if i not in combo]

            self.fold_f = pd.concat([utils.select_from_df(self.func, expression=f"run = {i}") for i in combo])
            self.fold_o = pd.concat([utils.select_from_df(self.onsets, expression=f"run = {i}") for i in combo])
            
            fit_ = self.single_fitter(
                self.fold_f,
                self.fold_o,
                **kwargs # all other parameters for NideconvFitter
            )

            self.fitters.append(fit_)

            # predict non-fitted run(s)
            df = self.predict_out_of_set(fit_, unfitted_runs)
            df["fold"] = ix
            self.r2_.append(df)

        self.r2_ = pd.concat(self.r2_)
        self.r2_.set_index(["subject","fold"], inplace=True)

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
        start = (ii.t0_-xlim[0])/tot
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
        force_neg=False,
        plot=False,
        nan_policy=True,
        debug=False,
        progress=False,
        thr_var=0,
        shift=0
        ):

        self.hrf = hrf
        self.TR = TR
        self.force_pos = force_pos
        self.force_neg = force_neg
        self.plot = plot
        self.shift = shift
        self.nan_policy = nan_policy
        self.debug = debug
        self.thr_var = thr_var
        self.progress = progress

        # get metrics
        self._get_metrics(
            TR=self.TR, 
            force_pos=self.force_pos,
            force_neg=self.force_neg,
            nan_policy=self.nan_policy
        )

    def return_metrics(self):
        return self.metrics.reset_index(drop=True)
    
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
        force_neg=False,
        nan_policy=False):

        # set shift to None because this doesn't deal with 2D arrays; just get the format right
        hrf = self._verify_input(self.hrf, TR=TR, shift=None)

        orig_cols = list(hrf.columns)

        # find columns abs(var)>0
        filtered_variance = (hrf.var(axis=0)>self.thr_var).values
        filtered_df = hrf.iloc[:,filtered_variance]
        cols = list(filtered_df.columns)
        utils.verbose(f" {filtered_df.shape[1]}/{hrf.shape[1]}\tvertices survived variance threshold of {self.thr_var}", self.debug)
        if len(cols) == 0:
            raise ValueError(f"Variance threshold of {self.thr_var} is too strict; no vertices survived")
        
        if not isinstance(force_neg, list):
            force_neg = [force_neg for _ in cols]

        if not isinstance(force_pos, list):
            force_pos = [force_pos for _ in cols]

        # initiate output
        incl = [
            "magnitude",
            "magnitude_ix",
            "fwhm",
            "fwhm_obj",
            "time_to_peak",
            "half_rise_time",
            "half_max",
            "rise_slope",
            "rise_slope_t",
            "positive_area",
            "undershoot"
        ]

        # print(force_pos)
        self.col_metrics = []
        self.col_fwhm = []

        # initialize empty dataframe
        self.metrics = pd.DataFrame(np.zeros((len(orig_cols),len(incl))), columns=incl)

        # separate loop for fancy progress bar
        if self.progress:
            with alive_bar(filtered_df.shape[1], force_tty=True) as bar:
                for ix,col in enumerate(filtered_df):
                    pars,fwhm_ = self._get_single_hrf_metrics(
                        filtered_df[col],
                        TR=self.TR,
                        force_pos=force_pos[ix],
                        force_neg=force_neg[ix],
                        plot=self.plot,
                        nan_policy=nan_policy,
                        shift=self.shift
                    )

                    # progress
                    bar()
                
                    self.col_metrics.append(pars)
                    self.col_fwhm.append(fwhm_)
        else:
            for ix,col in enumerate(filtered_df):
                pars,fwhm_ = self._get_single_hrf_metrics(
                    filtered_df[col],
                    TR=self.TR,
                    force_pos=force_pos[ix],
                    force_neg=force_neg[ix],
                    plot=self.plot,
                    nan_policy=nan_policy,
                    shift=self.shift
                )
            
                self.col_metrics.append(pars)
                self.col_fwhm.append(fwhm_)

        if len(self.col_metrics) > 0:
            self.col_metrics = pd.concat(self.col_metrics)
        
        # insert into empty dataframe
        self.metrics.iloc[filtered_variance,:] = self.col_metrics.values.copy()

        if len(orig_cols)>1:
            self.metrics["vox"] = np.arange(0,len(orig_cols))

        # set to integer
        if "magnitude_ix" in list(self.metrics.columns):
            self.metrics["magnitude_ix"] = self.metrics["magnitude_ix"].astype(int)

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
    def _verify_input(self, hrf, TR=None, shift=None):
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

        # shift to zero
        if isinstance(shift, int):
            if shift > 1:
                bsl = hrf.values[:shift].mean()
            else:
                bsl = hrf.values[0]

            if bsl>0:
                hrf -= bsl
            elif bsl<0:
                hrf += abs(bsl)

        return hrf

    @classmethod
    def plot_profile_for_debugging(
        self, 
        hrf, 
        axs=None,
        figsize=(5,5),
        **kwargs):

        if not isinstance(axs, mpl.axes._axes.Axes):
            _,axs = plt.subplots(figsize=figsize)

        time = self._get_time(hrf)
        plotting.LazyPlot(
            hrf.values.squeeze(),
            xx=list(time),
            x_label="time (s)",
            y_label="magnitude",
            axs=axs,
            color="r",
            line_width=2,
            **kwargs
        )

    @classmethod
    def _get_riseslope(
        self, 
        hrf, 
        force_pos=False, 
        force_neg=False,
        nan_policy=False):

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
        try:
            if not force_pos:
                if negative:
                    val = np.array([np.amin(diff)])
                else:
                    val = np.array([np.amax(diff)])
            else:
                val = np.array([np.amax(diff)])
            
            final_val = val[0]
            val_ix = utils.find_nearest(diff,final_val)[0]
            val_t = time[val_ix]

        except Exception as e:
            if not nan_policy:
                self.plot_profile_for_debugging(hrf)
                raise ValueError(f"Could not extract rise-slope from this profile: {e}")
            else:
                val_t = final_val = np.nan 
        
        return final_val,val_t

    @classmethod
    def _get_amplitude(self, hrf, force_pos=False, force_neg=False):

        # fetch time stamps
        time = self._get_time(hrf)

        # check negative:
        negative = self._check_negative(
            hrf, 
            force_neg=force_neg,
            force_pos=force_pos
        )

        if not force_pos:
            if negative:
                mag_tmp = hrf.min(axis=0).values
            else:
                mag_tmp = hrf.max(axis=0).values
        else:
            mag_tmp = hrf.max(axis=0).values

        mag_ix = utils.find_nearest(hrf.values.squeeze(), mag_tmp)[0]

        return {
            "amplitude": mag_tmp[0],
            "t": time[mag_ix],
            "t_ix": mag_ix
        }
    
    @classmethod
    def _get_auc(
        self, 
        hrf, 
        force_pos=False, 
        force_neg=False,
        nan_policy=False):

        # get time 
        time = self._get_time(hrf)
        dx = np.diff(time)[0]

        # get amplitude of HRF
        mag = self._get_amplitude(
            hrf, 
            force_pos=force_pos,
            force_neg=force_neg
        )

        # first check if the period before the peak has zero crossings
        zeros = np.zeros_like(hrf)[:mag["t_ix"]]
        xx = np.arange(0, zeros.shape[0])

        try:
            coords = utils.find_intersection(
                xx,
                zeros,
                hrf.values[:mag["t_ix"]]
            )

            HAS_ZEROS_BEFORE_PEAK = True
        except:
            HAS_ZEROS_BEFORE_PEAK = False
        
        if HAS_ZEROS_BEFORE_PEAK:
            # if we found multiple coordinates, we have some bump/dip before peak
            coords = [int(i[0][0]) for i in coords]
            coords.sort()

            if len(coords) > 1:
                first_cross = coords[-1]
            else:
                # assume single zero point before peak
                first_cross = coords[0]
            
        else:
            # if no zeros before peak, take start of curve as starting curve
            first_cross = 0

        # print(f"1st crossing = {first_cross}")
        # find the next zero crossing after "first_cross" that is after peak index
        zeros = np.zeros_like(hrf)
        xx = np.arange(0, zeros.shape[0])        

        try:
            coords = utils.find_intersection(
                xx,
                zeros,
                hrf.values.squeeze()
            )

            FAILED = False
        except:
            FAILED = True

        # if there's no zero-crossings, take end of interval
        third_cross = None
        if FAILED:
            second_cross = hrf.shape[0]-1
        else:

            # filter coordinates for elements BEFORE peak
            coord_list = [int(i[0][0]) for i in coords if int(i[0][0])>mag["t_ix"]]
            if len(coord_list)>0:

                # sort
                coord_list.sort()
                # index first element as second zero 
                second_cross = coord_list[0]

                # take last element as undershoot
                if len(coord_list)<2:
                    third_cross = hrf.shape[0]-1
            else:
                second_cross = hrf.shape[0]-1

            # if multple coords after peak we might have negative undershoot
            if len(coord_list)>1:
                # check if sample are not too close to one another;AUC needs at least 2 samples
                ix = 1
                third_cross = coord_list[ix]
                # print(coord_list)
                # print(third_cross)
                while (third_cross-second_cross)<2:
                    ix += 1
                    if ix<len(coord_list):
                        third_cross = coord_list[ix]
                    else:
                        third_cross = hrf.shape[0]-1

        # print(f"2nd crossing = {second_cross}")
        # print(f"3rd crossing = {third_cross}")
        # connect A and B (bulk of response)
        zeros = np.zeros_like(hrf.iloc[first_cross:second_cross])
        xx = np.arange(0, zeros.shape[0])

        try:
            ab_area = metrics.auc(xx, hrf.iloc[first_cross:second_cross])*dx
        except:
            if not nan_policy:
                self.plot_profile_for_debugging(hrf)
                raise ValueError()
            else:
                ab_area = np.nan
        
        # check for under-/overshoot
        ac_area = np.nan
        if isinstance(third_cross, int):
            # print(f"1st cross: {first_cross}\t2nd cross: {second_cross}\t3rd cross: {third_cross}")
            zeros = np.zeros_like(hrf.iloc[second_cross:third_cross])
            xx = np.arange(0, zeros.shape[0])
            try:
                ac_area = abs(metrics.auc(xx, hrf.iloc[second_cross:third_cross]))*dx
            except:
                if not nan_policy:
                    self.plot_profile_for_debugging(hrf, add_hline=0)
                    raise ValueError()
                else:
                    ac_area = np.nan
            
        return {
            "pos_area": ab_area,
            "undershoot": ac_area
        }


    @classmethod    
    def _get_fwhm(
        self,
        hrf,
        force_pos=False, 
        force_neg=False,
        nan_policy=False,
        add_fct=0.5):

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
        end_ix += int(end_ix*add_fct)

        try:
            # get fwhm around max amplitude
            fwhm_val = FWHM(
                time, 
                hrf.values, 
                negative=negative)
            
            fwhm_dict = {
                "fwhm": fwhm_val.fwhm,
                "half_rise": fwhm_val.t0_,
                "half_max": fwhm_val.half_max,
                "obj": fwhm_val
            }
        except Exception as e:
            if not nan_policy:
                self.plot_profile_for_debugging(hrf)
                raise ValueError(f"Could not extract FWHM from this profile: {e}")
            else:
                fwhm_dict = {
                    "fwhm": np.nan,
                    "half_rise": np.nan,
                    "half_max": np.nan,
                    "obj": np.nan
                }
        
        return fwhm_dict
    
    @classmethod
    def _get_single_hrf_metrics(
        self, 
        hrf, 
        TR=None, 
        force_pos=False,
        force_neg=False,
        plot=False,
        nan_policy=False,
        debug=False,
        shift=None,
        **kwargs):

        # verify input type
        hrf = self._verify_input(hrf, TR=TR, shift=shift)
        utils.verbose(f"var = {abs(hrf.var().iloc[0])}", debug)

        if plot:
            self.plot_profile_for_debugging(hrf, add_hline=0)

        # get magnitude and amplitude
        mag = self._get_amplitude(
            hrf, 
            force_pos=force_pos, 
            force_neg=force_neg)

        # fwhm
        fwhm_obj = self._get_fwhm(
            hrf, 
            force_pos=force_pos, 
            force_neg=force_neg,
            nan_policy=nan_policy)

        # rise slope
        rise_tmp, rise_t = self._get_riseslope(
            hrf, 
            force_pos=force_pos, 
            force_neg=force_neg,
            nan_policy=nan_policy)

        # positive area + undershoot
        auc = self._get_auc(
            hrf,
            force_pos=force_pos,
            force_neg=force_neg,
            nan_policy=nan_policy
        )

        ddict =  {
            "magnitude": [mag["amplitude"]], 
            "magnitude_ix": [mag["t_ix"]], 
            "fwhm": [fwhm_obj["fwhm"]], 
            "fwhm_obj": [fwhm_obj["obj"]], 
            "time_to_peak": [mag["t"]],
            "half_rise_time": [fwhm_obj["half_rise"]],
            "half_max": [fwhm_obj["half_max"]],
            "rise_slope": [rise_tmp],
            "rise_slope_t": [rise_t],
            "positive_area": [auc["pos_area"]],
            "undershoot": [auc["undershoot"]]
        }

        df = pd.DataFrame(ddict)

        return df,fwhm_obj

class FWHM():

    def __init__(
        self, 
        x, 
        hrf, 
        negative=False,
        y_tol=0.01,
        resample=500,
        ):
        
        self.x = x
        self.hrf = hrf
        self.y_tol = y_tol
        self.resample = resample
        self.negative = negative
        
        if x.shape[0]<self.resample:
            
            # print(f"resample vectors of shape {x.shape} to {self.resample}")
            self.use_x = glm.resample_stim_vector(self.x,self.resample, interpolate="linear")
            self.use_rf = glm.resample_stim_vector(self.hrf,self.resample, interpolate="linear")

        else:
            self.use_x = x.copy()
            self.use_rf = hrf.copy()

        self.hmx = self.half_max_x(self.use_x,self.use_rf)

        if len(self.hmx) < 2:
            raise ValueError(f"Could not find two points close to half max..")
        
        self.ts = [i.squeeze()[0] for i in self.hmx]
        self.ts.sort()

        self.t0_ = self.ts[0]
        self.t1_ = self.ts[1]
        self.fwhm = abs(self.t1_-self.t0_)
        
        if self.negative:
            self.half_max = min(self.use_rf)/2
        else:
            self.half_max = max(self.use_rf)/2

        self.half_max = self.half_max[0]

    # def lin_interp(self, x, y, i, half):
    #     return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

    def half_max_x(self, x, y):
        
        if not self.negative:
            half = max(y)/2.0
        else:
            half = min(y)/2.0

        return utils.find_intersection(x,y,np.full_like(y,half))

        # signs = np.sign(np.add(y, -half))
        # zero_crossings = (signs[0:-2] != signs[1:-1])
        # zero_crossings_i = np.where(zero_crossings)[0]
        # return [self.lin_interp(x, y, zero_crossings_i[0], half),
        #         self.lin_interp(x, y, zero_crossings_i[1], half)]

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

    res = {}
    for lbl,np in zip(["pos","neg"],[False,True]):
        res[lbl] = {}
        args["negative"] = np
        output = minimize(
            error_function, 
            starting_params, 
            args=(
                args, 
                data,
                make_prediction),
            method='trust-constr',
            bounds=bounds,
            tol=ftol,
            options=dict(xtol=xtol)
        )   

        pred = make_prediction(
            output['x'],
            **args
        )

        res[lbl]["pars"] = output['x']
        res[lbl]["r2"] = metrics.r2_score(data,pred)  

    r2_pos = res["pos"]["r2"]
    r2_neg = res["neg"]["r2"]
    if r2_pos>r2_neg:
        return res["pos"]["pars"],"pos"
    else:
        return res["neg"]["pars"],"neg"
    
    # return starting_params, "pos"
    
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
        interval=[0,30],
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

        # set number of jobs if None is specified (default = 1)
        if not isinstance(self.n_jobs, int):
            self.n_jobs = self.data.shape[0]
        
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
        
    def profiles_from_parameters(
        self, 
        resample_to_shape=None, 
        negative=None, 
        read_index=False):

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
    

class Epoch(InitFitter):

    def __init__(
        self, 
        func, 
        onsets, 
        TR=0.105, 
        interval=[-2,14],
        merge=False
        ):
        
        self.func = func
        self.onsets = onsets
        self.TR = TR
        self.interval = interval
        self.merge = merge

        # prepare data
        super().__init__(
            self.func, 
            self.onsets, 
            self.TR,
            merge=self.merge
        )

        # epoch data based on present identifiers (e.g., task/run)
        self.epoch_input()

    @staticmethod
    def correct_baseline(
        d_, 
        bsl=20, 
        verbose=False
        ):

        if isinstance(d_, pd.DataFrame):
            d_vals = d_.values
            return_df = True
        else:
            d_vals = d_.copy()
            return_df = False

        if d_vals.ndim<2:
            d_vals = d_vals[...,np.newaxis]
        
        shift_cols = []
        for col in range(d_vals.shape[-1]):
            
            col_vals = d_vals[:,col]
            m_ = col_vals[:bsl].mean()
            if m_ < 0:
                utils.verbose(f"Shifting profile UP with {round(abs(m_),2)}", verbose)
                d_shift = col_vals+abs(m_)
            else:
                utils.verbose(f"Shifting profile DOWN with {round(m_,2)}", verbose)
                d_shift = col_vals-m_

            if d_shift.ndim<2:
                d_shift = d_shift[...,np.newaxis]

            shift_cols.append(d_shift)

        shift_cols = np.concatenate(shift_cols, axis=1)
        if return_df:
            return pd.DataFrame(shift_cols, index=d_.index)
        else:
            return d_shift

    @staticmethod
    def _get_epoch_timepoints(interval,TR):
        total_length = interval[1] - interval[0]
        timepoints = np.linspace(
            interval[0],
            interval[1],
            int(total_length * (1/TR) * 1),
            endpoint=False)
        
        return timepoints

    def epoch_events(
        self,
        df_func,
        df_onsets, 
        index=False,
        ):

        ev_epochs = []

        info = {}
        for el in ["event_type","subject","run"]:
            info[el] = utils.get_unique_ids(df_onsets, id=el)

        df_idx = ["subject","run","event_type","epoch","t"]
        for ev in info["event_type"]:

            epochs = []
            ev_onsets = utils.select_from_df(df_onsets, expression=f"event_type = {ev}")
            for i,t in enumerate(ev_onsets.onset.values):
                
                if ev not in ["response","blink"]:
                    # find closest ID to interval
                    t_start = t-abs(self.interval[0])
                    ix_start = int(t_start/self.TR)

                    samples = self.interval[1]+abs(self.interval[0])
                    n_sampl = int(samples/self.TR)
                    ix_end = ix_start+n_sampl

                    # extract from data
                    stim_epoch = df_func.iloc[ix_start:ix_end,:].values #[...,np.newaxis]       
                    time = self._get_epoch_timepoints(self.interval, self.TR)

                    df_stim_epoch = pd.DataFrame(stim_epoch)

                    if len(time) != len(stim_epoch):
                        print(f"WARNING: could not extract full epoch around event; onset={round(t,2)}s, t_start={round(t_start,2)}s with {n_sampl} samples ({samples}s). Epoch has {len(stim_epoch)} samples")

                    df_stim_epoch["t"],df_stim_epoch["epoch"],df_stim_epoch["event_type"] = time[:len(stim_epoch)],i,ev
                    epochs.append(df_stim_epoch)
            
            # concatenate into 3d array (time,ev,stimulus_nr), average, and store in dataframe
            ev_epoch = pd.concat(epochs)

            # format dataframe
            ev_epoch["subject"], ev_epoch["run"] = info["subject"][0],info["run"][0]
            
            ev_epochs.append(ev_epoch)

        # concatenate into dataframe
        df_epochs = pd.concat(ev_epochs)

        # set index or not
        if index:
            df_epochs.set_index(df_idx, inplace=True)

        return df_epochs

    def epoch_runs(
        self,
        df_func,
        df_onsets, 
        index=False,
        ):

        # loop through runs
        self.run_ids = utils.get_unique_ids(df_func, id="run")
        # print(f"task-{task}\t| runs = {run_ids}")
        run_df = []
        for run in self.run_ids:
            
            expr = f"run = {run}"
            run_func = utils.select_from_df(df_func, expression=expr)
            run_stims = utils.select_from_df(df_onsets, expression=expr)

            # get epochs
            df = self.epoch_events(
                run_func,
                run_stims,
                index=index
            )

            run_df.append(df)

        run_df = pd.concat(run_df)

        return run_df
    
    def epoch_tasks(
        self,
        df_func,
        df_onsets
        ):

        # read task IDs
        self.task_ids = utils.get_unique_ids(df_func, id="task")

        # loop through task IDs
        task_df = [] 
        for task in self.task_ids:

            # extract task-specific dataframes
            expr = f"task = {task}"
            task_func = utils.select_from_df(df_func, expression=expr)
            task_stims = utils.select_from_df(df_onsets, expression=expr)

            df = self.epoch_runs(
                task_func,
                task_stims,
                index=False
            )

            df["task"] = task
            task_df.append(df)

        return pd.concat(task_df)
    
    def epoch_subjects(
        self,
        df_func,
        df_onsets
        ):

        self.sub_ids = utils.get_unique_ids(self.func, id="subject")
        
        # loop through subject IDs
        sub_df = [] 
        for sub in self.sub_ids:

            # extract task-specific dataframes
            expr = f"subject = {sub}"
            self.sub_func = utils.select_from_df(df_func, expression=expr)
            self.sub_stims = utils.select_from_df(df_onsets, expression=expr)

            try:
                self.task_ids = utils.get_unique_ids(self.sub_func, id="task")
            except:
                self.task_ids = None

            if isinstance(self.task_ids, list):
                sub_epoch = self.epoch_tasks(self.sub_func, self.sub_stims)
                self.idx = ["subject","task","run","event_type","epoch","t"]
            else:
                sub_epoch = self.epoch_runs(self.sub_func, self.sub_stims)
                self.idx = ["subject","run","event_type","epoch","t"]

            sub_epoch.set_index(self.idx, inplace=True)        
            sub_df.append(sub_epoch)

        sub_df = pd.concat(sub_df)

        return sub_df
    
    def epoch_input(self):

        # run Epoching
        self.df_epoch = self.epoch_subjects(
            self.func,
            self.onsets
        )

        # copy column names
        self.df_epoch.columns = self.func.columns

class GLM(InitFitter):

    def __init__(
        self, 
        func, 
        onsets, 
        TR=0.105, 
        merge=False,
        **kwargs
        ):
        
        self.func = func
        self.onsets = onsets
        self.TR = TR
        self.merge = merge

        # prepare data
        super().__init__(
            self.func, 
            self.onsets, 
            self.TR,
            merge=self.merge
        )

        # epoch data based on present identifiers (e.g., task/run)
        self.glm_input(**kwargs)

    @classmethod
    def get_copes(
        self,
        evs,
        derivative=False,
        dispersion=False,
        add_intercept=True,
        **kwargs
        ):

        copes = []
        if not dispersion and not derivative:
            copes = np.eye(len(evs))
        else:
            start = 0
            for i in enumerate(evs):
                if dispersion and derivative:
                    fc = 3
                elif dispersion or derivative:
                    fc = 2
                else:
                    fc = 1

                n_cols = len(evs)*fc
                arr = np.zeros((n_cols))
                arr[start:start+fc] = 1
                copes.append(arr)

                start += fc
            
        copes = np.array(copes)
        if add_intercept:
            icpt = np.zeros((len(evs),1))
            copes = np.hstack((icpt,copes))

        return copes

    @classmethod
    def single_glm(
        self,
        func,
        onsets,
        **kwargs
        ):

        defaults = {
            "add_intercept": True,
            "hrf_pars": "glover",
            "derivative": True,
            "TR": 0.105,
            "osf": 100
        }

        # update kwargs with defaults
        for key,val in defaults.items():
            kwargs = utils.update_kwargs(
                kwargs,
                key,
                val
            )

        # decide contrasts
        evs = utils.get_unique_ids(onsets, id="event_type")
        c_vec = self.get_copes(
            evs,
            **kwargs
        )

        fitter = glm.GenericGLM(
            onsets, 
            func,
            **kwargs
        )

        fitter.create_design()
        fitter.fit(copes=c_vec)

        fm = self.format_output(fitter)
        return fm
    
    @classmethod
    def format_fm(self, glm_obj):

        # full model predictions = dot product of design matrix & betas
        pred_fm = glm_obj.results["x_conv"]@glm_obj.results["betas"]

        # input will be dataframe, per definition, so copy indices
        return pd.DataFrame(pred_fm, index=glm_obj.orig.index)    

    @classmethod
    def format_ev(self, glm_obj):

        evs = utils.get_unique_ids(glm_obj.onsets, id="event_type")
        dm_columns = glm_obj.results["dm"].columns.to_list()
        preds = []

        for ix,ev in enumerate(evs):

            beta_idx = [ix for ix,ii in enumerate(dm_columns) if ev in ii or "regressor" in ii or "intercept" in ii]

            # get betas for all voxels
            betas = glm_obj.results["betas"][beta_idx]
            event = glm_obj.results["x_conv"][:,beta_idx]
            ev_preds = event@betas  

            df_ev = pd.DataFrame(ev_preds, index=glm_obj.orig.index)
            idx = list(df_ev.index.names)
            df_ev["event_type"] = ev
            df_ev.reset_index(inplace=True)
            idx.insert(-1, "event_type")
            df_ev.set_index(idx, inplace=True)
            preds.append(df_ev)

        # get predictions
        return pd.concat(preds)

    @classmethod
    def format_r2(self, glm_obj):
        idx = {}
        for i in ["subject","task","run"]:
            try:
                IDs = utils.get_unique_ids(glm_obj.orig, id=i)
            except:
                IDs = []

            if len(IDs)>0:
                idx[i] = IDs[0]

        df_r2 = pd.DataFrame(glm_obj.results["r2"][np.newaxis,...])
        for key,val in idx.items():
            df_r2[key] = val

        df_r2.set_index(list(idx.keys()), inplace=True)
        return df_r2
    
    @classmethod
    def format_tstats(self, glm_obj):
        idx = {}
        for i in ["subject","task","run"]:
            try:
                IDs = utils.get_unique_ids(glm_obj.orig, id=i)
            except:
                IDs = []

            if len(IDs)>0:
                idx[i] = IDs[0]

        evs = utils.get_unique_ids(glm_obj.onsets, id="event_type")
        df_t = pd.DataFrame(glm_obj.results["tstats"])
        for key,val in idx.items():
            df_t[key] = val

        df_t["event_type"] = evs
        ix_list = list(idx.keys())
        ix_list.insert(-1, "event_type")
        df_t.set_index(ix_list, inplace=True)
        return df_t    

    @classmethod
    def format_obj(self, glm_obj):
        idx = {}
        for i in ["subject","task","run"]:
            try:
                IDs = utils.get_unique_ids(glm_obj.orig, id=i)
            except:
                IDs = []

            if len(IDs)>0:
                idx[i] = IDs[0]

        evs = utils.get_unique_ids(glm_obj.onsets, id="event_type")
        df_t = pd.DataFrame(glm_obj.results["tstats"])
        for key,val in idx.items():
            df_t[key] = val

        df_t["event_type"] = evs
        ix_list = list(idx.keys())
        ix_list.insert(-1, "event_type")
        df_t.set_index(ix_list, inplace=True)
        return df_t 
    
    @classmethod
    def format_output(self, glm_obj):
        
        # full model
        df_fm = self.format_fm(glm_obj)

        # sort out ev-predictions
        df_em = self.format_ev(glm_obj)
        
        # format r2
        df_r2 = self.format_r2(glm_obj)

        # format r2
        df_tstats = self.format_tstats(glm_obj)

        ddict = {
            "obj": glm_obj,
            "full_model": df_fm,
            "ev_model": df_em,
            "r2": df_r2,
            "tstats": df_tstats
        }

        return ddict

    @classmethod
    def glm_runs(
        self,
        df_func,
        df_onsets, 
        **kwargs
        ):

        # loop through runs
        self.run_ids = utils.get_unique_ids(df_func, id="run")
        # print(f"task-{task}\t| runs = {run_ids}")
        run_df = []
        for run in self.run_ids:
            
            expr = f"run = {run}"
            run_func = utils.select_from_df(df_func, expression=expr)
            run_stims = utils.select_from_df(df_onsets, expression=expr)

            # get glms
            df = self.single_glm(
                run_func,
                run_stims,
                **kwargs
            )

            run_df.append(df)

        return run_df
    
    @classmethod
    def glm_tasks(
        self,
        df_func,
        df_onsets,
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
            task_stims = utils.select_from_df(df_onsets, expression=expr)

            df = self.glm_runs(
                task_func,
                task_stims,
                **kwargs
            )

            task_df += df
        
        return task_df

    @classmethod
    def glm_subjects(
        self,
        df_func,
        df_onsets,
        **kwargs
        ):

        self.sub_ids = utils.get_unique_ids(df_func, id="subject")
        
        # loop through subject IDs
        sub_df = [] 
        for sub in self.sub_ids:

            # extract task-specific dataframes
            expr = f"subject = {sub}"
            self.sub_func = utils.select_from_df(df_func, expression=expr)
            self.sub_stims = utils.select_from_df(df_onsets, expression=expr)

            try:
                self.task_ids = utils.get_unique_ids(self.sub_func, id="task")
            except:
                self.task_ids = None

            if isinstance(self.task_ids, list):
                sub_glm = self.glm_tasks(
                    self.sub_func, 
                    self.sub_stims,
                    **kwargs
                )
                self.idx = ["subject","task","run"]
            else:
                sub_glm = self.glm_runs(
                    self.sub_func, 
                    self.sub_stims,
                    **kwargs
                )
                self.idx = ["subject","run"]

            # sub_glm.set_index(self.idx, inplace=True)        
            sub_df += sub_glm

        # sub_df = pd.concat(sub_df)
        return sub_df
    
    def glm_input(self, **kwargs):

        # run glm
        self.df_glm = self.glm_subjects(
            self.func,
            self.onsets,
            **kwargs
        )

        concat_elements = [
            "full_model",
            "ev_model",
            "r2",
            "tstats"
        ]

        self.glm_output = {}
        for conc in concat_elements:
            self.glm_output[conc] = []
            for i in self.df_glm:
                self.glm_output[conc].append(i[conc])

            if len(self.glm_output[conc])>0:
                self.glm_output[conc] = pd.concat(self.glm_output[conc])

    def get_result(self):
        return self.glm_output
    
    def find_max_r2(self, df=None):
        if not isinstance(df, pd.DataFrame):
            df = self.glm_output["r2"]

        max_r2 = np.amax(df.values)
        max_idx = np.where(df.values==max_r2)[-1][0]
        return max_r2,max_idx
    
    def find_r2(self, vox_nr, df=None):

        if not isinstance(df, pd.DataFrame):
            df = self.glm_output["r2"]

        r2 = df.iloc[:,vox_nr].values
        if len(r2)>0:
            r2 = r2.mean()
        else:
            r2 = r2[0]

        return r2
    
    def plot_ev_predictions(
        self,
        data=None,
        subject=None,
        task=None,
        run=None,
        vox_nr="max",
        cmap="inferno",
        axs=None,
        figsize=(14,4),
        full=False,
        full_only=False,
        full_color="k",
        r2_dec=4,
        **kwargs
        ):

        # get data
        preds = self.glm_output["ev_model"]
        r2 = self.glm_output["r2"]
        full_preds = self.glm_output["full_model"]
        evs = utils.get_unique_ids(preds, id="event_type")
        if not isinstance(data, pd.DataFrame):
            data = self.func.copy()

        # apply some filters
        expr = []
        if isinstance(subject, str):
            expr.append(f"subject = {subject}")
        
        if isinstance(task, str):
            expr.append(f"task = {task}")

        if isinstance(run, (int,str)):
            expr.append(f"run = {run}")

        if len(expr)>0:
            preds = utils.multiselect_from_df(preds, expression=expr)
            r2 = utils.multiselect_from_df(r2, expression=expr)
            data = utils.multiselect_from_df(data, expression=expr)
            full_preds = utils.multiselect_from_df(full_preds, expression=expr)

        if not isinstance(axs, mpl.axes._axes.Axes):
            fig,axs = plt.subplots(figsize=figsize)

        # set defaults for actual datapoints
        markers = ['.']
        colors = ["#cccccc"]
        linewidth = [0.5]
        if not isinstance(vox_nr, int):
            r2_val,vox_nr = self.find_max_r2(df=r2)
        else:
            r2_val = self.find_r2(vox_nr, df=r2)
    
        signals = [data.iloc[:,vox_nr].values]
        labels = ['input signal']
        if not full_only:

            for ev in evs:

                # get predictions
                ev_preds = utils.select_from_df(
                    preds,
                    expression=f"event_type = {ev}"
                )

                signals.append(ev_preds.iloc[:,vox_nr].values)
                labels.append(f"Event '{ev}'")
                markers.append(None)
                linewidth.append(3)

            colors = [*colors, *sns.color_palette(cmap, len(evs))]
        else:
            full = True

        # append full model
        if full:
            signals.append(full_preds.iloc[:,vox_nr].values)
            labels.append(f"full model")
            markers.append(None)
            linewidth.append(2)
            colors.append(full_color)

        if not "title" in list(kwargs.keys()):
            kwargs = utils.update_kwargs(
                kwargs,
                "title",
                f"model fit vox {vox_nr+1}/{data.shape[1]} (r2={round(r2_val,r2_dec)})",
            )

        pl = plotting.LazyPlot(
            signals,
            y_label="Activity (a.u.)",
            x_label="volumes",
            labels=labels,
            axs=axs,
            markers=markers,
            color=colors,
            line_width=linewidth,
            **kwargs
        )
