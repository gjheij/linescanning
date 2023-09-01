import numpy as np
from linescanning import (
    utils,
    plotting,
    prf
)
import pandas as pd
from typing import Union

class ITI():
    
    """ITI

    Class to generate a set of inter-stimulus-intervals (ITIs) in an iterative manner that follows a negative exponential. Such a design is optimal for fMRI experiments aimed to deconvolve responses of different events

    Parameters
    ----------
    tmin: Union[int,float], optional
        Minimal ITI duration, by default 3
    tmax: Union[int,float], optional
        Maximal ITI duration, by default 18
    tmean: Union[int,float], optional
        Mean ITI duration, by default 6
    n_trials: int, optional
        Number of trials in experiment, by default None
    leeway: Union[int,float], optional
        Allow some leeway, by default 0
    stim_duration: Union[int,float], optional
        Stimulus duration, by default 1. Used to build the onset dataframe
    total_duration: Union[int,float], optional
        Max duration of the experiment, by default None. If a value is specified, the ITIs will be iteratively generated so that the total time does not exceed this value
    start_duration: Union[int,float], optional
        Baseline at the start of the experiment, by default 0
    end_duration: Union[int,float], optional
        Baseline at the end of the experiment, by default 0
    TR: Union[int,float], optional
        Repetition time, by default 0.105. Used for indexing of the onset dataframe
    verbose: bool, optional
        Turn on verbose, by default False
    events: Union[str,list], optional
        Specify event names, by default 'stim'. Used for generation of the dataframe. If the experiment involves 1 event, the event name is "stim", otherwise these names will be randomly assigned to the ITIs.

    Example
    ----------
    >>> iti_obj = simulate.ITI(
    >>>     tmin=3,
    >>>     tmax=18,
    >>>     tmean=6,
    >>>     n_trials=32,
    >>>     leeway=0,
    >>>     stim_duration=2,
    >>>     total_duration=360,
    >>>     start_duration=20,
    >>>     end_duration=20,
    >>>     verbose=True,
    >>>     events=["act","norm"]
    >>>     )
    """
    
    def __init__(
        self,
        tmin: Union[int,float]=3,
        tmax: Union[int,float]=18,
        tmean: Union[int,float]=6,
        n_trials: int=None,
        leeway: Union[int,float]=0,
        stim_duration: Union[int,float]=1,
        total_duration: Union[int,float]=None,
        start_duration: Union[int,float]=0,
        end_duration: Union[int,float]=0,
        TR: Union[int,float]=0.105,
        verbose: bool=False,
        events: Union[str,list]='stim',
        seq: list=None):
        
        self.minimal_duration   = tmin
        self.maximal_duration   = tmax
        self.mean_duration      = tmean
        self.n_trials           = n_trials
        self.leeway_duration    = leeway
        self.total_duration     = total_duration
        self.stim_duration      = stim_duration
        self.start_duration     = start_duration
        self.end_duration       = end_duration
        self.TR                 = TR
        self.events             = events
        self.verbose            = verbose
        self.seq                = seq

        if isinstance(self.events, str):
            self.events = [self.events]

        # get itis based on negative exponential
        self.itis = self.iterative_itis(
            mean_duration=self.mean_duration,
            minimal_duration=self.minimal_duration,
            maximal_duration=self.maximal_duration,
            n_trials=self.n_trials,
            leeway=self.leeway_duration,
            verbose=self.verbose)

        # calculate full experiment time
        self.total_experiment_time = self.itis.sum() + self.start_duration + self.end_duration + (self.n_trials*self.stim_duration)

        # check if we have to limit total time
        if isinstance(self.total_duration, (int,float)):
            while self.total_experiment_time > self.total_duration: 
                utils.verbose(f"Total experiment time: {round(self.total_experiment_time,2)}s is too long", self.verbose)

                self.itis = self.iterative_itis(
                    mean_duration=self.mean_duration,
                    minimal_duration=self.minimal_duration,
                    maximal_duration=self.maximal_duration,
                    n_trials=self.n_trials,
                    leeway=self.leeway_duration,
                    verbose=self.verbose)
                
                self.total_experiment_time = self.itis.sum() + self.start_duration + self.end_duration + (self.n_trials*self.stim_duration)

        self.n_samples = round(self.total_experiment_time/self.TR)
        utils.verbose((f"Total experiment time: {round(self.total_experiment_time,2)}s (= {self.n_samples} samples)"), self.verbose)

        self.onset_df = self.itis_to_onsets(
            self.itis,
            start_duration=self.start_duration,
            stim_duration=self.stim_duration,
            TR=self.TR,
            events=self.events,
            seq=self.seq)
        
    def get_itis(self):
        return self.itis
    
    def get_onsets(self):
        return self.onset_df
    
    def plot_iti_distribution(self, *args, **kwargs):
        """plot_iti_distribution

        Plot the distribution of a given set of ITIs. Aside from a few defaults such as axis names, all args from :class:`linescanning.plotting.LazyHist` can be used.

        Example
        ----------
        >>> iti_obj.plot_iti_distribution()
        """

        self.iti_plot = plotting.LazyHist(
            self.itis,
            # kde=True,
            hist=True,
            fill=False,
            y_label2="count",
            x_label2="ITI (s)",
            # hist_kwargs={"alpha": 0.4},
            # kde_kwargs={"linewidth": 4},
            *args,
            **kwargs)

    @staticmethod
    def itis_to_onsets(
        itis: np.ndarray, 
        start_duration: Union[int,float]=20, 
        stim_duration: Union[int,float]=1, 
        TR: Union[int,float]=0.105, 
        events: Union[str,list]='stim',
        seq: Union[list,np.ndarray]=None,
        shuffle: bool=True):

        """itis_to_onsets

        Transform a set of ITIs to a `pandas.DataFrame` trough :class:`linescanning.dataset.ParseExpToolsFile`. 

        Parameters
        ----------
        itis: np.ndarray
            array representing the ITIs
        start_duration: int, float, optional
            Baseline at the start of the experiment, by default 0
        end_duration: int, float, optional
            Baseline at the end of the experiment, by default 0            
        TR: int, float, optional
            Repetition time, by default 0.105. Used for indexing of the onset dataframe
        events: str, list, optional
            Specify event names, by default 'stim'. Used for generation of the dataframe. If the experiment involves 1 event, the event name is "stim", otherwise these names will be randomly assigned to the ITIs.
        seq: list, np.ndarray, optional
            Define stimulus sequence; must consist of integers (0,1,2) corresponding to the events in `events`
        shuffle: bool, optional
            Randomize the sequence order if `seq=None` and `events` is an instance of `list` or `np.ndarray`. Default behavior is `True`

        Returns
        ----------
        pandas.DataFrame
            dataframe formatted like :class:`linescanning.dataset.ParseExpToolsFile`
        """
        
        onsets = []
        start_ = start_duration
        n_trials = len(itis)
        for ix,iti in enumerate(itis):
            
            # start duration + iti = onset time
            start_ += iti

            # append to onsets
            onsets.append(start_)

            # add stimulus duration 
            start_ += stim_duration
        
        # prepare onsets, condition, and duration as arrays
        onsets = np.array(onsets)[...,np.newaxis]
        
        # check if we got multiple events for itis
        if isinstance(events, list):
            n_events = len(events)

            # get stim sequence
            if not isinstance(seq, (list,np.ndarray)):
                presented_stims = np.r_[[np.full(n_trials//n_events, ii, dtype=int) for ii in range(n_events)]].flatten()
                np.random.shuffle(presented_stims)
            else:
                presented_stims = seq

            condition = np.array([events[ii] for ii in presented_stims])[...,np.newaxis]
        else:
            condition = np.full(onsets.shape, events)

        duration = np.full(onsets.shape, stim_duration)

        # combine
        onsets = np.hstack((onsets, condition, duration))

        # make a dataframe
        columns = ['onset', 'event_type', 'duration']
        
        # add indices
        onsets_df = pd.DataFrame(onsets, columns=columns)
        onsets_df["subject"],onsets_df["run"] = 1,1

        onsets_df['event_type'] = onsets_df['event_type'].astype(str)
        onsets_df['onset'] = onsets_df['onset'].astype(float)
        onsets_df['duration'] = onsets_df['duration'].astype(float)

        onsets_df = onsets_df.set_index(["subject","run","event_type"])
        
        return onsets_df
    
    def to_file(self, fname):
        if isinstance(fname, str):
            if fname.endswith("txt"):
                np.savetxt(fname, self.itis)
            elif fname.endswith("npy"):
                np.save(fname, self.itis)
            elif fname.endswith("csv"):
                pd.DataFrame({"itis": self.itis}).to_csv(fname)
            elif fname.endswith("tsv"):
                pd.DataFrame({"itis": self.itis}).to_csv(fname, sep="\t")
            else:
                raise TypeError(f"File has unfamiliar extension. File can be saved as 'txt', 'npy', 'csv', or 'tsv'")
        else:
            raise ValueError("No filename specified..")

    @staticmethod
    def create_prf_design(
        df: pd.DataFrame, 
        n_trs: int=None, 
        tr: Union[int,float]=0.105, 
        stim_at_half_tr: bool=False, 
        stims: Union[str,list]=None,
        events: Union[str,list]=None,
        make_square: bool=False,
        n_pix: int=None):

        """create_prf_design

        Create a pRF-like design given the dataframe of onsets.

        Returns
        ----------
        _type_
            _description_

        Example
        ----------
        >>> 
        """
        # if input is an instance of linescanning.simulate.ITI, read info from there
        if isinstance(df, ITI):
            n_trs = df.n_samples
            tr = df.TR
            df = df.get_onsets()
        
        if not isinstance(stims, list):
            stims = [stims]

        if not isinstance(events, list):
            events = [events]

        dm = np.zeros((*stims[0].shape, n_trs))
        try:
            df = df.reset_index()
        except:
            pass

        # backwards compatibility for older versions of lineprf
        for ii in range(df.shape[0]):

            # find time at the middle of TR
            onset_time = df.iloc[ii].onset
            tr_in_samples = (onset_time/tr)

            if stim_at_half_tr:
                tr_in_samples += (0.5*tr)

            tr_in_samples = round(tr_in_samples)

            # find event type
            event_type = df.iloc[ii].event_type

            # get stimulus
            add_stim = stims[events.index(event_type)]

            # get duration
            dur_in_samples = round(df.iloc[ii].duration/tr)

            # inset in design
            # print(f"#{ii+1}= {round(onset_time, 2)}\t| {tr_in_samples} samples\t| duration = {dur_in_samples} samples")
            dm[...,tr_in_samples:tr_in_samples+dur_in_samples] = np.tile(add_stim[...,np.newaxis], dur_in_samples)

        # check if we should make the design a square
        if make_square:
            offset = int((dm.shape[1]-dm.shape[0])/2)
            dm = dm[:, offset:(offset+dm.shape[0])]

            # check if we should also resample to n_pix (can only be done if make_square=True)
            if isinstance(n_pix, int):
                if dm.shape[0] != n_pix:
                    dm = utils.resample2d(dm, n_pix)

        return dm

    @staticmethod
    def _return_itis(
        mean_duration: Union[int,float], 
        minimal_duration: Union[int,float], 
        maximal_duration: Union[int,float], 
        n_trials:int):

        # iti function based on negative exponential
        itis = np.random.exponential(scale=mean_duration-minimal_duration, size=n_trials)
        itis += minimal_duration
        itis[itis>maximal_duration] = maximal_duration
        return itis
    
    @staticmethod
    def iterative_itis(
        mean_duration: Union[int,float]=6, 
        minimal_duration: Union[int,float]=3, 
        maximal_duration: Union[int,float]=18, 
        n_trials: int=None, 
        leeway: Union[int,float]=0, 
        verbose: bool=False):
        
        nits = 0
        itis = ITI._return_itis(
            mean_duration=mean_duration,
            minimal_duration=minimal_duration,
            maximal_duration=maximal_duration,
            n_trials=n_trials)

        total_iti_duration = n_trials * mean_duration
        min_iti_duration = total_iti_duration - leeway
        max_iti_duration = total_iti_duration + leeway
        while (itis.sum() < min_iti_duration) | (itis.sum() > max_iti_duration):
            itis = ITI._return_itis(
                mean_duration=mean_duration,
                minimal_duration=minimal_duration,
                maximal_duration=maximal_duration,
                n_trials=n_trials)
            nits += 1

        utils.verbose(f'ITIs created with total ITI duration of {round(itis.sum(),2)}s after {nits} iterations', verbose)    

        return itis    
    
def prediction_from_obj(
    pars,
    iti_obj,
    model="norm",
    stims=None,
    TR=0.105,
    ):

    # make square because the pRF has square dimensions
    dm = iti_obj.create_prf_design(
        iti_obj,
        stims=stims,
        events=iti_obj.events,
        make_square=True)

    # initialize object without actual data; mainly insert design matrix
    obj_ = prf.pRFmodelFitting(
        None,
        design_matrix=dm,
        TR=TR,
        verbose=False,
        model=model
    )
    
    # load parameters
    obj_.load_params(
        pars, 
        model=model, 
        stage="iter")

    # get timecourse + make plot
    _,_,_,tc_def = obj_.plot_vox(
        vox_nr=0,
        model=model,
        make_figure=False)

    return tc_def

def variance_from_prf_design(
    pars,
    model="norm",
    TR=0.105,
    stims=None,
    **kwargs  
    ):

    iti_obj = ITI(**kwargs)
    tc_def = prediction_from_obj(
        pars,
        iti_obj,
        stims=stims,
        TR=TR,
        model=model)

    # get variance
    return np.var(tc_def),iti_obj

def optimize_stimulus_order(
    pars,
    model="norm",
    TR=0.105,
    stims=None,
    order=None,
    **kwargs  
    ):

    presented_stims = order.copy()
    np.random.shuffle(presented_stims)

    var_def,obj_ = variance_from_prf_design(
        pars,
        model=model,
        stims=stims,
        TR=TR,
        seq=presented_stims,
        **kwargs
    )

    return var_def,obj_

def optimize_stimulus_isi(
    pars,
    model="norm",
    TR=0.105,
    stims=None,
    **kwargs  
    ):

    var_def, obj_ = variance_from_prf_design(
        pars,
        model=model,
        stims=stims,
        TR=TR,
        **kwargs
    )

    return var_def,obj_    
