from . import glm, plotting, preproc, utils
import matplotlib.pyplot as plt
import nibabel as nb
from nilearn.signal import _standardize
from niworkflows.reports import core
import numpy as np
import os
from pathlib import Path
import pandas as pd
from scipy import io, stats
from time import strftime
from uuid import uuid4
import warnings

try:
    import hedfpy
    HEDFPY_AVAILABLE = True
except:
    HEDFPY_AVAILABLE = False

opj = os.path.join
pd.options.mode.chained_assignment = None # disable warning thrown by string2float
warnings.filterwarnings("ignore")

def check_input_is_list(obj, var=None, list_element=0, matcher="func_file"):

    if hasattr(obj, var):
        attr = getattr(obj, var)
    else:
        raise ValueError(f"Class does not have '{var}'-attribute")

    if isinstance(attr, (list,np.ndarray)):
        if len(attr) != len(getattr(obj,matcher)):
            raise ValueError(f"Length of '{var}' ({len(attr)}) does not match number of func files ({len(getattr(obj,matcher))}). Either specify a list of equal lenghts or 1 integer value for all volumes")

        return attr[list_element]
    else:
        return attr


class ParseEyetrackerFile():

    """ParseEyetrackerFile

    Class for parsing edf-files created during experiments with Exptools2. The class will read in the file, read when the experiment actually started, correct onset times for this start time and time deleted because of removing the first few volumes (to do this correctly, set the `TR` and `deleted_first_timepoints`). You can also provide a numpy array/file containing eye blinks that should be added to the onset times in real-world time (seconds). In principle, it will return a pandas DataFrame indexed by subject and run that can be easily concatenated over runs. This function relies on the naming used when programming the experiment. In the `session.py` file, you should have created `phase_names=['iti', 'stim']`; the class will use these things to parse the file.

    Parameters
    ----------
    edf_file: str, list
        path pointing to the output file of the experiment; can be a list of multiple 
    subject: int
        subject number in the returned pandas DataFrame (should start with 1, ..., n)
    run: int
        run number you'd like to have the onset times for
    low_pass_pupil_f: float, optional
        Low-pass cutoff frequency
    high_pass_pupil_f: float, optional
        High-pass cutoff frequency
    TR: float
        repetition time to correct onset times for deleted volumes
    deleted_first_timepoints: int
        number of volumes to delete to correct onset times for deleted volumes

    Examples
    ----------
    >>> from linescanning.utils import ParseExpToolsFile
    >>> file = 'some/path/to/exptoolsfile.tsv'
    >>> parsed_file = ParseExpToolsFile(file, subject=1, run=1, button=True)
    >>> onsets = parsed_file.get_onset_df()

    >>> # If you want to get all your subjects and runs in 1 nideconv compatible dataframe, you can do something like this:
    >>> onsets = []
    >>> run_subjects = ['001','002','003']
    >>> for sub in run_subjects:
    >>>     path_tsv_files = os.path.join(f'some/path/sub-{sub}')
    >>>     f = os.listdir(path_tsv_files)
    >>>     nr_runs = []; [nr_runs.append(os.path.join(path_tsv_files, r)) for r in f if "events.tsv" in r]
    >>> 
    >>>     for run in range(1,len(nr_runs)+1):
    >>>         sub_idx = run_subjects.index(sub)+1
    >>>         onsets.append(ParseExpToolsFile(df_onsets, subject=sub_idx, run=run).get_onset_df())
    >>>         
    >>> onsets = pd.concat(onsets).set_index(['subject', 'run', 'event_type'])
    """

    def __init__(
        self, 
        edf_file, 
        subject=1, 
        run=1, 
        low_pass_pupil_f=6.0, 
        high_pass_pupil_f=0.01,
        func_file=None, 
        TR=0.105, 
        verbose=False, 
        use_bids=True,
        nr_vols=None):

        if not HEDFPY_AVAILABLE:
            raise ModuleNotFoundError("could not find 'hedfpy', so this functionality is disabled")
        else:
            print("I was able to import hedfpy")

        self.edf_file           = edf_file
        self.func_file          = func_file
        self.sub                = subject
        self.run                = run
        self.TR                 = TR
        self.low_pass_pupil_f   = low_pass_pupil_f
        self.high_pass_pupil_f  = high_pass_pupil_f
        self.verbose            = verbose
        self.use_bids           = use_bids
        self.include_blinks     = False
        self.nr_vols            = nr_vols

        # add all files to h5-file
        if isinstance(self.edf_file, str) or isinstance(self.edf_file, list):
            
            if self.verbose:
                print("\nEYETRACKER")

            self.preprocess_edf_files()
            self.ho.close_hdf_file()

    def preprocess_edf_files(self):

        # deal with edf-files
        if isinstance(self.edf_file, str):
            edfs = [self.edf_file]
        elif isinstance(self.edf_file, list):
            edfs = self.edf_file.copy()
        else:
            raise ValueError(f"Input must be 'str' or 'list', not '{type(self.edf_file)}'")

        # deal with edf-files
        if self.func_file != None:
            if isinstance(self.func_file, str):
                self.func_file = [str(self.func_file)]
            elif isinstance(self.func_file, list):
                self.func_file = self.func_file.copy()
            else:
                raise ValueError(f"Input must be 'str' or 'list', not '{type(self.edf_file)}'")

            self.nr_vols = self.vols(self.func_file)
        else:
            if not isinstance(self.nr_vols, int):
                raise ValueError(f"Nr of volumes must be specified (or pass the functional file)")

        h5_file = opj(os.path.dirname(edfs[0]), f"eye.h5")
        self.ho = hedfpy.HDFEyeOperator(h5_file)
        if not os.path.exists(h5_file):
            for i, edf_file in enumerate(edfs):

                if not os.path.exists(edf_file):
                    raise FileNotFoundError(f"Could not read specified file: '{edf_file}'")

                if self.use_bids:
                    bids_comps = utils.split_bids_components(edf_file)
                    for el in ['sub', 'run']:
                        setattr(self, el, bids_comps[el])
                else:
                    self.run = i+1

                alias = f"run_{self.run}"

                self.ho.add_edf_file(edf_file)
                self.ho.edf_message_data_to_hdf(alias=alias)
                self.ho.edf_gaze_data_to_hdf(
                    alias=alias,
                    pupil_hp=self.high_pass_pupil_f,
                    pupil_lp=self.low_pass_pupil_f)

        else:
            self.ho.open_hdf_file()

        self.df_eye         = []
        self.df_blinks      = []
        self.df_space_func  = []
        self.df_space_eye   = []
        self.df_saccades    = []
        for i, edf_file in enumerate(edfs):

            if self.verbose:
                print(f"Preprocessing {edf_file}")

            if self.use_bids:
                bids_comps = utils.split_bids_components(edf_file)
                for el in ['sub', 'run']:
                    setattr(self, el, bids_comps[el])
            else:
                self.run = i+1
            
            # check if we got multiple TRs for different edf-files
            use_TR = check_input_is_list(
                self, 
                "TR", 
                list_element=self.run,
                matcher="edf_file")

            # full output from 'fetch_relevant_info' > use sub as differentiator if multiple files were given
            self.data = self.fetch_relevant_info(TR=use_TR, nr_vols=self.nr_vols)

            # collect outputs
            self.df_blinks.append(self.fetch_eyeblinks())
            self.df_space_func.append(self.fetch_eye_func_time())
            self.df_space_eye.append(self.fetch_eye_tracker_time())
            self.df_saccades.append(self.fetch_saccades())

        self.df_blinks = pd.concat(self.df_blinks).set_index(['subject', 'run', 'event_type'])
        self.df_space_func = pd.concat(self.df_space_func).set_index(['subject', 'run', 't'])
        self.df_space_eye = pd.concat(self.df_space_eye).set_index(['subject', 'run', 't'])
        self.df_saccades = pd.concat(self.df_saccades).set_index(['subject', 'run', 'event_type'])

    def fetch_blinks_run(self, run=1, return_type='df'):
        blink_df = utils.select_from_df(
            self.df_blinks, 
            expression=(f"run = {run}"), 
            index=['subject', 'run', 'event_type'])

        if return_type == "df":
            return blink_df
        else:
            return blink_df.values

    def fetch_eyeblinks(self):
        return self.data['blink_events']
    
    def fetch_saccades(self):
        return self.data['saccades']        

    def fetch_eye_func_time(self):
        return self.data['space_func']

    def fetch_eye_tracker_time(self):
        return self.data['space_eye']

    def fetch_relevant_info(
        self,
        nr_vols=None,
        TR=None):

        # set alias
        alias = f'run_{self.run}'
        if self.verbose:
            print(" Alias:       ", alias)

        # load times per session:
        trial_times = self.ho.read_session_data(alias, 'trials')
        trial_phase_times = self.ho.read_session_data(alias, 'trial_phases')

        # fetch duration of scan
        func_time = nr_vols*TR

        # get block parameters
        self.session_start_EL_time = trial_times.iloc[0, :][0]
        self.sample_rate = self.ho.sample_rate_during_period(alias)

        # add number of fMRI*samplerate as stop EL time
        self.session_stop_EL_time = self.session_start_EL_time+(func_time*self.sample_rate)
        self.time_period = [self.session_start_EL_time,self.session_stop_EL_time]

        eye = self.ho.eye_during_period(self.time_period, alias)

        if self.verbose:
            print(" Sample rate: ", self.sample_rate)
            print(" Start time:  ", self.time_period[0])
            print(" Stop time:   ", self.time_period[1])

        # set some stuff required for successful plotting with seconds on the x-axis
        div = False
        if self.sample_rate == 500:
            n_samples = int(self.time_period[1]-self.time_period[0])/2
            duration_sec = n_samples*(1/self.sample_rate)*2

            div = True
        elif self.sample_rate == 1000:
            n_samples = int(self.time_period[1]-self.time_period[0])
            duration_sec = n_samples*(1/self.sample_rate)
        else:
            raise ValueError(f"Did not recognize sample_rate of {self.sample_rate}")

        if self.verbose:
            print(" Duration:     {}s [{} samples]".format(duration_sec, n_samples))

        # Fetch a bunch of data
        extract = [
            "pupil", 
            "pupil_int",
            "gaze_x_int",
            "gaze_y_int",
            "gaze_x",
            "gaze_y"]           
        
        if self.verbose:
            print(f" Fetching:     {extract}")

        df_space_eye = {} 
        for extr in extract:
            data = np.squeeze(
                self.ho.signal_during_period(
                    time_period=self.time_period, 
                    alias=alias, 
                    signal=extr, 
                    requested_eye=eye).values)

            df_space_eye[extr] = data
        
        df_space_func = {}
        rs_col = list(df_space_eye.keys())
        for col in rs_col:
            df_space_func[col] = glm.resample_stim_vector(df_space_eye[col], nr_vols)   

            # high-pass filter
            df_space_func[f"{col}_hp"] = preproc.highpass_dct(
                df_space_func[col], 
                self.high_pass_pupil_f,
                TR=TR)[0]

            # percent signal change
            for hh in [col,f"{col}_hp"]:
                dd = df_space_func[hh]
                df_space_func[f"{hh}_psc"] = np.squeeze(
                    utils.percent_change(
                        dd, 
                        0,
                        baseline=dd.shape[0]))

        # add start time to it
        start_exp_time = trial_times.iloc[0, :][-1]

        if self.verbose:
            print(" Start time exp = ", round(start_exp_time, 2))

        # get onset time of blinks, cluster blinks that occur within 350 ms
        bb = self.ho.blinks_during_period(self.time_period, alias=alias)

        onsets = bb["start_timestamp"].values
        duration = bb["duration"].values/self.sample_rate

        # convert the onsets to seconds in experiment
        onsets = (onsets-self.time_period[0])*(1/self.sample_rate)

        # normal eye blink is 1 blink every 4 seconds, throw warning if we found more than a blink per second
        # ref: https://www.sciencedirect.com/science/article/abs/pii/S0014483599906607
        blink_rate = len(onsets)/duration_sec

        if self.verbose:
            print(f" Found {len(onsets)} blinks [{round(blink_rate, 2)} blinks per second]")

        # extract saccades
        df_saccades = self.ho.detect_saccades_during_period(time_period=self.time_period, alias=alias)
        
        if len(df_saccades) > 0:
            if self.verbose:
                print(f" Found {len(df_saccades)} saccades")

            # convert saccade onset to seconds relative to start of run
            df_saccades["onset"] = df_saccades["expanded_start_time"]/self.sample_rate

            # add some stuff for indexing
            for tt,mm in zip(["subject","run","event_type"],[self.sub,self.run,"saccade"]):
                df_saccades[tt] = mm

        # build dataframe with relevant information
        df_space_eye = pd.DataFrame(df_space_eye)
        df_space_func = pd.DataFrame(df_space_func)

        # index
        df_space_eye['subject'], df_space_eye['run'], df_space_eye['t'] = self.sub, self.run, list((1/self.sample_rate)*np.arange(df_space_eye.shape[0]))

        # index
        df_space_func['subject'], df_space_func['run'], df_space_func['t'] = self.sub, self.run, list(TR*np.arange(df_space_func.shape[0]))

        # index
        df_blink_events = pd.DataFrame(onsets, columns=['onset'])

        for tt,mm in zip(['subject','run','event_type'],[self.sub,self.run,"blink"]):
            df_blink_events[tt] = mm

        if self.verbose:
            print("Done")

        return_dict = {
            "space_eye": df_space_eye,
            "space_func": df_space_func,
            "blink_events": df_blink_events}

        if len(df_saccades) > 0:
            return_dict["saccades"] = df_saccades

        return return_dict

    def vols(self, func_file):
        if func_file.endswith("gz") or func_file.endswith('nii'):
            img = nb.load(func_file)
            nr_vols = img.get_fdata().shape[-1]
            self.TR2 = img.header['pixdim'][4]
        elif func_file.endswith("mat"):
            raw = io.loadmat(func_file)
            tag = list(raw.keys())[-1]
            raw = raw[tag]
            nr_vols = raw.shape[-1]
        else:
            raise ValueError(f"Could not derive number of volumes for file '{func_file}'")

        return nr_vols

class ParseExpToolsFile(ParseEyetrackerFile):

    """ParseExpToolsFile()

    Class for parsing tsv-files created during experiments with Exptools2. The class will read in the file, read when the experiment actually started, correct onset times for this start time and time deleted because of removing the first few volumes (to do this correctly, set the `TR` and `deleted_first_timepoints`). You can also provide a numpy array/file containing eye blinks that should be added to the onset times in real-world time (seconds). In principle, it will return a pandas DataFrame indexed by subject and run that can be easily concatenated over runs. This function relies on the naming used when programming the experiment. In the `session.py` file, you should have created `phase_names=['iti', 'stim']`; the class will use these things to parse the file.

    Parameters
    ----------
    tsv_file: str, list
        path pointing to the output file of the experiment
    subject: int
        subject number in the returned pandas DataFrame (should start with 1, ..., n)
    run: int
        run number you'd like to have the onset times for
    button: bool
        boolean whether to include onset times of button responses (default is false). ['space'] will be ignored as response
    TR: float
        repetition time to correct onset times for deleted volumes
    deleted_first_timepoints: int
        number of volumes to delete to correct onset times for deleted volumes. Can be specified for each individual run if `tsv_file` is a list
    use_bids: bool, optional
        If true, we'll read BIDS-components such as 'sub', 'run', 'task', etc from the input file and use those as indexers, rather than sequential 1,2,3.
    funcs: str, list, optional
        List of functional files that is being passed down down to :class:`linescanning.dataset.ParseEyetrackerFile`. Required for correct resampling to functional space
    edfs: str, list, optional
        List of eyetracking output files that is being passed down down to :class:`linescanning.dataset.ParseEyetrackerFile`.
    verbose: bool, optional
        Print details to the terminal, default is False
    phase_onset: int, optional
        Which phase of exptools-trial should be considered the actual stimulus trial. Usually, `phase_onset=0` means the interstimulus interval. Therefore, default = 1
    stim_duration: str, int, optional
        If desired, add stimulus duration to onset dataframe. Can be one of 'None', 'stim' (to use duration from exptools'  log file) or any given integer
    add_events: str, list, optional
        Add additional events to onset dataframe. Must be an existing column in the exptools log file. For intance, `responses` and `event_type = stim` are read in by default, but if we have a separate column containing the onset of some target (e.g., 'target_onset'), we can add these times to the dataframe with `add_events='target_onset'`.
    event_names: str, list, optional
        Custom names for manually added events through `add_events` if the column names are not the names you want to use in the dataframe. E.g., if I find `target_onset` too long of a name, I can specify `event_names='target'`. If `add_events` is a list, then `event_names` must be a list of equal length if custom names are desired. By default we'll take the names from `add_events`
    RTs: bool, optional
        If we have a design that required some response to a stimulus, we can request the reaction times. Default = False
    RT_relative_to: str, optional
        If `RTs=True`, we need to know relative to what time the button response should be offset. Only correct responses are considered, as there's a conditional statement that requires the present of the reference time (e.g., `target_onset`) and button response. If there's a response but no reference time, the reaction time cannot be calculated. If you do not have a separate reference time column, you can specify `RT_relative_to='start'` to calculate the reaction time relative to onset time. If `RT_relative_to != 'start'`, I'll assume you had a target in your experiment in X/n_trials. From this, we can calculate the accuracy and save that to `self.df_accuracy`, while reaction times are saved in`self.df_rts`

    Examples
    ----------
    >>> from linescanning.utils import ParseExpToolsFile
    >>> file = 'some/path/to/exptoolsfile.tsv'
    >>> parsed_file = ParseExpToolsFile(file, subject=1, run=1, button=True)
    >>> onsets = parsed_file.get_onset_df()

    >>> # If you want to get all your subjects and runs in 1 nideconv compatible dataframe, you can do something like this:
    >>> onsets = []
    >>> run_subjects = ['001','002','003']
    >>> for sub in run_subjects:
    >>>     path_tsv_files = os.path.join(f'some/path/sub-{sub}')
    >>>     f = os.listdir(path_tsv_files)
    >>>     nr_runs = []; [nr_runs.append(os.path.join(path_tsv_files, r)) for r in f if "events.tsv" in r]
    >>> 
    >>>     for run in range(1,len(nr_runs)+1):
    >>>         sub_idx = run_subjects.index(sub)+1
    >>>         onsets.append(ParseExpToolsFile(df_onsets, subject=sub_idx, run=run).get_onset_df())
    >>>         
    >>> onsets = pd.concat(onsets).set_index(['subject', 'run', 'event_type'])
    """

    def __init__(
        self, 
        tsv_file, 
        subject=1, 
        run=1, 
        button=False, 
        RTs=False,
        RT_relative_to=None,
        TR=0.105, 
        deleted_first_timepoints=0, 
        edfs=None, 
        funcs=None, 
        use_bids=True,
        verbose=False,
        phase_onset=1,
        stim_duration=None,
        add_events=None,
        event_names=None,
        **kwargs):

        self.tsv_file                       = tsv_file
        self.sub                            = subject
        self.run                            = run
        self.TR                             = TR
        self.deleted_first_timepoints       = deleted_first_timepoints
        self.button                         = button
        self.funcs                          = funcs
        self.edfs                           = edfs
        self.use_bids                       = use_bids
        self.verbose                        = verbose
        self.phase_onset                    = phase_onset
        self.stim_duration                  = stim_duration
        self.RTs                            = RTs
        self.RT_relative_to                 = RT_relative_to
        self.add_events                     = add_events
        self.event_names                    = event_names
        self.__dict__.update(kwargs)

        if self.edfs != None:
            super().__init__(
                self.edfs, 
                subject=self.sub, 
                func_file=self.funcs, 
                TR=self.TR, 
                use_bids=self.use_bids, 
                verbose=self.verbose,
                **kwargs)

        if self.verbose:
            print("\nEXPTOOLS")

        if isinstance(self.tsv_file, str):
            self.tsv_file = [self.tsv_file]

        if isinstance(self.tsv_file, list):
            df_onsets = []
            df_rts = []
            df_accuracy = []
            for run, onset_file in enumerate(self.tsv_file):

                if self.use_bids:
                    bids_comps = utils.split_bids_components(onset_file)
                    for el in ['sub', 'run']:
                        setattr(self, el, bids_comps[el])
                else:
                    self.run = run+1

                # check if we got different nr of vols to delete per run
                delete_vols = check_input_is_list(
                    self, 
                    "deleted_first_timepoints", 
                    list_element=run,
                    matcher="tsv_file")

                # check if we got different stimulus durations per run
                duration = check_input_is_list(
                    self, 
                    var="stim_duration", 
                    list_element=run,
                    matcher="tsv_file"
                    )

                # read in the exptools-file
                self.preprocess_exptools_file(
                    onset_file,
                    run=self.run,
                    delete_vols=delete_vols,
                    phase_onset=self.phase_onset,
                    duration=duration)

                # append to df
                df_onsets.append(self.get_onset_df(index=False))

                # check if we got RTs
                try:
                    df_rts.append(self.get_rts_df(index=False))
                except:
                    pass

                # check if we got accuracy (only if RT_relative_to != 'start')
                try:
                    df_accuracy.append(self.get_accuracy(index=False))
                except:
                    pass


            # concatemate df
            self.df_onsets = pd.concat(df_onsets).set_index(['subject', 'run', 'event_type'])

            # rts
            try:
                self.df_rts = pd.concat(df_rts).set_index(['subject', 'run'])
            except:
                pass

            # accuracy
            try:
                self.df_accuracy = pd.concat(df_accuracy).set_index(['subject', 'run'])
            except:
                pass

        # get events per run
        self.events_per_run = self.events_per_run()


    def events_per_run(self):
        n_runs = np.unique(self.df_onsets.reset_index()['run'].values)
        events = {}
        for run in n_runs:
            df = utils.select_from_df(self.df_onsets, expression=f"run = {run}", index=None)
            events[run] = np.unique(df['event_type'].values)

        return events

    def events_single_run(self, run=1):
        return self.events_per_run[run]

    def preprocess_exptools_file(self, tsv_file, run=1, delete_vols=0, phase_onset=1, duration=None):

        if self.verbose:
            print(f"Preprocessing {tsv_file}")
        with open(tsv_file) as f:
            self.data = pd.read_csv(f, delimiter='\t')

        # trim onsets to first 't'
        delete_time         = delete_vols*self.TR
        self.start_time     = float(self.data.loc[(self.data['event_type'] == "pulse") & (self.data['phase'] == 0)]['onset'].values[0])
        self.trimmed        = self.data.loc[(self.data['event_type'] == "stim") & (self.data['phase'] == phase_onset)].iloc[1:,:]
        self.onset_times    = self.trimmed['onset'].values[...,np.newaxis]

        skip_duration = False
        if isinstance(duration, float) or isinstance(duration, int):
            self.durations = np.full_like(self.onset_times, float(duration))
        elif duration == None:
            skip_duration = True
        else:
            self.durations = self.trimmed['duration'].values[...,np.newaxis]

        self.condition = self.trimmed['condition'].values[..., np.newaxis]
        if self.verbose:
            print(f" 1st 't' @{round(self.start_time,2)}s")
        
        # add button presses
        if self.button:

            # get dataframe with responses
            self.response_df = self.data.loc[(self.data['event_type'] == "response") & (self.data['response'] != 'space')]

            # get the onset times
            self.response_times = self.response_df['onset'].values[...,np.newaxis]

            # stack onset times
            self.onset_times = np.vstack([self.onset_times, self.response_times])

            # make a condition column
            self.response_condition = self.response_df['response'].values[...,np.newaxis]

            # stack it onto existing condition array
            self.condition = np.vstack([self.condition, self.response_condition])
        
        # check if we should include other events
        if isinstance(self.add_events, str):
            self.add_events = [self.add_events]

        if isinstance(self.event_names, str):
            self.event_names = [self.event_names]            

        if isinstance(self.add_events, list):
            if isinstance(self.event_names, list):
                if len(self.event_names) != len(self.add_events):
                    raise ValueError(f"Length ({len(self.add_events)}) of added events {self.add_events} does not equal the length ({len(self.event_names)}) of requested event names {self.event_names}")
            else:
                self.event_names = self.add_events.copy()

            for ix,ev in enumerate(self.add_events):
                ev_times = np.array([ii for ii in np.unique(self.data[ev].values)])

                # filter for nan (https://stackoverflow.com/a/11620982)
                ev_times = ev_times[~np.isnan(ev_times)][...,np.newaxis]

                # create condition
                ev_names = np.full(ev_times.shape, self.event_names[ix])

                # add times and names to array
                self.onset_times = np.vstack((self.onset_times, ev_times))
                self.condition = np.vstack((self.condition, ev_names))

        # check if we should add duration (can't be used in combination with add_events)
        if not skip_duration:
            if isinstance(self.add_events, list):
                raise TypeError(f"Cannot do this operation because I don't know the durations for the added events. Please consider using 'stim_duration!={self.stim_duration}' or 'add_events=None'")
            self.onset = np.hstack((self.onset_times, self.condition, self.durations))
        else:
            self.onset = np.hstack((self.onset_times, self.condition))

        # sort array based on onset times (https://stackoverflow.com/a/2828121)        
        self.onset = self.onset[self.onset[:,0].argsort()]

        # correct for start time of experiment and deleted time due to removal of inital volumes
        self.onset[:, 0] = self.onset[:, 0] - (self.start_time + delete_time)

        if self.verbose:
            print(f" Cutting {round(self.start_time + delete_time,2)}s from onsets")

            if not skip_duration:
                print(f" Avg duration = {round(self.durations.mean(),2)}s")

        # make dataframe
        if skip_duration:
            columns = ['onset', 'event_type']
        else:
            columns = ['onset', 'event_type', 'duration']
            
        self.onset_df = self.index_onset(self.onset, columns=columns, subject=self.sub, run=run)

        # check if we should do reaction times
        if self.RTs:
            if not isinstance(self.RT_relative_to, str):
                raise ValueError(f"Need a reference column to calculate reaction times (RTs), not '{self.RT_relative_to}'")
            
            # get response times
            if not hasattr(self, "response_df"):
                self.response_df = self.data.loc[(self.data['event_type'] == "response") & (self.data['response'] != 'space')]
            
            # fetch trials were target happened
            self.id_target = {}
            self.id_no_target = []
            self.false_alarms = []
            self.correct_rejection = []
            for idx in self.trimmed['trial_nr'].values:

                # cross-check
                trial_overview = self.data.loc[(self.data['trial_nr'] == idx)]
                
                # get the values of reference column; skip if all elements are nan
                if self.RT_relative_to != "start":
                    ref_values = np.unique(trial_overview[self.RT_relative_to].values)
                    has_nans = np.isnan(ref_values)

                    # should be safe as any response will not have a value in reference column
                    if False in has_nans:
                        
                        # increase number of target to remain agnostic about design
                        id_no_nan = np.where(has_nans == False)[0]
                        if len(id_no_nan) != 0:
                            self.id_target[idx] = ref_values[id_no_nan]
                    else:
                        # append no targets
                        self.id_no_target.append(idx)
                        
                        # false alarm = no target but response
                        if 'response' in list(trial_overview['event_type'].values):
                            self.false_alarms.append(idx)
                        else:
                            # correct rejection = no target and no response
                            self.correct_rejection.append(idx)

            self.rts = []
            self.hits = []
            for target_trial in self.id_target.keys():
        
                # get corresponding reference value
                if self.RT_relative_to != "start":
                    ref_value = self.id_target[target_trial]
                else:
                    ref_value = trial_overview.loc[(trial_overview['event_type'] == 'stim')]['onset'].values
                
                # get response value (length will be 0 if dataframe is empty)
                response_value = self.response_df.loc[(self.response_df['trial_nr'] == target_trial)]['onset'].values

                # check if lengths of reference value and response value are equal
                if len(response_value) == len(ref_value):
                    rt = response_value - ref_value

                    # ignore negative reaction times..
                    if rt > 0:
                        self.rts.append(rt)
                
            if len(self.rts) >= 1:
                self.rts = np.array(self.rts)
            else:
                self.rts = np.array([0])

            if len(self.id_target) != 0:
                self.hits = len(self.rts)/len(self.id_target)
                self.miss = (len(self.id_target)-len(self.rts))/len(self.id_target)
                self.fa = len(self.false_alarms)/len(self.id_no_target)
                self.cr = len(self.correct_rejection)/len(self.id_no_target)

                # set FA to something if 0 to avoid d' = inf
                if self.fa == float(0):
                    self.fa = 0.5*(1/len(self.trimmed['trial_nr'].values))
                    add_fa_txt = f"[added {round(self.fa,2)} to avoid d' = inf]"
                else:
                    add_fa_txt = ""

                # set HITS to something if 0 to avoid d' = inf
                if self.hits == float(0):
                    self.hits = 0.5*(1/len(self.trimmed['trial_nr'].values))
                    add_hits_txt = f"[added {round(self.hits,2)} to avoid d' = inf]"
                else:
                    add_hits_txt = ""

                # calculate d-prime
                self.hitZ = stats.norm.ppf(self.hits)
                self.faZ = stats.norm.ppf(self.fa)
                self.dPrime = self.hitZ-self.faZ

                # d-prime=0 is considered as pure guessing.
                # d-prime=1 is considered as good measure of signal sensitivity/detectability.
                # d-prime=2 is considered as awesome.
                
                self.accuracy_df = self.index_accuracy(np.array([self.hits, self.miss, self.fa, self.cr, self.dPrime], dtype=float)[np.newaxis,...], columns=['hits','miss','fa','cr','d_prime'], subject=self.sub, run=run)

            if self.verbose:
                if hasattr(self, 'dPrime'):
                    print(f" Hits:\t{round(self.hits,2)}\t({len(self.rts)}/{len(self.id_target)})\t{add_hits_txt}")
                    print(f" Miss:\t{round(self.miss,2)}\t({(len(self.id_target)-len(self.rts))}/{len(self.id_target)})")
                    print(f" FA:\t{round(self.fa,2)}\t({len(self.false_alarms)}/{len(self.id_no_target)})\t{add_fa_txt}")
                    print(f" CR:\t{round(self.cr,2)}\t({len(self.correct_rejection)}/{len(self.id_no_target)})")
                    print(f" D':\t{round(self.dPrime,2)}\t(0=guessing;1=good;2=awesome)")
                
                print(f" Average reaction time (RT) = {round(self.rts.mean(),2)}s (relative to '{self.RT_relative_to}').")

            # parse into dataframe
            self.rts_df = self.index_rts(self.rts, columns=["RTs"], subject=self.sub, run=run)


    @staticmethod
    def index_onset(array, columns=None, subject=1, run=1, TR=0.105, set_index=False):
        
        if columns == None:
            df = pd.DataFrame(array)
        else:
            df = pd.DataFrame(array, columns=columns)
            
        df['subject']       = subject
        df['run']           = run
        df['event_type']    = df['event_type'].astype(str)
        df['onset']         = df['onset'].astype(float)

        # check if we got duration
        try:
            df['duration'] = df['duration'].astype(float)  
        except:
            pass

        if set_index:
            return df.set_index(['subject', 'run', 'event_type'])
        else:
            return df        

    @staticmethod
    def index_rts(array, columns=None, subject=1, run=1, set_index=False):
        
        if columns == None:
            df = pd.DataFrame(array)
        else:
            df = pd.DataFrame(array, columns=columns)
            
        df['subject']   = subject
        df['run']       = run
        df['RTs']       = df['RTs'].astype(float)

        if set_index:
            return df.set_index(['subject', 'run'])
        else:
            return df        

    @staticmethod
    def index_accuracy(array, columns=None, subject=1, run=1, set_index=False):
        
        if columns == None:
            df = pd.DataFrame(array)
        else:
            df = pd.DataFrame(array, columns=columns)
            
        df['subject']   = subject
        df['run']       = run

        if set_index:
            return df.set_index(['subject', 'run'])
        else:
            return df                   

    def get_onset_df(self, index=False):
        """Return the indexed DataFrame containing onset times"""

        if index:
            return self.onset_df.set_index(['subject', 'run', 'event_type'])
        else:
            return self.onset_df

    def get_rts_df(self, index=False):
        """Return the indexed DataFrame containing reaction times"""

        if index:
            return self.rts_df.set_index(['subject', 'run'])
        else:
            return self.rts_df

    def get_accuracy(self, index=False):
        """Return the indexed DataFrame containing reaction times"""

        if index:
            return self.accuracy_df.set_index(['subject', 'run'])
        else:
            return self.accuracy_df             

    def onsets_to_fsl(self, fmt='3-column', amplitude=1, output_base=None):
        """onsets_to_fsl

        This function creates a text file with a single column containing the onset times of a given condition. Such a file can be used for SPM or FSL modeling, but it should be noted that the onset times have been corrected for the deleted volumes at the beginning. So make sure your inputting the correct functional data in these cases.

        Parameters
        ----------
        subject: int
            subject number you'd like to have the onset times for
        run: int
            run number you'd like to have the onset times for
        condition: str
            name of the condition you'd like to have the onset times for as specified in the data frame
        fname: str
            path to output name for text file

        Returns
        ----------
        str
            if `fname` was specified, a new file will be created and `fname` will be returned as string pointing to that file

        list
            if `fname` was *None*, the list of onset times will be returned
        """

        onsets = self.df_onsets.copy()
        subj_list = self.get_subjects(onsets)
        for sub in subj_list:
            df = utils.select_from_df(onsets, expression=f"subject = {sub}")

            n_runs = self.get_runs(df)

            for run in n_runs:
                onsets_per_run = utils.select_from_df(df, expression=f"run = {run}")
                events_per_run = self.get_events(onsets_per_run)

                for ix, ev in enumerate(events_per_run):

                    onsets_per_event = utils.select_from_df(onsets_per_run, expression=f"event_type = {events_per_run[ix]}")
                    if output_base == None:
                        if isinstance(self.tsv_file, list):
                            outdir = os.path.dirname(self.tsv_file[0])
                        elif isinstance(self.tsv_file, str):
                            outdir = os.path.dirname(self.tsv_file)
                        else:
                            outdir = os.getcwd()

                        fname = opj(outdir, f"{ev}_run-{run}.txt")
                    else:
                        fname = f"{output_base}{ix+1}_run-{run}.txt"

                    # fetch the onsets
                    event_onsets = onsets_per_event['onset'].values[..., np.newaxis]
                    if fmt == "3-column":

                        # check if we got duration
                        if 'duration' in list(onsets_per_event.columns):
                            duration_arr = onsets_per_event['duration'].values[..., np.newaxis]
                        else:
                            duration_arr = np.ones_like(onsets_per_event)

                        amplitude_arr = np.full_like(event_onsets, amplitude)
                        three_col = np.hstack((event_onsets, duration_arr, amplitude_arr))

                        print(f"Writing {fname}; {three_col.shape}")
                        np.savetxt(fname, three_col, delimiter='\t', fmt='%1.3f')
                    else:
                        np.savetxt(fname, event_onsets, delimiter='\t', fmt='%1.3f')

    @staticmethod
    def get_subjects(df):
        try:
            df = df.reset_index()
        except:
            pass

        return np.unique(df['subject'].values)

    @staticmethod
    def get_runs(df):
        try:
            df = df.reset_index()
        except:
            pass

        return np.unique(df['run'].values)

    @staticmethod
    def get_events(df):
        try:
            df = df.reset_index()
        except:
            pass

        return np.unique(df['event_type'].values)

class ParsePhysioFile():

    """ParsePhysioFile
    
    In similar style to :class:`linescanning.utils.ParseExpToolsFile` and :class:`linescanning.utils.ParseFuncFile`, we use this class to read in physiology-files created with the PhysIO-toolbox (https://www.tnu.ethz.ch/en/software/tapas/documentations/physio-toolbox) (via `call_spmphysio` for instance). Using the *.mat*-file created with `PhysIO`, we can also attempt to extract `heart rate variability` measures. If this file cannot be found, this operation will be skipped

    Parameters
    ----------
    physio_file: str
        path pointing to the regressor file created with PhysIO (e.g., `call_spmphysio`)
    physio_mat: str
        path pointing to the *.mat*-file created with PhysIO (e.g., `call_spmphysio`)
    subject: int
        subject number in the returned pandas DataFrame (should start with 1, ..., n)
    run: int
        run number you'd like to have the onset times for
    TR: float
        repetition time to correct onset times for deleted volumes
    orders: list
        list of orders used to create the regressor files (see `call_spmphysio`, but default = [2,2,2,]). This one is necessary to create the correct column names for the dataframe
    deleted_first_timepoints: int, optional
        number of volumes deleted at the beginning of the timeseries
    deleted_last_timepoints: int, optional
        number of volumes deleted at the end of the timeseries

    Example
    ----------
    >>> physio_file = opj(os.path.dirname(func_file), "sub-001_ses-1_task-SR_run-1_physio.txt")
    >>> physio_mat  = opj(os.path.dirname(func_file), "sub-001_ses-1_task-SR_run-1_physio.mat")
    >>> physio = utils.ParsePhysioFile(
    >>>     physio_file,
    >>>     physio_mat=physio_mat,
    >>>     subject=func.subject,
    >>>     run=func.run,
    >>>     TR=func.TR,
    >>>     deleted_first_timepoints=func.deleted_first_timepoints,
    >>>     deleted_last_timepoints=func.deleted_last_timepoints)
    >>> physio_df   = physio.get_physio(index=False)
    """

    def __init__(
        self, 
        physio_file, 
        physio_mat=None, 
        subject=1, 
        run=1, 
        TR=0.105, 
        orders=[3,4,1], 
        deleted_first_timepoints=0, 
        deleted_last_timepoints=0, 
        use_bids=False, 
        verbose=True,
        **kwargs):

        self.physio_file                = physio_file
        self.physio_mat                 = physio_mat
        self.sub                        = subject
        self.run                        = run
        self.TR                         = TR
        self.orders                     = orders
        self.deleted_first_timepoints   = deleted_first_timepoints
        self.deleted_last_timepoints    = deleted_last_timepoints
        self.physio_mat                 = physio_mat
        self.use_bids                   = use_bids
        self.verbose                    = verbose
        self.__dict__.update(kwargs)

        print("\nPHYSIO")
        
        self.physio_cols = [f'c_{i}' for i in range(self.orders[0])] + [f'r_{i}' for i in range(self.orders[1])] + [f'cr_{i}' for i in range(self.orders[2])]

        if isinstance(self.physio_file, str):
            self.physio_file = [self.physio_file]

        if isinstance(self.physio_mat, str):
            self.physio_mat = [self.physio_mat]
                
        if isinstance(self.physio_file, list):

            df_physio = []
            for run, func in enumerate(self.physio_file):

                if self.verbose:
                    print(f"Preprocessing {func}")

                if self.use_bids:
                    bids_comps = utils.split_bids_components(func)
                    for el in ['sub', 'run']:
                        setattr(self, el, bids_comps[el])
                else:
                    self.run = run+1

                # check if deleted_first_timepoints is list or not
                delete_first = check_input_is_list(
                    self, 
                    var="deleted_first_timepoints",
                    list_element=run,
                    matcher="func_file")

                # check if deleted_last_timepoints is list or not
                delete_last = check_input_is_list(
                    self, 
                    var="deleted_last_timepoints", 
                    list_element=run,
                    matcher="func_file")

                if self.physio_mat != None:
                    if isinstance(self.physio_mat, list):
                        if len(self.physio_mat) == len(self.physio_file):
                            mat_file = self.physio_mat[run]
                        else:
                            raise ValueError(f"Length of mat-files ({len(self.physio_mat)}) does not match length of physio-files ({len(self.physio_mat)})")
                    else:
                        raise ValueError("Please specify a list of mat-files of equal lengths to that of the list of physio files")
                else:
                    mat_file = None

                self.preprocess_physio_file(
                    func, 
                    physio_mat=mat_file,
                    deleted_first_timepoints=delete_first,
                    deleted_last_timepoints=delete_last)

                df_physio.append(self.get_physio(index=False))

            self.df_physio = pd.concat(df_physio).set_index(['subject', 'run', 't'])
        
    def preprocess_physio_file(
        self, 
        physio_tsv, 
        physio_mat=None, 
        deleted_first_timepoints=0, 
        deleted_last_timepoints=0):

        self.physio_data = pd.read_csv(
            physio_tsv,
            header=None,
            sep="\t",
            engine='python',
            skiprows=deleted_first_timepoints,
            usecols=list(range(0, len(self.physio_cols))))

        self.physio_df = pd.DataFrame(self.physio_data)
        self.physio_df.drop(self.physio_df.tail(deleted_last_timepoints).index,inplace=True)
        self.physio_df.columns = self.physio_cols

        # Try to get the heart rate
        if physio_mat != None:

            self.mat = io.loadmat(physio_mat)
            try:
                self.hr = self.mat['physio']['ons_secs'][0][0][0][0][12]
            except:
                print(" WARNING: no heart rate trace found..")
            
            try:
                self.rvt = self.mat['physio']['ons_secs'][0][0][0][0][13]
            except:
                print(" WARNING: no respiration trace found..")

            # trim beginning and end
            for trace in ['hr', 'rvt']:
                if hasattr(self, trace):
                    if deleted_last_timepoints != 0:
                        self.physio_df[trace] = getattr(self, trace)[deleted_first_timepoints:-deleted_last_timepoints,:]
                    else:
                        self.physio_df[trace] = getattr(self, trace)[deleted_first_timepoints:, :]

        self.physio_df['subject'], self.physio_df['run'], self.physio_df['t'] = self.sub, self.run, list(self.TR*np.arange(self.physio_df.shape[0]))

    def get_physio(self, index=True):
        if index:
            return self.physio_df.set_index(['subject', 'run', 't'])
        else:
            return self.physio_df


class ParseFuncFile(ParseExpToolsFile, ParsePhysioFile):

    """ParseFuncFile

    Class for parsing func-files created with Luisa's reconstruction. It can do filtering, conversion to percent signal change, and create power spectra. It is supposed to look similar to :class:`linescanning.utils.ParseExpToolsFile` to make it easy to translate between the functional data and experimental data.

    Parameters
    ----------
    func_file: str, list
        path or list of paths pointing to the output file of the experiment
    subject: int, optional
        subject number in the returned pandas DataFrame (should start with 1, ..., n)
    run: int, optional
        run number you'd like to have the onset times forpsc_nilearn: b
    baseline: float, int, optional
        Duration of the baseline used to calculate the percent-signal change. This method is the default over `psc_nilearn`
    baseline_units: str, optional
        Units of the baseline. Use `seconds`, `sec`, or `s` to imply that `baseline` is in seconds. We'll convert it to volumes internally. If `deleted_first_timepoints` is specified, `baseline` will be corrected for that as well.
    psc_nilearn: bool, optional
        Use nilearn method of calculating percent signal change. This method uses the mean of the entire timecourse, rather than the baseline period. Overwrites `baseline` and `baseline_units`. Default is False.
    standardize: str, optional
        method of standardization (e.g., "zscore" or "psc")
    low_pass: bool, optional
        Temporally smooth the data. It's a bit of a shame if this is needed. The preferred option is to use aCompCor with `filter_confs=0.2`
    lb: float, optional
        lower bound for signal filtering
    TR: float, optional
        repetition time to correct onset times for deleted volumes
    deleted_first_timepoints: int, list, optional
        number of volumes deleted at the beginning of the timeseries. Can be specified for each individual run if `func_file` is a list
    deleted_last_timepoints: int, list, optional
        number of volumes deleted at the end of the timeseries. Can be specified for each individual run if `func_file` is a list
    window_size: int, optional
        size of window for rolling median and Savitsky-Golay filter
    poly_order: int, optional
        The order of the polynomial used to fit the samples. polyorder must be less than window_length.
    use_bids: bool, optional
        If true, we'll read BIDS-components such as 'sub', 'run', 'task', etc from the input file and use those as indexers, rather than sequential 1,2,3.
    verbose: bool, optional
        Print details to the terminal, default is False
    retroicor: bool, optional
        WIP: implementation of retroicor, requires the specification of `phys_file` and `phys_mat` containing the output from the PhysIO-toolbox
    n_components: int, optional
        Number of components to use for WM/CSF PCA during aCompCor
    select_component: int, optional
        If `verbose=True` and `aCompcor=True`, we'll create a scree-plot of the PCA components. With this flag, you can re-run this call but regress out only this particular component. [Deprecated: `filter_confs` is much more effective]
    filter_confs: float, optional
        High-pass filter the components from the PCA during aCompCor. This seems to be pretty effective. Default is 0.2Hz.
    ses1_2_ls: str, optional:
        Transformation mapping `ses-1` anatomy to current linescanning-session, ideally the multi-slice image that is acquired directly before the first `1slice`-image. Default is None.
    run_2_run: str, list, optional
        (List of) Transformation(s) mapping the slices of subsequent runs to the first acquired `1slice` image. Default is None.
    save_as: str, optional
        Directory + basename for several figures that can be created during the process (mainly during aCompCor)
    transpose: bool, optional
        The data needs to be in the format of <time,voxels>. We'll be trying to force the input data into this format, but sometimes this breaks. This flag serves as an opportunity to flip whatever the default is for a particular input file (e.g., `gii`, `npy`, or `np.ndarray`), so that your final dataframe has the format it needs to have. For gifti-input, we transpose by default. `transpose=True` turns this transposing *off*. For `npy`-inputs, we do **NOT** transpose (we assume the numpy arrays are already in <time,voxels> format). `transpose=True` will transpose this input.

    Example
    ----------
    >>> from linescanning import utils
    >>> func_file = utils.get_file_from_substring(f"run-1_bold.mat", opj('sub-001', 'ses-1', 'func'))
    >>> func = utils.ParseFuncFile(func_file, subject=1, run=1, deleted_first_timepoints=100, deleted_last_timepoints=300)
    >>> raw = func.get_raw(index=True)
    >>> psc = func.get_psc(index=True)
    """

    def __init__(
        self, 
        func_file, 
        subject=1, 
        run=1,
        filter_strategy="hp",
        TR=0.105, 
        lb=0.01,
        deleted_first_timepoints=0, 
        deleted_last_timepoints=0, 
        window_size=11,
        poly_order=3,
        attribute_tag=None,
        hdf_key="df",
        tsv_file=None,
        edf_file=None,
        phys_file=None,
        phys_mat=None,
        use_bids=True,
        button=False,
        verbose=True,
        retroicor=False,
        acompcor=False,
        ica=False,
        n_components=5,
        func_tag=None,
        select_component=None,
        standardization="psc",
        filter_confs=0.2,
        keep_comps=None,
        ses1_2_ls=None,
        run_2_run=None,
        save_as=None,
        gm_range=[355, 375],
        tissue_thresholds=[0.7,0.7,0.7],
        save_ext="svg",
        report=False,
        transpose=False,
        baseline=20,
        baseline_units="seconds",
        psc_nilearn=False,
        foldover="FH",
        shift=0,
        **kwargs):

        self.sub                        = subject
        self.run                        = run
        self.TR                         = TR
        self.lb                         = lb
        self.deleted_first_timepoints   = deleted_first_timepoints
        self.deleted_last_timepoints    = deleted_last_timepoints
        self.window_size                = window_size
        self.poly_order                 = poly_order
        self.attribute_tag              = attribute_tag
        self.hdf_key                    = hdf_key
        self.button                     = button
        self.func_file                  = func_file
        self.tsv_file                   = tsv_file
        self.edf_file                   = edf_file
        self.phys_file                  = phys_file
        self.phys_mat                   = phys_mat
        self.use_bids                   = use_bids
        self.verbose                    = verbose
        self.retroicor                  = retroicor
        self.acompcor                   = acompcor
        self.foldover                   = foldover
        self.shift                      = shift
        self.func_tag                   = func_tag
        self.n_components               = n_components
        self.select_component           = select_component
        self.filter_confs               = filter_confs
        self.standardization            = standardization
        self.ses1_2_ls                  = ses1_2_ls
        self.run_2_run                  = run_2_run
        self.save_as                    = save_as
        self.gm_range                   = gm_range
        self.tissue_thresholds          = tissue_thresholds
        self.save_ext                   = save_ext
        self.filter_strategy            = filter_strategy
        self.report                     = report
        self.transpose                  = transpose
        self.baseline                   = baseline
        self.baseline_units             = baseline_units
        self.psc_nilearn                = psc_nilearn
        self.ica                        = ica
        self.keep_comps                 = keep_comps
        self.__dict__.update(kwargs)

        # sampling rate and nyquist freq
        self.fs = 1/self.TR
        self.fn = self.fs/2

        if self.phys_file != None: 
                                                        
            ParsePhysioFile.__init__(
                self, 
                self.phys_file, 
                physio_mat=self.phys_mat, 
                use_bids=self.use_bids,
                TR=self.TR,
                deleted_first_timepoints=self.deleted_first_timepoints,
                deleted_last_timepoints=self.deleted_last_timepoints,
                **kwargs)
        
        if self.acompcor:
            if isinstance(self.ref_slice, str):
                self.ref_slice = [self.ref_slice]

        if self.verbose:
            print("\nFUNCTIONAL")

        if isinstance(self.func_file, str) or isinstance(self.func_file, np.ndarray):
            self.func_file = [self.func_file]
                
        # start boilerplate
        self.func_pre_desc = """
Functional data preprocessing

For each of the {num_bold} BOLD run(s) found per subject (across all tasks and sessions), the following preprocessing was performed.
""".format(num_bold=len(self.func_file))

        if isinstance(self.func_file, list):

            # initiate some dataframes
            self.df_psc     = []    # psc-data (filtered or not)
            self.df_raw     = []    # raw-data (filtered or not)
            self.df_retro   = []    # z-score data (retroicor'ed, `if retroicor=True`)
            self.df_r2      = []    # r2 for portions of retroicor-regressors (e.g., 'all', 'cardiac', etc)
            self.df_acomp   = []    # aCompCor'ed data
            self.df_zscore  = []    # zscore-d data
            self.df_ica     = []    # ica'ed data    
            self.ica_objs   = []    # keep track of all ICA objects
            self.acomp_objs = []    # keep track of all aCompCor elements
            self.df_gm_only = []    # aCompCor'ed data, only GM voxels
            self.gm_per_run = []    # keep track of GM-voxel indices

            # reports
            for run_id, func in enumerate(self.func_file):
                
                if self.verbose:
                    if isinstance(func, str):
                        print(f"Preprocessing {func}")
                    elif isinstance(func, np.ndarray):
                        print(f"Preprocessing array {run_id+1} in list")
                        
                        # override use_bids. Can't be use with numpy arrays
                        self.use_bids = False
                    else:
                        raise ValueError(f"Unknown input type '{type(func)}'. Must be string or numpy-array")

                if self.use_bids:
                    bids_comps = utils.split_bids_components(func)
                    for el in ['sub', 'run', 'ses']:
                        setattr(self, el, bids_comps[el])
                else:
                    self.run = run_id+1
                    self.ses = None
                
                # make LSprep output directory
                if self.report:
                    if self.save_as == None:
                        try:
                            self.lsprep_dir = opj(os.environ.get("DIR_DATA_DERIV"), 'lsprep')
                        except:
                            raise ValueError(f"Please specify an output directory with 'save_as='")
                    else:
                        self.lsprep_dir = save_as

                    if self.ses != None:
                        self.lsprep_full = opj(self.lsprep_dir, f'sub-{self.sub}', f'ses-{self.ses}')
                    else:
                        self.lsprep_full = opj(self.lsprep_dir, f'sub-{self.sub}')
                        
                    # make figure directory
                    self.run_uuid = f"{strftime('%Y%m%d-%H%M%S')}_{uuid4()}"
                    self.lsprep_figures = opj(self.lsprep_dir, f'sub-{self.sub}', 'figures')
                    self.lsprep_runid = opj(self.lsprep_dir, f'sub-{self.sub}', 'log', self.run_uuid)
                    self.lsprep_logs = opj(self.lsprep_dir, 'logs')
                    for dir in self.lsprep_full, self.lsprep_figures, self.lsprep_logs, self.lsprep_runid:
                        if not os.path.exists(dir):
                            os.makedirs(dir, exist_ok=True)

                # check if deleted_first_timepoints is list or not
                delete_first = check_input_is_list(
                    self, 
                    var="deleted_first_timepoints", 
                    list_element=run_id,
                    matcher="func_file")

                # check if deleted_last_timepoints is list or not
                delete_last = check_input_is_list(
                    self, 
                    var="deleted_last_timepoints", 
                    list_element=run_id,
                    matcher="func_file")

                # check if baseline is list or not
                baseline = check_input_is_list(
                    self, 
                    var="baseline", 
                    list_element=run_id,
                    matcher="func_file")

                # check if shift is list or not
                shift = check_input_is_list(
                    self, 
                    var="shift", 
                    list_element=run_id,
                    matcher="func_file")                    

                if self.acompcor:
                    if len(self.ref_slice) > 1:
                        ref_slice = self.ref_slice[run_id]
                    else:
                        ref_slice = self.ref_slice[0]
                else:
                    ref_slice = None

                if self.verbose:
                    print(f" Filtering strategy: '{self.filter_strategy}'")
                    print(f" Standardization strategy: '{self.standardization}'")

                self.preprocess_func_file(
                    func, 
                    run=self.run, 
                    deleted_first_timepoints=delete_first,
                    deleted_last_timepoints=delete_last,
                    acompcor=self.acompcor,
                    reference_slice=ref_slice,
                    baseline=baseline,
                    shift=shift,
                    **kwargs)
                
                if self.standardization == "psc":
                    self.df_psc.append(
                        self.get_data(
                            index=False, 
                            filter_strategy=self.filter_strategy, 
                            dtype='psc', 
                            acompcor=self.acompcor, 
                            ica=self.ica))

                elif self.standardization == "zscore":
                    if not self.acompcor:
                        self.df_zscore.append(
                            self.get_data(
                                index=False, 
                                filter_strategy=self.filter_strategy, 
                                dtype='zscore',
                                acompcor=False, 
                                ica=False))

                self.df_raw.append(
                    self.get_data(
                        index=False, 
                        filter_strategy=None, 
                        dtype='raw'))

                if self.retroicor:
                    self.df_retro.append(self.get_retroicor(index=False))
                    self.df_r2.append(self.r2_physio_df)

                if self.acompcor:
                    acomp_data = self.get_data(
                        index=False, 
                        filter_strategy=self.filter_strategy, 
                        dtype=self.standardization,
                        acompcor=True,
                        ica=False)

                    self.df_acomp.append(acomp_data)
                    
                    # append the linescanning.preproc.aCompCor object in case we have multiple runs
                    self.acomp_objs.append(self.acomp)

                    # select GM-voxels based on segmentations in case we have single run
                    self.select_gm_voxels = [ii for ii in self.acomp.gm_voxels if ii in range(*self.gm_range)]
                    self.gm_per_run.append(self.select_gm_voxels)
                    
                    # fetch the data
                    self.df_gm_only.append(utils.select_from_df(acomp_data, expression='ribbon', indices=self.select_gm_voxels))

                if self.ica:
                    self.df_ica.append(
                        self.get_data(
                            index=False, 
                            filter_strategy=self.filter_strategy, 
                            dtype=self.standardization, 
                            acompcor=False, 
                            ica=True))

                    self.ica_objs.append(self.ica_obj)

            # check for standardization method
            if self.standardization == "psc":
                self.df_func_psc = pd.concat(self.df_psc)
            elif self.standardization == "zscore":
                if not self.acompcor:
                    self.df_func_zscore = pd.concat(self.df_zscore)

            # we'll always have raw data
            self.df_func_raw = pd.concat(self.df_raw)

            if self.retroicor:
                try:
                    self.df_func_retroicor = pd.concat(self.df_retro).set_index(['subject', 'run', 't'])
                    self.df_physio_r2 = pd.concat(self.df_r2)
                except:
                    raise ValueError("RETROICOR did not complete successfully..")

            if self.acompcor:           
                
                # check if elements of list contain dataframes
                if all(elem is None for elem in self.df_acomp):
                    print("WARNING: aCompCor did not execute properly. All runs have 'None'")
                else:
                    try:
                        self.df_func_acomp = pd.concat(self.df_acomp).set_index(['subject', 'run', 't'])
                    except:
                        self.df_func_acomp = pd.concat(self.df_acomp)

                # decide on GM-voxels across runs by averaging tissue probabilities
                if len(self.acomp_objs) > 1:
                    self.select_voxels_across_runs()
                    self.gm_df = utils.select_from_df(
                        self.df_func_acomp, 
                        expression='ribbon', 
                        indices=self.voxel_classification['gm'])
                    
                    self.ribbon_voxels = [ii for ii in range(*self.gm_range) if ii in self.voxel_classification['gm']]
                    self.ribbon_df = utils.select_from_df(
                        self.df_func_acomp, 
                        expression='ribbon', 
                        indices=self.ribbon_voxels)
                else:
                    self.gm_df = self.df_gm_only[0].copy()

            if self.ica:           
                
                # check if elements of list contain dataframes
                if all(elem is None for elem in self.df_ica):
                    print("WARNING: ICA did not execute properly. All runs have 'None'")
                else:
                    try:
                        self.df_func_ica = pd.concat(self.df_ica).set_index(['subject', 'run', 't'])
                    except:
                        self.df_func_ica = pd.concat(self.df_ica)                    

        # now that we have nicely formatted functional data, initialize the ParseExpToolsFile-class
        if self.tsv_file != None: 
            ParseExpToolsFile.__init__(
                self,
                self.tsv_file, 
                subject=self.sub, 
                deleted_first_timepoints=self.deleted_first_timepoints, 
                TR=self.TR, 
                edfs=self.edf_file, 
                funcs=self.func_file, 
                use_bids=self.use_bids,
                button=self.button,
                verbose=self.verbose,
                **kwargs)

        # write boilerplate
        if self.report:
            self.citation_file = Path(opj(self.lsprep_logs, "CITATION.md"))
            self.citation_file.write_text(self.desc_func)

            # write report
            self.config = str(Path(utils.__file__).parents[1]/'misc'/'default.yml')
            if not os.path.exists(self.config):
                raise FileNotFoundError(f"Could not find 'default.yml'-file in '{str(Path(utils.__file__).parents[1]/'misc')}'")
            
            if self.report:
                self.report_obj = core.Report(
                    os.path.dirname(self.lsprep_dir),
                    self.run_uuid,
                    subject_id=self.sub,
                    packagename="lsprep",
                    config=self.config)

                # generate report
                self.report_obj.generate_report()

                if self.verbose:
                    print(f" Saving report to {str(self.report_obj.out_dir/self.report_obj.out_filename)}")                                      

    def preprocess_func_file(
        self, 
        func_file, 
        run=1, 
        deleted_first_timepoints=0, 
        deleted_last_timepoints=0,
        acompcor=False,
        reference_slice=None,
        baseline=None,
        shift=0,
        **kwargs):

        #----------------------------------------------------------------------------------------------------------------------------------------------------
        # BASIC DATA LOADING

        # Load in datasets with tag "wcsmtSNR"
        if isinstance(func_file, str):
            if func_file.endswith("mat"):

                # load matlab file
                self.ts_wcsmtSNR = io.loadmat(func_file)

                # decide which key to read from the .mat file
                if self.func_tag == None:
                    self.tag = list(self.ts_wcsmtSNR.keys())[-1]
                else:
                    self.tag = self.func_tag

                # select data
                self.ts_wcsmtSNR    = self.ts_wcsmtSNR[self.tag]
                self.ts_complex     = self.ts_wcsmtSNR
                self.ts_magnitude   = np.abs(self.ts_wcsmtSNR)

            elif func_file.endswith('gii'):
                self.gif_obj = ParseGiftiFile(func_file)

                self.ts_magnitude = self.gif_obj.data
                if not self.transpose:
                    self.ts_magnitude = self.gif_obj.data.T

            elif func_file.endswith("npy"):
                self.ts_magnitude = np.load(func_file)

                if self.transpose:
                    self.ts_magnitude = np.load(func_file).T

            elif func_file.endswith("nii") or func_file.endswith("gz"):
                fdata = nb.load(func_file).get_fdata()

                # cifti nii's are already 2D
                if fdata.ndim > 2:
                    xdim,ydim,zdim,time_points = fdata.shape
                    self.ts_magnitude = fdata.reshape(xdim*ydim*zdim, time_points)
                else:
                    self.ts_magnitude = fdata.copy().T

        elif isinstance(func_file, np.ndarray):
            self.ts_magnitude = func_file.copy()
            
        else:
            raise NotImplementedError(f"Input type {type(func_file)} not supported")

        # check baseline
        if not self.psc_nilearn:
            if self.baseline_units == "seconds" or self.baseline_units == "s" or self.baseline_units == "sec":
                baseline_vols_old = int(np.round(baseline*self.fs, 0))
                if self.verbose:
                    print(
                        f" Baseline is {baseline} seconds, or {baseline_vols_old} TRs")
            else:
                baseline_vols_old = baseline
                if self.verbose:
                    print(f" Baseline is {baseline} TRs")

            # correct for deleted samples
            baseline_vols = baseline_vols_old-self.deleted_first_timepoints
            txt = f" (also cut from baseline (was {baseline_vols_old}, now {baseline_vols} TRs)"
        else:
            txt = ""
            baseline_vols = 0

        # trim beginning and end
        if deleted_last_timepoints != 0:
            self.desc_trim = f""" {deleted_first_timepoints} were removed from the beginning of the functional data."""
            self.ts_corrected = self.ts_magnitude[:,deleted_first_timepoints:-deleted_last_timepoints]
        else:
            self.desc_trim = ""
            self.ts_corrected = self.ts_magnitude[:,deleted_first_timepoints:]

        if self.verbose:
            print(f" Cutting {deleted_first_timepoints} volumes from beginning{txt} | {deleted_last_timepoints} volumes from end")

        self.vox_cols = [f'vox {x}' for x in range(self.ts_corrected.shape[0])]

        #----------------------------------------------------------------------------------------------------------------------------------------------------
        # STANDARDIZATION OF UNFILTERED DATA & CREATE DATAFRAMES

        # dataframe of raw, unfiltered data
        self.data_raw = self.ts_corrected.copy()
        self.data_raw_df = self.index_func(
            self.data_raw, 
            columns=self.vox_cols, 
            subject=self.sub, 
            run=run, 
            TR=self.TR,
            set_index=True)

        # dataframe of unfiltered PSC-data
        self.data_psc = utils.percent_change(
            self.data_raw, 
            1, 
            baseline=baseline_vols,
            nilearn=self.psc_nilearn)

        self.data_psc_df = self.index_func(
            self.data_psc,
            columns=self.vox_cols, 
            subject=self.sub,
            run=run, 
            TR=self.TR, 
            set_index=True)

        # dataframe of unfiltered z-scored data
        self.data_zscore = _standardize(self.data_raw.T, standardize='zscore').T
        self.data_zscore_df = self.index_func(
            self.data_zscore,
            columns=self.vox_cols, 
            subject=self.sub, 
            run=run, 
            TR=self.TR,
            set_index=True)

        #----------------------------------------------------------------------------------------------------------------------------------------------------
        # HIGH PASS FILTER
        self.clean_tag = None
        if self.filter_strategy != "raw":

            self.desc_filt = f"""DCT-high pass filter [removes low frequencies <{self.lb} Hz] was applied. """

            if self.verbose:
                print(f" DCT-high pass filter [removes low frequencies <{self.lb} Hz] to correct low-frequency drifts.")

            self.hp_raw, self._cosine_drift = preproc.highpass_dct(self.data_raw, self.lb, TR=self.TR)
            self.hp_raw_df = self.index_func(
                self.hp_raw,
                columns=self.vox_cols, 
                subject=self.sub, 
                run=run, 
                TR=self.TR,
                set_index=True)

            # dataframe of high-passed PSC-data (set NaN to 0)
            self.hp_psc = np.nan_to_num(utils.percent_change(
                self.hp_raw,
                1, 
                baseline=baseline_vols,
                nilearn=self.psc_nilearn))
                
            self.hp_psc_df = self.index_func(
                self.hp_psc,
                columns=self.vox_cols, 
                subject=self.sub,
                run=run, 
                TR=self.TR, 
                set_index=True)

            # dataframe of high-passed z-scored data
            self.hp_zscore = _standardize(self.hp_raw.T, standardize='zscore').T
            self.hp_zscore_df = self.index_func(
                self.hp_zscore,
                columns=self.vox_cols, 
                subject=self.sub, 
                run=run, 
                TR=self.TR,
                set_index=True)

            # save SD and Mean so we can go from zscore back to original
            self.zscore_SD = self.hp_raw.std(axis=-1, keepdims=True)
            self.zscore_M = self.hp_raw.mean(axis=-1, keepdims=True)

            # don't save figures if report=False
            if self.report:
                save_as = self.lsprep_figures
            else:
                save_as = None

            #----------------------------------------------------------------------------------------------------------------------------------------------------
            # ACOMPCOR AFTER HIGH-PASS FILTERING
            if acompcor:

                self.desc_filt += f"""Data was then z-scored and fed into a custom implementation of `aCompCor` 
(https://github.com/gjheij/linescanning/blob/main/linescanning/preproc.py), which is tailored for line-scanning data: """

                # do some checks beforehand
                if reference_slice != None:
                    if self.use_bids:
                        bids_comps = utils.split_bids_components(reference_slice)
                        setattr(self, "target_session", bids_comps['ses'])
                        setattr(self, "subject", f"sub-{bids_comps['sub']}")
                    else:
                        assert hasattr(self, "target_session"), f"Please specify a target_session with 'target_session=<int>'"
                        assert hasattr(self, "subject"), f"Please specify a subject with 'target_session=<int>'"

                # check the transformations inputs
                assert hasattr(self, "ses1_2_ls"), f"Please specify a transformation matrix mapping FreeSurfer to ses-{self.target_session}"

                if hasattr(self, "run_2_run"):
                    if isinstance(self.run_2_run, list):
                        run_trafo =  utils.get_file_from_substring(f"to-run{self.run}", self.run_2_run)
                        self.trafos = [self.ses1_2_ls, run_trafo]
                    else:
                        if self.run_2_run != None:
                            self.trafos = [self.ses1_2_ls, self.run_2_run]
                        else:
                            self.trafos = [self.ses1_2_ls]
                else:
                    self.trafos = self.ses1_2_ls            

                # aCompCor implemented in `preproc` module
                self.acomp = preproc.aCompCor(
                    self.hp_zscore_df,
                    subject=f"sub-{self.sub}",
                    run=self.run,
                    trg_session=self.target_session,
                    reference_slice=reference_slice,
                    trafo_list=self.trafos,
                    n_components=self.n_components,
                    filter_confs=self.filter_confs,
                    save_as=save_as,
                    select_component=self.select_component, 
                    summary_plot=self.report,
                    TR=self.TR,
                    foldover=self.foldover,
                    shift=shift,
                    verbose=self.verbose,
                    save_ext=self.save_ext,
                    **kwargs)
                
                self.clean_tag = "acompcor"
                self.clean_data = self.acomp.acomp_data

                if self.ica:
                    raise TypeError("ICA cannot be used in conjunction with aCompCor. Please set 'ica=False'")

            if self.ica:
                if acompcor:
                    raise TypeError("aCompCor cannot be used in conjunction with ICA. Please set 'acompcor=False'")

                if self.verbose:
                    print(f" Running FastICA with {self.n_components} components")

                self.ica_obj = preproc.ICA(
                    self.hp_zscore_df,
                    subject=f"sub-{self.sub}",
                    run=self.run,
                    n_components=self.n_components,
                    TR=self.TR,
                    filter_confs=self.filter_confs,
                    keep_comps=self.keep_comps,
                    verbose=self.verbose,
                    summary_plot=self.report,
                    melodic_plot=self.report,
                    ribbon=tuple(self.gm_range),
                    save_as=save_as,
                    save_ext=self.save_ext
                )

                # regress
                self.ica_obj.regress()

                self.clean_tag = "ica"
                self.clean_data = self.ica_obj.ica_data

            if hasattr(self, "clean_data"):
                self.tmp_df = self.index_func(
                    self.clean_data,
                    columns=self.vox_cols, 
                    subject=self.sub, 
                    run=run, 
                    TR=self.TR,
                    set_index=True)

                setattr(self, f"hp_{self.clean_tag}_df", self.tmp_df)
                
                # multiply by SD and add mean
                self.tmp_raw = (self.clean_data * self.zscore_SD) + self.zscore_M
                setattr(self, f"hp_{self.clean_tag}_raw", self.tmp_raw)

                self.tmp_raw_df = self.index_func(
                    self.tmp_raw,
                    columns=self.vox_cols, 
                    subject=self.sub, 
                    run=run, 
                    TR=self.TR,
                    set_index=True)

                setattr(self, f"hp_{self.clean_tag}_raw_df", self.tmp_raw)

                # make percent signal
                self.hp_tmp_psc = np.nan_to_num(utils.percent_change(
                    self.tmp_raw,
                    1, 
                    baseline=baseline_vols,
                    nilearn=self.psc_nilearn))

                setattr(self, f"hp_{self.clean_tag}_psc", self.hp_tmp_psc)

                self.hp_tmp_psc_df = self.index_func(
                    self.hp_tmp_psc,
                    columns=self.vox_cols, 
                    subject=self.sub, 
                    run=run, 
                    TR=self.TR,
                    set_index=True)            
                
                setattr(self, f"hp_{self.clean_tag}_psc_df", self.hp_tmp_psc_df)

                if self.clean_tag == "acompcor":
                    self.desc_filt += self.acomp.__desc__
                elif self.clean_tag == "ica":
                    self.desc_filt += self.ica_obj.__desc__

                self.desc_filt += f"""
Output from {self.clean_tag} was then converted back to un-zscored data by multipying by the standard deviation and adding the mean back. """

            #----------------------------------------------------------------------------------------------------------------------------------------------------
            # LOW PASS FILTER
            if "lp" in self.filter_strategy:

                self.desc_filt += f"""
The data was then low-pass filtered using a Savitsky-Golay filter [removes high frequences] (window={self.window_size}, 
order={self.poly_order}). """

                if acompcor or self.ica:
                    info = f" Using {self.clean_tag}-data for low-pass filtering"
                    data_for_filtering = self.get_data(index=True, filter_strategy="hp", dtype=self.standardization, acompcor=acompcor, ica=self.ica).T.values
                    out_attr = f"lp_{self.clean_tag}_{self.standardization}"
                elif hasattr(self, f"hp_{self.standardization}"):
                    info = " Using high-pass filtered data for low-pass filtering"
                    data_for_filtering = getattr(self, f"hp_{self.standardization}")
                    out_attr = f"lp_{self.standardization}"
                else:
                    info = " Using unfiltered/un-aCompCor'ed data for low-pass filtering"
                    data_for_filtering = getattr(self, f"data_{self.standardization}")
                    out_attr = f"lp_data_{self.standardization}"

                if self.verbose:
                    print(info)
                    print(f" Savitsky-Golay low-pass filter [removes high frequences] (window={self.window_size}, order={self.poly_order})")

                tmp_filtered = preproc.lowpass_savgol(data_for_filtering, window_length=self.window_size, polyorder=self.poly_order)

                tmp_filtered_df = self.index_func(
                    tmp_filtered,
                    columns=self.vox_cols,
                    subject=self.sub,
                    run=run,
                    TR=self.TR,
                    set_index=True)

                setattr(self, out_attr, tmp_filtered.copy())
                setattr(self, f'{out_attr}_df', tmp_filtered_df.copy())

        else:
            self.desc_filt = ""
            self.clean_tag = None

        # get basic qualities
        self.basic_qa(
            self.ts_corrected, 
            run=run, 
            make_figure=True)

        # final
        self.desc_func = self.func_pre_desc + self.desc_trim + self.desc_filt
    
    def basic_qa(self, data, run=1, make_figure=False):
        
        # tsnr
        tsnr_pre = utils.calculate_tsnr(data, -1)
        mean_tsnr_pre = float(np.nanmean(np.ravel(tsnr_pre)))

        # variance
        var_pre = np.var(data, axis=-1)
        mean_var_pre = float(var_pre.mean())

        tsnr_inputs = [tsnr_pre]
        var_inputs = [var_pre]
        colors = "#1B9E77"
        tsnr_lbl = None
        var_lbl = None
        tsnr_lines = {
            "pos": mean_tsnr_pre,
            "color": colors
        }

        var_lines = {
            "pos": mean_var_pre,
            "color": colors
        }
        if self.verbose:
            if self.clean_tag == None:
                info = "no cleaning"
            else:
                info = f"before '{self.clean_tag}'"

            print(f" tSNR [{info}]: {round(mean_tsnr_pre,2)}\t| variance: {round(mean_var_pre,2)}")

        if self.clean_tag == "acompcor" or self.clean_tag == "ica":

            # get aCompCor/ICA'ed tSNR
            tsnr_post = utils.calculate_tsnr(getattr(self, f"hp_{self.clean_tag}_raw"),-1)
            mean_tsnr_post = float(np.nanmean(np.ravel(tsnr_post)))

            # variance
            var_post = np.var(getattr(self, f"hp_{self.clean_tag}_raw"), axis=-1)
            mean_var_post = float(var_post.mean())

            # sort out plotting stuff
            tsnr_inputs += [tsnr_post]
            var_inputs += [var_post]
            colors = [colors, "#D95F02"]
            tsnr_lbl = [f'no {self.clean_tag} [{round(mean_tsnr_pre,2)}]', f'{self.clean_tag} [{round(mean_tsnr_post,2)}]']
            var_lbl = [f'no {self.clean_tag} [{round(mean_var_pre,2)}]', f'{self.clean_tag} [{round(mean_var_post,2)}]']            
            tsnr_lines = {
                "pos": [mean_tsnr_pre,mean_tsnr_post],
                "color": colors
            }

            var_lines = {
                "pos": [mean_var_pre,mean_var_post],
                "color": colors
            }

            if self.verbose:
                print(f" tSNR [after '{self.clean_tag}']:  {round(mean_tsnr_post,2)}\t| variance: {round(mean_var_post,2)}")

        if make_figure:
            # initiate figure
            fig = plt.figure(figsize=(24,7))
            gs = fig.add_gridspec(1,3, width_ratios=[10,10,10])

            # imshow for apparent motion
            ax1 = fig.add_subplot(gs[0])
            ax1.imshow(np.rot90(data.T), aspect=8/1)
            ax1.set_xlabel("volumes", fontsize=16)
            ax1.set_ylabel("voxels", fontsize=16)
            ax1.set_title("Stability of position", fontsize=16)
            
            defs = plotting.Defaults()
            ax1.tick_params(
                width=defs.tick_width, 
                length=defs.tick_length,
                labelsize=defs.label_size)

            vox_ticks = [0,data.shape[0]//2,data.shape[0]]
            ax1.set_xticks([0,data.shape[-1]])
            ax1.set_yticks(vox_ticks)

            for axis in ['top', 'bottom', 'left', 'right']:
                ax1.spines[axis].set_linewidth(0.5)

            # line plot for tSNR over voxels
            ax2 = fig.add_subplot(gs[1])
            plotting.LazyPlot(
                tsnr_inputs,
                axs=ax2,
                title=f"tSNR across the line",
                font_size=16,
                linewidth=2,
                color=colors,
                x_label="voxels",
                y_label="tSNR (a.u.)",
                labels=tsnr_lbl,
                add_hline=tsnr_lines,
                x_ticks=vox_ticks,
                line_width=2)

            ax3 = fig.add_subplot(gs[2])
            plotting.LazyPlot(
                var_inputs,
                axs=ax3,
                title=f"Variance across the line",
                font_size=16,
                linewidth=2,
                color=colors,
                x_label="voxels",
                y_label="Variance",
                labels=var_lbl,
                add_hline=var_lines,
                x_ticks=vox_ticks,
                line_width=2)            

            plt.close(fig)
            if self.report:
                fname = opj(self.lsprep_figures, f"sub-{self.sub}_ses-{self.ses}_run-{run}_desc-qa.{self.save_ext}")
                fig.savefig(fname, bbox_inches='tight', dpi=300)
                
    def select_voxels_across_runs(self):

        fig = plt.figure(figsize=(24,12))
        gs = fig.add_gridspec(3,1, hspace=0.6)

        self.voxel_classification = {}
        self.tissue_probabilities = {}

        # get the nice blue from CSF-regressor
        if hasattr(self.acomp_objs[0], "regressor_voxel_colors"):
            use_color = self.acomp_objs[0].regressor_voxel_colors[0]
        else:
            use_color = "#0062C7"

        for ix, (t_type, seg) in enumerate(zip(["white matter","csf","gray matter"], ["wm","csf","gm"])):
            
            self.tissue_probabilities[seg] = []
            for compcor in self.acomp_objs:
                compcor.segmentations_to_beam()
                self.tissue_probabilities[seg].append(
                    compcor.segmentations_in_beam[compcor.subject][seg][..., np.newaxis])

            img         = np.concatenate((self.tissue_probabilities[seg]), axis=-1)
            avg_runs    = img.mean(axis=-1).mean(axis=-1)
            # avg_err     = stats.sem(avg_runs, axis=-1)
            avg_err     = img.mean(axis=-1).std(axis=-1)

            # add indication for new classification
            self.voxel_classification[seg] = np.where(avg_runs > self.tissue_thresholds[ix])[0]

            if self.verbose:
                ax = fig.add_subplot(gs[ix])
                add_hline = {
                    'pos': self.tissue_thresholds[ix], 
                    'color': 'k', 
                    'ls': '--', 
                    'lw': 1}

                for ii in self.voxel_classification[seg]:
                    ax.axvline(ii, alpha=0.3, color="#cccccc")

                plotting.LazyPlot(
                    avg_runs,
                    axs=ax,
                    error=avg_err,
                    title=f"tissue probabilities '{t_type}'",
                    color=use_color,
                    line_width=2,
                    y_ticks=[0,1],
                    x_ticks=[0,avg_runs.shape[0]//2,avg_runs.shape[0]],
                    x_lim=[0,avg_runs.shape[0]],
                    add_hline=add_hline)

        if self.report:
            fname = opj(self.lsprep_figures, f"sub-{self.sub}_ses-{self.ses}_desc-tissue_classification.{self.save_ext}")
            fig.savefig(fname, bbox_inches='tight', dpi=300)

    def get_data(self, filter_strategy=None, index=False, dtype="psc", acompcor=False, ica=False):

        if dtype != "psc" and dtype != "zscore" and dtype != "raw":
            raise ValueError(f"Requested data type '{dtype}' is not supported. Use 'psc', 'zscore', or 'raw'")

        return_data = None
        allowed = [None, "raw", "hp", "lp"]

        if acompcor or ica:
            tag = f"{self.clean_tag}_{dtype}"
        else:
            tag = dtype

        if filter_strategy == None or filter_strategy == "raw":
            attr = f"data_{tag}_df"
        elif filter_strategy == "lp":
            attr = f"lp_{tag}_df"
        elif filter_strategy == "hp":
            attr = f"hp_{tag}_df"
        else:
            raise ValueError(f"Unknown attribute '{filter_strategy}'. Must be one of: {allowed}")

        if hasattr(self, attr):
            # print(f" Fetching attribute: {attr}")
            return_data = getattr(self, attr)
        else:
            raise ValueError(f"{self} does not have an attribute called '{attr}'")

        if isinstance(return_data, pd.DataFrame):
            if index:
                try:
                    return return_data.set_index(['subject', 'run', 't'])
                except:
                    return return_data
            else:
                return return_data

    @staticmethod
    def index_func(array, columns=None, subject=1, run=1, TR=0.105, set_index=False):
    
        if columns == None:
            df = pd.DataFrame(array.T)
        else:
            df = pd.DataFrame(array.T, columns=columns)
            
        df['subject']   = subject
        df['run']       = run
        df['t']         = list(TR*np.arange(df.shape[0]))

        if set_index:
            return df.set_index(['subject', 'run', 't'])
        else:
            return df

class Dataset(ParseFuncFile):
    """Dataset

    Main class for retrieving, formatting, and preprocessing of all datatypes including fMRI (2D), eyetracker (*.edf), physiology (*.log [WIP]), and experiment files derived from `Exptools2` (*.tsv). If you leave `subject` and `run` empty, these elements will be derived from the file names. So if you have BIDS-like files, leave them empty and the dataframe will be created for you with the correct subject/run IDs. 

    Inherits from :class:`linescanning.dataset.ParseFuncFile`, so all arguments from that class are available and are passed on via `**kwargs`.	Only `func_file` and `verbose` are required. The first one is necessary because if the input is an **h5**-file, we'll set the attributes accordingly. Otherwise :class:`linescanning.dataset.ParseFuncFile` is invoked. `verbose` is required for aesthetic reasons. Given that :class:`linescanning.dataset.ParseFuncFile` inherits in turn from :class:`linescanning.dataset.ParseExpToolsFile`, you can pass the arguments for that class here as well.
    
    Parameters
    ----------
    func_file: str, list
        path or list of paths pointing to the output file of the experiment
    verbose: bool, optional
        Print details to the terminal, default is False

    Example
    ----------
    >>> from linescanning import dataset, utils
    >>> func_dir = "/some/dir"
    >>> exp     = utils.get_file_from_substring("tsv", func_dir)
    >>> funcs   = utils.get_file_from_substring("bold.mat", func_dir)
    >>> # 
    >>> # only cut from SR-runs
    >>> delete_first = 100
    >>> delete_last = 0
    >>> #
    >>> window = 19
    >>> order = 3
    >>> data = dataset.Dataset(
    >>>     funcs,
    >>>     deleted_first_timepoints=delete_first,
    >>>     deleted_last_timepoints=delete_last,
    >>>     tsv_file=exp,
    >>>     verbose=True)
    >>> #
    >>> # retrieve data
    >>> fmri = data.fetch_fmri()
    >>> onsets = data.fetch_onsets()
    """

    def __init__(
        self, 
        func_file,  
        verbose=False,         
        **kwargs):

        self.verbose = verbose
        if self.verbose:
            print("DATASET")
        
        self.read_attributes = [
            'df_func_psc', 
            'df_func_raw', 
            'df_retro_zscore', 
            'df_onsets', 
            'df_space_func', 
            'blink_events']

        if isinstance(func_file, str) and func_file.endswith(".h5"):
            if self.verbose:
                print(f" Reading from {func_file}")
            self.from_hdf(func_file)
        else:
            super().__init__(func_file, verbose=self.verbose, **kwargs)

        if self.verbose:
            print("\nDATASET: created")

    def fetch_fmri(self, strip_index=False, dtype=None):

        if dtype == None:
            if self.acompcor:
                dtype = "acompcor"
            else:
                if self.ica:
                    dtype = "ica"
                else:
                    if hasattr(self, "standardization"):
                        dtype = self.standardization
                    else:
                        dtype = "psc"
        
        if dtype == "psc":
            attr = 'df_func_psc'
        elif dtype == "retroicor":
            attr = 'df_func_retroicor'
        elif dtype == "raw" or dtype == None:
            attr = 'df_func_raw'
        elif dtype == "zscore":
            attr = 'df_func_zscore'
        elif dtype == "acompcor":
            attr = 'df_func_acomp'
        elif dtype == "ica":
            attr = 'df_func_ica'
        else:
            raise ValueError(f"Unknown option '{dtype}'. Must be 'psc', 'retroicor', 'acompcor', or 'zscore'")

        if hasattr(self, attr):
            
            if self.verbose:
                print(f"Fetching dataframe from attribute '{attr}'")
                
            df = getattr(self, attr)
            if strip_index:
                return df.reset_index().drop(labels=['subject', 'run', 't'], axis=1) 
            else:
                return df
        else:
            print(f"Could not find '{attr}' attribute")
            
    def fetch_onsets(self, strip_index=False):
        if hasattr(self, 'df_onsets'):
            if strip_index:
                return self.df_onsets.reset_index().drop(labels=list(self.df_onsets.index.names), axis=1)
            else:
                return self.df_onsets
        else:
            print("No event-data was provided")

    def fetch_rts(self, strip_index=False):
        if hasattr(self, 'df_rts'):
            if strip_index:
                return self.df_rts.reset_index().drop(labels=list(self.df_rts.index.names), axis=1)
            else:
                return self.df_rts
        else:
            print("No reaction times were provided")            

    def fetch_accuracy(self, strip_index=False):
        if hasattr(self, 'df_accuracy'):
            if strip_index:
                return self.df_accuracy.reset_index().drop(labels=list(self.df_accuracy.index.names), axis=1)
            else:
                return self.df_accuracy
        else:
            print("No accuracy measurements found")                 

    def fetch_physio(self, strip_index=False):
        if hasattr(self, 'df_physio'):
            if strip_index:
                return self.df_physio.reset_index().drop(labels=list(self.df_physio.index.names), axis=1)
            else:
                return self.df_physio
        else:
            print("No physio-data was provided")            

    def fetch_trace(self, strip_index=False):
        if hasattr(self, 'df_space_func'):
            if strip_index:
                return self.df_space_func.reset_index().drop(labels=list(self.df_space_func.index.names), axis=1)
            else:
                return self.df_space_func
        else:
            print("No eyetracking-data was provided")

    def fetch_blinks(self, strip_index=False):
        if hasattr(self, 'blink_events'):
            return self.blink_events
        else:
            print("No eyetracking-data was provided")

    def to_hdf(self, output_file=None, overwrite=False):

        if output_file == None:
            if hasattr(self, "lsprep_full"):
                self.h5_file = opj(self.lsprep_full, "dataset.h5")
            else:
                raise ValueError("No output file specified")
        else:
            self.h5_file = output_file

        if overwrite:
            if os.path.exists(self.h5_file):
                store = pd.HDFStore(self.h5_file)
                store.close()
                os.remove(self.h5_file)

        if self.verbose:
            print(f"Saving to {self.h5_file}")

        for attr in self.read_attributes:
            if hasattr(self, attr):
                
                if self.verbose:
                    print(f" Storing attribute: {attr}")
                    
                add_df = getattr(self, attr)
                if os.path.exists(self.h5_file):
                    add_df.to_hdf(self.h5_file, key=attr, append=True, mode='r+', format='t')
                else:
                    store = pd.HDFStore(self.h5_file)
                    store.close()
                    add_df.to_hdf(self.h5_file, key=attr, mode='w', format='t')
        
        if self.verbose:
            print("Done")

        store = pd.HDFStore(self.h5_file)
        store.close()

    def from_hdf(self, input_file=None):

        if input_file == None:
            if hasattr(self, "lsprep_full"):
                self.h5_file = opj(self.lsprep_full, "dataset.h5")
            else:
                raise ValueError("No output file specified")
        else:
            self.h5_file = input_file

        hdf_store = pd.HDFStore(self.h5_file)
        hdf_keys = hdf_store.keys()
        for key in hdf_keys:
            key = key.strip("/")
            
            if self.verbose:
                print(f" Setting attribute: {key}")

            setattr(self, key, hdf_store.get(key))

        hdf_store.close()         

    def to4D(self, fname=None, desc=None, dtype=None, mask=None):

        # get dataset
        df = self.fetch_fmri(dtype=dtype)

        subj_list = self.get_subjects(df)
        file_counter = 0
        for sub in subj_list:
            
            # get subject-specific data
            data_per_subj = utils.select_from_df(df, expression=f"subject = {sub}")

            # get run IDs
            n_runs = self.get_runs(df)
            for run in n_runs:

                # get run-specific data
                data_per_run = utils.select_from_df(data_per_subj, expression=f"run = {run}")

                # get corresponding reference image from self.func_file either based on index (if use_bids=False), or based on BIDS-elements (use_bids=True)
                if self.use_bids:
                    ref_img = utils.get_file_from_substring([f'sub-{sub}', f'run-{run}'], self.func_file)
                else:
                    ref_img = self.func_file[file_counter]
                
                if self.verbose:
                    print(f"Ref img = {ref_img}")

                if isinstance(ref_img, nb.Nifti1Image):
                    ref_img = ref_img
                elif isinstance(ref_img, str):
                    if ref_img.endswith("gz") or ref_img.endswith("nii"):
                        ref_img = nb.load(ref_img)
                    else:
                        raise TypeError(f"Unknown reference type '{ref_img}'. Must be a string pointing to 'nii' or 'nii.gz' file")
                else:
                    raise ValueError("'ref_img' must either be string pointing to nifti image or a nb.Nifti1Image object")
                
                # get information of reference image
                dims = ref_img.get_fdata().shape
                aff = ref_img.affine
                hdr = ref_img.header

                if self.verbose:
                    print(f"Ref shape = {dims}")
                
                data_per_run = data_per_run.values
                # time is initially first axis, so transpose
                if data_per_run.shape[-1] != dims[-1]:
                    if self.verbose:
                        print(f"Data shape = {data_per_run.shape}; transposing..")
                    data_per_run = data_per_run.T
                else:
                    if self.verbose:
                        print(f"Data shape = {data_per_run.shape}; all good..")

                if self.verbose:
                    print(f"Final shape = {data_per_run.shape}")

                # check if we have mask
                if isinstance(mask, nb.Nifti1Image) or isinstance(mask, str) or isinstance(mask, list):
                    
                    if self.verbose:
                        print("Masking with given mask-object")

                    if isinstance(mask, nb.Nifti1Image):
                        mask = mask
                    elif isinstance(mask, str):
                        if mask.endswith("gz") or mask.endswith("nii"):
                            mask = nb.load(mask)
                        else:
                            raise TypeError(f"Unknown reference type '{mask}'. Must be a string pointing to 'nii' or 'nii.gz' file")
                    elif isinstance(mask, list):
                        # select mask based on BIDS-components or index
                        if self.use_bids:
                            mask = utils.get_file_from_substring([f'sub-{sub}', f'run-{run}'], mask)
                        else:
                            mask = mask[file_counter]
                    else:
                        raise TypeError(f"Unknown input '{type(mask)}', must be nibabel.Nifti1Image-object or string pointing to nifti-image")

                    # mask array
                    mask_data = mask.get_fdata()
                    mask_data = mask_data.reshape(np.prod(mask_data.shape))
                    brain_idc = np.where(mask_data > 0)[0]
                    data_masked = np.zeros_like(data_per_run)

                    # fill zeroed array with brandata
                    data_masked[brain_idc,:] = data_per_run[brain_idc,:]

                    # overwrite
                    data_per_run = data_masked.copy()

                # reshape
                data_per_run = data_per_run.reshape(*dims)

                # save
                if not isinstance(fname, str):
                    if isinstance(desc, str):
                        fname = f"sub-{sub}_run-{run}_desc-{desc}.nii.gz"
                    else:
                        fname = f"sub-{sub}_run-{run}.nii.gz"
                else:
                    if isinstance(desc, str):
                        fname = f"{fname}_run-{run}_desc-{desc}.nii.gz"
                    else:
                        fname = f"{fname}_run-{run}.nii.gz"

                print(f"Writing {fname}")
                nb.Nifti1Image(data_per_run, affine=aff, header=hdr).to_filename(fname)

                file_counter += 1

class DatasetCollector():
    def __init__(self, dataset_objects):

        self.datasets = dataset_objects
        if len(self.datasets) != None:
            self.data = []
            self.onsets = []
            for dataset in self.datasets:
                self.data.append(dataset.fetch_fmri())

                # check if we got onsets
                if hasattr(dataset, 'df_onsets'):
                    onsets = True
                    self.onsets.append(dataset.fetch_onsets())
                else:
                    onsets = False

            self.data = pd.concat(self.data)
            if onsets:
                self.onsets = pd.concat(self.onsets)

# this is basically a wrapper around pybest.utils.load_gifti
class ParseGiftiFile():

    def __init__(self, gifti_file, set_tr=None):

        self.gifti_file = gifti_file
        self.f_gif = nb.load(self.gifti_file)
        self.data = np.vstack([arr.data for arr in self.f_gif.darrays])
        self.set_tr = set_tr

        if set_tr != None:
            if len(self.f_gif.darrays[0].metadata) == 0:
                self.f_gif = self.set_metadata()
            elif int(float(self.f_gif.darrays[0].metadata['TimeStep'])) == 0:
                # int(float) construction from https://stackoverflow.com/questions/1841565/valueerror-invalid-literal-for-int-with-base-10
                self.f_gif = self.set_metadata()
            elif int(float(self.f_gif.darrays[0].metadata['TimeStep'])) == set_tr:
                pass
            else:
                raise ValueError("Could not update TR..")
        
        self.meta = self.f_gif.darrays[0].metadata
        self.TR_ms = float(self.meta['TimeStep'])
        self.TR_sec = float(self.meta['TimeStep']) / 1000

    def set_metadata(self):
        
        # define metadata
        image_metadata = nb.gifti.GiftiMetaData().from_dict({'TimeStep': str(float(self.set_tr))})

        # copy old data and combine it with metadata
        darray = nb.gifti.GiftiDataArray(self.data, meta=image_metadata)

        # store in new gifti image object
        gifti_image = nb.GiftiImage()

        # add data to this object
        gifti_image.add_gifti_data_array(darray)
        
        # save in same file name
        nb.save(gifti_image, self.gifti_file)

        return gifti_image
