import hedfpy
from . import glm, utils
import nibabel as nb
from nilearn import signal
from nilearn.glm.first_level.design_matrix import _cosine_drift
from nitime.timeseries import TimeSeries
from nitime.analysis import SpectralAnalyzer
import numpy as np
import os
import pandas as pd
from scipy import io
from scipy import fft
from scipy.signal import savgol_filter
import warnings

opj = os.path.join
pd.options.mode.chained_assignment = None # disable warning thrown by string2float
warnings.filterwarnings("ignore")

def check_input_is_list(obj, var=None, list_element=0):

    if hasattr(obj, var):
        attr = getattr(obj, var)
    else:
        raise ValueError(f"Class does not have '{var}'-attribute")
    
    if isinstance(attr, list) or isinstance(attr, np.ndarray):
        if len(attr) != len(obj.func_file):
            raise ValueError(f"Length of '{var}' ({len(attr)}) does not match number of func files ({len(obj.func_file)}). Either specify a list of equal lenghts or 1 integer value for all volumes")

        return attr[list_element]
    else:
        return attr

class ParseEyetrackerFile():

    """ParseEyetrackerFile()

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

    def __init__(self, 
                 edf_file, 
                 subject=1, 
                 low_pass_pupil_f=6.0, 
                 high_pass_pupil_f=0.01,
                 func_file=None, 
                 TR1=0.105, 
                 TR2=None, 
                 verbose=False, 
                 use_bids=True):

        self.edf_file           = edf_file
        self.func_file          = func_file
        self.sub                = subject
        self.TR1                = TR1
        self.TR2                = TR2
        self.low_pass_pupil_f   = low_pass_pupil_f
        self.high_pass_pupil_f  = high_pass_pupil_f
        self.verbose            = verbose
        self.use_bids           = use_bids
        self.include_blinks     = False

        print("\nEYETRACKER")
        # add all files to h5-file
        if isinstance(self.edf_file, str) or isinstance(self.edf_file, list):
            self.preprocess_edf_files()
            self.include_blinks = True

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

        h5_file = opj(os.path.dirname(edfs[0]), f"eye.h5")
        self.ho = hedfpy.HDFEyeOperator(h5_file)
        if not os.path.exists(h5_file):
            for i, edf_file in enumerate(edfs):

                if self.use_bids:
                    comps = utils.split_bids_components(edf_file)
                    try:
                        run_ID = comps['run']
                    except:
                        run_ID = i+1
                else:
                    run_ID = i+1

                alias = f"run_{run_ID}"

                self.ho.add_edf_file(edf_file)
                self.ho.edf_message_data_to_hdf(alias=alias)
                self.ho.edf_gaze_data_to_hdf(alias=alias,
                                             pupil_hp=self.high_pass_pupil_f,
                                             pupil_lp=self.low_pass_pupil_f)
        else:
            self.ho.open_hdf_file()

        self.df_eye         = []
        self.blink_events   = []
        self.eye_in_func    = []
        for i, edf_file in enumerate(edfs):

            if self.verbose:
                print(f"Dealing with {edf_file}")

            if self.use_bids:
                bids_comps = utils.split_bids_components(edf_file)
                self.sub, run_ID = bids_comps['sub'], bids_comps['run']
            else:
                run_ID = i+1

            # full output from 'fetch_relevant_info' > use sub as differentiator if multiple files were given
            if self.use_bids:
                self.data = self.fetch_relevant_info(sub=self.sub, run=run_ID)
            else:
                self.data = self.fetch_relevant_info(run=run_ID)

            # collect outputs
            self.blink_events.append(self.fetch_eyeblinks())
            self.eye_in_func.append(self.fetch_eye_func_time())

        self.blink_events = pd.concat(self.blink_events).set_index(['subject', 'run', 'event_type'])
        self.eye_in_func = pd.concat(self.eye_in_func).set_index(['subject', 'run', 't'])

    def fetch_blinks_run(self, run=1, return_type='df'):
        blink_df = utils.select_from_df(self.blink_events, expression=(f"run = {run}"), index=['subject', 'run', 'event_type'])

        if return_type == "df":
            return blink_df
        else:
            return blink_df.values

    def fetch_eyeblinks(self):
        return self.data['blink_events']

    def fetch_eye_func_time(self):
        return self.data['space_func']

    def fetch_eye_tracker_time(self):
        return self.data['space_eye']

    def fetch_relevant_info(self, sub=None, run=1):

        # set alias
        alias = f'run_{run}'
        if self.verbose:
            print(" Alias:       ", alias)

        # load times per session:
        trial_times = self.ho.read_session_data(alias, 'trials')
        trial_phase_times = self.ho.read_session_data(alias, 'trial_phases')

        # read func data file to get nr of volumes
        if sub != None:
            func = utils.get_file_from_substring([f"sub-{sub}_", f'run-{run}'], self.func_file)
        else:
            func = utils.get_file_from_substring(f'run-{run}', self.func_file)

        nr_vols = self.vols(func)

        if func.endswith("nii") or func.endswith("gz"):
            TR = self.TR2
        elif func.endswith('mat'):
            TR = self.TR1
        else:
            TR = 0.105

        # fetch duration of scan
        func_time = nr_vols*TR

        # get block parameters
        session_start_EL_time = trial_times.iloc[0, :][0]
        sample_rate = self.ho.sample_rate_during_period(alias)
        # add number of fMRI*samplerate as stop EL time
        session_stop_EL_time = session_start_EL_time+(func_time*sample_rate)

        eye = self.ho.eye_during_period(
            [session_start_EL_time, session_stop_EL_time], alias)

        if self.verbose:
            print(" Sample rate: ", sample_rate)
            print(" Start time:  ", session_start_EL_time)
            print(" Stop time:   ", session_stop_EL_time)

        # set some stuff required for successful plotting with seconds on the x-axis
        div = False
        if sample_rate == 500:
            n_samples = int(session_stop_EL_time-session_start_EL_time)/2
            duration_sec = n_samples*(1/sample_rate)*2

            div = True
        elif sample_rate == 1000:
            n_samples = int(session_stop_EL_time-session_start_EL_time)
            duration_sec = n_samples*(1/sample_rate)
        else:
            raise ValueError(f"Did not recognize sample_rate of {sample_rate}")

        if self.verbose:
            print(" Duration:     {}s [{} samples]".format(
                duration_sec, n_samples))

        # Fetch a bunch of data
        pupil_raw = np.squeeze(self.ho.signal_during_period(time_period=[
                               session_start_EL_time, session_stop_EL_time+1], alias=alias, signal='pupil', requested_eye=eye))
        pupil_int = np.squeeze(self.ho.signal_during_period(time_period=[
                               session_start_EL_time, session_stop_EL_time+1], alias=alias, signal='pupil_int', requested_eye=eye))
        pupil_bp = np.squeeze(self.ho.signal_during_period(time_period=[
                              session_start_EL_time, session_stop_EL_time+1], alias=alias, signal='pupil_bp', requested_eye=eye))
        pupil_lp = np.squeeze(self.ho.signal_during_period(time_period=[
                              session_start_EL_time, session_stop_EL_time+1], alias=alias, signal='pupil_lp', requested_eye=eye))
        pupil_hp = np.squeeze(self.ho.signal_during_period(time_period=[
                              session_start_EL_time, session_stop_EL_time+1], alias=alias, signal='pupil_hp', requested_eye=eye))
        pupil_bp_psc = np.squeeze(self.ho.signal_during_period(time_period=[
                                  session_start_EL_time, session_stop_EL_time+1], alias=alias, signal='pupil_bp_psc', requested_eye=eye))
        pupil_bp_psc_c = np.squeeze(self.ho.signal_during_period(time_period=[
                                    session_start_EL_time, session_stop_EL_time+1], alias=alias, signal='pupil_bp_clean_psc', requested_eye=eye))

        # Do some plotting
        if not div:
            x = np.arange(0, duration_sec, (1/sample_rate))
        else:
            x = np.arange(0, duration_sec, (1/(sample_rate/2)))

        # resample to match functional data
        resamp = glm.resample_stim_vector(pupil_bp_psc_c.values, nr_vols)
        resamp1 = glm.resample_stim_vector(pupil_raw.values, nr_vols)

        # add start time to it
        start_exp_time = trial_times.iloc[0, :][-1]

        if self.verbose:
            print(" Start time exp = ", round(start_exp_time, 2))

        # get onset time of blinks, cluster blinks that occur within 350 ms
        onsets = self.filter_for_eyeblinks(pupil_raw.to_numpy(),
                                           skip_time=10,
                                           filt_window=500,
                                           sample_rate=sample_rate,
                                           exp_start=start_exp_time)

        # normal eye blink is 1 blink every 4 seconds, throw warning if we found more than a blink per second
        # ref: https://www.sciencedirect.com/science/article/abs/pii/S0014483599906607
        blink_rate = len(onsets) / duration_sec

        if self.verbose:
            print(" Found {} blinks [{} blinks per second]".format(
                len(onsets), round(blink_rate, 2)))

        if blink_rate > 1 or blink_rate < 0.1:
            print(
                f"WARNING for run-{run}: found {round(blink_rate,2)} blinks per second; normal blink rate is 0.25 blinks per second ({len(onsets)} in {duration_sec}s)")

        # if verbose:
        #     print("Saving blink onsets times and pupil size trace")
        # np.save(opj(func_dir, '{}_ses-2_task-LR_{}_eyeblinks.npy'.format(subject, alias.replace('_','-'))), onsets.T)
        # np.save(opj(func_dir, '{}_ses-2_task-LR_{}_pupilsize.npy'.format(subject, alias.replace('_','-'))), resamp)

        # build dataframe with relevant information
        df_space_eye = pd.DataFrame({"pupil_raw": pupil_raw,
                                     "pupil_int": pupil_int,
                                     "pupil_bp": pupil_bp,
                                     "pupil_lp": pupil_lp,
                                     "pupil_hp": pupil_hp,
                                     "pupil_bp_psc": pupil_bp_psc,
                                     "pupil_bp_psc_c": pupil_bp_psc_c})

        # index
        df_space_eye['subject'], df_space_eye['run'] = self.sub, run

        df_space_func = pd.DataFrame({"pupil_raw_2_func": resamp,
                                      "pupil_psc_2_func": resamp1})

        # index
        df_space_func['subject'], df_space_func['run'], df_space_func['t'] = self.sub, run, list(
            TR*np.arange(df_space_func.shape[0]))

        # index
        df_blink_events = pd.DataFrame(onsets.T, columns=['onsets'])
        df_blink_events['subject'], df_blink_events['run'], df_blink_events['event_type'] = self.sub, run, "blink"

        if self.verbose:
            print("Done")

        return {"space_eye": df_space_eye,
                "space_func": df_space_func,
                "blink_events": df_blink_events}

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
            raise ValueError(
                f"Could not derive number of volumes for file '{func_file}'")

        return nr_vols

    @staticmethod
    def filter_for_eyeblinks(arr, skip_time=None, filt_window=350, sample_rate=500, exp_start=None):
        """filter_for_eyeblinks

        This function reads where a blink occurred and will filter onset times with a particular window of 
        occurrance. For instance, a blink generally takes about 100 ms, so any onsets within 100 ms of each
        other can't be physiologically correct. The function will find the first onset time, checks for onset
        times within the 100ms window using the sampling rate, and return the filtered onset times.

        Parameters
        -----------
        arr: np.ndarray
            Array to-be-filtered. If obtained from 'signal_during_period', use 'to_numpy()' as input
        skip_time: int
            skip the first <skip_time> seconds from sampled data to leave out any unphysiological events (default = None)
        filt_window: float
            consider events within <filt_window> as one blink. Given in seconds, default is set to 350ms (0.35s). See: `https://bionumbers.hms.harvard.edu/bionumber.aspx?id=100706&ver=0`
        sample_rate: int
            sampling rate of data, used together with <filt_window> to get the amount of data points that need to be clustered as 1 event
        exp_start: float
            add the start of the experiment time to the onset times. Otherwise timing is re-
            lative to 0, so it's not synced with the experiment.

        Returns
        ----------
        onset times: np.ndarray
            numpy array containing onset times in seconds
        """

        blink_onsets = np.where(arr == 0)[0]

        blink = 0
        filter = True
        blink_arr = []
        while filter:

            try:
                start_blink = blink_onsets[blink]
                end_blink = start_blink+int((filt_window/1000*sample_rate))

                for ii in np.arange(start_blink+1, end_blink):
                    if ii in blink_onsets:
                        blink_onsets = np.delete(
                            blink_onsets, np.where(blink_onsets == ii))

                blink_arr.append(blink_onsets[blink])

                blink += 1
            except:
                filter = False

        onsets = np.array(blink_arr)
        onsets = onsets*(1/sample_rate)

        if skip_time:
            for pp in onsets:
                if pp < skip_time:
                    onsets = np.delete(onsets, np.where(onsets == pp))

        if exp_start:
            onsets = onsets+exp_start

        return onsets


    def get_onset_df(self, index=False):

        if index:
            return self.onset_df.set_index(['subject', 'run', 'event_type'])
        else:
            return self.onset_df

    def onsets_to_txt(self, subject=1, run=1, condition='right', fname=None):
        """onset_to_txt

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

        df = self.onset_df.set_index(['subject', 'run', 'event_type'])
        onsets = list(df['onset'][subject][run]
                      [condition].to_numpy().flatten().T)

        if not fname:
            return onsets
        else:
            np.savetxt(fname, onsets, fmt='%1.3f')
            return fname


class ParseExpToolsFile(ParseEyetrackerFile):

    """ParseExpToolsFile()

    Class for parsing tsv-files created during experiments with Exptools2. The class will read in the file, read when the experiment actually started, correct onset times for this start time and time deleted because of removing the first few volumes (to do this correctly, set the `TR` and `deleted_first_timepoints`). You can also provide a numpy array/file containing eye blinks that should be added to the onset times in real-world time (seconds). In principle, it will return a pandas DataFrame indexed by subject and run that can be easily concatenated over runs. This function relies on the naming used when programming the experiment. In the `session.py` file, you should have created `phase_names=['iti', 'stim']`; the class will use these things to parse the file.

    Parameters
    ----------
    tsv_file: str
        path pointing to the output file of the experiment
    subject: int
        subject number in the returned pandas DataFrame (should start with 1, ..., n)
    run: int
        run number you'd like to have the onset times for
    button: bool
        boolean whether to include onset times of button responses (default is false)
    blinks: str, np.ndarray
        string or array containing the onset times of eye blinks as extracted with hedfpy
    TR: float
        repetition time to correct onset times for deleted volumes
    deleted_first_timepoints: int
        number of volumes to delete to correct onset times for deleted volumes. Can be specified for each individual run if `tsv_file` is a list
    use_bids: bool, optional
        If true, we'll read BIDS-components such as 'sub', 'run', 'task', etc from the input file and use those as indexers, rather than sequential 1,2,3.

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

    def __init__(self, 
                 tsv_file, 
                 subject=1, 
                 run=1, 
                 button=False, 
                 blinks=None, 
                 TR=0.105, 
                 deleted_first_timepoints=0, 
                 edfs=None, 
                 funcs=None, 
                 use_bids=True,
                 verbose=False):

        self.tsv_file                       = tsv_file
        self.sub                            = int(subject)
        self.run                            = int(run)
        self.TR                             = TR
        self.deleted_first_timepoints       = deleted_first_timepoints
        self.button                         = button
        self.blinks                         = blinks
        self.funcs                          = funcs
        self.edfs                           = edfs
        self.use_bids                       = use_bids
        self.verbose                        = verbose

        super().__init__(self.edfs, 
                         subject=self.sub, 
                         func_file=self.funcs, 
                         TR1=self.TR, 
                         use_bids=self.use_bids, 
                         verbose=self.verbose)

        print("EXPTOOLS\n")
        if isinstance(self.tsv_file, str):

            # try to read bids-components from file-name if None's were specified
            if self.use_bids:
                bids_comps = utils.split_bids_components(self.func_file)
                for el in ['sub', 'run']:
                    if getattr(self, el) == None:
                        setattr(self, el, bids_comps[el])

            if not self.tsv_file.endswith("h5"):
                if self.include_blinks:
                    self.blinks = self.fetch_blinks_run(run=self.run)
                    self.preprocess_exptools_file(self.tsv_file, run=self.run, delete_vols=self.deleted_first_timepoints)
                else:
                    self.preprocess_exptools_file(self.tsv_file, run=self.run)
                self.df_onsets = self.get_onset_df(index=True)
            else:
                if self.attribute_tag == None:
                    hdf_store = pd.HDFStore(self.tsv_file)
                    hdf_keys = hdf_store.keys()
                    for key in hdf_keys:
                        key = key.strip("/")
                        setattr(self, key, hdf_store.get(key))
                else:
                    self.from_hdf(self.tsv_file, self.attribute_tag, key=self.hdf_key)


        elif isinstance(self.tsv_file, list):
            df_onsets = []
            for run, onset_file in enumerate(self.tsv_file):
                bids_comps = utils.split_bids_components(onset_file)
                for el in ['sub', 'run']:
                    setattr(self, el, bids_comps[el])

                # include eyeblinks?
                if self.include_blinks:
                    self.blinks = self.fetch_blinks_run(run=self.run)

                # check if we got different nr of vols to delete per run
                delete_vols = check_input_is_list(self, "deleted_first_timepoints", list_element=run)

                # read in the exptools-file
                self.preprocess_exptools_file(onset_file, run=self.run, delete_vols=delete_vols)

                # append to df
                df_onsets.append(self.get_onset_df(index=False))

            # concatemate df
            self.df_onsets = pd.concat(df_onsets).set_index(['subject', 'run', 'event_type'])

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

    def preprocess_exptools_file(self, tsv_file, run=1, delete_vols=0):
        
        data_onsets = []
        with open(tsv_file) as f:
            timings = pd.read_csv(f, delimiter='\t')
            data_onsets.append(pd.DataFrame(timings))

        delete_time         = delete_vols*self.TR
        self.data           = data_onsets[0]
        self.start_times    = pd.DataFrame(self.data[(self.data['response'] == 't') & (self.data['trial_nr'] == 1) & (self.data['phase'] == 0)][['onset']])
        self.data_cut_start = self.data.drop([q for q in np.arange(0,self.start_times.index[0])])
        self.onset_times    = pd.DataFrame(self.data_cut_start[(self.data_cut_start['event_type'] == 'stim') & (self.data_cut_start['condition'].notnull()) | (self.data_cut_start['response'] == 'b')][['onset', 'condition']]['onset'])
        self.condition      = pd.DataFrame(self.data_cut_start[(self.data_cut_start['event_type'] == 'stim') & (self.data_cut_start['condition'].notnull()) | (self.data_cut_start['response'] == 'b')]['condition'])

        # add button presses
        if self.button:
            self.response = self.data_cut_start[(self.data_cut_start['response'] == 'b')]
            self.condition.loc[self.response.index] = 'response'

        self.onset = np.concatenate((self.onset_times, self.condition), axis=1)

        # add eyeblinks
        if isinstance(self.blinks, np.ndarray) or isinstance(self.blinks, str):
            if isinstance(self.blinks, np.ndarray):
                self.eye_blinks = self.blinks
            elif isinstance(self.blinks, str):
                if self.blinks.endwith(".npy"):
                    self.eye_blinks = np.load(self.blinks)
                else:
                    raise ValueError(f"Could not recognize type of {self.blinks}. Should be numpy array or string to numpy file")

            self.eye_blinks = self.eye_blinks.astype('object').flatten()
            tmp = self.onset[:,0].flatten()

            # combine and sort timings
            comb = np.concatenate((self.eye_blinks, tmp))
            comb = np.sort(comb)[...,np.newaxis]

            # add back event types by checking timing values in both arrays
            event_array = []
            for ii in comb:

                if ii in self.onset:
                    idx = np.where(self.onset == ii)[0][0]
                    event_array.append(self.onset[idx][-1])
                else:
                    idx = np.where(self.eye_blinks == ii)[0]
                    event_array.append('blink')

            event_array = np.array(event_array)[...,np.newaxis]

            self.onset = np.concatenate((comb, event_array), axis=1)

        # correct for start time of experiment and deleted time due to removal of inital volumes
        self.onset[:, 0] = self.onset[:, 0] - \
            float(self.start_times['onset'] + delete_time)

        # make dataframe
        self.onset_df = self.index_onset(self.onset, columns=['onset', 'event_type'], subject=self.sub, run=run)

    @staticmethod
    def index_onset(array, columns=None, subject=1, run=1, TR=0.105, set_index=False):
        
        if columns == None:
            df = pd.DataFrame(array)
        else:
            df = pd.DataFrame(array, columns=columns)
            
        df['subject'], df['run']    = subject, run
        df['event_type']            = df['event_type'].astype(str)
        df['onset']                 = df['onset'].astype(float)

        if set_index:
            return df.set_index(['subject', 'event_type'])
        else:
            return df        

    def get_onset_df(self, index=False):
        """Return the indexed DataFrame containing onset times"""

        if index:
            return self.onset_df.set_index(['subject', 'run', 'event_type'])
        else:
            return self.onset_df

    def onsets_to_txt(self, subject=1, run=1, condition='right', fname=None):
        """onset_to_txt

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

        df = self.onset_df.set_index(['subject', 'run', 'event_type'])
        onsets = list(df['onset'][subject][run][condition].to_numpy().flatten().T)

        if not fname:
            return onsets
        else:
            np.savetxt(fname, onsets, fmt='%1.3f')
            return fname


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
    >>> physio = utils.ParsePhysioFile(physio_file,
    >>>                                physio_mat=physio_mat,
    >>>                                subject=func.subject,
    >>>                                run=func.run,
    >>>                                TR=func.TR,
    >>>                                deleted_first_timepoints=func.deleted_first_timepoints,
    >>>                                deleted_last_timepoints=func.deleted_last_timepoints)
    >>> physio_df   = physio.get_physio(index=False)
    """

    def __init__(self, 
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
                
            # try to read bids-components from file-name if None's were specified
            if self.use_bids:
                bids_comps = utils.split_bids_components(self.physio_file)
                for el in ['sub', 'run']:
                    if getattr(self, el) == None:
                        setattr(self, el, bids_comps[el])
            
            # preprocess
            if self.verbose:
                print(f"Preprocessing {self.physio_file}")
            self.preprocess_physio_file(self.physio_file,
                                        physio_mat=self.physio_mat,
                                        deleted_first_timepoints=self.deleted_first_timepoints,
                                        deleted_last_timepoints=self.deleted_last_timepoints)

            self.df_physio = self.get_physio(index=True)
                
        elif isinstance(self.physio_file, list):

            df_physio = []
            for run, func in enumerate(self.physio_file):
                if self.verbose:
                    print(f"Preprocessing {func}")

                if self.use_bids:
                    bids_comps = utils.split_bids_components(func)
                    for el in ['sub', 'run']:
                        setattr(self, el, bids_comps[el])
                else:
                    self.run = run

                # check if deleted_first_timepoints is list or not
                delete_first = check_input_is_list(self, var="deleted_first_timepoints", list_element=run)

                # check if deleted_last_timepoints is list or not
                delete_last = check_input_is_list(self, var="deleted_last_timepoints", list_element=run)

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

                self.preprocess_physio_file(func, 
                                            physio_mat=mat_file,
                                            deleted_first_timepoints=delete_first,
                                            deleted_last_timepoints=delete_last)

                df_physio.append(self.get_physio(index=False))

            self.df_physio = pd.concat(df_physio).set_index(['subject', 'run', 't'])
        
    def preprocess_physio_file(self, 
                               physio_tsv, 
                               physio_mat=None, 
                               deleted_first_timepoints=0, 
                               deleted_last_timepoints=0):

        self.physio_data = pd.read_csv(physio_tsv,
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
            self.hr = self.mat['physio']['ons_secs'][0][0][0][0][12]

            # trim beginning and end
            if deleted_last_timepoints != 0:
                self.physio_df['hr'] = self.hr[:,deleted_first_timepoints:-deleted_last_timepoints]
            else:
                self.physio_df['hr'] = self.hr[:,deleted_first_timepoints:]

            # self.physio_df['hr'] = (self.physio_df['hr'] - self.physio_df['hr'].mean())/self.physio_df['hr'].std(ddof=0)

        self.physio_df['subject'], self.physio_df['run'], self.physio_df['t'] = self.sub, self.run, list(self.TR*np.arange(self.physio_df.shape[0]))

    def get_physio(self, index=True):
        if index:
            return self.physio_df.set_index(['subject', 'run', 't'])
        else:
            return self.physio_df


class ParseFuncFile(ParseExpToolsFile, ParsePhysioFile):

    """ParseFuncFile()

    Class for parsing func-files created with Luisa's reconstruction. It can do filtering, conversion to percent signal change, and create power spectra. It is supposed to look similar to :class:`linescanning.utils.ParseExpToolsFile` to make it easy to translate between the functional data and experimental data.

    Parameters
    ----------
    func_file: str, list
        path or list of paths pointing to the output file of the experiment
    subject: int, optional
        subject number in the returned pandas DataFrame (should start with 1, ..., n)
    run: int, optional
        run number you'd like to have the onset times for
    bp_filter: str, optional
        method for filtering. Possible options include: "boxcar", "hanning", "rolling", "iir", "fir", and fourier". If you just want percent signal changed non-filtered data, choose `bp_filter="rolling"` and select `self.dct_psc_df` as data
    standardize: str, optional
        method of standardization (e.g., "zscore" or "psc"). If `bp_filter=="rolling"`, data is automatically converted to *percent signal change*
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

    Example
    ----------
    >>> from linescanning import utils
    >>> func_file = utils.get_file_from_substring(f"run-1_bold.mat", opj('sub-001', 'ses-1', 'func'))
    >>> func = utils.ParseFuncFile(func_file, subject=1, run=1, deleted_first_timepoints=100, deleted_last_timepoints=300)
    >>> raw = func.get_raw(index=True)
    >>> psc = func.get_psc(index=True)
    """

    def __init__(self, 
                 func_file, 
                 subject=1, 
                 run=1,
                 high_pass=True,
                 low_pass=True,
                 lb=0.01, 
                 TR=0.105, 
                 deleted_first_timepoints=0, 
                 deleted_last_timepoints=0, 
                 window_size=19,
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
                 **kwargs):

        self.sub                        = subject
        self.run                        = run
        self.TR                         = TR
        self.lb                         = lb
        self.high_pass                  = high_pass
        self.low_pass                   = low_pass
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
        self.__dict__.update(kwargs)
        
        if self.phys_file != None: 
            
            # super(ParsePhysioFile, self).__init__(**kwargs)                                              
            ParsePhysioFile.__init__(self, 
                                     self.phys_file, 
                                     physio_mat=self.phys_mat, 
                                     use_bids=self.use_bids,
                                     TR=self.TR,
                                     deleted_first_timepoints=self.deleted_first_timepoints,
                                     deleted_last_timepoints=self.deleted_last_timepoints,
                                     **kwargs)
        
        print("\nFUNCTIONAL")

        if isinstance(self.func_file, str):
            if not self.func_file.endswith("h5"):
                
                # try to read bids-components from file-name if None's were specified
                if self.use_bids:
                    bids_comps = utils.split_bids_components(self.func_file)
                    for el in ['sub', 'run']:
                        if getattr(self, el) == None:
                            setattr(self, el, bids_comps[el])
                
                # preprocess
                if self.verbose:
                    print(f"Preprocessing {self.func_file}")

                self.preprocess_func_file(self.func_file,
                                          run=self.run, 
                                          lowpass=self.low_pass, 
                                          highpass=self.high_pass, 
                                          deleted_first_timepoints=self.deleted_first_timepoints,
                                          deleted_last_timepoints=self.deleted_last_timepoints)

                self.df_func_psc = self.get_psc(index=True)
                self.df_func_raw = self.get_raw(index=True)
            else:
                if self.attribute_tag == None:
                    hdf_store = pd.HDFStore(func_file)
                    hdf_keys = hdf_store.keys()
                    for key in hdf_keys:
                        key = key.strip("/")
                        setattr(self, key, hdf_store.get(key))
                else:
                    self.from_hdf(func_file, self.attribute_tag, key=self.hdf_key)
                
        elif isinstance(func_file, list):
            df_func     = []
            df_raw      = []
            df_retro    = []
            df_r2       = []
            for run, func in enumerate(self.func_file):
                if self.verbose:
                    print(f"Preprocessing {func}")
                if self.use_bids:
                    bids_comps = utils.split_bids_components(func)
                    for el in ['sub', 'run']:
                        setattr(self, el, bids_comps[el])
                else:
                    self.run = run

                # check if deleted_first_timepoints is list or not
                delete_first = check_input_is_list(self, var="deleted_first_timepoints", list_element=run)

                # check if deleted_last_timepoints is list or not
                delete_last = check_input_is_list(self, var="deleted_last_timepoints", list_element=run)

                self.preprocess_func_file(func, 
                                          run=self.run, 
                                          lowpass=self.low_pass, 
                                          highpass=self.high_pass, 
                                          deleted_first_timepoints=delete_first,
                                          deleted_last_timepoints=delete_last)
            
                df_func.append(self.get_psc(index=False))
                df_raw.append(self.get_raw(index=False))

                if retroicor:
                    df_retro.append(self.get_retroicor(index=False))
                    df_r2.append(self.r2_physio_df)

            self.df_func_psc = pd.concat(df_func).set_index(['subject', 'run', 't'])
            self.df_func_raw = pd.concat(df_raw).set_index(['subject', 'run', 't'])

            if retroicor:
                try:
                    self.df_retro_zscore = pd.concat(df_retro).set_index(['subject', 'run', 't'])
                    self.df_r2 = pd.concat(df_r2).set_index(['subject', 'run', 'vox'])
                except:
                    raise ValueError("RETROICOR did not complete successfully..")


        # now that we have nicely formatted functional data, initialize the ParseExpToolsFile-class
        if self.tsv_file != None: 
            ParseExpToolsFile.__init__(self,
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

    def preprocess_func_file(self, 
                             func_file, 
                             run=1, 
                             lowpass=None, 
                             highpass=None, 
                             deleted_first_timepoints=0, 
                             deleted_last_timepoints=0):

        # Load in datasets with tag "wcsmtSNR"
        if func_file.endswith("nii") or func_file.endswith("gz"):
            raise NotImplementedError("This datatype is not supported because of different nr of voxels compared to line data.. This make concatenation of dataframes not possible")
        
        if func_file.endswith("mat"):
            self.ts_wcsmtSNR    = io.loadmat(func_file)
            self.tag            = list(self.ts_wcsmtSNR.keys())[-1]
            self.ts_wcsmtSNR    = self.ts_wcsmtSNR[self.tag]
            self.ts_complex     = self.ts_wcsmtSNR
            self.ts_magnitude   = np.abs(self.ts_wcsmtSNR)
        elif func_file.endswith("npy"):
            self.ts_magnitude   = np.load(func_file)
        elif func_file.endswith("nii") or func_file.endswith("gz"):
            raise NotImplementedError()

        # trim beginning and end
        if deleted_last_timepoints != 0:
            self.ts_corrected = self.ts_magnitude[:,deleted_first_timepoints:-deleted_last_timepoints]
        else:
            self.ts_corrected = self.ts_magnitude[:,deleted_first_timepoints:]

        self.vox_cols = [f'vox {x}' for x in range(self.ts_corrected.shape[0])]
        self.raw_data = self.index_func(self.ts_corrected, 
                                        columns=self.vox_cols, 
                                        subject=self.sub, 
                                        run=run, 
                                        TR=self.TR)

        # apply some filtering
        if highpass:
            if self.verbose:
                print(f" DCT-high pass filter [removes low frequencies <{self.lb} Hz]")
            
            self.high_passed, self._cosine_drift = self.highpass_dct(self.ts_corrected, self.lb, TR=self.TR)
            self.high_passed_psc    = utils.percent_change(self.high_passed, -1)
            self.high_passed_df     = self.index_func(self.high_passed_psc, 
                                                      columns=self.vox_cols, 
                                                      subject=self.sub,
                                                      run=run, 
                                                      TR=self.TR, 
                                                      set_index=True)
        if lowpass:
            if self.verbose:
                print(f" Savitsky-Golay low pass filter [removes high frequences] (window = {self.window_size}, order = {self.poly_order})")            
            if hasattr(self, "high_passed"):
                use_data = self.high_passed.copy()
            else:
                use_data = self.ts_corrected.copy()  

            self.low_passed = self.lowpass_savgol(use_data, window_length=self.window_size, polyorder=self.poly_order)

        # check if we should attempt RETROICOR
        if self.retroicor:

            # we should have df_physio dataframe from ParsePhysioFile
            if hasattr(self, "df_physio"):
                try:
                    # select subset of df_physio. Run IDs must correspond!
                    self.confs = utils.select_from_df(self.df_physio, expression=f"run = {self.run}")
                except:
                    raise ValueError(f"Could not extract dataframe from 'df_physio' with expression: 'run = {self.run}'")

                # check what data we should use
                if hasattr(self, "lowpass"):
                    data = self.lowpass.copy()
                elif hasattr(self, "highpass"):
                    data = self.highpass.copy()
                else:
                    data = self.ts_corrected.copy()

                self.z_score = signal.clean(data, standardize=True)

                if "hr" in list(self.confs.columns):
                    self.confs = self.confs.drop(columns=['hr'])

                # regress out the confounds with signal.clean
                if self.verbose:
                    print(" RETROICOR with physio-regressors")

                cardiac     = utils.select_from_df(self.confs, expression='ribbon', indices=[0,self.orders[0]])
                respiration = utils.select_from_df(self.confs, expression='ribbon', indices=[self.orders[0],self.orders[0]+self.orders[1]])
                interaction = utils.select_from_df(self.confs, expression='ribbon', indices=[self.orders[0]+self.orders[1],len(list(self.confs.columns))])

                self.clean_all          = signal.clean(self.z_score.T, standardize=False, confounds=self.confs.values).T
                self.clean_resp         = signal.clean(self.z_score.T, standardize=False, confounds=respiration.values).T
                self.clean_cardiac      = signal.clean(self.z_score.T, standardize=False, confounds=cardiac.values).T
                self.clean_interaction  = signal.clean(self.z_score.T, standardize=False, confounds=interaction.values).T

                # create the dataframes
                self.z_score_df    = self.index_func(self.z_score, columns=self.vox_cols, subject=self.sub, run=run, TR=self.TR)

                self.z_score_retroicor_df    = self.index_func(self.clean_all, columns=self.vox_cols, subject=self.sub, run=run, TR=self.TR)

                print(self.z_score.shape)
                self.r2_all          = 1-(np.var(self.clean_all, -1) / np.var(self.z_score, -1))
                self.r2_resp         = 1-(np.var(self.clean_resp, -1) / np.var(self.z_score, -1))
                self.r2_cardiac      = 1-(np.var(self.clean_cardiac, -1) / np.var(self.z_score, -1))
                self.r2_interaction  = 1-(np.var(self.clean_interaction, -1) / np.var(self.z_score, -1))
                
                # save in a subject X run X voxel manner
                self.r2_physio = {'all': self.r2_all,   
                                  'respiration': self.r2_resp, 
                                  'cardiac': self.r2_cardiac, 
                                  'interaction': self.r2_interaction}

                self.r2_physio_df = pd.DataFrame(self.r2_physio)
                self.r2_physio_df['subject'], self.r2_physio_df['run'], self.r2_physio_df['vox'] = self.sub, run, np.arange(0,self.r2_all.shape[0])

            else:
                print(" Could not find df_physio.. Retroicor failed")

        # percent signal change       
        if hasattr(self, "low_passed"):
            if self.verbose:
                print(" Using low-passed [SG-filtered] data for PSC")
            self.data = self.low_passed.copy()
        elif hasattr(self, "high_passed"):
            if self.verbose:
                print(" Using high-passed [DCT-filtered] data for PSC")            
            self.data = self.high_passed.copy()
        else:
            if self.verbose:
                print(" Using non-filtered data for PSC")            
            self.data = self.ts_corrected.copy()

        self.psc_data       = utils.percent_change(self.data, -1)
        self.psc_data_df    = self.index_func(self.psc_data, columns=self.vox_cols, subject=self.sub, run=run, TR=self.TR)

        self.psc_raw       = utils.percent_change(self.ts_corrected, -1)
        self.psc_raw_df    = self.index_func(self.psc_raw, columns=self.vox_cols, subject=self.sub, run=run, TR=self.TR)        

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

    @staticmethod
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
            _description_, by default 0.105

        Returns
        ----------
        _type_
            _description_

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
        dct_data        = signal.clean(func.T, detrend=False, standardize=False, confounds=hp_set).T

        return dct_data, hp_set

    @staticmethod
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
        
        return savgol_filter(func, window_length, polyorder, axis=-1)

    @staticmethod
    def get_freq(func, TR=0.105, spectrum_type='psd', clip_power=None):
        
        if spectrum_type != "fft":
            TC      = TimeSeries(np.asarray(func), sampling_interval=1/TR)
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
                power = np.clip(power, 0, clip_power)

            return freq,power

        else:
            TR      = 0.105
            freq    = np.fft.fftshift(np.fft.fftfreq(func.shape[0], d=TR))
            power   = np.abs(np.fft.fftshift(np.fft.fft(func)))**2/func.shape[0]

            if clip_power != None:
                power[power>clip_power] = 0    

            return freq, power

    def get_zscore(self, index=False):
        if hasattr(self, 'z_score_df'):
            if index:
                return self.z_score_df.set_index(['subject', 'run', 't'])
            else:
                return self.z_score_df

    def get_retroicor(self, index=False):
        if hasattr(self, 'z_score_retroicor_df'):
            if index:
                return self.z_score_retroicor_df.set_index(['subject', 'run', 't'])
            else:
                return self.z_score_retroicor_df                

    def get_psc(self, index=False, psc=True):
        if psc:
            if hasattr(self, 'psc_data_df'):
                if index:
                    return self.psc_data_df.set_index(['subject', 'run', 't'])
                else:
                    return self.psc_data_df
        else:
            raise NotImplementedError("Not sure yet what this should be..")

    def get_raw(self, index=False, psc=True):
        if psc:
            if hasattr(self, 'psc_raw_df'):
                if index:
                    return self.psc_raw_df.set_index(['subject', 'run', 't'])
                else:
                    return self.psc_raw_df
        else:
            if hasattr(self, 'raw_data'):
                if index:
                    return self.raw_data.set_index(['subject', 'run', 't'])
                else:
                    return self.raw_data

    def to_hdf(self, attribute_tag, output_file, key="df"):
        if hasattr(self, attribute_tag):
            add_df = getattr(self, attribute_tag)
            if os.path.exists(output_file):
                add_df.to_hdf(output_file,key=key, append=True, mode='r+', format='t')
            else:
                add_df.to_hdf(output_file, key=key, mode='w')
            

    def from_hdf(self, input_file, attribute_tag, key="df"):
        setattr(self, attribute_tag, pd.read_hdf(input_file, key))

class Dataset(ParseFuncFile):
    """Dataset

    Main class for retrieving, formatting, and preprocessing of all datatypes including fMRI (2D), eyetracker (*.edf), physiology (*.log [WIP]), and experiment files derived from `Exptools2` (*.tsv). If you leave `subject` and `run` empty, these elements will be derived from the file names. So if you have BIDS-like files, leave them empty and the dataframe will be created for you with the correct subject/run IDs
    
    Parameters
    ----------
    func_file: str, list
        path or list of paths pointing to the output file of the experiment
    tsv_file: str
        path pointing to the output file of the experiment 
    edf_file: str, list
        path pointing to the output file of the experiment; can be a list of multiple               
    subject: int, optional
        subject number in the returned pandas DataFrame (should start with 1, ..., n)
    run: int, optional
        run number you'd like to have the onset times for
    lb: float, optional
        lower bound for signal filtering
    TR: float, optional
        repetition time to correct onset times for deleted volumes
    button: bool
        boolean whether to include onset times of button responses (default is false)        
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
    >>> data = dataset.Dataset(funcs,
    >>>                        deleted_first_timepoints=delete_first,
    >>>                        deleted_last_timepoints=delete_last,
    >>>                        window_size=window,
    >>>                        high_pass=True,
    >>>                        low_pass=True,
    >>>                        poly_order=order,
    >>>                        tsv_file=exp,
    >>>                        verbose=True)
    >>> #
    >>> # retrieve data
    >>> fmri = data.fetch_fmri()
    >>> onsets = data.fetch_onsets()
    """

    def __init__(self, 
                 func_file,
                 subject=1,
                 run=1,
                 TR=0.105, 
                 tsv_file=None,
                 edf_file=None,
                 phys_file=None,
                 phys_mat=None,
                 high_pass=True,
                 low_pass=True,
                 button=False,
                 lb=0.01, 
                 deleted_first_timepoints=0, 
                 deleted_last_timepoints=0, 
                 window_size=20,
                 poly_order=3,
                 attribute_tag=None,
                 hdf_key="df",
                 use_bids=True,
                 verbose=False,
                 retroicor=False,
                 **kwargs):

        self.sub                        = subject
        self.run                        = run
        self.TR                         = TR
        self.lb                         = lb
        self.high_pass                  = high_pass
        self.low_pass                   = low_pass
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
        self.__dict__.update(kwargs)

        super().__init__(self.func_file,
                         TR=self.TR,
                         subject=self.sub,
                         run=self.run,
                         lb=self.lb,
                         low_pass=self.low_pass,
                         high_pass=self.high_pass,
                         deleted_first_timepoints=self.deleted_first_timepoints,
                         deleted_last_timepoints=self.deleted_last_timepoints,
                         window_size=self.window_size,
                         poly_order=self.poly_order,
                         tsv_file=self.tsv_file,
                         edf_file=self.edf_file,
                         phys_file=self.phys_file,
                         phys_mat=self.phys_mat,
                         use_bids=self.use_bids,
                         verbose=self.verbose,
                         retroicor=self.retroicor,
                         **kwargs)

    print("DATASET")
    def fetch_fmri(self, strip_index=False, type='filt+psc'):
        if type == "filt+psc":
            attr = 'df_func_psc'
        elif type == "retroicor":
            attr = 'df_retro_zscore'
        elif type == "raw+psc":
            attr = 'df_func_raw'
        else:
            raise ValueError(f"Unknown option '{type}'. Must be 'filt+psc', 'retroicor', or 'raw+psc'")

        if hasattr(self, attr):
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

    def fetch_physio(self, strip_index=False):
        if hasattr(self, 'df_physio'):
            if strip_index:
                return self.df_physio.reset_index().drop(labels=list(self.df_physio.index.names), axis=1)
            else:
                return self.df_physio
        else:
            print("No physio-data was provided")            

    def fetch_trace(self, strip_index=False):
        if hasattr(self, 'eye_in_func'):
            if strip_index:
                return self.eye_in_func.reset_index().drop(labels=list(self.eye_in_func.index.names), axis=1)
            else:
                return self.eye_in_func
        else:
            print("No eyetracking-data was provided")

    def fetch_blinks(self, strip_index=False):
        if hasattr(self, 'blink_events'):
            return self.blink_events
        else:
            print("No eyetracking-data was provided")

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
