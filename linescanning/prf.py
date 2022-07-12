import ast
from datetime import datetime, timedelta
from linescanning import utils, plotting, dataset, preproc
import math
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from past.utils import old_div
from prfpy import rf,stimulus
from prfpy.fit import Iso2DGaussianFitter, Norm_Iso2DGaussianFitter
from prfpy.model import Iso2DGaussianModel, Norm_Iso2DGaussianModel
import random
from scipy.ndimage import rotate
from scipy import signal, io
import subprocess
import time
import yaml
import pickle

opj = os.path.join

def get_prfdesign(screenshot_path,
                  n_pix=40,
                  dm_edges_clipping=[0,0,0,0]):
    """
    get_prfdesign

    Basically Marco's gist, but then incorporated in the repo. It takes the directory of screenshots and creates a vis_design.mat file, telling pRFpy at what point are certain stimulus was presented.

    Parameters
    ----------
    screenshot_path: str
        string describing the path to the directory with png-files
    n_pix: int
        size of the design matrix (basically resolution). The larger the number, the more demanding for the CPU. It's best to have some value which can be divided with 1080, as this is easier to downsample. Default is 40, but 270 seems to be a good trade-off between resolution and CPU-demands
    dm_edges_clipping: list
        people don't always see the entirety of the screen so it's important to check what the subject can actually see by showing them the cross of for instance the BOLD-screen (the matlab one, not the linux one) and clip the image accordingly

    Returns
    ----------
    numpy.ndarray
        array with shape <n_pix,n_pix,timepoints> representing a binary paradigm

    Example
    ----------
    >>> dm = get_prfdesign('path/to/dir/with/pngs', n_pix=270, dm_edges_clipping=[6,1,0,1])
    """

    image_list = os.listdir(screenshot_path)

    # get first image to get screen size
    img = (255*mpimg.imread(opj(screenshot_path, image_list[0]))).astype('int')

    # there is one more MR image than screenshot
    design_matrix = np.zeros((img.shape[0], img.shape[0], 1+len(image_list)))

    for image_file in image_list:
        
        # assuming last three numbers before .png are the screenshot number
        img_number = int(image_file[-7:-4])-1
        
        # subtract one to start from zero
        img = (255*mpimg.imread(opj(screenshot_path, image_file))).astype('int')
        
        # make it square
        if img.shape[0] != img.shape[1]:
            offset = int((img.shape[1]-img.shape[0])/2)
            img = img[:, offset:(offset+img.shape[0])]

        # binarize image into dm matrix
        # assumes: standard RGB255 format; only colors present in image are black, white, grey, red, green.
        design_matrix[...,img_number][np.where(((img[...,0] == 0) & (
            img[...,1] == 0)) | ((img[...,0] == 255) & (img[...,1] == 255)))] = 1

        design_matrix[...,img_number][np.where(((img[...,0] == img[...,1]) & (
            img[...,1] == img[...,2]) & (img[...,0] != 127)))] = 1

    # downsample with scipy.interpolate.interp2d
    dm_resampled = np.zeros((n_pix, n_pix, design_matrix.shape[-1]))
    
    for img in range(design_matrix.shape[-1]):
        dm_resampled[...,img] = utils.resample2d(design_matrix[...,img], n_pix)
    
    #clipping edges
    #top, bottom, left, right
    dm_resampled[:dm_edges_clipping[0], :, :] = 0
    dm_resampled[(dm_resampled.shape[0]-dm_edges_clipping[1]):, :, :] = 0
    dm_resampled[:, :dm_edges_clipping[2], :] = 0
    dm_resampled[:, (dm_resampled.shape[0]-dm_edges_clipping[3]):, :] = 0
    # print("  Design matrix completed")

    return dm_resampled


def radius(stim_arr):

    """radius

    Return the radius of a given stimulus within a stimulus grid.

    Parameters
    ----------
    stim_arr: numpy.ndarray
        arrays containing the stimulus as created with :func:`linescanning.prf.make_stims`

    Returns
    ----------
    float
        radius of circular stimulus within stim_arr
    """

    max_idc = np.where(stim_arr == float(1))[0]
    try:
        lb, rb = max_idc[0], max_idc[-1]
        radius = (rb-lb)/2
    except:
        radius = 0

    return radius


def create_circular_mask(h, w, center=None, radius=None):

    """create_circular_mask

    Creates a circular mask in a numpy meshgrid to create filled stimuli or mask out parts of a circular stimulus to create concentric stimuli.

    Parameters
    ----------
    h: int
        height of meshgrid
    w: int
        width of meshgrid
    center: tuple, list
        center of the circle to create
    radius: float
        radius of stimulus to draw

    Returns
    ----------
    numpy.ndarray
        np.meshgrid with input size with the specified circle set to ones
    """

    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def make_stims(n_pix, prf_object, dim_stim=2, factr=4, concentric=False, concentric_size=0.65):

    """make_stims

    Creates a list of stimuli to create size/response curves.

    Parameters
    ----------
    n_pix: int
        number of pixels in the grid to use
    prf_object: prfpy.stimulus.PRFStimulus2D
        representation the pRF in visual space
    dim_stim: int
        number of dimensions to use: 2 for circle, 1 for bar
    factr: int
        factor with which to increase stimulus size
    concentric: boolean
        If true, concentric stimuli will be made. For that, the next argument is required
    concentric_size: float
        proportion of stimulus size that needs to be masked out again

    Returns
    ----------
    list
        list of numpy.ndarrays with meshgrid-size containing the stimuli. Can be plotted with :func:`linescanning.prf.plot_stims`
    """

    ss_deg = 2.0 * np.degrees(np.arctan(prf_object.screen_size_cm /(2.0*prf_object.screen_distance_cm)))
    x = np.linspace(-ss_deg/2, ss_deg/2, n_pix)

    if dim_stim == 1:
        stims = [np.zeros_like(n_pix) for n in range(int(n_pix/4))]
    else:
        factr=4
        stims = [np.zeros((n_pix, n_pix)) for n in range(int(n_pix/(2*factr)))]
        stim_sizes=[]

    for pp, stim in enumerate(stims):

        if dim_stim == 1:
            #2d rectangle or 1d
            stim[int(stim.shape[0]/2)-pp:int(stim.shape[0]/2)+pp] = 1
        else:
            #2d circle
            xx,yy = np.meshgrid(x,x)
            stim[((xx**2+yy**2)**0.5)<(x.max()*pp/(len(stims)*factr))] = 1
            stim_sizes.append(2*(x.max()*pp/(len(stims)*factr)))

            # make concentric rings
            if concentric:
                stim_rad = radius(stim)

                if stim_rad > 2:
                    mask = create_circular_mask(x.shape[0], x.shape[0], radius=concentric_size*stim_rad)
                    stim[mask] = 0

    if dim_stim > 1:
        return stims, stim_sizes
    else:
        return stims


def make_stims2(n_pix, prf_object, dim_stim=2, degrees=[0,6], concentric=False, concentric_size=0.65):

    """make_stims2

    Creates a list of stimuli to create size/response curves. In contrast to the first version, this version uses :func:`linescanning.prf.create_circular_mask` more intensively. This allows for the manual specification of visual degrees to use, as it uses an adaptation of psychopy's `deg2pix` function to translate the visual degree to pixel space to create a numpy mask.

    Parameters
    ----------
    n_pix: int
        number of pixels in the grid to use
    prf_object: prfpy.stimulus.PRFStimulus2D
        representation the pRF in visual space
    dim_stim: int
        number of dimensions to use: 2 for circle, 1 for bar
    degrees: list
        list representing the range of visual degrees to create the stimuli with
    concentric: boolean
        If true, concentric stimuli will be made. For that, the next argument is required
    concentric_size: float
        proportion of stimulus size that needs to be masked out again

    Returns
    ----------
    list
        list of numpy.ndarrays with meshgrid-size containing the stimuli. Can be plotted with :func:`linescanning.prf.plot_stims`
    """

    degrees = np.linspace(*degrees, num=33)

    if dim_stim == 1:
        stims = [np.zeros_like(n_pix) for n in range(int(n_pix/4))]
    else:
        factr=4
        stims = [np.zeros((n_pix, n_pix)) for n in range(int(n_pix/(2*factr)))]
        stim_sizes=[]

    for pp, stim in enumerate(stims):

        if dim_stim == 1:
            #2d rectangle or 1d
            stim[int(stim.shape[0]/2)-pp:int(stim.shape[0]/2)+pp] = 1
        else:
            #2d circle
            mask = create_circular_mask(n_pix, n_pix, radius=deg2pix(degrees[pp], prf_object))
            stim[mask] = 1

            # make concentric rings
            if concentric:
                stim_rad = radius(stim)

                if stim_rad > 2:
                    mask = create_circular_mask(n_pix, n_pix, radius=concentric_size*stim_rad)
                    stim[mask] = 0

    if dim_stim > 1:
        return stims, degrees
    else:
        return stims


def plot_stims(stims, n_rows=3, n_cols=11, figsize=(60,10)):

    """plot_stims

    Plots all stimuli contained in `stims` as created in `prf.make_stims`. Assumesthat about 33 stimuli have been produced, otherwise axes ordering will be messy..

    Parameters
    ----------
    stims: list
        list of numpy.ndarrays with meshgrid-size containing the stimuli

    Returns
    ----------
    matplotlib.pyplot plot
    """

    # if len(stims) != 33:
    #     raise Exception(f"Ideally, the number of created stimuli is 33 for this function.. Length = {len(stims)}")

    # , sharex=True, sharey=True)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)

    # for ix, ii in enumerate(stims):
    #
    #     if ix < n_cols:
    #         # print(f"ROW1 > {ix}")
    #         axs[0, ix].imshow(ii)
    #         axs[0, ix].axis('off')
    #     elif n_cols <= ix <= (2*n_cols)-1:
    #         new_ix = ix-n_cols
    #         # print(f"ROW 2 > old: {ix}; new: {new_ix}")
    #         axs[1, new_ix].imshow(ii)
    #         axs[1, new_ix].axis('off')
    #     else:
    #         new_ix = ix-(n_cols*2)
    #         # print(f"ROW 3 > old: {ix}; new: {new_ix}")
    #         axs[2, new_ix].imshow(ii)
    #         axs[2, new_ix].axis('off')

    # All the above is done much simpler and cleaner with this:
    for i, ax in enumerate(fig.axes):
        ax.imshow(stims[i])
        ax.axis('off')

    plt.tight_layout()

def make_prf(prf_object, mu_x=0, mu_y=0, size=None, resize_pix=None):
    """make_prf

    Create an instantiation of a pRF using the parameters obtained during fitting.

    Parameters
    ----------
    prf_object: prfpy.stimulus.PRFStimulus2D
        representation the pRF in visual space
    mu_x: float
        x-component of pRF. Leave default if you want to create size/response functions
    mu_y: float
        y-component of pRF. Leave default if you want to create size/response functions
    size: float
        size of pRF, optional
    resize_pix: int
        resolution of pRF to resample to. For instance, if you've used a low-resolution design matrix, but you'd like a prettier image, you can set `resize` to something higher than the original (54 >> 270, for example). By default not used.

    Returns
    ----------
    numpy.ndarray
        meshgrid containing Gaussian characteristics of the pRF. Can be plotted with :func:`linescanning.plotting.LazyPRF`
    """

    prf = np.rot90(rf.gauss2D_iso_cart(x=prf_object.x_coordinates[..., np.newaxis],
                                       y=prf_object.y_coordinates[..., np.newaxis],
                                       mu=(mu_x, mu_y),
                                       sigma=size,
                                       normalize_RFs=False).T, axes=(1, 2))
    if isinstance(resize_pix, int):
        prf_squeezed = np.squeeze(prf, axis=0)
        prf = utils.resample2d(prf_squeezed, resize_pix)[np.newaxis,...] 
        
    return prf


# From Marco's https://github.com/VU-Cog-Sci/prfpy_tools/blob/master/utils/postproc_utils.py
def norm_2d_sr_function(a,b,c,d,s_1,s_2,x,y,stims,mu_x=0,mu_y=0):
    """create size/response function given set of parameters and stimuli"""

    xx,yy = np.meshgrid(x,y)

    sr_function = (a*np.sum(np.exp(-((xx-mu_x)**2+(yy-mu_y)**2)/(2*s_1**2))*stims, axis=(-1,-2)) + b) / (c*np.sum(np.exp(-((xx-mu_x)**2+(yy-mu_y)**2)/(2*s_2**2))*stims, axis=(-1,-2)) + d) - b/d
    return sr_function


# From Marco's https://github.com/VU-Cog-Sci/prfpy_tools/blob/master/utils/postproc_utils.py
def norm_1d_sr_function(a,b,c,d,s_1,s_2,x,stims,mu_x=0):
    """Create size/response function for 1D stimuli"""

    sr_function = (a*np.sum(np.exp(-(x-mu_x)**2/(2*s_1**2))*stims, axis=-1) + b) / (c*np.sum(np.exp(-(x-mu_x)**2/(2*s_2**2))*stims, axis=-1) + d) - b/d
    return sr_function


# Adapted from psychopy
def pix2deg(pixels, prf_object, scrSizePix=[270, 270]):
    """Convert size in pixels to size in degrees for a given Monitor object""" 

    # get monitor params and raise error if necess
    scrWidthCm = prf_object.screen_size_cm
    dist = prf_object.screen_distance_cm

    cmSize = pixels * float(scrWidthCm) / scrSizePix[0]
    return old_div(cmSize, (dist * 0.017455))


# Adapted from psychopy
def deg2pix(degrees, prf_object, scrSizePix=[270, 270]):
    """Convert size in degrees to size in pixels for a given pRF object"""

    # get monitor params and raise error if necess
    scrWidthCm = prf_object.screen_size_cm
    dist = prf_object.screen_distance_cm

    cmSize = np.array(degrees) * dist * 0.017455
    return round(cmSize * scrSizePix[0] / float(scrWidthCm), 0)


# get onset times from select_stims
def get_onset_idx(design):

    idx = []
    for ii in range(design.shape[-1]):

        if np.amax(design[..., ii]) != 0:
            idx.append(ii)

    idx = np.array(idx)

    return idx


# create design matrix by selecting
def select_stims(settings_dict, stim_library, frames=225, baseline_frames=15, randomize=False, shuffle=True, verbose=False, TR=1.5, return_onsets=False, settings_fn=None):

    """select_stims

    Function to create a fictitious design matrix based on the 'lib'-variable containing a library of 4 different types of bar stimuli: 'hori' (horizontal), 'vert' (vertical) 'rot_45' (horizontal bars rotated 45 degrees counterclockwise), and 'rot_135' (horizontal bars rotated 135 degrees counterclockwise). Using a 'settings'-dictionary, you can specify how many and which of each of these stimuli you want.

    Parameters
    ----------
    settings_dict: dict
        dictionary containing the number of stimuli and index of stimuli (as per the library) that you would like to include. An example is:

        >>> use_settings = {'hori': [0, 18],
        >>>                 'vert': [4, 25],
        >>>                 'rot_45': [4, 18],
        >>>                 'rot_135': [0, 5]}
        
        this selects 0 horizontal stimuli, 4x the 25th vertical stimulus in 'lib', 4x the 18th sti-
        mulus of the 45 degree rotated stimuli, and 0 of the 135 degree rotated stimuli.
    stim_library: dict
        library containing different types of stimuli. Can be create with :func:`linescanning.prf.create_stim_library`
    frames: int 
        number of frames your design matrix should have. Should be the same as your functional data in terms of TR (default = 225)!
    baseline_frames: int 
        number of baseline frames before stimulus presentation begins. Only needs to be specified if 'randomize' is set to False (default = 15).
    randomize: bool 
        boolean whether you want to completely randomize your experiment or not. Generally you'll want to set this to false, to create a regular intervals between stimuli (default = False).
    shuffle: bool 
        if randomize is turned off, we can still shuffle the order of the stimulus presentations. This you can do by setting shuffle to True (default = True)
    verbose: bool 
        Set to True if you want some messages along the way (default = False)

    Returns
    ----------
    numpy.ndarray
        <n_pix,n_pix,frames> numpy array representing your design
    """

    # except:
    #     raise ValueError(f"Could not open 'stimulus_library.pkl'-file in {pname}")

    lib = stim_library.copy()
    
    onset_idx = None
    index_shuf = None

    try:
        hori        = settings_dict['hori']
        vert        = settings_dict['vert']
        rot_45      = settings_dict['rot_45']
        rot_135     = settings_dict['rot_135']
        filled      = settings_dict['filled']
        conc        = settings_dict['conc']
    except:
        print(select_stims.__doc__)
        raise Exception("Could not extract relevant information; check the doc for an example of settings_dict")

    total_stims = np.sum([settings_dict[ii][0] for ii in settings_dict.keys()])

    if verbose:
        print(f"Creating design with {total_stims} stimuli and length of {frames}")

    stims = []

    # get stims for each orientation
    for ori in settings_dict.keys():

        if verbose:
            print(f" Using {settings_dict[ori][0]} stimuli of '{ori}'-orientation")

        if settings_dict[ori][0] != 0:

            if isinstance(settings_dict[ori][1], np.ndarray) or isinstance(settings_dict[ori][1], list):
                idc_list = settings_dict[ori][1]

                onset_idx = list(np.tile(idc_list, settings_dict[ori][0]))

                for ii in range(settings_dict[ori][0]):
                    stims.append(np.array([lib[ori][ii] for ii in idc_list]).T)

                stims = np.concatenate(stims, axis=-1)

            else:
                if not settings_dict[ori][1]:
                    idc = np.random.choice(33,1)[0]
                else:
                    idc = settings_dict[ori][1]

                for ii in range(settings_dict[ori][0]):
                    stims.append(lib[ori][idc])

    if verbose:
        if isinstance(stims, np.ndarray):
            print(f"Stimuli have shape {stims.shape}")
        else:
            print(f"Stimuli have shape {len(stims)}")

    # fill to end of frames
    nr_stims = len(stims)
    fill_stims = frames - nr_stims

    # completely randomize trials and baseline
    if randomize:
        for ii in range(fill_stims):
            stims.append(np.zeros_like(stims[0]))

        stims = np.array(stims)

        idc = np.arange(0,frames)
        np.random.shuffle(idc)
        stims = stims[idc,:,:]

        return stims.T

    # make regular intervals between stimuli
    else:
        if isinstance(stims, np.ndarray):
            ref = stims[...,0]
            actual_stims = stims.shape[-1]
        else:
            ref = stims[0]
            actual_stims = total_stims

        # do ISI calculation as in exptools experiment:
        import yaml

        # from exptools2.core.session._load_settings()
        with open(settings_fn, 'r', encoding='utf8') as f_in:
            settings = yaml.safe_load(f_in)

        total_iti_duration = actual_stims * settings['design'].get('mean_iti_duration')
        min_iti_duration = total_iti_duration - settings['design'].get('total_iti_duration_leeway'),
        max_iti_duration = total_iti_duration + settings['design'].get('total_iti_duration_leeway')

        def return_itis(mean_duration, minimal_duration, maximal_duration, n_trials):
            itis = np.random.exponential(scale=mean_duration-minimal_duration, size=n_trials)
            itis += minimal_duration
            itis[itis>maximal_duration] = maximal_duration
            return itis

        nits = 0
        itis = return_itis(
            mean_duration=settings['design'].get('mean_iti_duration'),
            minimal_duration=settings['design'].get('minimal_iti_duration'),
            maximal_duration=settings['design'].get('maximal_iti_duration'),
            n_trials=actual_stims)
        while (itis.sum() < min_iti_duration) | (itis.sum() > max_iti_duration):
            itis = return_itis(
                mean_duration=settings['design'].get('mean_iti_duration'),
                minimal_duration=settings['design'].get('minimal_iti_duration'),
                maximal_duration=settings['design'].get('maximal_iti_duration'),
                n_trials=actual_stims)
            nits += 1

        if verbose:
            print(f'ITIs created with total ITI duration of {itis.sum()} after {nits} iterations')

        # fetch new nr of frames based on ITIs
        frames = int(itis.sum()/TR)

        if verbose:
            print(f"New nr of frames = {frames} based on ITIs")

        # new nr of frames to fill
        fill_stim = frames - actual_stims

        # fetch how many frames do not have stimulus
        frames_no_stim = fill_stims - baseline_frames

        # fetch how many frames should be presented between each stimulus
        chunk_size = int(round((frames_no_stim)/actual_stims, 0))

        # create (n_pix,n_pix,baseline_frames) array
        baseline = np.dstack([np.zeros_like(ref) for i in range(baseline_frames)])

        # initialize list
        chunk_list = []

        # loop through stimuli
        for ii in np.arange(0,actual_stims):

            if isinstance(stims, np.ndarray):
                stim_array = stims[...,ii]
            else:
                stim_array = stims[ii]

            # create (n_pix,n_pix,chunk_size) array
            isi_chunk = np.dstack([np.zeros_like(ref) for i in range(int(itis[ii]/TR))])

            # create chunk with stimulus + isi
            add_chunk = np.dstack((stim_array[..., np.newaxis], isi_chunk))

            # if verbose:
            #     print(f" Added chunk nr {ii} with shape: {add_chunk.shape}")

            chunk_list.append(add_chunk)

        # shuffle the order of stimulus presentations
        if shuffle:

            if onset_idx:

                # I do this so we can keep track of which stimulus in the list was presented
                chunk_list_shuff = []
                stim_shuff = []
                index_shuf = np.arange(0,len(onset_idx))
                random.shuffle(index_shuf)

                for i in index_shuf:
                    chunk_list_shuff.append(chunk_list[i])
                    stim_shuff.append(onset_idx[i])

            else:
                random.shuffle(chunk_list)
                chunk_list_shuff = chunk_list

        # add baseline at the beginning
        chunk_list_shuff.insert(0,baseline)

        # concatenate into a numpy array
        design = np.concatenate(chunk_list_shuff, axis=-1)

        # if verbose:
        #     print(f"Current design shape = {design.shape}")

        # # add or cut frames at the end to meet the nr of frames
        # if design.shape[-1] > frames:
        #     if verbose:
        #         print(f"Removing {design.shape[-1]-frames} frames")

        #     design = design[:,:,:frames]
        # elif design.shape[-1] < frames:
        #     if verbose:
        #         print(f"Adding {frames-design.shape[-1]} frames")
        #     design = np.dstack((design, np.dstack([np.zeros_like(ref) for i in range(frames-design.shape[-1])])))

        if verbose:
            print(f"Design has shape: {design.shape}")

        if return_onsets:
            # # create onset df
            # try:
            if verbose:
                print("Creating onset dataframe")

            onset_frames = get_onset_idx(design)
            onset_times = onset_frames*TR

            onset_dict = {'onset': onset_times,
                          'event_type': stim_shuff}

            onset_df = pd.DataFrame(onset_dict)

            if verbose:
                print("Done")

            return design, onset_df

            # except:
            #     if verbose:
            #         print("Not returning onset dataframe")


        else:

            if verbose:
                print("Done")

            return design

def prf_neighbouring_vertices(subject, hemi='lh', vertex=None, prf_params=None, compare=False):

    """prf_neighbouring_vertices

    Function to extract pRF-parameters from vertices neighbouring a particular vertex of interest. Internally uses "call_prfinfo" to extract the parameters, so it assumes a particular kind of data structure (for now..)

    Parameters
    ----------
    subject: str
        string used as subject ID (e.g., 'sub-001')
    hemi: str 
        string ('lh'|'rh') used to annotate the hemisphere which is required to index the correct pRF parameters
    vertex: int
        vertex from which to extract neighbouring vertices
    prf_params: str, np.ndarray
        if you do not want to depend on fixed data structures, you can specify the pRF-parameters directly with a string pointing to a numpy-file or the numpy-array itself
    compare: bool
        if True, it will compare the parameters of neighbouring vertices with the parameters of <vertex>

    Returns
    ----------
    list
        list of dictionaries containing the pRF-parameters for each vertex

    list
        list of the neighbouring vertices

    Notes
    ----------
    Note that the last element in both is the information about the requested vertex itself!
    """

    # fetch the surface used for vertex extraction
    surf = utils.get_file_from_substring(f"{hemi}.fiducial", opj(os.environ['SUBJECTS_DIR'], subject, 'surf'))

    # define the command to fetch neighbouring vertices
    cmd_2 = ('mris_info', '--vx', str(vertex), surf)

    # read the output
    L = utils.decode(subprocess.check_output(cmd_2)).splitlines()

    # read number of neighbours
    n_neighbours = int(L[4].split(' ')[-1])

    if n_neighbours < 1:
        raise ValueError("Vertex only has a few neighbours..?")

    # loop through neighbouring vertices to extract their vertices
    verts = []
    list_neighbours = np.arange(0,n_neighbours)
    for ii in list_neighbours:
        line = list(filter(None, L[5+ii].split(' ')))
        if line[0] == 'nbr' and line[1] == str(list_neighbours[ii]):
            verts.append(int(line[2]))

    verts.append(vertex)

    # extract pRF parameters for each vertex
    prfs = {}
    for ii in verts:
        cmd_1 = ('call_prfinfo', '-s', str(subject), '-v', str(ii), '-h', str(hemi))
        prfs[ii] = ast.literal_eval(utils.decode(subprocess.check_output(cmd_1)).splitlines()[0])

    if compare:
        for ii in verts[:-1]:
            print(f'Vertex {ii}:')
            for el in ['x', 'y', 'size']:
                x = round((prfs[ii][el]/prfs[list(prfs.keys())[-1]][el])*100,2)
                print(f" {el} = {x}% similar")

    return prfs, verts

def create_line_prf_matrix(log_dir, 
                           nr_trs=None,
                           stim_at_half_TR=False,
                           TR=0.105, 
                           n_pix=270, 
                           deleted_first_timepoints=0, 
                           deleted_last_timepoints=0, 
                           stim_duration=1, 
                           baseline_before=24,
                           baseline_after=24,
                           skip_first_img=True,
                           verbose=False,
                           dm_edges_clipping=[0,0,0,0]):

    """create_line_prf_matrix

    Adaptation of get_prfdesign. Uses a similar principle, but rather than having an image per TR, we have images with onset times. With this function we can fill in an empty design matrix with size of n_samples to create the design matrix. This holds for cases where the stimulus duration is modulo the repetition time (e.g., stim duration = 0.315, repetition time = 0.105). In cases where the stimulus duration is incongruent with the repetition time, we can loop through the repetition times and find, in the onsets dataframe, the closest time and select that stimulus for our design matrix. This later case works for the partial pRF experiment that we run with 3D-EPI 1.1s TR acquisition, and should work with the line-scanning acquisition as well.

    Parameters
    ----------
    log_dir: str
        path to the log-directory as per the output of exptools experiments. We should have a "Screenshots"-directory in there and a *.tsv*-file containing the onset times
    nr_trs: int
        number of TRs in an acquisition run; this one is required if the stimulus duration is not modulo the repetition time
    TR: float (default = 0.105s)
        repetition time of scan acquisition. Needed to convert the onset times from seconds to TRs
    stim_at_half_TR: bool
        set to True if you want the stimulus halfway through the TR. This flag can be used in combination with <nr_trs>, when the stimulus duration is not modulo the repetition time
    deleted_first_timepoints: int 
        number of volumes to skip in the beginning; should be the same as used in ParseFunc-file and/or ParseFuncFile (if used in concert) (default = 0)
    deleted_last_timepoints: int
        number of volumes to skip at the end; should be the same as used in ParseFuncFile and/or ParseFuncFile (if used in concert) (default = 0)
    n_pix: int 
        size determinant of the design matrix. Note, <n_pix> is hardcoded as well in the function itself because I ran stuff on my laptop and that screen has a different size than the stimulus computer @Spinoza (default = 270px)
    stim_duration: float
        length of stimulus presentation (in seconds; default = 1s)
    skip_first_img: boolean 
        depending on how you coded the screenshot capturing, the first empty screen is something also captured. With this flag we can skip that image (default = True).
    dm_edges_clipping: list
        people don't always see the entirety of the screen so it's important to check what the subject can actually see by showing them the cross of for instance the BOLD-screen (the matlab one, not the linux one) and clip the image accordingly
        
    Returns
    ----------
    np.ndarray
        design matrix with size <n_pix, n_pix, n_samples>
    """

    screenshot_path = utils.get_file_from_substring("Screenshot", log_dir)
    image_list = os.listdir(screenshot_path)
    image_list.sort()
    tr_in_duration = int(stim_duration/TR)
    
    # whether the stimulus duration is modulo the repetition time or not differs. In case it is, 
    # we can use the loop below. If not, we need to loop through the TRs, find the onset time 
    # closest to it, and select that particular image

    # get first image to get screen size
    img = (255*mpimg.imread(opj(screenshot_path, image_list[0]))).astype('int')

    # there is one more MR image than screenshot
    design_matrix = np.zeros((img.shape[0], img.shape[0], 1+len(image_list)))
    if not round(stim_duration % TR, 5) > 0:


        point = 0
        for ii in image_list:
            img_fn = opj(screenshot_path, ii)
            img = (255*mpimg.imread(img_fn)).astype('int')

            if img.shape[0] != img.shape[1]:
                offset = int((img.shape[1]-img.shape[0])/2)
                img = img[:, offset:(offset+img.shape[0])]

            # first binarize, then downsample
            design_matrix[:, :, point:point+tr_in_duration][np.where(((img[...,0] == 0) & (
            img[...,1] == 0)) | ((img[...,0] == 255) & (img[...,1] == 255)))] = 1

            point += tr_in_duration

        # downsample
        dm_resampled = np.zeros((n_pix, n_pix, design_matrix.shape[-1]))
        
        for img in range(design_matrix.shape[-1]):
            dm_resampled[...,img] = utils.resample2d(design_matrix[...,img], n_pix)

        #clipping edges
        #top, bottom, left, right
        dm_resampled[:dm_edges_clipping[0], :, :] = 0
        dm_resampled[(dm_resampled.shape[0]-dm_edges_clipping[1]):, :, :] = 0
        dm_resampled[:, :dm_edges_clipping[2], :] = 0
        dm_resampled[:, (dm_resampled.shape[0]-dm_edges_clipping[3]):, :] = 0

        # deal with baseline and deleted volumes
        baseline_before = np.zeros((n_pix,n_pix,int(baseline_before/TR)))
        if deleted_first_timepoints != 0:
            baseline_before = baseline_before[...,deleted_first_timepoints:]
        
        print(f"Baseline before has shape: {baseline_before.shape}")

        baseline_after = np.zeros((n_pix,n_pix,int(baseline_after/TR)))    
        if deleted_last_timepoints != 0:
            baseline_after = baseline_after[...,:-deleted_last_timepoints]
        
        print(f"Design itself has shape: {dm_resampled.shape}")
        print(f"Baseline after has shape: {baseline_after.shape}")
        dm = np.dstack((baseline_before, dm_resampled, baseline_after))
        
        return dm
    else:

        if verbose:
            print("Reading onset times from log-file")

        # get the onsets
        onsets = dataset.ParseExpToolsFile(utils.get_file_from_substring(".tsv", log_dir), 
                                           TR=TR, 
                                           deleted_first_timepoints=deleted_first_timepoints, 
                                           use_bids=False,
                                           verbose=verbose,
                                           phase_onset=0)
    
        if verbose:
            print("Creating design matrix (can take a few minutes with thousands of TRs)")
        
        trial_df = onsets.df_onsets.copy()
        for tr in range(nr_trs):
    
            # find time at the middle of TR
            if stim_at_half_TR:
                tr_in_sec = (tr * onsets.TR)+0.5*onsets.TR
            else:
                tr_in_sec = (tr * onsets.TR)

            # ix now represents the trial ID in the onset dataframe, which starts at the first 't'
            ix,_ = utils.find_nearest(trial_df['onset'].values, tr_in_sec)
            
            # zero-pad number https://stackoverflow.com/questions/2189800/how-to-find-length-of-digits-in-an-integer
            zfilling = len(str(len(os.listdir(screenshot_path))))
            img_number = str(ix).zfill(zfilling)
            try:
                image_file = utils.get_file_from_substring(f"{img_number}.png", screenshot_path)
            except:
                image_file = None
            
            if image_file != None:
                
                img = (255*mpimg.imread(image_file)).astype('int')

                if img.shape[0] != img.shape[1]:
                    offset = int((img.shape[1]-img.shape[0])/2)
                    img = img[:, offset:(offset+img.shape[0])]

                # binarize image into dm matrix
                # assumes: standard RGB255 format; only colors present in image are black, white, grey, red, green.
                design_matrix[:, :, tr][np.where(((img[:, :, 0] == 0) & (
                    img[:, :, 1] == 0)) | ((img[:, :, 0] == 255) & (img[:, :, 1] == 255)))] = 1
        
        # downsample
        dm_resampled = np.zeros((n_pix, n_pix, design_matrix.shape[-1]))
        
        for img in range(design_matrix.shape[-1]):
            dm_resampled[...,img] = utils.resample2d(design_matrix[...,img], n_pix)

        #clipping edges
        #top, bottom, left, right
        dm_resampled[:dm_edges_clipping[0], :, :] = 0
        dm_resampled[(dm_resampled.shape[0]-dm_edges_clipping[1]):, :, :] = 0
        dm_resampled[:, :dm_edges_clipping[2], :] = 0
        dm_resampled[:, (dm_resampled.shape[0]-dm_edges_clipping[3]):, :] = 0

        return dm_resampled

def create_stim_library(n_pix, prf_stim, range_around_center=[40,40], concentricity=0.5, stim_factor=4, beam_size=26):

    """Create library of stimuli to build a design matrix with"""

    # create filled stimuli
    stims_fill, stims_fill_sizes = make_stims(n_pix, prf_stim, factr=stim_factor)

    # create concentric stimuli
    stims_conc, stims_fill_conc = make_stims(n_pix, prf_stim, concentric=True, concentric_size=concentricity)    

    # internally prf.make_stims calculates the number of stimuli. Use that number for the other stimuli as well
    n_stim = len(stims_fill)

    # create aperture around stimuli
    aperture = create_circular_mask(n_pix,n_pix)

    # center bars
    beam_locations = np.linspace(n_pix//2-range_around_center[0],n_pix//2+range_around_center[0],num=n_stim).astype(int)

    # Create horizontal bars
    horizontal = [np.zeros((n_pix, n_pix)) for n in beam_locations]
    for pp, stim in enumerate(horizontal):

        total = beam_locations[pp]+beam_size

        if total <= n_pix:
            stim[beam_locations[pp]:beam_locations[pp]+beam_size,:] = 1
        else:
            remains = total - n_pix
            stim[beam_locations[pp]:(beam_locations[pp]+beam_size)-remains,:] = 1

        stim[~aperture.astype(bool)] = 0

    # create vertical bars
    vertical = [np.zeros((n_pix, n_pix)) for n in beam_locations]
    for pp, stim in enumerate(vertical):

        total = beam_locations[pp]+beam_size

        if total <= n_pix:
            stim[:,beam_locations[pp]:beam_locations[pp]+beam_size] = 1
        else:
            remains = total - n_pix
            stim[:,beam_locations[pp]:(beam_locations[pp]+beam_size)-remains] = 1

        stim[~aperture.astype(bool)] = 0

    # rotate vertical bars by 45 degrees
    rot45 = []
    for pp, stim in enumerate(vertical):

        stim_rot = rotate(stim, 45, reshape=False, order=1)
        rot45.append(stim_rot)

    # rotate vertical bars by 135 degrees
    rot135 = []
    for pp, stim in enumerate(vertical):

        stim_rot = rotate(stim, 135, reshape=False, order=1)
        rot135.append(stim_rot)

    lib = {'hori': horizontal,
           'vert': vertical,
           'rot_45': rot45,
           'rot_135': rot135,
           'filled': stims_fill,
           'conc': stims_conc}

    return lib


def distance_centers(prf1,prf2):

    x1,y1 = prf1[0],prf1[1]
    x2,y2 = prf2[0],prf2[1]
    
    # Calculating distance
    return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2) * 1.0)
 

def generate_model_params(model='gauss', dm=None, TR=1.5, outputdir=None, settings=None, fit_hrf=False):

    """generate_model_params

    Function to generate a yaml-file with parameters to use. Utilizes the analysis_settings.yml as basis, and adds information along the way, such as grid specifications and model type.

    Parameters
    ----------
    model: str
        model type to utilize.
    dm: numpy.ndarray
        design matrix, n_pix X n_pix X volumes. Needed to create the pRF stimulus
    TR: float
        repetition time; can be fetched from gifti-file; default = 1.5s
    outputdir: str
        path-like object pointing to the prf-directory so each analysis comes with its own analysis-settings file  
    settings: str
        load the settings from a settings file generate earlier
    fit_hrf: bool
        Whether or not to fit two extra parameters for hrf derivative and dispersion. The default is False.

    Returns
    ----------
    str
        saves a settings-file in the output directory

    yml
        returns the content of settings-file

    prfpy.stimlus.PRFStimulus2D
        pRF object
    """
    
    if settings != None:
        yml_file = settings
    else:
        yml_file = utils.get_file_from_substring("prf_analysis.yml", opj(os.path.dirname(os.path.dirname(utils.__file__)), 'misc'))

    with open(yml_file) as file:
        
        data = yaml.safe_load(file)

        prf_stim = stimulus.PRFStimulus2D(screen_size_cm=data['screen_size_cm'],
                                          screen_distance_cm=data['screen_distance_cm'],
                                          design_matrix=dm,
                                          TR=TR)

        ss = prf_stim.screen_size_degrees

        # define grids
        sizes, eccs, polars = data['max_ecc_size'] * np.linspace(0.125,1,data['grid_nr'])**2, \
                              data['max_ecc_size'] * np.linspace(0.05,1,data['grid_nr'])**2, \
                              np.linspace(0, 2*np.pi, data['grid_nr'], endpoint=False)

        grids = {'screensize_degrees': float(ss), 'grids': {'sizes': [float(item) for item in sizes], 'eccs': [float(item) for item in eccs], 'polars': [float(item) for item in polars]}}
        
        # define bounds
        if model == "gauss":
            gauss_bounds = [(-1.5*data['max_ecc_size'], 1.5*data['max_ecc_size']),  # x
                            (-1.5*data['max_ecc_size'], 1.5*data['max_ecc_size']),  # y
                            (data['eps'], 1.5*ss),                                  # prf size
                            (-80, 80),                                              # prf amplitude
                            (-60, 60)]                                              # bold baseline

            bounds = {'bounds': {'x': list(gauss_bounds[0]), 
                                 'y': list(gauss_bounds[1]), 
                                 'size': [float(item) for item in gauss_bounds[2]], 
                                 'prf_ampl': list(gauss_bounds[3]), 
                                 'bold_bsl': list(gauss_bounds[4])}}

        else:
            norm_bounds = [(-1.5*data['max_ecc_size'], 1.5*data['max_ecc_size']),   # x
                           (-1.5*data['max_ecc_size'], 1.5*data['max_ecc_size']),   # y
                           (data['eps'], 1.5*ss),                                   # prf size
                           (-80, 80),                                               # prf amplitude
                           (-60, 60),                                               # bold baseline
                           (0, 1000),                                               # surround amplitude
                           (data['eps'], 3*ss),                                     # surround size
                           (0, 1000),                                               # neural baseline
                           (1e-6, 1000)]                                            # surround baseline

            bounds = {'bounds': {'x': list(norm_bounds[0]), 
                                 'y': list(norm_bounds[1]), 
                                 'size': [float(item) for item in norm_bounds[2]], 
                                 'prf_ampl': list(norm_bounds[3]), 
                                 'bold_bsl': list(norm_bounds[4]),
                                 'surr_ampl': list(norm_bounds[5]),
                                 'surr_size': [float(item) for item in norm_bounds[6]],
                                 'neur_bsl': list(norm_bounds[7]),
                                 'surr_bsl': list(norm_bounds[8])}}

        if fit_hrf:
            bounds['bounds']["hrf_deriv"] = [0,10]
            bounds['bounds']["hrf_disp"] = [0,0]

        # update settings file if we've generated a new one
        if settings == None:
            data.update(bounds)
            data.update(grids)
            data.update({'model': model})
            data.update({'TR': TR})

        date = datetime.now().strftime("%Y%m%d")

    if settings == None:

        if isinstance(outputdir, str):
            fname = opj(outputdir, f'{date}_model-{model}_desc-settings.yml')
            with open(fname, 'w') as yml_file:
                yaml.safe_dump(data, yml_file)

            return data, fname, prf_stim
        
        else:
            return data, settings, prf_stim

    else:
        return data, settings, prf_stim


class pRFmodelFitting(): 
    """pRFmodelFitting

    Main class to perform all the pRF-fitting. As of now, the Gaussian and Divisive Normalization models are implemented. For each model, an analysis-file is produced an stored in <output_dir> for later reference.

    Parameters
    ----------
    data: numpy.ndarray
        <voxels,time> numpy array | when reading in the data later again, the format must be <time,voxels>. This is highly annoying, but it seems to be required for predictions to work properly.
    design_matrix: numpy.ndarray
        <n_pix, n_pix, time> numpy array containing the paradigm
    TR: float
        repetition time of acquisition; required for the analysis file. If you're using gifti-input, you can fetch the TR from that file with `gifti = linescanning.utils.ParseGiftiFile(gii_file).TR_sec`. If you're file have been created with fMRIprep or call_vol2fsaverage, this should work.
    model: str
        as of now, either 'gauss', or 'norm' is accepted
    stage: str
        can either be 'grid' or 'grid+iter', in combination with the <model> flag
    output_dir: str
        directory to store all files in; should be somewhere in <project>/derivatives/prf/<subject>
    output_base: str
        basename for output files; should be something like <subject>_<ses-?>_<task-?>
    write_files: bool
        save files (True) or not (False). Should be used in combination with <output_dir> and <output_base>
    old_params: tuple, optional
        A tuple of a numpy.ndarray and string describing the nature of the old parameters. Can be inserted at any stage of the fitting. Generally you'd insert Gaussian iterfit parameters before fitting the normalization model.

        >>> old_params = (<voxels, parameters>, 'gauss+iter')
        >>> old_params = (<voxels, parameters>, 'gauss+grid')
    hrf: np.ndarray
        <1,time_points> describing the HRF. Can be created with :func:`linescanning.glm.double_gamma`, then add an
        axis before the timepoints:

        >>> dt = 1
        >>> time_points = np.linspace(0,36,np.rint(float(36)/dt).astype(int))
        >>> hrf_custom = linescanning.glm.double_gamma(time_points, lag=6)
        >>> hrf_custom = hrf_custom[np.newaxis,...]
    fit_hrf: bool
        fit the HRF with the pRF-parameters as implemented in `prfpy`
    nr_jobs: int, optional
        Set the number of jobs. By default 1.
    verbose: bool
        print messages to the terminal

    Returns
    ----------
    npy-files
        For each model (gauss/normalization), a npy-file with the grid/iterative parameters and an npy-file with the predictions (*only* for iterative stage!). The format of these files is as follows: <output_dir>/<output_base>_model-<gauss|norm>_stage-<grid|iter>_desc-<prf_params|predictions>.npy

    Example
    ----------
    >>> from linescanning.prf import pRFmodelFitting
    >>> fitting = pRFmodelFitting(func, design_matrix=dm, model='gauss')

    >>> # we can use this class to read in existing parameter files
    >>> prf_pfov = opj(prf_new, "sub-003_ses-3_task-pRF_acq-3DEPI_model-norm_stage-iter_desc-prf_params.npy")
    >>> yml = utils.get_file_from_substring("settings", prf_new)
    >>> if isinstance(yml, list):
    >>>     yml = yml[-1]
    >>> #
    >>> modelling_pfov = prf.pRFmodelFitting(partial_nan.T,
    >>>                                      design_matrix=design_pfov,
    >>>                                      settings=yml,
    >>>                                      model="norm",
    >>>                                      output_dir=prf_new,
    >>>                                      output_base="sub-003_ses-3_task-pRF_acq-3DEPI")
    >>> #
    >>> modelling_pfov.load_params(np.load(prf_pfov), model='norm', stage='iter')
    """

    def __init__(self, 
                 data, 
                 design_matrix=None, 
                 TR=1.5, 
                 model="gauss", 
                 stage="grid+iter", 
                 output_dir=None, 
                 write_files=False, 
                 output_base=None,
                 old_params=None,
                 verbose=True,
                 hrf=None,
                 fit_hrf=False,
                 settings=None,
                 nr_jobs=1000,
                 rsq_threshold=None,
                 prf_stim=None,
                 model_obj=None,
                 **kwargs):
    

        self.data           = data
        self.design_matrix  = design_matrix
        self.TR             = TR
        self.model          = model
        self.stage          = stage
        self.output_dir     = output_dir
        self.output_base    = output_base
        self.write_files    = write_files
        self.settings_fn    = settings
        self.verbose        = verbose
        self.hrf            = hrf
        self.old_params     = old_params
        self.nr_jobs        = nr_jobs
        self.prf_stim       = prf_stim
        self.model_obj      = model_obj
        self.rsq_threshold  = rsq_threshold
        self.fit_hrf        = fit_hrf
        self.__dict__.update(kwargs)

        #----------------------------------------------------------------------------------------------------------------------------------------------------------
        # Fetch the settings
        self.settings, self.settings_file, self.prf_stim_ = generate_model_params(model=self.model, 
                                                                                dm=self.design_matrix, 
                                                                                outputdir=self.output_dir, 
                                                                                TR=self.TR,
                                                                                settings=self.settings_fn,
                                                                                fit_hrf=self.fit_hrf)

        # overwrite rsq-threshold from settings file
        if self.rsq_threshold != None:
            if self.verbose:
                print(f"Setting rsq-threshold to user-defined value: {self.rsq_threshold}")
            self.rsq = self.rsq_threshold
        else:
            if self.verbose:
                print(f"Setting rsq-threshold to default value: {self.settings['rsq_threshold']}")
            self.rsq = self.settings['rsq_threshold']

        # check if we got a pRF-stim object
        if self.prf_stim != None:
            if self.verbose:
                print("Using user-defined pRF-stimulus object")
            self.prf_stim = self.prf_stim
        else:
            self.prf_stim = self.prf_stim_
        
        # read the settings
        if isinstance(self.settings_fn, str):
            if self.verbose:
                print(f"Using settings file: {self.settings_fn}")
    
        if self.model.lower() != "gauss" and self.model.lower() != "norm":
            raise ValueError(f"Model specification needs to be either 'gauss' or 'norm'; got {model}.")

        #----------------------------------------------------------------------------------------------------------------------------------------------------------
        # whichever model you run, run the Gaussian first

        ## Define model
        self.gaussian_model = Iso2DGaussianModel(stimulus=self.prf_stim,
                                                 filter_predictions=self.settings['filter_predictions'],
                                                 filter_type='sg',
                                                 hrf=self.hrf,
                                                 filter_params={'window_length': self.settings['filter_window_length'],
                                                                'polyorder': self.settings['filter_polyorder']})

        self.norm_model = Norm_Iso2DGaussianModel(stimulus=self.prf_stim,
                                                  filter_predictions=self.settings['filter_predictions'],
                                                  filter_type='sg',
                                                  hrf=self.hrf,                                                  
                                                  filter_params={'window_length': self.settings['filter_window_length'],
                                                                 'polyorder': self.settings['filter_polyorder']})

        if self.model_obj != None:
            if self.verbose:
                print(f"Setting {self.model_obj} as '{model}_model'-attribute")
            setattr(self, f'{model}_model', self.model_obj)

    def fit(self):

        ## Initiate fitter
        # check whether we got old parameters so we can skip Gaussian fit:
        if isinstance(self.old_params, np.ndarray) or isinstance(self.old_params, str):
            if isinstance(self.old_params, np.ndarray):
                pass
            elif isinstance(self.old_params, str):
                self.old_params = np.load(self.old_params)
            else:
                raise ValueError(f"old_params must be a string pointing to a npy-file or a np.ndarray, not '{type(self.old_params)}'")

            self.gaussian_fitter = Iso2DGaussianFitter(data=self.data, 
                                                       model=self.gaussian_model, 
                                                       fit_css=False,
                                                       fit_hrf=self.fit_hrf)

            # set inserted params as gridsearch_params and iterative_search_params 
            self.gaussian_fitter.gridsearch_params = self.old_params.copy()         # needed for the rsq-mask
            self.gaussian_fitter.iterative_search_params = self.old_params.copy()   # actual parameters
            
            # set gaussian_fitter as previous_gaussian_fitter
            self.previous_gaussian_fitter = self.gaussian_fitter

        if not hasattr(self, "previous_gaussian_fitter"):

            ### add check whether we need to transpose data
            self.gaussian_fitter = Iso2DGaussianFitter(data=self.data, 
                                                       model=self.gaussian_model, 
                                                       fit_css=False,
                                                       fit_hrf=self.fit_hrf)
            
            if self.verbose:
                print("Starting gauss grid fit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

            ## start grid fit
            start = time.time()
            self.gaussian_fitter.grid_fit(ecc_grid=self.settings['grids']['eccs'],
                                          polar_grid=self.settings['grids']['polars'],
                                          size_grid=self.settings['grids']['sizes'],
                                          pos_prfs_only=self.settings['pos_prfs_only'])
            
            elapsed = (time.time() - start)

            self.gauss_grid = utils.filter_for_nans(self.gaussian_fitter.gridsearch_params)
            if self.verbose:
                print("Gaussian gridfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+
                        ". voxels/vertices above "+str(self.rsq)+": "+str(np.sum(self.gauss_grid[:, -1]>self.rsq))+" out of "+
                        str(self.gaussian_fitter.data.shape[0]))
                print(f"Gridfit took {str(timedelta(seconds=elapsed))}")
                print("Mean rsq>"+str(self.rsq)+": "+str(round(np.nanmean(self.gauss_grid[self.gauss_grid[:, -1]>self.rsq, -1]),2)))
            
            if self.write_files:
                self.save_params(model="gauss", stage="grid", predictions=False)

            #----------------------------------------------------------------------------------------------------------------------------------------------------------
            # Check if we should do Gaussian iterfit        
            ## Run iterative fit with Gaussian model
            if self.stage == "grid+iter":

                start = time.time()
                if self.verbose:
                    print("Starting gauss iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

                # fetch bounds
                self.gauss_bounds = self.fetch_bounds(model='gauss') 

                # fit
                self.gaussian_fitter.iterative_fit(rsq_threshold=self.rsq, bounds=self.gauss_bounds)

                # print summary
                elapsed = (time.time() - start)              
                self.gauss_iter = utils.filter_for_nans(self.gaussian_fitter.iterative_search_params)
                if self.verbose:
                    print("Gaussian iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(self.rsq)+": "+str(round(np.nanmean(self.gauss_iter[self.gaussian_fitter.rsq_mask, -1]),2)))
                    print(f"Iterfit took {str(timedelta(seconds=elapsed))}")

                # save intermediate files
                if self.write_files:
                    self.save_params(model="gauss", stage="iter", predictions=True)


        #----------------------------------------------------------------------------------------------------------------------------------------------------------
        # Check if we should do DN-model
        if self.model.lower() == "norm":            
            
            ## Define settings/grids/fitter/bounds etcs
            settings, settings_file, self.prf_stim = generate_model_params(model='norm', dm=self.design_matrix, outputdir=self.output_dir, fit_hrf=self.fit_hrf)
            
            if self.verbose:
                print(f"Using settings file: {settings_file}")

            # overwrite rsq-threshold from settings file
            if self.rsq_threshold != None:
                self.rsq = self.rsq_threshold
            else:
                self.rsq = self.settings['rsq_threshold']                

            
            # define fitter; previous_gaussian_fitter can be specified in kwargs
            if not hasattr(self, "previous_gaussian_fitter"):
                # check if we have gaussian fit
                if hasattr(self, "gaussian_fitter"):
                    self.norm_fitter = Norm_Iso2DGaussianFitter(self.norm_model, self.data, fit_hrf=self.fit_hrf, previous_gaussian_fitter=self.gaussian_fitter)
                else:
                    # we might have manually injected parameters
                    self.norm_fitter = Norm_Iso2DGaussianFitter(self.norm_model, self.data)
            else:
                # inject existing-model object > advised when fitting the HRF
                self.norm_fitter = Norm_Iso2DGaussianFitter(self.norm_model, self.data, fit_hrf=self.fit_hrf, previous_gaussian_fitter=self.previous_gaussian_fitter)

            # read grid from settings file
            self.surround_amplitude_grid = np.array(self.settings['norm']['surround_amplitude_grid'], dtype='float32')
            self.surround_size_grid      = np.array(self.settings['norm']['surround_size_grid'], dtype='float32')
            self.neural_baseline_grid    = np.array(self.settings['norm']['neural_baseline_grid'], dtype='float32')
            self.surround_baseline_grid  = np.array(self.settings['norm']['surround_baseline_grid'], dtype='float32')

            # fetch bounds from settings > HRF bounds are automatically appended if fit_hrf=True
            self.norm_bounds = self.fetch_bounds(model='norm')

            #----------------------------------------------------------------------------------------------------------------------------------------------------------
            ## Start grid fit
            start = time.time()
            if self.verbose:
                print("Starting norm grid fit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

            # let the fitter find the parameters; easier when fitting the HRF as indexing gets complicated
            if not hasattr(self, "old_params_filt"):
                self.old_params_filt = None

            self.norm_fitter.grid_fit(self.surround_amplitude_grid, 
                                      self.surround_size_grid, 
                                      self.neural_baseline_grid,
                                      self.surround_baseline_grid, 
                                      gaussian_params=self.old_params_filt,
                                      n_batches=self.nr_jobs,
                                      verbose=self.verbose,
                                      rsq_threshold=self.rsq)

            elapsed = (time.time() - start)

            ### save grid parameters
            self.norm_grid = utils.filter_for_nans(self.norm_fitter.gridsearch_params)
            if self.verbose:
                print("Norm gridfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(self.rsq)+": "+str(round(np.nanmean(self.norm_grid[self.norm_fitter.gridsearch_rsq_mask, -1]),2)))
                print(f"Gridfit took {str(timedelta(seconds=elapsed))}")

            if self.write_files:
                self.save_params(model="norm", stage="grid")

            #----------------------------------------------------------------------------------------------------------------------------------------------------------
            # Check if we should do iterative fitting with DN model
            if self.stage == "grid+iter":
                start = time.time()
                if self.verbose:
                    print("Starting norm iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

                self.norm_fitter.iterative_fit(rsq_threshold=settings['rsq_threshold'], bounds=self.norm_bounds)
                
                elapsed = (time.time() - start)  

                ### save iterative parameters
                self.norm_iter = utils.filter_for_nans(self.norm_fitter.iterative_search_params)

                if self.verbose:
                    print("Norm iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(self.rsq)+": "+str(round(np.mean(self.norm_iter[self.norm_fitter.rsq_mask, -1]),2)))
                    print(f"Iterfit took {str(timedelta(seconds=elapsed))}")

                if self.write_files:
                    self.save_params(model="norm", stage="iter", predictions=True)

    def fetch_bounds(self, model=None):
        if model == "norm":
            bounds = [tuple(self.settings['bounds']['x']),                  # x
                      tuple(self.settings['bounds']['y']),                  # y
                      tuple(self.settings['bounds']['size']),               # prf size
                      tuple(self.settings['bounds']['prf_ampl']),           # prf amplitude
                      tuple(self.settings['bounds']['bold_bsl']),           # bold baseline
                      tuple(self.settings['bounds']['surr_ampl']),          # surround amplitude
                      tuple(self.settings['bounds']['surr_size']),          # surround size
                      tuple(self.settings['bounds']['neur_bsl']),           # neural baseline
                      tuple(self.settings['bounds']['surr_bsl'])]           # surround baseline

        elif model == "gauss":
            bounds = [tuple(self.settings['bounds']['x']),                  # x
                      tuple(self.settings['bounds']['y']),                  # y
                      tuple(self.settings['bounds']['size']),               # prf size
                      tuple(self.settings['bounds']['prf_ampl']),           # prf amplitude
                      tuple(self.settings['bounds']['bold_bsl'])]           # bold baseline   

        else:
            raise ValueError(f"Unrecognized model '{model}'")               

        if self.fit_hrf:
            bounds.append(tuple(self.settings["bounds"]['hrf_deriv']))      # HRF time derivative
            bounds.append(tuple(self.settings["bounds"]['hrf_disp']))       # HRF dispersion derivative

        return bounds

    def load_params(self, params_file, model='gauss', stage='iter', acq=None, run=None):

        """Load in a numpy array into the class; allows for quick plotting of voxel timecourses"""

        if isinstance(params_file, str):
            if params_file.endswith('npy'):
                params = np.load(params_file)
            elif params_file.endswith('pkl'):
                with open(params_file, 'rb') as input:
                    data = pickle.load(input)
                    
                params = data['pars']
                self.settings = data['settings']
                
                if self.verbose:
                    print("Reading settings from pickle-file (safest option; overwrites other settings)")

        elif isinstance(params_file, np.ndarray):
            params = params_file.copy()
        elif isinstance(params_file, pd.DataFrame):
            dict_keys = list(params_file.keys())
            if not "hemi" in dict_keys:
                # got normalization parameter file
                params = np.array((params_file['x'][0],
                                  params_file['y'][0],
                                  params_file['prf_size'][0],
                                  params_file['A'][0],
                                  params_file['bold_bsl'][0],
                                  params_file['B'][0],
                                  params_file['C'][0],
                                  params_file['surr_size'][0],
                                  params_file['D'][0],
                                  params_file['r2'][0]))
            else:
                raise NotImplementedError()
        else:
            raise ValueError(f"Unrecognized input type for '{params_file}'")

        setattr(self, f'{model}_{stage}', params)

        if acq != None:
            search_list = [model, stage, acq, "predictions.npy"]
        else:
            search_list = [model, stage, "predictions.npy"]

        if run != None:
            search_list.extend(f"run-{run}")          

        preds = utils.get_file_from_substring(search_list, os.path.dirname(self.settings_fn), return_msg=None)
        if preds != None:
            if isinstance(preds, list):
                raise ValueError(f"Found multiple instances for {search_list}: {preds}")
            else:
                print(f"Predictions: {preds}")

            setattr(self, f'{model}_{stage}_predictions', np.load(preds))

    def make_predictions(self, vox_nr=None, model='gauss', stage='iter'):
        
        if model == "gauss":
            use_model = self.gaussian_model
        elif model == "norm":
            use_model = self.norm_model

        if hasattr(self, f"{model}_{stage}"):
            params = getattr(self, f"{model}_{stage}")
            if params.ndim == 1:
                if vox_nr != None:
                    if vox_nr == "best":
                        vox,_ = utils.find_nearest(params[...,-1], np.amax(params[...,-1]))
                    else:
                        vox = vox_nr

                try:
                    params = params[vox,...]
                except:
                    params = params

                return use_model.return_prediction(*params[:-1]).T, params, vox

            elif params.ndim == 2:
                if vox_nr != None:
                    if vox_nr == "best":
                        vox,_ = utils.find_nearest(params[...,-1], np.amax(params[...,-1]))
                    else:
                        vox = vox_nr                    
                    params = params[vox,...]
                    return use_model.return_prediction(*params[:-1]).T, params, vox
                else:
                    predictions = []
                    for vox in range(params.shape[0]):
                        pars = params[vox,...]
                        predictions.append(use_model.return_prediction(*pars[:-1]).T)
                    
                    return np.squeeze(np.array(predictions), axis=-1)

            else:
                raise ValueError(f"No parameters found..")
        else:
            raise ValueError(f"Could not find {stage} parameters for {model}")

    def plot_vox(self, 
                 vox_nr=0, 
                 model='gauss', 
                 stage='iter', 
                 make_figure=True, 
                 xkcd=False, 
                 title=None, 
                 font_size=18,
                 transpose=False,
                 freq_spectrum=False,
                 freq_type='fft',
                 clip_power=None,
                 save_as=None,
                 axis_type="volumes",
                 resize_pix=None,
                 **kwargs):

        """plot real and predicted timecourses for a voxel. Also returns parameters, the numpy array representing the pRF in visual space, and timecourse of data"""

        self.prediction, params, vox = self.make_predictions(vox_nr=vox_nr, model=model, stage=stage)

        if hasattr(self, f"{model}_{stage}_predictions"):
            self.prediction = getattr(self, f"{model}_{stage}_predictions")[vox_nr]

        if make_figure:
            prf_array = make_prf(self.prf_stim, 
                                 size=params[2], 
                                 mu_x=params[0], 
                                 mu_y=params[1], 
                                 resize_pix=resize_pix)

            if freq_spectrum:
                fig = plt.figure(constrained_layout=True, figsize=(20,5))
                gs00 = fig.add_gridspec(1,3, width_ratios=[10,20,10])
            else:
                fig = plt.figure(constrained_layout=True, figsize=(15,5))
                gs00 = fig.add_gridspec(1,2, width_ratios=[10,20])

            # make plot 
            ax1 = fig.add_subplot(gs00[0])
            plotting.LazyPRF(prf_array, vf_extent=self.settings['vf_extent'], ax=ax1, xkcd=xkcd)

            # make plot 
            ax2 = fig.add_subplot(gs00[1])

            # annoying indexing issues.. lots of inconsistencies in array shapes.
            if transpose:
                tc = self.data.T[vox,...]
            else:
                tc = self.data[vox,...]

            if title != None:
                if title == "pars":
                    set_title = [round(ii,2) for ii in params]
                else:
                    set_title = title
            else:
                set_title = None

            if axis_type == "time":
                x_label = "time (s)"
                x_axis = np.array(list(np.arange(0,tc.shape[0])*self.TR))
            else:
                x_axis = None
                x_label = "volumes"

            plotting.LazyPlot([tc, self.prediction],
                              xx=x_axis,
                              color=['#cccccc', 'r'], 
                              labels=['real', 'pred'], 
                              add_hline='default',
                              x_label=x_label,
                              y_label="amplitude",
                              axs=ax2,
                              title=set_title,
                              xkcd=xkcd,
                              font_size=font_size,
                              line_width=[0.5, 3],
                              markers=['.', None],
                              **kwargs)

            if freq_spectrum:
                ax3 = fig.add_subplot(gs00[2])
                self.freq = preproc.get_freq(tc, TR=self.TR, spectrum_type=freq_type, clip_power=clip_power)

                plotting.LazyPlot(self.freq[1],
                                  xx=self.freq[0],
                                  color="#1B9E77", 
                                  x_label="frequency (Hz)",
                                  y_label="power (a.u.)",
                                  axs=ax3,
                                  title=freq_type,
                                  xkcd=xkcd,
                                  font_size=font_size,
                                  x_lim=[0,0.5],
                                  line_width=2,
                                  **kwargs)
            
            if save_as:
                print(f"Writing {save_as}")
                fig.savefig(save_as)

            return params, prf_array, tc
        else:
            return params, self.prediction

    def save_params(self, model="gauss", stage="grid", predictions=False):

        if hasattr(self, f"{model}_{stage}"):
            # write simple numpy files
            params = getattr(self, f"{model}_{stage}")
            output = opj(self.output_dir, f'{self.output_base}_model-{model}_stage-{stage}_desc-prf_params.npy')
            np.save(output, params)

            # write a pickle-file with relevant outputs
            pkl_file = opj(self.output_dir, f'{self.output_base}_model-{model}_stage-{stage}_desc-prf_params.pkl')
            out_dict = {}
            out_dict['pars'] = params
            out_dict['settings'] = self.settings
            out_dict['predictions'] = self.make_predictions(model=model, stage=stage)

            with open(pkl_file, 'wb') as f:
                f.dump(out_dict, f)

            if self.verbose:
                print(f"Save {stage}-fit parameters in {output}")

            if predictions:
                predictions = out_dict['predictions']
                output = opj(self.output_dir, f'{self.output_base}_model-{model}_stage-{stage}_desc-predictions.npy')
                np.save(output, predictions)
                
                if self.verbose:
                    print(f"Save {stage}-fit predictions in {output}")
        else:
            raise ValueError(f"class does not have attribute '{model}_{stage}'. Not saving parameters")

def find_most_similar_prf(reference_prf, look_in_params, verbose=False, return_nr='all', r2_thresh=0.5):

    """find_most_similar_prf

    find a pRF with similar characteristics in one array given the specifications of another pRF

    Parameters
    ----------
    reference_prf: numpy.ndarray
        pRF-parameters from the reference pRF, where `reference_prf[0]` = **x**, `reference_prf[1]` = **y**, and `reference_prf[2]` = **size**
    look_in_params: numpy.ndarray
        array of pRF-parameters in which we will be looking for `reference_prf`
    verbose: bool
        turn on/off verbose
    return_nr: str, int
        same parameter as in :func:`linescanning.utils.find_nearest`, where we can specify how many matches we want to have returned (default = "all") 
    r2_thresh: float
        after finding matching pRF, throw away fits below `r2_thresh`

    Returns
    ----------
    numpy.ndarray
        array containing the indices of `look_in_params` that conform the search
    """

    x_par,_ = utils.find_nearest(look_in_params[...,0], reference_prf[0], return_nr='all')

    if verbose:
        print(f"{x_par.shape} survived x-parameter matching")

    y_par,_ = utils.find_nearest(look_in_params[x_par,1], reference_prf[1], return_nr='all')
    xy_par = x_par[y_par]

    if verbose:
        print(f"{xy_par.shape} survived y-parameter matching")

    size_par,_ = utils.find_nearest(look_in_params[xy_par,2], reference_prf[2], return_nr='all')
    xysize_par = xy_par[size_par]

    if verbose:
        print(f"{xysize_par.shape} survived size-parameter matching")

    # filter indices with r2
    if r2_thresh != None:
        filt = look_in_params[xysize_par][...,-1] > r2_thresh
        true_idc = np.where(filt == True)
        xysize_par = xysize_par[true_idc]

        if verbose:
            print(f"{xysize_par.shape} survived r2>{r2_thresh}")

    if len(xysize_par) == 0:
        raise ValueError(f"Could not find similar pRFs. Maybe lower r2-threshold?")
        
    if return_nr == "all":
        return xysize_par
    else:
        return xysize_par[:return_nr]


class SizeResponse():
    """SizeResponse

    Perform size-response related operations given a pRF-stimulus/parameters. Simulate the pRF-response using a set of growing stimuli using :func:`linescanning.prf.make_stims`, create size response functions, and find stimulus sizes that best reflect the difference between two given SR-curves. Only needs a *prfpy.stimulus.PRFStimulus2D*-object and a set of parameters derived from a Divisive Normalization model.

    Parameters
    ----------
    prf_stim: prfpy.stimulus.PRFStimulus2D
        Object describing the nature of the stimulus
    params: numpy.ndarray
        array with shape (10,) as per the output of a Divisive Normalization fit operation.
    subject_info: `linescanning.utils.VertexInfo`-object
        Subject information collected in `linescanning.utils.VertexInfo` that can be used for :func:`linescanning.prf.SizeResponse.save_target_params`

    Example
    ----------
    >>> from linescanning import utils
    >>> # Collect subject-relevant information in class
    >>> subject = "sub-001"
    >>> hemi = "lh"
    >>> #
    >>> if hemi == "lh":
    >>>     hemi_tag = "hemi-L"
    >>> elif hemi == "rh":
    >>>     hemi_tag = "hemi-R"
    >>> #
    >>> subject_info = utils.CollectSubject(subject, derivatives=opj('<path_to_project>', 'derivatives'), settings='recent', hemi="lh")
    >>> #
    >>> # Get and plot fMRI signal
    >>> data_fn = utils.get_file_from_substring(f"avg_bold_{hemi_tag}.npy", subject_info.prfdir)
    >>> data = np.load(data_fn)[...,subject_info.return_target_vertex(hemi=hemi)]
    >>> #
    >>> # insert old parameters
    >>> insert_params =(np.array(subject_info.target_params)[np.newaxis,...],"gauss+iter")
    >>> #
    >>> # initiate class
    >>> fitting = prf.pRFmodelFitting(data[...,np.newaxis].T, 
    >>>                               design_matrix=subject_info.design_matrix, 
    >>>                               TR=subject_info.settings['TR'], 
    >>>                               model="norm", 
    >>>                               stage="grid", 
    >>>                               old_params=insert_params, 
    >>>                               verbose=False, 
    >>>                               output_dir=subject_info.prfdir, 
    >>>                               nr_jobs=1)
    >>> #
    >>> # fit
    >>> fitting.fit()
    >>> #
    >>> new_params = fitting.norm_grid[0]
    >>> #
    >>> # size response functions
    >>> SR = prf.SizeResponse(fitting.prf_stim, new_params)

    >>> # add subject info object
    >>> SR = prf.SizeResponse(fitting.prf_stim, new_params, subject_info=subject_info)        
    """
    
    def __init__(self, prf_stim, params, subject_info=None):

        self.prf_stim = prf_stim
        self.n_pix = self.prf_stim.design_matrix.shape[0]
        self.params = params

        if isinstance(self.params, np.ndarray):
            self.params_df = self.parse_normalization_parameters(self.params)
        elif isinstance(self.params, pd.DataFrame):
            self.params_df = self.params.copy()
            
        self.subject_info = subject_info

        # define visual field in degree of visual angle
        self.ss_deg = 3.0 * np.degrees(np.arctan(self.prf_stim.screen_size_cm /(2.0*self.prf_stim.screen_distance_cm)))
        self.x = np.linspace(-self.ss_deg/2, self.ss_deg/2, self.n_pix)


    def make_stimuli(self, factor=4):
        """create stimuli for Size-Response curve simulation. See :func:`linescanning.prf.make_stims`"""
        # create stimuli
        self.stims_fill, self.stims_fill_sizes = make_stims(self.n_pix, self.prf_stim, factr=factor)
    
    @staticmethod
    def parse_normalization_parameters(params, save_as=None):
        """store the Divisive Normalization model parameters in a DataFrame"""

        # see: https://github.com/VU-Cog-Sci/prfpy_tools/blob/master/utils/postproc_utils.py#L377
        if params.ndim == 1:
            params_dict = {"x": [params[0]], 
                           "y": [params[1]], 
                           "prf_size": [params[2]],
                           "prf_ampl": [params[3]],
                           "bold_bsl": [params[4]],
                           "neur_bsl": [params[7]],
                           "surr_ampl": [params[5]],
                           "surr_size": [params[6]], 
                           "surr_bsl": [params[4]],
                           "A": [params[3]], 
                           "B": [params[7]], #/params[:,3], 
                           "C": [params[5]], 
                           "D": [params[8]],
                           "ratio (B/D)": [params[7]/params[8]],
                           "r2": [params[-1]],
                           "size ratio": [params[6]/params[2]],
                           "suppression index": [(params[5]*params[6]**2)/(params[3]*params[2]**2)]}        

        elif params.ndim == 2:
            params_dict = {"x": params[:,0], 
                           "y": params[:,1], 
                           "prf_size": params[:,2],
                           "prf_ampl": params[:,3],
                           "bold_bsl": params[:,4],
                           "neur_bsl": params[:,7],
                           "surr_ampl": params[:,5],
                           "surr_size": params[:,6], 
                           "surr_bsl": params[:,4],
                           "A": params[:,3], 
                           "B": params[:,7], #/params[:,3], 
                           "C": params[:,5], 
                           "D": params[:,8],
                           "ratio (B/D)": params[:,7]/params[:,8],
                           "r2": params[:,-1],
                           "size ratio": params[:,6]/params[:,2],
                           "suppression index": (params[:,5]*params[:,6]**2)/(params[:,3]*params[:,2]**2)}
        else:
            raise TypeError(f"Parameters must have 1 or 2 dimensions, not {params.ndim}")

        df = pd.DataFrame(params_dict)
        if save_as:
            df.to_csv(save_as)
            return df
        else:
            return df

    
    def make_sr_function(self, center_prf=True, scale_factor=None, normalize=True):
        """create Size-Response function. If you want to ignore the actual location of the pRF, set `center_prf=True`. You can also scale the pRF-size with a factor `scale_factor`, for instance if you want to simulate pRF-sizes across depth."""
        if center_prf:
            mu_x, mu_y = 0,0
        else:
            mu_x = self.params_df['x'][0]
            mu_y = self.params_df['y'][0]

        if scale_factor != None:
            prf_size = self.params_df['prf_size'][0]*scale_factor
        else:
            prf_size = self.params_df['prf_size'][0]

        func = norm_2d_sr_function(self.params_df['A'][0], self.params_df['B'][0], self.params_df['C'][0], self.params_df['D'][0], prf_size, self.params_df['surr_size'][0], self.x, self.x, self.stims_fill, mu_x=mu_x, mu_y=mu_y)

        if normalize:
            return func / func.max()
        else:
            return func

    def find_stim_sizes(self, curve1, curve2):
        """find_stim_sizes

        Function to fetch the stimulus sizes that optimally disentangle the responses of two pRFs given the Size-Response curves. Starts by finding the maximum response for each curve, then finds the intersect, then finds the stimulus sizes before and after the intersect where the response difference is largest.

        Parameters
        ----------
        curve1: numpy.ndarray
            Array representing the first SR-curve (as per output for :func:`linescanning.prf.SizeResponse.make_sr_function`)
        curve2: numpy.ndarray
            Array representing the second SR-curve (as per output for :func:`linescanning.prf.SizeResponse.make_sr_function`)

        Returns
        ----------
        numpy.ndarray
            array containing the optimal stimulus sizes as per :func:`linescanning.prf.SizeResponse.make_stims`

        Example
        ----------
        >>> # follows up on example in *linescanning.prf.SizeResponse*
        >>> SR.make_stimuli()
        >>> sr_curve1 = SR.make_sr_function(center_prf=True)
        >>> sr_curve2 = SR.make_sr_function(center_prf=True, scale_factor=0.8) # decrease pRF size of second pRF by 20%
        >>> use_stim_sizes = SR.find_stim_sizes(sr_curve1, sr_curve2)
        """

        # find the peaks of individual curves
        sr_diff = curve1-curve2
        size_indices = signal.find_peaks(abs(sr_diff))[0]
        size_indices = np.append(size_indices, 
                                 np.array((utils.find_max_val(curve1)[0],
                                           utils.find_max_val(curve2)[0])))

        # append sizes of max of curves
        use_stim_sizes = []
        for size_index in size_indices:
            use_stim_sizes.append(self.stims_fill_sizes[size_index])

        # find intersection of curves
        x_size, y_size = utils.find_intersection(self.stims_fill_sizes, curve1, curve2)
        use_stim_sizes.append(y_size)
        use_stim_sizes.sort()

        return use_stim_sizes


    def plot_stim_size(self, stim_size, vf_extent=[-5,-5], ax=None):
        """plot output of :func:`linescanning.prf.SizeResponse.find_stim_sizes`"""

        if ax == None:
            fig,ax = plt.subplots(figsize=(6,6))

        stim_ix = utils.find_nearest(self.stims_fill_sizes, stim_size)[0]
        cmap_blue = utils.make_binary_cm((8,178,240))

        im = ax.imshow(self.stims_fill[stim_ix], extent=vf_extent+vf_extent, cmap=cmap_blue)
        ax.axvline(0, color='k', linestyle='dashed', lw=0.5)
        ax.axhline(0, color='k', linestyle='dashed', lw=0.5)
        ax.axis('off')
        patch = patches.Circle((0,0), radius=vf_extent[-1], transform=ax.transData)
        im.set_clip_path(patch)


    def save_target_params(self, fname=None, hemi="L"):
        """Write best_vertices-type file for normalization parameters + full normalization parameters in numpy-file"""
            
        ecc = np.sqrt(self.params_df['x'][0]**2+self.params_df['y'][0]**2)
        polar = np.angle(self.params_df['x'][0]+self.params_df['y'][0]*1j)

        if hemi == "lh":
            hemi = "L"
        elif hemi == "rh":
            hemi = "R"

        if self.subject_info != None:
            if fname == None:
                prf_bestvertex = opj(self.subject_info.cx_dir, f'{self.subject_info.subject}_model-norm_desc-best_vertices.csv')
            else:
                prf_bestvertex = fname

            vert_info = self.subject_info.vert_info.data.copy()
            best_vertex = pd.DataFrame({"hemi":     [hemi],
                                        "x":        [self.params_df['x'][0]],
                                        "y":        [self.params_df['y'][0]],
                                        "size":     [self.params_df['prf_size'][0]],
                                        "beta":     [self.params_df['prf_ampl'][0]],
                                        "baseline": [self.params_df['bold_bsl'][0]],
                                        "r2":       [self.params_df['r2'][0]],
                                        "ecc":      [ecc],
                                        "polar":    [polar],
                                        "index":    [vert_info['index'][hemi]],
                                        "position": [vert_info['position'][hemi]],
                                        "normal":   [vert_info['normal'][hemi]]})
            
            best_vertex = best_vertex.set_index(['hemi'])
            best_vertex.to_csv(prf_bestvertex)
            
        self.params_df.to_csv(opj(self.subject_info.cx_dir, f'{self.subject_info.subject}_hemi-{hemi}_desc-normalization_parameters.csv'))
        np.save(opj(self.subject_info.cx_dir, f'{self.subject_info.subject}_hemi-{hemi}_desc-normalization_parameters.npy'), self.params)

class CollectSubject:
    """CollectSubject

    Simple class to fetch pRF-related settings given a subject. Collects the design matrix, settings, and target vertex information. The `ses`-flag decides from which session the pRF-parameters to be used. You can either specify an *analysis_yaml* file containing information about a pRF-analysis, or specify *settings='recent'* to fetch the most recent analysis file in the pRF-directory of the subject. The latter is generally fine if you want information about the stimulus.

    Parameters
    ----------
    subject: str
        subject ID as used throughout the pipeline
    derivatives: str, optional
        Derivatives directory, by default None. 
    cx_dir: str, optional
        path to subject-specific pycortex directory
    prf_dir: str, optional
        subject-specific pRF directory, by default None. `derivatives` will be ignore if this flag is used
    ses: int, optional
        Source session of pRF-parameters to use, by default 1
    analysis_yaml: str, optional
        String pointing to an existing file, by default None. 
    hemi: str, optional
        Hemisphere to extract target vertex from, by default "lh"
    settings: str, optional
        Fetch most recent settings file rather than `analysis_yaml`, by default None. 
    model: str, optional
        This flag can be set to read in a specific 'best_vertex' file as the location parameters sometimes differ between a Gaussian and DN-fit.

    Example
    ----------
    >>> from linescanning import utils
    >>> subject_info = utils.CollectSubject(subject, derivatives=<path_to_derivatives>, settings='recent', hemi="lh")
    """

    def __init__(self, subject, derivatives=None, cx_dir=None, prf_dir=None, ses=1, analysis_yaml=None, hemi="lh", settings=None, model="gauss", correct_screen=True, verbose=True):

        self.subject        = subject
        self.derivatives    = derivatives
        self.cx_dir         = cx_dir
        self.prf_dir        = prf_dir
        self.prf_ses        = ses
        self.hemi           = hemi
        self.model          = model
        self.analysis_yaml  = analysis_yaml
        self.correct_screen = correct_screen
        self.verbose        = verbose

        if self.hemi == "lh" or self.hemi.lower() == "l" or self.hemi.lower() == "left":
            self.hemi_tag = "L"
        elif self.hemi == "rh" or self.hemi.lower() == "r" or self.hemi.lower() == "right":
            self.hemi_tag = "R"
        else:
            self.hemi_tag = "both"        

        # set pRF directory
        if self.prf_dir == None:
            if derivatives != None:
                self.prf_dir = opj(self.derivatives, 'prf', self.subject, f'ses-{self.prf_ses}')

        # get design matrix, vertex info, and analysis file
        if self.prf_dir != None:
            self.design_fn      = utils.get_file_from_substring("vis_design.mat", self.prf_dir)
            self.design_matrix  = io.loadmat(self.design_fn)['stim']
            self.func_data_lr   = np.load(utils.get_file_from_substring("avg_bold_hemi-LR.npy", self.prf_dir))
            self.func_data_l    = np.load(utils.get_file_from_substring("avg_bold_hemi-L.npy", self.prf_dir))
            self.func_data_r    = np.load(utils.get_file_from_substring("avg_bold_hemi-R.npy", self.prf_dir))
        # load specific analysis file
        if self.analysis_yaml != None:
            self.settings = yaml.safe_load(self.analysis_yaml)

            with open(self.analysis_yaml) as file:
                self.settings = yaml.safe_load(file)      

        try:
            self.gauss_iter_pars = np.load(utils.get_file_from_substring(["model-gauss", "stage-iter", "params"], self.prf_dir))
        except:
            pass
        
        # load the most recent analysis file. This is fine for screens/stimulus information
        if settings == "recent":
            self.analysis_yaml = opj(self.prf_dir, sorted([ii for ii in os.listdir(self.prf_dir) if "desc-settings" in ii])[-1])
        
            with open(self.analysis_yaml) as file:
                self.settings = yaml.safe_load(file)

        if self.cx_dir != None:
            self.vert_fn        = utils.get_file_from_substring([self.model, "best_vertices.csv"], self.cx_dir)
            self.vert_info      = utils.VertexInfo(self.vert_fn, subject=self.subject, hemi=self.hemi)
        
        # fetch target vertex parameters
        if hasattr(self, "vert_info"):
            self.target_params = self.return_prf_params(hemi=self.hemi)
            self.target_vertex = self.return_target_vertex(hemi=self.hemi)

        # create pRF if settings were specified
        if hasattr(self, "settings"):
            self.prf_stim = stimulus.PRFStimulus2D(screen_size_cm=self.settings['screen_size_cm'], screen_distance_cm=self.settings['screen_distance_cm'], design_matrix=self.design_matrix,TR=self.settings['TR'])
            self.prf_array = make_prf(self.prf_stim, size=self.target_params[2], mu_x=self.target_params[0], mu_y=self.target_params[1])

        try:
            self.normalization_params_df    = pd.read_csv(utils.get_file_from_substring([f"hemi-{self.hemi_tag}", "normalization", "csv"], self.cx_dir), index_col=0)
            self.normalization_params       = np.load(utils.get_file_from_substring([f"hemi-{self.hemi_tag}", "normalization", "npy"], self.cx_dir))

            if self.correct_screen:
                self.normalization_params = self.normalization_params*1.08
                
        except:
            self.normalization_params_df    = None
            self.normalization_params       = None

        if self.prf_dir != None:
            self.modelling      = pRFmodelFitting(self.func_data_lr,
                                                  design_matrix=self.design_matrix,
                                                  settings=self.analysis_yaml,
                                                  verbose=self.verbose)

            if self.model == "gauss":
                if hasattr(self, "gauss_iter_pars"):
                    self.pars = self.gauss_iter_pars.copy()
                else:
                    raise AttributeError("Could not find 'gauss_iter_pars' attribute")
            else:
                self.pars = self.normalization_params.copy()

            self.modelling.load_params(self.pars, model=self.model, stage="iter")

    def return_prf_params(self, hemi="lh"):
        """return pRF parameters from :class:`linescanning.utils.VertexInfo`"""
        return self.vert_info.get('prf', hemi=hemi)

    def return_target_vertex(self, hemi="lh"):
        """return the vertex ID of target vertex"""
        return self.vert_info.get('index', hemi=hemi)

    def target_prediction_prf(self, xkcd=False, line_width=1, freq_spectrum=None, save_as=None, **kwargs):
        _, self.prediction, _ = self.modelling.plot_vox(vox_nr=self.target_vertex, 
                                                        model=self.model, 
                                                        stage='iter', 
                                                        make_figure=True, 
                                                        xkcd=xkcd,
                                                        title='pars',
                                                        transpose=True, 
                                                        line_width=line_width,
                                                        freq_spectrum=freq_spectrum, 
                                                        freq_type="fft",
                                                        save_as=save_as,
                                                        **kwargs)        