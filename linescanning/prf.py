import ast
from datetime import datetime, timedelta
from linescanning import (
    utils, 
    plotting, 
    dataset, 
    preproc)
import math
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from past.utils import old_div
from prfpy import rf,stimulus
from prfpy.fit import *
from prfpy.model import *
import random
from scipy.ndimage import rotate
from scipy import signal, io
import subprocess
import time
import yaml
import pickle

opj = os.path.join

def normalize_prf(src,targ):

    norm = targ.copy()
    norm[0] -= src[0]
    norm[1] -= src[1]
    norm[2] /= src[2]

    return norm
    
def read_par_file(prf_file):

    """read_par_file

    Function to read in files containing pRF-estimates from various sources. Currently, supports the following inputs: `np.ndarray`, `npy`-file, `mat`-file (will read in the data from the last item in `list(prf_file.keys())`), `pkl`-file (assumes has key 'pars' in it). Because we generally save the design matrix as a *.mat*-file, we can use the same function to read in that file and obtain the design matrix in `np.ndarray`-format.

    Returns
    ----------
    np.ndarray
        array containing the pRF-estimates or other parameters (e.g., design matrix embedded in `mat`-file)

    Raises
    ----------
    TypeError
        If `pkl`-file was given, but attribute `pars` does not exist inside the file
    TypeError
        If the input is not a `np.ndarray`, and not a string ending on `npy`, `mat`, or `pkl`

    Example
    ----------
    >>> # read in pRF-estimates file
    >>> from linescanning import prf
    >>> prf_file = "sub-01_ses-1_task-2R_model-norm_stage-iter_desc-prf_params.pkl"
    >>> pars = prf.read_par_file(prf_file)

    >>> # read design matrix
    >>> from linescanning import prf
    >>> dm_file = "design_task-2R.mat"
    >>> dm = prf.read_par_file(dm_file)
    """

    if isinstance(prf_file, str):
        if prf_file.endswith("npy"):
            pars = np.load(prf_file)
        elif prf_file.endswith("mat"):
            tmp = io.loadmat(prf_file)
            pars = tmp[list(tmp.keys())[-1]] # find last key in list
        elif prf_file.endswith("pkl"):
            with open(prf_file, 'rb') as input:
                data = pickle.load(input)

                try:
                    pars = data['pars']
                except:
                    raise TypeError(f"Pickle-file did not arise from 'pRFmodelFitting'. I'm looking for 'pars', but got '{data.keys()}'")
    elif isinstance(prf_file, np.ndarray):
        pars = prf_file.copy()
    else:
        raise TypeError(f"Unknown input file '{prf_file}'. Must be one of ['str', 'npy', 'pkl'] or a numpy.ndarray")

    return pars
     
def get_prfdesign(screenshot_path, n_pix=40, dm_edges_clipping=[0,0,0,0]):
    """
    get_prfdesign

    Basically Marco's gist, but then incorporated in the repo. It takes the directory of screenshots and creates a vis_design.mat file, telling pRFpy at what point are certain stimulus was presented.

    Parameters
    ----------
    screenshot_path: str
        string describing the path to the directory with png-files
    n_pix: int
        size of the design matrix (basically resolution). The larger the number, the more demanding for the CPU. It's best to have some value which can be divided with 1080, as this is easier to downsample. Default is 40, but 270 seems to be a good trade-off between resolution and CPU-demands
    dm_edges_clipping: list, dict, optional
        people don't always see the entirety of the screen so it's important to check what the subject can actually see by showing them the cross of for instance the BOLD-screen (the matlab one, not the linux one) and clip the image accordingly. This is a list of 4 values, which are the number of pixels to clip from the left, right, top and bottom of the image. Default is [0,0,0,0], which means no clipping. Negative values will be set to 0.

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

    #clipping edges; top, bottom, left, right
    if isinstance(dm_edges_clipping, dict):
        dm_edges_clipping = [
            dm_edges_clipping['top'],
            dm_edges_clipping['bottom'],
            dm_edges_clipping['left'],
            dm_edges_clipping['right']]

    # ensure absolute values; should be a list by now anyway
    dm_edges_clipping = [abs(ele) for ele in dm_edges_clipping]

    design_matrix[:dm_edges_clipping[0], :, :] = 0
    design_matrix[(design_matrix.shape[0]-dm_edges_clipping[1]):, :, :] = 0
    design_matrix[:, :dm_edges_clipping[2], :] = 0
    design_matrix[:, (design_matrix.shape[0]-dm_edges_clipping[3]):, :] = 0

    # downsample (resample2d can also deal with 3D input)
    if n_pix != design_matrix.shape[0]:
        dm_resampled = utils.resample2d(design_matrix, n_pix)
        dm_resampled[dm_resampled<0.9] = 0
        return dm_resampled
    else:
        return design_matrix


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


def make_stims(x, dim_stim=2, factor=4, concentric=False, concentric_size=0.65):

    """make_stims

    Creates a list of stimuli to create size/response curves.

    Parameters
    ----------
    x: array
        visual field delineation
    prf_object: prfpy.stimulus.PRFStimulus2D
        representation the pRF in visual space
    dim_stim: int
        number of dimensions to use: 2 for circle, 1 for bar
    factor: int
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

    if dim_stim == 1:
        stims = [np.zeros_like(x) for n in range(int(x.shape[-1]/factor))]
    else:
        stims = [np.zeros((x.shape[0],x.shape[0])) for n in range(int(x.shape[-1]/(2*factor)))]
        stim_sizes=[]

    for pp, stim in enumerate(stims):

        if dim_stim == 1:
            #2d rectangle or 1d
            stim[int(stim.shape[0]/2)-pp:int(stim.shape[0]/2)+pp] = 1
        else:
            #2d circle
            xx,yy = np.meshgrid(x,x)
            stim[((xx**2+yy**2)**0.5)<(x.max()*pp/(len(stims)*factor))] = 1
            stim_sizes.append(2*(x.max()*pp/(len(stims)*factor)))

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


def plot_stims(
    stims, 
    n_rows=5, 
    n_cols=10, 
    figsize=None, 
    extent=[-5,5],
    save_as=None,
    axis_on=False,
    interval=1):

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

    if figsize == None:
        figsize = (n_cols*3,n_rows*3)

    fig,_ = plt.subplots(n_rows, n_cols, figsize=figsize)
    start = 0
    for i, ax in enumerate(fig.axes):
        ax.imshow(stims[start], extent=extent+extent)
        
        if not axis_on:
            ax.axis('off')

        start += interval

    plt.tight_layout()
    if isinstance(save_as, str):
        fig.savefig(save_as, facecolor='white', dpi=300, bbox_inches='tight')
    # return fig

def make_prf(prf_object, mu_x=0, mu_y=0, size=None, resize_pix=None, **kwargs):
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

    prf = np.rot90(rf.gauss2D_iso_cart(
        x=prf_object.x_coordinates[..., np.newaxis],
        y=prf_object.y_coordinates[..., np.newaxis],
        mu=(mu_x, mu_y),
        sigma=size,
        normalize_RFs=False).T, axes=(1, 2))

    # spatially smooth for visualization
    if isinstance(resize_pix, int):
        prf_squeezed = np.squeeze(prf, axis=0)
        prf = utils.resample2d(prf_squeezed, resize_pix)
    
    # save a bunch of problems if returned array is 2D
    if prf.ndim > 2:
        prf = np.squeeze(prf,axis=0)
        
    return prf


# From Marco's https://github.com/VU-Cog-Sci/prfpytools/blob/master/prfpytools/postproc_utils.py
def norm_2d_sr_function(a,b,c,d,s_1,s_2,x,y,stims,mu_x=0,mu_y=0):
    """create size/response function given set of parameters and stimuli"""

    xx,yy = np.meshgrid(x,y)

    sr_function = (a*np.sum(np.exp(-((xx-mu_x)**2+(yy-mu_y)**2)/(2*s_1**2))*stims, axis=(-1,-2)) + b) / (c*np.sum(np.exp(-((xx-mu_x)**2+(yy-mu_y)**2)/(2*s_2**2))*stims, axis=(-1,-2)) + d) - b/d
    return sr_function

# from https://github.com/mdaghlian/MD_toy_dn_scotoma/blob/2d0c208307536fd1853a6befdc38be7fe0d969fb/dn_scripts/RF.py
def gauss_1d_function(x, mu, sigma):
    """Create 1d gaussian from parameters"""
    gauss1d = np.exp(-((x-mu)**2) /(2*(sigma**2)))
    
    return gauss1d

# From Marco's https://github.com/VU-Cog-Sci/prfpytools/blob/master/prfpytools/postproc_utils.py
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
    verbose: bool, optional
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

def prf_neighbouring_vertices(subject, hemi='lh', vertex=None, prf_params=None, compare=False, vertices_only=False):

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
    vertices_only: bool
        only return the vertices, not their pRF-parameters (which might depend on a certain project structure)

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

    if not isinstance(vertex, int):
        raise ValueError("Must specify vertex from which to extract neighbours")

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

    if not vertices_only:
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
    else:
        return verts

def create_line_prf_matrix(
    log_dir, 
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
    dm_edges_clipping: list, dict, optional
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

    #clipping edges; top, bottom, left, right
    if isinstance(dm_edges_clipping, dict):
        dm_edges_clipping = [
            dm_edges_clipping['top'],
            dm_edges_clipping['bottom'],
            dm_edges_clipping['left'],
            dm_edges_clipping['right']]    

    # whether the stimulus duration is modulo the repetition time or not differs. In case it is, 
    # we can use the loop below. If not, we need to loop through the TRs, find the onset time 
    # closest to it, and select that particular image

    # get first image to get screen size
    img = (255*mpimg.imread(opj(screenshot_path, image_list[0]))).astype('int')

    if not round(stim_duration % TR, 5) > 0:

        # define native resolution design matrix
        design_matrix = np.zeros((img.shape[0],img.shape[0],len(image_list*tr_in_duration)))

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

        #top, bottom, left, right
        design_matrix[:dm_edges_clipping[0], :, :] = 0
        design_matrix[(design_matrix.shape[0]-dm_edges_clipping[1]):, :, :] = 0
        design_matrix[:, :dm_edges_clipping[2], :] = 0
        design_matrix[:, (design_matrix.shape[0]-dm_edges_clipping[3]):, :] = 0

        # downsample (resample2d can also deal with 3D input)
        dm_resampled = utils.resample2d(design_matrix, n_pix)

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
        
        # define empty design matrix
        design_matrix = np.zeros((img.shape[0], img.shape[0], nr_trs))

        if verbose:
            print("Reading onset times from log-file")

        # get the onsets
        onsets = dataset.ParseExpToolsFile(
            utils.get_file_from_substring(".tsv", log_dir), 
            TR=TR, 
            deleted_first_timepoints=deleted_first_timepoints, 
            use_bids=False,
            verbose=verbose,
            phase_onset=0)
        trial_df = onsets.get_onset_df()

        settings_fn = utils.get_file_from_substring(['yml'], log_dir)
        with open(settings_fn) as f:
            settings = yaml.safe_load(f)
            
        # check baseline first
        baseline_start = settings['design'].get('start_duration')

        if verbose:
            print("Creating design matrix (can take a few minutes with thousands of TRs)")

        # compatible with lineprf2
        if "baseline" in np.unique(trial_df['event_type'].values):

            # find blank onsets; parse ranges into list of tuples and check if tr_in_sec falls in that range or not
            # https://stackoverflow.com/a/6054040
            blank_periods = trial_df.loc[(trial_df['event_type'] == 'blank')]['onset'].values
            blank_duration = settings['design'].get('inter_sweep_blank')
            blank_ranges = [(i,i+blank_duration) for i in blank_periods]

            for tr in range(nr_trs):
                
                # find time at the middle of TR
                if stim_at_half_TR:
                    tr_in_sec = (tr * onsets.TR)+0.5*onsets.TR
                else:
                    tr_in_sec = (tr * onsets.TR)

                # start doing stuff if tr_in_sec is greater than baseline
                if tr_in_sec > baseline_start:
                    
                    # check blank area
                    if not any(lower <= tr_in_sec <= upper for (lower, upper) in blank_ranges):

                        # ix now represents the trial ID in the onset dataframe, which starts at the first 't'
                        ix,_ = utils.find_nearest(trial_df['onset'].values, tr_in_sec)
                        image_file = utils.get_file_from_substring(f"Screenshots{ix}.png", screenshot_path, return_msg=None)
                        
                        if image_file != None:
                            
                            img = (255*mpimg.imread(image_file)).astype('int')

                            if img.shape[0] != img.shape[1]:
                                offset = int((img.shape[1]-img.shape[0])/2)
                                img = img[:, offset:(offset+img.shape[0])]

                            # binarize image into dm matrix; i use ranges because I have another cue in the screenshots
                            design_matrix[..., tr][np.where(((img[..., 0] < 40) & (img[..., 1] < 40)) |
                                                            ((img[..., 0] > 200) & (img[..., 1] > 200)))] = 1
        else:

            # backwards compatibility for older versions of lineprf
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
                    design_matrix[..., tr][np.where(((img[..., 0] < 40) & (img[..., 1] < 40)) |
                                                    ((img[..., 0] > 200) & (img[..., 1] > 200)))] = 1
        
        #top, bottom, left, right
        design_matrix[:dm_edges_clipping[0], :, :] = 0
        design_matrix[(design_matrix.shape[0]-dm_edges_clipping[1]):, :, :] = 0
        design_matrix[:, :dm_edges_clipping[2], :] = 0
        design_matrix[:, (design_matrix.shape[0]-dm_edges_clipping[3]):, :] = 0

        # downsample (resample2d can also deal with 3D input)
        if n_pix != design_matrix.shape[0]:
            dm_resampled = utils.resample2d(design_matrix, n_pix)
            dm_resampled[dm_resampled<0.9] = 0
            return dm_resampled
        else:
            return design_matrix

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
 

def generate_model_params(model='gauss', dm=None, TR=1.5, fit_hrf=False, verbose=False):

    """generate_model_params

    Function to generate a yaml-file with parameters to use. Utilizes the [prf_analysis.yml](https://github.com/gjheij/linescanning/blob/main/misc/prf_analysis.yml) as basis, but it's advised to copy the file to the `DIR_DATA_HOME/code`-folder so you have a project-specific template. Depending on the model, we'll add grids and bounds and return a dictionary of settings. This dictionary is saved in the `pkl`-file as per the output of :class:`linescanning.prf.pRFmodelFitting`. 

    Parameters
    ----------
    model: str
        model type to utilize.
    dm: numpy.ndarray
        design matrix, n_pix X n_pix X volumes. Needed to create the pRF stimulus
    TR: float
        repetition time; can be fetched from gifti-file; default = 1.5
    fit_hrf: bool
        Whether or not to fit two extra parameters for hrf derivative and dispersion. The default is False.

    Returns
    ----------
    dict
        Dictionary containing the settings
    :class:`prfpy.stimlus.PRFStimulus2D`-object
        pRF stimulus object
    """
    
    # check if we have project-specific template; otherwise take linescanning-repo template
    # it's in a try-except loop because sometimes os.environ.get fails, causing a premature error..
    try:
        yml_file = utils.get_file_from_substring("prf_analysis.yml", opj(os.environ.get("DIR_DATA_HOME"), 'code'), return_msg=None)
    except:
        yml_file = None

    if yml_file == None:
        yml_file = utils.get_file_from_substring("prf_analysis.yml", opj(os.path.dirname(os.path.dirname(utils.__file__)), 'misc'))

    if verbose:
        print(f"Reading settings from '{yml_file}'", flush=True)

    with open(yml_file) as file:
        
        settings = yaml.safe_load(file)

        prf_stim = stimulus.PRFStimulus2D(
            screen_size_cm=settings['screen_size_cm'],
            screen_distance_cm=settings['screen_distance_cm'],
            design_matrix=dm,
            TR=TR)

        ss = prf_stim.screen_size_degrees
        max_ecc_size = ss/2.0

        # define grids
        sizes = max_ecc_size * np.linspace(0.25, 1, settings['grid_nr'])**2
        eccs = max_ecc_size * np.linspace(0.1, 1, settings['grid_nr'])**2
        polars = np.linspace(0, 2*np.pi, settings['grid_nr'])

        coord_bounds = (-1.5*max_ecc_size, 1.5*max_ecc_size)
        prf_size = (0.2, 1.5*ss)

        grids = {'screensize_degrees': float(ss), 
                 'grids': {'sizes': [float(item) for item in sizes], 
                           'eccs': [float(item) for item in eccs], 
                           'polars': [float(item) for item in polars]}}
        
        allowed_models = ['gauss', 'css', 'dog', 'norm']
        if model not in allowed_models:
            raise ValueError(f"Model must be one of {allowed_models}. Not '{model}'")

        # define bounds
        if model == "gauss":
            gauss_bounds = [coord_bounds,                                           # x
                            coord_bounds,                                           # y
                            prf_size,                                               # prf size
                            settings['prf_ampl'],                                   # prf amplitude
                            settings['bold_bsl']]                                   # bold baseline

            bounds = {'bounds': {'x': list(gauss_bounds[0]), 
                                 'y': list(gauss_bounds[1]), 
                                 'size': [float(item) for item in gauss_bounds[2]], 
                                 'prf_ampl': gauss_bounds[3], 
                                 'bold_bsl': gauss_bounds[4]}}

        elif model == "css":

            css_bounds = [coord_bounds,                                             # x
                          coord_bounds,                                             # y
                          prf_size,                                                 # prf size
                          settings['prf_ampl'],                                     # prf amplitude
                          settings['bold_bsl'],                                     # bold baseline
                          settings['css_exponent']]                                 # CSS exponent

            bounds = {'bounds': {'x': list(css_bounds[0]),
                                 'y': list(css_bounds[1]),
                                 'size': [float(item) for item in css_bounds[2]],
                                 'prf_ampl': css_bounds[3],
                                 'bold_bsl': css_bounds[4],
                                 'css_exponent': css_bounds[5]}}

        elif model == "dog":

            dog_bounds = [coord_bounds,                                             # x
                          coord_bounds,                                             # y
                          prf_size,                                                 # prf size
                          settings['prf_ampl'],                                     # prf amplitude
                          settings['bold_bsl'],                                     # bold baseline
                          settings[model]['surround_amplitude_bound'],              # surround amplitude
                          (settings['eps'], 3*ss)]                                  # surround size

            bounds = {'bounds': {'x': list(dog_bounds[0]),
                                 'y': list(dog_bounds[1]),
                                 'size': [float(item) for item in dog_bounds[2]],
                                 'prf_ampl': dog_bounds[3],
                                 'bold_bsl': dog_bounds[4],
                                 'surr_ampl': dog_bounds[5],
                                 'surr_size': [float(item) for item in dog_bounds[6]]}}

        elif model == "norm":
            norm_bounds = [coord_bounds,                                            # x
                           coord_bounds,                                            # x
                           prf_size,                                                # prf size
                           settings['prf_ampl'],                                    # prf amplitude
                           settings['bold_bsl'],                                    # bold baseline
                           settings[model]['surround_amplitude_bound'],             # surround amplitude
                           (settings['eps'], 3*ss),                                 # surround size
                           settings[model]['neural_baseline_bound'],                # neural baseline
                           settings[model]['surround_baseline_bound']]              # surround baseline

            bounds = {'bounds': {'x': list(norm_bounds[0]), 
                                 'y': list(norm_bounds[1]), 
                                 'size': [float(item) for item in norm_bounds[2]], 
                                 'prf_ampl': norm_bounds[3], 
                                 'bold_bsl': norm_bounds[4],
                                 'surr_ampl': norm_bounds[5],
                                 'surr_size': [float(item) for item in norm_bounds[6]],
                                 'neur_bsl': list(norm_bounds[7]),
                                 'surr_bsl': [float(item) for item in norm_bounds[8]]}}

        if fit_hrf:
            bounds['bounds']["hrf_deriv"] = settings["hrf"]["deriv_bound"]
            bounds['bounds']["hrf_disp"] = settings["hrf"]["disp_bound"]

        # update settings file if we've generated a new one
        settings.update(bounds)
        settings.update(grids)
        settings.update({'model': model})
        settings.update({'TR': TR})

    return settings, prf_stim

class GaussianModel():

    def __init__(self):     

        self.gaussian_fitter = Iso2DGaussianFitter(
            data=self.data,
            model=self.gauss_model,
            fit_css=False,
            fit_hrf=self.fit_hrf)

        if isinstance(self.old_params, np.ndarray):
            
            # check if dimensions make sense
            if self.old_params.shape[0] != self.data.shape[0]:
                if self.verbose:
                    print(f"Matching old parameters of shape {self.old_params.shape} to data of shape {self.data.shape[0],self.old_params.shape[-1]}", flush=True)
                self.old_params = np.tile(self.old_params, (self.data.shape[0],1))

            # set inserted params as gridsearch_params and iterative_search_params
            # needed for the rsq-mask
            self.gaussian_fitter.gridsearch_params = self.old_params.copy()
            self.gaussian_fitter.iterative_search_params = self.old_params.copy()   # actual parameters

            # set gaussian_fitter as previous_gaussian_fitter
            self.previous_gaussian_fitter = self.gaussian_fitter

            if self.verbose:
                print(f"Inserting parameters from {type(self.old_params)} as 'iterative_search_params' in {self}", flush=True)            

    def gridfit(self):
        if self.verbose:
            print("Starting gauss grid fit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'), flush=True)

        ## start grid fit
        start = time.time()
        self.gaussian_fitter.grid_fit(
            ecc_grid=self.settings['grids']['eccs'],
            polar_grid=self.settings['grids']['polars'],
            size_grid=self.settings['grids']['sizes'],
            fixed_grid_baseline=self.fix_grid_baseline,
            grid_bounds=[tuple(self.settings['bounds']['prf_ampl'])])
        
        elapsed = (time.time() - start)

        self.gauss_grid = utils.filter_for_nans(self.gaussian_fitter.gridsearch_params)
        mean_rsq = np.mean(self.gauss_grid[self.gauss_grid[:, -1]>self.settings['rsq_threshold'], -1])
        if self.verbose:
            print("Completed Gaussian gridfit at {start_time}. Voxels/vertices above {rsq}: {nr}/{total}".format(
                start_time=datetime.now().strftime('%Y/%m/%d %H:%M:%S'),
                rsq=str(self.settings['rsq_threshold']),
                nr=str(np.sum(self.gauss_grid[:, -1]>self.settings['rsq_threshold'])),
                total=str(self.gaussian_fitter.data.shape[0])), flush=True)
                
            print(f"Gridfit took {str(timedelta(seconds=elapsed))}")
            print("Mean rsq>{rsq}: {m_rsq}".format(
                rsq=self.settings['rsq_threshold'],
                m_rsq=str(round(mean_rsq,2))), flush=True)
        
        if self.write_files:
            if self.save_grid:
                self.save_params(model="gauss", stage="grid") 

    def iterfit(self):

        start = time.time()
        if self.verbose:
            print("Starting gauss iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'), flush=True)

        # fetch bounds
        self.gauss_bounds = self.fetch_bounds(model='gauss')

        if isinstance(self.fix_parameters, list):
            if self.verbose:
                print(f"Fixing parameters (idx): {self.fix_parameters}", flush=True)

            for el in self.fix_parameters:
                self.gauss_bounds[el] = tuple(
                    [self.gaussian_fitter.gridsearch_params[0,el],
                    self.gaussian_fitter.gridsearch_params[0,el]])

        # fit
        if self.constraints[0] == "tc":
            constr = []
        elif self.constraints[0] == "bgfs":
            constr = None
        else:
            raise ValueError(f"Unknown optimizer '{self.constraints[0]}'. Must be one of ['tc', 'bgfs']")

        self.gaussian_fitter.iterative_fit(
            rsq_threshold=self.settings['rsq_threshold'], 
            bounds=self.gauss_bounds,
            constraints=constr,
            xtol=self.settings['xtol'],
            ftol=self.settings['ftol'])

        # print summary
        elapsed = (time.time() - start)              
        self.gauss_iter = utils.filter_for_nans(self.gaussian_fitter.iterative_search_params)
        if self.verbose:
            mean_rsq = np.nanmean(self.gauss_iter[self.gaussian_fitter.rsq_mask, -1])
            print("Completed gauss iterfit at {start_time}. Mean rsq>{rsq}: {m_rsq}".format(
                start_time=datetime.now().strftime('%Y/%m/%d %H:%M:%S'),
                rsq=self.settings['rsq_threshold'],
                m_rsq=str(round(mean_rsq,2))), flush=True)

            print(f"Iterfit took {str(timedelta(seconds=elapsed))}", flush=True)

        # save intermediate files
        if self.write_files:
            self.save_params(model="gauss", stage="iter")  
            
class ExtendedModel():

    def __init__(self):

        if self.model == "dog":
            self.active_fitter = DoG_Iso2DGaussianFitter
        elif self.model == "css":
            self.active_fitter = CSS_Iso2DGaussianFitter
        elif self.model == "norm":
            self.active_fitter = Norm_Iso2DGaussianFitter

        self.active_model = getattr(self, f"{self.model}_model")

        # define fitter; previous_gaussian_fitter can be specified in kwargs
        if not hasattr(self, "previous_gaussian_fitter"):
            
            # check if we have gaussian fit
            if hasattr(self, "gaussian_fitter"):
                self.tmp_fitter = self.active_fitter(
                    self.active_model,
                    self.data, 
                    fit_hrf=self.fit_hrf, 
                    previous_gaussian_fitter=self.gaussian_fitter)
            else:
                # we might have manually injected parameters
                self.tmp_fitter = self.active_fitter(self.active_model, self.data, fit_hrf=self.fit_hrf)
        else:
            # inject existing-model object > advised when fitting the HRF
            self.tmp_fitter = self.active_fitter(
                self.active_model, 
                self.data, 
                fit_hrf=self.fit_hrf, 
                previous_gaussian_fitter=self.previous_gaussian_fitter)

    def gridfit(self):
        
        #----------------------------------------------------------------------------------------------------------------------------------------------------------

        # define grids into lists so we can call gridfit ones
        if self.model == "css":
            self.grid_list      = [np.array(self.settings['css']['css_exponent_grid'], dtype='float32')]
            self.grid_bounds    = [tuple(self.settings['bounds']['prf_ampl'])]
        elif self.model == "dog":
            self.grid_list      = [np.array(self.settings['dog']['dog_surround_amplitude_grid'], dtype='float32'),
                                   np.array(self.settings['dog']['dog_surround_size_grid'], dtype='float32')]
            self.grid_bounds    = [tuple(self.settings['prf_ampl']),
                                   tuple(self.settings['bounds']['surr_ampl'])]
        elif self.model == "norm":
            self.grid_list      = [np.array(self.settings['norm']['surround_amplitude_grid'], dtype='float32'),
                                   np.array(self.settings['norm']['surround_size_grid'], dtype='float32'),
                                   np.array(self.settings['norm']['neural_baseline_grid'], dtype='float32'),
                                   np.array(self.settings['norm']['surround_baseline_grid'], dtype='float32')]
            self.grid_bounds    = [tuple(self.settings['bounds']['prf_ampl']),
                                   tuple(self.settings['bounds']['neur_bsl'])]

        # grid fit
        if not self.skip_grid:

            ## Start grid fit
            start = time.time()
            if self.verbose:
                print(f"Starting {self.model} gridfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'), flush=True)

            self.tmp_fitter.grid_fit(
                *self.grid_list,
                n_batches=self.nr_jobs,
                rsq_threshold=self.settings['rsq_threshold'],
                fixed_grid_baseline=self.fix_grid_baseline,
                grid_bounds=self.grid_bounds)

            elapsed = (time.time() - start)

            ### save grid parameters
            filtered_ = utils.filter_for_nans(self.tmp_fitter.gridsearch_params)
            setattr(self, f"{self.model}_grid", filtered_)
            if self.verbose:
                mean_rsq = np.mean(filtered_[self.tmp_fitter.gridsearch_rsq_mask, -1])
                print("Completed {model} gridfit at {start_time}. Mean rsq>{rsq}: {m_rsq}".format(
                    model=self.model,
                    start_time=datetime.now().strftime('%Y/%m/%d %H:%M:%S'),
                    rsq=str(self.settings['rsq_threshold']),
                    m_rsq=str(round(mean_rsq,2))), flush=True)
                    
                print(f"Gridfit took {str(timedelta(seconds=elapsed))}", flush=True)

            if self.write_files:
                if self.save_grid:
                    self.save_params(model=self.model, stage="grid")
        else:
            if self.verbose:
                print(f"Setting {(type(self.gaussian_fitter.iterative_search_params))} as 'gridsearch_params' in {self.tmp_fitter}", flush=True)
                
            self.tmp_fitter.gridsearch_params = self.gaussian_fitter.iterative_search_params

        setattr(self, f"{self.model}_fitter", self.tmp_fitter)

    def iterfit(self):

        # fetch bounds from settings > HRF bounds are automatically appended if fit_hrf=True
        self.tmp_bounds = self.fetch_bounds(model=self.model)
        if isinstance(self.fix_parameters, list):
            if self.verbose:
                print(f"Fixing parameters (idx): {self.fix_parameters}", flush=True)

            for el in self.fix_parameters:
                self.tmp_bounds[el] = tuple(
                    [self.gaussian_fitter.iterative_search_params[0,el],
                    self.gaussian_fitter.iterative_search_params[0,el]])

        setattr(self, f"{self.model}_bounds", self.tmp_bounds)        
        
        start = time.time()
        if self.verbose:
            print(f"Starting {self.model} iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'), flush=True)

        # fit
        if self.constraints[1] == "tc":
            constr = []
        elif self.constraints[1] == "bgfs":
            constr = None
        else:
            raise ValueError(f"Unknown optimizer '{self.constraints[0]}'. Must be one of {self.allowed_optimizers}")        

        self.tmp_fitter.iterative_fit(
            rsq_threshold=self.settings['rsq_threshold'], 
            bounds=self.tmp_bounds,
            constraints=constr,
            xtol=self.settings['xtol'],
            ftol=self.settings['ftol'])

        elapsed = (time.time() - start)  

        ### save iterative parameters
        filtered_ = utils.filter_for_nans(self.tmp_fitter.iterative_search_params)
        setattr(self, f"{self.model}_iter", filtered_)

        if self.verbose:
            mean_rsq = np.mean(filtered_[self.tmp_fitter.rsq_mask, -1])
            print("Completed {model} iterfit at {start_time}. Mean rsq>{rsq}: {m_rsq}".format(
                model=self.model,
                start_time=datetime.now().strftime('%Y/%m/%d %H:%M:%S'),
                rsq=self.settings['rsq_threshold'],
                m_rsq=str(round(mean_rsq,2))), flush=True)

            print(f"Iterfit took {str(timedelta(seconds=elapsed))}", flush=True)

        if self.write_files:
            self.save_params(model=self.model, stage="iter")   

        setattr(self, f"{self.model}_fitter", self.tmp_fitter)

class pRFmodelFitting(GaussianModel, ExtendedModel):
    """pRFmodelFitting

    Main class to perform all the pRF-fitting. By default, we'll first look for a `prf_analysis.yml`-file in `DIR_DATA_HOME/code`. If there's no file there, we'll take the file provided with the *linescanning*-repository (https://github.com/gjheij/linescanning/blob/main/misc/prf_analysis.yml). Generally, the input data is expected to be percent-signal changed where the timecourses are shifted such that the median of the timepoints *without* stimulus is set to 0 (see https://github.com/gjheij/linescanning/blob/main/bin/call_prf#L267 how this works). 

    Parameters
    ----------
    data: numpy.ndarray
        <voxels,time> numpy array | when reading in the data later again, the format must be <time,voxels>. This is highly annoying, but it seems to be required for predictions to work properly.
    design_matrix: numpy.ndarray
        <n_pix, n_pix, time> numpy array containing the paradigm
    TR: float
        repetition time of acquisition; required for the analysis file. If you're using gifti-input, you can fetch the TR from that file with `gifti = linescanning.utils.ParseGiftiFile(gii_file).TR_sec`. If you're file have been created with `fMRIprep` or `call_vol2fsaverage`, this should work.
    model: str
        as of now, one of ['gauss','dog','css','norm'] is accepted
    stage: str
        Can technically be anything. By default, grid-fits are executed, but if `stage` contains `iter` (e.g., `stage="iter"`), an iterative fit is executed as well.
    output_dir: str
        directory to store all files in; should be somewhere in <project>/derivatives/prf/<subject>
    output_base: str
        basename for output files; should be something like <subject>_<ses-?>_<task-?>
    write_files: bool
        save files (True) or not (False). Should be used in combination with <output_dir> and <output_base>
    old_params: np.ndarray, str, optional
        A string pointing to an existing file or a numpy array. Internally, the parameters will be assigned to `Iso2DGaussianFitter.gridsearch_params` and `Iso2DGaussianFitter.iterative_search_params`. This fitter object is then assigned to `Iso2DGaussianFitter.previous_gaussian_fitter`, which is then directly inserted in one of the extended models (e.g., `DN`, `DoG`, or `CSS`).
    hrf: np.ndarray, list, optional
        <1,time_points> describing the HRF. Can be created with :func:`linescanning.glm.double_gamma`, then add an axis before the timepoints:

        >>> dt = 1
        >>> time_points = np.linspace(0,36,np.rint(float(36)/dt).astype(int))
        >>> hrf_custom = linescanning.glm.double_gamma(time_points, lag=6)
        >>> hrf_custom = hrf_custom[np.newaxis,...]

        Can also be a list of three parameters for the SPM-functions. Defaults to [1,1,0], for the HRF, and the time derivative. 
    fit_hrf: bool
        fit the HRF with the pRF-parameters as implemented in `prfpy`
    nr_jobs: int, optional
        Set the number of jobs. By default 1.
    verbose: bool, optional
        Set to True if you want some messages along the way (default = False)
    constraints: list, str, optional
        Specify optimizers. Use `tc` for trust-constrained minimization (= slower, but less local. I.e., can move away from the grid), or `bfgs` for L-BFGS-B minimization (lot faster, but more local. I.e., stays closer to the grid). If `tc` or `bgfs`, this minimizer is used for both stages. You can also specify a list where the first element is for *Gaussian*-model and second element is for extended model, e.g., `['tc', 'bgfs']`. Generally, the extended model has more parameters so the fit is slower, while the Gaussian-model runs fine with slow optimization (`tc`). Default = `tc` for both stages as it leads to improved *r2*. 
    save_grid: bool, optional
        Save grid-parameters; can save clogging up of directories. Default = True
    any: optional
        You can also provide a value for any key/item present in the default settings-file, e.g., `screen_size_cm`, `rsq_threshold`, `TR`, you name it. This will overwrite the default setting, without the requirement of specifying a new file. Allows for quick tests/adaptations to the default settings. This is especially useful in cases where - pls no.. - you have different screen distances in one dataset.

    Returns
    ----------
    pkl-file
        For each model, a pickle file with the settings, parameters, and predictions all in one

    Example
    ----------
    >>> from linescanning.prf import pRFmodelFitting
    >>> fitting = pRFmodelFitting(func, design_matrix=dm, model='gauss')

    >>> # we can use this class to read in existing parameter files
    >>> prf_pfov = opj(prf_new, "sub-003_ses-3_task-pRF_acq-3DEPI_model-norm_stage-iter_desc-prf_params.npy")
    >>> modelling_pfov = prf.pRFmodelFitting(
    >>>     partial_nan.T,
    >>>     design_matrix=design_pfov,
    >>>     stage="grid+iter",
    >>>     model="norm",
    >>>     output_dir=prf_new,
    >>>     output_base="sub-003_ses-3_task-pRF_acq-3DEPI")
    >>> #
    >>> modelling_pfov.load_params(np.load(prf_pfov), model='norm', stage='iter')

    Notes
    ----------
    See https://linescanning.readthedocs.io/en/latest/examples/prfmodelfitter.html for more elaborate example of fitting, loading, visualization, and HRF-fitting
    """

    def __init__(
        self, 
        data, 
        design_matrix=None, 
        TR=1.5, 
        model="gauss", 
        stage="iter", 
        output_dir=None, 
        write_files=False, 
        output_base=None,
        old_params=None,
        verbose=True,
        hrf=None,
        fit_hrf=False,
        settings=None,
        nr_jobs=1000,
        prf_stim=None,
        model_obj=None,
        fix_bold_baseline=False,
        fix_parameters=None,
        constraints="tc",
        save_grid=True,
        skip_grid=False,
        **kwargs):

        self.data               = data
        self.design_matrix      = design_matrix
        self.TR                 = TR
        self.model              = model
        self.stage              = stage
        self.output_dir         = output_dir
        self.output_base        = output_base
        self.write_files        = write_files
        self.settings_fn        = settings
        self.verbose            = verbose
        self.hrf                = hrf
        self.old_params         = old_params
        self.nr_jobs            = nr_jobs
        self.prf_stim           = prf_stim
        self.model_obj          = model_obj
        self.fit_hrf            = fit_hrf
        self.fix_bold_baseline  = fix_bold_baseline
        self.fix_parameters     = fix_parameters
        self.constraints        = constraints
        self.save_grid          = save_grid
        self.skip_grid          = skip_grid
        self.__dict__.update(kwargs)

        # make data 2D
        if isinstance(self.data, np.ndarray):
            if self.data.ndim == 1:
                self.data = self.data[np.newaxis,...]
        
        # read design matrix if needed
        if isinstance(self.design_matrix, str):
            if self.verbose:
                print(f"Reading design matrix from '{self.design_matrix}'", flush=True)
            self.design_matrix = read_par_file(self.design_matrix)

        #----------------------------------------------------------------------------------------------------------------------------------------------------------
        # Fetch the settings
        self.define_settings()

        # overwrite bold baseline tuple if bold baseline should be fixed
        if self.fix_bold_baseline:
            self.settings['bounds']['bold_bsl'] = [0,0]
            self.fix_grid_baseline = 0
            if self.verbose:
                print(f"Fixing baseline at {self.settings['bounds']['bold_bsl']}", flush=True)
        else:
            self.fix_grid_baseline = None

        # check if we got a pRF-stim object
        if self.prf_stim != None:
            if self.verbose:
                print("Using user-defined pRF-stimulus object", flush=True)
            self.prf_stim = self.prf_stim
        else:
            self.prf_stim = self.prf_stim_
        
        self.allowed_models = ["gauss", "css", "dog", "norm"]
        if self.model.lower() not in self.allowed_models:
            raise ValueError(f"Model specification needs to be one of {self.allowed_models}; got {model}.")

        # sort out HRF
        self.define_hrf()

        # update settings
        self.update_settings()

        # check validity of constraints
        self.allowed_optimizers = ['tc', 'bgfs']
        if isinstance(self.constraints, str):
            self.constraints = [self.constraints, self.constraints]

        for optimizer in self.constraints:
            if optimizer not in self.allowed_optimizers:
                raise ValueError(f"Unknown optimizer '{optimizer}'. Must be one of {self.allowed_optimizers}")
        
        if self.verbose:
            print(f"Using constraint(s): {self.constraints}")
            
        #----------------------------------------------------------------------------------------------------------------------------------------------------------
        # whichever model you run, run the Gaussian first

        ## Define models
        self.gauss_model    = Iso2DGaussianModel(stimulus=self.prf_stim, hrf=self.hrf)
        self.css_model      = CSS_Iso2DGaussianModel(stimulus=self.prf_stim, hrf=self.hrf)
        self.dog_model      = DoG_Iso2DGaussianModel(stimulus=self.prf_stim, hrf=self.hrf)
        self.norm_model     = Norm_Iso2DGaussianModel(stimulus=self.prf_stim, hrf=self.hrf)

        if self.model_obj != None:
            if self.verbose:
                print(f"Setting {self.model_obj} as '{model}_model'-attribute", flush=True)
            setattr(self, f'{model}_model', self.model_obj)

    def define_settings(self):

        self.settings, self.prf_stim_ = generate_model_params(
            model=self.model, 
            dm=self.design_matrix, 
            TR=self.TR,
            fit_hrf=self.fit_hrf,
            verbose=self.verbose)

    def define_hrf(self):

        # make compatible with prfpy's HRF-class?
        if isinstance(self.hrf, np.ndarray):
            if self.hrf.ndim < 2:
                self.hrf = self.hrf[np.newaxis,...]
            
            if self.verbose:
                print(f"Instantiate HRF with: '{type(self.hrf)}'", flush=True)

            try:
                self.hrf = HRF(self.hrf)
            except:
                raise ValueError("Cannot use HRF-class. Please specify list of three parameters")

        elif isinstance(self.hrf, list):
            if self.verbose:
                print(f"Instantiate HRF with: {self.hrf}", flush=True)

            try:
                self.hrf = HRF()
                self.hrf.create_spm_hrf(hrf_params=self.hrf, TR=self.TR, force=True)
            except:
                pass
        
        elif isinstance(self.hrf, str):
            if self.verbose:
                print(f"Instantiate 'direct' HRF", flush=True)
            
            self.hrf = "direct"
            
        else:
            # try to read HRF parameters from settings
            try:
                hrf_pars = self.settings['hrf']['pars']
            except:
                hrf_pars = [1,1,0]

            if self.verbose:
                print(f"Instantiate HRF with: {hrf_pars}", flush=True)
            
            try:
                self.hrf = HRF()
                self.hrf.create_spm_hrf(hrf_params=hrf_pars, TR=self.TR, force=True)
            except:
                self.hrf = hrf_pars.copy()

    def update_settings(self):

        # overwrite settings with custom settings
        for setting in list(self.settings.keys()):
            if hasattr(self, setting):

                # hrf separate because it can be a list or object depending on prfpy-version
                if setting != "hrf":

                    # only update if different
                    if getattr(self, setting) != self.settings[setting]:

                        if self.verbose:
                            print(f"Setting '{setting}' to user-defined value: {getattr(self, setting)} (was: {self.settings[setting]})", flush=True)

                        self.settings[setting] = getattr(self, setting)

    def fit(self):

        # check whether we got old parameters so we can skip Gaussian fit:
        if isinstance(self.old_params, np.ndarray) or isinstance(self.old_params, str):
            
            # agnostic read-in of existing parameter file
            self.old_params = read_par_file(self.old_params)
            if self.old_params.ndim == 1:
                self.old_params = self.old_params[np.newaxis,...]

            # initiate Gaussian model
            GaussianModel.__init__(self)

        if not hasattr(self, "previous_gaussian_fitter"):

            GaussianModel.__init__(self)
            GaussianModel.gridfit(self)

            #----------------------------------------------------------------------------------------------------------------------------------------------------------
            # Check if we should do Gaussian iterfit        
            if 'iter' in self.stage:
                GaussianModel.iterfit(self)

        else:
            if self.verbose:
                print(f"Gaussian fitter: {self.previous_gaussian_fitter}", flush=True)

                # assume old parameters are grid parameters
                if self.model == "gauss" and "iter" in self.stage:
                    GaussianModel.iterfit(self)

        #----------------------------------------------------------------------------------------------------------------------------------------------------------
        # Check if we should do DN-model
        if self.model.lower() != "gauss":            
            
            ## Define settings/grids/fitter/bounds etcs
            self.define_settings()

            if self.fix_bold_baseline:
                self.settings['bounds']['bold_bsl'] = [0,0]

            # overwrite settings with custom settings
            self.update_settings()

            # initiate and do grid fit
            ExtendedModel.__init__(self)
            ExtendedModel.gridfit(self)

            if "iter" in self.stage:
                ExtendedModel.iterfit(self)

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

        elif model == "css":

            bounds = [tuple(self.settings['bounds']['x']),                  # x
                      tuple(self.settings['bounds']['y']),                  # y
                      tuple(self.settings['bounds']['size']),               # prf size
                      tuple(self.settings['bounds']['prf_ampl']),           # prf amplitude
                      tuple(self.settings['bounds']['bold_bsl']),           # bold baseline
                      tuple(self.settings['bounds']['css_exponent'])]       # CSS exponent

        elif model == "dog":

            bounds = [tuple(self.settings['bounds']['x']),                  # x
                      tuple(self.settings['bounds']['y']),                  # y
                      tuple(self.settings['bounds']['size']),               # prf size
                      tuple(self.settings['bounds']['prf_ampl']),           # prf amplitude
                      tuple(self.settings['bounds']['bold_bsl']),           # bold baseline
                      tuple(self.settings['bounds']['surr_ampl']),          # surround amplitude
                      tuple(self.settings['bounds']['surr_size'])]          # surround size

        else:
            raise ValueError(f"Unrecognized model '{model}'")               

        if self.fit_hrf:
            bounds.append(tuple(self.settings["bounds"]['hrf_deriv']))      # HRF time derivative
            bounds.append(tuple(self.settings["bounds"]['hrf_disp']))       # HRF dispersion derivative

        return bounds
    
    def load_params(self, params_file, model='gauss', stage='iter', hemi=None):
        """load_params

        Attribute of `self`, with which you can load in an existing file with pRF-estimates. If the input file is a `pkl`-file, you'll get access to all the required information about your analysis: the settings, the input data, the predictions, and the design matrix. To do this, we need the `params_file` as well as information about the origin of the file; which model (e.g., `gauss`) and stage (e.g., 'iter') did the file arise from. We'll then internally set the parameters to the specified model and stage, making it compatible with :func:`linescanning.prf.pRFmodelFitting.plot_vox()`. It can also be a `numpy.ndarray` or `pandas.DataFrame`, but then less information about the analysis is known. 

        Parameters
        ----------
        params_file: str, pandas.DataFrame, numpy.ndarray
            Input to load in. You'll get the most out of it when you use a `pkl`-file created with :class:`linescanning.prf.pRFmodelFitting`, as you'll have access to the predictions, timecourses, parameters, and settings. If you do not have this, you can also enter a `numpy.ndarray` or `pandas.DataFrame`. The later is assumed to be a result from :class:`linescanning.prf.SizeResponse`, in that it's a *Divisive-Normalization* model result, with the following columns: `['x','y','prf_size','A','bold_bsl','B','C','surr_size','D','r2']` and indexed by `hemi` (`L`/`R`).
        model: str, optional
            Model from which the pRF-estimates came from, by default 'gauss'
        stage: str, optional
            Stage from which the pRF-estimates came from, by default 'iter'
        hemi: str, optional
            In case `params_file` is a `pandas.DataFrame` indexed by `hemi` and you want a particular subset of the dataframe

        Raises
        ----------
        ValueError
            If input is `pandas.DataFrame`, but input does not have index `hemi`
        ValueError
            If input is not one of `str`, `numpy.ndarray`, `pandas.DataFrame`

        Example
        ----------
        >>> # we initiate the model as per usual
        >>> gauss_load = prf.pRFmodelFitting(
        >>>     data.T,
        >>>     design_matrix=design,
        >>>     TR=1.5,
        >>>     verbose=True)
        >>> #
        >>> gauss_load.load_params("sub-01_ses-1_model-norm_stage-iter_desc-prf_params.pkl", model="gauss", stage="iter")

        Notes
        ----------
        Also see https://linescanning.readthedocs.io/en/latest/examples/prfmodelfitter.html
        """

        if isinstance(params_file, str):
            if params_file.endswith('npy'):
                params = np.load(params_file)
            elif params_file.endswith('pkl'):
                with open(params_file, 'rb') as input:
                    data = pickle.load(input)
                    
                params = data['pars']
                self.settings = data['settings']
                setattr(self, f'{model}_{stage}_predictions', data['predictions'])

                # try to set the HRFs if fit_hrf=True
                try:
                    setattr(self, f'{model}_{stage}_hrf', data['hrf'])
                except:
                    pass
                
                if self.verbose:
                    print("Reading settings from pickle-file (safest option; overwrites other settings)")

        elif isinstance(params_file, np.ndarray):
            params = params_file.copy()
        elif isinstance(params_file, list):
            params = np.array(params_file)
        elif isinstance(params_file, pd.DataFrame):
            if hemi:
                # got normalization parameter file
                params = np.array((
                    params_file['x'][hemi],
                    params_file['y'][hemi],
                    params_file['prf_size'][hemi],
                    params_file['A'][hemi],
                    params_file['bold_bsl'][hemi],
                    params_file['C'][hemi],
                    params_file['surr_size'][hemi],
                    params_file['B'][hemi],
                    params_file['D'][hemi],
                    params_file['r2'][hemi]))
            else:
                params = np.array((
                    params_file['x'],
                    params_file['y'],
                    params_file['prf_size'],
                    params_file['A'],
                    params_file['bold_bsl'],
                    params_file['C'],
                    params_file['surr_size'],
                    params_file['B'],
                    params_file['D'],
                    params_file['r2']))

        else:
            raise ValueError(f"Unrecognized input type for '{params_file}' ({type(params_file)})")

        if self.verbose:
            print(f"Inserting parameters from {type(params_file)} as '{model}_{stage}' in {self}")

        setattr(self, f'{model}_{stage}', params)

    def make_predictions(self, vox_nr=None, model='gauss', stage='iter'):
        
        try:
            use_model = getattr(self, f"{model}_model")
        except:
            raise ValueError(f"{self}-object does not have attribute '{self.model}_model'")

        if hasattr(self, f"{model}_{stage}"):
            params = getattr(self, f"{model}_{stage}")
            if params.ndim == 1:
                params = params[np.newaxis,...]

            if vox_nr != None:
                if vox_nr == "best":
                    vox,_ = utils.find_nearest(params[...,-1], np.amax(params[...,-1]))
                else:
                    vox = vox_nr
                

                params = params[vox,...]
                pred = use_model.return_prediction(*params[:-1]).T
                return pred, params, vox
            else:
                predictions = []
                for vox in range(params.shape[0]):
                    pars = params[vox,...]
                    predictions.append(use_model.return_prediction(*pars[:-1]).T)
                
                return np.squeeze(np.array(predictions), axis=-1)

        else:
            raise ValueError(f"Could not find {stage} parameters for {model}")

    def plot_vox(
        self, 
        vox_nr=0,
        model='gauss', 
        stage='iter', 
        make_figure=True, 
        xkcd=False, 
        title=None, 
        transpose=False,
        freq_spectrum=False,
        freq_type='fft',
        clip_power=None,
        save_as=None,
        axis_type="time",
        resize_pix=270,
        add_tc={"tc": None},
        axs=None,
        **kwargs):    
        
        """plot_vox

        Quick function to plot the pRF-location in visual space as well as the raw timecourse + prediction. This is done based on voxel-indexing, `vox_nr`. You'll need to specify the `model` and `stage` flags to select the correct pRF-estimates. If you do not want a figure, but just the outputs, you can set `make_figure=False`. Other flags are customizations, such as adding a power spectrum, font size, and xkcd-style plotting. `axis_type` refers to the nature of the x-axis. Can either be `volumes` or `time` (generally time is more informative). Finally, if you have a slightly lower resolution design matrix, you can upsample your pRF-location with a given pixel size (e.g., `270`). This is only for aesthetics in the figure.

        Parameters
        ----------
        vox_nr: int, np.ndarray, optional
            Voxel/vertex index to create the plot for, by default 0. Can also be a 1D array
        model: str, optional
            Which *model* to select the pRF-estimates from, by default 'gauss'
        stage: str, optional
            Which *stage* to select the pRF-estimates from, by default 'iter'
        make_figure: bool, optional
            Make the figure (`make_figure=True`) and output `params`, the `np.ndarray` representing the pRF in visual space, the BOLD timecourse, and the `prediction`; or `make_figure=False` and only return `params` and `prediction`, by default True
        xkcd: bool, optional
            Make the plot in xkcd-style, by default False
        title: str, optional
            Title of the timecourse plot. If `title='pars'`, we'll set the parameters as title. This can be useful to quickly check parameters, by default None
        font_size: int, optional
            Fontsize for x/y-labels and title, by default 18
        transpose: bool, optional
            Depending on how the predictions are loaded, you might need to transpose. Generally, if you've ran the fitting before plotting, this should be fine. Rule of thumb: if you get an indexing error, try `transpose=True`.
        freq_spectrum: bool, optional
            Add a frequency spectrum of the timecourse, by default False
        freq_type: str, optional
            Which type of frequency sprectrum, by default 'fft' (see also :func:`linescanning.preproc.get_freq` or https://linescanning.readthedocs.io/en/latest/classes/preproc.html#linescanning.preproc.get_freq)
        clip_power: int, optional
            Clip the power of the spectrum to enhance visualization, by default None
        save_as: str, optional
            Save the figure as `save_as`, by default None
        axis_type: str, optional
            Type of x-axis, by default "volumes". Can also be 'time' to use time-dimension, rather than volume-dimension
        resize_pix: int, optional
            Spatially smooth your pRF if you've used a low-resolution design matrix, by default None. Generally, `resize_pix=270` results in pleasing pRF-depictions
        add_tc: dict, np.ndarray, optional
            Allows an additional timecourse to be plotted with the data and the prediction. Can be either a numpy array or a dictionary with the following keys:

            - "tc": the timecourse to add. If `add_tc == np.ndarray`, then "tc" will be set internally
            - "color": any valid `matplotlib`-color (e.g., RGB/Hex). Default = "b"
            - "lw": line width of the timecourse. Default = 2
            - "marker": marker type of the timecourse. Default = None
            - "label": label to add to the legend. Default = "extra"

            Some examples:            
            >>> add_tc=np.array([]) # simplest
            >>> add_tc={"tc": np.array([]), "label": "grid", "marker": "."} # set some extra options

        Returns
        ----------
        np.ndarray
            1D-array representing the parameters of you selected voxel
        np.ndarray
            2D-array representing the pRF of your selected voxel in visual space
        np.ndarray
            1D-array representing the raw BOLD timecourse of your selected voxel
        np.ndarray
            1D-array representing the prediction of your selected voxel given `model`, `stage`, and `voxel`

        Example
        ----------
        >>> from linescanning.prf import pRFmodelFitting
        >>> #
        >>> # define the model
        >>> fitting = pRFmodelFitting(func, design_matrix=dm, model='gauss')
        >>> #
        >>> # fit
        >>> fitting.fit()
        >>> #
        >>> # plot the 1st voxel
        >>> fitting.plot_vox(
        >>>     vox_nr=0,
        >>>     title='pars',
        >>>     model='gauss',
        >>>     stage='iter')

        Notes
        ----------
        
        - To silence output, use `_,_,_,_= fitting.plot_vox()`
        - Also check https://linescanning.readthedocs.io/en/latest/examples/prfmodelfitter.html for more examples
        """

        if isinstance(vox_nr, np.ndarray):
            if len(vox_nr) == 1:
                vox_nr = vox_nr[0]
            else:
                raise ValueError(f"Array of length {len(vox_nr)} was given. Can only be 1")

        self.prediction, params, vox = self.make_predictions(
            vox_nr=vox_nr, 
            model=model, 
            stage=stage)

        if hasattr(self, f"{model}_{stage}_predictions"):
            self.prediction = getattr(self, f"{model}_{stage}_predictions")[vox_nr]

        prf_array = make_prf(
            self.prf_stim, 
            size=params[2], 
            mu_x=params[0], 
            mu_y=params[1], 
            resize_pix=resize_pix)
        
        if hasattr(self, "data"):
            if isinstance(self.data, np.ndarray):
                # annoying indexing issues.. lots of inconsistencies in array shapes.
                if transpose:
                    tc = self.data.T[vox,...]
                else:
                    tc = self.data[vox,...]

                # set default lists
                data_list = [tc, self.prediction]
                color_list = ['#cccccc', 'r']
                label_list = ['real', 'pred']
                lw_list = [0.5,3]
                marker_list = ['.', None]                
            else:
                tc = None
                data_list = [self.prediction]
                color_list = ['r']
                label_list = None
                lw_list = [3]
                marker_list = None

        if make_figure:
            
            if not isinstance(axs, list):
                fig = plt.figure(constrained_layout=True, figsize=(15,5))
                if freq_spectrum:
                    gs = fig.add_gridspec(1,3, width_ratios=[10,20,10])
                else:
                    gs = fig.add_gridspec(1,2, width_ratios=[10,20])

                ax1 = fig.add_subplot(gs[0])
                ax2 = fig.add_subplot(gs[1])
            else:
                ax1 = axs[0]
                ax2 = axs[1]

            # make plot 
            plotting.LazyPRF(
                prf_array, 
                vf_extent=self.settings['vf_extent'], 
                ax=ax1, 
                xkcd=xkcd,
                **kwargs)

            # make plot 
            if title != None:
                if title == "pars":
                    set_title = [round(ii,2) for ii in params]
                else:
                    set_title = title
            else:
                set_title = None

            if axis_type == "time":
                x_label = "time (s)"
                x_axis = np.array(list(np.arange(0,self.prediction.shape[0])*self.TR))
            else:
                x_axis = np.array(list(np.arange(0,self.prediction.shape[0])))
                x_label = "volumes"

            # add additional timecourse
            if isinstance(add_tc, np.ndarray):
                add_tc = {"tc": add_tc}
            
            if isinstance(add_tc, dict):
                if isinstance(add_tc["tc"], np.ndarray):
                    default_dict = {
                        "color": "b",
                        "lw": 2,
                        "marker": None,
                        "label": "extra"
                    }

                    for key in list(default_dict.keys()):
                        if key not in list(add_tc.keys()):
                            add_tc[key] = default_dict[key]

                    data_list += [add_tc["tc"]]
                    color_list += [add_tc["color"]]
                    label_list += [add_tc["label"]]
                    lw_list += [add_tc["lw"]]
                    marker_list += [add_tc["marker"]]

            plotting.LazyPlot(
                data_list,
                xx=x_axis,
                color=color_list, 
                labels=label_list, 
                add_hline=0,
                x_label=x_label,
                y_label="amplitude",
                axs=ax2,
                title=set_title,
                xkcd=xkcd,
                line_width=lw_list,
                markers=marker_list,
                x_lim=[0,x_axis[-1]],
                x_ticks=[0,x_axis[-1]],
                **kwargs)

            if freq_spectrum:
                if not isinstance(axs, list):
                    ax3 = fig.add_subplot(gs[2])
                else:
                    if not len(axs) == 3:
                        raise ValueError("Need an extra axis in the list for this function")
                    
                    ax3 = axs[2]

                self.freq = preproc.get_freq(tc, TR=self.TR, spectrum_type=freq_type, clip_power=clip_power)

                plotting.LazyPlot(
                    self.freq[1],
                    xx=self.freq[0],
                    color="#1B9E77", 
                    x_label="frequency (Hz)",
                    y_label="power (a.u.)",
                    axs=ax3,
                    title=freq_type,
                    xkcd=xkcd,
                    x_lim=[0,0.5],
                    line_width=2,
                    **kwargs)

            if save_as:
                print(f"Writing {save_as}")
                fig.savefig(save_as)

        return params, prf_array, tc, self.prediction

    def save_params(self, model="gauss", stage="grid"):
        
        if hasattr(self, f"{model}_{stage}"):

            # get parameters given model and stage
            params = getattr(self, f"{model}_{stage}")

            # write a pickle-file with relevant outputs
            pkl_file = opj(self.output_dir, f'{self.output_base}_model-{model}_stage-{stage}_desc-prf_params.pkl')
            out_dict = {}
            out_dict['pars'] = params
            out_dict['settings'] = self.settings
            out_dict['predictions'] = self.make_predictions(model=model, stage=stage)

            # # save HRFs if applicable
            # if self.fit_hrf:
                
            #     if stage != "grid":
            #         try:
            #             self.hrfs = [self.hrf.create_spm_hrf(hrf_params=[1,*params[ii,-3:-1]], TR=self.TR, force=True).values.T for ii in range(self.data.shape[0])]
            #         except:
            #             raise IOError(f"Could not create HRFs")

            #         out_dict['hrf'] = self.hrfs

            if self.verbose:
                print(f"Save {stage}-fit parameters in {pkl_file}")

            f = open(pkl_file, "wb")
            pickle.dump(out_dict, f)
            f.close()

        else:
            raise ValueError(f"{self} does not have attribute '{model}_{stage}'. Not saving parameters")

def find_most_similar_prf(reference_prf, look_in_params, verbose=False, return_nr='all', r2_thresh=0.5):

    """find_most_similar_prf

    find a pRF with similar characteristics in one array given the specifications of another pRF

    Parameters
    ----------
    reference_prf: numpy.ndarray
        pRF-parameters from the reference pRF, where `reference_prf[0]` = **x**, `reference_prf[1]` = **y**, and `reference_prf[2]` = **size**
    look_in_params: numpy.ndarray
        array of pRF-parameters in which we will be looking for `reference_prf`
    verbose: bool, optional
        Set to True if you want some messages along the way (default = False)
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
    prf_stim: :class:`prfpy.stimulus.PRFStimulus2D`
        Object describing the nature of the stimulus
    params: numpy.ndarray
        array with shape (10,) as per the output of a Divisive Normalization fit operation.
    subject_info: :class:`linescanning.utils.VertexInfo`-object
        Subject information collected in :class:`linescanning.utils.VertexInfo` that can be used for :func:`linescanning.prf.SizeResponse.save_target_params`

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
    >>> subject_info = prf.CollectSubject(
    >>>     subject, 
    >>>     prf_dir=prf_dir, 
    >>>     cx_dir=cx_dir, 
    >>>     hemi=hemi, 
    >>>     resize_pix=270,
    >>>     verbose=False)
    >>> #
    >>> # Get and plot fMRI signal
    >>> data_fn = utils.get_file_from_substring(f"avg_bold_{hemi_tag}.npy", subject_info.prfdir)
    >>> data = np.load(data_fn)[...,subject_info.return_target_vertex(hemi=hemi)]
    >>> #
    >>> # insert old parameters
    >>> insert_params = subject_info.target_params
    >>> #
    >>> # initiate class
    >>> fitting = prf.pRFmodelFitting(
    >>>     data[...,np.newaxis].T, 
    >>>     design_matrix=subject_info.design_matrix, 
    >>>     TR=subject_info.settings['TR'], 
    >>>     model="norm", 
    >>>     stage="grid", 
    >>>     old_params=insert_params, 
    >>>     verbose=False, 
    >>>     output_dir=subject_info.prfdir, 
    >>>     nr_jobs=1)
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
    
    def __init__(self, prf_stim, params, model="norm", subject_info=None):

        self.prf_stim = prf_stim
        self.n_pix = self.prf_stim.design_matrix.shape[0]
        self.params = params
        self.model = model

        if isinstance(self.params, np.ndarray):
            self.params_df = self.parameters_to_df(self.params, model=self.model)
        elif isinstance(self.params, pd.DataFrame):
            self.params_df = self.params.copy()
            
        self.subject_info = subject_info

        # define visual field in degree of visual angle
        self.ss_deg = 2*np.degrees(np.arctan(self.prf_stim.screen_size_cm /(2.0*self.prf_stim.screen_distance_cm)))
        self.x = np.linspace(-self.ss_deg/2, self.ss_deg/2, 1000)
        self.dx = self.n_pix/len(self.x)


    def make_stimuli(self, factor=4):
        """create stimuli for Size-Response curve simulation. See :func:`linescanning.prf.make_stims`"""
        # create stimuli
        self.stims_fill, self.stims_fill_sizes = make_stims(self.x, factor=factor)
    
    def find_pref_size(self, size=None):
        if not hasattr(self, "stims_fill"):
            self.make_stimuli(factor=2)

        idx,_ = utils.find_nearest(self.stims_fill_sizes, size)
        return idx,self.stims_fill[idx]

    @staticmethod
    def parameters_to_df(params, model="norm", save_as=None):
        """store the Divisive Normalization model parameters in a DataFrame"""

        if params.ndim == 1:
            params = params[np.newaxis,:]

        # see: https://github.com/VU-Cog-Sci/prfpy_tools/blob/master/utils/postproc_utils.py#L377
        if model == "gauss":
            params_dict = {
                "x": params[:,0], 
                "y": params[:,1], 
                "prf_size": params[:,2],
                "prf_ampl": params[:,3],
                "bold_bsl": params[:,4],
                "r2": params[:,-1],
                "ecc": np.sqrt(params[:,0]**2+params[:,1]**2),
                "polar": np.angle(params[:,0]+params[:,1]*1j)
            }
        elif model == "norm":
                
            params_dict = {
                "x": params[:,0], 
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
                "suppression index": (params[:,5]*params[:,6]**2)/(params[:,3]*params[:,2]**2),
                "ecc": np.sqrt(params[:,0]**2+params[:,1]**2),
                "polar": np.angle(params[:,0]+params[:,1]*1j)}
        else:
            raise NotImplementedError(f"Im lazy.. Still need to sort out models ['css' and 'dog']")

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

        func = norm_2d_sr_function(
            self.dx**2*self.params_df['A'][0], 
            self.params_df['B'][0], 
            self.dx**2*self.params_df['C'][0], 
            self.params_df['D'][0], 
            prf_size, 
            self.params_df['surr_size'][0], 
            self.x, 
            self.x, 
            self.stims_fill, 
            mu_x=mu_x, 
            mu_y=mu_y)

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
        _,y_size = utils.find_intersection(self.stims_fill_sizes, curve1, curve2)
        use_stim_sizes.append(y_size)
        use_stim_sizes.sort()

        # append a stimulus of size 3dva if len(use_stim_sizes) == 4
        if len(use_stim_sizes) == 4:
            use_stim_sizes.append(2.5)

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


    def save_target_params(self, fname=None, hemi="L", stim_sizes=None, factor=None):
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

            if factor != None:
                print(f"Correcting for closer BOLD-screen [factor={factor}]")
            else:
                factor = 1

            vert_info = self.subject_info.vert_info.data.copy()
            data_dict = {"hemi":     [hemi],
                         "x":        [self.params_df['x'][0]*factor],
                         "y":        [self.params_df['y'][0]*factor],
                         "size":     [self.params_df['prf_size'][0]*factor],
                         "beta":     [self.params_df['prf_ampl'][0]],
                         "baseline": [self.params_df['bold_bsl'][0]],
                         "r2":       [self.params_df['r2'][0]],
                         "ecc":      [ecc],
                         "polar":    [polar],
                         "index":    [vert_info['index'][hemi]],
                         "position": [vert_info['position'][hemi]],
                         "normal":   [vert_info['normal'][hemi]]}

            # add stim sizes
            if isinstance(stim_sizes, list):
                stim_sizes = np.array(stim_sizes)

            if isinstance(stim_sizes, np.ndarray):
                data_dict['stim_sizes'] = [stim_sizes*factor]
                
            best_vertex = pd.DataFrame(data_dict)
            
            # append to existing file
            if os.path.exists(prf_bestvertex):
                tmp = pd.read_csv(prf_bestvertex, index_col=0).reset_index()

                if hemi in tmp['hemi'].values:
                    tmp = tmp[tmp.hemi != hemi]

                best_vertex = pd.concat((best_vertex, tmp))

            best_vertex.set_index('hemi').to_csv(prf_bestvertex)
        
        # save full parameters as well
        pars_file = opj(self.subject_info.cx_dir, f'{self.subject_info.subject}_model-norm_desc-params.csv') 
        self.params_df['hemi'] = hemi
        if os.path.exists(pars_file):
            tmp = pd.read_csv(pars_file, index_col=0).reset_index()

            if hemi in tmp['hemi'].values:
                tmp = tmp[tmp.hemi != hemi]

            self.params_df = pd.concat((self.params_df, tmp))

        self.params_df.set_index('hemi').to_csv(pars_file)

class CollectSubject(pRFmodelFitting):
    """CollectSubject

    Simple class to fetch pRF-related settings given a subject. Collects the design matrix, settings, and target vertex information. The `ses`-flag decides from which session the pRF-parameters to be used. You can either specify an *analysis_yaml* file containing information about a pRF-analysis, or specify *settings='recent'* to fetch the most recent analysis file in the pRF-directory of the subject. The latter is generally fine if you want information about the stimulus.

    Parameters
    ----------
    subject: str
        subject ID as used throughout the pipeline
    ses: int, optional
        Source session of pRF-parameters to use, by default 1
    derivatives: str, optional
        Derivatives directory, by default None. 
    cx_dir: str, optional
        path to subject-specific pycortex directory
    prf_dir: str, optional
        subject-specific pRF directory, by default None. `derivatives` will be ignore if this flag is used
    hemi: str, optional
        Hemisphere to extract target vertex from, by default "lh"
    model: str, optional
        By default `gauss`, which reads in the gaussian iterative fit parameters in a :class:`linescanning.prf.pRFmodelFitting`-object. Can be any of the allowed models ('gauss', 'css', 'dog', norm).
    verbose: bool, optional
        Set to True if you want some messages along the way (default = True)
    resize_pix: int
        resolution of pRF to resample to. For instance, if you've used a low-resolution design matrix, but you'd like a prettier image, you can set `resize` to something higher than the original (54 >> 270, for example). By default not used.
    best_vertex: bool, optional
        Signifies that we should load in the parameters of the target vertex used in line-scanning experiments. If *True*, the 'best_vertex'-parameters given `model` will be read in. If *False*, the iterative fit parameters from `model` will be read in.
    filter_list: list, optional
        Extra filters to search for particular design matrix. By default, we'll look for a file with ["design", "mat"], but if you have multiple files that correspond to this list you can add extra elements to pick the file you need, e.g., `filter_list=["acq-3DEPI"]`

    Example
    ----------
    >>> from linescanning import utils
    >>> subject_info = utils.CollectSubject(subject, derivatives=<path_to_derivatives>, settings='recent', hemi="lh")
    """

    def __init__(
        self, 
        subject, 
        ses=1, 
        derivatives=None, 
        cx_dir=None, 
        prf_dir=None, 
        hemi="lh", 
        model="gauss", 
        verbose=True, 
        best_vertex=False,
        filter_list=[],
        fit_hrf=False,
        **kwargs):

        self.subject        = subject
        self.derivatives    = derivatives
        self.cx_dir         = cx_dir
        self.prf_dir        = prf_dir
        self.prf_ses        = ses
        self.hemi           = hemi
        self.model          = model
        self.verbose        = verbose
        self.best_vertex    = best_vertex
        self.filter_list    = filter_list
        self.fit_hrf        = fit_hrf
        self.__dict__.update(kwargs)

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
            self.design_fn = utils.get_file_from_substring(["design", ".mat"]+self.filter_list, self.prf_dir)

            # error message in case multiple files are found
            if isinstance(self.design_fn, list):
                raise ValueError(f"Found multiple files matching the criteria: ['design', 'mat']; {self.design_fn}")

            self.design_matrix = read_par_file(self.design_fn) # read_par_file doesn't care whether it's a npy-/mat-/pkl-file

            try:
                self.func_data_lr = np.load(utils.get_file_from_substring(["hemi-LR_", "avg_bold", ".npy"], self.prf_dir))
                self.func_data_l = np.load(utils.get_file_from_substring(["hemi-L_", "avg_bold", ".npy"], self.prf_dir))
                self.func_data_r = np.load(utils.get_file_from_substring(["hemi-R_", "avg_bold", ".npy"], self.prf_dir))
            except:
                try:
                    self.func_data_lr = np.load(utils.get_file_from_substring(["desc-data.npy"]+self.filter_list, self.prf_dir))
                except:
                    print(f"WARNING: could not load functional data from '{self.prf_dir}'")

        # try to read iterative fit parameters
        allowed_models = ['gauss', 'css', 'dog', 'norm']
        look_for = [f"model-{model}", "stage-iter", "params.pkl"]
        
        if self.fit_hrf:
            look_for += ["hrf-true_"]

        for model in allowed_models:
            par_file = utils.get_file_from_substring(look_for, self.prf_dir, return_msg=None)
            if isinstance(par_file, str):
                setattr(self, f'{model}_iter_pars', read_par_file(par_file))
        
        # set pycortex directory
        if self.cx_dir == None:
            if derivatives != None:
                self.cx_dir = opj(self.derivatives, 'pycortex', self.subject)

        if self.best_vertex:
            
            if self.cx_dir != None and os.path.exists(self.cx_dir):
                self.vert_fn = utils.get_file_from_substring([self.model, "best_vertices.csv"], self.cx_dir)

                if self.verbose:
                    print(f"Reading {self.vert_fn}")

                self.vert_info = utils.VertexInfo(
                    self.vert_fn, 
                    subject=self.subject, 
                    hemi=self.hemi)
            
            # fetch target vertex parameters
            if hasattr(self, "vert_info"):
                self.target_params = self.return_prf_params(hemi=self.hemi)
                self.target_vertex = self.return_target_vertex(hemi=self.hemi)

            # find the csv file with parameters (generally available with 'norm')
            try:
                self.params_fn = utils.get_file_from_substring([f"model-{self.model}", "desc-params.csv"], self.cx_dir)
                self.normalization_params_df = pd.read_csv(self.params_fn, index_col=0)            
                self.pars = self.normalization_params_df.copy()
            except:
                self.pars = np.array(self.target_params)[np.newaxis,...]
            
            if self.hemi_tag == "L":
                tmp_data = self.func_data_l.copy()
            else:
                tmp_data = self.func_data_r.copy()
            
            self.data = tmp_data[...,self.target_vertex][np.newaxis,...]

        else:
            if hasattr(self, f"{self.model}_iter_pars"):
                self.pars = getattr(self, f"{self.model}_iter_pars")
            else:
                print(f"WARNING: Could not find '{self.model}_iter_pars' attribute")

            if hasattr(self, "func_data_lr"):
                self.data = self.func_data_lr.T

        # initialize the model
        if hasattr(self, "data"):
            super().__init__(
                self.data,
                design_matrix=self.design_matrix,
                verbose=self.verbose,
                model=self.model)

            if hasattr(self, "pars"):
                self.load_params(
                    self.pars, 
                    model=self.model, 
                    stage="iter", 
                    hemi=self.hemi_tag)
        else:
            print("WARNING: could not initiate model due to missing data. Some functions may not work.")

    def return_prf_params(self, hemi="lh"):
        """return pRF parameters from :class:`linescanning.utils.VertexInfo`"""
        return self.vert_info.get('prf', hemi=hemi)

    def return_target_vertex(self, hemi="lh"):
        """return the vertex ID of target vertex"""
        return self.vert_info.get('index', hemi=hemi)

    def target_prediction_prf(self, xkcd=False, freq_spectrum=None, save_as=None, make_figure=True, **kwargs):
        
        self.targ_pars, self.targ_prf, self.targ_tc, self.targ_pred = self.plot_vox(
            vox_nr=0, 
            model=self.model, 
            stage='iter', 
            make_figure=make_figure, 
            xkcd=xkcd,
            title='pars',
            freq_spectrum=freq_spectrum, 
            freq_type="fft",
            save_as=save_as,
            **kwargs)

        return {
            "pars": self.targ_pars,
            "prf": self.targ_prf,
            "tc": self.targ_tc,
            "pred": self.targ_pred}

def baseline_from_dm(dm, n_trs=7):
    shifted_dm = np.zeros_like(dm)
    shifted_dm[..., n_trs:] = dm[..., :-n_trs]
    return np.where((np.sum(dm, axis=(0, 1)) == 0) & (np.sum(shifted_dm, axis=(0, 1)) == 0))[0]
