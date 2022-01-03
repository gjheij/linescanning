import ast
from datetime import datetime, timedelta
from joblib.externals.cloudpickle.cloudpickle import instance
from linescanning import utils
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
import seaborn as sns
import subprocess
import time
import yaml

opj = os.path.join

def get_prfdesign(screenshot_path,
                  n_pix=40,
                  dm_edges_clipping=[0, 0, 0, 0]):
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

    # there is one more MR image than screenshot
    design_matrix = np.zeros((n_pix, n_pix, 1+len(image_list)))

    for image_file in image_list:
        # assuming last three numbers before .png are the screenshot number
        img_number = int(image_file[-7:-4])-1
        # subtract one to start from zero
        img = (255*mpimg.imread(opj(screenshot_path, image_file))).astype('int')
        # make it square
        if img.shape[0] != img.shape[1]:
            offset = int((img.shape[1]-img.shape[0])/2)
            img = img[:, offset:(offset+img.shape[0])]

        # downsample
        downsampling_constant = int(img.shape[1]/n_pix)
        downsampled_img = img[::downsampling_constant, ::downsampling_constant]
#        import matplotlib.pyplot as pl
#        fig = pl.figure()
#        pl.imshow(downsampled_img)
#        fig.close()

        if downsampled_img[:, :, 0].shape != design_matrix[..., 0].shape:
            print("please choose a n_pix value that is a divisor of " +
                  str(img.shape[0]))

        # binarize image into dm matrix
        # assumes: standard RGB255 format; only colors present in image are black, white, grey, red, green.
        design_matrix[:, :, img_number][np.where(((downsampled_img[:, :, 0] == 0) & (
            downsampled_img[:, :, 1] == 0)) | ((downsampled_img[:, :, 0] == 255) & (downsampled_img[:, :, 1] == 255)))] = 1

        design_matrix[:, :, img_number][np.where(((downsampled_img[:, :, 0] == downsampled_img[:, :, 1]) & (
            downsampled_img[:, :, 1] == downsampled_img[:, :, 2]) & (downsampled_img[:, :, 0] != 127)))] = 1

    #clipping edges
    #top, bottom, left, right
    design_matrix[:dm_edges_clipping[0], :, :] = 0
    design_matrix[(design_matrix.shape[0]-dm_edges_clipping[1]):, :, :] = 0
    design_matrix[:, :dm_edges_clipping[2], :] = 0
    design_matrix[:, (design_matrix.shape[0]-dm_edges_clipping[3]):, :] = 0
    # print("  Design matrix completed")

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

def make_prf(prf_object, mu_x=0, mu_y=0, size=None):
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
        size of pRF

    Returns
    ----------
    numpy.ndarray
        meshgrid containing Gaussian characteristics of the pRF. Can be plotted with :func:`linescanning.prf.plot_prf`
    """

    prf = np.rot90(rf.gauss2D_iso_cart(x=prf_object.x_coordinates[..., np.newaxis],
                                       y=prf_object.y_coordinates[..., np.newaxis],
                                       mu=(mu_x, mu_y),
                                       sigma=size,
                                       normalize_RFs=False).T, axes=(1, 2))

    return prf


def plot_prf(prf,vf_extent, return_axis=False, save_as=None):

    """
    plot_prf

    Plot the geometric location of the Gaussian pRF.

    Parameters
    ----------
    prf: numpy.ndarray
        instantiation of `gauss2D_iso_cart`
    vf_extent: list
        the space the pRF lives in

    Returns
    ----------
    matplotlib.pyplot plot
    """

    fig, ax = plt.subplots()
    ax.axvline(0, color='white', linestyle='dashed', lw=0.5)
    ax.axhline(0, color='white', linestyle='dashed', lw=0.5)
    im = ax.imshow(np.squeeze(prf, axis=0), extent=vf_extent+vf_extent, cmap='magma')
    patch = patches.Circle((0, 0), radius=vf_extent[-1], transform=ax.transData)
    im.set_clip_path(patch)
    ax.axis('off')

    if save_as:
        plt.savefig(save_as, transparant=True)
    if return_axis:
        return ax


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

        use_settings = {'hori': [0, 18],
                        'vert': [4, 25],
                        'rot_45': [4, 18],
                        'rot_135': [0, 5]}

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
                           n_samples=None, 
                           TR=0.105, 
                           n_pix=275, 
                           deleted_first_timepoints=0, 
                           deleted_last_timepoints=0, 
                           stim_duration=1, 
                           baseline_before=24,
                           baseline_after=24,
                           skip_first_img=True):

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
    deleted_first: int 
        number of volumes to skip in the beginning; should be the same as used in ParseFunc-file and/or ParseFuncFile (if used in concert) (default = 0)
    deleted_last: int
        number of volumes to skip at the end; should be the same as used in ParseFuncFile and/or ParseFuncFile (if used in concert) (default = 0)
    n_pix: int 
        size determinant of the design matrix. Note, <n_pix> is hardcoded as well in the function itself because I ran stuff on my laptop and that screen has a different size than the stimulus computer @Spinoza (default = 270px)
    stim_duration: float
        length of stimulus presentation (in seconds; default = 1s)
    skip_first_img: boolean 
        depending on how you coded the screenshot capturing, the first empty screen is something also captured. With this flag we can skip that image (default = True).

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

    if not round(stim_duration % TR, 5) > 0:

        design_matrix = np.zeros((n_pix,n_pix,len(image_list*tr_in_duration)))

        point = 0
        for ii in image_list:
            img_fn = opj(screenshot_path, ii)
            img = (255*mpimg.imread(img_fn)).astype('int')

            if img.shape[0] != img.shape[1]:
                offset = int((img.shape[1]-img.shape[0])/2)
                img = img[:, offset:(offset+img.shape[0])]

            downsampling_constant = int(img.shape[1]/n_pix)
            downsampled_img = img[::downsampling_constant, ::downsampling_constant]

            design_matrix[:, :, point:point+tr_in_duration][np.where(((downsampled_img[:, :, 0] == 0) & (
            downsampled_img[:, :, 1] == 0)) | ((downsampled_img[:, :, 0] == 255) & (downsampled_img[:, :, 1] == 255)))] = 1

            point += tr_in_duration

        # deal with baseline and deleted volumes
        baseline_before = np.zeros((n_pix,n_pix,int(baseline_before/TR)))
        if deleted_first_timepoints != 0:
            baseline_before = baseline_before[...,deleted_first_timepoints:]

        print(f"Baseline before has shape: {baseline_before.shape}")

        baseline_after = np.zeros((n_pix,n_pix,int(baseline_after/TR)))    
        if deleted_last_timepoints != 0:
            baseline_after = baseline_after[...,:-deleted_last_timepoints]
        
        print(f"Design itself has shape: {design_matrix.shape}")
        print(f"Baseline after has shape: {baseline_after.shape}")
        dm = np.dstack((baseline_before, design_matrix, baseline_after))
        
        return dm
    else:

        design_matrix = np.zeros((n_pix, n_pix, nr_trs))

        # get the onsets
        onsets = utils.ParseExpToolsFile(utils.get_file_from_substring(".tsv", log_dir), TR=TR, delete_vols=deleted_first_timepoints)
        trial_df = onsets.get_onset_df()
        for tr in range(nr_trs):
    
            # find time at the middle of TR
            if stim_at_half_TR:
                tr_in_sec = (tr * onsets.TR)+0.5*onsets.TR
            else:
                tr_in_sec = (tr * onsets.TR)

            ix,_ = utils.find_nearest(trial_df['onset'].values, tr_in_sec)
            
            # zero-pad number
            # find how much we need to zfill > 
            # https://stackoverflow.com/questions/2189800/how-to-find-length-of-digits-in-an-integer
            zfilling = len(str(len(os.listdir(screenshot_path))))
            if ix > 1:
                img_number = str(ix-1).zfill(zfilling)
            else:
                img_number = "1".zfill(zfilling)

            image_file = utils.get_file_from_substring(img_number, screenshot_path)
            
            if image_file != None:

                img = (255*mpimg.imread(image_file)).astype('int')

                if img.shape[0] != img.shape[1]:
                    offset = int((img.shape[1]-img.shape[0])/2)
                    img = img[:, offset:(offset+img.shape[0])]

                # downsample
                downsampling_constant = int(img.shape[1]/n_pix)
                downsampled_img = img[::downsampling_constant, ::downsampling_constant]

                if downsampled_img[:, :, 0].shape != design_matrix[..., 0].shape:
                    print("please choose a n_pix value that is a divisor of " +
                        str(img.shape[0]))

                # binarize image into dm matrix
                # assumes: standard RGB255 format; only colors present in image are black, white, grey, red, green.
                design_matrix[:, :, tr][np.where(((downsampled_img[:, :, 0] == 0) & (
                    downsampled_img[:, :, 1] == 0)) | ((downsampled_img[:, :, 0] == 255) & (downsampled_img[:, :, 1] == 255)))] = 1

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
    return math.sqrt(math.pow(x2 - x1, 2) +
                math.pow(y2 - y1, 2) * 1.0)
 

def generate_model_params(model='gauss', dm=None, TR=1.5, outputdir=None, settings=None):

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
        yml_file = utils.get_file_from_substring("prf_analysis", opj(os.environ['DIR_SCRIPTS'], 'data'))

    with open(yml_file) as file:
        
        data = yaml.safe_load(file)

        prf_stim = stimulus.PRFStimulus2D(screen_size_cm=data['screen_size_cm'],
                                          screen_distance_cm=data['screen_size_cm'],
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

        # update settings file if we've generated a new one
        if settings == None:
            data.update(bounds)
            data.update(grids)
            data.update({'model': model})
            data.update({'TR': TR})

        date = datetime.now().strftime("%Y%m%d")

    if settings == None:
        fname = opj(outputdir, f'{date}_model-{model}_desc-settings.yml')
        with open(fname, 'w') as yml_file:
            yaml.safe_dump(data, yml_file)

        return data, fname, prf_stim

    else:
        return data, settings, prf_stim


class pRFmodelFitting(): 

    """pRFmodelFitting

    Main class to perform all the pRF-fitting. As of now, the Gaussian and Divisive Normalization models are implemented. For each model, an analysis-file is produced an stored in <output_dir> for later reference.

    Parameters
    ----------
    data: numpy.ndarray
        <voxels,time> numpy array
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
    hrf: np.ndarray
        <1,time_points> describing the HRF. Can be created with :func:`linescanning.glm.double_gamma`, then add an
        axis before the timepoints:
        
        >>> time_points = np.linspace(0,36,np.rint(float(36)/dt).astype(int))
        >>> hrf_custom = linescanning.glm.double_gamma(time_points, lag=6)
        >>> hrf_custom = hrf_custom[np.newaxis,...]
    verbose: bool
        print messages to the terminal

    Example
    ----------
    >>> from linescanning.prf import pRFmodelFitting
    >>> fitting = pRFmodelFitting(func, design_matrix=dm, model='gauss')
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
                 verbose=True,
                 hrf=None,
                 settings=None):
    

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

        #----------------------------------------------------------------------------------------------------------------------------------------------------------
        # Fetch the settings
        self.settings, self.settings_file, self.prf_stim = generate_model_params(model=self.model, 
                                                                                 dm=self.design_matrix, 
                                                                                 outputdir=self.output_dir, 
                                                                                 TR=self.TR,
                                                                                 settings=self.settings_fn)

        if verbose:
            print(f"Using settings file: {self.settings_file}")
    
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


    def fit(self):
        ## Initiate fitter
        ### add check whether we need to transpose data
        self.gaussian_fitter = Iso2DGaussianFitter(data=self.data, model=self.gaussian_model, fit_css=False)
        
        if self.verbose:
            print("Starting gauss grid fit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

        ## start grid fit
        start = time.time()
        self.gaussian_fitter.grid_fit(ecc_grid=self.settings['grids']['eccs'],
                                    polar_grid=self.settings['grids']['polars'],
                                    size_grid=self.settings['grids']['sizes'],
                                    pos_prfs_only=self.settings['pos_prfs_only'])
        
        elapsed = (time.time() - start)

        if self.verbose:
            print("Gaussian gridfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+
                    ". voxels/vertices above "+str(self.settings['rsq_threshold'])+": "+str(np.sum(self.gaussian_fitter.gridsearch_params[:, -1]>self.settings['rsq_threshold']))+" out of "+
                    str(self.gaussian_fitter.data.shape[0]))
            print(f"Gridfit took {str(timedelta(seconds=elapsed))}")
            print("Mean rsq>"+str(self.settings['rsq_threshold'])+": "+str(np.mean(self.gaussian_fitter.gridsearch_params[self.gaussian_fitter.gridsearch_params[:, -1]>self.settings['rsq_threshold'], -1])))
        
        self.gauss_grid = utils.filter_for_nans(self.gaussian_fitter.gridsearch_params)
        if self.write_files:
            self.save_params(model="gauss", stage="grid", verbose=self.verbose, output_dir=self.output_dir, output_base=self.output_base)

        #----------------------------------------------------------------------------------------------------------------------------------------------------------
        # Check if we should do Gaussian iterfit

        ## Run iterative fit with Gaussian model
        if self.stage == "grid+iter":

            start = time.time()
            print("Starting gauss iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

            self.gauss_bounds = [tuple(self.settings['bounds']['x']),              # x
                                 tuple(self.settings['bounds']['y']),               # y
                                 tuple(self.settings['bounds']['size']),            # prf size
                                 tuple(self.settings['bounds']['prf_ampl']),        # prf amplitude
                                 tuple(self.settings['bounds']['bold_bsl'])]        # bold baseline    

            self.gaussian_fitter.iterative_fit(rsq_threshold=self.settings['rsq_threshold'],
                                            bounds=self.gauss_bounds)

            elapsed = (time.time() - start)

            if self.verbose:
                print("Gaussian iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(self.settings['rsq_threshold'])+": "+str(np.mean(self.gaussian_fitter.iterative_search_params[self.gaussian_fitter.rsq_mask, -1])))
                print(f"Iterfit took {str(timedelta(seconds=elapsed))}")
            
            self.gauss_iter = utils.filter_for_nans(self.gaussian_fitter.iterative_search_params)
            if self.write_files:
                self.save_params(model="gauss", stage="iter", verbose=self.verbose, output_dir=self.output_dir, output_base=self.output_base)

        #----------------------------------------------------------------------------------------------------------------------------------------------------------
        # Check if we should do DN-model
        if self.model.lower() == "norm":
            
            ## Define settings/grids/fitter/bounds etcs
            settings, settings_file = generate_model_params(model='norm', dm=self.design_matrix, outputdir=self.output_dir)
            print(f"Using settings file: {settings_file}")

            self.old_params_arr = self.gaussian_fitter.iterative_search_params

            # make n_units x 4 array, with X,Y,size,r2
            self.old_params_filt = np.hstack((self.old_params_arr[:,:3], self.old_params_arr[:,-1][...,np.newaxis]))

            self.norm_model = Norm_Iso2DGaussianModel(stimulus=self.prf_stim,
                                                    filter_predictions=self.settings['filter_predictions'],
                                                    filter_type='sg',
                                                    filter_params={'window_length': self.settings['filter_window_length'], 'polyorder': self.settings['filter_polyorder']})

            self.norm_fitter = Norm_Iso2DGaussianFitter(self.norm_model, self.data)

            self.surround_amplitude_grid = np.array(self.settings['norm']['surround_amplitude_grid'], dtype='float32')
            self.surround_size_grid      = np.array(self.settings['norm']['surround_size_grid'], dtype='float32')
            self.neural_baseline_grid    = np.array(self.settings['norm']['neural_baseline_grid'], dtype='float32')
            self.surround_baseline_grid  = np.array(self.settings['norm']['surround_baseline_grid'], dtype='float32')

            self.norm_bounds = [tuple(settings['bounds']['x']),               # x
                                tuple(settings['bounds']['y']),               # y
                                tuple(settings['bounds']['size']),            # prf size
                                tuple(settings['bounds']['prf_ampl']),        # prf amplitude
                                tuple(settings['bounds']['bold_bsl']),        # bold baseline
                                tuple(settings['bounds']['surr_ampl']),       # surround amplitude
                                tuple(settings['bounds']['surr_size']),       # surround size
                                tuple(settings['bounds']['neur_bsl']),        # neural baseline
                                tuple(settings['bounds']['surr_bsl'])]        # surround baseline

            #----------------------------------------------------------------------------------------------------------------------------------------------------------
            ## Start grid fit
            start = time.time()
            print("Starting norm grid fit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
            self.norm_fitter.grid_fit(self.surround_amplitude_grid, 
                                    self.surround_size_grid, 
                                    self.neural_baseline_grid,
                                    self.surround_baseline_grid, 
                                    gaussian_params=self.old_params_filt)

            elapsed = (time.time() - start)

            if self.verbose:
                print("Norm gridfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(self.settings['rsq_threshold'])+": "+str(np.mean(self.norm_fitter.gridsearch_params[self.norm_fitter.gridsearch_rsq_mask, -1])))
                print(f"Gridfit took {str(timedelta(seconds=elapsed))}")
            
            ### save grid parameters
            self.norm_grid = utils.filter_for_nans(self.norm_fitter.gridsearch_params)
            if self.write_files:
                self.save_params(model="norm", stage="grid", verbose=self.verbose, output_dir=self.output_dir, output_base=self.output_base)

            #----------------------------------------------------------------------------------------------------------------------------------------------------------
            ## Check if we should do iterative fitting with DN model
            if self.stage == "grid+iter":
                start = time.time()
                if self.verbose:
                    print("Starting norm iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

                self.norm_fitter.iterative_fit(rsq_threshold=settings['rsq_threshold'],
                                            bounds=self.norm_bounds,
                                            starting_params=self.norm_fitter.gridsearch_params)
                
                elapsed = (time.time() - start)  
                print("Norm iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(settings['rsq_threshold'])+": "+str(np.mean(self.norm_fitter.iterative_search_params[self.norm_fitter.rsq_mask, -1])))
                print(f"Iterfit took {str(timedelta(seconds=elapsed))}")

                ### save iterative parameters
                self.norm_iter = utils.filter_for_nans(self.norm_fitter.iterative_search_params)
                if self.write_files:
                    self.save_params(model="norm", stage="iter", verbose=self.verbose, output_dir=self.output_dir, output_base=self.output_base)


    def load_params(self, params_file, model='gauss', stage='iter'):

        """Load in a numpy array into the class; allows for quick plotting of voxel timecourses"""

        if isinstance(params_file, str):
            params = np.load(params_file)

        elif isinstance(params_file, np.ndarray):
            params = params_file.copy()
        else:
            raise ValueError("Unrecognized input type for 'params_file'")

        setattr(self, f'{model}_{stage}', params)
        
    
    def plot_vox(self, vox_nr='best', model='gauss', stage='iter'):

        """plot real and predicted timecourses for a voxel"""
        
        if model == "gauss":
            use_model = self.gaussian_model
        elif model == "norm":
            use_model = self.norm_model

        if hasattr(self, f"{model}_{stage}"):
            params = getattr(self, f"{model}_{stage}")

            if vox_nr == "best":
                vox,_ = utils.find_nearest(params[...,-1], np.amax(params[...,-1]))
            else:
                vox = vox_nr

            params = params[vox, ...]
        else:
            raise ValueError(f"Could not find {stage} parameters for {model}")

        self.prediction = use_model.return_prediction(*params[:-1]).T

        prf_array = make_prf(self.prf_stim, size=params[2], mu_x=params[0], mu_y=params[1])

        fig = plt.figure(constrained_layout=True, figsize=(20,5))
        gs00 = fig.add_gridspec(1,2, width_ratios=[10,20])

        # make plot 
        ax1 = fig.add_subplot(gs00[0])
        ax1.axvline(0, color='white', linestyle='dashed', lw=0.5)
        ax1.axhline(0, color='white', linestyle='dashed', lw=0.5)
        im = ax1.imshow(np.squeeze(prf_array,axis=0), extent=self.settings['vf_extent']+self.settings['vf_extent'], cmap='magma')
        patch = patches.Circle((0, 0), radius=self.settings['vf_extent'][-1], transform=ax1.transData)
        im.set_clip_path(patch)
        ax1.axis('off')

        # make plot 
        ax2 = fig.add_subplot(gs00[1])
        ax2.axhline(0, color='k', linestyle='dashed', lw=0.5)

        # annoying indexing issues.. lots of inconsistencies in array shapes.
        try:
            im = ax2.plot(self.data.T[vox,...], lw=2, color='#08B2F0', label='real')
        except:
            im = ax2.plot(self.data[vox,...], lw=2, color='#08B2F0', label='real')

        im = ax2.plot(self.prediction, lw=2, color='#cccccc', label='pred')
        ax2.set_ylabel("BOLD amplitude (z-score)").set_fontsize(14)
        ax2.set_xlabel("Volumes").set_fontsize(14)
        ax2.legend(frameon=False)
        sns.despine(offset=10)

        return params

    def save_params(self, model="gauss", stage="grid", verbose=False, output_dir=None, output_base=None):

        if hasattr(self, f"{model}_{stage}"):
            params = getattr(self, f"{model}_{stage}")
            output = opj(output_dir, f'{output_base}_model-{model}_stage-{stage}_desc-prf_params.npy')
            np.save(output, params)

            if verbose:
                print(f"Save {stage}-fit parameters in {output}")
        
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

    if return_nr == "all":
        return xysize_par
    else:
        return xysize_par[:return_nr]



