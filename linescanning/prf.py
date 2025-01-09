import mkl
mkl.set_num_threads=1
standard_max_threads = mkl.get_max_threads()

import ast
from datetime import datetime, timedelta
from linescanning import (
    utils, 
    plotting, 
    dataset, 
    fitting,
    preproc
)
import math
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from joblib import Parallel, delayed
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

    # check that we have enough parameters
    for ii,par in zip([src,targ],["source","target"]):
        if len(ii) < 3:
            raise ValueError(f"need at least three parameters (x,y,sigma) for '{par}' pRF, got {len(ii)}.")

    # get overlap prior to normalization
    x = distance_centers(src,targ)/src[2]
    
    # normalize
    norm = targ.copy()
    norm[0] -= src[0]
    norm[1] -= src[1]
    norm[2] /= src[2]

    # get new distance
    x_prime = distance_centers([0,0], norm)

    # shift position so that distances are the same again
    p_prime = []
    for ix,ii in enumerate(["x","y"]):
        corrected_position = norm[ix]/x_prime * x
        p_prime.append(corrected_position)

    # p_prime only has position, add size back to it
    p_prime.append(norm[2])

    return p_prime
    
def read_par_file(prf_file, key="pars"):

    """read_par_file

    Function to read in files containing pRF-estimates from various sources. Currently, supports the following inputs: `np.ndarray`, `npy`-file, `mat`-file (will read in the data from the last item in `list(prf_file.keys())`), `pkl`-file (assumes has key 'pars' in it). Because we generally save the design matrix as a *.mat*-file, we can use the same function to read in that file and obtain the design matrix in `np.ndarray`-format. Inputs `pd.DataFrame` will be converted to `np.ndarray` with :func:`linescanning.prf.Parameters`, assuming certain column names to be present.

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
        if os.path.exists(prf_file):
            if prf_file.endswith("npy"):
                pars = np.load(prf_file)
            elif prf_file.endswith("mat"):
                tmp = io.loadmat(prf_file)
                pars = tmp[list(tmp.keys())[-1]] # find last key in list
            elif prf_file.endswith("pkl"):
                with open(prf_file, 'rb') as input:
                    data = pickle.load(input)

                    try:
                        pars = data[key]
                    except:
                        raise TypeError(f"Pickle-file did not arise from 'pRFmodelFitting'. I'm looking for '{key}', but got '{data.keys()}'")
        else:
            raise TypeError(f"Input '{prf_file}' is not a file..")
        
    elif isinstance(prf_file, np.ndarray):
        pars = prf_file.copy()
    elif isinstance(prf_file, pd.DataFrame):
        pars = Parameters(prf_file).to_array()
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


def make_stims(
    xy, 
    dim_stim=2, 
    factor=4, 
    concentric_size=0.65, 
    loc=(0,0),
    dt="fill"):

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
    loc: tuple, optional
        location to center stimuli on. Default is (0,0)
    dt: str
        Set the type of stimulus that needs to be created. by default 'fill', other options are 'full' [fullscreen stimulus], 'concentric' [circles with holes], 'hole' [fullscreen with growing hole]
    concentric_size: float
        proportion of stimulus size that needs to be masked out again
    Returns
    ----------
    list
        list of numpy.ndarrays with meshgrid-size containing the stimuli. Can be plotted with :func:`linescanning.prf.plot_stims`
    """

    if not isinstance(xy,tuple):
        xy = (xy,xy)

    if dim_stim == 1:
        stims = [np.zeros_like(xy[0]) for n in range(int(xy.shape[-1]/factor))]
    else:
        stims = [np.zeros((xy[1].shape[0],xy[0].shape[0]), dtype=int) for n in range(int(xy[0].shape[-1]/(2*factor)))]
        stim_sizes=[]
    
    if dt != "full":
        for pp, stim in enumerate(stims):

            if dim_stim == 1:
                #2d rectangle or 1d
                stim[int(stim.shape[0]/2)-pp:int(stim.shape[0]/2)+pp] = 1
            else:
                #2d circle
                xx,yy = np.meshgrid(xy[0],xy[1])
                stim[(((xx-loc[0])**2+(yy-loc[1])**2)**0.5)<(xy[0].max()*pp/(len(stims)*factor))] = 1
                stim_sizes.append(2*(xy[0].max()*pp/(len(stims)*factor)))

                # make concentric rings
                if dt == "concentric":
                    stim_rad = radius(stim)

                    if stim_rad > 2:
                        mask = create_circular_mask(xy.shape[1], xy.shape[0], radius=concentric_size*stim_rad)
                        stim[mask] = 0

        if dt == "hole":
            stims_hole = []
            for stim in stims:
                stim_inv = np.ones_like(stim)
                stim_inv[stim>0] = 0
                stims_hole.append(stim_inv)

            stims = stims_hole.copy()

    else:
        stims = [np.ones((xy[1].shape[0],xy[0].shape[0]))]
        stim_sizes.append(np.inf)

    if dim_stim > 1:
        return np.concatenate([ii[...,np.newaxis] for ii in stims], axis=-1), stim_sizes
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
    n_cols=10, 
    figsize=None, 
    extent=([-5,5],[-5,5]),
    save_as=None,
    axis_on=False,
    prf_object=None,
    as_circle=False,
    prf_color="g",
    interval=1,
    add_fixation=False,
    flipud=True,
    stim_color=None):

    """plot_stims

    Plots all stimuli contained in `stims` as created in `prf.make_stims`. Assumesthat about 33 stimuli have been produced, otherwise axes ordering will be messy..

    Parameters
    ----------
    stims: list
        list of numpy.ndarrays with meshgrid-size containing the stimuli
    n_cols: int, optional
        number of columns to use. Number of rows will be automatically derived from the number of `stims` and `interval`
    figsize: tuple, optional
        custom figure size; otherwise will be determined from the number of columns and rows
    save_as: str, optional 
        save the figure
    axis_on: bool, optional
        show the axes in the figures. Default is false
    prf_object: np.ndarray, optional
        add a spatial representation of a pRF to the figure
    as_circle: bool, optional
        plot `prf_object` as a simple circle rather than full on pRF-object. The outline of the circle will denote the size of the pRF as per `prfpy`'s convention
    prf_color: str, optional
        color of `prf_object`. If `as_circle==True`, the outline will have this color. If not, a binary colormap will be created using :func:`linescanning.utils.make_binary_cm()`
    interval: int, optional
        show the stimuli with a certain interval. This prevents a mega-amount of figures that needs to be created.
    stim_color: str, optional
        color of `stims`. Binary colormap will be created using :func:`linescanning.utils.make_binary_cm()`. If false, colormap will be `viridis`
    add_fixation: bool, optional
        add fixation cross add (0,0). If `stim_color` is specified, the color will be black; with viridis, the cross will be white

    Returns
    ----------
    matplotlib.pyplot plot
    """

    # get number of plots
    if isinstance(stims, list):
        n_elements = len(stims)
        scalar = stims[0].shape[1]/stims[0].shape[0]
    else:
        n_elements = stims.shape[-1]
        scalar = stims.shape[1]/stims.shape[0]

    # calculate nr of rows based on nr of elements (account for interval-flag)
    n_elements = int(round(n_elements/interval))
    n_rows = int(round(n_elements/n_cols))

    if figsize == None:
        figsize = (24,n_rows*scalar)

    fig,_ = plt.subplots(
        n_rows, 
        n_cols, 
        figsize=figsize,
        constrained_layout=True)

    if isinstance(prf_object, np.ndarray):
        add_prf = True
    else:
        add_prf = False

    if isinstance(stim_color, (tuple,str)):
        stim_cm = utils.make_binary_cm(stim_color)
        cross_hair_color = "black"
    else:
        stim_cm = "viridis"
        cross_hair_color = "white"

    start = 0
    for i, ax in enumerate(fig.axes):

        try:
            # decide input type
            if isinstance(stims, list):
                data = stims[start]
            else:
                data = stims[...,start]
            
            # decide if we need to flip; actual screenshot do no have to be flipped
            if flipud:
                data = np.flipud(data)

            # plot
            ax.imshow(data, extent=extent[0]+extent[1], cmap=stim_cm)
            
            if add_fixation:
                # line on x-axis
                ax.axvline(
                    0, 
                    color=cross_hair_color, 
                    linestyle='dashed', 
                    lw=0.5,
                    zorder=1)

                # line on y-axis
                ax.axhline(
                    0, 
                    color=cross_hair_color, 
                    linestyle='dashed', 
                    lw=0.5,
                    zorder=1)

            if add_prf:
                if as_circle:
                    prf_ = plt.Circle(
                        (prf_object[0],prf_object[1]),
                        prf_object[2],
                        ec=prf_color,
                        fill=False,
                        lw=2)

                    ax.add_artist(prf_) 

                else:
                    cm = utils.make_binary_cm(prf_color)
                    prf_ = make_prf(prf_object)
                    plotting.LazyPRF(
                        prf_, 
                        extent,
                        axs=ax,
                        cmap=cm,
                        cross_color="k",
                        edge_color=None,
                        shrink_factor=0.9,
                        vf_only=True)

            if not axis_on:
                ax.axis('off')

            start += interval
        except:
            ax.axis("off")

    plt.tight_layout()
    if isinstance(save_as, str):
        fig.savefig(save_as, facecolor='white', dpi=300, bbox_inches='tight')
    # return fig

def make_prf(
    prf_object, 
    params,
    model="gauss",
    resize_pix=None, 
    **kwargs):

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

    # prf = np.rot90(rf.gauss2D_iso_cart(
    #     x=prf_object.x_coordinates[..., np.newaxis],
    #     y=prf_object.y_coordinates[..., np.newaxis],
    #     mu=(params[0], params[1]),
    #     sigma=params[2],
    #     normalize_RFs=False).T, axes=(1, 2))

    prf = create_model_rf_wrapper(
        model,
        prf_object,
        params)

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

    if isinstance(stims, np.ndarray):
        n_stims = stims.shape[-1]
    elif isinstance(stims, list):
        n_stims = len(stims)
    else:
        raise TypeError(f"Stimuli must be a list or np.ndarray, not {type(stims)}")

    sr_function = ((a[...,np.newaxis]*np.sum(np.tile(np.exp(-((xx[...,np.newaxis]-mu_x)**2+(yy[...,np.newaxis]-mu_y)**2)/(2*s_1**2))[...,np.newaxis],n_stims)*stims[:,:,np.newaxis,:],axis=(0,1)) +b[...,np.newaxis])/\
    ((c[...,np.newaxis]*np.sum(np.tile(np.exp(-((xx[...,np.newaxis]-mu_x)**2+(yy[...,np.newaxis]-mu_y)**2)/(2*s_2**2))[...,np.newaxis],n_stims)*stims[:,:,np.newaxis,:],axis=(0,1)) +d[...,np.newaxis])) - (b/d)[...,np.newaxis])    
    
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
def pix2deg(
    pixels, 
    prf_object=None, 
    scrSizePix=[270, 270],
    scrWidthCm=39.3,
    scrDist=196
    ):
    """Convert size in pixels to size in degrees for a given Monitor object""" 

    # get monitor params and raise error if necess
    if not isinstance(scrWidthCm, (int,float)) and not isinstance(scrDist, (int,float)):
        scrWidthCm = prf_object.screen_size_cm
        scrDist = prf_object.screen_distance_cm

    cmSize = pixels * float(scrWidthCm) / scrSizePix[0]
    return old_div(cmSize, (scrDist * 0.017455))


# Adapted from psychopy
def deg2pix(
    degrees, 
    prf_object=None, 
    scrSizePix=[270, 270],
    scrWidthCm=39.3,
    scrDist=196
    ):
    """Convert size in degrees to size in pixels for a given Monitor object""" 

    # get monitor params and raise error if necess
    if not isinstance(scrWidthCm, (int,float)) and not isinstance(scrDist, (int,float)):
        scrWidthCm = prf_object.screen_size_cm
        scrDist = prf_object.screen_distance_cm

    cmSize = np.array(degrees) * scrDist * 0.017455
    return int(round(cmSize * scrSizePix[0] / float(scrWidthCm), 0))


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
        
        utils.verbose(f"Design has shape: {design.shape}", verbose)
        if return_onsets:
            # # create onset df
            # try:
            utils.verbose("Creating onset dataframe", verbose)
            onset_frames = get_onset_idx(design)
            onset_times = onset_frames*TR

            onset_dict = {'onset': onset_times,
                          'event_type': stim_shuff}

            onset_df = pd.DataFrame(onset_dict)
            utils.verbose("Done", verbose)

            return design, onset_df

        else:

            utils.verbose("Done", verbose)
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

    if isinstance(screenshot_path, list):
        raise TypeError(f"Found multiple ({len(screenshot_path)}) instances for 'Screenshot'. Maybe hidden directories from MacOS?; \n{screenshot_path}")

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

        utils.verbose("Creating design matrix (can take a few minutes with thousands of TRs)", verbose)

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
                search_for = f"Screenshots{img_number}.png"
                try:
                    image_file = utils.get_file_from_substring(search_for, screenshot_path)
                except:
                    image_file = None

                if isinstance(image_file, list):
                    raise ValueError(f"Found multiple ({len(image_file)}) files for '{search_for}': {image_file}")
                
                if image_file != None:
                    
                    img = (255*mpimg.imread(image_file)).astype('int')

                    if img.shape[0] != img.shape[1]:
                        offset = int((img.shape[1]-img.shape[0])/2)
                        img = img[:, offset:(offset+img.shape[0])]

                    # binarize image into dm matrix
                    # assumes: standard RGB255 format; only colors present in image are black, white, grey, red, green.
                    design_matrix[..., tr][np.where(((img[..., 0] < 40) & (img[..., 1] < 40)) |
                                                    ((img[..., 0] > 200) & (img[..., 1] > 200)))] = 1
                else:
                    print(f"WARNING; could not find image for trial #{tr} ({search_for})")

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
 

def generate_model_params(
    model='gauss', 
    dm=None, 
    TR=1.5, 
    fit_hrf=False, 
    verbose=False,
    old_settings=None):

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
    old_settings: dict, optional
        Dictionary describing an existing set of settings (e.g., from previous fit)
        
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

    if not isinstance(old_settings, dict):
        if isinstance(yml_file, str):
            utils.verbose(f"Reading settings from '{yml_file}'", verbose)
            with open(yml_file) as file:
                settings = yaml.safe_load(file)
        else:
            raise ValueError("Could not read settings")
    else:
        utils.verbose(f"Reading manually specified settings", verbose)
        settings = old_settings

    if isinstance(dm, str):
        dm = read_par_file(dm)
    
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

    hrf1 = np.linspace(0,10,10)
    hrf2 = np.linspace(0,0,1)

    grids = {
        'screensize_degrees': float(ss), 
        'grids': {
            'sizes': [float(item) for item in sizes], 
            'eccs': [float(item) for item in eccs], 
            'polars': [float(item) for item in polars],
            'hrf1': hrf1,
            'hrf2': hrf2
        }
    }
    
    allowed_models = ['gauss', 'css', 'dog', 'norm', 'abc', 'abd']
    if model not in allowed_models:
        raise ValueError(f"Model must be one of {allowed_models}. Not '{model}'")

    # define standard bounds, then add model-specific stuff
    standard_bounds = [
        coord_bounds,           # x
        coord_bounds,           # y
        prf_size,               # prf size
        settings['prf_ampl'],   # prf amplitude
        settings['bold_bsl']    # bold baseline
    ]
    
    # define bounds
    if model == "gauss":
        gauss_bounds = standard_bounds.copy()
        bounds = {
            'bounds': {
                'x': list(gauss_bounds[0]), 
                'y': list(gauss_bounds[1]), 
                'size': [float(item) for item in gauss_bounds[2]], 
                'prf_ampl': gauss_bounds[3], 
                'bold_bsl': gauss_bounds[4]
            }
        }

    elif model == "css":
        css_bounds = standard_bounds.copy() + [settings['css_exponent']] # CSS exponent
        bounds = {
            'bounds': {
                'x': list(css_bounds[0]),
                'y': list(css_bounds[1]),
                'size': [float(item) for item in css_bounds[2]],
                'prf_ampl': css_bounds[3],
                'bold_bsl': css_bounds[4],
                'css_exponent': css_bounds[5]
            }
        }

    elif model == "dog":
        dog_bounds = standard_bounds.copy() + [
            settings[model]['surround_amplitude_bound'],    # surround amplitude
            (settings['eps'], 3*ss)                         # surround size
        ]

        bounds = {
            'bounds': {
                'x': list(dog_bounds[0]),
                'y': list(dog_bounds[1]),
                'size': [float(item) for item in dog_bounds[2]],
                'prf_ampl': dog_bounds[3],
                'bold_bsl': dog_bounds[4],
                'surr_ampl': dog_bounds[5],
                'surr_size': [float(item) for item in dog_bounds[6]]
            }
        }
    else:
        # set D-param to 1
        if model == "abc":
            neur_bsl_bound = [1,1]
            surr_bsl_bound = settings['norm']['surround_baseline_bound']
        # set C-param to 1
        elif model == "abd":
            neur_bsl_bound = settings['norm']['neural_baseline_bound']
            surr_bsl_bound = [1,1]
        # ful model
        else:
            neur_bsl_bound = settings['norm']['neural_baseline_bound']
            surr_bsl_bound = settings['norm']['surround_baseline_bound']

        norm_bounds = standard_bounds.copy() + [
            settings['norm']['surround_amplitude_bound'],    # surround amplitude
            (settings['eps'], 3*ss),                        # surround size
            neur_bsl_bound,                                 # neural baseline
            surr_bsl_bound                                  # surround baseline
        ]

        bounds = {
            'bounds': {
                'x': list(norm_bounds[0]), 
                'y': list(norm_bounds[1]), 
                'size': [float(item) for item in norm_bounds[2]], 
                'prf_ampl': norm_bounds[3], 
                'bold_bsl': norm_bounds[4],
                'surr_ampl': norm_bounds[5],
                'surr_size': [float(item) for item in norm_bounds[6]],
                'neur_bsl': list(norm_bounds[7]),
                'surr_bsl': [float(item) for item in norm_bounds[8]]
            }
        }

    if fit_hrf:
        bounds['bounds']["hrf_deriv"] = settings["hrf"]["deriv_bound"]
        bounds['bounds']["hrf_disp"] = settings["hrf"]["disp_bound"]

    # update settings file if we've generated a new one
    settings.update(bounds)
    settings.update(grids)
    settings.update({'model': model})
    settings.update({'TR': TR})

    # print important settings
    utils.verbose("\n---------------------------------------------------------------------------------------------------", verbose)
    utils.verbose("Check these important settings!", verbose)
    utils.verbose(f" Screen distance: {settings['screen_distance_cm']}cm", verbose)
    utils.verbose(f" Screen size: {settings['screen_size_cm']}cm", verbose)
    utils.verbose(f" TR: {settings['TR']}s", verbose)
    utils.verbose("---------------------------------------------------------------------------------------------------\n", verbose)

    # return
    return settings, prf_stim

class GaussianModel():

    def __init__(self):     

        self.gaussian_fitter = Iso2DGaussianFitter(
            data=self.data,
            model=self.gauss_model,
            fit_css=False,
            fit_hrf=self.fit_hrf,
            n_jobs=self.nr_jobs)

        if isinstance(self.old_params, np.ndarray):
            
            # check if dimensions make sense
            if self.old_params.shape[0] != self.data.shape[0]:
                utils.verbose(f"Matching old parameters of shape {self.old_params.shape} to data of shape {self.data.shape[0],self.old_params.shape[-1]}", self.verbose)
                self.old_params = np.tile(self.old_params, (self.data.shape[0],1))

            # set inserted params as gridsearch_params and iterative_search_params
            # needed for the rsq-mask
            self.gaussian_fitter.gridsearch_params = self.old_params.copy()
            self.gaussian_fitter.iterative_search_params = self.old_params.copy()   # actual parameters

            # set gaussian_fitter as previous_gaussian_fitter
            self.previous_gaussian_fitter = self.gaussian_fitter

            utils.verbose(f"Inserting parameters from {type(self.old_params)} as 'iterative_search_params' in {self}", self.verbose)         

    def gridfit(self):
        ## start grid fit
        utils.verbose(f"Starting gauss gridfit {self.data.shape} at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}", self.verbose)
        
        start = time.time()
        self.gaussian_fitter.grid_fit(
            ecc_grid=self.settings['grids']['eccs'],
            polar_grid=self.settings['grids']['polars'],
            size_grid=self.settings['grids']['sizes'],
            fixed_grid_baseline=self.fix_grid_baseline,
            verbose=self.verbose,
            grid_bounds=[tuple(self.settings['bounds']['prf_ampl'])],
            n_batches=self.nr_jobs)
        
        elapsed = (time.time() - start)

        self.gauss_grid = utils.filter_for_nans(self.gaussian_fitter.gridsearch_params)
        mean_rsq = np.mean(self.gauss_grid[self.gauss_grid[:, -1]>self.settings['rsq_threshold'], -1])
        
        # verbose stuff
        start_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        nr = np.sum(self.gauss_grid[:, -1]>self.settings['rsq_threshold'])
        total = self.gaussian_fitter.data.shape[0]
        utils.verbose(f"Completed Gaussian gridfit at {start_time}. Voxels/vertices above {self.settings['rsq_threshold']}: {nr}/{total}", self.verbose)
        utils.verbose(f"Gridfit took {timedelta(seconds=elapsed)} | Mean rsq>{self.settings['rsq_threshold']}: {round(mean_rsq,2)}", self.verbose)
        
        if self.write_files:
            if self.save_grid:
                self.save_params(model="gauss", stage="grid") 

    def iterfit(self):

        start = time.time()
        utils.verbose(f"Starting gauss iterfit {self.data.shape} at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}", self.verbose)

        # fetch bounds
        self.gauss_bounds = self.fetch_bounds(model='gauss')

        if isinstance(self.fix_parameters, (list,int)):
            utils.verbose(f"Fixing parameters (idx): {self.fix_parameters}", self.verbose)

            self.gauss_bounds = self.unitwise_bounds(
                self.fix_parameters,
                model="gauss",
                ref_pars=self.gaussian_fitter.gridsearch_params
            )

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

        # verbose stuff
        mean_rsq = np.nanmean(self.gauss_iter[self.gaussian_fitter.rsq_mask, -1])
        start_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')

        utils.verbose(f"Completed Gaussian iterfit at {start_time}. Mean rsq>{self.settings['rsq_threshold']}: {round(mean_rsq,2)}", self.verbose)
        utils.verbose(f"Iterfit took {timedelta(seconds=elapsed)}", self.verbose)

        # save intermediate files
        if self.write_files:
            self.save_params(model="gauss", stage="iter")  
            
class ExtendedModel():

    def __init__(self):

        if self.model == "dog":
            self.active_fitter = DoG_Iso2DGaussianFitter
        elif self.model == "css":
            self.active_fitter = CSS_Iso2DGaussianFitter
        elif self.model in ["norm","abc","abd"]:
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
                    previous_gaussian_fitter=self.gaussian_fitter,
                    n_jobs=self.nr_jobs)
            else:
                # we might have manually injected parameters
                self.tmp_fitter = self.active_fitter(
                    self.active_model, 
                    self.data, 
                    fit_hrf=self.fit_hrf,
                    n_jobs=self.nr_jobs)
        else:
            # inject existing-model object > advised when fitting the HRF
            self.tmp_fitter = self.active_fitter(
                self.active_model, 
                self.data, 
                fit_hrf=self.fit_hrf, 
                previous_gaussian_fitter=self.previous_gaussian_fitter,
                n_jobs=self.nr_jobs)

    def gridfit(self):
        
        #----------------------------------------------------------------------------------------------------------------------------------------------------------

        # define grids into lists so we can call gridfit ones
        if self.model == "css":
            self.grid_list = [
                np.array(self.settings['css']['css_exponent_grid'], dtype='float32')
            ]

            self.grid_bounds = [
                tuple(self.settings['bounds']['prf_ampl'])
            ]

        elif self.model == "dog":
            self.grid_list = [
                np.array(self.settings['dog']['dog_surround_amplitude_grid'], dtype='float32'),
                np.array(self.settings['dog']['dog_surround_size_grid'], dtype='float32')
            ]      
            self.grid_bounds = [
                tuple(self.settings['prf_ampl']),
                tuple(self.settings['bounds']['surr_ampl'])
            ]

        elif self.model in ["norm","abc","abd"]:
            self.grid_list = [
                np.array(self.settings['norm']['surround_amplitude_grid'], dtype='float32'),
                np.array(self.settings['norm']['surround_size_grid'], dtype='float32'),
                np.array(self.settings['norm']['neural_baseline_grid'], dtype='float32'),
                np.array(self.settings['norm']['surround_baseline_grid'], dtype='float32')
            ]

            self.grid_bounds = [
                tuple(self.settings['bounds']['prf_ampl']),
                tuple(self.settings['bounds']['neur_bsl'])
            ]
        
        # grid fit
        if not self.skip_grid:

            if not self.use_grid_bounds:
                utils.verbose("Ignoring grid bounds", self.verbose)
                self.grid_bounds = None

            ## Start grid fit
            start = time.time()
            utils.verbose(f"Starting {self.model} gridfit {self.data.shape} at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}", self.verbose)

            self.tmp_fitter.grid_fit(
                *self.grid_list,
                n_batches=self.nr_jobs,
                verbose=self.verbose,
                rsq_threshold=self.settings['rsq_threshold'],
                fixed_grid_baseline=self.fix_grid_baseline,
                grid_bounds=self.grid_bounds)

            elapsed = (time.time() - start)

            ### save grid parameters
            filtered_ = utils.filter_for_nans(self.tmp_fitter.gridsearch_params)
            setattr(self, f"{self.model}_grid", filtered_)

            # verbose stuff
            mean_rsq = np.mean(filtered_[self.tmp_fitter.gridsearch_rsq_mask, -1])
            start_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
            utils.verbose(f"Completed {self.model} gridfit at {start_time}. Mean rsq>{self.settings['rsq_threshold']}: {round(mean_rsq,2)}", self.verbose)
            utils.verbose(f"Gridfit took {timedelta(seconds=elapsed)}", self.verbose)

            if self.write_files:
                if self.save_grid:
                    self.save_params(model=self.model, stage="grid")
        else:
            utils.verbose(f"Setting {(type(self.gaussian_fitter.iterative_search_params))} as 'gridsearch_params' in {self.tmp_fitter}", self.verbose)
            self.tmp_fitter.gridsearch_params = self.gaussian_fitter.iterative_search_params

        setattr(self, f"{self.model}_fitter", self.tmp_fitter)

    def iterfit(self):

        # fetch bounds from settings > HRF bounds are automatically appended if fit_hrf=True
        self.tmp_bounds = self.fetch_bounds(model=self.model)
        if isinstance(self.fix_parameters, (list,int)):
            utils.verbose(f"Fixing parameters (idx): {self.fix_parameters}", self.verbose)

            self.tmp_bounds = self.unitwise_bounds(
                self.fix_parameters,
                model=self.model,
                ref_pars=self.previous_gaussian_fitter.iterative_search_params
            )

        setattr(self, f"{self.model}_bounds", self.tmp_bounds)        
        
        start = time.time()
        utils.verbose(f"Starting {self.model} iterfit {self.data.shape}  at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}", self.verbose)

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

        # verbose stuff
        mean_rsq = np.mean(filtered_[self.tmp_fitter.rsq_mask, -1])
        start_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        utils.verbose(f"Completed {self.model} iterfit at {start_time}. Mean rsq>{self.settings['rsq_threshold']}: {round(mean_rsq,2)}", self.verbose)
        utils.verbose(f"Iterfit took {timedelta(seconds=elapsed)}", self.verbose)

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
    skip_settings: bool, optional
        avoids overwriting of particular settings (e.g., screen_distance_cm), and takes them from the reference analysis file instead
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
        fit_hrf=True,
        settings=None,
        nr_jobs=1,
        prf_stim=None,
        model_obj=None,
        fix_bold_baseline=False,
        fix_parameters=None,
        constraints="tc",
        save_grid=True,
        skip_grid=False,
        use_grid_bounds=True,
        mask=None,
        transpose=False,
        skip_settings=False,
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
        self.use_grid_bounds    = use_grid_bounds
        self.mask               = mask
        self.transpose          = transpose
        self.skip_settings      = skip_settings
        self.__dict__.update(kwargs)

        # read design matrix if needed
        if isinstance(self.design_matrix, str):
            utils.verbose(f"Reading design matrix from '{self.design_matrix}'", self.verbose)
            self.design_matrix = read_par_file(self.design_matrix)

        # read design matrix if needed
        if isinstance(self.data, str):
            utils.verbose(f"Reading data from '{self.data}'", self.verbose)
            self.data = read_par_file(self.data)            

        # adjust design matrix to data
        # make data 2D
        if isinstance(self.data, np.ndarray):
            if self.data.ndim == 1:
                self.data = self.data[np.newaxis,...]

            if self.transpose:
                self.data = self.data.T
                
            if self.design_matrix.shape[-1] != self.data.shape[-1]:
                missed_vols = self.design_matrix.shape[-1]-self.data.shape[-1]

                # transpose if this value is negative; means shapes are shifted
                if missed_vols < 0:
                    self.data = self.data.T
                    utils.verbose(f"Skipped volumes was negative ({missed_vols}), transposing data to {self.data.shape}", self.verbose)
                    missed_vols = self.design_matrix.shape[-1]-self.data.shape[-1]

                self.design_matrix = self.design_matrix[...,missed_vols:]

                utils.verbose(f"Design has {missed_vols} more volumes than timecourses, trimming from beginning of design to {self.design_matrix.shape}", self.verbose)

        #----------------------------------------------------------------------------------------------------------------------------------------------------------
        # Fetch the settings
        self.define_settings()

        # overwrite bold baseline tuple if bold baseline should be fixed
        if self.fix_bold_baseline:
            self.settings['bounds']['bold_bsl'] = [0,0]
            self.fix_grid_baseline = 0

            utils.verbose(f"Fixing baseline at {self.settings['bounds']['bold_bsl']}", self.verbose)
        else:
            self.fix_grid_baseline = None

        # check if we got a pRF-stim object
        if self.prf_stim != None:
            utils.verbose("Using user-defined pRF-stimulus object", self.verbose)
            self.prf_stim = self.prf_stim
        else:
            self.prf_stim = self.prf_stim_
        
        self.allowed_models = ["gauss", "css", "dog", "norm", "abc", "abd"]
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
        
        utils.verbose(f"Using constraint(s): {self.constraints}", self.verbose)
            
        #----------------------------------------------------------------------------------------------------------------------------------------------------------
        # whichever model you run, run the Gaussian first

        ## Define models
        self.gauss_model    = Iso2DGaussianModel(stimulus=self.prf_stim, hrf=self.hrf)
        self.css_model      = CSS_Iso2DGaussianModel(stimulus=self.prf_stim, hrf=self.hrf)
        self.dog_model      = DoG_Iso2DGaussianModel(stimulus=self.prf_stim, hrf=self.hrf)

        if self.model in ["norm","abc","abd"]:
            setattr(self, f"{self.model}_model", Norm_Iso2DGaussianModel(stimulus=self.prf_stim, hrf=self.hrf))

        if self.model_obj != None:
            utils.verbose(f"Setting {self.model_obj} as '{model}_model'-attribute", self.verbose)
            setattr(self, f'{model}_model', self.model_obj)

    def define_settings(self, old_settings=None):

        self.settings, self.prf_stim_ = generate_model_params(
            model=self.model, 
            dm=self.design_matrix, 
            TR=self.TR,
            fit_hrf=self.fit_hrf,
            verbose=self.verbose,
            old_settings=old_settings)

    def define_hrf(self):

        # make compatible with prfpy's HRF-class?
        if isinstance(self.hrf, np.ndarray):
            if self.hrf.ndim < 2:
                self.hrf = self.hrf[np.newaxis,...]
            
            utils.verbose(f"Instantiate HRF with: '{type(self.hrf)}' (fit={self.fit_hrf})", self.verbose)
            try:
                self.hrf = HRF(self.hrf)
            except:
                raise ValueError("Cannot use HRF-class. Please specify list of three parameters")

        elif isinstance(self.hrf, list):
            utils.verbose(f"Instantiate HRF with: {self.hrf} (fit={self.fit_hrf})", self.verbose)
            try:
                self.hrf = HRF()
                self.hrf.create_spm_hrf(hrf_params=self.hrf, TR=self.TR, force=True)
            except:
                pass
        
        elif isinstance(self.hrf, str):
            self.hrf = "direct"  
            utils.verbose(f"Instantiate '{self.hrf}' HRF (fit={self.fit_hrf})", self.verbose)
        else:
            # try to read HRF parameters from settings
            try:
                hrf_pars = self.settings['hrf']['pars']
            except:
                hrf_pars = [1,1,0]

            utils.verbose(f"Instantiate HRF with: {hrf_pars} (fit={self.fit_hrf})", self.verbose)
            
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
                        utils.verbose(f"Setting '{setting}' to user-defined value: {getattr(self, setting)} (was: {self.settings[setting]})", self.verbose)

                        self.settings[setting] = getattr(self, setting)

    def fit(self):

        # check whether we got old parameters so we can skip Gaussian fit:
        if isinstance(self.old_params, (np.ndarray,str)):
            
            # agnostic read-in of existing parameter file; if pkl-file, settings will be overwritting to ensure consistency
            self.old_params = self.load_params(self.old_params, return_pars=True, skip_settings=self.skip_settings)

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
                self.previous_gaussian_fitter = self.gaussian_fitter

        else:
            utils.verbose(f"Gaussian fitter: {self.previous_gaussian_fitter}", self.verbose)

            # assume old parameters are grid parameters
            if self.model == "gauss" and "iter" in self.stage:

                # we need a separete generation of settings if we have 'old_params' with the HRF fitted already, otherwise we get 'AssertionError: Unequal bounds (7) and parameters (9)' because it tries to add another set of parameters for the HRF while they already exist

                if self.previous_gaussian_fitter.gridsearch_params.shape[-1] > 6:

                    # HRF already fitted, so set to false and update settings
                    utils.verbose(f"HRF already fitted, setting 'fit_hrf' in '{self.previous_gaussian_fitter}' to False", self.verbose)
                    self.previous_gaussian_fitter.fit_hrf = False

                GaussianModel.iterfit(self)
                self.previous_gaussian_fitter = self.gaussian_fitter

        #----------------------------------------------------------------------------------------------------------------------------------------------------------
        # Check if we should do DN-model
        if self.model.lower() != "gauss":

            ## Define settings/grids/fitter/bounds etcs
            self.define_settings(old_settings=self.settings)

            if self.fix_bold_baseline:
                self.settings['bounds']['bold_bsl'] = [0,0]

            # overwrite settings with custom settings
            self.update_settings()

            # initiate and do grid fit
            ExtendedModel.__init__(self)

            # if existing parameters already have more than the usual parameters given a model, we assume that the HRF has been fitted
            max_shape = self.get_max_parameter_shape(self.model)
            if self.previous_gaussian_fitter.gridsearch_params.shape[-1] > max_shape:

                # HRF already fitted, so set to false and update settings
                utils.verbose(f"HRF already fitted, setting 'fit_hrf' in '{self.tmp_fitter}' to False", self.verbose)
                self.tmp_fitter.fit_hrf = False

            ExtendedModel.gridfit(self)

            if "iter" in self.stage:
                ExtendedModel.iterfit(self)

    def get_max_parameter_shape(self, model):
        if model == "dog":
            max_shape = 8
        elif model == "css":
            max_shape = 7
        elif model in ["norm","abc","abd"]:
            max_shape = 10
        else:
            raise ValueError(f"Model must be one of {self.allowed_models}, not '{model}'")

        return max_shape
    
    def unitwise_bounds(
        self,
        idx_list,
        model=None,
        ref_pars=None
        ):

        if isinstance(idx_list, int):
            idx_list = [idx_list]
        
        if not isinstance(model, str):
            raise ValueError(f"Please specify a model from {self.allowed_models}")
        
        if not isinstance(ref_pars, np.ndarray):
            raise ValueError(f"Please specify an array representing parameters whose values need to be used to fix")
        
        new_bounds = []
        for b in range(ref_pars.shape[0]):
            tmp = np.array(self.fetch_bounds(model=model))
            for idx in idx_list:
                tmp[idx,:] = ref_pars[b,idx]
            new_bounds.append(tmp)

        return new_bounds
    
    def fetch_bounds(self, model=None):
        
        bounds = [
            tuple(self.settings['bounds']['x']),                # x
            tuple(self.settings['bounds']['y']),                # y
            tuple(self.settings['bounds']['size']),             # prf size
            tuple(self.settings['bounds']['prf_ampl']),         # prf amplitude
            tuple(self.settings['bounds']['bold_bsl'])          # bold baseline   
        ]
            
        if model == "norm":
            bounds += [
                tuple(self.settings['bounds']['surr_ampl']),    # surround amplitude
                tuple(self.settings['bounds']['surr_size']),    # surround size
                tuple(self.settings['bounds']['neur_bsl']),     # neural baseline
                tuple(self.settings['bounds']['surr_bsl'])      # surround baseline
            ]

        elif model == "css":
            bounds += [tuple(self.settings['bounds']['css_exponent'])]  # CSS exponent

        elif model == "dog":
            bounds += [
                tuple(self.settings['bounds']['surr_ampl']),    # surround amplitude
                tuple(self.settings['bounds']['surr_size'])     # surround size
            ]              

        if self.fit_hrf:
            bounds.append(tuple(self.settings["bounds"]['hrf_deriv']))      # HRF time derivative
            bounds.append(tuple(self.settings["bounds"]['hrf_disp']))       # HRF dispersion derivative

        return bounds
    
    def load_params(
        self, 
        params_file, 
        model='gauss', 
        stage='iter', 
        hemi=None,
        skip_settings=False,
        return_pars=False):

        """load_params

        Attribute of `self`, with which you can load in an existing file with pRF-estimates. If the input file is a `pkl`-file, you'll get access to all the required information about your analysis: the settings, the input data, the predictions, and the design matrix. To do this, we need the `params_file` as well as information about the origin of the file; which model (e.g., `gauss`) and stage (e.g., 'iter') did the file arise from. We'll then internally set the parameters to the specified model and stage, making it compatible with :func:`linescanning.prf.pRFmodelFitting.plot_vox()`. It can also be a `numpy.ndarray` or `pandas.DataFrame`, but then less information about the analysis is known. 

        Parameters
        ----------
        params_file: str, dict, pandas.DataFrame, numpy.ndarray
            Input to load in. You'll get the most out of it when you use a `pkl`-file created with :class:`linescanning.prf.pRFmodelFitting`, as you'll have access to the predictions, timecourses, parameters, and settings. If you do not have this, you can also enter a `numpy.ndarray` or `pandas.DataFrame`. The later is assumed to be a result from :class:`linescanning.prf.SizeResponse`, in that it's a *Divisive-Normalization* model result, with the following columns: `['x','y','prf_size','A','bold_bsl','B','C','surr_size','D','r2']` and indexed by `hemi` (`L`/`R`). Can also be a dictionary collecting filenames as values, and model names (e.g., 'gauss','norm') as keys. 
        model: str, list optional
            Model from which the pRF-estimates came from, by default 'gauss'
        stage: str, optional
            Stage from which the pRF-estimates came from, by default 'iter'
        hemi: str, optional
            In case `params_file` is a `pandas.DataFrame` indexed by `hemi` and you want a particular subset of the dataframe
        return_pars: bool, optional
            Instead of setting the parameters as attributes given `model` and `stage`, just return the parameters as output. This will still update the settings internally if the input was a pickle-file containing the **settings** key.

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

        if not isinstance(params_file, list):
            if isinstance(params_file, (str, np.ndarray, pd.DataFrame)):
                params_file = [params_file]
            elif isinstance(params_file, dict):
                model = list(params_file.keys())
                params_file = [params_file[i] for i in model]
            else:
                raise TypeError(f"Cannot deal with {params_file} of type {type(params_file)}")
        
        for ix,par_file in enumerate(params_file):
            
            mm = model
            if isinstance(par_file, str):

                # try to read model- from the filename
                if not isinstance(model, (list,str)):
                    try:
                        comps = utils.split_bids_components(par_file)
                    except:
                        comps = []
                    
                    if "model" in comps:
                        mm = comps["model"]
                else:
                    if isinstance(model, list):
                        mm = model[ix]
                    else:
                        mm = model

                if par_file.endswith('npy'):
                    params = np.load(par_file)
                elif par_file.endswith('pkl'):
                    with open(par_file, 'rb') as input:
                        data = pickle.load(input)
                        
                    params = data['pars']
                    for el in ["predictions","hrf"]:
                        if el in list(data.keys()):
                            setattr(self, f'{mm}_{stage}_predictions', data['predictions'])
                
                    if not skip_settings:
                        utils.verbose(f"Reading settings from '{par_file}' (safest option; overwrites other settings)", self.verbose)
                        self.settings = data['settings']

                        # print important settings
                        utils.verbose("\n---------------------------------------------------------------------------------------------------", self.verbose)
                        utils.verbose("Check these important updated settings!", self.verbose)
                        utils.verbose(f" Screen distance: {self.settings['screen_distance_cm']}cm", self.verbose)
                        utils.verbose(f" Screen size: {self.settings['screen_size_cm']}cm", self.verbose)
                        utils.verbose(f" TR: {self.settings['TR']}s", self.verbose)
                        utils.verbose("---------------------------------------------------------------------------------------------------\n", self.verbose)

            elif isinstance(par_file, np.ndarray):
                params = par_file.copy()
            elif isinstance(par_file, list):
                params = np.array(par_file)
            elif isinstance(par_file, pd.DataFrame):
                if hemi:
                    # got normalization parameter file
                    params = np.array((
                        par_file['x'][hemi],
                        par_file['y'][hemi],
                        par_file['prf_size'][hemi],
                        par_file['A'][hemi],
                        par_file['bold_bsl'][hemi],
                        par_file['C'][hemi],
                        par_file['surr_size'][hemi],
                        par_file['B'][hemi],
                        par_file['D'][hemi],
                        par_file['r2'][hemi]))
                else:
                    params = Parameters(par_file, model=mm).to_array()

            else:
                raise ValueError(f"Unrecognized input type for '{par_file}' ({type(par_file)})")

            if return_pars:
                return params
            else:
                utils.verbose(f"Inserting parameters from {type(par_file)} as '{mm}_{stage}' in {self}", self.verbose)
                setattr(self, f'{mm}_{stage}', params)

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
        n_time=5,
        add_tc={"tc": None},
        axs=None,
        normalize=False,
        figsize=(15,5),
        wratios=None,
        force_int=False,
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
        normalize: bool, optional
            Normalize the prediction to the max amplitude. Default = False

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

        if normalize:
            self.prediction /= self.prediction.max()
            
        prf_array = make_prf(
            self.prf_stim, 
            params,
            model=model,
            # size=params[2], 
            # mu_x=params[0], 
            # mu_y=params[1], 
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
                fig = plt.figure(constrained_layout=True, figsize=figsize)
                if freq_spectrum:
                    if not isinstance(wratios, list):
                        wratios = [10,20,10]

                    gs = fig.add_gridspec(1,3, width_ratios=wratios)
                else:
                    if not isinstance(wratios, list):
                        wratios = [10,20]

                    gs = fig.add_gridspec(1,2, width_ratios=wratios)

                ax1 = fig.add_subplot(gs[0])
                ax2 = fig.add_subplot(gs[1])
            else:
                ax1 = axs[0]
                ax2 = axs[1]

            # make plot 
            cross_col = "white"
            if model == "gauss":
                cross_col = "black"

            plotting.LazyPRF(
                prf_array, 
                cross_color=cross_col,
                vf_extent=self.settings['vf_extent'], 
                axs=ax1, 
                xkcd=xkcd,
                **kwargs)

            # make plot 
            if isinstance(title, str):
                if title == "pars":
                    title = str([round(ii,2) for ii in params])

            if axis_type == "time":
                x_label = "time (s)"
                x_axis = np.array(list(np.arange(0,self.prediction.shape[0])*self.TR))
            else:
                x_axis = np.array(list(np.arange(0,self.prediction.shape[0])))
                x_label = "volumes"
            
            if "x_label" in list(kwargs.keys()):
                x_label = kwargs["x_label"]
                kwargs.pop("x_label")
            
            # add additional timecourse
            if isinstance(add_tc, np.ndarray):
                add_tc = {"tc": add_tc}
            
            if isinstance(add_tc, dict):
                if isinstance(add_tc["tc"], np.ndarray):
                    default_dict = {
                        "color": "b",
                        "lw": 2,
                        "marker": None,
                        "label": "extra"}

                    for key in list(default_dict.keys()):
                        if key not in list(add_tc.keys()):
                            add_tc[key] = default_dict[key]

                    for ll,it in zip(
                        [data_list,color_list,label_list,lw_list,marker_list],
                        ["tc","color","label","lw","marker"]):

                        ll += [add_tc[it]]

            diff = x_axis[-1]-x_axis[0]
            step_size = diff/n_time

            x_ticks = np.linspace(
                x_axis[0],
                x_axis[-1], 
                num=n_time, 
                endpoint=True
            )

            if force_int:
                x_ticks = [int(round(i,0)) for i in x_ticks]

            ddict = {
                "x_ticks": x_ticks
            }

            for key,val in ddict.items():
                kwargs = utils.update_kwargs(
                    kwargs,
                    key,
                    val
                )

            plotting.LazyPlot(
                data_list,
                xx=x_axis,
                color=color_list, 
                labels=label_list, 
                add_hline=0,
                x_label=x_label,
                y_label="amplitude",
                axs=ax2,
                title=title,
                xkcd=xkcd,
                line_width=lw_list,
                markers=marker_list,
                # x_lim=[0,x_axis.shape[0]],
                **kwargs
            )

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
                fig.savefig(
                    save_as, 
                    dpi=300, 
                    bbox_inches="tight",
                    facecolor="white")

        return params, prf_array, tc, self.prediction

    def save_params(
        self, 
        model="gauss", 
        stage="grid",
        output_base=None,
        output_dir=None
        ):
        
        if hasattr(self, f"{model}_{stage}"):
            
            # set output stuff
            for flag,el in zip([output_base,output_dir],["output_base","output_dir"]):
                if isinstance(flag, str):
                    setattr(self, el)
                else:
                    if not hasattr(self, el):
                        raise ValueError(f"'{el}' is not set. Use the flags in 'save_params' or define in {self}")

            # define pickle
            pkl_file = opj(self.output_dir, f'{self.output_base}_model-{model}_stage-{stage}_desc-prf_params.pkl')

            # get parameters given model and stage
            params = getattr(self, f"{model}_{stage}")

            # write a pickle-file with relevant outputs
            out_dict = {}
            out_dict['pars'] = params
            out_dict['settings'] = self.settings
            # out_dict['predictions'] = self.make_predictions(model=model, stage=stage)

            utils.verbose(f"Save {stage}-fit parameters in {pkl_file}", self.verbose)
            with open(pkl_file, "wb") as f:
                pickle.dump(out_dict, f)

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

    utils.verbose(f"{x_par.shape} survived x-parameter matching", verbose)

    y_par,_ = utils.find_nearest(look_in_params[x_par,1], reference_prf[1], return_nr='all')
    xy_par = x_par[y_par]

    utils.verbose(f"{xy_par.shape} survived y-parameter matching", verbose)

    size_par,_ = utils.find_nearest(look_in_params[xy_par,2], reference_prf[2], return_nr='all')
    xysize_par = xy_par[size_par]

    utils.verbose(f"{xysize_par.shape} survived size-parameter matching", verbose)

    # filter indices with r2
    if r2_thresh != None:
        filt = look_in_params[xysize_par][...,-1] > r2_thresh
        true_idc = np.where(filt == True)
        xysize_par = xysize_par[true_idc]

        utils.verbose(f"{xysize_par.shape} survived r2>{r2_thresh}", verbose)

    if len(xysize_par) == 0:
        raise ValueError(f"Could not find similar pRFs. Maybe lower r2-threshold?")
        
    if return_nr == "all":
        return xysize_par
    else:
        return xysize_par[:return_nr]


class SizeResponse():
    """SizeResponse

    Perform size-response related operations given a pRF-stimulus/parameters. Simulate the pRF-response using a set of growing stimuli (or growing holes) using :func:`linescanning.prf.make_stims`, create size/hole response functions, and find stimulus sizes that maximize activation and suppression.

    Parameters
    ----------
    params: numpy.ndarray, str, pd.DataFrame
        Output of a Divisive Normalization fit 
    prf_stim: :class:`prfpy.stimulus.PRFStimulus2D`
        Object describing the nature of the stimulus
    subject_info: :class:`linescanning.utils.VertexInfo`-object
        Subject information collected in :class:`linescanning.utils.VertexInfo` that can be used for :func:`linescanning.prf.SizeResponse.save_target_params`
    model: str, optional
        Model from which `params` is derived. Default = "norm"
    subject_info: :class:`linescanning.prf.CollectSubject`, optional
        Object collection defaults given a subject. Mainly used to generate the `prf_stim` object, which is used to derive information about screen settings, which can also be set with the flags below. Therefore, not mandatory.
    downsample_factor: int, optional
        The default screen is [1920,1080]. This is rather large and can be complicated CPU wise. This flag downsamples the screen size by selecting every `downsample_factor` along the x and y axis
    n_pix: int, optional
        Number of pixels in the design matrix to simulate (smaller is faster)
    screen_distance_cm: int, optional
        Distance from viewer to screen. Default is the MRI-scanner value of 196 cm
    screen_size_cm: float, tuple, list, optional
        Dimensions of the screen in **centimeters**. Default is the BOLD screen: [70,39.3]. Specify a single value to create square stimuli
    screen_size_px: float, tuple, list, optional
        Dimensions of the screen in **pixels**. Default is the BOLD screen: [1920,1080]. Specify a single value to create square stimuli

    Example
    ----------
    >>> from linescanning import prf
    >>> # define file with pRF estimates
    >>> in_file = "sub-01_ses-1_task-2R_model-norm_stage-iter_desc-prf_params.pkl"
    >>> #
    >>> # read the input file into a dataframe
    >>> df_for_srfs = prf.Parameters(in_file, model="norm").to_df()
    >>> #
    >>> # initialize class
    >>> SR_ = prf.SizeResponse(
    >>>     params=df_for_srfs, 
    >>>     model="norm",
    >>>     screen_distance_cm=196,
    >>>     screen_size_cm=[70,39.3],
    >>>     screen_size_px=[1920,1080])

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
    
    def __init__(
        self, 
        prf_stim=None, 
        params=None, 
        model="norm", 
        subject_info=None,
        downsample_factor=6,
        n_pix=100,
        screen_distance_cm=196,
        screen_size_cm=[70,39.3],
        screen_size_px=[1920,1080],
        verbose=False):

        self.prf_stim = prf_stim
        self.params = params
        self.model = model
        self.subject_info = subject_info
        self.ds = downsample_factor
        self.n_pix = n_pix
        self.screen_distance_cm = screen_distance_cm
        self.screen_size_cm = screen_size_cm
        self.screen_size_px = screen_size_px
        self.verbose = verbose

        # overwrite info in CollectSubject object was specified
        if isinstance(self.subject_info, CollectSubject):
            if not isinstance(self.params, (np.ndarray,str,pd.DataFrame)):
                self.params = self.subject_info.pars.copy()
                self.prf_stim = self.subject_info.prf_stim
            
            self.n_pix = self.prf_stim.design_matrix.shape[0]
            self.screen_distance_cm = [self.prf_stim.screen_distance_cm,self.prf_stim.screen_distance_cm]
            self.screen_size_cm = self.prf_stim.screen_size_cm
            
            # overwrite verbose-flag from CollectSubject
            self.verbose = verbose

        if isinstance(self.params, str):
            self.params = read_par_file(self.params)

        if isinstance(self.params, np.ndarray):
            self.params_df = Parameters(self.params, model=self.model).to_df()
        elif isinstance(self.params, pd.DataFrame):
            self.params_df = self.params.copy()
        
        # parse potential floats into list
        for ii in ["screen_size_cm", "screen_size_px"]:
            el = getattr(self, ii)
            if isinstance(el, float):
                setattr(self, ii, [el,el])

        # define visual field in degree of visual angle
        self.ss_deg_x = 2*np.degrees(np.arctan(self.screen_size_cm[0]/(2.0*self.screen_distance_cm)))
        self.x = np.linspace(-self.ss_deg_x/2, self.ss_deg_x/2, self.screen_size_px[0])[::self.ds]

        self.ss_deg_y = 2*np.degrees(np.arctan(self.screen_size_cm[1]/(2.0*self.screen_distance_cm)))
        self.y = np.linspace(-self.ss_deg_y/2, self.ss_deg_y/2, self.screen_size_px[1])[::self.ds]
        self.dx = self.n_pix/len(self.x)

        # define visual extent:
        self.x_ext = [-self.ss_deg_x/2,self.ss_deg_x/2]
        self.y_ext = [-self.ss_deg_y/2,self.ss_deg_y/2]
        self.vf_extent = (self.x_ext,self.y_ext)
    
    def make_stimuli(
        self, 
        factor=4, 
        dt="fill", 
        *args, 
        **kwargs):

        """create stimuli for Size-Response curve simulation. See :func:`linescanning.prf.make_stims`"""
        # create stimuli
        
        stims, sizes = make_stims(
            (self.x,self.y), 
            factor=factor, 
            dt=dt,
            *args,
            **kwargs)
        
        return stims,sizes

    def find_pref_size(self, size=None, *args, **kwargs):
        if not hasattr(self, "stims_fill"):
            stims, sizes = self.make_stimuli(factor=2, *args, **kwargs)

        idx,_ = utils.find_nearest(sizes, size)
        return idx,sizes[idx]

    def make_sr_function(
        self, 
        params, 
        stims=None,
        center_prf=True,
        normalize=False):

        if not isinstance(params, (pd.DataFrame,pd.Series)):
            params = Parameters(params, model="norm").to_df()

        # get x,y,size parameters
        if center_prf:
            mu_x, mu_y = np.zeros((params.shape[0])),np.zeros((params.shape[0]))
        else:
            mu_x = params.x.values
            mu_y = params.y.values

        if mu_x.ndim == 0:
            mu_x = mu_x[...,np.newaxis]
        
        if mu_y.ndim == 0:
            mu_y = mu_y[...,np.newaxis]            

        sr = norm_2d_sr_function(
            self.dx**2*params.A.values, 
            params.B.values, 
            self.dx**2*params.C.values, 
            params.D.values, 
            params.prf_size.values, 
            params.surr_size.values, 
            self.x, 
            self.y, 
            stims, 
            mu_x=mu_x, 
            mu_y=mu_y)

        if normalize:
            sr /= sr.max()

        return sr

    def batch_sr_function(
        self,
        params=None,
        center_prf=True, 
        normalize=None, 
        stims=None,
        sizes=None,
        batch_size=100,
        parallel=True,
        thresh=0,
        max_jobs=20
        ):

        """create Size-Response function. If you want to ignore the actual location of the pRF, set `center_prf=True`. You can also scale the pRF-size with a factor `scale_factor`, for instance if you want to simulate pRF-sizes across depth."""
        
        if not isinstance(params, np.ndarray):
            params = Parameters(params, model="norm").to_df()
        
        # reset indices, but keep vertex IDs
        params = params.reset_index(drop=False)

        # get indices where prf size != 0
        idc_valid = list(params.loc[params.prf_size > thresh].index)
        utils.verbose(f"{len(idc_valid)}/{params.shape[0]} vertices > {thresh}", self.verbose)

        # filter out zeros
        df_filtered = params.loc[params.index[idc_valid]]
                
        # use batches
        if df_filtered.shape[0] > batch_size and center_prf:
            
            n_batches = int(np.ceil(df_filtered.shape[0]/batch_size))
            if parallel:
                if n_batches > max_jobs:
                    n_batches = max_jobs
                    batch_size = int(np.ceil(df_filtered.shape[0]/n_batches))

            utils.verbose(f"Split data in {n_batches} batches of {batch_size} vertices", self.verbose)

            # parse dataframe into list of batch dataframes for parallellization
            act = 0
            df_batches = []
            for batch in range(n_batches):
                interval = (batch*batch_size)+batch_size

                try:
                    df_batch = df_filtered.iloc[act:interval,:]
                except:
                    df_batch = df_filtered.iloc[act:,:]

                df_batches.append(df_batch)
                act += batch_size

            # loop through dfs
            if parallel:
                utils.verbose("Running parallel jobs", self.verbose)
                dd = Parallel(n_jobs=n_batches,verbose=True)(
                delayed(self.make_sr_function)(
                    batch, 
                    stims=stims,
                    center_prf=center_prf)
                    for batch in df_batches
                )
            else:
                utils.verbose("Running serial jobs", self.verbose)

                # use same stimulus sequence for all pRFs
                dd = []
                for ix,batch in enumerate(df_batches):
                    start = time.time()
                    tmp = self.make_sr_function(batch, stims=stims)
                    elapsed = (time.time() - start)
                    utils.verbose(f"batch #{ix}: {tmp.shape} | process took {timedelta(seconds=elapsed)}", self.verbose)
                    dd.append(tmp)

            # concatenate into single array
            func = np.concatenate(dd)

            if normalize == "max":
                func /= func.max(axis=0)

            n_stims = stims.shape[-1]
        else:
            # make single SR function
            if center_prf:
                func = self.make_sr_function(
                    df_filtered,
                    stims=stims,
                    center_prf=True)

                n_stims = stims.shape[-1]

            else:
                # custom stimuli for each pRF
                func = []
                utils.verbose(f"Creating unique stimulus (type='{stims}') set for each pRF", self.verbose)
                collect_stims = []
                for rf_ix in range(df_filtered.shape[0]):
                    
                    # get parameters
                    rf = pd.DataFrame(df_filtered.iloc[rf_ix]).T

                    # make stimulus
                    rf_stims,rf_sizes = self.make_stimuli(
                        factor=1, 
                        dt=stims, 
                        loc=(rf.x.values[0],rf.y.values[0]))

                    collect_stims.append(rf_stims)
                    n_stims = rf_stims.shape[-1]

                    # get srf
                    dd = self.make_sr_function(
                        rf, 
                        stims=rf_stims,
                        center_prf=False)
                    
                    func.append(dd)

                stims = rf_stims.copy()
                sizes = rf_sizes.copy()
                func = np.concatenate(func)

            if normalize == "max":
                func /= func.max()

        # initialize empty output array
        srf = np.zeros((params.shape[0], n_stims))
        srf[idc_valid,:] = func

        df_srf = pd.DataFrame(srf.T)
        df_srf["sizes"],df_srf["stim_nr"] = sizes,np.arange(0,len(sizes), dtype=int)
        
        return df_srf

    def find_stim_sizes(
        self, 
        curve1, 
        curve2=None, 
        t="max", 
        dt="fill",
        sizes=None,
        max_size=5,
        return_ampl=False):
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
        if isinstance(curve2,(np.ndarray,list)):

            # do some formatting stuff
            if isinstance(curve2, list):
                curve2 = np.array(curve2)
            
            if curve2.ndim > 1:
                curve2 = curve2.squeeze()

            if curve1.ndim > 1:
                curve1 = curve1.squeeze()

            if not isinstance(sizes, (list,np.ndarray)):
                raise ValueError(f"sizes must be a list or array consisting of stimulus sizes, not {sizes} of type ({type(sizes)})")
            
            # get intersection
            sr_diff = curve1-curve2
            size_indices = signal.find_peaks(abs(sr_diff))[0]

            # append sizes of max of curves
            use_stim_sizes = []
            for size_index in size_indices:
                use_stim_sizes.append(sizes[size_index])

            # # find intersection of curves
            y_size,_ = utils.find_intersection(sizes, curve1, curve2)
            use_stim_sizes.append(y_size[0][0])
            use_stim_sizes.sort()

            # find equidistance 
            eq1 = use_stim_sizes[0]+(use_stim_sizes[1]-use_stim_sizes[0])/2
            eq2 = use_stim_sizes[1]+(use_stim_sizes[2]-use_stim_sizes[1])/2

            use_stim_sizes += [eq1,eq2]
            use_stim_sizes.sort()
            
            return use_stim_sizes
        else:
            # find maximum
            if t == "max":
                ffunc = np.amax
            else:
                ffunc = np.amin

            max_idx = utils.find_nearest(sizes, max_size)[0]
            extr_val = ffunc(curve1[:max_idx])
            size_index = np.where(curve1[:max_idx]==extr_val)[0][0]
            
            if not isinstance(sizes, (list,np.ndarray)):
                sizes = getattr(self, f"stims_{dt}_sizes")
            
            if not return_ampl:
                return sizes[size_index]
            else:
                return sizes[size_index],extr_val

    def plot_stim_size(
        self, 
        stim, 
        vf_extent=([-5,5],[-5,5]), 
        ax=None, 
        clip=True, 
        cmap=(8,178,240),
        axis=False):
        
        """plot output of :func:`linescanning.prf.SizeResponse.find_stim_sizes`"""

        import matplotlib as mpl
        if not isinstance(ax, mpl.axes._axes.Axes):
            _,ax = plt.subplots(figsize=(6,6))        
        
        cmap_blue = utils.make_binary_cm(cmap)
        im = ax.imshow(np.flipud(stim), extent=vf_extent[0]+vf_extent[1], cmap=cmap_blue)

        ax.axvline(0, color='k', linestyle='dashed', lw=0.5)
        ax.axhline(0, color='k', linestyle='dashed', lw=0.5)

        if not axis:
            ax.axis('off')

        if clip:
            patch = patches.Circle((0,0), radius=vf_extent[0][-1], transform=ax.transData)
            im.set_clip_path(patch)

    def save_target_params(
        self, 
        fname=None, 
        hemi="L", 
        stim_sizes=None):
        """Write best_vertices-type file for normalization parameters + full normalization parameters in numpy-file"""
            
        if hemi == "lh":
            hemi = "L"
        elif hemi == "rh":
            hemi = "R"

        if self.subject_info != None:
            if fname == None:
                prf_bestvertex = opj(self.subject_info.cx_dir, f'{self.subject_info.subject}_model-norm_desc-best_vertices.csv')
            else:
                prf_bestvertex = fname

            # write existing pRF parameters to dictionary
            vert_info = self.subject_info.vert_info.data.copy()
            data_dict = {}
            data_dict["hemi"] = [hemi]
            for ii in list(self.params_df.keys()):
                data_dict[ii] = [self.params_df[ii]]

            # add custom stuff
            for ii in ["index","position","normal"]:
                data_dict[ii] = [vert_info[ii]]

            # add stim sizes
            if isinstance(stim_sizes, (int,float)):
                stim_sizes = [stim_sizes]

            if isinstance(stim_sizes, list):
                data_dict['stim_sizes'] = np.array(stim_sizes)
                            
            # append to existing file
            if os.path.exists(prf_bestvertex):

                # can index an indexed dataframe
                try:
                    df = pd.read_csv(prf_bestvertex).set_index(["hemi"])
                except:
                    df = pd.read_csv(prf_bestvertex)
                
                if not "stim_sizes" in list(df.columns):
                    df["stim_sizes"] = None

                df["stim_sizes"][hemi] = data_dict['stim_sizes']
            else:
                df = pd.DataFrame(data_dict)

            # can index an indexed dataframe
            try:
                df = df.set_index(["hemi"])
            except:
                pass

            df.to_csv(prf_bestvertex)
        
        # # save full parameters as well
        # pars_file = opj(self.subject_info.cx_dir, f'{self.subject_info.subject}_model-norm_desc-params.csv') 
        # self.params_df['hemi'] = hemi
        # if os.path.exists(pars_file):
        #     tmp = pd.read_csv(pars_file).reset_index()

        #     if hemi in tmp['hemi'].values:
        #         tmp = tmp[tmp.hemi != hemi]

        #     self.params_df = pd.concat((self.params_df, tmp))

        # self.params_df.set_index('hemi').to_csv(pars_file)

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
        v1=False,
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
        self.v1_data        = v1
        self.__dict__.update(kwargs)

        if self.hemi.lower() in ["lh","l","left"]:
            self.hemi_tag = "L"
        elif self.hemi.lower() in ["rh","r","right"]:
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

            search_for = ["avg_bold", ".npy"]
            if self.v1_data:
                search_for += ["_roi-V1"]
                exclude = None
            else:
                exclude = "_roi-V1"
            
            try:
                for item,tag in zip(
                    ["_lr","_l","_r"],
                    ["hemi-LR_","hemi-L_","hemi-R_"]):

                    data = np.load(
                        utils.get_file_from_substring(
                        search_for+[tag], 
                        self.prf_dir, 
                        exclude=exclude))
                    
                    setattr(self, f"func_data{item}", data)
                    
            except:
                try:
                    self.func_data_lr = np.load(utils.get_file_from_substring(["desc-data.npy"]+self.filter_list, self.prf_dir))
                except:
                    print(f"WARNING: could not load all functional data from '{self.prf_dir}'")

        # try to read iterative fit parameters
        allowed_models = ['gauss', 'css', 'dog', 'norm', 'abc', 'abd']
        look_for = ["stage-iter", "params.pkl"]
        
        # add some filters
        if self.v1_data:
            look_for += ["_roi-V1"]
            exclude = None
        else:
            exclude = "_roi-V1"

        if self.fit_hrf:
            look_for += ["hrf-true_"]

        utils.verbose(f"Reading full-cortex pRF estimates with {look_for}", self.verbose)
        for model in allowed_models:
            par_file = utils.get_file_from_substring(look_for+[f"model-{model}"], self.prf_dir, return_msg=None, exclude=exclude)
            if isinstance(par_file, str):
                utils.verbose(f" model: {model}:\t{par_file}", self.verbose)
                setattr(self, f'{model}_iter_pars_file', par_file)
                setattr(self, f'{model}_iter_pars', read_par_file(par_file))
                setattr(self, f'{model}_iter_pars_df', Parameters(getattr(self, f'{model}_iter_pars'), model=model).to_df())
                setattr(self, f'{model}_iter_pars_arr', Parameters(getattr(self, f'{model}_iter_pars_df'), model=model).to_array())
        
        # set pycortex directory
        if self.cx_dir == None:
            if derivatives != None:
                self.cx_dir = opj(self.derivatives, 'pycortex', self.subject)
            
        if self.cx_dir != None and os.path.exists(self.cx_dir):
            self.vert_fn = utils.get_file_from_substring([self.model, "best_vertices.csv"], self.cx_dir, return_msg=None)

            if isinstance(self.vert_fn, str):
                utils.verbose(f"Reading {self.vert_fn}", self.verbose)
                self.vert_info = utils.VertexInfo(
                    self.vert_fn, 
                    subject=self.subject, 
                    hemi=self.hemi)
        
        # fetch target vertex parameters
        if hasattr(self, "vert_info"):
            self.target_vertex = self.return_target_vertex(hemi=self.hemi)
            utils.verbose(f"Target vertex: {self.target_vertex}", self.verbose)

            self.target_params = getattr(self, f'{model}_iter_pars_arr')[self.target_vertex,:]

        # find the csv file with parameters (generally available with 'norm')
        if self.best_vertex:
            try:
                self.params_fn = utils.get_file_from_substring([f"model-{self.model}", "desc-params.csv"], self.cx_dir)
                self.normalization_params_df = pd.read_csv(self.params_fn, index_col=0)            
                self.pars = self.normalization_params_df.copy()
            except:
                self.pars = np.array(self.target_params)[np.newaxis,...]
            
            tmp_data = getattr(self, f"func_data_{self.hemi_tag.lower()}").copy()
            self.data = tmp_data[...,self.target_vertex][np.newaxis,...]
            txt = "target vertex"
        else:
            if hasattr(self, f"{self.model}_iter_pars"):
                self.pars = getattr(self, f"{self.model}_iter_pars")
            else:
                print(f"WARNING: Could not find '{self.model}_iter_pars' attribute")

            if hasattr(self, "func_data_lr"):
                self.data = self.func_data_lr.T

            txt = "full cortex"

        if self.verbose:
            print(f"Shape final parameters: {self.pars.shape} [{txt}]")

        # initialize the model
        if hasattr(self, "data"):
            super().__init__(
                self.data,
                design_matrix=self.design_matrix,
                verbose=self.verbose,
                model=self.model
            )

            if hasattr(self, "pars"):
                self.load_params(
                    self.pars, 
                    model=self.model, 
                    stage="iter", 
                    hemi=self.hemi_tag,
                    skip_settings=self.skip_settings
                )
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

class Parameters():

    def __init__(
        self,
        params,
        model="gauss"):

        self.params = params
        self.model = model
        self.allow_models = ["gauss","dog","css","norm",'abc','abd']

        if isinstance(self.params, str):
            self.params = read_par_file(self.params)
            
    def to_df(self):
        
        if isinstance(self.params, pd.DataFrame):
            return self.params
        
        if not isinstance(self.params, np.ndarray):
            raise ValueError(f"Input must be np.ndarray, not '{type(self.params)}'")

        if self.params.ndim == 1:
            self.params = self.params[np.newaxis,:]

        # see: https://github.com/VU-Cog-Sci/prfpy_tools/blob/master/utils/postproc_utils.py#L377
        if self.model in self.allow_models:
            if self.model == "gauss":
                params_dict = {
                    "x": self.params[:,0], 
                    "y": self.params[:,1], 
                    "prf_size": self.params[:,2],
                    "prf_ampl": self.params[:,3],
                    "bold_bsl": self.params[:,4],
                    "r2": self.params[:,-1],
                    "ecc": np.sqrt(self.params[:,0]**2+self.params[:,1]**2),
                    "polar": np.angle(self.params[:,0]+self.params[:,1]*1j)
                }

                if self.params.shape[-1] > 6:
                    params_dict["hrf_deriv"] = self.params[:,-3]
                    params_dict["hrf_disp"] = self.params[:,-2]

            elif self.model in ["norm","abc","abd"]:
                    
                params_dict = {
                    "x": self.params[:,0], 
                    "y": self.params[:,1], 
                    "prf_size": self.params[:,2],
                    "prf_ampl": self.params[:,3],
                    "bold_bsl": self.params[:,4],
                    "surr_ampl": self.params[:,5],
                    "surr_size": self.params[:,6], 
                    "neur_bsl": self.params[:,7],
                    "surr_bsl": self.params[:,8],
                    "A": self.params[:,3], 
                    "B": self.params[:,7], #/params[:,3], 
                    "C": self.params[:,5], 
                    "D": self.params[:,8],
                    "ratio (B/D)": self.params[:,7]/self.params[:,8],
                    "r2": self.params[:,-1],
                    "size ratio": self.params[:,6]/self.params[:,2],
                    "suppression index": (self.params[:,5]*self.params[:,6]**2)/(self.params[:,3]*self.params[:,2]**2),
                    "ecc": np.sqrt(self.params[:,0]**2+self.params[:,1]**2),
                    "polar": np.angle(self.params[:,0]+self.params[:,1]*1j)}

                if self.params.shape[-1] > 10:
                    params_dict["hrf_deriv"] = self.params[:,-3]
                    params_dict["hrf_dsip"] = self.params[:,-2]

            elif self.model == "dog":
                params_dict = {
                    "x": self.params[:,0], 
                    "y": self.params[:,1], 
                    "prf_size": self.params[:,2],
                    "prf_ampl": self.params[:,3],
                    "bold_bsl": self.params[:,4],
                    "surr_ampl": self.params[:,5],
                    "surr_size": self.params[:,6], 
                    "r2": self.params[:,-1],
                    "size ratio": self.params[:,6]/self.params[:,2],
                    "suppression index": (self.params[:,5]*self.params[:,6]**2)/(self.params[:,3]*self.params[:,2]**2),
                    "ecc": np.sqrt(self.params[:,0]**2+self.params[:,1]**2),
                    "polar": np.angle(self.params[:,0]+self.params[:,1]*1j)}

                
                if self.params.shape[-1] > 8:
                    params_dict["hrf_deriv"] = self.params[:,-3]
                    params_dict["hrf_dsip"] = self.params[:,-2]

            elif self.model == "css":
                params_dict = {
                    "x": self.params[:,0], 
                    "y": self.params[:,1], 
                    "prf_size": self.params[:,2],
                    "prf_ampl": self.params[:,3],
                    "bold_bsl": self.params[:,4],
                    "css_exp": self.params[:,5],
                    "r2": self.params[:,-1],
                    "ecc": np.sqrt(self.params[:,0]**2+self.params[:,1]**2),
                    "polar": np.angle(self.params[:,0]+self.params[:,1]*1j)}     
                
                if self.params.shape[-1] > 7:
                    params_dict["hrf_deriv"] = self.params[:,-3]
                    params_dict["hrf_dsip"] = self.params[:,-2]

        else:
            raise ValueError(f"Model must be one of {self.allow_models}. Not '{self.model}'")

        return pd.DataFrame(params_dict)

    def to_array(self):
        
        if not isinstance(self.params, pd.DataFrame):
            raise ValueError(f"Input must be pd.DataFrame, not '{type(self.params)}'")

        if self.model == "gauss":
            item_list = ["x","y","prf_size","prf_ampl","bold_bsl","hrf_deriv","hrf_disp","r2"]
        elif self.model in ["norm","abc","abd"]:
            item_list = ["x","y","prf_size","prf_ampl","bold_bsl","surr_ampl","surr_size","neur_bsl","surr_bsl","hrf_deriv","hrf_disp","r2"]
        elif self.model == "dog":
            item_list = ["x","y","prf_size","prf_ampl","bold_bsl","surr_ampl","surr_size","hrf_deriv","hrf_disp","r2"]
        elif self.model == "css":
            item_list = ["x","y","prf_size","prf_ampl","bold_bsl","css_exp","hrf_deriv","hrf_disp","r2"]
        else:
            raise ValueError(f"Model must be one of 'gauss','norm','dog','css'; not '{self.model}'")

        self.parr = []

        # parallel counter in case HRF-parameters are not present
        ct = 0
        for ii in item_list:
            if ii in list(self.params.columns):
                pars = self.params[ii].values
                if not np.isnan(pars).all():
                    self.parr.append(pars[...,np.newaxis])
                    ct += 1
            
        return np.concatenate(self.parr,axis=1)

def create_model_rf_wrapper(model,stim,params,normalize_RFs=False):
    prf = params[3][...,np.newaxis,np.newaxis]*np.rot90(
        rf.gauss2D_iso_cart(
            x=stim.x_coordinates[...,np.newaxis],
            y=stim.y_coordinates[...,np.newaxis],
            mu=(params[0], params[1]),
            sigma=params[2],
            normalize_RFs=normalize_RFs).T, axes=(1,2))

    if model == 'css':
        prf **= params[5][...,np.newaxis,np.newaxis]

    elif model == 'dog':
        prf -= params[5][...,np.newaxis,np.newaxis]*np.rot90(
            rf.gauss2D_iso_cart(
                x=stim.x_coordinates[...,np.newaxis],
                y=stim.y_coordinates[...,np.newaxis],
                mu=(params[0], params[1]),
                sigma=params[6],
                normalize_RFs=normalize_RFs).T, axes=(1,2))

    elif model in ["norm","abc","abd"]:
        prf += params[7][...,np.newaxis,np.newaxis]
        prf /= (params[5][...,np.newaxis,np.newaxis]*np.rot90(
            rf.gauss2D_iso_cart(
                x=stim.x_coordinates[...,np.newaxis],
                y=stim.y_coordinates[...,np.newaxis],
                mu=(params[0], params[1]),
                sigma=params[6],
                normalize_RFs=normalize_RFs).T, 
            axes=(1,2)) + params[8][...,np.newaxis,np.newaxis])

        prf -= (params[7]/params[8])[...,np.newaxis,np.newaxis]

    return prf

class FormatTimeCourses():

    def __init__(
        self, 
        data, 
        *args, 
        **kwargs
        ):

        # format
        self.data = data
        self.formatted_data, self.n_verts = self.format_hemi_data(
            self.data,
            *args,
            **kwargs
        )
    
    def return_data(self):
        return self.formatted_data.copy()
    
    @classmethod
    def average_iterations(
        self, 
        data, 
        n_folds=None, 
        debug=False
        ):

        if not isinstance(n_folds, int):
            raise TypeError(f"folds-argument must be integer, not {n_folds} of type {type(n_folds)}")
        
        n_vols,n_vertices = data.shape
        iter_size = n_vols//n_folds
        
        chunk_list = []
        start = 0
        for i in range(n_folds):

            # try to fetch values, if steps are out of bounds, zero-pad the timecourse
            end_point = start+iter_size
            utils.verbose(f"chunk #{i+1}\t{start}-{end_point}", debug)
            if end_point <= n_vols:
                chunk = data[start:end_point]
            else:
                raise ValueError(f"Data with {n_vertices} vertices cannot be split evenly in chunks of {iter_size}. Use '--cut_vols <n_vols>' to remove volumes at the beginning to align the data")

            chunk_list.append(chunk[...,np.newaxis])
            start += iter_size

        chunked_array = np.concatenate(chunk_list, axis=-1).mean(axis=-1)
        return chunked_array

    def get_verts(self):
        return self.n_verts
    
    @classmethod
    def format_hemi_data(
        self,
        pair,
        gifti=False,
        psc=False,
        n_folds=None,
        dm=None,
        *args,
        **kwargs
        ):

        # different function for giftis
        if gifti:
            hemi_data = [dataset.ParseGiftiFile(pair[ix]).data for ix in range(len(pair))]
        else:
            hemi_data = [np.load(pair[ix]) for ix in range(len(pair))]

        n_verts = [ii.shape[-1] for ii in hemi_data]

        # stack into array (time,voxels)
        hemi_data = np.hstack(hemi_data)

        # cut vols?
        if "cut_vols" in list(kwargs.keys()):
            # print(f"cutting {kwargs['cut_vols']} away")
            hemi_data = hemi_data[kwargs["cut_vols"]:,:]
            kwargs.pop("cut_vols")

        # check if we got multiple repeats within runs
        if isinstance(n_folds, int):
            hemi_data = self.average_iterations(
                hemi_data, 
                n_folds=n_folds, 
                *args, 
                **kwargs
            )

        # do percent change following marco's method
        if psc:

            if not isinstance(dm, (str,np.ndarray,dict)):
                raise ValueError(f"Please specify a path representing the design matrix, or a numpy array, not {dm} of type {type(dm)}")
            
            if hemi_data.shape[0] != dm.shape[-1]:
                diff = dm.shape[-1]-hemi_data.shape[0]
                # print(f"cutting {diff} from design")
                dm = dm[...,diff:]

            hemi_data = utils.percent_change(
                hemi_data,
                0,
                prf=True,
                dm=dm
            )

        return hemi_data,n_verts

class Profile1D(pRFmodelFitting):

    def __init__(
        self,
        pars,
        n_pix=100,
        center=True,
        plot_kws={},
        metrics_kws={},
        dm_kws={},
        model="norm",
        first_cross=True,
        **kwargs
        ):

        self.pars = pars
        self.n_pix = n_pix
        self.center = center
        self.plot_kws = plot_kws
        self.metrics_kws = metrics_kws
        self.dm_kws = dm_kws
        self.model = model
        self.first_cross = first_cross

        if isinstance(self.pars, pd.DataFrame):
            self.pars = Parameters(self.pars, model=self.model).to_array().squeeze()

        if self.center:
            self.pars[:2] = 0

        self.dm = self.create_dm(
            n_pix=self.n_pix,
            center=self.center,
            pars=self.pars,
            **self.dm_kws
        )

        super().__init__(
            None,
            design_matrix=self.dm,
            TR=1,
            hrf="direct",
            model=self.model,
            **kwargs
        )

        # skip plot by default
        ddict = {
            "make_figure": False,
            "axis_type": "volumes"
        }

        for key,val in ddict.items():
            plot_kws = utils.update_kwargs(
                plot_kws,
                key,
                val
            )

        self.load_params(self.pars, model=self.model)
        _,self.prf_o,_,self.prof_1d = self.plot_vox(
            vox_nr=0, 
            model=self.model, 
            resize_pix=self.n_pix,
            **plot_kws
        )

        self.prof_1d = fitting.Epoch.correct_baseline(self.prof_1d, bsl=1)

        # find fwhm in pixels
        self.metrics = fitting.HRFMetrics(
            self.prof_1d, 
            TR=self.TR,
            **self.metrics_kws
        ).return_metrics()

        self.fwhm = self.metrics.iloc[0].fwhm

        self.fwhm_deg = pix2deg(
            self.fwhm,
            scrSizePix=(self.n_pix,self.n_pix),
            scrWidthCm=self.settings["screen_size_cm"],
            scrDist=self.settings["screen_distance_cm"]
        )
        
        # try to find zero crossings
        try:
            self.zero_cross = self.find_crossings(self.prof_1d)
        except:
            print(f'Could not find zero-crossings of 1d profile..')

        if hasattr(self, "zero_cross"):

            # sometimes the outer borders of the image are counted as crossing; this flag dictates to only take the first crossings
            self.zero_cross_pix = [i[0][0] for i in self.zero_cross]
            if self.first_cross:
                if len(self.zero_cross_pix)>2:
                    self.zero_cross_pix = self.zero_cross_pix[1:3]
                    
            self.zero_diff = np.diff(sorted(self.zero_cross_pix))[0]
            self.zero_deg = pix2deg(
                self.zero_diff,
                scrSizePix=(self.n_pix,self.n_pix),
                scrWidthCm=self.settings["screen_size_cm"],
                scrDist=self.settings["screen_distance_cm"]
            )
            
    @classmethod
    def create_dm(
        self, 
        n_pix=100,
        center=True,
        pars=None,
        **kwargs
        ):

        # create impulse design matrix (single pixel traversing left-to-right)
        if center:
            y_pos = n_pix//2
        else:
            y_pos = int(deg2pix(
                pars[1],
                scrSizePix=[n_pix,n_pix],
                **kwargs
            ))

            y_pos += n_pix//2
            # raise NotImplementedError(f"Still need to implement translating a position in DVA to pixel")

        tiny_dm = np.zeros((n_pix,n_pix,n_pix))
        for ii in range(tiny_dm.shape[-1]):
            tiny_dm[y_pos-1,ii,ii] = 1

        return tiny_dm

    def plot_dm(self, **kwargs):

        for key,val in zip(["interval","n_cols","figsize"],[2,10,(16,8)]):
            kwargs = utils.update_kwargs(
                kwargs,
                key,
                val
            )

        plot_stims(
            self.dm, 
            **kwargs
        )

    @classmethod
    def find_crossings(self, curve):
        return utils.find_intersection(
            np.arange(0,curve.shape[0]), 
            curve, 
            np.zeros_like(curve)
        )
