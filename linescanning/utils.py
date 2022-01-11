import cortex
import csv
import json
from . import prf
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nibabel as nb
from nilearn.signal import clean
from nilearn.glm.first_level.design_matrix import _cosine_drift as dct_set
from nitime.timeseries import TimeSeries
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer, NormalizationAnalyzer
import numpy as np
import os
from numpy.lib.arraysetops import isin
import pandas as pd
from PIL import ImageColor
from prfpy import gauss2D_iso_cart
from prfpy import stimulus
import random
from re import I
from scipy import io
from scipy.signal import detrend
import seaborn as sns
from shapely import geometry
import subprocess
import warnings
import yaml

opj = os.path.join
pd.options.mode.chained_assignment = None # disable warning thrown by string2float
warnings.filterwarnings("ignore")

def copy_hdr(source_img,dest_img):
    """copy_hdr

    Similar functionality as fslcpgeom but than more rigorious using Nibabel. Copies the ENTIRE header, including affine, quaternion rotations, and dimensions.

    Parameters
    ----------
    source_img: str, nibabel.Nifti1Image
        source image from which to derive the header information
    dest_img: str, nibabel.Nifti1Image
        destination image to which to copy the header from <source image> to

    Returns
    ----------
    nibabel.Nifti1Image
        `source_img` with updated header information

    Example
    ----------
    >>> new_img = copy_hdr(img1,img2)
    """

    if isinstance(source_img, nb.Nifti1Image):
        src_img = source_img
    elif isinstance(source_img, str):
        src_img = nb.load(source_img)

    if isinstance(dest_img, nb.Nifti1Image):
        targ_img = dest_img
    elif isinstance(dest_img, str):
        targ_img = nb.load(dest_img)

    new = nb.Nifti1Image(targ_img.get_fdata(), affine=src_img.affine, header=src_img.header)
    return new

class color:
    # """color
    
    # Add some color to the terminal.

    # Example
    # ----------
    # >>> print("set orientation to " + utils.color.BOLD + utils.color.RED + "SOME TEXT THAT'LL BE IN TED" + utils.color.END)
    # """
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def find_nearest(array, value, return_nr=1):
    """find_nearest

    Find the index and value in an array given a value. You can either choose to have 1 item (the `closest`) returned, or the 5 nearest items (`return_nr=5`), or everything you're interested in (`return_nr="all"`)

    Parameters
    ----------
    array: numpy.ndarray
        array to search in
    value: float
        value to search for in `array`
    return_nr: int, str, optional
        number of elements to return after searching for elements in `array` that are close to `value`. Can either be an integer or a string *all*

    Returns
    ----------
    int
        integer representing the index of the element in `array` closest to `value`. 

    list
        if `return_nr` > 1, a list of indices will be returned

    numpy.ndarray
        value in `array` at the index closest to `value`
    """
    
    array = np.asarray(array)

    if return_nr == 1:
        idx = (np.abs(array-value)).argmin()
        return idx, array[idx]
    else:
        try:
            idx = (np.abs(array-value))

            if return_nr == "all":
                idc_list = np.sort(np.where(idx == 0.0)[0])
            else:
                idc_list = np.sort(np.where(idx == 0.0)[0])[:return_nr]
            return idc_list, array[idc_list]

        except Exception:
            print("Could not perform this operation")


def replace_string(fn, str1, str2, fn_sep='_'):
    """replace_string

    Replace a string with another string given a filename

    Parameters
    ----------
    fn: str
        filename in which we need to replace something
    str1: str
        string-to-be-replaced
    str2: str
        string-to-replace-str1-with
    fn_sep: str
        what type of element can we use to split the filename into chunks that we can replace

    Returns
    ----------
    str
        filename with replaced substring
    """

    split_name = fn.split(os.sep)[-1].split(fn_sep)
    idx = [(i, split_name.index(str1))
           for i, split_name in enumerate(split_name) if str1 in split_name][0][0]
    split_name[idx] = split_name[idx].replace(split_name[idx], str2)
    new_filename = fn_sep.join(split_name)
    new_filename = opj(os.path.dirname(fn), new_filename)

    return new_filename


def convert2unit(v, method="np"):
    """convert vector to unit vector"""
    import numpy as np

    if method.lower() == "np":
        v_hat = v / np.linalg.norm(v)
        return v_hat
    elif method.lower() == "mesh":
        # https://sites.google.com/site/dlampetest/python/calculating-normals-of-a-triangle-mesh-using-numpy
        lens = np.sqrt( v[:,0]**2 + v[:,1]**2 + v[:,2]**2 )
        v[:,0] /= lens
        v[:,1] /= lens
        v[:,2] /= lens
        return v

def string2float(string_array):
    """string2float

    This function converts a array in string representation to a regular float array. This can happen, for instance, when you've stored a numpy array in a pandas dataframe (such is the case with the 'normal' vector). It starts by splitting based on empty spaces, filter these, and convert any remaining elements to floats and returns these in an array.

    Parameters
    ----------
    string_array: str
        string to be converted to a valid numpy array with float values

    Returns
    ----------
    numpy.ndarray
        array containing elements in float rather than in string representation

    Example
    ----------
    >>> string2float('[ -7.42 -92.97 -15.28]')
    array([ -7.42, -92.97, -15.28])
    """

    if type(string_array) == str:
        new = string_array[1:-1].split(' ')[0:]
        new = list(filter(None, new))
        new = [float(i) for i in new]
        new = np.array(new)

        return new

    else:
        # array is already in non-string format
        return string_array


def get_module_nr(key_word):
    """get_module_nr

    Fetches the module number from the master script given an input string. It sends a command using sed and grep to the bash command line. Won't work on windows! See `call_bashhelper` for more information (that version is actually more accurate as it allows additions to the `master` usage, while that's hardcoded in this one..)

    Parameters
    ----------
    key_word: str
        search string of the module your interested in. Should at least match otherwise the function will not find anything. For instance, if we want to know which module the creation of the sinus mask is, we can do:

    Example
    ----------
    >>> get_module_nr('sinus')
    '12'
    """

    cmd = "sed -n \'50,85p\' {master} | grep -A0 \"{key}\" | grep -Eo \"[0-9]{{1,2}}\" | head -n 1".format(
        master=opj(os.environ['DIR_SCRIPTS'], 'shell', 'master'), key=key_word)
    # print(cmd)
    mod = subprocess.getoutput(cmd)

    return mod


def bids_fullfile(bids_image):
    """get full path to a BIDS-image filename"""

    fullfile = opj(bids_image.dirname, bids_image.filename)
    return fullfile


def ctx_config():
    """fetch path to `Pycortex`'s configuration file"""

    f = cortex.options.usercfg
    return f


def decode(obj):
    """decode an object"""
    if isinstance(obj, bytes):
        obj = obj.decode()
    return obj


def reverse_sign(x):
    """reverse_sign

    Inverts the sign given set of values. Can be either one value or an array of values that need to be inverted

    Parameters
    ----------
    x: int,float,list,numpy.ndarray
        input that needs inverting, either one value or a list
    
    Returns
    ----------
    the inverse of whatever the input `x` was

    Example
    ----------
    >>> # input is integer
    >>> x = 5
    >>> reverse_sign(x)
    -5
    >>> # input is array
    >>> x = np.array([2, -2340, 2345,123342, 123])
    >>> In [6]: reverse_sign(x)
    array([-2.00000e+00,  2.34000e+03, -2.34500e+03, -1.23342e+05,-1.23000e+02])
    >>> # input is float
    >>> x = 5.0
    >>> reverse_sign(x)
    -5.0
    """

    import numpy as np

    inverted = ()

    if isinstance(x, int) or isinstance(x, float) or isinstance(x, np.float32):
        if x > 0:
            inverted = -x
        else:
            inverted = abs(x)
    elif isinstance(x, np.ndarray):
        for i in x:
            if float(i) > 0:
                val = -float(i)
            else:
                val = abs(float(i))

            inverted = np.append(inverted, val)

    return inverted


def remove_files(path, string, ext=False):
    """remove_files

    Remove files from a given path that containg a string as extension (`ext=True`), or at the
    start of the file (`ext=False`)

    Parameters
    ----------
    path: str
        path to the directory from which we need to remove files
    string: str
        tag for files we need to remove
    ext: str, optional
        only remove files containing `string` that end with `ext`
    """

    files_in_directory = os.listdir(path)

    if ext:
        filtered_files = [file for file in files_in_directory if file.endswith(string)]
    else:
        filtered_files = [file for file in files_in_directory if file.startswith(string)]

    for file in filtered_files:
        path_to_file = os.path.join(path, file)
        os.remove(path_to_file)


def get_file_from_substring(filt, path):
    """get_file_from_substring

    This function returns the file given a path and a substring. Avoids annoying stuff with glob. Now also allows multiple filters to be applied to the list of files in the directory. The idea here is to construct a binary matrix of shape (files_in_directory, nr_of_filters), and test for each filter if it exists in the filename. If all filters are present in a file, then the entire row should be 1. This is what we'll be looking for. If multiple files are found in this manner, a list of paths is returned. If only 1 file was found, the string representing the filepath will be returned. 

    Parameters
    ----------
    filt: str, list
        tag for files we need to select. Now also support a list of multiple filters. 
    path: str
        path to the directory from which we need to remove files
    
    Returns
    ----------
    str
        path to the files containing `string`. If no files could be found, `None` is returned

    list
        list of paths if multiple files were found

    Raises
    ----------
    ValueError
        If no files usingn the specified filters could be found

    Example
    ----------
    >>> get_file_from_substring("R2", "/path/to/prf")
    '/path/to/prf/r2.npy'
    >>> get_file_from_substring(['gauss', 'best_vertices'], "path/to/pycortex/sub-xxx")
    '/path/to/pycortex/sub-xxx/sub-xxx_model-gauss_desc-best_vertices.csv'
    >>> get_file_from_substring(['best_vertices'], "path/to/pycortex/sub-xxx")
    ['/path/to/pycortex/sub-xxx/sub-xxx_model-gauss_desc-best_vertices.csv',
    '/path/to/pycortex/sub-xxx/sub-xxx_model-norm_desc-best_vertices.csv']    
    """
    
    if isinstance(filt, str):
        filt = [filt]

    if isinstance(filt, list):
        # list and sort all files in the directory
        files_in_directory = sorted(os.listdir(path))

        # the idea is to create a binary matrix for the files in 'path', loop through the filters, and find the row where all values are 1
        filt_array = np.zeros((len(files_in_directory), len(filt)))
        for ix,f in enumerate(files_in_directory):
            for filt_ix,filt_opt in enumerate(filt):
                filt_array[ix,filt_ix] = filt_opt in f

        # now we have a binary <number of files x number of filters> array. If all filters were available in a file, the entire row should be 1, 
        # so we're going to look for those rows
        full_match = np.ones(len(filt))
        full_match_idc = np.where(np.all(filt_array==full_match,axis=1))[0]
        
        if len(full_match_idc) == 1:
            fname = files_in_directory[full_match_idc[0]]
            return opj(path, fname)
        elif len(full_match_idc) > 1:
            match_list = []
            for match in full_match_idc:
                fname = files_in_directory[match]
                match_list.append(opj(path, fname))
            return match_list
        else:
            raise ValueError(f"Could not find file with filters: {filt}")

def get_bids_file(layout, filter=None):
    """get_bids_file

    This search function is more tailored for BIDSified data, and requires a list of BIDS-filenames as per output for `l = BIDSLayout(dir, validate=False)` & `fn = l.get(session='1', datatype='anat')` for instance. From this list the script will look the list of given filters.

    Parameters
    ----------
    layout: :abbr:`BIDS (Brain Imaging Data Structure)` layout object
        BIDS-layout object obtained with `BIDSLayout`
    filter: str, optional
        filter for particular strings

    Returns
    ----------
    str
        filenames meeting the specifications (i.e., existing in `layout` and containing strings specified in `filters`)

    Example
    ----------
    >>> layout = BIDSLayout(somedir).get(session='1', datatype='anat')
    >>> fn = get_bids_file(layout, filter=['str1', 'str2', 'str3'])
    """

    import warnings
    warnings.filterwarnings("ignore")

    l = []
    for i in layout:
        if all(f in i for f in filter) == True:
            l.append(i)

    if len(l) == 1:
        return l[0]
    else:
        return l


def get_matrixfromants(mat, invert=False):
    """get_matrixfromants

    This function greps the rotation and translation matrices from the matrix-file create by `antsRegistration`. It basically does the same as on of the ANTs functions, but still..

    Parameters
    ----------
    mat: str
        string pointing to a *.mat*-file containing the transformation.
    invert: bool
        Boolean for inverting the matrix (`invert=False`) or not (`invert=True`)

    Return
    ----------
    numpy.ndarray
        (4,4) array representing the transformation matrix
    """

    try:
        genaff = io.loadmat(mat)
        key = list(genaff.keys())[0]
        matrix = np.hstack((genaff[key][0:9].reshape(
            3, 3), genaff[key][9:].reshape(3, 1)))
        matrix = np.vstack([matrix, [0, 0, 0, 1]])
    except:
        # assuming I just got a matrix
        matrix = np.loadtxt(mat)
        
    if invert == True:
        matrix = np.linalg.inv(matrix)

    return matrix

def make_chicken_csv(coord, input="ras", output_file=None, vol=0.343):
    """make_chicken_csv

    This function creates a .csv-file like the chicken.csv example from ANTs to warp a coordinate using a transformation file. ANTs assumes the input coordinate to be LPS, but this function
    can deal with RAS-coordinates too. (see https://github.com/stnava/chicken for the reason of this function's name)

    Parameters
    ----------
    coord: np.ndarray
        numpy array containing the three coordinates in x,y,z direction
    input: str
        specify whether your coordinates uses RAS or LPS convention (default is RAS, and will be converted to LPS to create the file)
    output_file: str
        path-like string pointing to an output file (.csv!)
    vol: float
        volume of voxels (pixdim_x*pixdim_y*pixdim_z). If you're using the standard 0.7 MP2RAGE, the default vol will be ok

    Returns
    ----------
    str
        path pointing to the `csv`-file containing the coordinate

    Example
    ----------
    >>> make_chicken_csv(np.array([-16.239,-67.23,-2.81]), output_file="sub-001_space-fs_desc-lpi.csv")
    "sub-001_space-fs_desc-lpi.csv"
    """

    if len(coord) > 3:
        coord = coord[:3]

    if input.lower() == "ras":
        # ras2lps
        LPS = np.array([[-1,0,0],
                        [0,-1,0],
                        [0,0,1]])

        coord = LPS @ coord

    # rows = ["x,y,z,t,label,mass,volume,count", f"{coord[0]},{coord[1]},{coord[2]},0,1,1,{vol},1"]
    with open(output_file, "w") as target:
        writer = csv.writer(target, delimiter=",")
        writer.writerow(["x","y","z","t","label","mass","volume","count"])
        writer.writerow([coord[0],coord[1],coord[2],0,1,1,vol,1])

    return output_file

def read_chicken_csv(chicken_file, return_type="lps"):
    """read_chicken_csv

    Function to get at least the coordinates from a csv file used with antsApplyTransformsToPoints. (see https://github.com/stnava/chicken for the reason of this function's name)

    Parameters
    ----------
    chicken_file: str
        path-like string pointing to an input file (.csv!)
    return_type: str
        specify the coordinate system that the output should be in

    Returns
    ----------
    numpy.ndarray
        (3,) array containing the coordinate in `chicken_file`

    Example
    ----------
    >>> read_chicken_csv("sub-001_space-fs_desc-lpi.csv")
    array([-16.239,-67.23,-2.81])
    """
    
    contents = pd.read_csv(chicken_file)
    coord = np.squeeze(contents.iloc[:,0:3].values)

    if return_type.lower() == "lps":
        return coord
    elif return_type.lower() == "ras":
        # ras2lps
        LPS = np.array([[-1,0,0],
                        [0,-1,0],
                        [0,0,1]])

        return LPS@coord

def fix_slicetiming(json_dir, TR=1.5, slc=60):

    """fix_slicetiming

    Function to fix the slicetiming in json file. Assumes there already is a key called SliceTiming in the json files. You'll only need to specify the directory the json-files are in, the TR, (default = 1.5), and the number of slices (default = 60)

    Parameters
    ----------
    json_dir: str
        path to folder containing json files
    TR: float
        repetition time
    slc: int
        number of slices

    Returns
    ----------
    str
        updated json-file

    Example
    ----------
    >>> fix_slicetiming('path/to/folder/with/json', TR=1.5, slc=60)
    """

    op = os.listdir(json_dir)

    for ii in op:
        if ii.endswith('.json'):
            with open(opj(json_dir,ii)) as in_file:
                data = json.load(in_file)

            data['SliceTiming'] = list(np.tile(np.linspace(0, TR, int(slc/3), endpoint=False), 3))

            with open(opj(json_dir,ii), 'w') as out_file:
                json.dump(data, out_file, indent=4)

class VertexInfo:

    """ VertexInfo
    
    This object reads a .csv file containing relevant information about the angles, vertex position, and normal vector.
    
    Parameters
    ----------
    infofile: str
        path to the information file containing `best_vertices` in the filename
    subject: str
        subject ID as used in `SUBJECTS_DIR`

    Returns
    ----------
    attr
        sets attributes in the class    
    """

    def __init__(self, infofile=None, subject=None):
        
        self.infofile = infofile
        self.data = pd.read_csv(self.infofile, index_col=0)
        
        # try to set the index to hemi. It will throw an error if you want to set the index while there already is an index.
        # E.g., initially we will set the index to 'hemi'. If we then later on read in that file again, the index is already 
        # set
        try:
            self.data = self.data.set_index('hemi')
        except:
            pass
        
        # check if arrays are in string format
        for hemi in ["L", "R"]:
            self.data['normal'][hemi]   = string2float(self.data['normal'][hemi])
            self.data['position'][hemi] = string2float(self.data['position'][hemi])
        
        self.subject = subject

    def get(self, keyword, hemi='both'):

        """return values from dataframe given keyword. Can be any column name or 'prf' for pRF-parameters"""

        keywords = np.array(self.data.columns)

        if keyword == "prf":

            if hemi == "both":
                return {"lh": [self.data[ii]['L'] for ii in ['x', 'y', 'size', 'beta', 'baseline', 'r2']],
                        "rh": [self.data[ii]['R'] for ii in ['x', 'y', 'size', 'beta', 'baseline', 'r2']]}
            elif hemi.lower() == "right" or hemi.lower() == "r" or hemi.lower() == "rh":
                return [self.data[ii]['R'] for ii in ['x', 'y', 'size', 'beta', 'baseline', 'r2']]
            elif hemi.lower() == "left" or hemi.lower() == "l" or hemi.lower() == "lh":
                return [self.data[ii]['L'] for ii in ['x', 'y', 'size', 'beta', 'baseline', 'r2']]

        else:

            if keyword not in keywords:
                raise ValueError(f"{keyword} does not exist in {keywords}")

            if hemi == "both":
                return {"lh": self.data[keyword]['L'],
                        "rh": self.data[keyword]['R']}
            elif hemi.lower() == "right" or hemi.lower() == "r" or hemi.lower() == "rh":
                return self.data[keyword]['R']
            elif hemi.lower() == "left" or hemi.lower() == "l" or hemi.lower() == "lh":
                return self.data[keyword]['L']


def make_binary_cm(color):

    """make_binary_cm

    This function creates a custom binary colormap using matplotlib based on the RGB code specified. Especially useful if you want to overlay in imshow, for instance. These RGB values will be converted to range between 0-1 so make sure you're specifying the actual RGB-values. I like `https://htmlcolorcodes.com` to look up RGB-values of my desired color. The snippet of code used here comes from https://kbkb-wx-python.blogspot.com/2015/12/python-transparent-colormap.html

    Parameters
    ----------
    <color>: tuple, str
        either  hex-code with (!!) '#' or a tuple consisting of:

        * <R>     int | red-channel (0-255)
        * <G>     int | green-channel (0-255)
        * <B>     int | blue-channel (0-255)
    
    Returns
    ----------
    matplotlib.colors.LinearSegmentedColormap object
        colormap to be used with `plt.imshow`

    Example
    ----------
    >>> cm = make_binary_cm((232,255,0))
    >>> cm
    <matplotlib.colors.LinearSegmentedColormap at 0x7f35f7154a30>
    >>> cm = make_binary_cm("#D01B47")
    >>> cm
    >>> <matplotlib.colors.LinearSegmentedColormap at 0x7f35f7154a30>
    """

    if isinstance(color, tuple):
        (R,G,B) = color
    elif isinstance(color, str):
        color = ImageColor.getcolor(color, "RGB")
        (R,G,B) = color

    if R > 1:
        R = R/255

    if G > 1:
        G = G/255

    if B > 1:
        B = B/255

    colors = [(R,G,B,c) for c in np.linspace(0,1,100)]
    cmap = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=5)

    return cmap


def percent_change(ts, ax):
    """convert timeseries to percent signal change via the nilearn method"""

    return (ts / np.expand_dims(np.mean(ts, ax), ax) - 1) * 100    


class ParseFuncFile():

    """ParseFuncFile()

    Class for parsing func-files created with Luisa's reconstruction. It can do filtering, conversion to percent signal change, and create power spectra. It is supposed to look similar to :class:`linescanning.utils.ParseExpToolsFile` to make it easy to translate between the functional data and experimental data.

    Parameters
    ----------
    func_file: str
        path pointing to the output file of the experiment
    subject: int, optional
        subject number in the returned pandas DataFrame (should start with 1, ..., n)
    run: int, optional
        run number you'd like to have the onset times for
    bp_filter: str, optional
        method for filtering. Possible options include: "boxcar", "hanning", "rolling", "iir", "fir", and fourier". If you just want percent signal changed non-filtered data, choose `bp_filter="rolling"` and select `self.dct_psc_df` as data
    standardize: str, optional
        method of standardization (e.g., "zscore" or "psc"). If `bp_filter=="rolling"`, data is automatically converted to *percent signal change*
    ub: float, optional
        upper bound for signal filtering
    lb: float, optional
        lower bound for signal filtering
    TR: float, optional
        repetition time to correct onset times for deleted volumes
    deleted_first_timepoints: int, optional
        number of volumes deleted at the beginning of the timeseries
    deleted_last_timepoints: int, optional
        number of volumes deleted at the end of the timeseries
    window_size: int, optional
        size of window for rolling median

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
                 bp_filter='boxcar', 
                 standardize='psc', 
                 ub=0.15, 
                 lb=0.01, 
                 TR=0.105, 
                 deleted_first_timepoints=38, 
                 deleted_last_timepoints=38, 
                 window_size=20):

        self.TR = TR
        self.ub = ub
        self.lb = lb
        self.subject = subject
        self.run = run
        self.deleted_first_timepoints = deleted_first_timepoints
        self.deleted_last_timepoints = deleted_last_timepoints
        self.standardize = standardize
        self.window_size = window_size

        # Load in datasets with tag "wcsmtSNR"
        self.ts_wcsmtSNR = io.loadmat(func_file)
        self.tag = list(self.ts_wcsmtSNR.keys())[-1]
        self.ts_wcsmtSNR = self.ts_wcsmtSNR[self.tag]

        self.ts_complex = self.ts_wcsmtSNR
        self.ts_magnitude = np.abs(self.ts_wcsmtSNR)

        # trim beginning and end
        if self.deleted_last_timepoints != 0:
            self.ts_corrected = self.ts_magnitude[:,self.deleted_first_timepoints:-self.deleted_last_timepoints]
        else:
            self.ts_corrected = self.ts_magnitude[:,self.deleted_first_timepoints:]

        self.vox_cols = [f'vox {x}' for x in range(self.ts_corrected.shape[0])]
        self.raw_data = pd.DataFrame(self.ts_corrected.T, columns=self.vox_cols)
        self.raw_data['subject'], self.raw_data['run'], self.raw_data['t'] = subject, run, list(self.TR*np.arange(self.raw_data.shape[0]))

        # nitime filtering?
        if isinstance(bp_filter, str):

            if bp_filter != "hanning" and bp_filter != "rolling":
                self.T = TimeSeries(self.ts_corrected, sampling_interval=self.TR)
                self.F = FilterAnalyzer(self.T, ub=self.ub, lb=self.lb)

                if bp_filter.lower() == "boxcar":
                    filter_type = self.F.filtered_boxcar
                elif bp_filter.lower() == "iir":
                    filter_type = self.F.iir
                elif bp_filter.lower() == "fir":
                    filter_type = self.F.fir
                elif bp_filter.lower() == "fourier":
                    filter_type = self.F.filtered_fourier
                else:
                    raise ValueError(f"'{bp_filter}' was requested, but must be one of: 'boxcar', 'iir', 'fir', 'fourier', 'hanning', 'rolling (WIP)'")

                # normalize
                self.psc = NormalizationAnalyzer(filter_type).percent_change.data
                self.zscore = NormalizationAnalyzer(filter_type).z_score.data

                self.mixed_zscore = pd.DataFrame(self.zscore.T, columns=self.vox_cols)
                self.mixed_psc = pd.DataFrame(self.psc.T, columns=self.vox_cols)

                self.mixed_zscore['subject'], self.mixed_zscore['run'], self.mixed_zscore['t'] = subject, run, list(self.TR*np.arange(self.zscore.shape[-1]))
                self.mixed_psc['subject'], self.mixed_psc['run'], self.mixed_psc['t'] = subject, run, list(self.TR*np.arange(self.psc.shape[-1]))

            elif bp_filter == "hanning":
                
                # filter by convolving Gaussian filter with data
                self.windowSize = 20
                self.window = np.hanning(self.windowSize)
                self.window = self.window / self.window.sum()
                
                self.hanning_filtered = []
                for ii in range(self.ts_corrected.shape[0]):
                    convolved = np.convolve(self.window, self.ts_corrected[ii,:], mode='valid')
                    self.hanning_filtered.append(convolved)

                self.hanning_filtered = np.array(self.hanning_filtered)
                # convert to percent change
                self.han_filt_psc = percent_change(self.hanning_filtered, -1)

                # detrend (high pass filter)
                self.han_filt_psc_det  = detrend(self.han_filt_psc, axis=-1)

                # make into dataframe
                self.han_filt_psc_det_df = pd.DataFrame(self.han_filt_psc_det.T, columns=self.vox_cols)
                self.han_filt_psc_det_df['subject'] = subject
                self.han_filt_psc_det_df['run'] = run
                self.han_filt_psc_det_df['t'] = list(self.TR*np.arange(self.han_filt_psc_det_df.shape[0]))

            elif bp_filter == "rolling":

                # Create high-pass filter and clean
                n_vol = self.ts_corrected.shape[1]
                tr = self.TR
                st_ref = 0  # offset frametimes by st_ref * tr
                ft = np.linspace(st_ref * self.TR, (n_vol + st_ref) * self.TR, n_vol, endpoint=False)

                self.hp_set = dct_set(self.lb, ft)
                self.filt_dct = clean(self.ts_corrected.T, detrend=False, standardize=self.standardize, confounds=self.hp_set)
                # self.dct_psc_df = pd.DataFrame(self.filt_dct, index=self.raw_data.index, columns=self.raw_data.columns)
                self.dct_psc_df = pd.DataFrame(self.filt_dct, columns=self.vox_cols)
                self.dct_psc_df['subject'], self.dct_psc_df['run'], self.dct_psc_df['t'] = subject, run, list(self.TR*np.arange(self.dct_psc_df.shape[0]))
                self.dct_psc_df = self.dct_psc_df.set_index(['subject', 'run', 't'])

                if self.window_size != None:
                    self.dct_psc_rol = self.dct_psc_df.rolling(self.window_size, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None).median()
                    self.dct_psc_rol = self.dct_psc_rol.fillna(0)
                else:
                    self.dct_psc_rol = self.dct_psc_df.copy()
                
    def get_psc(self, index=False):
        if hasattr(self, 'mixed_psc'):
            if index:
                return self.mixed_psc.set_index(['subject', 'run', 't'])
            else:
                return self.mixed_psc
        else:
            raise ValueError("Did not filter data; use 'bp_filter=True' to get percent-signal change data")

    def get_zscore(self, index=False):
        if hasattr(self, 'mixed_zscore'):
            if index:
                return self.mixed_zscore.set_index(['subject', 'run', 't'])
            else:
                return self.mixed_zscore
        else:
            raise ValueError("Did not filter data; use 'bp_filter=True' to get zscored data")

    def get_raw(self, index=False):
        if index:
            return self.raw_data.set_index(['subject', 'run', 't'])
        else:
            return self.raw_data

    def get_hanning(self, index=False):
        if index:
            return self.han_filt_psc_det_df.set_index(['subject', 'run', 't'])
        else:
            return self.han_filt_psc_det_df

    def get_rolling(self, index=False):
        if hasattr(self, 'dct_psc_rol'):
            if index:
                return self.dct_psc_rol.set_index(['subject', 'run', 't'])
            else:
                return self.dct_psc_rol
        else:
            raise ValueError("Did not filter data; use 'bp_filter=rolling' to get rolling median'ed data")

    def get_freq(self, datatype='raw', spectrum_type='psd'):

        """return power & frequency spectrum from timeseries"""

        if datatype == "raw":
            self.TC = self.raw_data.copy()
        elif datatype == "psc":
            self.TC = self.mixed_psc.copy()
        elif datatype == "zscore":
            self.TC = self.mixed_psc.copy()

        self.TC = TimeSeries(np.asarray(self.TC), sampling_interval=self.TR)
        self.spectra = SpectralAnalyzer(self.TC)

        if spectrum_type == "psd":
            selected_spectrum = self.spectra.psd
        elif spectrum_type == "fft":
            selected_spectrum = self.spectra.spectrum_fourier
        elif spectrum_type == "periodogram":
            selected_spectrum = self.spectra.periodogram
        elif spectrum_type == "mtaper":
            selected_spectrum = self.spectra.spectrum_multi_taper
        else:
            raise ValueError(f"Requested spectrum was '{spectrum_type}'; available options are: 'psd', 'fft', 'periodogram', or 'mtaper'")

        return selected_spectrum[0], selected_spectrum[1]

class ParseExpToolsFile(object):

    """ParseExpToolsFile()

    Class for parsing tsv-files created during experiments with Exptools2. The class will read in the file, read when the experiment actually started, correct onset times for this start time and time deleted because of removing the first few volumes (to do this correctly, set the `TR` and `delete_vols`). You can also provide a numpy array/file containing eye blinks that should be added to the onset times in real-world time (seconds). In principle, it will return a pandas DataFrame indexed by subject and run that can be easily concatenated over runs. This function relies on the naming used when programming the experiment. In the `session.py` file, you should have created `phase_names=['iti', 'stim']`; the class will use these things to parse the file.

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
    delete_vols: int
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

    def __init__(self, tsv_file, subject=1, run=1, button=False, blinks=None, TR=0.105, delete_vols=38):

        """Initialize object and do all of the parsing/correction/reading"""

        self.subject = int(subject)
        self.run = int(run)
        self.deleted_volumes = delete_vols
        self.TR = TR
        self.delete_time = self.deleted_volumes*self.TR

        data_onsets = []
        with open(tsv_file) as f:
            timings = pd.read_csv(f, delimiter='\t')
            data_onsets.append(pd.DataFrame(timings))

        self.data = data_onsets[0]
        self.start_times = pd.DataFrame(self.data[(self.data['response'] == 't') & (self.data['trial_nr'] == 1) & (self.data['phase'] == 0)][['onset']])
        self.data_cut_start = self.data.drop([q for q in np.arange(0,self.start_times.index[0])])
        self.onset_times = pd.DataFrame(self.data_cut_start[(self.data_cut_start['event_type'] == 'stim') & (self.data_cut_start['condition'].notnull()) | (self.data_cut_start['response'] == 'b')][['onset', 'condition']]['onset'])
        self.condition = pd.DataFrame(self.data_cut_start[(self.data_cut_start['event_type'] == 'stim') & (self.data_cut_start['condition'].notnull()) | (self.data_cut_start['response'] == 'b')]['condition'])

        # add button presses
        if button:
            self.response = self.data_cut_start[(self.data_cut_start['response'] == 'b')]
            self.condition.loc[self.response.index] = 'response'

        self.onset = np.concatenate((self.onset_times, self.condition), axis=1)

        # add eyeblinks
        if isinstance(blinks, np.ndarray) or isinstance(blinks, str):
            if isinstance(blinks, np.ndarray):
                self.eye_blinks = blinks
            elif isinstance(blinks, str):
                if blinks.endwith(".npy"):
                    self.eye_blinks = np.load(blinks)
                else:
                    raise ValueError(f"Could not recognize type of {blinks}. Should be numpy array or string to numpy file")

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
        self.onset[:,0] = self.onset[:,0]-float(self.start_times['onset'] + self.delete_time)

        # create nideconv-compatible dataframe
        self.onset_df = pd.DataFrame(self.onset, columns=['onset', 'event_type'])
        self.onset_df['subject'], self.onset_df['run'] = self.subject, self.run
        self.onset_df['event_type'] = self.onset_df['event_type'].astype(str)
        self.onset_df['onset'] = self.onset_df['onset'].astype(float)

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

    def __init__(self, physio_file, physio_mat=None, subject=1, run=1, TR=0.105, orders=[2,2,2], deleted_first_timepoints=38, deleted_last_timepoints=38):

        self.physio_file = physio_file
        self.subject = subject
        self.run = run
        self.TR = TR
        self.orders = orders
        self.deleted_first_timepoints = deleted_first_timepoints
        self.deleted_last_timepoints = deleted_last_timepoints
        self.physio_mat = physio_mat

        self.physio_cols = [f'c_{i}' for i in range(self.orders[0])] + [f'r_{i}' for i in range(self.orders[1])] + [f'cr_{i}' for i in range(self.orders[2])]
        
        self.physio_data = pd.read_csv(self.physio_file,
                                       header=None,
                                       sep="\t",
                                       engine='python',
                                       skiprows=deleted_first_timepoints,
                                       usecols=list(range(0, len(self.physio_cols))))

        self.physio_df = pd.DataFrame(self.physio_data)
        self.physio_df.drop(self.physio_df.tail(self.deleted_last_timepoints).index,inplace=True)
        self.physio_df.columns = self.physio_cols

        # Try to get the heart rate
        if self.physio_mat != None:
            self.mat = io.loadmat(self.physio_mat)
            self.hr = self.mat['physio']['ons_secs'][0][0][0][0][12]
            self.physio_df['hr'] = self.hr[self.deleted_first_timepoints:-self.deleted_last_timepoints,...]
            self.physio_df['hr'] = (self.physio_df['hr'] - self.physio_df['hr'].mean())/self.physio_df['hr'].std(ddof=0)
        self.physio_df['subject'], self.physio_df['run'], self.physio_df['t'] = self.subject, self.run, list(self.TR*np.arange(self.physio_df.shape[0]))

    def get_physio(self, index=True):
        if index:
            return self.physio_df.set_index(['subject', 'run', 't'])
        else:
            return self.physio_df


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
    

def filter_for_nans(array):
    """filter out NaNs from an array"""

    if np.isnan(array).any():
        return np.nan_to_num(array)
    else:
        return array

def find_max_val(array):
    """find the index of maximum value given an array"""
    return np.where(array == np.amax(array))[0]


def read_fs_reg(dat_file):
    """read_fs_reg

    Read a `.dat`-formatted registration file from FreeSurfer

    Parameters
    ----------
    dat_file: str
        path pointing to the registration file

    Returns
    ----------
    nump.ndarray
        (4,4) numpy array containing the transformation
    """
    with open(dat_file) as f:
        d = f.readlines()[4:-1]

        return np.array([[float(s) for s in dd.split() if s] for dd in d])


def random_timeseries(intercept, volatility, nr):
    """random_timeseries

    Create a random timecourse by multiplying an intercept with a random Gaussian distribution.

    Parameters
    ----------
    intercept: float
        starting point of timecourse
    volatility: float
        this factor is multiplied with the Gaussian distribution before multiplied with the intercept
    nr: int
        length of timecourse 

    Returns
    ----------
    numpy.ndarray
        array of length `nr`

    Example
    ----------
    >>> from linescanning import utils
    >>> ts = utils.random_timeseries(1.2, 0.5, 100)

    Notes
    ----------
    Source: https://stackoverflow.com/questions/67977231/how-to-generate-random-time-series-data-with-noise-in-python-3
    """
    time_series = [intercept, ]
    for _ in range(nr):
        time_series.append(time_series[-1] + intercept * random.gauss(0,1) * volatility)
    return np.array(time_series[:-1])


class LazyPlot():

    def __init__(self, ts, xx=None, error=None, error_alpha=0.3, x_label=None, y_label=None, title=None, xkcd=False, color=None, figsize=(12, 5), cmap='viridis', save_as=None,  labels=None, font_size=12, add_hline=None, add_vline=None, line_width=1, axs=None):
        """__init__

        Class for plotting because I'm lazy and I don't want to go through the `matplotlib` motion everything I quickly want to visualize something. This class makes that a lot easier. It allows single inputs, lists with multiple timecourses, labels, error shadings, and much more.

        Parameters
        ----------
        ts: list, numpy.ndarray
            Input data. Can either be a single list, or a list of multiple numpy arrays. If you want labels, custom colors, or error bars, these inputs must come in lists of similar length as `ts`!
        xx: list, numpy.ndarray, optional
            X-axis array
        error: list, numpy.ndarray, optional
            Error data with the same length/shape as the input timeseries, by default None. Can be either a numpy.ndarray for 1 timeseries, or a list of numpy.ndarrays for multiple timeseries
        error_alpha: float, optional
            Opacity level for error shadings, by default 0.3
        x_label: str, optional
            Label of x-axis, by default None
        y_label: str, optional
            Label of y-axis, by default None
        labels: str, list, optional
            String (if 1 timeseries) or list (with the length of `ts`) of colors, by default None. Labels for the timeseries to be used in the legend
        title: str, optional
            Plot title, by default None
        xkcd: bool, optional
            Plot the figre in XKCD-style (cartoon), by default False
        color: str, list, optional
            String (if 1 timeseries) or list (with the length of `ts`) of colors, by default None. If nothing is specified, we'll use `cmap` to create a color palette
        cmap: str, optional
            Color palette to use for colors if no individual colors are specified, by default 'viridis'
        figsize: tuple, optional
            Figure dimensions as per usual matplotlib conventions, by default (12,5)
        save_as: str, optional
            Save the plot, by default None. If you want to use figures in Inkscape, save them as PDFs to retain high resolution
        font_size: int, optional
            Font size of titles and axis labels, by default 12
        add_hline: dict, optional
            Dictionary for a horizontal line through the plot, by default None. Collects the following items:

            >>> add_hline = {'pos' 0,       # position
            >>>              'color': 'k',  # color
            >>>              'lw': 1,       # linewidth
            >>>              'ls': '--'}    # linestyle
            You can get the settings above by specifying *add_hline='defaults'*
        add_vline: [type], optional
            Dictionary for a vertical line through the plot, by default None. Same keys as `add_hline`
        line_width: int, list, optional
            Line widths for either all graphs (then *int*) or a *list* with the number of elements as requested graphs, default = 1.
        axs: <AxesSubplot:>, optional
            Matplotlib axis to store the figure on
            
        Example
        ----------
        >>> # create a bunch of timeseries
        >>> from linescanning import utils
        >>> ts = utils.random_timeseries(1.2, 0.0, 100)
        >>> ts1 = utils.random_timeseries(1.2, 0.3, 100)
        >>> ts2 = utils.random_timeseries(1.2, 0.5, 100)
        >>> ts3 = utils.random_timeseries(1.2, 0.8, 100)
        >>> ts4 = utils.random_timeseries(1.2, 1, 100)
        
        >>> # plot 1 timecourse
        >>> glm.LazyPlot(ts2, figsize=(20, 5))
        <linescanning.glm.LazyPlot at 0x7f839b0289d0>

        >>> # plot multiple timecourses, add labels, and save file
        >>> glm.LazyPlot([ts, ts1, ts2, ts3, ts4], figsize=(20, 5), save_as="/mnt/d/FSL/shared/spinoza/programs/project_repos/PlayGround/results/test_LazyPlot.pdf", labels=['vol=0', 'vol=0.3', 'vol=0.5', 'vol=0.8', 'vol=1.0'])
        <linescanning.glm.LazyPlot at 0x7f839b2177c0>

        >>> # add horizontal line at y=0
        >>> hline = {'pos': 0, 'color': 'k', 'lw': 0.5, 'ls': '--'}
        >>> >>> glm.LazyPlot(ts2, figsize=(20, 5), add_hline=hline)
        <linescanning.glm.LazyPlot at 0x7f839b053580>

        >>> # add shaded error bars
        >>> from scipy.stats import sem
        >>> stack = np.hstack((ts1[...,np.newaxis],ts2[...,np.newaxis],ts4[...,np.newaxis])) # make some stack
        >>> avg = stack.mean(axis=-1) # calculate mean
        >>> err = sem(stack, axis=-1) # calculate error
        >>> glm.LazyPlot(avg, figsize=(20, 5), error=err)
        <linescanning.glm.LazyPlot at 0x7f839b0d5220>
        """

        self.array = ts
        self.xx = xx
        self.error = error
        self.error_alpha = error_alpha
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.xkcd = xkcd
        self.color = color
        self.figsize = figsize
        self.cmap = cmap
        self.save_as = save_as
        self.labels = labels
        self.font_size = font_size
        self.add_hline = add_hline
        self.add_vline = add_vline
        self.axs = axs
        self.line_width = line_width

        if self.xkcd:
            with plt.xkcd():
                self.fontname = "Humor Sans"
                self.plot()
        else:
            self.fontname = "Arial"
            self.plot()

        if save_as:
            plt.savefig(self.save_as, transparent=True)

    def plot(self):

        if self.axs == None:
            fig, axs = plt.subplots(figsize=self.figsize)
        else:
            axs = self.axs

        if isinstance(self.array, list):

            if not isinstance(self.color, list):
                color_list = sns.color_palette(self.cmap, len(self.array))
            else:
                color_list = self.color
                if len(color_list) != len(self.array):
                    raise ValueError(f"Length color list {len(color_list)} does not match length of data list ({len(self.array)}")

            for idx, el in enumerate(self.array):

                if isinstance(self.line_width, list):
                    if len(self.line_width) != len(self.array):
                        raise ValueError(f"Length of line width lenghts {len(self.line_width)} does not match length of data list ({len(self.array)}")
                        
                    use_width=self.line_width[idx]
                elif isinstance(self.line_width, int):
                    use_width=self.line_width
                else:
                    use_width=""

                # decide on x-axis
                if not isinstance(self.xx, np.ndarray) and not isinstance(self.xx, list):
                    x = np.arange(0, len(el))
                else:
                    x = self.xx.copy()

                if self.labels:
                    axs.plot(x, el, color=color_list[idx], label=self.labels[idx], lw=use_width)
                else:
                    axs.plot(x, el, color=color_list[idx], lw=use_width)

                if isinstance(self.error, list) or isinstance(self.error, np.ndarray):
                    yerr = self.error[idx]
                    if np.isscalar(yerr) or len(yerr) == len(el):
                        ymin = el - yerr
                        ymax = el + yerr
                    elif len(yerr) == 2:
                        ymin, ymax = yerr
                    x = np.arange(0, len(el))
                    axs.fill_between(
                        x, ymax, ymin, color=self.color_list[idx], alpha=self.error_alpha)
        else:
            if not self.color:
                self.color = sns.color_palette(self.cmap, 1)[0]

            # decide on x-axis
            if not isinstance(self.xx, np.ndarray) and not isinstance(self.xx, list):
                x = np.arange(0, len(self.array))
            else:
                x = self.xx.copy()

            axs.plot(x, self.array, color=self.color, label=self.labels, lw=self.line_width)

            if isinstance(self.error, list) or isinstance(self.error, np.ndarray):
                if np.isscalar(self.error) or len(self.error) == len(self.array):
                    ymin = self.array - self.error
                    ymax = self.array + self.error
                elif len(self.error) == 2:
                    ymin, ymax = self.error
                x = np.arange(0, len(self.array))
                axs.fill_between(x, ymax, ymin, color=self.color, alpha=self.error_alpha)

        if self.labels:
            axs.legend(frameon=False)

        if self.x_label:
            axs.set_xlabel(self.x_label, fontname=self.fontname, fontsize=self.font_size)

        if self.y_label:
            axs.set_ylabel(self.y_label, fontname=self.fontname, fontsize=self.font_size)

        if self.title:
            axs.set_title(self.title, fontname=self.fontname, fontsize=self.font_size)

        if self.add_vline:
            if self.add_vline == "defaults":
                self.add_vline={'pos': 0, 'color': 'k', 'ls': 'dashed', 'lw': 0.5}

            axs.axvline(self.add_vline['pos'], color=self.add_vline['color'], lw=self.add_vline['lw'], ls=self.add_vline['ls'])

        if self.add_hline:
            if self.add_hline == "defaults":
                self.add_hline={'pos': 0, 'color': 'k', 'ls': 'dashed', 'lw': 0.5}

            axs.axhline(self.add_hline['pos'], color=self.add_hline['color'], lw=self.add_hline['lw'], ls=self.add_hline['ls'])

        sns.despine(offset=10)


def squeeze_generic(a, axes_to_keep):
    """squeeze_generic

    Numpy squeeze implementation keeping <axes_to_keep> dimensions.

    Parameters
    ----------
    a: numpy.ndarray
        array to be squeezed
    axes_to_keep: tuple, range
        tuple of axes to keep from original input

    Returns
    ----------
    numpy.ndarray
        `axes_to_keep` from `a`

    Example
    ----------
    >>> a = np.random.rand(3,5,1)
    >>> squeeze_generic(a, axes_to_keep=range(2)).shape
    (3, 5)

    Notes
    ----------
    From: https://stackoverflow.com/questions/57472104/is-it-possible-to-squeeze-all-but-n-dimensions-using-numpy
    """
    out_s = [s for i, s in enumerate(a.shape) if i in axes_to_keep or s != 1]
    return a.reshape(out_s)


def find_intersection(xx, curve1, curve2):
    """find_intersection

    Find the intersection coordinates given two functions using `Shapely`.

    Parameters
    ----------
    xx: numpy.ndarray
        array describing the x-axis values
    curve1: numpy.ndarray
        array describing the first curve
    curve2: numpy.ndarray
        array describing the first curve

    Returns
    ----------
    tuple
        x,y coordinates where *curve1* and *curve2* intersect

    Raises
    ----------
    ValueError
        if no intersection coordinates could be found

    Example
    ----------
    See [refer to linescanning.prf.SizeResponse.find_stim_sizes]
    """

    first_line = geometry.LineString(np.column_stack((xx, curve1)))
    second_line = geometry.LineString(np.column_stack((xx, curve2)))
    intersection = first_line.intersection(second_line)

    try:
        x_coord, y_coord = geometry.LineString(intersection).xy[0]
    except:
        raise ValueError("Could not find intersection between curves..")

    return (x_coord, y_coord)


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

    def __init__(self, subject, derivatives=None, cx_dir=None, prf_dir=None, ses=1, analysis_yaml=None, hemi="lh", settings=None, model="gauss"):

        self.subject        = subject
        self.derivatives    = derivatives
        self.cx_dir         = cx_dir
        self.prf_dir        = prf_dir
        self.prf_ses        = ses
        self.hemi           = hemi
        self.model          = model

        # set pRF directory
        if self.prf_dir == None:
            if derivatives != None:
                self.prf_dir = opj(self.derivatives, 'prf', self.subject, f'ses-{self.prf_ses}')

        # get design matrix, vertex info, and analysis file
        self.design_fn      = get_file_from_substring("vis_design.mat", self.prf_dir)
        self.design_matrix  = io.loadmat(self.design_fn)['stim']
        self.vert_fn        = get_file_from_substring([self.model, "best_vertices.csv"], self.cx_dir)
        self.vert_info      = VertexInfo(self.vert_fn, subject=self.subject)
        self.analysis_yaml  = analysis_yaml

        # load specific analysis file
        if self.analysis_yaml != None:
            self.settings = yaml.safe_load(self.analysis_yaml)

            with open(self.analysis_yaml) as file:
                self.settings = yaml.safe_load(file)            
        
        # load the most recent analysis file. This is fine for screens/stimulus information
        if settings == "recent":
            analysis_yaml = opj(self.prf_dir, sorted([ii for ii in os.listdir(self.prf_dir) if "desc-settings" in ii])[-1])

            with open(analysis_yaml) as file:
                self.settings = yaml.safe_load(file)
        
        # fetch target vertex parameters
        self.target_params = self.return_prf_params(hemi=self.hemi)

        # create pRF if settings were specified
        if hasattr(self, "settings"):
            self.prf_stim = stimulus.PRFStimulus2D(screen_size_cm=self.settings['screen_size_cm'], screen_distance_cm=self.settings['screen_distance_cm'], design_matrix=self.design_matrix,TR=self.settings['TR'])
            self.prf_array = prf.make_prf(self.prf_stim, size=self.target_params[2], mu_x=self.target_params[0], mu_y=self.target_params[1])


    def return_prf_params(self, hemi="lh"):
        """return pRF parameters from :class:`linescanning.utils.VertexInfo`"""
        return self.vert_info.get('prf', hemi=hemi)

    def return_target_vertex(self, hemi="lh"):
        """return the vertex ID of target vertex"""
        return self.vert_info.get('index', hemi=hemi)
