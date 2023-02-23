import csv
import fnmatch
import json
import math
import matplotlib.colors as mcolors
import nibabel as nb
from nilearn.signal import _standardize
import numpy as np
import operator
import os
import pandas as pd
from PIL import ImageColor
import random
from scipy import io, interpolate
from shapely import geometry
import subprocess
import warnings

opj = os.path.join
pd.options.mode.chained_assignment = None # disable warning thrown by string2float
warnings.filterwarnings("ignore")

def verbose(msg, verbose, flush=True, **kwargs):
    if verbose:
        print(msg, flush=flush, **kwargs)

def calculate_tsnr(data,ax):
    mean_d = np.mean(data,axis=ax)
    std_d = np.std(data,axis=ax)
    tsnr = mean_d/std_d
    tsnr[np.where(np.isinf(tsnr))] = np.nan

    return tsnr
    
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

def str2operator(ops):

    if ops == "and" or ops == "&" or ops == "&&":
        return operator.and_
    elif ops == "or" or ops == "|" or ops == "||":
        return operator.or_
    elif ops == "is not" or ops == "!=":
        return operator.ne
    elif ops == "is" or ops == "==" or ops == "=":
        return operator.eq
    elif ops == "gt" or ops == ">":
        return operator.gt
    elif ops == "lt" or ops == "<":
            return operator.lt
    elif ops == "ge" or ops == ">=":
        return operator.ge
    elif ops == "le" or ops == "<=":
            return operator.le
    else:
        raise NotImplementedError()

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

    if not isinstance(fn, str):
        raise ValueError(f"Input must be string, not {fn} of type {type(fn)}")
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

def string2list(string_array):
    """string2list

    This function converts a array in string representation to a list of string. This can happen, for instance, when you use bash to give a list of strings to python, where ast.literal_eval fails.

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
    >>> string2list('[tc,bgfs]')
    ['tc', 'bgfs']
    """

    if type(string_array) == str:
        new = string_array.split(',')[0:]
        new = list(filter(None, new))

        return new

    else:
        # array is already in non-string format
        return string_array

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

def match_lists_on(ref_list, search_list, matcher="run"):
    """match_lists_on

    Match two list based on a BIDS-specifier such as 'sub', 'run', etc. Can be any key that is extracted using :func:`linescanning.utils.split_bids_components`.

    Parameters
    ----------
    ref_list: list
        List to use as reference
    search_list: list
        List to search for items in `ref_list`
    matcher: str, optional
        BIDS-identifier, by default "run"

    Returns
    ----------
    list
        new `search_list` filtered for items in `ref_list`

    Example
    ----------
    >>> # Let's say I have functional files for 3 runs
    >>> func_file
    >>> ['sub-003_ses-3_task-SR_run-3_bold.mat',
    >>> 'sub-003_ses-3_task-SR_run-4_bold.mat',
    >>> 'sub-003_ses-3_task-SR_run-6_bold.mat']

    >>> # and anatomical slices for 5 runs
    >>> anat_slices
    >>> ['sub-003_ses-3_acq-1slice_run-2_T1w.nii.gz',
    >>> 'sub-003_ses-3_acq-1slice_run-3_T1w.nii.gz',
    >>> 'sub-003_ses-3_acq-1slice_run-4_T1w.nii.gz',
    >>> 'sub-003_ses-3_acq-1slice_run-5_T1w.nii.gz',
    >>> 'sub-003_ses-3_acq-1slice_run-6_T1w.nii.gz']

    >>> # I can then use `match_list_on` to find the anatomical slices corresponding to the functional files
    >>> from linescanning import utils
    >>> utils.match_lists_on(func_file, anat_slices, matcher='run')
    >>> ['sub-003_ses-3_acq-1slice_run-3_T1w.nii.gz',
    >>> 'sub-003_ses-3_acq-1slice_run-4_T1w.nii.gz',
    >>> 'sub-003_ses-3_acq-1slice_run-6_T1w.nii.gz']
    """

    new_list = []
    for ii in ref_list:
        comps = split_bids_components(ii)
        ff = get_file_from_substring(f"{matcher}-{comps[matcher]}", search_list, return_msg="None")

        if ff != None:
            if ff == search_list:
                raise ValueError(f"Output list is equal to input list with identifier '{matcher}'. Please use unique identifier")
            new_list.append(ff)

    return new_list

def get_file_from_substring(filt, path, return_msg='error', exclude=None):
    """get_file_from_substring

    This function returns the file given a path and a substring. Avoids annoying stuff with glob. Now also allows multiple filters to be applied to the list of files in the directory. The idea here is to construct a binary matrix of shape (files_in_directory, nr_of_filters), and test for each filter if it exists in the filename. If all filters are present in a file, then the entire row should be 1. This is what we'll be looking for. If multiple files are found in this manner, a list of paths is returned. If only 1 file was found, the string representing the filepath will be returned. 

    Parameters
    ----------
    filt: str, list
        tag for files we need to select. Now also support a list of multiple filters. 
    path: str
        path to the directory from which we need to remove files
    return_msg: str, optional
        whether to raise an error (*return_msg='error') or return None (*return_msg=None*). Default = 'error'.
    exclude: str, optional:
        Specify string to exclude from options. This criteria will be ensued after finding files that conform to `filt` as final filter.

    Returns
    ----------
    str
        path to the files containing `string`. If no files could be found, `None` is returned

    list
        list of paths if multiple files were found

    Raises
    ----------
    FileNotFoundError
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
    
    input_is_list = False
    if isinstance(filt, str):
        filt = [filt]

    if isinstance(filt, list):
        # list and sort all files in the directory
        if isinstance(path, str):
            files_in_directory = sorted(os.listdir(path))
        elif isinstance(path, list):
            input_is_list = True
            files_in_directory = path.copy()
        else:
            raise ValueError("Unknown input type; should be string to path or list of files")

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
            if input_is_list:
                return fname
            else:
                f = opj(path, fname)
                if exclude != None:
                    if exclude not in f:
                        return opj(path, fname)
                    else:
                        if return_msg == "error":
                            raise FileNotFoundError(f"Could not find file with filters: {filt} and exclusion of [{exclude}] in '{path}'")
                        else:
                            return None
                else:
                    return opj(path, fname)
                
        elif len(full_match_idc) > 1:
            match_list = []
            for match in full_match_idc:
                fname = files_in_directory[match]
                if input_is_list:
                    match_list.append(fname)         
                else:
                    match_list.append(opj(path, fname))
            if exclude != None:
                exl_list = [f for f in match_list if exclude not in f]
                
                # return the string if there's only 1 element
                if len(exl_list) == 1:
                    return exl_list[0]
                else:
                    return exl_list
            else:
                return match_list
            # return match_list
        else:
            if return_msg == "error":
                raise FileNotFoundError(f"Could not find file with filters: {filt} in {path}")
            else:
                return None

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

    if mat.endswith(".mat"):
        genaff = io.loadmat(mat)
        key = list(genaff.keys())[0]
        matrix = np.hstack((genaff[key][0:9].reshape(
            3, 3), genaff[key][9:].reshape(3, 1)))
        matrix = np.vstack([matrix, [0, 0, 0, 1]])
    elif mat.endswith(".txt"):
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

    def __init__(self, infofile=None, subject=None, hemi="lh"):
        
        self.infofile = infofile
        self.data = pd.read_csv(self.infofile, index_col=0)
        
        # try to set the index to hemi. It will throw an error if you want to set the index while there already is an index.
        # E.g., initially we will set the index to 'hemi'. If we then later on read in that file again, the index is already 
        # set
        try:
            self.data = self.data.set_index('hemi')
        except:
            pass
            
        if hemi == "lh" or hemi.lower() == "l" or hemi.lower() == "left":
            self.hemi = "L"
        elif hemi == "rh" or hemi.lower() == "r" or hemi.lower() == "right":
            self.hemi = "R"
        else:
            self.hemi = "both"
        
        if self.hemi == "both":
            # check if arrays are in string format
            for hemi in ["L", "R"]:
                self.data['normal'][hemi]   = string2float(self.data['normal'][hemi])
                self.data['position'][hemi] = string2float(self.data['position'][hemi])
        else:
            self.data['normal'][self.hemi]   = string2float(self.data['normal'][self.hemi])
            self.data['position'][self.hemi] = string2float(self.data['position'][self.hemi])            
        
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

def percent_change(ts, ax, nilearn=False, baseline=20):
    """percent_change

    Function to convert input data to percent signal change. Two options are current supported: the nilearn method (`nilearn=True`), where the mean of the entire timecourse if subtracted from the timecourse, and the baseline method (`nilearn=False`), where the median of `baseline` is subtracted from the timecourse.

    Parameters
    ----------
    ts: numpy.ndarray
        Array representing the data to be converted to percent signal change. Should be of shape (n_voxels, n_timepoints)
    ax: int
        Axis over which to perform the conversion. If shape (n_voxels, n_timepoints), then ax=1. If shape (n_timepoints, n_voxels), then ax=0.
    nilearn: bool, optional
        Use nilearn method, by default False
    baseline: int, list, np.ndarray optional
        Use custom method where only the median of the baseline (instead of the full timecourse) is subtracted, by default 20. Length should be in `volumes`, not `seconds`. Can also be a list or numpy array (1d) of indices which are to be considered as baseline. The list of indices should be corrected for any deleted volumes at the beginning.

    Returns
    ----------
    numpy.ndarray
        Array with the same size as `ts` (voxels,time), but with percent signal change.

    Raises
    ----------
    ValueError
        If `ax` > 2
    """
    
    if ts.ndim == 1:
        ts = ts[:,np.newaxis]
        ax = 0
        
    if nilearn:
        if ax == 0:
            psc = _standardize(ts, standardize='psc')
        else:
            psc = _standardize(ts.T, standardize='psc').T
    else:

        # first step of PSC; set NaNs to zero if dividing by 0 (in case of crappy timecourses)
        ts_m = ts*np.expand_dims(np.nan_to_num((100/np.mean(ts, axis=ax))), ax)

        # get median of baseline
        if isinstance(baseline, np.ndarray):
            baseline = list(baseline)

        if ax == 0:
            if isinstance(baseline, list):
                median_baseline = np.median(ts_m[baseline,:], axis=0)
            else:
                median_baseline = np.median(ts_m[:baseline,:], axis=0)
        elif ax == 1:
            if isinstance(baseline, list):
                median_baseline = np.median(ts_m[:,baseline], axis=1)
            else:
                median_baseline = np.median(ts_m[:,:baseline], axis=1)
        else:
            raise ValueError("ax must be 0 or 1")

        # subtract
        psc = ts_m-np.expand_dims(median_baseline,ax)
        
    return psc

def select_from_df(df, expression="run = 1", index=True, indices=None, match_exact=True):
    """select_from_df

    Select a subset of a dataframe based on an expression. Dataframe should be indexed by the variable you want to select on or have the variable specified in the expression argument as column name. If index is True, the dataframe will be indexed by the selected variable. If indices is specified, the dataframe will be indexed by the indices specified through a list (only select the elements in the list) or a `range`-object (select within range).

    Parameters
    ----------
    df: pandas.DataFrame
        input dataframe
    expression: str, optional
        what subject of the dataframe to select, by default "run = 1". The expression must consist of a variable name and an operator. The operator can be any of the following: '=', '>', '<', '>=', '<=', '!=', separated by spaces. You can also change 2 operations by specifying the `&`-operator between the two expressions. If you want to use `indices`, specify `expression="ribbon"`. 
    index: bool, optional
        return output dataframe with the same indexing as `df`, by default True
    indices: list, range, numpy.ndarray, optional
        List, range, or numpy array of indices to select from `df`, by default None
    match_exact: bool, optional:
        When you insert a list of strings with `indices` to be filtered from the dataframe, you can either request that the items of `indices` should **match** exactly (`match_exact=True`, default) the column names of `df`, or whether the columns of `df` should **contain** the items of `indices` (`match_exact=False`).

    Returns
    ----------
    pandas.DataFrame
        new dataframe where `expression` or `indices` were selected from `df`

    Raises
    ----------
    TypeError
        If `indices` is not a tuple, list, or array

    Notes
    ----------
    See https://linescanning.readthedocs.io/en/latest/examples/nideconv.html for an example of how to use this function (do ctrl+F and enter "select_from_df").
    """    

    if expression == "ribbon":
        
        if isinstance(indices, tuple):
            return df.iloc[:,indices[0]:indices[1]]
        elif isinstance(indices, list):
            if all(isinstance(item, str) for item in indices):
                if match_exact:
                    return df[df.columns[df.columns.isin(indices)]]
                else:
                    df_tmp = []
                    for item in indices:
                        df_tmp.append(df[df.columns[df.columns.str.contains(item)]])

                    return pd.concat(df_tmp, axis=1)
            else:
                return df.iloc[:,indices]
        elif isinstance(indices, np.ndarray):
            return df.iloc[:,list(indices)]
        else:
            raise TypeError(f"Unknown type '{type(indices)}' for indices; must be a tuple of 2 values representing a range, or a list/array of indices to select")
    else:
        # fetch existing indices
        idc = list(df.index.names)
        if idc[0] != None:
            reindex = True
        else:
            reindex = False

        # sometimes throws an error if you're trying to reindex a non-indexed dataframe
        try:
            df = df.reset_index()
        except:
            pass
        
        sub_df = df.copy()
        if isinstance(expression, str):
            expression = [expression]

        if isinstance(expression, (tuple,list)):

            expressions = expression[::2]
            operators = expression[1::2]

            if len(expressions) == 1:

                col1,operator1,val1 = expressions[0].split()
                ops1 = str2operator(operator1)
                
                # use dtype of whatever dtype the colum is
                search_value = np.array([val1], dtype=type(sub_df.reset_index()[col1].values[0]))
                sub_df = sub_df.loc[ops1(sub_df[col1], search_value[0])]
                
            if len(expressions) == 2:
                col1,operator1,val1 = expressions[0].split()
                col2,operator2,val2 = expressions[1].split()

                main_ops = str2operator(operators[0])
                ops1 = str2operator(operator1)
                ops2 = str2operator(operator2)

                # check if we should interpret values invididually as integers
                search_value1 = np.array([val1], dtype=type(sub_df.reset_index()[col1].values[0]))[0]
                search_value2 = np.array([val2], dtype=type(sub_df.reset_index()[col2].values[0]))[0]

                sub_df = sub_df.loc[main_ops(ops1(sub_df[col1], search_value1), ops2(sub_df[col2], search_value2))]

        # first check if we should do indexing
        if index != None:
            # then check if we actually have something to index
            if reindex:
                if idc[0] != None:
                    sub_df = sub_df.set_index(idc)

        return sub_df

def split_bids_components(fname):

    comp_list = fname.split('_')
    comps = {}
    
    ids = ['sub', 'ses', 'task', 'acq', 'rec', 'run', 'space', 'hemi', 'model', 'stage', 'desc', 'vox']
    for el in comp_list:
        for i in ids:
            if i in el:
                comp = el.split('-')[-1]

                if "." in comp:
                    ic = comp.index(".")
                    if ic > 0:
                        ex = 0
                    else:
                        ex = -1

                    comp = comp.split(".")[ex]
                
                # if i == "run":
                #     comp = int(comp)

                comps[i] = comp

    if len(comps) != 0:
        return comps
    else:
        print(f"Could not find any element of {ids} in {fname}")

def subjects_in_list(input):

    subj = []
    for ii in input:
        subj.append(split_bids_components(ii)["sub"])

    return list(np.unique(np.array(subj)))

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


def disassemble_fmriprep_wf(wf_path, subj_ID, prefix="sub-"):
    """disassemble_fmriprep_wf

    Parses the workflow-folder from fMRIPrep into its constituents to recreate a filename. Searches for the following keys: `['ses', 'task', 'acq', 'run']`.

    Parameters
    ----------
    wf_path: str
        Path to workflow-folder
    subj_ID: str
        Subject ID to append to `prefix`
    prefix: str, optional
        Forms together with `subj_ID` the beginning of the new filename. By default "sub-"

    Returns
    ----------
    str
        filename based on constituent file parts

    Example
    ----------
    >>> from linescanning.utils import disassemble_fmriprep_wf
    >>> wf_dir = "func_preproc_ses_2_task_pRF_run_1_acq_3DEPI_wf"
    >>> fname = disassemble_fmriprep_wf(wf_dir, "001")
    >>> fname
    'sub-001_ses-2_task-pRF_acq-3DEPI_run-1'
    """
    wf_name = [ii for ii in wf_path.split(os.sep) if "func_preproc" in ii][0]
    wf_elem = wf_name.split("_")
    fname = [f"{prefix}{subj_ID}"]

    for tag in ['ses', 'task', 'acq', 'run']:

        if tag in wf_elem:
            idx = wf_elem.index(tag)+1
            fname.append(f"{tag}-{wf_elem[idx]}")

    fname = "_".join(fname)
    return fname


def resample2d(array:np.ndarray, new_size:int, kind='linear'):
    """resample2d

    Resamples a 2D (or 3D) array with :func:`scipy.interpolate.interp2d` to `new_size`. If input is 2D, we'll loop over the final axis.

    Parameters
    ----------
    array: np.ndarray
        Array to be interpolated. Ideally axis have the same size.
    new_size: int
        New size of array
    kind: str, optional
        Interpolation method, by default 'linear'

    Returns
    ----------
    np.ndarray
        If 2D: resampled array of shape `(new_size,new_size)`
        If 3D: resampled array of shape `(new_size,new_size, array.shape[-1])`
    """
    # set up interpolater
    x = np.array(range(array.shape[0]))
    y = np.array(range(array.shape[1]))

    # define new grid
    xnew = np.linspace(0, x.shape[0], new_size)
    ynew = np.linspace(0, y.shape[0], new_size)

    if array.ndim > 2:
        new = np.zeros((new_size,new_size,array.shape[-1]))

        for dd in range(array.shape[-1]):
            f = interpolate.interp2d(x, y, array[...,dd], kind=kind)
            new[...,dd] = f(xnew,ynew)

        return new    
    else:
        f = interpolate.interp2d(x, y, array, kind=kind)
        return f(xnew,ynew)

class FindFiles():

    def __init__(self, directory, extension, exclude=None):

        self.directory = directory
        self.extension = extension
        self.exclude = exclude
        self.files = []

        for filename in self.find_files(self.directory, f'*{self.extension}'):
            self.files.append(filename)

        self.files.sort()

        if isinstance(self.exclude, str):
            self.files = get_file_from_substring([], self.files, exclude=self.exclude)

    @staticmethod
    def find_files(directory, pattern):
        for root, dirs, files in os.walk(directory):
            for basename in files:
                if fnmatch.fnmatch(basename, pattern):
                    filename = os.path.join(root, basename)
                    yield filename
