import csv
import fnmatch
import json
import math
import matplotlib.colors as mcolors
from matplotlib import cm
import nibabel as nb
from nilearn import signal
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
import itertools

opj = os.path.join
pd.options.mode.chained_assignment = None # disable warning thrown by string2float
warnings.filterwarnings("ignore")

# Define a function 'pairwise' that iterates over all pairs of consecutive items in a list
def pairwise(l1):
    # Create an empty list 'temp' to store the pairs
    temp = []

    # Iterate through the list elements up to the second-to-last element
    for i in range(len(l1) - 1):
        # Get the current element and the next element in the list
        current_element, next_element = l1[i], l1[i + 1]

        # Create a tuple 'x' containing the current and next elements
        x = (current_element, next_element)

        # Append the tuple 'x' to the 'temp' list
        temp.append(x)

    # Return the list of pairs
    return temp

def normalize(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))

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

    if ops in ["and","&","&&"]:
        return operator.and_
    elif ops in ["or","|","||"]:
        return operator.or_
    elif ops in ["is not","!="]:
        return operator.ne
    elif ops in ["is","==","="]:
        return operator.eq
    elif ops in ["gt",">"]:
        return operator.gt
    elif ops in ["lt","<"]:
            return operator.lt
    elif ops in ["ge",">="]:
        return operator.ge
    elif ops in ["le","<="]:
            return operator.le
    elif ops in ["x","*"]:
        return operator.mul
    elif ops == "/":
        return operator.truediv
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
        idx = np.nanargmin((np.abs(array-value)))
        return idx, array[idx]
    else:
        
        # check nan indices
        nans = np.isnan(array)

        # initialize output
        idx = np.full_like(array, np.nan)

        # loop through values in array
        for qq,ii in enumerate(array):

            # don't do anything if value is nan
            if not nans[qq]:
                idx[qq] = np.abs(ii-value)
        
        # sort
        idx = np.argsort(idx)

        # return everything
        if return_nr == "all":
            idc_list = idx.copy()
        else:
            # return closest X values
            idc_list = idx[:return_nr]
        
        return idc_list, array[idc_list]

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

def string2list(string_array, make_float=False):
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

        if make_float:
            new = [float(ii) for ii in new]
            
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
        new = [float(i.strip(",")) for i in new]
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

    if isinstance(matcher, str):
        matcher = [matcher]

    new_list = []
    for ii in ref_list:
        comps = split_bids_components(ii)

        # loop through elements in 'matcher' list
        search_for = [f"{ii}-{comps[ii]}" for ii in matcher]
        ff = get_file_from_substring(search_for, search_list, return_msg="None")

        if ff != None:
            if ff == search_list:
                raise ValueError(f"Output list is equal to input list with identifier '{matcher}'. Please use unique identifier")
            new_list.append(ff)

    return new_list

def get_unique_ids(df, id=None, sort=True, as_int=False, drop_na=True):
    try:
        df = df.reset_index()
    except:
        pass

    if not isinstance(id, str):
        raise ValueError(f"Please specify a identifier from the dataframe")
    
    try:
        a = df[id].values
        if not sort:
            indexes = np.unique(a, return_index=True)[1]
            ret_list =  [a[index] for index in sorted(indexes)]
        else:
            ret_list = list(np.unique(a))

        # https://stackoverflow.com/a/50297200
        if drop_na:
            ret_list = [x for x in ret_list if x == x]

        if as_int:
            ret_list = [int(i) for i in ret_list]
        return ret_list
        
    except Exception as e:
        raise RuntimeError(f"Could not find '{id}' in {list(df.columns)}")
    
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

    if isinstance(exclude, str):
        exclude = [exclude]

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
                if isinstance(exclude, list):
                    if not any(x in f for x in exclude):
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

            if isinstance(exclude, list):
                exl_list = []
                for f in match_list:
                    if not any(x in f for x in exclude):
                        exl_list.append(f)
                
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
        matrix = np.hstack((genaff[key][0:9].reshape(3, 3), genaff[key][9:].reshape(3, 1)))
        matrix = np.vstack([matrix, [0, 0, 0, 1]])
    elif mat.endswith(".txt"):
        matrix = np.loadtxt(mat)

        
    if invert == True:
        matrix = np.linalg.inv(matrix)

    return matrix

def ants_truncate_intensities(
    in_file, 
    out_file, 
    lower=0.01, 
    upper=0.99, 
    n_bins=256):

    import nibabel as nb
    import os

    if not isinstance(in_file, str):
        raise TypeError(f"Input must be a string pointing to a nifti file or a nb.Nifti1Image-object, not '{type(in_file)}'")
    else:
        dims = nb.load(in_file).header["dim"][0]

    if not isinstance(out_file, str):
        out_file = os.path.abspath(in_file)
        raise TypeError(f"out_file must be a string, not '{type(out_file)}'")
    else:
        out_file = os.path.abspath(out_file)

    cmd = f"ImageMath {dims} {out_file} TruncateImageIntensity {in_file} {lower} {upper} {n_bins}"
    
    # print command if verb = True
    print(cmd)
    os.system(cmd)
    
    return out_file

def ants_to_spm_moco(affine, deg=False, convention="SPM"):

    """SPM output = x [LR], y [AP], z [SI], rx, ry, rz. ANTs employs an LPS system, so y value should be switched"""
    dx, dy, dz = affine[9:]

    if convention == "SPM":
        dy = reverse_sign(dy)

    rot_x = np.arcsin(affine[6])
    cos_rot_x = np.cos(rot_x)
    rot_y = np.arctan2(affine[7] / cos_rot_x, affine[8] / cos_rot_x)
    rot_z = np.arctan2(affine[3] / cos_rot_x, affine[0] / cos_rot_x)

    if deg:
        rx,ry,rz = np.degrees(rot_x),np.degrees(rot_y),np.degrees(rot_z)
    else:
        rx,ry,rz = rot_x,rot_y,rot_z

    moco_pars = np.array([dx,dy,dz,rx,ry,rz])
    return moco_pars

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

def get_vertex_nr(subject, as_list=False, debug=False, fs_dir=None):

    if not isinstance(fs_dir, str):
        fs_dir = os.environ.get("SUBJECTS_DIR")

    n_verts_fs = []
    for i in ['lh', 'rh']:
        
        surf = opj(fs_dir, subject, 'surf', f'{i}.white')
        verbose(surf, debug)
        if not os.path.exists(surf):
            raise FileNotFoundError(f"Could not find file '{surf}'")
        
        try:
            verts = nb.freesurfer.io.read_geometry(surf)[0].shape[0]
        except:
            annot = opj(fs_dir, subject, 'label', f'{i}.aparc.a2009s.annot')
            verts = nb.freesurfer.io.read_annot(annot)[0].shape[0]
            
        n_verts_fs.append(verts)

    if as_list:
        return n_verts_fs
    else:
        return sum(n_verts_fs)

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
        self.data = pd.read_csv(self.infofile)
        
        # try to set the index to hemi. It will throw an error if you want to set the index while there already is an index.
        # E.g., initially we will set the index to 'hemi'. If we then later on read in that file again, the index is already 
        # set
        try:
            self.data = self.data.set_index('hemi')
        except:
            pass
            
        if hemi.lower() in ["lh","l","left"]:
            self.hemi = "L"
        elif hemi.lower() in ["rh","r","right"]:
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
                return {
                    "lh": [self.data[ii]['L'] for ii in ['x', 'y', 'size', 'beta', 'baseline', 'r2']],
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

def convert_to_rgb(color, as_integer=False):
    if isinstance(color, tuple):
        (R,G,B) = color
    elif isinstance(color, str):
        if len(color) == 1:
            color = mcolors.to_rgb(color)
        else:
            color = ImageColor.getcolor(color, "RGB")
            
        (R,G,B) = color
    
    if not as_integer:
        rgb = []
        for v in [R,G,B]:
            if v>1:
                v /= 255
            rgb.append(v)
        R,G,B = rgb
    else:
        rgb = []
        for v in [R,G,B]:
            if v<=1:
                v = int(v*255)
            rgb.append(v)
        R,G,B = rgb
        
    return (R,G,B)

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
    
    # convert input to RGB
    R,G,B = convert_to_rgb(color)

    colors = [(R,G,B,c) for c in np.linspace(0,1,100)]
    cmap = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=5)

    return cmap

def find_missing(lst):
    return [i for x, y in zip(lst, lst[1:])
        for i in range(x + 1, y) if y - x > 1]
        
def make_between_cm(
    col1,
    col2,
    as_list=False,
    **kwargs):

    input_list = [col1,col2]

    # scale to 0-1
    col_list = []
    for color in input_list:

        scaled_color = convert_to_rgb(color)
        col_list.append(scaled_color)

    cm = mcolors.LinearSegmentedColormap.from_list("", col_list, **kwargs)

    if as_list:
        return [mcolors.rgb2hex(cm(i)) for i in range(cm.N)]
    else:
        return cm

def make_stats_cm(
    direction, 
    lower_neg=(51,0,248),
    upper_neg=(151,253,253), 
    lower_pos=(217,36,36),
    upper_pos=(251,255,72),
    invert=False,
    ):

    if direction not in ["pos","neg"]:
        raise ValueError(f"direction must be one of 'pos' or 'neg', not '{direction}'")
    
    if direction == "pos":
        input_list = [lower_pos,upper_pos]
    else:
        input_list = [lower_neg, upper_neg]

    if invert:
        input_list = input_list[::-1]

    # scale to 0-1
    col_list = []
    for color in input_list:
        scaled_color = convert_to_rgb(color)
        col_list.append(scaled_color)

    return mcolors.LinearSegmentedColormap.from_list("", col_list)

def percent_change(
    ts, 
    ax, 
    nilearn=False, 
    baseline=20,
    prf=False,
    dm=None
    ):
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
    
    if prf:
        from linescanning import prf

        # format data
        if ts.ndim == 1:
            ts = ts[...,np.newaxis]

        # read design matrix
        if isinstance(dm, str):
            dm = prf.read_par_file(dm)

        # calculate mean
        avg = np.mean(ts, axis=ax)
        ts *= (100/avg)

        # find points with no stimulus
        timepoints_no_stim = prf.baseline_from_dm(dm)

        # find timecourses with no stimulus
        if ax == 0:
            med_bsl = ts[timepoints_no_stim,:]
        else:
            med_bsl = ts[:,timepoints_no_stim]

        # calculat median over baseline
        median_baseline = np.median(med_bsl, axis=ax)

        # shift to zero
        ts -= median_baseline

        return ts
    else:
        if nilearn:
            if ax == 0:
                psc = signal._standardize(ts, standardize='psc')
            else:
                psc = signal._standardize(ts.T, standardize='psc').T
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

def unique_combinations(elements, l=2):
    """
    Precondition: `elements` does not contain duplicates.
    Postcondition: Returns unique combinations of length 2 from `elements`.

    >>> unique_combinations(["apple", "orange", "banana"])
    [("apple", "orange"), ("apple", "banana"), ("orange", "banana")]
    """
    return list(itertools.combinations(elements, l))

def select_from_df(
    df, 
    expression=None, 
    index=True, 
    indices=None, 
    match_exact=True
    ):
    """select_from_df

    Select a subset of a dataframe based on an expression. Dataframe should be indexed by the variable you want to select on or have the variable specified in the expression argument as column name. If index is True, the dataframe will be indexed by the selected variable. If indices is specified, the dataframe will be indexed by the indices specified through a list (only select the elements in the list) or a `range`-object (select within range).

    Parameters
    ----------
    df: pandas.DataFrame
        input dataframe
    expression: str, optional
        what subject of the dataframe to select, by default None. The expression must consist of a variable name and an operator. The operator can be any of the following: '=', '>', '<', '>=', '<=', '!=', separated by spaces. You can also change 2 operations by specifying the `&`-operator between the two expressions. If you want to use `indices`, specify `expression="ribbon"`. 
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

    # not the biggest fan of a function within a function, but this allows easier translation of expressions/operators
    def sort_expressions(expression):
        expr_ = expression.split(" ")
        if len(expr_)>3:
            for ix,i in enumerate(expr_):
                try:
                    _ = str2operator(i)
                    break
                except:
                    pass

            col1 = " ".join(expr_[:ix])
            val1 = " ".join(expr_[(ix+1):])
            operator1 = expr_[ix]
        else:
            col1,operator1,val1 = expr_

        return col1,operator1,val1
    
    if not isinstance(expression, (str,tuple,list)):
        raise ValueError(f"Please specify expressions to apply to the dataframe. Input is '{expression}' of type ({type(expression)})")
    
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

                # find operator index
                col1,operator1,val1 = sort_expressions(expressions[0])

                # convert to operator function
                ops1 = str2operator(operator1)
                
                # use dtype of whatever dtype the colum is
                search_value = np.array([val1], dtype=type(sub_df[col1].values[0]))
                sub_df = sub_df.loc[ops1(sub_df[col1], search_value[0])]
                
            if len(expressions) == 2:
                col1,operator1,val1 = sort_expressions(expressions[0])
                col2,operator2,val2 = sort_expressions(expressions[1])

                main_ops = str2operator(operators[0])
                ops1 = str2operator(operator1)
                ops2 = str2operator(operator2)

                # check if we should interpret values invididually as integers
                search_value1 = np.array([val1], dtype=type(sub_df[col1].values[0]))[0]
                search_value2 = np.array([val2], dtype=type(sub_df[col2].values[0]))[0]

                sub_df = sub_df.loc[main_ops(ops1(sub_df[col1], search_value1), ops2(sub_df[col2], search_value2))]

        # first check if we should do indexing
        if index != None:
            # then check if we actually have something to index
            if reindex:
                if idc[0] != None:
                    sub_df = sub_df.set_index(idc)

        if sub_df.shape[0] == 0:
            raise ValueError(f"The following expression(s) resulted in an empty dataframe: {expression}")
        
        return sub_df

def multiselect_from_df(df, expression=[]):

    if not isinstance(expression, list):
        raise TypeError(f"expression must be list of tuples (see docs utils.select_from_df), not {type(expression)}")
    
    if len(expression) == 0:
        raise ValueError(f"List is empty")
    
    start_df = df.copy()
    for expr in expression:
        df = select_from_df(df, expression=expr)

    return df

def split_bids_components(fname, entities=False):

    comp_list = fname.split('_')
    comps = {}
    
    ids = ['sub', 'ses', 'task', 'acq', 'rec', 'run', 'space', 'hemi', 'model', 'stage', 'desc', 'vox']

    full_entities = [
        "subject",
        "session",
        "task",
        "reconstruction",
        "acquisition",
        "run"
    ]
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

        if entities:
            return comps, full_entities
        else:
            return comps
    else:
        raise ValueError(f"Could not find any element of {ids} in {fname}")

class BIDSFile():

    def __init__(self, bids_file):
        self.bids_file = os.path.abspath(bids_file)

    def get_bids_basepath(self, *args):
        return self._get_bids_basepath(self.bids_file, *args)
    
    def get_bids_root(self, *args):
        return self._get_bids_root(self.bids_file, *args)

    def get_bids_workbase(self, *args):
        return self._get_bids_workbase(self.bids_file, *args) 

    def get_bids_workflow(self, **kwargs):
        return assemble_fmriprep_wf(self.bids_file, **kwargs)   

    # def get_bids_root(self):
    @staticmethod
    def _get_bids_basepath(file, pref="sub"):
        sp = file.split(os.sep)
        for i in sp:
            if i.startswith(pref) and not i.endswith('.nii.gz'):
                base_path = os.sep.join(sp[sp.index(i)+1:-1])
                break

        return base_path
    
    # def get_bids_root(self):
    @staticmethod
    def _get_bids_workbase(file, pref="sub"):
        sp = file.split(os.sep)
        for i in sp:
            if i.startswith(pref) and not i.endswith('.nii.gz'):
                base_path = os.sep.join(sp[sp.index(i):-2])
                break

        return base_path    
    
    @staticmethod
    def _get_bids_root(file, pref="sub"):
        sp = file.split(os.sep)
        for i in sp:
            if i.startswith(pref) and not i.endswith('.nii.gz'):
                bids_root = os.sep.join(sp[:sp.index(i)])
                break

        return bids_root
    
    def get_bids_ids(self, **kwargs):
        return split_bids_components(self.bids_file, **kwargs)

def get_ids(func_list, bids="task"):

    ids = []
    if isinstance(func_list, list):
        for ff in func_list:
            if isinstance(ff, str):
                bids_comps = split_bids_components(ff)
                if bids in list(bids_comps.keys()):
                    ids.append(bids_comps[bids])
        
    if len(ids) > 0:
        ids = list(np.unique(np.array(ids)))

        return ids
    else:
        return []
        
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
    geom = first_line.intersection(second_line)
    
    try:
        if isinstance(geom, geometry.multipoint.MultiPoint):
            # multiple coordinates
            coords = [i.coords._coords for i in list(geom.geoms)]
        elif isinstance(geom, geometry.point.Point):
            # single coordinate
            coords = [geom.coords._coords]
        elif isinstance(geom, geometry.collection.GeometryCollection):
            # coordinates + line segments
            mapper = geometry.mapping(geom)
            coords = []
            for el in mapper["geometries"]:
                coor = np.array(el["coordinates"])
                if coor.ndim > 1:
                    coor = coor[0]
                coords.append(coor[np.newaxis,...]) # to make indexing same as above
        else:
            raise ValueError(f"Can't deal with output of type {type(geom)}")
            # coords = geom
    except:
        raise ValueError("Could not find intersection between curves..")
    
    return coords

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

def assemble_fmriprep_wf(bold_path, wf_only=False):
    """assemble_fmriprep_wf

    Parses the bold file into a workflow name for fMRIPrep into its constituents to recreate a filename. Searches for the following keys: `['ses', 'task', 'acq', 'run']`.

    Parameters
    ----------
    bold_path: str
        Path to bold-file
    wf_only: bool, optional
        If `sub` tag is found in `bold_path`, we can reconstruct the full workflow folder including preceding `single_subject_<sub_id>_wf`. If you do not want this, set `wf_only` to **False**.

    Returns
    ----------
    str
        filename based on constituent file parts

    Example
    ----------
    >>> from linescanning.utils import disassemble_fmriprep_wf
    >>> bold_file = "sub-008_ses-2_task-SRFi_acq-3DEPI_run-1_desc-preproc_bold.nii.gz"
    >>> wf_name = assemble_fmriprep_wf(bold_file)
    >>> wf_name
    >>> 'single_subject_008_wf/func_preproc_ses_2_task_SRFi_run_1_acq_3DEPI_wf'

    >>> # workflow name only
    >>> wf_name = assemble_fmriprep_wf(bold_file, wf_only=True)
    >>> wf_name
    >>> 'func_preproc_ses_2_task_SRFi_run_1_acq_3DEPI_wf'
    """
    bids_comps = split_bids_components(os.path.basename(bold_path))
    fname = ["func_preproc"]

    for tag in ['ses', 'task', 'run', 'acq']:
        if tag in list(bids_comps.keys()):
            fname.append(f"{tag}_{bids_comps[tag]}")
    
    base_dir = ""
    fname = "_".join(fname)+"_wf"
    if 'sub' in list(bids_comps.keys()):
        base_dir = f"single_subject_{bids_comps['sub']}_wf"

        if wf_only:
            return fname
        else:
            return opj(base_dir, fname)
    else:
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

    def __init__(self, directory, extension, exclude=None, maxdepth=None, filters=None):

        self.directory = directory
        self.extension = extension
        self.exclude = exclude
        self.maxdepth = maxdepth
        self.filters = filters
        self.files = []

        for filename in self.find_files(self.directory, f'*{self.extension}', maxdepth=self.maxdepth):
            if not filename.startswith('._'):
                self.files.append(filename)

        self.files.sort()

        if isinstance(self.exclude, (str,list)) or isinstance(self.filters, (list, str)):
            if isinstance(self.filters, str):
                self.filters =  [self.filters]
            elif isinstance(self.filters, list):
                pass
            else:
                self.filters = []

            self.files = get_file_from_substring(self.filters, self.files, exclude=self.exclude)

    @staticmethod
    def find_files(directory, pattern, maxdepth=None):
        
        start = None
        if isinstance(maxdepth, int):
            start = 0

        for root, dirs, files in os.walk(directory, followlinks=True):

            for basename in files:
                if fnmatch.fnmatch(basename, pattern):
                    filename = os.path.join(root, basename)
                    yield filename

            if isinstance(maxdepth, int):
                if start > maxdepth:
                    break 

            if isinstance(start, int):
                start += 1 

def round_decimals_down(number:float, decimals:int=2):
    """
    Returns a value rounded down to a specific number of decimal places.
    see: https://kodify.net/python/math/round-decimals/#round-decimal-places-up-and-down-round
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)

    factor = 10 ** decimals
    return math.floor(number * factor) / factor

def round_decimals_up(number:float, decimals:int=2):
    """
    Returns a value rounded up to a specific number of decimal places.
    see: https://kodify.net/python/math/round-decimals/#round-decimal-places-up-and-down-round
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor

def make_polar_cmap():

    top = cm.get_cmap('hsv', 256)
    bottom = cm.get_cmap('hsv', 256)

    newcolors = np.vstack((top(np.linspace(0, 1, 256)), bottom(np.linspace(0, 1, 256))))
    cmap = mcolors.ListedColormap(newcolors, name='hsvx2')

    return cmap

# https://stackoverflow.com/a/50692782
def paste_slices(tup):
  pos, w, max_w = tup
  wall_min = max(pos, 0)
  wall_max = min(pos+w, max_w)
  block_min = -min(pos, 0)
  block_max = max_w-max(pos+w, max_w)
  block_max = block_max if block_max != 0 else None
  return slice(wall_min, wall_max), slice(block_min, block_max)

def paste(large, small, loc=(0,0)):
  loc_zip = zip(loc, small.shape, large.shape)
  large_slices, small_slices = zip(*map(paste_slices, loc_zip))
  large[large_slices] = small[small_slices]


def SDT(hits, misses, fas, crs):
    from scipy import stats
    Z = stats.norm.ppf

    """ returns a dict with d-prime measures given hits, misses, false alarms, and correct rejections"""
    # Floors an ceilings are replaced by half hits and half FA's
    half_hit = 0.5 / (hits + misses)
    half_fa = 0.5 / (fas + crs)
 
    # Calculate hit_rate and avoid d' infinity
    hit_rate = hits / (hits + misses)
    if hit_rate == 1: 
        hit_rate = 1 - half_hit
    if hit_rate == 0: 
        hit_rate = half_hit
 
    # Calculate false alarm rate and avoid d' infinity
    fa_rate = fas / (fas + crs)
    if fa_rate == 1: 
        fa_rate = 1 - half_fa
    if fa_rate == 0: 
        fa_rate = half_fa
 
    # Return d', beta, c and Ad'
    out = {}
    out['d'] = Z(hit_rate) - Z(fa_rate)
    out['beta'] = math.exp((Z(fa_rate)**2 - Z(hit_rate)**2) / 2)
    out['c'] = -(Z(hit_rate) + Z(fa_rate)) / 2
    out['Ad'] = stats.norm.cdf(out['d'] / math.sqrt(2))
    out['hit'] = hit_rate
    out['fa'] = fa_rate
    
    return(out)

def update_kwargs(kwargs, el, val, force=False):
    if not force:
        if not el in list(kwargs.keys()):
            kwargs[el] = val
    else:
        kwargs[el] = val
        
    return kwargs

def create_sinewave(
    amplitude=1,
    frequency=1,
    phase=0,
    sampling_rate=100,
    duration=1
    ):

    """create_sinewave

    Parameters
    ----------
    amplitude: int,float
        Amplitude of the wave. Default = 1
    frequency: int,float
        Frequency of the wave in Hertz. Default = 1
    phase: int,float
        Phase shift of the wave in radians
    sampling_rate: int
        Number of samples per second
    duration: int
        Duration of the wave in seconds

    Returns
    ----------
    A tuple with first element the numpy array containing the sine wave. The second element is the time axis

    Example
    ----------
    >>> from linescanning import utils
    >>> wave,time = utils.create_sinewave()

    >>> from linescanning import utils
    >>> wave,time = utils.create_sinewave(
    >>>     frequency=4,
    >>>     amplitude=0.2
    >>> )
    """

    time = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
    sine_wave = amplitude * np.sin(2*np.pi*frequency*time+phase)
    return sine_wave,time
