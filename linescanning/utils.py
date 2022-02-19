import csv
import json
from . import prf, glm, plotting
import lmfit
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nibabel as nb
import nideconv as nd
import numpy as np
import operator
import os
import pandas as pd
from PIL import ImageColor
from prfpy import stimulus
import random
from scipy import io, stats
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

def str2operator(ops):

    if ops == "and" or ops == "&" or ops == "&&":
        return operator.and_
    elif ops == "or" or ops == "|" or ops == "||":
        return operator.or_
    elif ops == "is not" or ops == "!=":
        return operator.ne
    elif ops == "is" or ops == "==" or ops == "=":
        return operator.eq
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
                return [f for f in match_list if exclude not in f]
            else:
                return match_list
            # return match_list
        else:
            if return_msg == "error":
                raise FileNotFoundError(f"Could not find file with filters: {filt} in {path}")
            else:
                return None

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

def percent_change(ts, ax):
    """convert timeseries to percent signal change via the nilearn method"""

    return (ts / np.expand_dims(np.mean(ts, ax), ax) - 1) * 100    

def select_from_df(df, expression="run = 1", index=True, indices=None):

    if expression == "ribbon":
        
        if indices == None:
            raise ValueError("You want specific voxels from DataFrame, but none were specified. Please specify indices=[start,stop]")

        return df.iloc[:,indices[0]:indices[1]]
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

        if isinstance(expression, tuple) or isinstance(expression, list):

            expressions = expression[::2]
            operators = expression[1::2]

            if len(expressions) == 1:

                col1,operator1,val1 = expressions[0].split()
                ops1 = str2operator(operator1)
                
                if len(val1) == 1:
                    val1 = int(val1)
                    
                sub_df = sub_df.loc[ops1(sub_df[col1], val1)]
                
            if len(expressions) == 2:
                col1,operator1,val1 = expressions[0].split()
                col2,operator2,val2 = expressions[1].split()

                main_ops = str2operator(operators[0])
                ops1 = str2operator(operator1)
                ops2 = str2operator(operator2)

                # check if we should interpret values invididually as integers
                if len(val1) == 1:
                    val1 = int(val1)

                if len(val2) == 1:
                    val2 = int(val2)

                sub_df = sub_df.loc[main_ops(ops1(sub_df[col1], val1), ops2(sub_df[col2], val2))]

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
    
    ids = ['ses', 'task', 'acq', 'rec', 'sub', 'desc', 'run']
    for el in comp_list:
        for i in ids:
            if i in el:
                comp = el.split('-')[-1]
                if i == "run":
                    comp = int(comp)

                comps[i] = comp

    if len(comps) != 0:
        return comps
    else:
        print(f"Could not find any element of {ids} in {fname}")


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

    def __init__(self, subject, derivatives=None, cx_dir=None, prf_dir=None, ses=1, analysis_yaml=None, hemi="lh", settings=None, model="gauss", correct_screen=True):

        self.subject        = subject
        self.derivatives    = derivatives
        self.cx_dir         = cx_dir
        self.prf_dir        = prf_dir
        self.prf_ses        = ses
        self.hemi           = hemi
        self.model          = model
        self.analysis_yaml  = analysis_yaml
        self.correct_screen = correct_screen

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
            self.design_fn      = get_file_from_substring("vis_design.mat", self.prf_dir)
            self.design_matrix  = io.loadmat(self.design_fn)['stim']
            self.func_data_lr   = np.load(get_file_from_substring("avg_bold_hemi-LR.npy", self.prf_dir))
            self.func_data_l    = np.load(get_file_from_substring("avg_bold_hemi-L.npy", self.prf_dir))
            self.func_data_r    = np.load(get_file_from_substring("avg_bold_hemi-R.npy", self.prf_dir))
        # load specific analysis file
        if self.analysis_yaml != None:
            self.settings = yaml.safe_load(self.analysis_yaml)

            with open(self.analysis_yaml) as file:
                self.settings = yaml.safe_load(file)            
        
        # load the most recent analysis file. This is fine for screens/stimulus information
        if settings == "recent":
            self.analysis_yaml = opj(self.prf_dir, sorted([ii for ii in os.listdir(self.prf_dir) if "desc-settings" in ii])[-1])
        
            with open(self.analysis_yaml) as file:
                self.settings = yaml.safe_load(file)

        if self.cx_dir != None:
            self.vert_fn        = get_file_from_substring([self.model, "best_vertices.csv"], self.cx_dir)
            self.vert_info      = VertexInfo(self.vert_fn, subject=self.subject, hemi=self.hemi)
        
        # fetch target vertex parameters
        if hasattr(self, "vert_info"):
            self.target_params = self.return_prf_params(hemi=self.hemi)
            self.target_vertex = self.return_target_vertex(hemi=self.hemi)

        # create pRF if settings were specified
        if hasattr(self, "settings"):
            self.prf_stim = stimulus.PRFStimulus2D(screen_size_cm=self.settings['screen_size_cm'], screen_distance_cm=self.settings['screen_distance_cm'], design_matrix=self.design_matrix,TR=self.settings['TR'])
            self.prf_array = prf.make_prf(self.prf_stim, size=self.target_params[2], mu_x=self.target_params[0], mu_y=self.target_params[1])

        try:
            self.normalization_params_df    = pd.read_csv(get_file_from_substring([f"hemi-{self.hemi_tag}", "normalization", "csv"], self.cx_dir), index_col=0)
            self.normalization_params       = np.load(get_file_from_substring([f"hemi-{self.hemi_tag}", "normalization", "npy"], self.cx_dir))

            if self.correct_screen:
                self.normalization_params = self.normalization_params*1.08
                
        except:
            self.normalization_params_df    = None
            self.normalization_params       = None

        if self.prf_dir != None:
            self.modelling      = prf.pRFmodelFitting(self.func_data_lr,
                                                      design_matrix=self.design_matrix,
                                                      settings=self.analysis_yaml)
                                
            self.modelling.load_params(self.normalization_params, model=self.model, stage="iter")

    def return_prf_params(self, hemi="lh"):
        """return pRF parameters from :class:`linescanning.utils.VertexInfo`"""
        return self.vert_info.get('prf', hemi=hemi)

    def return_target_vertex(self, hemi="lh"):
        """return the vertex ID of target vertex"""
        return self.vert_info.get('index', hemi=hemi)

    def target_prediction_prf(self, xkcd=False, line_width=1):
        _, self.prediction, _ = self.modelling.plot_vox(vox_nr=self.target_vertex, 
                                                        model=self.model, 
                                                        stage='iter', 
                                                        make_figure=True, 
                                                        xkcd=xkcd,
                                                        title='pars',
                                                        transpose=True, 
                                                        line_width=line_width)

class CurveFitter():
    """CurveFitter

    Simple class to perform a quick curve fitting procedure on `y_data`. You can either specify your own function with `func`, or select a polynomial of order `order` (currently up until 3rd-order is included). Internally uses `lmfit.Model` to perform the fitting, allowing for access to confidence intervals.

    Parameters
    ----------
    y_data: np.ndarray
        Data-points to perform fitting on
    x: np.ndarray, optional
        Array describing the x-axis, by default None. If `None`, we'll take `np.arange` of `y_data.shape[0]`. 
    func: <function> object, optional
        Use custom function describing the behavior of the fit, by default None. If `none`, we'll assume either a linear or polynomial fit (up to 3rd order)
    order: str, int, optional
        Order of polynomial fit, by default "3rd". Can either be '1st'|1, '2nd'|2, or '3rd'|3
    verbose: bool, optional
        Print summary of fit, by default True
    interpolate: str, optional
        Method of interpolation for an upsampled version of the predicted data (default = 1000 samples)

    Raises
    ----------
    NotImplementedError
        If `func=None` and no valid polynomial order (see above) was specified

    Example
    ----------
    >>> # imports
    >>> from linescanning import utils
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> # define data points
    >>> data = np.array([5.436, 5.467, 5.293, 0.99 , 2.603, 1.902, 2.317])
    >>> # create instantiation of CurveFitter
    >>> cf = utils.CurveFitter(data, order=3, verbose=False)
    >>> # initiate figure with axis to be fed into LazyPlot
    >>> fig, axs = plt.subplots(figsize=(8,8))
    >>> # plot original data points
    >>> axs.plot(cf.x, data, 'o', color="#DE3163", alpha=0.6)
    >>> # plot upsampled fit with 95% confidence intervals as shaded error
    >>> plotting.LazyPlot(cf.y_pred_upsampled,
    >>>                   xx=cf.x_pred_upsampled,
    >>>                   error=cf.ci_upsampled,
    >>>                   axs=axs,
    >>>                   color="#cccccc",
    >>>                   x_label="x-axis",
    >>>                   y_label="y-axis",
    >>>                   title="Curve-fitting with polynomial (3rd-order)",
    >>>                   set_xlim_zero=False,
    >>>                   sns_trim=True,
    >>>                   line_width=1,
    >>>                   font_size=20)
    >>> plt.show()
    """

    def __init__(self, y_data, x=None, func=None, order="3rd", verbose=True, interpolate='linear'):

        self.y_data         = y_data
        self.func           = func
        self.order          = order
        self.x              = x
        self.verbose        = verbose
        self.interpolate    = interpolate

        if self.func == None:
            if order == "1st" or order == 1:
                self.func = self.first_order
            elif order == "2nd" or order == 2:
                self.func = self.second_order
            elif order == "3rd" or order == 3:
                self.func = self.third_order
            else:
                raise NotImplementedError(f"polynomial of order {order} is not available")

        if self.x == None:
            self.x = np.arange(self.y_data.shape[0])

        self.pmodel = lmfit.Model(self.func)
        self.params = self.pmodel.make_params(a=1, b=2, c=1, d=1)
        self.result = self.pmodel.fit(self.y_data, self.params, x=self.x)

        if self.verbose:
            print(self.result.fit_report())

        # create predictions & confidence intervals that are compatible with LazyPlot
        self.y_pred             = self.result.best_fit
        self.x_pred_upsampled   = np.linspace(self.x[0], self.x[-1], 1000)
        self.y_pred_upsampled   = self.result.eval(x=self.x_pred_upsampled)
        self.ci                 = self.result.eval_uncertainty()
        self.ci_upsampled       = glm.resample_stim_vector(self.ci, len(self.x_pred_upsampled), interpolate=self.interpolate)

    @staticmethod
    def first_order(x, a, b):
        return a * x + b
    
    @staticmethod
    def second_order(x, a, b, c):
        return a * x + b * x**2 + c
    
    @staticmethod
    def third_order(x, a, b, c, d):
	    return (a * x) + (b * x**2) + (c * x**3) + d


class NideconvFitter():
    """NideconvFitter

    Wrapper class around :class:`nideconv.GroupResponseFitter` to promote reprocudibility, avoid annoyances with pandas indexing, and flexibility when performing multiple deconvolutions in an analysis. Works fluently with :class:`linescanning.dataset.Dataset` and :func:`linescanning.utils.select_from_df`. Because our data format generally involved ~720 voxels, we can specify the range which represents the grey matter ribbon around our target vertex, e.g., `[355,364]`, and select the subset of the main functional dataframe to use as input for this class (see also example).

    Main inputs are the dataframe with fMRI-data, the onset timings, followed by specific settings for the deconvolution. Rigde-regression is not yet available as method, because 2D-dataframes aren't supported yet. This is a work-in-progress.

    Parameters
    ----------
    func: pd.DataFrame
        Dataframe as per the output of :func:`linescanning.dataset.Datasets.fetch_fmri()`, containing the fMRI data indexed on subject, run, and t.
    onsets: pd.DataFrame
        Dataframe as per the output of :func:`linescanning.dataset.Datasets.fetch_onsets()`, containing the onset timings data indexed on subject, run, and event_type.
    TR: float, optional
        Repetition time, by default 0.105. Use to calculate the sampling frequency (1/TR)
    confounds: pd.DataFrame, optional
        Confound dataframe with the same format as `func`, by default None
    basis_sets: str, optional
        Type of basis sets to use, by default "fourier". Should be 'fourier' or 'fir'.
    fit_type: str, optional
        Type of minimization strategy to employ, by default "ols". Should be 'ols' or 'ridge' (though the latter isn't implemented properly yet)
    n_regressors: int, optional
        Number of regressors to use, by default 9
    add_intercept: bool, optional
        Fit the intercept, by default False
    verbose: bool, optional
        _description_, by default False
    lump_events: bool, optional
        If ple are  in the onset dataframe, we can lump the events together and consider all onset times as 1 event, by default False
    interval: list, optional
        Interval to fit the regressors over, by default [0,12]

    Example
    ----------
    >>> from linescanning import utils, dataset
    >>> func_file
    >>> ['sub-003_ses-3_task-SR_run-3_bold.mat',
    >>> 'sub-003_ses-3_task-SR_run-4_bold.mat',
    >>> 'sub-003_ses-3_task-SR_run-6_bold.mat']
    >>> ribbon = [356,363]
    >>> window = 19
    >>> order = 3
    >>> 
    >>> ## window 5 TR poly 2
    >>> data_obj = dataset.Dataset(func_file,
    >>>                            deleted_first_timepoints=50,
    >>>                            deleted_last_timepoints=50,
    >>>                            window_size=window,
    >>>                            high_pass=True,
    >>>                            tsv_file=exp_file,
    >>>                            poly_order=order,
    >>>                            use_bids=True)
    >>> 
    >>> df_func     = data_obj.fetch_fmri()
    >>> df_onsets   = data_obj.fetch_onsets()
    >>> 
    >>> # pick out the voxels representing the GM-ribbon
    >>> df_ribbon = utils.select_from_df(df_func, expression='ribbon', indices=ribbon)
    >>> nd_fit = utils.NideconvFitter(df_ribbon,
    >>>                               df_onsets,
    >>>                               confounds=None,
    >>>                               basis_sets='fourier',
    >>>                               n_regressors=19,
    >>>                               lump_events=False,
    >>>                               TR=0.105,
    >>>                               interval=[0,12],
    >>>                               add_intercept=True,
    >>>                               verbose=True)

    Notes
    ---------
    Several plotting options are available:

    * `plot_average_per_event`: for each event, average over the voxels present in the dataframe
    * `plot_average_per_voxel`: for each voxel, plot the response to each event
    * `plot_hrf_across_depth`: for each voxel, fetch the peak HRF response and fit a 3rd-order polynomial to the points (utilizes :class:`linescanning.utils.CurveFitter`)

    See also https://linescanning.readthedocs.io/en/latest/examples/nideconv.html for more details.
    """

    def __init__(self, func, onsets, TR=0.105, confounds=None, basis_sets="fourier", fit_type="ols", n_regressors=9, add_intercept=False, verbose=False, lump_events=False, interval=[0,12]):

        self.func           = func
        self.onsets         = onsets
        self.confounds      = confounds
        self.basis_sets     = basis_sets 
        self.fit_type       = fit_type
        self.n_regressors   = n_regressors
        self.add_intercept  = add_intercept
        self.verbose        = verbose
        self.lump_events    = lump_events
        self.TR             = TR
        self.fs             = 1/self.TR
        self.interval       = interval

        if self.lump_events:
            self.lumped_onsets = self.onsets.copy().reset_index()
            self.lumped_onsets['event_type'] = 'stim'
            self.lumped_onsets = self.lumped_onsets.set_index(['subject', 'run', 'event_type'])
            self.used_onsets = self.lumped_onsets.copy()
        else:
            self.used_onsets = self.onsets.copy()        
        
        # get the model
        self.define_model()

        # specify the events
        self.define_events()

        # fit
        self.fit()

    def timecourses_condition(self):

        # get the condition-wise timecoursee
        self.tc_condition = self.model.get_conditionwise_timecourses()

        # rename 'event type' to 'event_type' so it's compatible with utils.select_from_df
        tmp = self.tc_condition.reset_index().rename(columns={"event type": "event_type"})
        self.tc_condition = tmp.set_index(['event_type', 'covariate', 'time'])

        # get the standard error of mean
        self.tc_error = self.model.get_subjectwise_timecourses().groupby(level=['event type', 'covariate', 'time']).sem()

        # set time axis
        self.time = self.tc_condition.groupby(['time']).mean().reset_index()['time'].values

    def define_model(self):

        self.model = nd.GroupResponseFitter(self.func,
                                            self.used_onsets,
                                            input_sample_rate=self.fs,
                                            concatenate_runs=False,
                                            confounds=self.confounds, 
                                            add_intercept=False)
    
    def define_events(self):
        
        if self.verbose:
            print(f"Selected '{self.basis_sets}'-basis sets")

        # define events
        self.cond = self.used_onsets.reset_index().event_type.unique()
        self.cond = np.array(sorted([event for event in self.cond if event != 'nan']))

        # add events to model
        for event in self.cond:
            if self.verbose:
                print(f"Adding event '{event}' to model")

            self.model.add_event(str(event), 
                                 basis_set=self.basis_sets,
                                 n_regressors=self.n_regressors, 
                                 interval=self.interval)

    def fit(self):

        if self.verbose:
            print(f"Fitting with '{self.fit_type}' minimization")

        if self.fit_type.lower() == "rigde":
            raise NotImplementedError("Ridge regression doesn't work with 2D-data yet, use 'ols' instead")
        elif self.fit_type.lower() == "ols":
            pass
        else:
            raise ValueError(f"Unrecognized minimizer '{self.fit_type}'; must be 'ols' or 'ridge'")
        
        # fitting
        self.model.fit(type=self.fit_type)

        if self.verbose:
            print("Done")

    def plot_average_per_event(self, add_offset=True, axs=None, title="Average HRF", **kwargs):

        if not hasattr(self, "tc_condition"):
            self.timecourses_condition()

        self.event_avg = []
        self.event_sem = []
        for ev in self.cond:
            # average over voxels (we have this iloc thing because that allows 'axis=1'). Groupby doesn't allow this
            avg     = self.tc_condition.loc[ev].iloc[:].mean(axis=1).values
            sem     = self.tc_condition.loc[ev].iloc[:].sem(axis=1).values

            if add_offset:
                if avg[0] > 0:
                    avg -= avg[0]
                else:
                    avg += abs(avg[0])

            self.event_avg.append(avg)
            self.event_sem.append(sem)

        plotting.LazyPlot(self.event_avg,
                          xx=self.time,
                          axs=axs,
                          error=self.event_sem,
                          title=title,
                          font_size=16,
                          **kwargs)

    def plot_average_per_voxel(self, add_offset=True, axs=None, n_cols=4, wspace=0, figsize=(30,15), make_figure=True, labels=None, **kwargs):
            
        if not hasattr(self, "tc_condition"):
            self.timecourses_condition()

        cols = list(self.tc_condition.columns)
        if cols > 30:
            raise Exception(f"{len(cols)} were requested. Maximum number of plots is set to 30")

        if n_cols != None:
            # initiate figure
            fig = plt.figure(figsize=figsize)
            n_rows = int(np.ceil(len(cols) / n_cols))
            gs = fig.add_gridspec(n_rows, n_cols, wspace=wspace)

        self.all_voxels_in_event = []
        for ix, col in enumerate(cols):

            # fetch data from specific voxel for each stimulus size
            self.voxel_in_events = []
            for idc,stim in enumerate(self.cond):
                col_data = self.tc_condition[col][stim].values
                
                if add_offset:
                    if col_data[0] > 0:
                        col_data -= col_data[0]
                    else:
                        col_data += abs(col_data[0])

                self.voxel_in_events.append(col_data)

            # this one is in case we want the voxels in 1 figure
            self.all_voxels_in_event.append(self.voxel_in_events[0])
            # draw legend once
            if ix == 0:
                set_labels = labels
            else:
                set_labels = None

            if make_figure:
                # plot all stimulus sizes for a voxel
                if n_cols != None:
                    ax = fig.add_subplot(gs[ix])
                    plotting.LazyPlot(self.voxel_in_events,
                                      xx=self.time,
                                      axs=ax,
                                      title=col,
                                      labels=set_labels,
                                      font_size=16,
                                      **kwargs)

        if make_figure:
            if n_cols == None:
                if labels:
                    labels = cols.copy()
                else:
                    labels = None
                
                if axs != None:
                    plotting.LazyPlot(self.all_voxels_in_event,
                                      xx=self.time,
                                      axs=axs,
                                      labels=labels,
                                      font_size=16,
                                      **kwargs)
                else:
                    plotting.LazyPlot(self.all_voxels_in_event,
                                      xx=self.time,
                                      font_size=16,
                                      labels=labels,
                                      **kwargs)


    def plot_hrf_across_depth(self, axs=None, figsize=(8,8), markers_cmap='viridis', ci_color="#cccccc", ci_alpha=0.6, **kwargs):

        if not hasattr(self, "all_voxels_in_event"):
            self.plot_timecourses(make_figure=False)
        
        self.max_vals = np.array([np.amax(self.col_data[ii]) for ii in range(len(self.col_data))])
        cf = CurveFitter(self.max_vals, order=3, verbose=False)
        
        if not axs:
            fig,axs = plt.subplots(figsize=figsize)

        color_list = sns.color_palette(markers_cmap, len(self.max_vals))
        for ix, mark in enumerate(self.max_vals):
            axs.plot(cf.x[ix], mark, 'o', color=color_list[ix], alpha=ci_alpha)

        plotting.LazyPlot(cf.y_pred_upsampled,
                        xx=cf.x_pred_upsampled,
                        axs=axs,
                        error=cf.ci_upsampled,
                        color=ci_color,
                        font_size=16,
                        **kwargs)
