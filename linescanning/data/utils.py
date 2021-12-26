import os
import sys

class color:
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

def replace_string(fn, str1, str2, fn_sep='_'):

    import os
    opj = os.path.join

    split_name = fn.split(os.sep)[-1].split(fn_sep)
    idx = [(i, split_name.index(str1)) for i, split_name in enumerate(split_name) if str1 in split_name][0][0]
    split_name[idx] = split_name[idx].replace(split_name[idx], str2)
    new_filename = fn_sep.join(split_name)
    new_filename = opj(os.path.dirname(fn), new_filename)

    return new_filename

def string2float(string_array):
    """
string2float

This function converts a array in string representation to a regular float array. This
can happen, for instance, when you've stored a numpy array in a pandas dataframe (such
is the case with the 'normal' vector).

It starts by splitting based on empty spaces, filter these, and convert any remaining
elements to floats and returns these in an array.

Args:
    string_array    : str
                    string to be converted to a valid numpy array with float values

Example:
    In [33]: string2float('[ -7.42 -92.97 -15.28]')
    Out[33]: array([ -7.42, -92.97, -15.28])

    """


    import numpy as np

    new = string_array[1:-1].split(' ')[0:]; new = list(filter(None, new)); new = [float(i) for i in new]; new = np.array(new)

    return new

def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):

    """
errorfill

This function plots your input array (x,y) with shaded error values ('yerr'). If you have
subplots, you can specify an axis to apply it to.

Args:
    x:      array containing x-axis values (something with np.linspace or whatnot)
    y:      array containing y-axis values (something of interest)
    yerr:   array containing error values over y (e.g., SEM)
    color:  hex/RGB-code of graph color
    alpha:  transparency of error fields
    ax:     axis (i.e., subplot) to apply it to

Example:
    errorfill(x, df['mean'], df['sem'],  color='#3C9DA7', alpha_fill=0.2)

    """

    import matplotlib.pyplot as plt
    import numpy as np

    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = next(ax._get_lines.prop_cycler)['color']
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


def get_max_coordinate(in_img):

    """
get_max_coordinate

fetches the point with the maximum value given an input image_file. Useful if you want to find the
voxel with the highest value after warping a binary file with ANTs. Mind you, it outputs the VOXEL
value, not the actual index of the array. The VOXEL value = idx_array+1 to account for different in-
dexing in nifti & python.

Args:
    in_img      : str|numpy.ndarray
                string to nifti image or numpy array

Returns:
    np.ndarray  if only 1 max value was detected
    list        list of np.ndarray containing the voxel coordinates with max value

Example:
    In [33]: get_max_coordinate('sub-001_space-ses1_hemi-L_vert-875.nii.gz')
    Out[33]: array([142,  48, 222])
    In [34]: get_max_coordinate('sub-001_space-ses1_hemi-R_vert-6002.nii.gz')
    Out[34]: [array([139,  35, 228]), array([139,  36, 228])]
    """

    import nibabel as nb
    import numpy as np

    if isinstance(in_img, np.ndarray):
        img_data = in_img
    elif isinstance(in_img, nb.Nifti1Image):
        img_data = in_img.get_fdata()
    elif isinstance(in_img, str):
        img_data = nb.load(in_img).get_fdata()
    else:
        raise NotImplementedError("Unknown input type; must either be a numpy array, a nibabel Nifti1Image, or a string pointing to a nifti-image")

    max_val = img_data.max()
    coord = np.array([np.where(img_data==max_val)[i]+1 for i in range(0,3)]) #.reshape(1,3).flatten()

    if coord.shape[-1] > 1:
        l = []
        for i in np.arange(0,coord.shape[-1]):
            l.append(coord[:,i])
    else:
        l = coord[:,0]

    return l

def read_fs_reg(dat_file):

    """reads the output from bbregister (e.g., register.dat) into a numpy array"""

    import numpy as np

    with open(dat_file) as f:
        d = f.readlines()[4:-1]

        return np.array([[float(s) for s in dd.split() if s] for dd in d])

def hrf_figure(x,y,color=None,order=3,stim_bars=None, alpha_error=0.1, alpha_bars=0.5, error_val=0.2, fig_save=None, title=None, filt="splrep"):

    """
hrf_figure

Draws an image given input timeseries 'y' and an 'x'-axis scale. You can select whether you want stimulation bars on the figure
at certain timepoints ('stim_bars'), array with error values ('error_val'), and a bunch of other things. Basically it's a for-
matting function

Args:
    x:              array representing the x-axis points
    y:              array representing the timecourse to be plotted
    color:          color of the graph (can be any of the matplotlib-specified color-type (RGB|HEX))
    stim_bars:      array of timepoints where to draw perpendicular lines on the figure to denote e.g., stimulation periods
    alpha_error:    float value for transparency of error bar shading
    error_val:      array of error values (default is set to one value, but can/should be an array across the timecourse)
    fig_save:       string to output name to save the figure
    filt:           string consisting either of "None, splrep, or savgol", denoting the type of filtering being performed on the
                    input timecourse
    order:          polynomial order of filtering

Example:
    see linescanning/notebooks/BUFF_nideconv.ipynb

    """

    from scipy.interpolate import splrep, splev
    from scipy.signal import savgol_filter
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure()

    if filt == "splrep":
        filt = splrep(x,y,s=3)
        filt_y = splev(x,filt)
    elif filt == "savgol":
        filt_y = savgol_filter(y, 11, 5)
    else:
        raise ValueError("Unknown filter specified; please use 'savgol', or 'splrep'")

    errorfill(x, filt_y, error_val,  color=color, alpha_fill=alpha_error)

    if stim_bars is not None:
        for b in stim_bars:
            plt.axvline(x=b, alpha=alpha_bars, color='darkgray')

    plt.axis('off')

    if title is not None:
        plt.title(title)

    if fig_save is not None:
        plt.savefig(fig_save, transparent=True)
    else:
        plt.show()


def get_base_dir():

    """
get_base_dir

Short function to determine the system where processing on and set a few paths based on that.
Relatively obsolete now that we can install the linescanning repository as a pacakage, but I
think it's still widely used in the notebooks. Will change that at some point..

Returns the base-directory (where 'projects' and 'programs') should be, and the place ('lin'|
'win')

Example:
    In [36]: d,p = get_base_dir()
    In [37]: d
    Out[37]: '/mnt/d/FSL/shared/spinoza'
    In [38]: p
    Out[38]: 'lin'

    """
    import getpass

    username = getpass.getuser()
    if username == "fsluser":
        base_dir = '/mnt/hgfs/shared/spinoza/'
        place = "lin"

    elif username == "Jurjen Heij":
        base_dir = "D:\\FSL\\shared\\spinoza"
        place = "win"

    elif username == "gjheij":
        base_dir = "/mnt/d/FSL/shared/spinoza"
        place = "lin"
    else:
        base_dir = "/data1/projects/MicroFunc/Jurjen"
        place = "spin"

    return base_dir, place

def get_module_nr(key_word):

    """
get_module_nr

Fetches the module number from the master script given an input string. It sends a command using sed
and grep to the bash command line. Won't work on windows!

Arg:
    key_word    : str
                search string of the module your interested in. Should at least match otherwise the
                function will not find anything. For instance, if we want to know which module the
                creation of the sinus mask is, we can do:

Example:
    In [42]: get_module_nr('sinus')
    Out[42]: '12'

Check for yourself in the master script whether module 12 indeed does create the sagittal sinus mask

    """

    import os
    import subprocess
    opj = os.path.join

    cmd = "sed -n \'50,85p\' {master} | grep -A0 \"{key}\" | grep -Eo \"[0-9]{{1,2}}\" | head -n 1".format(master=opj(os.environ['DIR_SCRIPTS'], 'shell', 'master'), key=key_word)
    # print(cmd)
    mod = subprocess.getoutput(cmd)

    return mod


def bids_fullfile(bids_image):

    import bids
    import os
    opj = os.path.join

    fullfile = opj(bids_image.dirname, bids_image.filename)

    return fullfile

def ctx_config():
    import cortex
    from cortex.options import config

    f = cortex.options.usercfg
    return f

def set_ctx_path(p=None, opt="update"):

    """
set_ctx_path

function that changes the filestore path in the cortex config file to make changing between projects
flexible.

Just specify the path to the new pycortex directory to change. If you do not specify a string, it
will default to what it finds in the os.environ['CTX'] variable as specified in the setup script

You can also ask for the current filestore path with "opt='show_fs'", or the path to the config script
with "opt='show_pn'". To update, leave 'opt' to 'update'

Usage:
    set_ctx_path('path/to/pycortex', "update")

    """

    import cortex
    import os

    if p == None:
        p = os.environ['CTX']

    conf = cortex.options.usercfg

    if opt == "show_pn":
        print(conf)

    f = open(conf, "r+")
    list_f = f.readlines()

    for i,j in enumerate(list_f):
        if "filestore" in j:

            if opt == "show_fs":
                print(j.strip('\n'))

            elif opt == "update":

                if j.split(' ')[2].split('\n')[0] != p:

                    list_f[i] = "filestore = {}".format(p+'\n')

                    print("new {}".format(list_f[i].split('\n')[0]))

    f = open(conf, "w+")
    for i in list_f:
        f.writelines(i)

    f.close()


def decode(obj):

    if isinstance(obj, bytes):
        obj = obj.decode()
    return obj


def reverse_sign(x):

    """
reverse_sign

inverts the sign given set of values. Can be either one value or an array of values that need to be inverted

args:
    x       : int|float|list
            input that needs inverting, either one value or a list

example:
    >> input is integer
        In [2]: x = 5
        In [3]: reverse_sign(x)
        Out[3]: -5

    >> input is array
        In [5]: x = np.array([2, -2340, 2345,123342, 123])
        In [6]: reverse_sign(x)
        Out[6]:
        array([-2.00000e+00,  2.34000e+03, -2.34500e+03, -1.23342e+05,
               -1.23000e+02])

    >> input is float
        In [7]: x = 5.0
        In [8]: reverse_sign(x)
        Out[8]: -5.0

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

def scanner_vs_itksnap():

    """
SCANNER VS ITKSNAP SOFTWARE ANGLES
--------------------------------------------------------------------------------------------------------
Translation values:
    ________|__ITKSNAP________________________________|__SCANNER_______________________________
    X (LR)  |  R = +mm (R = pos; L is neg)            |  L = +mm (R = neg; L = pos)
    Y (AP)  |  A = +mm (A = pos; P = neg)             |  P = +mm (A = neg; P = pos)
    Z (FH)  |  F = +mm (F = pos; H = neg)             |  H = +mm (F = neg; H = pos)

Orientation values:
    ________|__ITKSNAP________________________________|__SCANNER_______________________________
    X (LR)  |  neg = counterclock; pos = clockwise    |  neg = clockwise;    pos = counterclock
    Y (AP)  |  neg = clockwise;    pos = counterclock |  neg = counterclock; pos = clockwise
    Z (FH)  |  neg = clockwise;    pos = counterclock |  neg = counterclock; pos = clockwise

As you can see, the values are interpreted inversely, meaning we need to flip the signs of the rotation
and translation if we want values corresponding to scanner customs.

Another important note to make: the voxel values from coordinates.csv are indexed starting from 0.
Other software packages, such as ITKSNAP indexes starting from 1. If you want the correct value from
coordinates.csv in ITK-Snap, add 1 to all dimensions.
    """

    print(scanner_vs_itksnap.__doc__)


def get_rotation_values(array, format="both", out_type="scanner", show_vals=False, in_type='ants'):

    """
get_rotation_values

This script converts the values from a 3x3 rotation matrix into either radians or degrees (deg = default).
Specify something completely if you want both in a dataframe (e.g., "both")
This script is called by call_getscannercoordinates

Args:
    array       : np.ndarray
                a 3x3 numpy array with float values describing the rotation matrix. You can also specify
                an ANTs or FSL matrix file by setting the 'in_type' flag. The rotation part will be read
                from this 4x4 array

    format      : str
                unit for output; either deg (degrees) or rad (radians) if something else, return both

    out_type    : str
                coordinate system to use. The scanner and ITK use different conventions (see scanner_vs_
                itksnap), so either scanner or itksnap format (default = scanner) need to be specified in
                order to return the correct direction of rotation

    show_vals   : bool
                if True the values are printed to the terminal, otherwise suppressed. Mostly used for
                quick visualization rather than actually functional..

Outputs:
    In [6]: r = np.array([[ 0.734, -0.675,  0.068],
       ...:               [ 0.672,  0.738,  0.062],
       ...:               [-0.092,  0,     0.996]])

    In [7]: get_rotation_values(r)
    Out[7]:
      world axis scanner axis    degrees   radians
    0          x           LR  -0.004181 -0.000073
    1          y           AP  -5.281444 -0.092179
    2          z           FH -42.453360 -0.740951

    """

    import numpy as np
    from scipy.spatial.transform import Rotation as R
    import pandas as pd

    if array.shape[0] == 3 and array.shape[1] == 3:

        rotation_array = array

    else:

        for row in range(0,3):
            if in_type == 'ants':
                rotation_array = ()
                rotation_row = np.array([float(i) for i in array[row].split(' ')[4:7]])
                rotation_array = np.append(rotation_array, rotation_row)
                rotation_array = rotation_array.reshape(3,3)
            elif in_type == 'fsl':
                rotation_array = array[:3,:3]

    # calculate rotations and invert angles if we're doing scanner angles
    if out_type == "scanner":
        degrees = reverse_sign(R.as_euler(R.from_matrix(rotation_array), 'xyz', degrees=True))
        radians = reverse_sign(R.as_euler(R.from_matrix(rotation_array), 'xyz', degrees=False))
    else:
        degrees = R.as_euler(R.from_matrix(rotation_array), 'xyz', degrees=True)
        radians = R.as_euler(R.from_matrix(rotation_array), 'xyz', degrees=False)


    data = {"world axis": ["x", "y", "z"],
            "scanner axis": ["LR", "AP", "FH"],
            "degrees": [float(i) for i in degrees],
            "radians": [float(i) for i in radians]}

    df = pd.DataFrame(data)

    if format == "deg":
        out = df["degrees"]
    elif format == "rad":
        out = df["radians"]
    else:
        out = df

    if show_vals == True:
        print(out)

    return out


def bool_mask(sm_curv, mask):
    """
bool_mask

bool_mask takes smoothed surface and a boolean mask to set the values in the
vertices of the mask to a high value, thereby creating that can be applied
to the minimal curvature data.

Args:
    sm_curv     : numpy.ndarray
                2D, containing surface coordinates
    mask        : numpy.ndarray
                2D, containing boolean coordinates of mask

Returns
-------
numpy.ndarray,
    2D, containing mask in surface space

    """

    import numpy as np

    curv = np.abs(sm_curv)
    curv[~mask] =1e8

    return curv


def get_vertex(curv, surf):
    """
get_vertex

Get the position of the best vertex best on minimal curvature array given
a mask and a minimal curvature map

Args:
    curv    : numpy.ndarray
            2D, containing curvature indices
    surf    : numpy.ndarray
            2D, containing boolean coordinates of mask

Returns:
    min_index:      idx of best vertex
    min_vertex:     position of best vertex
    min_vert_mask:  mask of best vertex

    """
    import numpy as np

    min_index = np.argmin(curv)
    min_vertex = surf[0][min_index]
    min_vert_mask = (surf[1] == min_vertex).sum(0)

    return min_index, min_vertex, min_vert_mask

def get_isocenter(img):
    """
get_isocenter

This function returns the RAS coordinates of the input image's origin. This resembles the
offset relative to the magnetic isocenter. You

Example:
    In [14]: img = 'sub-001_space-ses1_hemi-R_vert-6002.nii.gz'
    In [15]: get_isocenter(img)
    Out[15]: array([  0.27998984,   1.49000375, -15.34000604])

    """

    import numpy as np
    import nibabel as nb

    # get origin in RAS
    if isinstance(img, nb.Nifti1Image):
        fn = img
    elif isinstance(img, str):
        fn = nb.load(img)
    else:
        raise ValueError("Unknown input format for img")

    vox_center = (np.array(fn.shape) - 1) / 2.
    origin = native_to_scanner(img, vox_center, addone=False)

    return origin


def native_to_scanner(anat, coord, inv=False, addone=True):
    """
native_to_scanner

This function returns the RAS coordinates in scanner space given a coordinate in native anatomy space.
Required inputs are an anatomical image to derive the VOX-to-RAS conversion from and a voxel coordinate.

Conversely, if you have a RAS coordinate, you can set the 'inv' flag to True to get the voxel coordinate
corresponding to that RAS-coordinate. To make the output 1x4, the addone-flag is set to true. Set to
false if you'd like 1x3 coordinate.

Args:
    anat    : str
            nifti image to derive the ras2vox (and vice versa) conversion

    coord   : numpy.ndarray
            numpy array containing a coordinate to convert

    inv     : bool
            False if 'coord' is voxel, True if 'coord' is RAS

    addone  : bool
            False if you don't want a trailing '1', returning a 1x3 array, or True if you want a
            trailing '1' to make a matrix homogenous

Example:
    >> vox2ras
        In [23]: native_to_scanner('sub-001_space-ses1_hemi-L_vert-875.nii.gz', np.array([142,  48, 222]))
        Out[23]: array([ -7.42000937, -92.96745521, -15.27866316,   1.        ])

    >> ras2vox
        In [24]: native_to_scanner('sub-001_space-ses1_hemi-L_vert-875.nii.gz', np.array([ -7.42000937, -92.96745521, -15.27866316]), inv=True)
        Out[24]: array([142,  48, 222,   1])

    >> disable trailing '1'
        In [25]: native_to_scanner('sub-001_space-ses1_hemi-L_vert-875.nii.gz', np.array([142,  48, 222]), addone=False)
        Out[25]: array([ -7.42000937, -92.96745521, -15.27866316])

    """

    import numpy as np
    import nibabel as nb
    import sys
    # from nibabel.affines import apply_affine

    if len(coord) != 3:
        coord = coord[:3]

    if isinstance(anat, str):
        anat_img = nb.load(anat)
    elif isinstance(anat, nb.Nifti1Image) or isinstance(anat, nb.freesurfer.mghformat.MGHImage):
        anat_img = anat
    else:
        raise ValueError("Unknown type for input image. Needs to be either a str or nibabel.Nifti1Image")

    if inv == False:
        coord = nb.affines.apply_affine(anat_img.affine, coord)
    else:
        coord = nb.affines.apply_affine(np.linalg.inv(anat_img.affine), coord)
        coord = [int(round(i,0)) for i in coord]

    if addone == True:
        coord = np.append(coord, [1], axis=0)

    return coord


def get_coordinate(vertex,subject,anat1=None,anat2=None,hemi=None,matrix=None,out_space=None,out_type=None):
    """
get_coordinate

Get the coordinate of the vertex in a given space. First we will calculate the
RAS-tkr coordinate corresponding to the vertex as given by FreeView (using
mris_info --vx). Then, we convert that coordinate to a coordinate in the orig.mgz
space (following https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems [3]).
This RAS coordinate can be used to calculate the voxel location in orig.mgz and
rawavg.mgz. We create a binary mask of this coordinate in voxel space.

If we want another space, say a second session, we need a transforma-
tion matrix that maps voxels from one to another. We need a fixed and a moving
image for that to succeed. In general we assume you have a matrix mapping the first
session to a second session, in that the "moving" image will be "rawavg.nii.gz".
We can apply this matrix to the earlier created binary mask and by determining the
maximum value, we can determine the voxel index that is most likely to correspond
to the point in orig.mgz.

Finally, we can output the values in RAS and VOX space

Args:
    vertex      : int
                Vertex location as per output of pycortex. This value will be conver-
                ted to tkrRAS with mris_info

    subject     : str
                Subject name (e.g., sub-xx) identical to how the subject is named in
                the FreeSurfer directory. From this we will extract the orig.mgz and
                rawavg.mgz ("session 1") files as well as surface files

    anat1       : str
                Path to the first session anatomy, which is in theory identical to
                rawavg.mgz. If rawavg.nii.gz does not exist, we will convert rawavg.mgz
                to nifti with mri_convert

    anat2       : str
                Path to a potential second session anatomy, or at least an anatomy in
                a different space

    mat         : list
                If we want to warp files from FreeSurfer space to session 2 space we
                also need the FreeSurfer to session 1 and session 1 to session 2 warp
                files (unfortunately we still need to find a way to concatenate these
                matrices.. One is FLIRT and the other is ANTs, but the conversion with
                c3d_affine_tool -ras2fsl doesn't work properly > it doesn't align the
                files as the ANTs-matrix does..)

                IMPORTANT: the files need to have the format: sub-xxx_from-x_to-x.mat,
                so we can detect where we need to apply which file.

    out_space   : str
                In which space would you like the output? Can be either "orig", "anat1",
                or "anat2"

    out_type    : str
                In which format would you like the output? Can be either "ras" for values
                related to the magnetic isocenter (default), or "vox" useful to confirm
                location

Output
    coordinate:   1x3 numpy.ndarray

if both out_space and out_type are set to None (default), a dataframe will be outputted
with all relevant information. This dataframe is also saved for later reference

    """
    print(f" Fetch RAS point from vertex for {hemi} hemisphere")

    import nibabel as nb
    import numpy as np
    import os, sys
    import subprocess
    from scipy.io import savemat
    import pandas as pd
    from .utils import get_max_coordinate
    opj = os.path.join

    place = get_base_dir()[1]

    if place == "lin" or place == "spin":
        pass
    else:
        raise OSError("We're not on a linux system? Either change the get_base_dir() function in utils or switch to linux for this function. It calls on bash, so it need to be a linux system")

    try:
        fs_dir = os.environ['SUBJECTS_DIR']
        if not os.path.exists(fs_dir):
            raise NameError(f"the directory SUBJECTS_DIR points to does not exist: {fs_dir}")
    except:
        raise ValueError(" Environmental variable SUBJECTS_DIR can't be found; you're probably not on a linux system or you haven't installed/configured FreeSurfer")

    orig_mgz = opj(fs_dir, subject, 'mri', 'orig.mgz')
    orig_nii = orig_mgz.split('.')[0]+".nii.gz"

    brainmask = opj(fs_dir, subject, 'mri', 'brainmask.mgz')

    if not os.path.isfile(orig_nii):
        print("  Converting orig.mgz to orig.nii.gz for later reference")
        cmd_txt = "mri_convert --in_type mgz --out_type nii {mgz} {nii}".format(mgz=orig_mgz, nii=orig_nii)
        os.system(cmd_txt)

    ##########################################################################################################################
    # STEP 1: MAP VERTEX TO ORIG

    # map tkr to orig (following https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems
    # option [3])!

    # BORIG
    print("  Create surface2orig transformation")
    cmd = ('mri_info', '--vox2ras', brainmask)
    L = decode(subprocess.check_output(cmd)).splitlines()
    borig = np.array([[np.float(s) for s in ll.split() if s] for ll in L])

    # RORIG
    cmd = ('mri_info', '--ras2vox', orig_mgz)
    L = decode(subprocess.check_output(cmd)).splitlines()
    rorig = np.array([[np.float(s) for s in ll.split() if s] for ll in L])

    # NORIG
    cmd = ('mri_info', '--vox2ras', orig_mgz)
    L = decode(subprocess.check_output(cmd)).splitlines()
    norig = np.array([[np.float(s) for s in ll.split() if s] for ll in L])

    # TORIG
    cmd = ('mri_info', '--vox2ras-tkr', orig_mgz)
    L = decode(subprocess.check_output(cmd)).splitlines()
    torig = np.array([[np.float(s) for s in ll.split() if s] for ll in L])

    # Combine into surf2orig matrix
    surf2orig = np.dot(norig, np.linalg.inv(torig))
    m = np.matrix(surf2orig)
    with open(opj(os.path.dirname(orig_mgz), "surf2orig.txt"),'wb') as f:
        for line in m:
            np.savetxt(f, line, fmt='%.2f')

    # print('\t'+str(surf2orig).replace('\n','\n\t'))

    # Get RAS coordinate
    if hemi == "left":
        surf = opj(fs_dir, subject, 'surf', 'lh.fiducial')
        tag = "L"
    elif hemi == "right":
        surf = opj(fs_dir, subject, 'surf', 'rh.fiducial')
        tag = "R"
    else:
        raise TypeError(f"Unknown option specified for hemisphere: {hemi}")

    cmd = ('mris_info', '--vx', str(vertex), surf)
    L = decode(subprocess.check_output(cmd)).splitlines()
    tkr_ras = L[1].split(' ')[1:]; tkr_ras = list(filter(None, tkr_ras)); tkr_ras = [round(float(i),2) for i in tkr_ras]; tkr_ras = np.array(tkr_ras)

    if len(tkr_ras) != 4:
        tkr_ras = np.append(tkr_ras, [1], axis=0); print("  tkr_ras   =", tkr_ras)

    coord_scan = np.dot(np.dot(norig,np.linalg.inv(torig)), tkr_ras); coord_scan = np.array([round(i,2) for i in coord_scan]); print("  scan ras  =", coord_scan)

    # Transform RAS to VOX > for some reason this will be the brainmask voxel, not the orig voxel
    vox_orig = [int(round(i,0)) for i in rorig@coord_scan]; vox_orig = np.array(vox_orig); print("  orig vox  =", vox_orig)

    brainmask_ras = np.array(borig@vox_orig); brainmask_ras = np.array([round(float(i),2) for i in brainmask_ras]); print("  brain_ras =", brainmask_ras[:3])
    brainmask_vox = [int(round(i,0)) for i in rorig@brainmask_ras]; brainmask_vox = np.array(brainmask_vox); print("  brain vox =", brainmask_vox)

    # The @ thing = dot product
    orig_ras    = np.array([round(float(i),2) for i in brainmask_ras])
    orig_vox    = brainmask_vox
    anat1_vox   = np.array(native_to_scanner(anat1, orig_ras, inv=True)); print("  anat1 vox =", anat1_vox)
    anat1_ras   = native_to_scanner(anat1, anat1_vox); anat1_ras = np.array([round(float(i),2) for i in anat1_ras]); print("  anat1 ras =", anat1_ras[:3])
    ##########################################################################################################################
    # STEP 2: MAKE MASK OF VOXEL POINT AND WARP IT TO SECOND ANATOMICAL FILE IF PRESENT

    if anat2 != None:

        # Create mask of point in session 1 anatomy
        point_orig = opj(fs_dir, subject, 'mri', f'{subject}_space-fs_hemi-{tag}_vert-{vertex}.nii.gz')
        if not os.path.isfile(point_orig):

            print("  Create image with voxel location in FreeSurfer space")
            empty_fs = np.zeros_like(nb.load(orig_nii).get_fdata())
            empty_fs[orig_vox[0],orig_vox[1],orig_vox[2]] = 1
            # idx = np.where(empty_fs==np.amax(empty_fs))

            empty_fs = nb.Nifti1Image(empty_fs, affine=nb.load(orig_nii).affine, header=nb.load(orig_nii).header)
            empty_fs.to_filename(point_orig)

        # Warp that mask to session 2 if necessary
        if matrix == None:
            raise Exception("No matrix-file specified, can't warp like this")

        point_ses2 = opj(fs_dir, subject, 'mri', f'{subject}_space-ses2_hemi-{tag}_vert-{vertex}.nii.gz')
        if os.path.isfile(point_ses2):
            os.remove(point_ses2)

        warpsession(point_orig, "fs", "ses2", fixed1=anat1, matrix1=matrix[0], fixed2=anat2, matrix2=matrix[1], topup=True)

        anat2_vox = get_max_coordinate(point_ses2)
        anat2_vox = np.array([anat2_vox[i]+1 for i in range(0,3)]); print("  Selecting voxel with max value =", anat2_vox)
        anat2_ras = native_to_scanner(anat2, anat2_vox, addone=False); anat2_ras = np.array([round(float(i),2) for i in anat2_ras])


    if out_space == "orig" and out_type == "ras":
        return orig_ras[:3]
    elif out_space == "orig" and out_type == "vox":
        return orig_vox[:3]
    elif out_space == "anat1" and out_type == "ras":
        return anat1_ras[:3]
    elif out_space == "anat1" and out_type == "vox":
        return anat1_vox[:3]
    elif out_space == "anat2" and out_type == "ras":
        return anat2_ras
    elif out_space == "anat2" and out_type == "vox":
        return anat2_vox
    else:
        df = pd.DataFrame({"space": ["orig", "anat1", "anat2"],
                           "vox": [orig_vox[:3], anat1_vox[:3], anat2_vox],
                           "ras": [orig_ras[:3], anat1_ras[:3], anat2_ras]})

        df['hemi'] = hemi

        # pid = os.getpid()
        # df.to_csv(opj(fs_dir, subject, 'mri', f"coordinates_pid{pid}.csv"))

        return df


def copy_hdr(source_img,dest_img):

    """
copy_hdr

Similar functionality as fslcpgeom but than more rigorious using Nibabel. Copies the ENTIRE header,
including affine, quaternion rotations, and dimensions.

Args:
    source_img      str|nibabel.Nifti1Image:
                    source image from which to derive the header information

    dest_img        str|nibabel.Nifti1Image:
                    destination image to which to copy the header from <source image> to

Returns:
    nibabel.Nifti1Image with updated header information

Example:
    new_img = copy_hdr(img1,img2)

    """

    import nibabel as nb

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


def get_normal(surf,vert):
    """
get_normal

Get the normal vector of the best vertex best on minimal curvature array given
a surface and vertex. Uses the mask output from get_translation

Args:
    surf    : numpy.ndarray
            2D, containing surface coordinates
    vert    : numpy.ndarray
            position of best vertex

Outputs:
    normal_best_vert:   1x3 numpy.ndarray

    """

    import numpy as np

    tris = surf[0][surf[1][vert]]
    normals = np.cross(tris[::,1] - tris[::,0], tris[::,2] - tris[::,0])
    normals /= np.linalg.norm(normals, axis=1)
    normal_best_vert = np.average(normals, axis=0)
    # unique_tris = np.unique(surf[1][vert].ravel())
    # unique_tris_vert = surf[0][unique_tris]

    normal_best_vert[1] = reverse_sign(normal_best_vert[1])
    return normal_best_vert


def correct_angle(x, verbose=False, only_angles=True):

    """
correct_angle

This function converts the angles obtained with normal2angle to angles that we can use 
on the scanner. The scanner doesn't like angles >45 degrees. If inserted, it will flip 
all kinds of parameters such as slice orientation and foldover. This function is WIP, 
and needs further investigation to make it robust

Args:
    <x>                 : float | np.ndarray
                        generally this should be literally the output from normal2angle, 
                        a (3,) array containing the angles relative to each axis.

    <verbose>           : bool (default = False)
                        print messages during the process

    <only_angles>       : bool (default = True)
                        if we are getting the angles for real, we need to decide what
                        the angle with the z-axis means. We do this by returning an
                        additional variable 'z_axis_represents_angle_around' so that
                        we know in '.utils.get_foldover' where to place this angle.
                        By default this is false, and it will only return converted 
                        angles. When doing the final conversion, the real one, turn 
                        this off.

Returns:
    <scanner_angles>    : np.ndarray
                        scanner representation of the input angles

    <z_axis meaning>    : str
                        if <only_angles> is set to False, it additionally returns an
                        "X" or "Y", which specifies around which axis (X or Y) the 
                        angle with the z-axis is to be used

    """                    

    import numpy as np

    if isinstance(x, np.ndarray) or isinstance(x, list):

        scanner_angles = np.zeros((3,))

        # set boolean for flipping RL angle in case of sagittal slice
        flip = False

        # flag for true coronal slice (i.e., angle with Y is already set to zero in scanner.Scanner())
        true_cor = False
        true_sag = False
        
        # the angle with the z-axis can be interpreted in terms of angle AROUND Y (AP) AND AROUND X (RL)
        # If the slice is in the sweetspot of coronal slices, we can interpret the angle WITH the z-axis
        # as angle AROUND the Y (AP) axis. Otherwise we need to interpret it as the angle AROUND the X
        # axis (RL)
        z_axis_represents_angle_around = "Y"

        # for good coronal slices, there are basically for options:
        # 1) Large X | Large Y > vector = center to top-left        (use angle as is)
        # 2) Small X | small Y > vector = center to bottom-right    (use angle as is)
        # 3) Small X | Large Y > vector = center to top-right       (flip sign)
        # 4) Large X | Small Y > vector = center to bottom-left     (flip sign)

        #-------------------------------------------------------------------------------------------------------------------------------
        # deal with x-axis
        if 0 <= x[0] <= 45: # here we can decide on case 2 and 3 (small X's)
            # 1) angles living between 0 deg and 45 deg can freely do so, nothing to update > most likely coronal slice
            scanner_angles[0] = x[0]

            # case 3) Small X | Large Y >> flip angle
            if x[1] >= 90:
                # in this situation, we can have two lines, mirrored over the RL-axis. The only thing that separates them is the angle with 
                # the Y-axis. If this is a large angle, we should have a negative value for the X-angle. If this is a small angle, this means
                # we should have a positive value for the X-angle
                scanner_angles[0] = reverse_sign(scanner_angles[0])

                if verbose:
                    print(f" Case 3 holds: Small X ({round(x[0],2)}) | Large Y ({round(x[1],2)})")
            else:
                # case 2) Small X | small Y >> use angle as is

                if verbose:
                    print(f" Case 2 holds: Small X ({round(x[0],2)}) | Small Y ({round(x[1],2)})")
                else:
                    pass

        elif 45 <= x[0] <= 90:
            # 2) these angles are a bit tricky. This means we're outside of the range for coronal slices, so we insert 
            #    the corrected value into the second position and set position 1 to 0

            # theoretically we could have a bang on coronal slice. In that case the y-axis angle 
            # has been set to zero and the angle with the x-axis represents the angle around the AP 
            # axis.
            if x[1] != 0:
                scanner_angles[0] = 999 # code for sag slice
                scanner_angles[1] = 90-x[0]
                
                if verbose:
                    print(" No coronal cases hold: we'll get a sagittal slice")
                    
                sag_slice = True
            else:
                # if the y-angle is already set to zero, it means we have a coronal slice. This means the only
                # angle we need it this angle. It will deal with the foldover in utils.get_console_settings
                scanner_angles[0] = x[0]
                scanner_angles[1] = 998 # code for fully 90 deg slice
                true_cor = True

                if verbose:
                    print(" We're dealing with a purely coronal slice. Only X-angle is required")

        elif 90 <= x[0] <= 180: # here we can decide on case 1 and 4 (large X's)
            # 3) such angles would mean we have a vector pointing in the opposite direction of case 1). We simply subtract
            #    it from 180 degrees
            scanner_angles[0] = 180-x[0]

            # case 1) Large X | Large Y >> use angle as is
            if x[1] >= 90:
                if verbose:
                    print(f" Case 1 holds: Large X ({round(x[0],2)}) | Large Y ({round(x[1],2)})")
                else:
                    pass
            else:
                # case 4) Large X | Small Y >> flip angle
                scanner_angles[0] = reverse_sign(scanner_angles[0])

                if verbose:
                    print(f" Case 4 holds: Large X ({round(x[0],2)}) | Small Y ({round(x[1],2)})")

            # in case we have a sagittal slice, the RL-angle decides the direction of the vector. A center-topright
            # vector is created with a positive value. This is decided by the RL-angle: a large RL angle means we have a 
            # center-topleft vector, a small RL angle means we have a center-topright vector.
            flip = True



        # if the above resulted in a sagittal slice, we need to convert the angles relative to the AP-axis. We have the
        # angle with the RL-axis, scanner_angles[0], and the angle with the AP-axis then is 90-angle_RL. Because of the
        # way the scanner interprets this, a center-topleft vector is created with the -(90-angle_RL).
        if 45 <= scanner_angles[0] <= 90:
            # Convert angle with x-axis to be compatible with sagittal slice
            if x[1] != 0:
                scanner_angles[1] = 90-scanner_angles[0]
                scanner_angles[0] = 999 # code for sagittal

                # overwrite the angle around the axis that is represented by the angle with the Z-axis
                z_axis_represents_angle_around = "X"
                
                # decide on the direction of the sagittal slice, center-topleft or center-topright depending on what the 
                # initial angle with the x-axis was. Big angle means center-topleft, small angle means center-topright
                if flip == True:
                    if verbose:
                        print(f" X angle was large ({round(x[0],2)}), inverting {round(scanner_angles[1],2)}")
                        print(f" Z angle = angle around RL-axis")
                    scanner_angles[1] = reverse_sign(scanner_angles[1])
                else:
                    if verbose:
                        print(f" X angle was small ({round(x[0],2)}), using {round(scanner_angles[1],2)} as is")
                    else:
                        pass

        else:
            if verbose:
                print(f" Z angle = angle around AP-axis")
            else:
                pass


        #-------------------------------------------------------------------------------------------------------------------------------
        # now, if we found a suitable angle for X, we can ignore Y because we only need one of the two to get our line
        # we could've obtained the Y-angle above if the X-angle was in the 45-90 range. In that case the first two positions
        # are already filled.
        #
        if x[0] == 0:
            # set straight to true sagittal slice
            scanner_angles[1] = x[1]
            true_sag = True
            scanner_angles[0] = 998 # code for full sagittal

            # overwrite the angle around the axis that is represented by the angle with the Z-axis
            z_axis_represents_angle_around = "X"
            if verbose:
                print(" We're dealing with a purely sagittal slice. Only Y-angle is required")
        elif scanner_angles[0] <= 45:
            scanner_angles[1] = 0
        elif scanner_angles[0] == 999:
            # we've already corrected this angle if we have a sagittal slice
            pass
        else:
            # deal with y-axis; same rules as for the x-axis apply
            # we did not manage to get a good angle for X, so we need to convert the angle of Y relative to the AP axis
            
            if 0 <= x[1] <= 45:
                # 1) angles living between 0 deg and 45 deg can freely do so, nothing to update > most likely coronal slice
                scanner_angles[1] = x[1]
            elif 45 <= x[1] <= 90:
                # 2) these angles are a bit tricky. 
                scanner_angles[1] = 90-x[1]
            elif 90 <= x[1] <= 180:
                # 3) such angles would mean we have a vector pointing in the opposite direction of case 1). We simply subtract
                #    it from 180 degrees
                scanner_angles[1] = 180-x[1]

        #-------------------------------------------------------------------------------------------------------------------------------
        # deal with z-axis > this is a special angle as it can reflect an angle around the Z-axis OR the Y-axis, depending on 
        # the slice orientation. If slice == coronal > z-axis = angle AROUND Y-axis. If slice == sagittal > z-axis is angle
        # around X-axis. Previously we've also flattened this angle to the YZ-plane, so it's now a planar angle.

        # The angle with the z-axis starts at 0, where it points in the superior direction [0,0,1]. It then starts to rotate
        # down towards the inferior axis [0,0,-1]. The start = 0 deg, the y-axis is the 90 deg mark, and inferior axis = 180.
        # The scanner interprets these angles from -45 (DOWN/UP lines) to 45 (UP/DOWN) degrees. Other angles will be wrapped.
        if true_cor == False and true_sag == False:
            # we need to do something with the z-axis
            if -45 <= x[2] <= 45:
                # 1) angles living between 0 deg and 45 deg can freely do so, nothing to update > most likely coronal slice
                scanner_angles[2] = x[2]
            elif 45 <= x[2] <= 90:
                
                # 2) these angles are a bit tricky. Here is means that the foldover direction is changed too
                scanner_angles[2] = x[2]-90
            elif 90 <= x[2] <= 180:
                # 3) such angles would mean we have a vector pointing in the opposite direction of case 1). We simply subtract
                #    it from 180 degrees
                scanner_angles[2] = 180-x[2]
                flip = True

            # check if we should have the opposite angle of the one we got.
            if 45 <= scanner_angles[2] <= 90:
                # this means we got the angle proximal to the vector and Z-axis. We need to opposite one
                scanner_angles[2] = 90-scanner_angles[2]

                if flip == True:
                    # depending on whether the initial angle was big, we need to invert the sign to be compatible
                    # with the scanner
                    scanner_angles[2] = reverse_sign(scanner_angles[2])

        else:
            scanner_angles[2] = x[2]

        # return the result
        if only_angles == True:
            return scanner_angles
        else:
            return scanner_angles, z_axis_represents_angle_around


def normal2angle(normal, unit="deg", system="RAS", return_axis=['x','y','z']):

    """
normal2angle

Convert the normal vector to angles representing the angle with the x,y,z axis. This
can be done by taking the arc cosine over the dot product of the normal vector and a
vector representing the axis of interest. E.g., the vector for x would be [1,0,0], for
y it would be [0,1,0], and for z it would be [0,0,1]. Using these vector representations
of the axis we can calculate the angle between these vectors and the normal vector. This
results in radians, so we convert it to degrees by multiplying it with 180/pi.

Notes:
- convert angles to sensible plane: https://www.youtube.com/watch?v=vVPwQgoSG2g: angles
  obtained with this method are not coplanar; they don't live in the same space. So an
  idea would be to decompose the normal vector into it's components so it lives in the
  XY-plane, and then calculate the angles.

Args:
    <normal>        : np.ndarray | list
                    array or list-like representation of the normal vector as per output of
                    pycortex or FreeSurfer (they will return the same normals)
    
    <unit>          : str (default = "deg")
                    unit of angles: "deg"rees or "rad"ians

    <system>        : str (default = "RAS")
                    coordinate system used as reference for calculating the angles. A right-
                    handed system is default (RAS)
                    see: http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm
                    The scanner works in LPS, so we'd need to define the x/y-axis differently
                    to get correct angles

    <return_axis>   : list (default = ['x','y','z'])
                    List of axes to return the angles for. For some functions we only need 
                    the first two axes, which we can retrieve by specifying 'return_axes=
                    ['x', 'y']'.

Returns:
    <angles>    : list
                list-like representation of the angles with each axis, first being the 
                x axis, second the y axis, and third the z-axis.

    """

    import numpy as np

    vector = np.zeros((3))
    vector[:len(normal)] = normal

    # convert to a unit vector in case we received an array with 2 values
    vector = convert2unit(vector)
    # print(f"Vector = {vector}")
    
    # Define empty 3x3 array to be filled with 1's or -1's depending on requested coordinate system
    COORD_SYS = np.eye(3)
    if system.upper() == "RAS":
        np.fill_diagonal(COORD_SYS, [1,1,1])
    elif system.upper() == "LPS":
        np.fill_diagonal(COORD_SYS, [-1,-1,1])
    else:
        raise NotImplementedError(f"You requested a(n) {system.upper()} system, but I can only deal with 'LPS' or 'RAS' for now")

    # this works if both vectors are normal vectors; otherwise this needs to be scaled by the dot-product of both vector
    # magnitudes
    angles = np.arccos(vector @ COORD_SYS)

    # convert to degree or radian
    if unit == "deg":    
        angles = angles*(180/np.pi)

    # we don't always need all axes to be returned, but we do so by default. 
    # Specify some options below.
    if return_axis == ['x','y','z']:
        return angles
    elif return_axis == ['x','y']:
        return angles[:2]
    elif return_axis == ['y','z']:
        return angles[:-2]        
    elif return_axis == ['x']:
        return angles[0]
    elif return_axis == ['y']:
        return angles[1]
    elif return_axis == ['z']:
        return angles[2]   
    else:
        raise NotImplementedError(f"Requested angles were '{return_axis}', this has not yet been implemented")     

def get_transformation(normal, vertex, hemi, idx):

    import numpy as np
    import pandas as pd
    import os


    # print(angles)
    # print(angles_corr)
    # gets the smallest angle from corrected angle array
    angles_corr = normal2angle(normal)
    min_angle = np.argmin(abs(angles_corr))

    if min_angle == 0:

        # most likely a coronal slice with line in LR-direction
        angle_AP = float(angles_corr[0])
        angle_LR = 0
        angle_FH = float(angles_corr[1])
        orientation = 'coronal'
        foldover = 'FH'

    elif min_angle == 1:

        # most likely a sagittal slice with line in AP-direction
        angle_AP = 0
        angle_LR = float(angles_corr[2])
        angle_FH = float(angles_corr[1])
        orientation = 'sagittal'
        foldover = 'FH'

    elif min_angle == 2:

        # most likely a coronal slice with line in FH-direction
        angle_AP = 0
        angle_LR = float(angles_corr[2])
        angle_FH = float(angles_corr[1])
        orientation = 'transverse'
        foldover = 'RL'

    def reverse_sign(x):
        if x > 0:
            val = -x
        else:
            val = abs(x)

        return val

    trans_LR = reverse_sign(vertex[0])
    trans_AP = reverse_sign(vertex[1])
    trans_FH = vertex[2]

    # Write the rotations describing the orientation of the line in the first session anatomy to a text file
    data = {"parameter": ["orientation", "foldover", "vertex", "LR_rot", "AP_rot", "FH_rot", "LR_trans","AP_trans", "FH_trans"],
            "value": [orientation, foldover, idx, angle_LR, angle_AP, angle_FH, trans_LR, trans_AP, trans_FH],
            "hemi": [hemi, hemi, hemi, hemi, hemi, hemi, hemi, hemi, hemi]}

    df = pd.DataFrame(data)

    return df

def get_offset(angle, n_voxels=10, s_voxels=0.25):

    import numpy as np

    width = n_voxels*s_voxels
    offset = (width/np.sin(np.radians(90-abs(angle))))/2
    #
    # print(90-abs(angle))
    # print(np.radians(90-abs(angle)))
    return offset

def get_ctxsurfmove(subj):

    """
get_ctxsurfmove

Following cortex.freesurfer module: "Freesurfer uses FOV/2 for center, let's set the surfaces to use the magnet isocenter",
where it adds an offset of [128, 128, 128]*the affine of the files in the 'anatomicals'-folder.

This short function fetches the offset added given a subject name, assuming a correct specification of the cortex-directory
as defined by 'database.default_filestore, cx_subject'

Usage:

offset = get_ctxsurfmove("sub-001")

    """

    import cortex
    import nibabel as nb
    import numpy as np
    import os
    opj = os.path.join

    anat = opj(cortex.database.default_filestore, subj, 'anatomicals', 'raw.nii.gz')
    if not os.path.exists(anat):
        raise FileNotFoundError(f'Could not find {anat}')

    trans = nb.load(anat).affine[:3, -1]
    surfmove = trans - np.sign(trans) * [128, 128, 128]

    return surfmove


def get_console_settings(angles, hemi, idx, z_axis_meaning="Y"):

    """
get_console_settings

Function that outputs what is to be inserted in the MR console. This function is the 
biggest source of misery during my PhD so far. Needs thorough investigation. The idea
is pretty simple: we have a set of angles obtained from normal2angle, we have conver-
ted them to angles that the scanner can understand (i.e., angles <45 degrees), and 
now we need to derive which ones to use in order to place the line along the normal
vector.

Args:
    <angles>    : np.ndarray
                literally the output from correct_angles, a (3,) numpy array with the
                'corrected' angles

    <hemi>      : str
                should be "L" or "R", is mainly for info reason. It's stored in the 
                dataframe so we can use it to index
    
    <idx>       : int
                this should be the integer representing the selected vertex. This is 
                also only stored in the dataframe. No operations are executed on it

    <z_axis>    : str
                this string specifies how to interpret the angle with the z-axis: as
                angle around the X (RL) or Y (AP) axis. This can be obtained by tur-
                ning off <only_angles> in .utils.correct_angle(). By default it's set
                to 'Y', as that means we're dealing with a coronal slice; the most 
                common one. Though we can also get sagittal slices, so make sure to
                do this dilligently.

    <foldover>  : str
                foldover direction of the OVS bands. Generally this will be FH, but 
                there are instances where that does not apply. It can be returned by
                utils.correct_angle(foldover=True)                

Returns:
    <df>        : pd.DataFrame
                a dataframe containing the information needed to place the line accor-
                dingly. It tells you the foldover direction, slice orientation, and
                angles

    """

    import pandas as pd
    import os

    # Get index of smallest angle
    # min_angle = np.argmin(abs(angles))
    print(f"Dealing with hemisphere: {hemi}")
    foldover = "FH"
    # print(min_angle)
    angle_x,angle_y,angle_z = angles[0],angles[1],angles[2]
    # also see comments in 'correct_angle' for an explanation of what is happening, but in short:
    # - based on the angle between the normal vector and the x-axis [-1,0,0] we decided whether
    #   we have a coronal or sagittal slice. Is the angle between 0-45, then we have a coronal
    #   slice, if it's between 45-90 degrees, we have a sagittal slice and the first angle is 
    #   set to zero
    #
    # - here we will first check if the first angle is zero. As said, if that's the case we have
    #   a sagittal slice.
    #
    # - because the scanner flips the use of (counter)clockwise-ness, we need to flip the angles
    #   for the XY-angles. The Z-angles are interpreted correctedly (negative = UP, positive =
    #   DOWN)
    #
    # Decide on the slice
    if angle_x != 999:

        # coronal slice
        orientation = 'coronal'
        angle_fh = angle_x

        # check if we should do something with the foldover in a coronal slice. This happens when the angle with the
        # z-axis is lower than -45 or larger than 45. We should then flip the angle and change the foldover.
        if angle_fh <= -45:
            print(f" Angle around Y-axis = {angle_fh}; adding 90 deg & setting foldover to LR")
            angle_fh += 90
            foldover = "LR"
        elif angle_fh >= 45:
            print(f" Angle around Y-axis = {angle_fh}; substracting 90 deg & setting foldover to LR")
            angle_fh -= 90
            foldover = "LR"

        # if the angle with the z-axis was large, we need to invert the angle (means right-pointing vector)
        if angle_z >= 90:
            angle_fh = reverse_sign(angle_fh)

    else:
        orientation = 'sagittal'
        angle_fh = angle_y
        # print(f"Angle FH = {round(angle_y,2)}")

        # check if we should do something with the foldover in a sagittal slice. This happens when the angle with the
        # z-axis is lower than -45 or larger than 45. We should then flip the angle and change the foldover.
        if angle_fh <= -45:
            print(f" Angle around X-axis = {angle_fh}; adding 90 deg & setting foldover to AP")
            angle_fh += 90
            foldover = "AP"
        elif angle_fh >= 45:
            print(f" Angle around X-axis = {angle_fh}; adding 90 deg & setting foldover to AP")
            # angle_fh -= 90
            foldover = "AP"

        # if the angle with the z-axis was large, we need to invert the angle (means down-pointing vector)
        if angle_z >= 90:
            angle_fh = reverse_sign(angle_fh)

    # if we have a sagittal slice, the angle with the z-axis represents the angle around the x-axis
    # if we have a coronal slice, the angle with the z-axis represents the angle around the y-axis
    # you can see how this makes sense by using your phone and place is coronally, and sagittally, 
    # then rotate with the angle with the z-axis that invisibly points to your ceiling
    #
    # Sometimes we also give a big angle as z-axis to know what way the vector is pointing. In that
    # case the angle is not actually used for the console.
    if not 998 in angles:
        if z_axis_meaning.upper() == "Y":
            angle_ap = angle_z
            angle_lr = 0
        elif z_axis_meaning.upper() == "X":
            angle_ap = 0
            angle_lr = angle_z
        else:
            raise ValueError(f"Z-axis means an angle around the {z_axis_meaning.upper()} axis?? Needs to be 'X' or 'Y'")
    else:
        angle_lr = 0
        angle_ap = 0


    data = {"parameter": ["orientation", "foldover", "vertex", "LR_rot", "AP_rot", "FH_rot"],
            "value": [orientation, foldover, idx, angle_lr, angle_ap, angle_fh]}    

    df = pd.DataFrame(data)
    df['hemi'] = hemi

    return df

def vertex_rotation(normal, hemi, idx, coord=None):

    import pandas as pd
    from .utils import normal2angle

    angles = normal2angle(normal)

    # Write the rotations describing the orientation of the line in the first session anatomy to a text file
    try:
        data = {"parameter": ["vertex", "LR_rot", "AP_rot", "FH_rot", "normal", "coord"],
                "value": [int(round(idx,0)), angles[0], angles[1], angles[2], normal, coord]}
    except:
        data = {"parameter": ["vertex", "LR_rot", "AP_rot", "FH_rot", "normal"],
                "value": [int(round(idx,0)), angles[0], angles[1], angles[2], normal]}
    data['hemi'] = hemi

    df = pd.DataFrame(data)

    return df

def convert2unit(v, method="np"):

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


# Define function to do some formatting for us
def format_graph(X=None,Y=None,axis=False,title=None,leg=False,box=False,out=None):

    import matplotlib.pyplot as plt

    """
format_graph

Function that formats the graphs. Takes the following arguments:
    - Label for X axis (str) > give name for x-axis
    - Label for Y axis (str) > give name for y-axis
    - Axis (False/True)      > draw axis lines or not
    - Title (if yes, str)    > title of figure
    - Legend (False/True)    > show legend or not
    - Box (False/True)       > draw box around legend or not
    - Out (str)              > save as str()
    """

    # Define axis names
    if X != None:
        plt.gca().set_xlabel(X)
    if Y != None:
        plt.gca().set_ylabel(Y)

    # Only draw x/y axis (not the full box)
    if axis == False:
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

    if title == None:
        pass
    elif title != "":
        plt.title(title)

    # Draw legend
    if leg == True:
        plt.legend()

        if box == False:
            plt.legend(frameon=False)

    # Save or not
    if out != None:
        plt.savefig(out)

def massp_to_table(label_file, out=None, nr_structures=31, unit="vox"):

    """
massp_to_table

This function creates a tabular output from the label image generated by call_nighresmassp (or any
other version of MASSP). Just input the 'label' file and hit go. You can either have the dataframe
returned or create a file by specifying an output name.

Args:
    label_file      str:
                    path to nifti image (MASSP-output 'label' file)

    out             str:
                    path to output name. You can either safe it as a csv, json, txt, or pickle file.
                    I think right now, json-file might be the cleanest.

    nr_structures   int:
                    set to 31 by default as per the massp script

    unit            str:
                    output unit (voxels or mm3). Should be 'vox' for voxel output, or 'mm' for mm
                    (seems to make sense huh..)

Usage:
    file = massp_to_table('sub-001_desc-massp_label.nii.gz', out='massp_lut.json')

Example:
    In [1]: file = massp_to_table('sub-001_desc-massp_label.nii.gz', out='massp_lut.json')
    In [2]: file
    Out[2]: 'massp_lut.json'
    In [3]: massp_to_table('sub-001_desc-massp_label.nii.gz', unit="mm")
    Out[3]:
    {'Str-l': 10702.2163,
     'Str-r': 10816.1125,
     'STN-l': 136.6179,
     'STN-r': 149.2731,
     'SN-l': 540.4317,
     'SN-r': 532.9537,
     ...
     }
    """

    from nighres.parcellation.massp import labels_17structures
    import nibabel as nb
    import numpy as np

    img = nb.load(label_file)
    img_data = img.get_fdata()

    d = {}
    for i in np.arange(0,nr_structures):
        d[labels_17structures[i]] = np.count_nonzero(img_data == i+1)

    # convert to mm3?
    if unit == "mm":
        # multiply dimensions to get mm^3 per voxel
        dims = img.header['pixdim']
        mm_dim = dims[1]*dims[2]*dims[3]

        # multiply this with all elements in dict
        for key in d:
            d[key] = d[key]*mm_dim


    if out:
        # different extensions are possible: csv, txt, json, or pickle
        ext =out.split('.')[-1]
        if ext == "csv":
            import csv
            w = csv.writer(open(out, "w"))
            for key, val in d.items():
                w.writerow([key, val])
        elif ext == "txt":
            f = open(out,"w")
            f.write( str(d) )
            f.close()
        elif ext == "json":
            import json
            json = json.dumps(d, indent=4)
            f = open(out,"w")
            f.write(json)
            f.close()
        elif ext == "pkl":
            import pickle
            f = open(out,"wb")
            pickle.dump(d,f)
            f.close()
        else:
            print(f"Unknown file extension '{ext}'. Use 'csv', 'txt', 'json', or 'pkl'. Returning dictionary itself")
            return d
            # raise ValueError(f"Unknown file extension '{ext}'. Use 'csv', 'txt', 'json', or 'pkl'. Returning dictionary itself")

        return out
    else:
        return d


class VertexInfo:

    """This object reads a .csv file containing relevant information about the angles, vertex position, and normal vector.
    It is a WIP-replacement for get_composite. Beware, this is my first attempt at object oriented programming.."""

    def __init__(self, infofile=None, subject=None):
        import pandas as pd
        self.infofile = infofile
        self.data = pd.read_csv(self.infofile, index_col=0)
        self.subject = subject

    def get_vertex(self, hemi="both"):

        if hemi == "both":
            return {"lh": int(self.data['value']['vertex'][0]),
                    "rh": int(self.data['value']['vertex'][1])}
        elif hemi.lower() == "right" or hemi.lower() == "r" or hemi.lower() == "rh":
            return int(self.data['value']['vertex'][1])
        elif hemi.lower() == "left" or hemi.lower() == "l" or hemi.lower() == "lh":
            return int(self.data['value']['vertex'][0])


    def get_normal(self, hemi="both"):
        """fetch normal vector from dataframe"""

        from .utils import string2float

        if hemi == "both":
            return {"lh": string2float(self.data['value']['normal'][0]),
                    "rh": string2float(self.data['value']['normal'][1])}
        elif hemi.lower() == "right" or hemi.lower() == "r" or hemi.lower() == "rh":
            return string2float(self.data['value']['normal'][1])
        elif hemi.lower() == "left" or hemi.lower() == "l" or hemi.lower() == "lh":
            return string2float(self.data['value']['normal'][0])

    def get_coord(self, hemi="both"):
        """fetch non-corrected coordinates from dataframe"""
        from .utils import string2float

        if hemi == "both":
            return {"lh": string2float(self.data['value']['coord'][0]),
                    "rh": string2float(self.data['value']['coord'][1])}
        elif hemi.lower() == "right" or hemi.lower() == "r" or hemi.lower() == "rh":
            return string2float(self.data['value']['coord'][1])
        elif hemi.lower() == "left" or hemi.lower() == "l" or hemi.lower() == "lh":
            return string2float(self.data['value']['coord'][0])


    @property
    def infofile(self):
        return self._infofile

    @infofile.setter
    def infofile(self, f):
        self._infofile = f


def get_composite(csv_file, idx_col=0):
    """

get_composite

return the dataframe given an input csv-file.

    """
    import pandas as pd

    df = pd.read_csv(csv_file, index_col=idx_col)

    return df


def get_prfdesign(screenshot_path,
                  n_pix=40,
                  dm_edges_clipping=[0,0,0,0]):

    """
get_prfdesign

Basically Marco's gist, but then incorporated in the repo. It takes the directory of screenshots
and creates a vis_design.mat file, telling pRFpy at what point are certain stimulus was presented.

Args:
    screenshot_path     : str
                        string describing the path to the directory with png-files

    n_pix               : int
                        size of the design matrix (basically resolution). The larger the number,
                        the more demanding for the CPU. It's best to have some value which can be
                        divided with 1080, as this is easier to downsample. Default is 40, but 270
                        seems to be a good trade-off between resolution and CPU-demands

    dm_edges_clipping   : list
                        people don't always see the entirety of the screen so it's important to
                        check what the subject can actually see by showing them the cross of for
                        instance the BOLD-screen (the matlab one, not the linux one) and clip the
                        image accordingly

Usage:
    dm = get_prfdesign('path/to/dir/with/pngs', n_pix=270, dm_edges_clipping=[6,1,0,1])

Notes:

    """

    import os
    import numpy as np
    import matplotlib.image as mpimg
    opj = os.path.join

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


        if downsampled_img[:,:,0].shape != design_matrix[...,0].shape:
            print("please choose a n_pix value that is a divisor of "+str(img.shape[0]))

        # binarize image into dm matrix
        # assumes: standard RGB255 format; only colors present in image are black, white, grey, red, green.
        design_matrix[:, :, img_number][np.where(((downsampled_img[:, :, 0] == 0) & (
            downsampled_img[:, :, 1] == 0)) | ((downsampled_img[:, :, 0] == 255) & (downsampled_img[:, :, 1] == 255)))] = 1

        design_matrix[:, :, img_number][np.where(((downsampled_img[:, :, 0] == downsampled_img[:, :, 1]) & (
            downsampled_img[:, :, 1] == downsampled_img[:, :, 2]) & (downsampled_img[:,:,0] != 127) ))] = 1

    #clipping edges
    #top, bottom, left, right
    design_matrix[:dm_edges_clipping[0],:,:] = 0
    design_matrix[(design_matrix.shape[0]-dm_edges_clipping[1]):,:,:] = 0
    design_matrix[:,:dm_edges_clipping[2],:] = 0
    design_matrix[:,(design_matrix.shape[0]-dm_edges_clipping[3]):,:] = 0
    print("  Design matrix completed")

    return design_matrix

def get_file_from_substring(string, path):
    """
get_file_from_substring

This function returns the file given a path and a substring. Avoids annoying stuff with glob.

Usage:
    file = get_file_from_substring("R2", "/path/to/prf")
    """

    import os
    opj = os.path.join

    for f in os.listdir(path):
        if string in f:
            return opj(path, f)

def get_bids_file(layout, filter=None):

    """
get_bids_file

This search function is more tailored for BIDSified data, and requires a list of BIDS-filenames as
per output for l = BIDSLayout(dir, validate=False) & fn = l.get(session='1', datatype='anat') for
instance. From this list the script will look the list of given filters .

Usage:
    layout = BIDSLayout(somedir).get(session='1', datatype='anat')
    fn = get_bids_file(layout, filter=['str1', 'str2', 'str3'])
"""

    from bids import BIDSLayout
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


def bin_fov(img, thresh=0,out=None, fsl=False):
    """
bin_fov

This function returns a binarized version of the input image. If no output name was specified,
it will return the dataframe in nifti-format

Args:
    img         : str
                path to input image

    thresh      : int
                threshold to use (default = 0)

    out         : str
                path to output image (default is None, and will return the data array of the image)

    fsl         : bool
                if you reeeally want a binary image also run fslmaths -bin.. Can only be in combina
                tion with an output image and on a linux system with FSL (default is false)
Example:
    file = bin_fov("/path/to/image.nii.gz")
    bin_fov("/path/to/image.nii.gz", thresh=1, out="/path/to/image.nii.gz", fsl=True)
    bin_fov("/path/to/image.nii.gz", thres=2)

    """

    from scipy import ndimage
    import nibabel as nb
    import os
    from .utils import get_base_dir

    img_file = nb.load(img)                                     # load file
    img_data = img_file.get_fdata()                             # get data
    # img_bin = (img_data > thresh).astype(int)                   # get binary mask where value != 0
    # img_bin = ndimage.binary_fill_holes(img_bin).astype(int)    # fill any holes
    img_data[img_data <= thresh] = 0
    img_bin_img = nb.Nifti1Image(img_data, header=img_file.header, affine=img_file.affine)

    if out != None:
        img_bin_img.to_filename(out)

        # also run fslmaths for proper binarization
        if fsl == True:
            cmd_txt = "fslmaths {in_img} -bin {out_img}".format(in_img=out, out_img=out)
            place = get_base_dir()[1]

            if place != "win":
                os.system(cmd_txt)

    else:
        return img_bin_img


def create_ctx_transform(subj,ctx_dir,in_file,warp_name,binarize=False,ctx_type=None,cmap=None):

    import os, sys
    import numpy as np
    import nibabel as nb
    import cortex
    opj = os.path.join

    """
create_ctx_transform

This script takes an input volume, creates a warp directory, and uses this to get the volume in the
surface space of pycortex

Args:
    <subj>          : str
                    subject name (e.g., sub-xxx)

    <pycortex dir>  : str
                    path to pycortex derivatives folder (excluding sub-xxx)

    <input file>    : str
                    string-like path to file to extract from

    <warp name>     : str
                    string-like path to name of the warp

    <binarize?>     : int
                    should we binarize the FOV? Relevant for partial anatomies, less so for the line
                    (0=no, 1=yes)

    <type>          : str
                    string defining the type of the file, either vertex or volume

Outputs
    output file       new volume containing the extracted data from input file

Example:
    create_ctx_transform(input.nii.gz output.nii.gz)

    """

    if os.path.isfile(in_file):

        if os.path.exists(opj(ctx_dir, subj, 'transforms', warp_name)):
            print("   found existing transform; using reference.nii.gz")
            f = opj(ctx_dir, subj, 'warped', warp_name, 'reference.nii.gz')
        else:

            if binarize == "True":

                split_names = in_file.split("_")
                split_names[-1] = "desc-bin_" + split_names[-1]
                out_file = "_".join(split_names)

                if os.path.isfile(out_file):
                    pass
                else:
                    print("   binarizing file ..")
                    bin_fov(in_file, out=out_file)

                f = out_file

            else:
                print(f"   binarizing parameter is {binarize}; using regular image")
                f = in_file

            print("   using {file}".format(file=os.path.basename(f)))
            # check if dimensions match
            nb_f = nb.load(f)
            hdr  = nb_f.header
            dims = hdr.get_data_shape()

            dim_check = dims[0] == dims[1] == dims[2]
            if dim_check == False:
                print(f"   ERROR: dimensions {dims} are not the same. Not compatible with FreeSurfer")
                sys.exit(1)
            else:
                pass

            ident = np.array([[1,0,0,0],
                              [0,1,0,0],
                              [0,0,1,0],
                              [0,0,0,1]])

            if os.path.exists(opj(ctx_dir, subj, warp_name)):
                print(f"   transform directory {warp_name} exist")
                pass
            else:
                print(f"   creating transform directory: {warp_name}")
                cortex.db.save_xfm(subj, warp_name, ident, reference=f)


        if ctx_type != "None":

            if cmap == "None":
                cmap = 'magma'

            if ctx_type == 'Vertex':
                print(f"   creating vertex ..")
                f_ctx = cortex.Vertex(f, subject=subj, cmap=cmap)

            elif ctx_type == 'Volume':
                print(f"   creating volume ..")
                f_ctx = cortex.Volume(f, subject=subj, cmap=cmap, xfmname=warp_name)

            else:

                f_ctx = None

        else:

            f_ctx = None

        # collect outputs
        if f_ctx == None:
            print("   no pycortex file created; returning initial file")
            return warp_name, f
        else:
            return warp_name, f_ctx

def rotate_normal(norm, xfm, system="RAS"):

    """
rotate_normal

applies the rotation part of an affine matrix to the normal vectorself.

Args:
    norm    : numpy.ndarray
            1x3 or 1x4 array If 1x4 array, the last value should be set to zero to avoid translations

    xfm     : numpy.ndarray | str
            4x4 affine numpy array or string pointing to the matrix-file, can also be 'identity', in 
            which case np.eye(4) will be used. This is handy for planning the line in session 1/Free-
            Surfer space

    system  : str
            use RAS (freesurfer) or LPS (ITK) coordinate system. This is important 
            as we need to apply the matrix in the coordinate system that the vector is living in.
            e.g., RAS vector = RAS matrix (not ANTs' default), LPS vector = LPS matrix. If LPS, 
            then get_matrixfromants is used, otherwise the matrix is first converted to ras with
            ConvertTransformFile and then read in with np.loadtxt.

            Of note: the results of LPS_vector @ LPS_matrix is the same as RAS_vector @ RAS_matrix

Example:
    rotate_normal(normal_vector, xfm, system="LPS")
 
    """

    import numpy as np

    if isinstance(xfm, str):

        if system.upper() == "RAS":
            if xfm != "identity":
                xfm_tmp = xfm.split('.')[0]+"_ras.txt"
                os.system(f"ConvertTransformFile 3 {xfm} {xfm_tmp} --ras --hm")
                xfm = np.loadtxt(xfm_tmp)
            else:
                xfm = np.eye(4)
            # os.remove(xfm_tmp)
        else:
            xfm = get_matrixfromants(xfm)

    # if len(norm) == 3:
    #     norm =  np.append(norm,[0])
    # elif len(norm) == 4:
    #     if norm[3] != 0:
    #         raise ValueError("The last value of array is not zero; this results in translations in the normal vector. Should be set to 0!")
    # else:
    #     raise ValueError(f"Odd number of elements in array.. Vector = {norm}")

    rot_norm = norm@xfm[:3,:3]

    return rot_norm[:3]

def get_matrixfromants(mat, invert=False):

    """
get_matrixfromants

This function greps the rotation and translation matrices from the matrix-file create by antsRegistration
It basically does the same as on of the ANTs functions, but still..

    """

    from scipy.io import loadmat
    import numpy as np

    try:
        genaff = loadmat(mat)
        matrix = np.hstack((genaff['AffineTransform_float_3_3'][0:9].reshape(3,3),genaff['AffineTransform_float_3_3'][9:].reshape(3,1)))
        matrix = np.vstack([matrix, [0,0,0,1]])

        if invert == True:
            matrix = np.linalg.inv(matrix)
    except:
        matrix = np.eye(4)

    return matrix


# def show_slices(slices):

#     import matplotlib.pyplot as plt
#     """ Function to display row of image slices """
#     fig, axes = plt.subplots(1, len(slices))
#     for i, slice in enumerate(slices):
#         axes[i].imshow(slice.T, cmap="gray", origin="lower")


def warpsession(in_img,
                in_space,
                out_space,
                fixed1=None,
                moving1=None,
                fixed2=None,
                moving2=None,
                matrix1=None,
                matrix2=None,
                topup=False,
                thresh=None):

    """
warpsessions

In this project, we use multiple spaces for out structural data: we have FreeSurfer-space in which we have our
surface reconstructions, session 1 on which the FreeSurfer reconstructions are based, and session 2, in which
we actually acquire the line-scanning data.

To easily translate between these spaces, we have this function. Depending on the input data, we can register
it to any space we want. It is important that the input files have the following structure:

    sub-xxx_desc-xxx_space-[fs/ses1/ses2].nii.gz (IMPORTANT!!! Add a 'space-' tag somewhere in the filename)

so we can easily swap the "space-" tag. At the minimum, the function requires an image to be moved, the space
it is in now (in_space), and the output space we need to warp it to (out_space). Based on the type of regi-
stration you'd like to do, you need to specify the matrix files and anatomical files.

Args:
    in_img      : str
                path to the input file

    in_space    : str
                what is the input session (e.g., "ses2", "ses1", or "fs")

    out_space   : str
                what is the output session (e.g., "ses2", "ses1", or "fs")

    fixed1      : str
                what is the first fixed image we need to use for registration

    matrix1     : str
                what is the first matrix we need to use for registration

    fixed2      : str
                what is the second fixed image we need to use for registration

    matrix2     : str
                what is the second matrix we need to use for registration

    topup       : bool (true/false)
                This is when you're doing multi-session registration of a point. The regi-
                stration can introduce other voxels to be coloured as well. This topup flag
                calculates the maximum point, sets that to 1 and the rest to zero before the
                next registration step.

                ONLY NEEDED WHEN YOU'RE DOING MULTI-SESSION REGISTRATION OF A SINGLE POINT!

    thresh      : list
                When not None, this will call on bin_fov to binarize the image given a
                threshold. Should be a list if you'd like double thresholding across multiple
                sessions, e.g., threshold with value X from session 2 to session 1, and
                another threshold from session 1 to FreeSurfer

Examples:
- I have a file in SESSION 2 space and I want to warp it to ..
    > SESSION 1:
        - matrix1 = .mat file mapping session 2 to session 1 (invert session to session 2 matrix)
        - fixed1  = space it needs to be moved to, so your session 1 anatomy (should be rawavg.nii.gz)

    > FREESURFER:
        - matrix1 = .mat file mapping session 2 to session 1
        - fixed1  = session 1 anatomy (rawavg.nii.gz)
        - matrix2 = .mat file mapping session 1 to FreeSurfer
        - fixed2  = FreeSurfer anatomy (should be orig.nii.gz)

- I have a file in SESSION 1 space and I want to warp it to ..
    > FREESURFER:
        - matrix1 = .mat file mapping session 1 to freesurfer
        - fixed1  = FreeSurfer anatomy (should be orig.nii.gz)

    > SESSION 2:
        - matrix1 = .mat file mapping session 1 to session 2
        - fixed1  = session 2 anatomy (lowres MP2RAGE)

- I have a flie in FREESURFER space and I want to warp it to ..
    > SESSION 1:
        - matrix1 = .mat file mapping session 1 to freesurfer (will be inverted here)
        - fixed1  = session 1 anatomy

    > SESSION 2:
        - matrix1 = .mat file mapping freesurfer to session 1
        - fixed1  = session 1 anatomy
        - matrix2 = .mat file mapping session 1 to session 2
        - fixed2  = session 2 anatomy (lowres MP2RAGE)

If there's a multi-session registration going on, the intermediate steps are automatically save.
So if you warp a file from session 2 to FreeSurfer, the intermediate session 1 file will also
be saved.


example:
    warpsession(in_img, "ses2", "fs", fixed1=ses1_anat, matrix1=ses1_to_ses2, fixed2=fs_anat, matrix2=ses1_to_fs)

    """
    import sys
    import os
    import numpy as np
    opj = os.path.join
    place = get_base_dir()[1]

    if not os.path.exists(in_img):
        raise FileNotFoundError(f'could not find input image {in_img}')

    if out_space == None:

        raise ValueError("Please specify the output space: 'fs', 'ses1', or 'ses2'")

    elif out_space == "fs":

        ##################################################################################################################3
        # OUTPUT SPACE = FREESURFER
        #  input = session 1 space > registration matrix between session 1 and freesurfer
        #  input = session 2 space > registration matrix between session 2 and session 1, and session 1 and FreeSurfer

        if in_space == "ses1":

            if matrix1 == None:
                raise ValueError("Need a registration matrix..")

            if fixed1 == None:
                raise ValueError("Need a fixed image (should be orig.nii.gz)..")

            # we're dealing with session 1 data, just need a matrix mapping session1 to freesurfer
            split_name = in_img.split('/')[-1].split('_')

            # find space index
            try:
                idx_space = [(i, split_name.index('space-')) for i, split_name in enumerate(split_name) if 'space-' in split_name][0][0]
            except:
                raise ValueError("Could not find 'space-' tag in filename. Make sure to define the space")

            split_name[idx_space] = split_name[idx_space].replace(split_name[idx_space], "space-fs")
            new_filename = "_".join(split_name)

            print(f"  warping {in_img} to FreeSurfer space")
            out_fs = opj(os.path.dirname(in_img), new_filename)
            if not os.path.exists(out_fs):
                cmd_txt = "{script} {fixed} {moving} {out} {mat}".format(script=opj('call_fslapplytransforms'),
                                                                         fixed=fixed1,
                                                                         moving=in_img,
                                                                         out=out_fs,
                                                                         mat=matrix1)
                if place != "win":
                    os.system(cmd_txt)
                else:
                    raise OSError("Can't run bash scripts on this OS")


        elif in_space == "ses2":

            for i in [matrix1, matrix2, fixed1, fixed2]:
                if i == None:
                    raise ValueError("need more files..")


            # warp session 2 to session 1
            split_name = in_img.split('/')[-1].split('_')

            # find space index
            try:
                idx_space = [(i, split_name.index('space-')) for i, split_name in enumerate(split_name) if 'space-' in split_name][0][0]
            except:
                raise ValueError("Could not find 'space-' tag in filename. Make sure to define the space")

            split_name[idx_space] = split_name[idx_space].replace(split_name[idx_space], "space-ses1")
            new_filename = "_".join(split_name)

            out_ses1 = opj(os.path.dirname(in_img), new_filename)
            if not os.path.exists(out_ses1):
                print(f"  warping {in_img} to session 1 space")
                cmd_txt = "call_antsapplytransforms {fixed} {moving} {out} {mat}".format(fixed=fixed1,moving=in_img, out=out_ses1, mat=matrix1)
                if place != "win":
                    os.system(cmd_txt)
                else:
                    raise OSError("Can't run bash scripts on this OS")


            # binarizing FOV?
            if thresh != None:
                print(f"  thresholding {out_ses1} with {thresh[0]} first..")
                from .utils import bin_fov
                bin_fov(out_ses1, thresh=thresh[0], out=out_ses1)

            # warp session 1 to freesurfer
            split_name = out_ses1.split('/')[-1].split('_')

            # find space index
            try:
                idx_space = [(i, split_name.index('space-')) for i, split_name in enumerate(split_name) if 'space-' in split_name][0][0]
            except:
                raise ValueError("Could not find 'space-' tag in filename. Make sure to define the space")

            split_name[idx_space] = split_name[idx_space].replace(split_name[idx_space], "space-fs")
            new_filename = "_".join(split_name)

            out_fs = opj(os.path.dirname(out_ses1), new_filename)
            if not os.path.exists(out_fs):
                print(f"  warping {out_ses1} to FreeSurfer")
                cmd_txt = "{script} {fixed} {moving} {out} {mat}".format(script=opj('call_fslapplytransforms'), fixed=fixed2,moving=out_ses1, out=out_fs, mat=matrix2)

                if place != "win":
                    os.system(cmd_txt)
                else:
                    raise OSError("Can't run bash scripts on this OS")

            if thresh != None:
                print(f"  thresholding {out_fs} with {thresh[1]}")
                from .utils import bin_fov
                bin_fov(out_fs, thresh=thresh[1], out=out_fs, fsl=True)

    elif out_space == "ses1":

        ##################################################################################################################3
        # OUTPUT SPACE = SESSION 1
        #  input = FreeSurfer space > registration matrix between session 1 and freesurfer (will be inverted here)
        #  input = session 2 space > registration matrix between session 2 and session 1

        if in_space == "fs":

            if matrix1 == None:
                raise ValueError("Need a registration matrix..")

            if fixed1 == None:
                raise ValueError("Need a fixed image (should be orig.nii.gz)..")

            # we're dealing with session 1 data, just need a matrix mapping session1 to freesurfer
            split_name = in_img.split('/')[-1].split('_')

            # find space index
            try:
                idx_space = [(i, split_name.index('space-')) for i, split_name in enumerate(split_name) if 'space-' in split_name][0][0]
            except:
                raise ValueError("Could not find 'space-' tag in filename. Make sure to define the space")

            split_name[idx_space] = split_name[idx_space].replace(split_name[idx_space], "space-fs")
            new_filename = "_".join(split_name)

            print(f"  warping {in_img} to FreeSurfer space")
            out_fs = opj(os.path.dirname(in_img), new_filename)
            if not os.path.exists(out_fs):
                cmd_txt = "{script} {fixed} {moving} {out} {mat}".format(script=opj('call_fslapplytransforms'), fixed=fixed1, moving=in_img, out=out_fs, mat=matrix1)

                if place != "win":
                    os.system(cmd_txt)
                else:
                    raise OSError("Can't run bash scripts on this OS")


        if in_space == "ses2":

            if matrix1 == None:
                raise ValueError("Need a registration matrix..")

            if fixed1 == None:
                raise ValueError("Need a fixed image (should be orig.nii.gz)..")

            # we're dealing with session 1 data, just need a matrix mapping session1 to freesurfer
            split_name = in_img.split('/')[-1].split('_')

            # find space index
            try:
                idx_space = [(i, split_name.index('space-')) for i, split_name in enumerate(split_name) if 'space-' in split_name][0][0]
            except:
                raise ValueError("Could not find 'space-' tag in filename. Make sure to define the space")

            split_name[idx_space] = split_name[idx_space].replace(split_name[idx_space], "space-fs")
            new_filename = "_".join(split_name)

            print(f"  warping {in_img} to FreeSurfer space")
            out_fs = opj(os.path.dirname(in_img), new_filename)
            if not os.path.exists(out_fs):
                cmd_txt = "{script} {fixed} {moving} {out} {mat}".format(script=opj('call_fslapplytransforms'), fixed=fixed1, moving=in_img, out=out_fs, mat=matrix1)

                if place != "win":
                    os.system(cmd_txt)
                else:
                    raise OSError("Can't run bash scripts on this OS")

    elif out_space == "ses2":

        ##################################################################################################################3
        # OUTPUT SPACE = SESSION 2
        #  input = FreeSurfer > registration matrix between session 2 and session 1, and session 1 and FreeSurfer
        #  input = session 1 space > registration matrix between session 2 and session 1

        if in_space == "fs":

            for i in [matrix1, matrix2, fixed1, fixed2]:
                if i == None:
                    raise ValueError("need more files..")


            # warp session 2 to session 1
            split_name = in_img.split('/')[-1].split('_')

            # find space index
            try:
                idx_space = [(i, split_name.index('space-')) for i, split_name in enumerate(split_name) if 'space-' in split_name][0][0]
            except:
                raise ValueError("Could not find 'space-' tag in filename. Make sure to define the space")

            split_name[idx_space] = split_name[idx_space].replace(split_name[idx_space], "space-ses1")
            new_filename = "_".join(split_name)

            out_ses1 = opj(os.path.dirname(in_img), new_filename)
            if os.path.exists(out_ses1):
                os.remove(out_ses1)

            print(f"  Warping {in_img} to session 1 space")
            cmd_txt = "{script} {fixed} {moving} {out} {mat}".format(script=opj('call_fslapplytransforms'), fixed=fixed1, moving=in_img, out=out_ses1, mat=matrix1)

            if place != "win":
                os.system(cmd_txt)
            else:
                raise OSError("Can't run bash scripts on this OS")

            # do some topping up of the point
            if topup == True:
                import nibabel as nb
                from .utils import get_max_coordinate

                # Retrieve coordinate with max value
                max_coord = get_max_coordinate(out_ses1)
                print(f"  Topping up {max_coord} in {out_ses1}")
                # Set that coordinate to 1 in new array and the rest to zero
                img = nb.load(out_ses1)
                empty_ses1 = np.zeros_like(img.get_fdata())
                empty_ses1[max_coord[0], max_coord[1], max_coord[2]] = 1
                empty_ses1 = nb.Nifti1Image(empty_ses1, affine=img.affine, header=img.header)

                # overwrite image
                os.remove(out_ses1)
                empty_ses1.to_filename(out_ses1)

            # warp session 1 to freesurfer
            split_name = out_ses1.split('/')[-1].split('_')

            # find space index
            try:
                idx_space = [(i, split_name.index('space-')) for i, split_name in enumerate(split_name) if 'space-' in split_name][0][0]
            except:
                raise ValueError("Could not find 'space-' tag in filename. Make sure to define the space")

            split_name[idx_space] = split_name[idx_space].replace(split_name[idx_space], "space-ses2")
            new_filename = "_".join(split_name)

            out_ses2 = opj(os.path.dirname(out_ses1), new_filename)
            if os.path.exists(out_ses2):
                os.remove(out_ses2)

            print(f"  Warping {out_ses1} to session 2")
            cmd_txt = "call_antsapplytransforms {fixed} {moving} {out} {mat}".format(fixed=fixed2,
                                                                                     moving=out_ses1,
                                                                                     out=out_ses2,
                                                                                     mat=matrix2)

            if place != "win":
                os.system(cmd_txt)
            else:
                raise OSError("Can't run bash scripts on this OS")

            # do some topping up of the point
            if topup == True:
                import nibabel as nb
                from .utils import get_max_coordinate

                # Retrieve coordinate with max value
                max_coord = get_max_coordinate(out_ses2)
                print(f"  Topping up {max_coord} in {out_ses2}")
                # Set that coordinate to 1 in new array and the rest to zero
                img = nb.load(out_ses2)
                empty_ses2 = np.zeros_like(img.get_fdata())
                empty_ses2[max_coord[0], max_coord[1], max_coord[2]] = 1
                empty_ses2 = nb.Nifti1Image(empty_ses2, affine=img.affine, header=img.header)

                # overwrite image
                os.remove(out_ses2)
                empty_ses2.to_filename(out_ses2)

        if in_space == "ses1":

            if matrix1 == None:
                raise ValueError("Need a registration matrix..")

            if fixed1 == None:
                raise ValueError("Need a fixed image (should be orig.nii.gz)..")

            # we're dealing with session 1 data, just need a matrix mapping session1 to freesurfer
            split_name = in_img.split('/')[-1].split('_')

            # find space index
            try:
                idx_space = [(i, split_name.index('space-')) for i, split_name in enumerate(split_name) if 'space-' in split_name][0][0]
            except:
                raise ValueError("Could not find 'space-' tag in filename. Make sure to define the space")

            split_name[idx_space] = split_name[idx_space].replace(split_name[idx_space], "space-ses2")
            new_filename = "_".join(split_name)

            print(f"  warping {in_img} to FreeSurfer space")
            out_fs = opj(os.path.dirname(in_img), new_filename)
            if not os.path.exists(out_fs):
                cmd_txt = "{script} {fixed} {moving} {out} {mat}".format(
                                                                         script='call_fslapplytransforms',
                                                                         fixed=fixed1, moving=in_img,
                                                                         out=out_fs,
                                                                         mat=matrix1
                                                                        )
                if place != "win":
                    os.system(cmd_txt)
                else:
                    raise OSError("Can't run bash scripts on this OS")


def view_maps(subj_nr, cxdir=None,prfdir=None):

    """
view_maps

Create webviewer containing the polar angle maps, best vertices, curvature, etc to gain a
better idea of the information in the best vertex and for figure plotting.

Args:
    subject     : str
                subject name without 'sub' (e.g., sub-xxx)

    cxdir       : str
                path to pycortex dir (e.g., derivatives/pycortex)

    prfdir      : str
                path to pRF dir (e.g., derivatives/prf)

Outputs:
    A webbrowser as per output of pycortex webgl

    """

    import os, sys, getopt
    import random
    import cortex
    import numpy as np
    import nibabel as nb
    import random
    opj = os.path.join

    if cxdir == None:
        cxdir = os.environ['CTX']

    if prfdir == None:
        prfdir = os.environ['PRF']


    place = get_base_dir()[1]
    subject = f'sub-{subj_nr}'

    if not os.path.isdir(opj(cxdir, subject)):
        print("  ERROR! pycortex directory does not exist")
        sys.exit(1)

    if not os.path.isdir(opj(prfdir, subject)):
        print("  ERROR! prf directory does not exist")
        sys.exit(1)

    # check if we already have pRF-files we can convert to vertices immediately
    try:
        r2      = get_file_from_substring("R2", opj(prfdir, subject))
        ecc     = get_file_from_substring("eccentricity", opj(prfdir, subject))
        polar   = get_file_from_substring("polar", opj(prfdir, subject))
        prf_lr  = opj(prfdir, subject, f'{subject}_desc-bestvertex_hemi-LR.npy')
    except:
        raise FileNotFoundError('could not find R2, eccentricity, and polar angle maps..')

    prf_lr  = np.load(prf_lr)
    r2      = np.load(r2)
    ecc     = np.load(ecc)
    polar   = np.load(polar)

    # r2_v        = cortex.Vertex(r2,subject=subject,vmin=0.02, vmax=0.8, cmap="hsv_alpha")
    ecc_v       = cortex.Vertex2D(ecc,r2,
                                  vmin=0,
                                  vmax=12,
                                  vmin2=0.05,
                                  vmax2=0.4,
                                  subject=subject,cmap='spectral_alpha')
    # polar_v     = cortex.Vertex2D(polar,r2,
    #                               vmin=-np.pi,
    #                               vmax=np.pi,
    #                               vmin2=0.05,
    #                               vmax2=0.4,
    #                               subject=subject,cmap='Retinotopy_RYBCR_2D')
    curv_data   = cortex.db.get_surfinfo(subject, type="curvature") # Choose RDYIbu
    thick_data  = cortex.db.get_surfinfo(subject, type="thickness") # Choose RDYIbu
    prf_lr_v    = cortex.Vertex(prf_lr, subject=subject, cmap='magma', vmin=-0.5, vmax=1)


    port = random.randint(1024,65536)

    if place == "spin":
        cortex.webshow({'curvature': curv_data,
                        'thickness': thick_data,
                        'eccentricity': ecc_v,
                        'best vertices': prf_lr_v
                        }, open_browser=False, port=port)

        txt = "Now run {script} {port} in your local terminal".format(script='/mnt/d/FSL/shared/spinoza/programs/linescanning/bin/call_webviewer.sh', port=port)
    else:
        # cortex.webshow({'curvature': curv_data,
        #                 'thickness': thick_data,
        #                 'r2': r2_v,
        #                 'best vertices': prf_lr_v
        #                 }, open_browser=True, port=port)

        cortex.webshow({'curvature': curv_data,
                        'thickness': thick_data,
                        'best vertices': prf_lr_v
                        }, open_browser=True, port=port)

        os.wait()
        txt = None

    if txt != None:
        print(txt)


def get_thickness(thick_map, hemi, vertex_nr):

    """
get_thickness

Fetch the cortical thickness given a vertex in a certain hemisphere.

Args:
    thick_map   : str
                thickness.npz created by pycortex (to be implemented: map created by user)

    hemi        : str
                which hemisphere do we need to fetch data from

    vertex      : int
                which vertex in the specified hemisphere

Example:
    In [1]: get_thickness("/path/to/thickness.npz", "left", 875)
    Out[2]: 1.3451

    """

    import numpy as np
    import cortex

    thick = np.load(thick_map)
    thick_hemi = thick[hemi]
    val = thick_hemi[vertex_nr]

    # invert value as it is a negative value
    return abs(val)


def get_linerange(thick_map, hemi, vertex_nr, direction, line_vox=720, vox_size=0.25):

    """
get_linerange

Fetches the range of the line that covers the cortical band given a thickness map as per output of
pycortex. It assumes that the vertex is located at the fiducial surface (middle of WM>Pial) and
that the middle of the line is positioned at the vertex. The line is 720 voxels long, that would
mean that 360 is approximately at the position of te vertex. The voxel size of the line is 0.25 mm
so the range = (360-0.5*thickness, 360+0.5*thickness).

Args:
    thick_map   : str
                thickness.npz created by pycortex (to be implemented: map created by user)

    hemi        : str
                which hemisphere do we need to fetch data from

    vertex      : int
                which vertex in the specified hemisphere

    direction   : str
                which border does the line hit first, 'pial' or 'wm'?

    line_vox    : int
                size of the line
    vox_size    : int
                voxel size we need to utilize

Example:
    In [1]: get_linerange("/path/to/thickness.npz", "left", 875, "wm")
    Out[1]: [560,572]

Returns:
    a list with the minimum and maximum range of the line covering the cortical band

Notes:
- Based on two VERY broad assumptions:
    1.) Vertex is located at fiducial surface, also in second session anatomy
    2.) Middle of line (360) is located at the vertex

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    """

    from .utils import get_thickness

    val         = get_thickness(thick_map, hemi, vertex_nr)
    vert_loc    = val/2
    nr_vox      = vert_loc/vox_size
    r_line      = [round((line_vox/2)-nr_vox), round((line_vox/2)+nr_vox)]

    if direction == "pial":
        print(f"  line first hits the {direction}-boundary: upper bound = wm; lower bound = pial")
    elif direction == "wm":
        print(f"  line first hits the {direction}-boundary: upper bound = pial; lower bound = wm")
    else:
        raise ValueError(f"Unkown option {direction}. Need 'pial' or 'wm'")

    return r_line

def ants_registration(fixed, moving,type="rigid",output=None):

    """
ants_registration

python wrapper for call_antsregistration to perform registration with ANTs in the python
environment. Requires the same input as call_antsregistration

Args:
    fixed       : str
                string to nifti reference image (.nii.gz)

    moving      : str
                moving (to-be-registered) image (.nii.gz)

    type        : str
                type of transformation (default = 'rigid')

    output base : str
                output basename (stuff will be appended)

Usage
    ants_registration("fixed.nii.gz", "moving.nii.gz", "output_")

    """

def ants_applytrafo(fixed, moving, trafo=None, invert=0, interp='nn', output=None, return_type="file"):

    """
ants_applytrafo

Python wrapper for call_antsapplytransforms to apply a given transformation, and a set fixed/moving
images. See call_antsapplytransforms for more information on the actual call.

Args:
    fixed       : str|nibabel.Nifti1Image
                string to nifti reference image (.nii.gz) or nibabel.Nifti1Image that will be con-
                verted temporarily to a file (fixed.nii.gz) in the working directory

    moving      : str|nibabel.Nifti1Image
                moving (to-be-registered) image (.nii.gz) or nibabel.Nifti1Image that will be con-
                verted temporarily to a file (moving.nii.gz) in the working directory

    trafo       : str|list
                list or single path to transformation files in order of application

    interp      : str
                interpolation type: 'lin' (linear), 'nn' (NearestNeighbor), gau (Gaussian), bspl<or-
                order>, cws (CosineWindowedSinc), wws (WelchWindowedSinc), hws (HammingWindowed-Sinc),
                lws (LanczosWindowedSinc); default = 'nn'

    invert      : int|list
                list or single integer with the length of trafo-list to specify which transforma-
                tions to invert or not. Default = 0 for all, meaning use as they are specified: do
                not invert

    output      : str
                output name for warped file

    return_type : str
                whether you'd like the filename returned (return_type='file') or a nibabel.Nifti1-
                Image (return_type="nb")
Usage
    f = ants_applytrafo(fixed.nii.gz, moving.nii.gz, trafo=[f1.mat,f2.mat], invert=[0,0], output=
                        "outputfile.nii.gz", return_type="file")
    """

    import os
    import numpy as np
    import nibabel as nb
    opj = os.path.join

    if isinstance(fixed, nb.Nifti1Image):
        fixed.to_filename("fixed.nii.gz")
        fix = "fixed.nii.gz"
    else:
        fix = fixed

    if isinstance(moving, nb.Nifti1Image):
        moving.to_filename("moving.nii.gz")
        mov = "moving.nii.gz"
    else:
        mov = moving

    if not fix or not mov or not trafo:
        print("NEED REFERENC+MOVING IMAGES AND A TRANSFORMATION FILE")
        print(ants_applytrafo.__doc__)
        sys.exit(1)

    # check if we got a list of trafo files; if so, transform it to the string format required by
    # call_antsapplytransforms
    if trafo != None:
        if isinstance(trafo, list):
            trafo_list = " ".join(trafo)
        else:
            trafo_list = trafo
    else:
        raise ValueError("Need a transformation file")

    # check if we got a list of inversion values; if so, transform it to the string format required
    # by call_antsapplytransforms
    if invert != None:
        if isinstance(invert, list) and isinstance(trafo, list):
            if len(invert) != len(trafo):
                raise ValueError("list of inversion does not equal list of transformations: {} vs {}".format(len(invert), len(trafo)))

            inv_list = " ".join(str(e) for e in invert)
        else:
            inv_list = invert

    # check interpolation type:
    interp_list = ['lin', 'nn', 'gau', 'bspl', 'cws', 'wss', 'hws', 'lzs']
    if not interp in interp_list:
        raise ValueError(f"{interp} is not a valid option. Must be one of {interp_list}")

    if not output and return_type.lower() != "file":
        output = opj(os.path.dirname(fixed), 'tmp.nii.gz')
    elif not output and return_type.lower() == "file":
        output = opj(os.path.dirname(fixed), 'mov2fix.nii.gz')
        # raise ValueError("Please specify an output name if you want a file")

    # build command
    try:
        cmd_txt = f'call_antsapplytransforms -i "{inv_list}" -t {interp} {fix} {mov} {output} "{trafo_list}"'
        print(cmd_txt)
        os.system(cmd_txt)
    except:
        raise OSError("Could not execute call_antsapplytransforms; check your distribution or install the linescanning repository")

    # remove temporary files
    if os.path.exists("fixed.nii.gz"):
        os.remove("fixed.nii.gz")

    if os.path.exists("moving.nii.gz"):
        os.remove("moving.nii.gz")

    # output stuff
    if os.path.exists(output):
        if return_type.lower() == "file":
            return output
        elif return_type.lower() == "nb":
            img = nb.load(output)
            # os.remove(output)
            return img
    else:
        raise FileNotFoundError(f"Could not find file '{output}'")


def reorient_img(img, code="RAS", out=None, qform="orig"):

    """
reorient_img

python wrapper for fslswapdim to reorient an input image given an orientation code.
Valid options are: RAS, AIL for now. You can also specify "nb" as code, which re-
orients the input image to nibabel's RAS+ convention.

If no output name is given, it will overwrite the input image.

Args:
    img     : str
            nifti-image to be reoriented

    code    : str
            code for new orientation

    out     : str
            string to output nifti image

    qform   : str|int
            set qform code to original (str 'orig' = default) or a specified in-
            teger

Usage
    reorient_img("input.nii.gz", code="RAS", out="output.nii.gz")
    reorient_img("input.nii.gz", code="AIL", qform=1)
    reorient_img("input.nii.gz", code="AIL", qform='orig')

    """

    import os
    import nibabel as nb
    import numpy as np

    if out != None:
        new = out
    else:
        new = img

    if code.upper() != "NB":

        try:
            os.environ['FSLDIR']
        except:
            raise OSError(f'Could not find FSLDIR variable. Are we on a linux system with FSL?')

        pairs = {"L": "LR", "R": "RL", "A": "AP", "P": "PA", "S": "SI", "I": "IS"}
        orient = "{} {} {}".format(pairs[code[0].upper()], pairs[code[1].upper()], pairs[code[2].upper()])
        cmd_txt = "fslswapdim {} {} {}".format(img, orient, new)
        os.system(cmd_txt)

    elif code.upper() == "NB":
        # use nibabel's RAS
        img_nb = nb.load(img)
        img_hdr = img_nb.header

        if qform == "orig":
            qform = img_hdr['qform_code']

        ras = nb.as_closest_canonical(img_nb)
        if qform != 0:
            ras.header['qform_code'] = np.array([qform], dtype=np.int16)
        else:
            # set to 1 if original qform code = 0
            ras.header['qform_code'] = np.array([1], dtype=np.int16)

        ras.to_filename(new)
    else:
        raise ValueError(f"Code '{code}' not yet implemented")

def rescale_img(array, lower, upper):

    """
ants_registration

python wrapper for call_antsregistration to perform registration with ANTs in the python
environment. Requires the same input as call_antsregistration

Args:
    fixed       : str
                reference image (.nii.gz)

    moving      : str
                moving (to-be-registered) image (.nii.gz)

    type        : str
                type of transformation (default = 'rigid')

    output base : str
                output basename (stuff will be appended)

Usage
    ants_registration("fixed.nii.gz", "moving.nii.gz", "output_")

    """

def rot_mat(Z=0,Y=0,X=0,deg=True,hm=True):

    """
rot_mat

Create a rotation matrix given a set of angles using the scipy rotation module.
You can specify three angles of which a matrix is returned.

Args:
    <Z> <Y> <X> : float|int
                angle around a particular angle. The order should be ZYX!

    <deg>       : bool
                is the input in degrees or in radians? (default = degrees)

    <hm>        : bool
                should we homogenize (make 4x4) array of output rotation matrix
                (default = yes)

Returns:
    3x3 or 4x4 np.ndarray depending on whether 'hm'-flag was set.

Example:
    In [1]: rot_mat(Z=-42.48, Y=5.26)
    Out[1]: array([[ 0.734, -0.672, -0.092,  0.   ],
                   [ 0.675,  0.738,  0.   ,  0.   ],
                   [ 0.068, -0.062,  0.996,  0.   ],
                   [ 0.   ,  0.   ,  0.   ,  1.   ]])

    """

    import numpy as np
    from scipy.spatial.transform import Rotation as R

    print(f'angle around Z = {Z}')
    print(f'angle around Y = {Y}')
    print(f'angle around X = {X}\n')

    r = R.from_euler('zyx', [Z,Y,X], degrees=deg)
    rot = r.as_matrix()

    if hm == True:
        rot_hm = np.eye(4)
        rot_hm[:3,:3] = rot

        return rot_hm
    else:
        return rot

def create_trafo(xfm, img=None, offset=None, return_type=None):

    """
create_trafo

Return a transformation matrix given a rotation element (3x3 array) and a trans-
lation coordinate (offset). If 'img' is specified, the function will return a
dictionary containing the transformation matrix and the transformed matrix of
the input image.

If 'offset' is set to None (default), the origin of the image - if specified -
will be taken as center of rotation. Otherwise it will be set to (0,0,0), what-
ever that may entail exactly..

In an ideal world, you'd specify an image and an offset to get the most accurate
results

Args:
    <xfm>      :  3x3 numpy.ndarray
                  homogenous transformation matrix that should be applied to the
                  input affine matrix

    <img>         : Nifti-1|str
                  Nifti-image (or string to) from which the affine matrix will be
                  derived

    <offset>      : 3x1 numpy.ndarray
                  numpy array containing the translation of the new affine

    <return_type> : str
                  should we return both the transformation matrix and transformed
                  affine of the input image?
                   - None: return both
                   - trafo: return transformation matrix
                   - affine: return transformed affine of input image
Returns:
    4x4 np.ndarray representing the new affine that should be applied to a parti-
    cular image

Example:
    In [1]: create_trafo(rot, img=layer_img,offset=ras_coord, return_type="trafo")
    Out[1]: array([[  0.734,  -0.672,   0.092,  -7.7  ],
                   [  0.675,   0.738,  -0.   , -94.46 ],
                   [ -0.068,   0.062,   0.996,   0.06 ],
                   [  0.   ,   0.   ,   0.   ,   1.   ]])

    """

    import nibabel as nb
    import numpy as np
    import linescanning.bin.utils.utils as line

    if img != None:
        if isinstance(img, nb.Nifti1Image):
            # img is a nifti image
            fn = img
        elif isinstance(img, str):
            # img is a string
            fn = nb.load(img)
        else:
            raise ValueError(f'Unknown input type. Needs to be either a str to nifti image or nibabel.Nifti1Image')
    else:
        fn = None

    # add rotation element
    dim_x,dim_y = xfm.shape
    if dim_x != dim_y:
        raise ValueError(f'dimension {dim_x} and {dim_y} do not correspond. Input needs to be either 3x3 or 4x4')
    else:
        if dim_x == 4 and dim_y == 4:
            # extract rotation element from homogenous matrix
            rot = xfm[:3,:3]
        else:
            # assume matrix = 3x3
            rot = xfm

    trafo = np.eye(4)
    trafo[:3,:3] = rot

    # add translation element
    if isinstance(offset, np.ndarray):
        if fn:
            origin = line.get_isocenter(fn)
            trafo[:3,-1] = offset - origin
            new_aff = trafo@fn.affine
        else:
            trafo[:3,-1] = offset
            new_aff = None
    else:
        if fn != None:
            origin = line.get_isocenter(fn)
        else:
            origin = np.array([0,0,0])

        trafo[:3,-1] = origin

        if fn:
            new_aff = trafo@fn.affine
        else:
            new_aff = None

    if return_type == None:
        if isinstance(new_aff, np.ndarray):
            return trafo, new_aff
        else:
            raise ValueError("you requested the transformed affine to be outputted, but this does not exist. Make sure you specified an input image")
    else:
        if return_type.lower() == "trafo":
            return trafo
        elif return_type.lower() == "affine":
            if isinstance(new_aff, np.ndarray):
                return new_aff
            else:
                raise ValueError("you requested the transformed affine to be outputted, but this does not exist. Make sure you specified an input image")

def create_line_from_slice(in_file, out_file=None, width=16, fold="FH"):

    """
create_line_from_slice

This creates a binary image of the outline of the line. The line's dimensions are 16 voxels of 0.25
mm x 2.5 mm (slice thickness) and 0.25 mm (frequency encoding direction). We know that the middle of
the line is at the center of the slice, so the entire line encompasses 8 voxels up/down from the cen-
ter. This is represented by the 'width' flag, set to 16 by default

Args:
    in_file     : str
                path to image that should be used as reference (generally this should be the 1 slice
                file of the first run or something)

    out_file    : str
                path specifying the output name of the newly created 'line' or 'beam' file

    width       : int
                how many voxels should we use to define the line. Remember that the center of the
                line is width/2, so if it's set to 16, we take 8 voxels below center and 8 voxels
                above center.

    fold        : str
                string denoting the type of foldover direction that was used. We can find this in the
                info-file in the pycortex directory and can either be FH (line = LR), or LR (line =
                FH)
Returns:
    nibabel.niimg or an actual file if the specified output name

Example:
    In [1]: img = create_line_from_slice("input.nii.gz")
    In [2]: img
    Out[2]: <nibabel.nifti1.Nifti1Image at 0x7f5a1de00df0>
    In [3]: img.to_filename('sub-001_ses-2_task-LR_run-8_bold.nii.gz')

    """

    import nibabel as nb
    import numpy as np

    in_img      = nb.load(in_file)
    in_data     = in_img.get_fdata()
    empty_img   = np.zeros_like(in_data)

    upper, lower = (empty_img.shape[0] // 2)+(int(width)/2),(empty_img.shape[0] // 2)-(int(width)/2)

    # print(fold.lower())
    if fold.lower() == "rl" or fold.lower() == "lr":
        beam = np.ones((int(width), empty_img.shape[0],1))
        empty_img[int(lower):int(upper)] = beam*1e8
    elif fold.lower() == "fh" or fold.lower() == "hf":
        beam = np.ones((empty_img.shape[0],int(width),1))
        empty_img[:,int(lower):int(upper)] = beam*1e8
    else:
        raise NotImplementedError(f"Unknown option {fold}, probably not implemented yet..")

    line = nb.Nifti1Image(empty_img, affine=in_img.affine, header=in_img.header)
    if out_file != None:
        line.to_filename(out_file)
    else:
        return line
