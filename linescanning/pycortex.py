import cortex
from linescanning import image, utils
import nibabel as nb
import numpy as np
import os
import random
import sys

opj = os.path.join

def set_ctx_path(p=None, opt="update"):
    """set_ctx_path

    Function that changes the filestore path in the cortex config file to make changing between projects flexible. Just specify the path to the new pycortex directory to change. If you do not specify a string, it will default to what it finds in the os.environ['CTX'] variable as specified in the setup script. You can also ask for the current filestore path with "opt='show_fs'", or the path to the config script with "opt='show_pn'". To update, leave 'opt' to 'update'.

    Parameters
    ----------
    p: str, optional
        either the path we need to set `filestore` to (in combination with `opt="update"`), or None (in combination with `opt="show_fs"` or `opt="show_pn"`)
    opt: str
        either "update" to update the `filestore` with `p`; "show_pn" to show the path to the configuration file; or "show_fs" to show the current `filestore`

    Example
    ----------
    >>> set_ctx_path('path/to/pycortex', "update")
    """

    if p == None:
        p = os.environ['CTX']

    conf = cortex.options.usercfg

    if opt == "show_pn":
        print(conf)

    f = open(conf, "r+")
    list_f = f.readlines()

    for i, j in enumerate(list_f):
        if "filestore =" in j:

            if opt == "show_fs":
                filestore = j.strip('\n').split('=')[-1].strip(' ')
                return filestore

            elif opt == "update":
                if j.split(' ')[2].split('\n')[0] != p:
                    list_f[i] = "filestore = {}".format(p+'\n')
                    print("new {}".format(list_f[i].split('\n')[0]))

    f = open(conf, "w+")
    for i in list_f:
        f.writelines(i)

    f.close()


def create_ctx_transform(subj, ctx_dir, in_file, warp_name, binarize=False, ctx_type=None, cmap=None):

    """create_ctx_transform

    This script takes an input volume, creates a warp directory, and uses this to get the volume in the
    surface space of pycortex

    Parameters
    ----------
    subj: str
        subject name (e.g., sub-xxx)
    pycortex dir: str
        path to pycortex derivatives folder (excluding sub-xxx)
    input file: str
        string-like path to file to extract from
    warp name: str
        string-like path to name of the warp
    binarize?: int
        should we binarize the FOV? Relevant for partial anatomies, less so for the line (0=no, 1=yes)
    type: str
        string defining the type of the file, either vertex or volume

    Returns
    ----------
    str
        new volume containing the extracted data from input file

    Example
    ----------
    >>> create_ctx_transform("sub-001", "/path/derivatives/pycortex", "input.nii.gz", "fs2ctx")
    """

    if os.path.isfile(in_file):

        if os.path.exists(opj(ctx_dir, subj, 'transforms', warp_name)):
            print("Found existing transform; using reference.nii.gz")
            f = opj(ctx_dir, subj, 'warped', warp_name, 'reference.nii.gz')
        else:

            if binarize == "True":

                split_names = in_file.split("_")
                split_names[-1] = "desc-bin_" + split_names[-1]
                out_file = "_".join(split_names)

                if os.path.isfile(out_file):
                    pass
                else:
                    print("Binarizing file ..")
                    image.bin_fov(in_file, out=out_file)

                f = out_file

            else:
                print(f"No binarization; using regular image")
                f = in_file

            print("Using {file}".format(file=os.path.basename(f)))
            # check if dimensions match
            nb_f = nb.load(f)
            hdr = nb_f.header
            dims = hdr.get_data_shape()

            dim_check = dims[0] == dims[1] == dims[2]
            if dim_check == False:
                print(f"ERROR: dimensions {dims} are not the same. Not compatible with FreeSurfer")
                sys.exit(1)
            else:
                pass

            ident = np.eye(4)

            if os.path.exists(opj(ctx_dir, subj, warp_name)):
                print(f"Transform directory {warp_name} exist")
                pass
            else:
                print(f"Creating transform directory: {warp_name}")
                cortex.db.save_xfm(subj, warp_name, ident, reference=f)

        if ctx_type:

            if not cmap:
                cmap = 'magma'

            if ctx_type == 'Vertex':
                print(f"Creating vertex ..")
                f_ctx = cortex.Vertex(f, subject=subj, cmap=cmap)

            elif ctx_type == 'Volume':
                print(f"Creating volume ..")
                f_ctx = cortex.Volume(f, subject=subj, cmap=cmap, xfmname=warp_name)

            else:

                f_ctx = None

        else:

            f_ctx = None

        # collect outputs
        if f_ctx == None:
            print("No pycortex file created; returning initial file")
            return warp_name, f
        else:
            return warp_name, f_ctx


def get_thickness(thick_map, hemi, vertex_nr):
    """get_thickness

    Fetch the cortical thickness given a vertex in a certain hemisphere.

    Parameters
    ----------
    thick_map: str
        thickness.npz created by pycortex (to be implemented: map created by user)
    hemi: str
        which hemisphere do we need to fetch data from
    vertex: int
        which vertex in the specified hemisphere

    Returns
    ----------
    float
        thickness measure given a hemisphere and vertex number

    Example
    ----------
    >>> get_thickness("/path/to/thickness.npz", "left", 875)
    1.3451
    """

    import numpy as np

    thick = np.load(thick_map)
    thick_hemi = thick[hemi]
    val = thick_hemi[vertex_nr]

    # invert value as it is a negative value
    return abs(val)


def view_maps(subj_nr, cxdir=None, prfdir=None):
    """view_maps

    Create webviewer containing the polar angle maps, best vertices, curvature, etc to gain a better idea of the information in the best vertex and for figure plotting.

    Parameters
    ----------
    subject: str
        subject name without 'sub' (e.g., sub-xxx)
    cxdir: str
        path to pycortex dir (e.g., derivatives/pycortex)
    prfdir: str
        path to pRF dir (e.g., derivatives/prf)
    """

    if cxdir == None:
        cxdir = os.environ['CTX']

    if prfdir == None:
        prfdir = os.environ['PRF']

    place = utils.get_base_dir()[1]
    subject = f'sub-{subj_nr}'

    if not os.path.isdir(opj(cxdir, subject)):
        print("  ERROR! pycortex directory does not exist")
        sys.exit(1)

    if not os.path.isdir(opj(prfdir, subject)):
        print("  ERROR! prf directory does not exist")
        sys.exit(1)

    # check if we already have pRF-files we can convert to vertices immediately
    try:
        r2 = utils.get_file_from_substring("R2", opj(prfdir, subject))
        ecc = utils.get_file_from_substring("eccentricity", opj(prfdir, subject))
        polar = utils.get_file_from_substring("polar", opj(prfdir, subject))
        prf_lr = opj(prfdir, subject, f'{subject}_desc-bestvertex_hemi-LR.npy')
    except:
        raise FileNotFoundError(
            'could not find R2, eccentricity, and polar angle maps..')

    prf_lr  = np.load(prf_lr)
    r2      = np.load(r2)
    ecc     = np.load(ecc)
    polar   = np.load(polar)

    # r2_v        = cortex.Vertex(r2,subject=subject,vmin=0.02, vmax=0.8, cmap="hsv_alpha")
    ecc_v = cortex.Vertex2D(ecc, r2,
                            vmin=0,
                            vmax=12,
                            vmin2=0.05,
                            vmax2=0.4,
                            subject=subject, cmap='spectral_alpha')
    # polar_v     = cortex.Vertex2D(polar,r2,
    #                               vmin=-np.pi,
    #                               vmax=np.pi,
    #                               vmin2=0.05,
    #                               vmax2=0.4,
    #                               subject=subject,cmap='Retinotopy_RYBCR_2D')
    curv_data = cortex.db.get_surfinfo(
        subject, type="curvature")  # Choose RDYIbu
    thick_data = cortex.db.get_surfinfo(
        subject, type="thickness")  # Choose RDYIbu
    prf_lr_v = cortex.Vertex(prf_lr, subject=subject,
                             cmap='magma', vmin=-0.5, vmax=1)

    port = random.randint(1024, 65536)

    if place == "spin":
        cortex.webshow({'curvature': curv_data,
                        'thickness': thick_data,
                        'eccentricity': ecc_v,
                        'best vertices': prf_lr_v
                        }, open_browser=False, port=port)

        txt = "Now run {script} {port} in your local terminal".format(
            script='/mnt/d/FSL/shared/spinoza/programs/linescanning/bin/call_webviewer.sh', port=port)
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


def get_ctxsurfmove(subj):

    """get_ctxsurfmove

    Following `cortex.freesurfer` module: "Freesurfer uses FOV/2 for center, let's set the surfaces to use the magnet isocenter", where it adds an offset of [128, 128, 128]*the affine of the files in the 'anatomicals'-folder. This short function fetches the offset added given a subject name, assuming a correct specification of the cortex-directory as defined by 'database.default_filestore, cx_subject'

    Parameters
    ----------
    subj: str
        subject name (e.g., sub-xxx)

    Returns
    ----------
    numpy.ndarray
        (4,4) array representing the inverse of the shift induced when importing a `FreeSurfer` subject into `Pycortex`

    Example
    ----------
    >>> offset = get_ctxsurfmove("sub-001")
    """

    anat = opj(cortex.database.default_filestore, subj, 'anatomicals', 'raw.nii.gz')
    if not os.path.exists(anat):
        raise FileNotFoundError(f'Could not find {anat}')

    trans = nb.load(anat).affine[:3, -1]
    surfmove = trans - np.sign(trans) * [128, 128, 128]

    return surfmove


def get_linerange(thick_map=None, hemi=None, vertex_nr=None, direction=None, line_vox=720, vox_size=0.25, method="ctx", tissue=None):

    """get_linerange

    Fetches the range of the line that covers the cortical band given a thickness map as per output of pycortex. It assumes that the vertex is located at the fiducial surface (middle of WM>Pial) and that the middle of the line is positioned at the vertex. The line is 720 voxels long, that would mean that 360 is approximately at the position of te vertex. The voxel size of the line is 0.25 mm so the range = (360-0.5*thickness, 360+0.5*thickness).

    Parameters
    ----------
    thick_map: str
        thickness.npz created by pycortex (to be implemented: map created by user)
    hemi: str
        which hemisphere do we need to fetch data from
    vertex: int
        which vertex in the specified hemisphere
    direction: str
        which border does the line hit first, 'pial' or 'wm'?
    line_vox: int
        size of the line
    vox_size: int
        voxel size we need to utilize
    method: str
        use the cortical thickness method with the parameters described above ("ctx") or use the Nighres cortex-segmentation ("nighres"). The assumptions described below hold true for the 'ctx' method, so the nighres-method is preferred. If you use this method, you'll need to give the cruise_cortex file to derive the line-range
    tissue: np.ndarray
        cortical segmentation array derived by calculating the average of the max contribtion of each tissue probability (see segmentation_to_line notebook). Only required if you have specified method="nighres"
    
    Returns
    ----------
    list 
        minimum and maximum range of the line covering the cortical band

    Example
    ----------
    >>> get_linerange("/path/to/thickness.npz", "left", 875, "wm")
    [560,572]

    Notes
    ----------
    Based on two VERY broad assumptions:

    * Vertex is located at fiducial surface, also in second session anatomy
    * Middle of line (360) is located at the vertex
    """

    if method == "ctx":

        val = get_thickness(thick_map, hemi, vertex_nr)
        vert_loc = val/2
        nr_vox = vert_loc/vox_size
        r_line = [round((line_vox/2)-nr_vox), round((line_vox/2)+nr_vox)]

        if direction == "pial":
            print(
                f"  line first hits the {direction}-boundary: upper bound = wm; lower bound = pial")
        elif direction == "wm":
            print(
                f"  line first hits the {direction}-boundary: upper bound = pial; lower bound = wm")
        else:
            raise ValueError(f"Unkown option {direction}. Need 'pial' or 'wm'")

    elif method == "nighres":

        # load in files
        if not isinstance(tissue, np.ndarray):
            raise ValueError("Tissue should be a numpy array")

        roi = tissue[345:360]
        start = np.where(roi == 1)[0][0] + 345
        stop = np.where(roi == 1)[0][-1] + 345
        # diff = stop-start

        r_line = [start, stop]

    else:
        raise NotImplementedError(f"Unknown option {method} specified for 'method'. Please use either 'ctx' or 'nighres'")

    return r_line

def make_ecc(subject, ecc=None, r2=None, vmin1=0, vmax1=12, cmap="nipy_spectral", r2_thresh=None):

    """Create eccentricity vertex map as a function of R2 from data array as per output for call_prf"""
    if isinstance(r2,np.ndarray):
        if r2_thresh != None:
            print(f"thresholding eccentricity-map with r2 = {r2_thresh}")
            thresholded_ecc = np.zeros_like(ecc)
            thresholded_ecc[r2>r2_thresh] = ecc[r2>r2_thresh]
            return cortex.Vertex(thresholded_ecc, vmin=vmin1, vmax=vmax1, subject=subject, cmap=cmap)
        else:
            return cortex.Vertex(ecc, vmin=vmin1, vmax=vmax1, subject=subject, cmap=cmap)
    else:
        return cortex.Vertex(ecc, vmin=vmin1, vmax=vmax1, subject=subject, cmap=cmap)


def make_polar(subject, polar=None, r2=None, vmin1=-np.pi, vmax1=np.pi, cmap="hsv_r", r2_thresh=None):

    """Create polar angle vertex map as a function of R2 from data array as per output for call_prf"""

    if isinstance(r2,np.ndarray):
        if r2_thresh != None:
            thresholded_polar = np.zeros_like(polar)
            thresholded_polar[r2>r2_thresh] = polar[r2>r2_thresh]
            return cortex.Vertex(thresholded_polar, vmin=vmin1, vmax=vmax1, subject=subject, cmap=cmap)
        else:
            return cortex.Vertex(polar, vmin=vmin1, vmax=vmax1, subject=subject, cmap=cmap)
    else:
        return cortex.Vertex(polar, vmin=vmin1, vmax=vmax1, subject=subject, cmap=cmap)


def make_r2(subject, r2=None, vmin1=0, vmax1=0.3, cmap="inferno"):

    """Create R2 vertex map as per output for call_prf"""    
    return cortex.Vertex(r2, subject=subject, vmin=vmin1, vmax=vmax1, cmap=cmap)


def make_vertex(subject, array=None, vmin=None, vmax=None, cmap="magma"):

    return cortex.Vertex(array, subject=subject, vmin=vmin, vmax=vmax, cmap=cmap)
