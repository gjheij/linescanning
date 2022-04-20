from linescanning import utils, optimal, pycortex
import os
import pathlib
import nibabel as nb
import nipype.interfaces.freesurfer as fs
import numpy as np
import subprocess
import sys
opj = os.path.join

def ants_registration(fixed=None,moving=None,reg_type="rigid",output=None):
    """ants_registration

    python wrapper for call_antsregistration to perform registration with ANTs in the python
    environment. Requires the same input as call_antsregistration

    Parameters
    ----------
    fixed: str
        string to nifti reference image (.nii.gz)
    moving: str
        moving (to-be-registered) image (.nii.gz)
    reg_type: str
        type of transformation (default = 'rigid')
    output base: str
        output basename (stuff will be appended)

    Returns
    ----------
    str:
        path to transform file 

    Examples
    ----------
    >>> trafo = ants_registration(fixed="fixed.nii.gz", moving="moving.nii.gz", output="output_")
    >>> trafo
    '/path/to/output_genaff.mat'
    """

    if os.sep in output:
        # our output path contains directories, create the path if it doesn't exist
        out_dir = os.path.dirname(output)
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    trafo = output+"genaff.mat" # this will be added by call_antsregistration
    if fixed and moving:
        try:
            cmd_txt = f"call_antsregistration {fixed} {moving} {output} {reg_type}"
            print(cmd_txt, '\n')
            os.system(cmd_txt)
        except:
            raise OSError("Could not execute call_antsregistration; check your distribution or install the linescanning repository")


    if os.path.exists(trafo):
        return trafo
    else:
        raise FileNotFoundError(f"Could not find file '{trafo}'")


def ants_applytrafo(fixed, moving, trafo=None, invert=0, interp='nn', output=None, return_type="file", verbose=False):
    """ants_applytrafo

    Python wrapper for call_antsapplytransforms to apply a given transformation, and a set fixed/moving
    images. See call_antsapplytransforms for more information on the actual call.

    Parameters
    ----------
    fixed: str|nibabel.Nifti1Image
        string to nifti reference image (.nii.gz) or nibabel.Nifti1Image that will be con-verted temporarily to a file (fixed.nii.gz) in the working directory

    moving: str|nibabel.Nifti1Image
        moving (to-be-registered) image (.nii.gz) or nibabel.Nifti1Image that will be converted temporarily to a file (moving.nii.gz) in the working directory

    trafo: str|list
        list or single path to transformation files in order of application

    interp: str
        interpolation type: 'lin' (linear), 'nn' (NearestNeighbor), gau (Gaussian), bspl<order>, cws  CosineWindowedSinc), wws (WelchWindowedSinc), hws (HammingWindowed-Sinc), lws (LanczosWindowedSinc); default = 'nn'

    invert: int|list
        list or single integer with the length of trafo-list to specify which transformations to invert or not. Default = 0 for all, meaning use as they are specified: do not invert

    output: str
        output name for warped file

    return_type: str
        whether you'd like the `filename` returned (return_type='file') or a `nibabel.Nifti1Image` (return_type="nb")
    
    Returns
    ----------
    str
        if `return_type="str"`, then a filename is returned

    nibabel.Nifti1Image
        if `return_type="nb"`, a `nibabel.Nifti1Image` is returned
    
    Example
    ----------
    >>> ants_applytrafo(fixed.nii.gz, moving.nii.gz, trafo=[f1.mat,f2.mat], invert=[0,0], output="outputfile.nii.gz", return_type="file")
    'outputfile.nii.gz'
    """

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
    interp_list = ['lin', 'mul', 'nn', 'gau', 'bspl', 'cws', 'wss', 'hws', 'lzs', 'gen']
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
        if verbose:
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

def ants_applytopoints(chicken_file, output_file, trafo_file):
    """ants_applytopoints

    Wrapper for antsApplyTransformToPoints to warp a coordinate specified in `chicken_file` to a new space using `trafo_file`. Run :func:`linescanning.utils.make_chicken_csv` and `call_antsregistration` from 1 space to another first.

    Parameters
    ----------
    chicken_file: str
        output from linescanning.utils.make_chicken_csv containing the coordinate that needs to be warped
    
    output_file: str
        output csv file containing the warped point in LPS-convention! Swap the first and second dimensions to make the coordinate RAS

    trafo_file: str
        ANTs-transformation file mapping between the spaces. You can also specify 'identity', in which case 'call_makeitkident' will be called

    Returns
    ----------
    str
        filename of output csv file containing the warped point in LPS-convention, `output_file`

    Example
    ----------
    >>> fn = ants_applytopoints("input.csv", "output.csv", "trafo.mat")
    >>> fn
    'output.csv'
    """

    if trafo_file == "identity":
        trafo_file = opj(os.path.dirname(output_file), 'identity.txt')
        cmd = f"call_createident {trafo_file}"
        os.system(cmd)

    cmd = f"antsApplyTransformsToPoints -d 3 -i {chicken_file} -o {output_file} -t [{trafo_file},1]"
    os.system(cmd)
    # print(cmd)

    return output_file


def vert2coord(subject,vert=None, surf=None):
    """vert2coord
    
    Fetch TKR coordinate of surface vertex using mris_info.

    Parameters
    -----------
    subject: str
        subject ID as used in `SUBJECTS_DIR`
    vert: int
        surface vertex in TKR space that we want to extract information from
    surf: str
        surface from which to extract `vert`.

    Returns
    ----------
    numpy.ndarray
        numpy array containing the coordinate of `vert` in surface `surf`
    """

    cmd = ('mris_info', '--vx', str(vert), surf)
    L = utils.decode(subprocess.check_output(cmd)).splitlines()
    tkr_ras = L[1].split(' ')[1:]; tkr_ras = list(filter(None, tkr_ras)); tkr_ras = np.array([round(float(i),2) for i in tkr_ras])

    return tkr_ras

def fs2tkr(subject, coord=None, ref="orig.mgz", fs_dir=None, strip_lead=True):    
    """fs2tkr

    Option [4] from https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems: "*I have a CRS from a voxel in the orig.mgz volume and want to compute the RAS in surface space (tkrRAS) for this point*"

    Parameters
    -----------
    subject: str
        subject ID as used in `SUBJECTS_DIR`
    coord: numpy.ndarray|list
        containing the coordinate in `FreeSurfer` RAS convention
    ref: str, optional
        reference image to use for translation to `Surface` RAS convention (default = `orig.mgz`)
    fs_dir: str, optional
        `FreeSurfer` directory (default = SUBJECTS_DIR)
    strip_lead: bool
        if `True`, a numpy array with shape (4,) with the last element being 1 is returned. If `False`, a numpy array with shape (3,) is returned

    Returns
    ----------
    numpy.ndarray
        numpy array containing the `coord` in Surface RAS convention
    """

    if fs_dir == None:
        fs_dir = os.environ['SUBJECTS_DIR']

    torig = get_vox2ras_tkr(opj(fs_dir, subject, 'mri', ref))

    if len(coord) == 3:
        coord = np.append(coord, 1)
    
    if coord.ndim > 1 or isinstance(coord, list):
        tkr = np.zeros_like(coord)
        for ix,ii in enumerate(coord):
            tkr[ix,...] = (torig @ ii)

        if strip_lead:
            return tkr[...,:3]
        else:
            return tkr

    else:
        tkr_ras = torig @ coord

        if strip_lead:
            return tkr_ras[:3]
        else:
            return tkr_ras

def tkr2ctx(subject, coord=None):
    """tkr2ctx

    Add the surfmove-correction that Pycortex internally applies to FreeSurfer coordinates (see :func:`linescanning.pycortex.get_ctxsurfmove`)

    Parameters
    -----------
    subject: str
        subject ID as used in `SUBJECTS_DIR`
    coord: numpy.ndarray|list
        numpy array or list containing the `coord` in Surface RAS convention

    Returns
    ----------
    numpy.ndarray
        numpy array containing the `coord` in Pycortex convention
    """

    sm = pycortex.get_ctxsurfmove(subject)

    if len(coord) != 3:
        coord = coord[:3]
    
    if len(sm) != 3:
        sm = sm[:3]

    return coord+sm

def ctx2vert(surf,coord=None,rtol=0.015):
    """ctx2vert

    Finds the closest vertices given an RAS coordinate as defined by pycortex. It uses np.closeall with a customizable rtol-value. If this function does not return anything, increase the rtol. If the original coordinate is not on the surface, results might differ.

    Procedure to view CRS in orig.mgz on the surface in Pycortex (also see fs2vert):
    >>> from linescanning import *
    >>> pp = optimal.CalcBestVertex(subject)
    >>> fs_coord = np.array([187,177,41])
    >>> fs2tkr = transform.fs2tkr('sub-001', coord=fs_coord)
    >>> tkr2ctx = transform.tkr2ctx('sub-001', coord=fs2tkr)
    >>> vert = transform.ctx2vert(pp.surface.lh_surf_data[0], coord=tkr2ctx)
    >>> import cortex
    >>> mm = pp.surface.label_to_mask(subject='sub-001', lh_arr=vert, hemi='lh')
    >>> cortex.webview(mm['whole_roi_v'])

    It runs through the surface-array and calculates the similarity between the RAS coordinates in there and the specified coordinate with a given tolerance. All elements of the coordinate should be as close to 1 as possible, showing more similarities between the given coordinate and the coordinate that has a vertex attached to it. Because of this iterative nature, the process takes a few seconds, so it's not that your system is slow per se. 
    Returns a list of indices where the coordinate has passed the tolerance test. You can verify this by entering the indices in FreeView on the given surface. E.g., if you used lh.fiducial, enter the vertex in that box.

    Parameters
    -----------
    surf: str
        surface to extract the vertex from (e.g., `lh.fiducial`)
    coord: numpy.ndarray|list
        numpy array or list containing the `coord` in Pycortex convention
    rtol: float
        error margin in `mm`

    Returns
    ----------
    int
        vertex number corresponding to `coord` in surface `surf`
    """

    qq = []
    for i in surf:
        qq.append(np.allclose((coord/i),1,rtol=rtol))

    return [i for i,x in enumerate(qq) if x]

def fs2vert(subject, coord=None, hemi="lh"):
    """ctx2vert

    Warp a CRS-point from orig.mgz to Pycortex as described in :func:`linescanning.transform.ctx2vert`.

    Parameters
    -----------
    subject: str
        subject ID as used in `SUBJECTS_DIR`
    coord: numpy.ndarray|list
        numpy array or list containing the `coord` in Pycortex convention
    hemi: str
        hemisphere to extract the vertex from (should be `lh` or `rh`)

    Returns
    ----------
    dict
        Dictionary collecting outputs under the following keys

        * vert_nr (int): vertex number corresponding to the input coordinate `coord`
        * vert_obj (dict): output from :func:`linescanning.optimal.CalcBestVert.label_to_mask`, consisting of numpy.ndarrays representing a boolean mask of the vertex
    """

    pp = optimal.CalcBestVertex(subject)
    tkr = fs2tkr(subject, coord=coord)
    ctx = tkr2ctx(subject, coord=tkr)

    if hemi.lower() == "lh" or hemi.lower() == "left" or hemi.lower() == "l":
        surf = pp.surface.lh_surf_data[0]
    elif hemi.lower() == "rh" or hemi.lower() == "right" or hemi.lower() == "r":
        surf = pp.surface.rh_surf_data[0]

    vert = ctx2vert(surf, coord=ctx)
    mm = pp.surface.label_to_mask(subject=subject, lh_arr=vert, hemi=hemi)

    return {'vert_nr': vert, 'vert_obj': mm}

def ctx2tkr(subject, img=None, coord=None, correct=True, hm=True, ret=True, pad_ones=True):

    """ctx2tkr

    Convert a coordinate from Pycortex to FreeSurfer's TKR (surface) definition. Basically it adds the offset back (https://gallantlab.github.io/pycortex/_modules/cortex/freesurfer.html) that is added by Pycortex. With this particular function, you can apply that offset matrix to a given image, or enter a list of coordinate to which to apply the offset to. To just get the offset, specify the subject (needed for all operations), with 'hm' set to False (3x1 coordinate array) or True (4x4 homogenous matrix with offset in translation column).

    Parameters
    ----------
    subject: str
        subject-ID corresponding to the name in the pycortex filestore directory (mandatory!! for looking up how much the image has been moved when imported from FreeSurfer to Pycortex)
    img: nb.Nifti1Image, str
        nifti image or path to nifti image to which to apply the offset to
    coord: list
        one or multiple coordinates to apply the offset to (e.g., left|right hemisphere) in list format (also when entering just 1 coordinate)
    correct: bool
        actually apply the matrix to an input image and return the nifti image with padded affine matrix. Should be used in combination with 'img'
    hm: bool
        if `True`: output a homogenous (4,4) matrix with the offset in the translation column
        if `False`: just the offset values (3,)
    ret: bool
        used in combination with the list of coordinates to return the corrected coordinates in a list. Artifact from when this was part of a class and included the corrected coordinates in the class object itself
    pad_ones: bool
        pad the coordinates with a '1' if the length does not match the (4,4) `surf2orig` matrix to ensure proper dot product

    Returns
    ----------
    np.ndarray
        offset coordinates; 4x4 or 3x1 (depending on the 'hm' flag) array containing the offset values
    list
        corrected coordinates if `coord`-input was a list
    nb.Nifti1Image
        new nifti image with corrected affine matrix

    Examples
    ----------
    >>> offset = ctx2tkr('sub-001', hm=False)
    >>> corr_coordinates = ctx2tkr('sub-001', coord=[coord1,coord2], ret=True)
    >>> corr_nifti = ctx2trk('sub-001', img=input.nii.gz, correct=True)
    """

    offset = pycortex.get_ctxsurfmove(subject)
    if hm == True:
        # make homogenous matrix
        tmp = np.eye(4)
        tmp[:3,-1] = offset

        ctx_offset = tmp
    else:
        ctx_offset = offset

    if correct == True:
        if img:
            dim1,dim2 = ctx_offset.shape
            if dim1 != 4 and dim2 != 4:
                tmp = np.eye(4)
                offset = tmp[:3,:3] = offset

            nb_img = nb.load(img)
            new_aff = offset@nb_img.affine
            return nb.Nifti1Image(nb_img.get_fdata(), affine=new_aff)

    if coord != None:
        corr = []
        if len(ctx_offset) != 1:
            try:
                offset = ctx_offset[:3,-1]
            except:
                raise ValueError(f"Got input with unhashable dimensions. Could either be a (3,) or (4,4) matrix, not {ctx_offset.shape}")
        else:
            offset = ctx_offset

        for i in coord:
            tkr_coords = i-offset

            if pad_ones == True:
                if len(tkr_coords) == 3:
                    tkr_coords = np.append(tkr_coords,1)

            corr.append(tkr_coords)

        if ret == True:
            return corr

    if not img and not coord:
        if hm == True:
            return tmp
        else:
            return offset


def tkr2fs(subject, coord=None, fs_dir=None, pad_ones=True):
    """tkr2fs

    Convert a coordinate from FreeSurfer's TKR (surface) definition to FreeSurfer's anatomical (volume) definition. This involves the procedure described here: https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems scenario [3]: "I have a point on the surface ("Vertex RAS" in tksurfer) and want to compute the Scanner RAS in orig.mgz that corresponds to this point".

    Parameters
    ----------
    subject: str
        subject-ID corresponding to the name in the pycortex filestore directory (mandatory!! for looking up how much the image has been moved when imported from FreeSurfer to Pycortex)
    img: nb.Nifti1Image, str
        nifti image or path to nifti image to which to apply the offset to
    coord: list
        one or multiple coordinates to apply the offset to (e.g., left|right hemisphere) in list format (also when entering just 1 coordinate)
    fs_dir: str, optional
        `FreeSurfer` directory (default = SUBJECTS_DIR)
    pad_ones: bool
        pad the coordinates with a '1' if the length does not match the (4,4) `surf2orig` matrix to ensure proper dot product

    Returns
    ----------
    np.ndarray
        (4,4) array containing the transformation from Surface RAS to Scanner RAS
    list
        corrected coordinates if `coord`-input was a list

    Examples
    ----------
    >>> off,coord = tkr2fs('sub-001', coord=[tkr_coord1,tkr_coord2])
    """

    if fs_dir == None:
        fs_dir = os.environ['SUBJECTS_DIR']

    orig_mgz = opj(fs_dir, subject, 'mri', 'orig.mgz')

    # NORIG
    norig = get_vox2ras(orig_mgz)

    # TORIG
    torig = get_vox2ras_tkr(orig_mgz)

    # Combine into surf2orig matrix
    surf2orig = norig @ np.linalg.inv(torig)
    if isinstance(coord, list) or isinstance(coord, np.ndarray):
        corr = []
        for i in coord:
            if len(i) != 4:
                c = np.append(i, 1)
            else:
                c = i # ;)

            scanner_ras = surf2orig @ c
            corr.append(scanner_ras)

        return corr
    else:
        corr = None

    if isinstance(corr, list):
        return corr
    else:
        return surf2orig

def rawavg2fs(subject, coord=None, fs_dir=None):
    """rawavg2fs

    Option [6] from https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems: "*I have a CRS from a voxel in my functional/diffusion/ASL/rawavg/etc "mov" volume and want to compute the CRS for the corresponding point in the orig.mgz*"

    Parameters
    -----------
    subject: str
        subject ID as used in `SUBJECTS_DIR`
    coord: numpy.ndarray|list
        containing the coordinate in `FreeSurfer` rawavg convention (voxels)
    fs_dir: str, optional
        `FreeSurfer` directory (default = SUBJECTS_DIR)

    Returns
    ----------
    numpy.ndarray
        numpy array containing the `coord` in `orig.mgz` voxel convention
    """

    if fs_dir == None:
        fs_dir = os.environ['SUBJECTS_DIR']

    orig = opj(fs_dir, subject, 'mri', 'orig.mgz')
    move = opj(fs_dir, subject, 'mri', 'rawavg.mgz')

    torig = get_vox2ras_tkr(orig)
    tmove = get_vox2ras_tkr(move)
    reg = get_tkrreg(subject, mov=move, targ=orig)

    if len(coord) != 4:
        coord = np.append(coord,1)

    orig_coord = np.linalg.inv(torig) @ np.linalg.inv(reg) @ tmove @ coord

    return np.array([int(round(i,0)) for i in orig_coord])

def fs2rawavg(subject, coord=None, fs_dir=None):
    """fs2rawavg

    Inverse of :func:`linescanning.transform.rawavg2fs`

    Parameters
    -----------
    subject: str
        subject ID as used in `SUBJECTS_DIR`
    coord: numpy.ndarray|list
        containing the coordinate in `FreeSurfer` rawavg convention (voxels)
    fs_dir: str, optional
        `FreeSurfer` directory (default = SUBJECTS_DIR)

    Returns
    ----------
    numpy.ndarray
        numpy array containing the `coord` in `rawavg.mgz` voxel convention
    """

    if fs_dir == None:
        fs_dir = os.environ['SUBJECTS_DIR']

    orig = opj(fs_dir, subject, 'mri', 'orig.mgz')
    move = opj(fs_dir, subject, 'mri', 'rawavg.mgz')

    torig = get_vox2ras_tkr(orig)
    tmove = get_vox2ras_tkr(move)
    reg = get_tkrreg(subject, mov=move, targ=orig)

    if len(coord) != 4:
        coord = np.append(coord,1)

    orig_coord = np.linalg.inv(torig) @ reg @ tmove @ coord

    # return np.array([int(round(i,0)) for i in orig_coord])    
    return orig_coord

def tkr2rawavg(subject,matrix=None,coord=None,reg=True,fs_dir=None, inv=False, out_type="voxel"):

    """tkr2rawavg

    Computes the transformation from FreeSurfer TKR space (orig.mgz) to the native space (rawavg.nii.gz). It a ssumes the FreeSurfer and native space differ, which is not necessarily the case if you already have isotropic native data. You can either specify the registration file as per the output of tkregister2 or have this function create that file (register.dat by default)

    Parameters
    ----------
    subject: str
        subject-ID corresponding to the name in the pycortex filestore directory (mandatory!! for looking up how much the image has been moved when imported from FreeSurfer to Pycortex)
    matrix: str
        path to ANTs-format registration file (either the .txt or .mat file, doesn't really matter for ants_applytrafo)
    reg: bool
        create the matrix mapping FreeSurfer to native instead of specifying a matrix. Only one or the other is needed.
    fs_dir: str
        `FreeSurfer` directory (default = SUBJECTS_DIR)
    inv: bool
        if the matrix file you specify with 'matrix' is actually mapping native to FreeSurfer, set this flag to True to invert the matrix
    out_type: str
        output either the voxel ('voxel') or RAS ('ras') coordinate

    Returns
    ----------
    np.ndarray
        (4,4) array containing the transformation from Surface RAS to Scanner RAS
                        
    list
        corrected coordinates if `coord`-input was a list

    Example
    ----------
    >>> off,coord = tkr2rawavg('sub-001', matrix="register.dat", coord=[tkr_coord1,tkr_coord2])
    """


    mov = opj(fs_dir, subject, 'mri', 'rawavg.mgz')
    tar = opj(fs_dir, subject, 'mri', 'orig.mgz')

    if reg == True:
        m = get_tkrreg(subject, mov=mov, targ=tar, fs_dir=fs_dir, return_type='arr')
    else:
        if matrix:
            m = utils.read_fs_reg(matrix)
        else:
            raise ValueError("Need a matrix file if 'reg'-flag is set to False. Either specify a file or set 'reg' to True")

    Tmove = get_vox2ras_tkr(mov)

    if coord:
        corr = []
        for i in coord:
            if len(i) != 4:
                c = np.append(i, 1)
            else:
                c = i # ;)
                
            vox = np.linalg.inv(Tmove) @ m @ c

            if out_type.lower() == "ras":
                vox2ras = get_vox2ras(opj(fs_dir, subject, 'mri', 'rawavg.mgz'))
                vox = vox2ras@vox
            elif out_type.lower() == "vox" or out_type.lower() == "voxel":
                vox = np.array([int(i) for i in vox])

            corr.append(vox)

        return corr
    else:
        corr = None

    if isinstance(corr, list):
        return corr
    else:
        return np.linalg.inv(Tmove) @ m


def rawavg2lowres(fixed, moving, matrix, inv=False, out_file=None):
    """rawavg2lowres

    Transform a file from `rawavg` in another session (e.g., `lowres`) via a registration matrix. Uses :func:`linescanning.transform.ants_applytransform`

    Parameters
    -----------
    fixed: str
        reference image (e.g., `lowres`)
    moving: str
        moving image (e.g., `rawavg`)
    matrix: str
        transformation file mapping `moving` to `fixed`
    inv: bool
        if transformation file maps `fixed` to `moving`, we can invert the matrix by setting `inv=True`
    out_file: str
        output file representing `moving` in `fixed`-space

    Returns
    ----------
    str
        filename of output file representing `moving` in `fixed`-space
    """

    if inv == False:
        do_invert = 0
    elif inv == True:
        do_invert = 1
    else:
        raise ValueError(f"Unknown input {inv} for 'invert'-flag. Specify True (invert input matrix) or False (do not invert input matrix)")

    # linear interpolation results in 1 coordinate in session 2 anatomy
    if matrix:
        if out_file:
            f = ants_applytrafo(fixed,moving,trafo=matrix, interp="lin", invert=do_invert, output=out_file, return_type="file")
        else:
            f = ants_applytrafo(fixed,moving,trafo=matrix, interp="lin", invert=do_invert, return_type="nb")

    return f


def get_tkrreg(subject, mov=None, targ=None, out_file=None, fs_dir=None, return_type='arr'):
    """get_tkrreg
    
    Implementation of `tkregister2` by sending a command to the command line.

    Parameters
    ----------
    subject: str
        subject-ID corresponding to the name in the pycortex filestore directory (mandatory!! for looking up how much the image has been moved when imported from FreeSurfer to Pycortex)
    mov: str
        moving image
    targ: str
        reference image (will default to `orig.mgz` if left empty)
    out_file: str
        output file of transformation file
    fs_dir: str
        `FreeSurfer` directory (default = SUBJECTS_DIR)
    return_type: str
        if `file`, `out_file` is returned
        if `arr`, `out_file` will be read into a numpy.ndarray

    Returns
    ----------
    str
        if `return_type=="file", `out_file` is returned

    numpy.ndarray
        if `return_type=="arr", `out_file` is returned
    """

    if fs_dir == None:
        fs_dir = os.environ['SUBJECTS_DIR']
        
    if not targ:
        targ = opj(fs_dir, subject, 'mri', 'orig.mgz')

    if not mov:
        targ = opj(fs_dir, subject, 'mri', 'rawavg.mgz')

    if not out_file:
        out_file = opj(os.path.dirname(targ), 'register.dat')

    cmd = f"tkregister2 --mov {mov} --targ {targ} --reg {out_file} --noedit --regheader"
    try:
        os.system(cmd)
    except:
        raise OSError("Could not run tkregister2")

    if return_type.lower() == 'arr':
        return utils.read_fs_reg(out_file)
    else:
        return out_file

def get_vox2ras_tkr(img):

    """fetch the vox2ras-tkr matrix from an image (Torig/Tmov on the FreeSurfer wiki)"""

    cmd = ('mri_info', '--vox2ras-tkr', img)
    L = utils.decode(subprocess.check_output(cmd)).splitlines()
    torig = np.array([[np.float(s) for s in ll.split() if s] for ll in L])

    return torig

def get_ras2vox(img):

    """fetch the ras2vox matrix from an image (Rorig on the FreeSurfer wiki)"""

    cmd = ('mri_info', '--ras2vox', img)
    L = utils.decode(subprocess.check_output(cmd)).splitlines()
    rorig = np.array([[np.float(s) for s in ll.split() if s] for ll in L])

    return rorig

def get_vox2ras(img):

    """fetch the vox2ras matrix from an image (Norig on the FreeSurfer wiki)"""

    cmd = ('mri_info', '--vox2ras', img)
    L = utils.decode(subprocess.check_output(cmd)).splitlines()
    norig = np.array([[np.float(s) for s in ll.split() if s] for ll in L])

    return norig

def ras2lps(coord):
    """represent an RAS coordinate in LPS convention"""

    # ras2lps = [-1,-1,1]
    if len(coord) == 4:
        coord = coord[:3]

    return coord*np.array([-1,-1,1])

def ras2las(coord):
    """represent an RAS coordinate in LAS convention"""
    # ras2las = [-1,1,1]
    if len(coord) == 4:
        coord = coord[:3]

    return coord*np.array([-1,1,1])
    
def ras2lpi(coord):
    """represent an RAS coordinate in LPO+I convention"""

    # ras2las = [-1,-1,-1]
    if len(coord) == 4:
        coord = coord[:3]

    return coord*np.array([-1,-1,-1])

def native_to_scanner(anat, coord, inv=False, addone=True):
    """native_to_scanner
    
    This function returns the RAS coordinates in scanner space given a coordinate in native anatomy space. Required inputs are an anatomical image to derive the VOX-to-RAS conversion from and a voxel coordinate. Conversely, if you have a RAS coordinate, you can set the 'inv' flag to True to get the voxel coordinate corresponding to that RAS-coordinate. To make the output 1x4, the addone-flag is set to true. Set to false if you'd like 1x3 coordinate.

    Parameters
    ----------
    anat: str
        nifti image to derive the ras2vox (and vice versa) conversion
    coord: numpy.ndarray
        numpy array containing a coordinate to convert
    inv: bool
        False if 'coord' is voxel, True if `coord` is RAS        
    addone: bool
        False if you don't want a trailing *1*, returning a 1x3 array, or True if you want a trailing *1* to make a matrix homogenous

    Returns
    ----------
    numpy.ndarray
        (3,) or (4,) array containing the input coordinate `coord` in scanner convention (if `coord` is in voxel convention and `inv==False`) or voxel convention (if `coord` is in scanner convention and `inv==True`)

    Examples
    ----------
    >>> # vox2ras
    >>> native_to_scanner('sub-001_space-ses1_hemi-L_vert-875.nii.gz', np.array([142,  48, 222]))
    array([ -7.42000937, -92.96745521, -15.27866316,   1.        ])
    >>> # ras2vox
    >>> native_to_scanner('sub-001_space-ses1_hemi-L_vert-875.nii.gz', np.array([ -7.42000937, -92.96745521, -15.27866316]), inv=True)
    array([142,  48, 222,   1])
    >>> # disable trailing '1'
    >>> native_to_scanner('sub-001_space-ses1_hemi-L_vert-875.nii.gz', np.array([142,  48, 222]), addone=False)
    array([ -7.42000937, -92.96745521, -15.27866316])            
    """

    if len(coord) != 3:
        coord = coord[:3]

    if isinstance(anat, str):
        anat_img = nb.load(anat)
    elif isinstance(anat, nb.Nifti1Image) or isinstance(anat, nb.freesurfer.mghformat.MGHImage):
        anat_img = anat
    else:
        raise ValueError(
            "Unknown type for input image. Needs to be either a str or nibabel.Nifti1Image")

    if inv == False:
        coord = nb.affines.apply_affine(anat_img.affine, coord)
    else:
        coord = nb.affines.apply_affine(np.linalg.inv(anat_img.affine), coord)
        coord = [int(round(i, 0)) for i in coord]

    if addone == True:
        coord = np.append(coord, [1], axis=0)

    return coord

def mri_surf2surf(src_file=None, src_subj=None, trg_subj=None, out_file=None, hemi=None, return_file=False):
    """mri_surf2surf
    
    From https://nipype.readthedocs.io/en/latest/api/generated/nipype.interfaces.freesurfer.utils.html#surfacetransform: 
    Wrapped executable: mri_surf2surf. Transform a surface file from one subject to another via a spherical registration. Both the source and target subject must reside in your Subjects Directory, and they must have been processed with recon-all, unless you are transforming to one of the icosahedron meshes.

    Parameters
    ----------
    src_file: str, optional
        Surface file with source values. Maps to a command-line argument: `--sval %s`, by default None
    src_subj: str, optional
        Subject id for source surface. Maps to a command-line argument: `--srcsubject %s`, by default None
    trg_subj: str, optional
        Subject id of target surface. Maps to a command-line argument: `--trgsubject %s`, by default None
    out_file: str, optional
         Surface file to write. Maps to a command-line argument: `--tval %s`, by default None
    hemi: str, optional
        Hemisphere to transform. Maps to a command-line argument: `--hemi %s`, by default None

    Example
    ----------
    >>> from linescanning.transform import mri_surf2surf
    >>> surf_file = mri_surf2surf(src_file="lh.V1_fsaverage", src_subj="fsaverage", trg_subj="sub-001", out_file="lh.V1_fsnative", hemi="lh")
    """

    lh_acceptable = ["lh", "left", "l"]
    rh_acceptable = ["rh", "right", "r"]

    if hemi.lower in lh_acceptable:
        hemi = "lh"
    elif hemi.lower in rh_acceptable:
        hemi = "rh"
    else:
        raise ValueError(f"Specified hemisphere = '{hemi}', must be one of {lh_acceptable} or {rh_acceptable}")

    sxfm = fs.SurfaceTransform()
    sxfm.inputs.source_file     = src_file
    sxfm.inputs.source_subject  = src_subj
    sxfm.inputs.target_subject  = trg_subj
    sxfm.inputs.out_file        = out_file
    sxfm.inputs.hemi            = hemi
    sxfm.run()

    if return_file:
        return out_file