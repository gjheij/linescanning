# pylint: disable=no-member,E1130,E1137
import json
from linescanning import transform, utils, optimal
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import subprocess

opj = os.path.join

def slice_timings_to_json(
    json_file,
    nr_slices=None,
    tr=None,
    mb_factor=1):

    write_file = True
    if isinstance(json_file, str):
        with open(json_file) as f:
            data = json.load(f)
    elif isinstance(json_file, dict):
        write_file = False
        data = json_file
    else:
        raise ValueError(f"Input must be a json-file or dictionary, not {type(json_file)}")

    # TR from json file is usually more reliable
    if "RepetitionTime" in list(data.keys()):
        if data["RepetitionTime"] != tr:
            utils.verbose(f"WARNING: specified TR ({tr}) does not match TR in json file ({data['RepetitionTime']}). Taking TR from json file!", True)

            tr = data["RepetitionTime"]

    # get the list of slice timings
    st = slice_timings(
        nr_slices=nr_slices,
        tr=tr,
        mb_factor=mb_factor)

    # add slicetiming key
    st_dict = {"SliceTiming": st}
    data.update(st_dict)
    
    # write file or return updated dictionary
    if write_file:
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        return data

def slice_timings(
    nr_slices=None,
    tr=None,
    mb_factor=1):

    return list(np.tile(np.linspace(0, tr, int(nr_slices/mb_factor), endpoint=False), mb_factor))

def bin_fov(img, thresh=0,out=None, fsl=False):
    """bin_fov

    This function returns a binarized version of the input image. If no output name was specified,
    it will return the dataframe in nifti-format.

    Parameters
    ----------
    img: str
        path to input image
    thresh: int
        threshold to use (default = 0)
    out: str
        path to output image (default is None, and will return the data array of the image)
    fsl: bool
        if you reeeally want a binary image also run fslmaths -bin.. Can only be in combination with an output image and on a linux system with FSL (default is false)
    
    Example
    ---------
    >>> from linescanning.image import bin_fov
    >>> file = bin_fov("/path/to/image.nii.gz")
    >>> bin_fov("/path/to/image.nii.gz", thresh=1, out="/path/to/image.nii.gz", fsl=True)
    >>> bin_fov("/path/to/image.nii.gz", thres=2)
    """

    img_file = nb.load(img)                                     # load file
    img_data = img_file.get_fdata()                             # get data
    empty = np.zeros_like(img_data)
    empty[img_data <= thresh] = 1
    img_bin_img = nb.Nifti1Image(empty, header=img_file.header, affine=img_file.affine)

    if out != None:
        img_bin_img.to_filename(out)

        # also run fslmaths for proper binarization
        if fsl == True:
            cmd_txt = "fslmaths {in_img} -bin {out_img}".format(in_img=out, out_img=out)

            try:
                os.system(cmd_txt)
            except:
                raise Exception(f"Could not execute command '{cmd_txt}'")
    else:
        return img_bin_img


def reorient_img(img, code="RAS", out=None, qform="orig"):
    """reorient_img

    Python wrapper for fslswapdim to reorient an input image given an orientation code. Valid options are: RAS, AIL for now. You can also specify "nb" as code, which reorients the input image to nibabel's RAS+ convention. If no output name is given, it will overwrite the input image.

    Parameters
    ----------
    img: str
        nifti-image to be reoriented
    code: str, optional
        code for new orientation (default = "RAS")
    out: str, optional
        string to output nifti image
    qform: str,int, optional
        set qform code to original (str 'orig' = default) or a specified integer

    Returns
    ----------
    str
        if `out=="None"`, then `img` will receive the new coordinate code, otherwise `out` is created.
        The function only operates, it doesn't actually return something that can be captured in a variable

    Examples
    ----------
    >>> reorient_img("input.nii.gz", code="RAS", out="output.nii.gz")
    >>> reorient_img("input.nii.gz", code="AIL", qform=1)
    >>> reorient_img("input.nii.gz", code="AIL", qform='orig')
    """

    if out != None:
        new = out
    else:
        new = img

    if code.upper() != "NB":

        try:
            os.environ['FSLDIR']
        except:
            raise OSError(
                f'Could not find FSLDIR variable. Are we on a linux system with FSL?')

        pairs = {"L": "LR", "R": "RL", "A": "AP",
                 "P": "PA", "S": "SI", "I": "IS"}
        orient = "{} {} {}".format(
            pairs[code[0].upper()], 
            pairs[code[1].upper()], 
            pairs[code[2].upper()])
        cmd_txt = "fslswapdim {} {} {}".format(img, orient, new)
        print(cmd_txt)
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


def create_line_from_slice(
    in_file, 
    out_file=None, 
    width=16, 
    fold="FH", 
    keep_input=False,
    shift=0):

    """create_line_from_slice

    This creates a binary image of the outline of the line. The line's dimensions are 16 voxels of 0.25mm x 2.5 mm (slice thickness) and 0.25 mm (frequency encoding direction). We know that the middle of the line is at the center of the slice, so the entire line encompasses 8 voxels up/down from the center. This is represented by the 'width' flag, set to 16 by default

    Parameters
    ----------
    in_file: str
        path to image that should be used as reference (generally this should be the 1 slice file of the first run or something)
    out_file: str, optional
        path specifying the output name of the newly created 'line' or 'beam' file
    width: int, optional
        how many voxels should we use to define the line. Remember that the center of the line is width/2, so if it's set to 16, we take 8 voxels below center and 8 voxels above center.
    fold: str, optional
        string denoting the type of foldover direction that was used. We can find this in the info-file in the pycortex directory and can either be *FH* (line = `LR`), or *LR* (line = `FH`)
    keep_input: boolean, optional
        Keep the native input of the input data rather than binarizing the input image
    shift: int, optional
        Sometimes the slice had to be moved in `foldover` direction to place the saturation slabs in the right position. We need to correct for this. This argument moves the line `shift` millimeter in `fold` direction. For instance, if `fold="FH"` and `shift=2`, the line will be moved up. Use negative values to move the line down.
        
    Returns
    ----------
    nibabel.Nifti1Image
        if `out_file==None`, a niimg-object is returned

    str
        an actual file if the specified output name is created (*NOT RETURNED*)

    Examples
    ----------
    >>> img = create_line_from_slice("input.nii.gz")
    >>> img
    <nibabel.nifti1.Nifti1Image at 0x7f5a1de00df0>
    >>> img.to_filename('sub-001_ses-2_task-LR_run-8_bold.nii.gz')
    """

    img = False

    if isinstance(in_file, str):
        img = True
        in_img = nb.load(in_file)
        in_data = in_img.get_fdata()
        vox_size = in_img.header["pixdim"][1]
    elif isinstance(in_file, np.ndarray):
        in_data = in_file.copy()
        vox_size = 0.25

    if len(in_data.shape) == 4:
        in_data = np.squeeze(in_data, 3)
        
    empty_img = np.zeros_like(in_data)

    upper, lower = int((empty_img.shape[0]//2)+(int(width)/2)), int((empty_img.shape[0]//2)-(int(width)/2))

    # account for shift (assuming voxel size 0.25mm)
    if shift != 0:
        shift = int(round(shift/vox_size,0))

        upper += shift
        lower += shift
        
    # print(fold.lower())
    if fold.lower() == "rl" or fold.lower() == "lr":
        
        # add axis if we're dealing with nifti-image, otherwise return 2D-array
        beam = np.ones((int(width), empty_img.shape[0]))
        if img:
            beam = beam[...,np.newaxis]

        if keep_input == False:
            empty_img[lower:upper] = beam
        elif keep_input == True:
            empty_img[lower:upper] = beam*in_data[lower:upper]

    elif fold.lower() == "fh" or fold.lower() == "hf":

        # add axis if we're dealing with nifti-image, otherwise return 2D-array
        beam = np.ones((empty_img.shape[0], int(width)))
        if img:
            beam = beam[...,np.newaxis]

        if keep_input == False:
            empty_img[:,lower:upper] = beam
        elif keep_input == True:
            empty_img[:,lower:upper] = beam*in_data[:,lower:upper ]
            
    else:
        raise NotImplementedError(f"Unknown option {fold}, probably not implemented yet..")
    
    if img:
        line = nb.Nifti1Image(empty_img, affine=in_img.affine,header=in_img.header)
        if out_file != None:
            line.to_filename(out_file)
        else:
            return line
    else:
        return empty_img

def create_ribbon_from_beam(
    in_file, 
    ribbon,
    out_file=None):

    """create_ribbon_from_beam

    This creates a binary image of the outline of the ribbon based on the beam image (see :func:`linescanning.image.create_line_from_slice`). The line's dimensions are 16 voxels of 0.25mm x 2.5 mm (slice thickness) and 0.25 mm (frequency encoding direction). We know that the middle of the line is at the center of the slice, so the entire line encompasses 8 voxels up/down from the center. We then select only the voxels from `ribbon`.

    Parameters
    ----------
    in_file: str
        path to image that should be used as reference (generally this should be the 1 slice file of the first run or something)
    ribbon: list, tuple
        input representing the ribbon, e.g., `(358,363)`
    out_file: str, optional
        path specifying the output name of the newly created 'line' or 'beam' file
    
    Returns
    ----------
    nibabel.Nifti1Image
        if `out_file==None`, a niimg-object is returned
    str
        an actual file if the specified output name is created (*NOT RETURNED*)
    """

    beam_img = nb.load(in_file)
    beam_data = beam_img.get_fdata()

    # insert ribbon voxels
    rib_beam = np.zeros_like(beam_data)
    beam_loc = np.where(beam_data>0)
    rib_beam[ribbon[0]:ribbon[1],beam_loc[1][0]:beam_loc[1][-1]+1] = 1

    # save
    rib_img = nb.Nifti1Image(rib_beam, affine=beam_img.affine, header=beam_img.header)
    if out_file != None:
        rib_img.to_filename(out_file)
    else:
        return rib_img

def get_max_coordinate(in_img):

    """get_max_coordinate

    Fetches the point with the maximum value given an input image_file. Useful if you want to find the voxel with the highest value after warping a binary file with ANTs. Mind you, it outputs the VOXEL value, not the actual index of the array. The VOXEL value = idx_array+1 to account for different indexing in nifti & python.

    Parameters
    ----------
    in_img: str,numpy.ndarray,nibabel.Nifti1Image
        nibabel.Nifti1Image object, string to nifti image or numpy array

    Returns
    ----------
    np.ndarray
        if only 1 max value was detected

    list
        list of np.ndarray containing the voxel coordinates with max value

    Examples
    ----------
    >>> get_max_coordinate('sub-001_space-ses1_hemi-L_vert-875.nii.gz')
    array([142,  48, 222])
    >>> get_max_coordinate('sub-001_space-ses1_hemi-R_vert-6002.nii.gz')
    [array([139,  35, 228]), array([139,  36, 228])]
    """

    if isinstance(in_img, np.ndarray):
        img_data = in_img
    elif isinstance(in_img, nb.Nifti1Image):
        img_data = in_img.get_fdata()
    elif isinstance(in_img, str):
        img_data = nb.load(in_img).get_fdata()
    else:
        raise NotImplementedError(
            "Unknown input type; must either be a numpy array, a nibabel Nifti1Image, or a string pointing to a nifti-image")

    max_val = img_data.max()
    # .reshape(1,3).flatten()
    coord = np.array([np.where(img_data == max_val)[i] for i in range(0, 3)])

    if coord.shape[-1] > 1:
        l = []
        for i in np.arange(0, coord.shape[-1]):
            l.append(coord[:, i])
    else:
        l = coord[:, 0]

    return l


def get_isocenter(img):
    """get_isocenter

    This function returns the RAS coordinates of the input image's origin. This resembles the
    offset relative to the magnetic isocenter

    Parameters
    ----------
    img: str,nibabel.Nifti1Image
        input image to extract the isocenter coordinate from

    Returns
    ----------
    numpy.ndarray
        array containing isocenter coordinate from `img`

    Example
    ----------
    >>> img = 'sub-001_space-ses1_hemi-R_vert-6002.nii.gz'
    >>> get_isocenter(img)
    array([  0.27998984,   1.49000375, -15.34000604])
    """

    # get origin in RAS
    if isinstance(img, nb.Nifti1Image):
        fn = img
    elif isinstance(img, str):
        fn = nb.load(img)
    else:
        raise ValueError("Unknown input format for img")

    vox_center = (np.array(fn.shape) - 1) / 2.
    origin = transform.native_to_scanner(img, vox_center, addone=False)

    return origin


def bin_fov(img, thresh=0, out=None, fsl=False):
    """bin_fov

    This function returns a binarized version of the input image. If no output name was specified,
    it will return the dataframe in nifti-format

    Parameters
    ----------
    img: str
        path to input image
    thresh: int
        threshold to use (default = 0)
    out: str
        path to output image (default is None, and will return the data array of the image)
    fsl: bool
        if you reeeally want a binary image also run fslmaths -bin.. Can only be in combination with an output image and on a linux system with FSL (default is false)
    
    Returns
    ----------
    nibabel.Nifti1Image
        niimg-object with a binarized FOV of input image `img`

    Example
    ----------
    >>> file = bin_fov("/path/to/image.nii.gz")
    >>> bin_fov("/path/to/image.nii.gz", thresh=1, out="/path/to/image.nii.gz", fsl=True)
    >>> bin_fov("/path/to/image.nii.gz", thres=2)
    """

    img_file = nb.load(img)                                     # load file
    img_data = img_file.get_fdata()                             # get data
    # img_bin = (img_data > thresh).astype(int)                   # get binary mask where value != 0
    # img_bin = ndimage.binary_fill_holes(img_bin).astype(int)    # fill any holes
    img_data[img_data <= thresh] = 0
    img_bin_img = nb.Nifti1Image(
        img_data, header=img_file.header, affine=img_file.affine)

    if out != None:
        img_bin_img.to_filename(out)

        # also run fslmaths for proper binarization
        if fsl == True:
            cmd_txt = "fslmaths {in_img} -bin {out_img}".format(
                in_img=out, out_img=out)
            place = utils.get_base_dir()[1]

            if place != "win":
                os.system(cmd_txt)

    else:
        return img_bin_img

def mgz2nii(input, output=None, return_type='file'):
    """wrapper for call_mriconvert to convert mgz image to nifti"""

    if not isinstance(input, str):
        raise ValueError("Please use a string-like path to the file")

    if not output:
        output = input.split('.')[0]+'.nii.gz'

    cmd = ('call_mriconvert', input, output)
    L = utils.decode(subprocess.check_output(cmd))

    if return_type == "file":
        return output
    elif return_type == "nifti":
        return nb.load(output)
    else:
        raise NotImplementedError(f"Can't deal with return_type {return_type}. Please use 'file' or 'nifti'")


def nii2mgz(input, output=None, return_type='file'):
    """wrapper for mri_convert to convert mgz image to nifti"""

    import os
    import nibabel as nb

    if not isinstance(input, str):
        raise ValueError("Please use a string-like path to the file")

    if not output:
        output = input.split('.')[0]+'.mgz'

    os.system(f'mri_convert --in_type nii --out_type mgz {input} {output}')

    if return_type == "file":
        return output
    elif return_type == "mgz":
        return nb.load(output)
    else:
        raise NotImplementedError(f"Can't deal with return_type {return_type}. Please use 'file' or 'nifti'")

def tissue2line(data, line=None):
    """tissue2line

    Project tissue probability maps to the line by calculating the probability of each tissue type in each voxel of the 16x720 beam and then average these to get a 1x720 line. Discrete tissues are assigned by means of the highest probability of a particular tissue type.

    Parameters
    ----------
    data: list,numpy.ndarray,str
        for tissue data: list of three numpy array/nifti images/strings describing the probability of white matter/gray matter and CSF
    line: str,nibabel.Nifti1Image,numpy.ndarray
        used for the direction of the line and should have the same dimensions as `data`. Generally this is the output from create_line_from_slice

    Returns
    ----------
    numpy.ndarray
        (1,720) array of your `data` in the line
    """

    # load in reference line data
    if isinstance(line, str):
        ref = nb.load(line).get_fdata()
    elif isinstance(line, nb.Nifti1Image):
        ref = line.get_fdata()
    elif isinstance(line, np.ndarray):
        ref = line
    else:
        raise ValueError("Unknown input type for line; should be a string, nifti-image, or numpy array")

    if isinstance(data, list):
        # we have receive a list, assuming tissue probability maps.
        
        if len(data) > 3:
            raise ValueError(f'Data contains {len(data)} items, this should be three: 1) WM prob, 2) GM prob, 3) CSF prob')

        if isinstance(data[0], str):
            input = [nb.load(i).get_fdata() for i in data]
        elif isinstance(data[0], nb.Nifti1Image):
            input = [i.get_fdata() for i in data]
        elif isinstance(data[0], np.ndarray):
            input = data

        # remove existing 4th dimension
        input = [np.squeeze(i, axis=3) for i in input if len(i.shape) == 4]

        for i in input: 
            if i.shape != ref.shape:
                raise ValueError(f"Dimensions of line [{ref.shape}] do not match dimension of input seg [{i.shape}]")

        # put wm/gm/csf in three channels of a numpy array
        prob_stack = np.dstack([input[0],input[1],input[2]])
        prob_stack_avg = np.average(prob_stack, axis=1)

        # normalize averages between 0-1
        scaler = MinMaxScaler()
        scaler.fit(prob_stack_avg)
        avg_norm = scaler.transform(prob_stack_avg)

        output = []
        lut = {'wm':2,'gm':1,'csf':0}

        # avg_norm has 3 columns; 1st = WM, 2nd = GM, 3rd = CSF
        for i,r in enumerate(avg_norm):

            max_val = np.amax(r)

            # check tissue type only if non-zero value. If all probabilities are 0 is should be set to zero regardless
            if max_val == 0:
                output.append(lut['csf'])
            else:
                # make list of each row for nicer indexing
                idx = list(r).index(max_val)
                if idx == 0:
                    # type = 'wm' = '1' in nighres segmentation
                    output.append(lut['wm'])
                elif idx == 1:
                    # type = 'gm' = '2' in nighres segmentation
                    output.append(lut['gm'])
                elif idx == 2:
                    # type = 'csf' = '0' in nighres segmentation
                    output.append(lut['csf'])

        output = np.array(output)[:,np.newaxis]
        return output


def layers2line(data):
    """layers2line

    Project layer delineations to the line by calculating the probability of each layer membership in each voxel of the 16x720 beam and then average these to get a 1x720 line. Discrete layers are assigned by means of the highest probability of a particular layer membership.

    Parameters
    ----------
    data: list, numpy.ndarray, str
        (720,720) numpy array with the middle 16 rows containing information about the layers
    line: str,nibabel.Nifti1Image,numpy.ndarray
        used for the direction of the line and should have the same dimensions as `data`. Generally this is the output from create_line_from_slice

    Returns
    ----------
    numpy.ndarray
        (1,720) array of your `data` in the line
    """

    data[data == 0] = np.nan # convert zeros to NaN
    avg = np.nanmean(data, axis=1) # average by ignoring zeros
    avg = np.nan_to_num(avg) # convert zeros back to NaN otherwise int() will fail

    avg_layers = np.array([int(i) for i in avg if np.isnan(i) == False])[:,np.newaxis]

    return avg_layers

def clip_image(img, thr=None, val=None, return_type="image", out_file=None): 

    """clip_image

    This function takes an image or numpy array, and takes the <thr>-value as threshold to clip the maximum value of the input. This is especially useful to increase contrast in the masked_mp2rage-files. For some reason, the masking can be detrimental for the contrast resulting in failing segmentations. 

    Parameters
    ----------
    img: numpy.ndarray, nibabel.Nifti1Image, str
        input data; can either be a numpy array, a Nifti-image, or a string pointing to a nifti-file.
    thr: float, optional
        cut-off to use as threshold of image intensity. This represents 0.5% occuring values, after which a drop of intensities is seen.
    val: float, optional
        cut-off to use as threshold of image intensity. This represents a hard threshold of literal image values
    return_type: str, optional
        output type; can either be a numpy array ('arr'), a Nifti-image ('nib'), or a file ('file'). In the case of the latter, an output name should be provided
    out_file: str, optional
        string to output name if 'return_type="file"' is specified

    Returns
    ----------
    nibabel.Nifti1Image
        if `return_type=="file"`, a niimg-object is returned

    numpy.ndarray
        if `return_type=="arr"`, an array with the shape of `img` is returned

    str
        if `return_type=="file"`, a new file is created and nothing is returned

    Example
    ----------
    >>> new_img = clip_image("input.nii.gz", thr=0.005, return_type="nib")
    >>> clip_image("input.nii.gz", return_type='file', out_file='output.nii.gz')
    """

    if not img:
        raise ValueError("Need an input image")

    if isinstance(img, np.ndarray):
        input = img
        aff,hdr = None,None
    elif isinstance(img, nb.Nifti1Image):
        input = img.get_fdata()
        aff,hdr = img.affine, img.header
    elif isinstance(img, str):
        input = nb.load(img).get_fdata()
        aff,hdr = nb.load(img).affine, nb.load(img).header
    else:
        raise ValueError("Unknown input type; should be either a numpy.ndarray, nibabel.Nifti1Image, or str to nifti-file")

    if thr:
        ax = plt.hist(input.ravel(), bins=256)
        freq, vals = ax[0], ax[1]
        thr_freq = np.amax(freq)*thr
        idx = utils.find_nearest(freq, thr_freq)[0]
        cutoff = vals[idx]
    else:
        if val:
            cutoff = val
        else:
            raise ValueError("Either specify 'thr' or 'val'")

    new_data = np.clip(input,0,cutoff)

    if return_type == "arr":
        return new_data
    elif return_type == "nib":
        if not isinstance(aff, np.ndarray) and not isinstance(hdr, nb.Nifti1Header):
            raise ValueError("Need an affine-matrix and header for this operation; did you input a Nifti-like object or path to nifti-file?")

        return nb.Nifti1Image(new_data, affine=aff, header=hdr)
    elif return_type == "file":
        if not isinstance(aff, np.ndarray) and not isinstance(hdr, nb.Nifti1Header):
            raise ValueError("Need an affine-matrix and header for this operation; did you input a Nifti-like object or path to nifti-file?")

        if not out_file:
            raise ValueError("Please specify an output name for this operation")

        nb.Nifti1Image(new_data, affine=aff, header=hdr).to_filename(out_file)
    else:
        raise ValueError(f"Unknown option {return_type} specified; please use 'arr', 'nib', or 'file'")

def tsnr(img,file_name=None, clip=None):

    """tsnr

    This function calculates the temporal SNR of an input image. Can also create 3D tSNR
    map, but generally will just output the mean tSNR

    Parameters
    ----------
    img: numpy.ndarray, nibabel.Nifti1Image, str
        input data; can either be a numpy array, a Nifti-image, or a string pointing to a nifti-file.
    file_name: str, optional
        path to tSNR-map file. Set to None by default
    clip: int, float, optional
        lip the values to tSNR. Default is None

    Returns
    ----------
    numpy.ndarray
        tSNR in array with shape of `img` (minus last dimension if `img.shape`>3)

    Example
    ----------
    >>> tsnr = tsnr('path/to/func.nii.gz')
    """

    # ignore divide-by-zero error
    np.seterr(divide='ignore', invalid='ignore')

    if isinstance(img, nb.Nifti1Image):
        img = img
    elif isinstance(img, str):
        img = nb.load(img)
    else:
        raise ValueError("Unknown input type. Can be string pointing to nifti-file or an nb.Nifti1Image object")

    data = img.get_fdata()
    affine = img.affine
    hdr = img.header
    
    mean_d = np.mean(data,axis=-1)
    std_d = np.std(data,axis=-1)
    tsnr = mean_d/std_d
    tsnr[np.where(np.isinf(tsnr))] = 0

    if isinstance(clip, (int,float)):
        tsnr[tsnr>clip] = 0

    # calculate mean
    mean_tsnr = np.nanmean(np.ravel(tsnr))

    if isinstance(file_name, str):
        nb.Nifti1Image(
            tsnr,
            affine=affine, 
            header=hdr).to_filename(file_name)

    return mean_tsnr

class ROI():

    def __init__(
        self,
        roi,
        ):

        self.roi = roi

        # read parc data once even for multiple ROIs
        self.read_aparc()
        self.roi_list = self.read_roi_list()

        # force list
        if isinstance(self.roi, str):
            self.roi = [self.roi]

        # loop through list and merge
        self.individual_masks = {}
        if isinstance(self.roi, list):
            for roi in self.roi:
                if roi not in self.roi_list:
                    raise ValueError(f"'{roi}' is not part of the aparc atlas. Options are {self.roi_list}")
                
                self.individual_masks[roi] = self.make_roi_mask(roi=roi)

        # merge
        if len(self.individual_masks)>1:
            self.roi_mask = np.zeros_like(self.individual_masks[self.roi[0]])
            for _,val in self.individual_masks.items():
                self.roi_mask[val>0] = 1
        else:
            self.roi_mask = self.individual_masks[self.roi[0]].copy()

    def get_rois(self):
        return self.read_roi_list()
    
    def return_mask(self):
        return self.roi_mask
    
    def read_aparc(self):

        # GET VERICES FOR A SPECIFIC ROI 
        self.parc_data = optimal.SurfaceCalc.read_fs_annot(
            subject='fsaverage',
            fs_annot="aparc",
            hemi="both"
        )

    def read_roi_list(self):
        if not hasattr(self, "parc_data"):
            self.read_aparc()

        roi_list = []
        for i in self.parc_data["lh"][-1]:
            roi_list.append(i.decode())

        roi_list = [i for i in roi_list if i != "unknown"]
        return roi_list
    
    def make_roi_mask(self, roi=None):
        
        if not isinstance(roi, str):
            raise ValueError(f"Please specify an ROI to extract")


        tmp = {}
        for ii in ['lh', 'rh']:
            
            # get data
            parc_hemi = self.parc_data[ii][0]

            # get index of specified ROI in list | hemi doesn't matter here
            for ix,i in enumerate(self.parc_data[ii][2]):
                if roi.encode() in i:
                    break
            
            tmp[ii] = (parc_hemi == ix).astype(int)

        # Concat Hemis
        concat_hemis = np.hstack((tmp['lh'],tmp['rh']))

        return concat_hemis
