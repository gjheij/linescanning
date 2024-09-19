# pylint: disable=no-member,E1130,E1137
import cortex
from datetime import datetime
import json
from linescanning import (
    planning,
    plotting,
    dataset, 
    pycortex, 
    transform, 
    utils,
    prf
    )
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import os
import pandas as pd
from scipy import stats
from typing import Union
import time
import warnings

warnings.filterwarnings('ignore')
opj = os.path.join

def set_threshold(name=None, borders=None, set_default=None):

    """set_threshold

    This function is utilized in call_pycortex2 to fetch the thresholds for multiple properties including pRF-parameters (eccentricity, r2, and polar angle), and structural (surface) properties such as cortical thickness and sulcal depth. To make verbosity nicer, you can specify a 'name' to print to the terminal what kind of property is being processed. Then, you can specify a range where the user input should fall in (default is None to not force a specific range). You can also set a default value if you do not wish to set a certain threshold (usually this is the minimum/maximum of your array depending on what kind of thresholding you're about to do, e.g., 'greater/smaller than <value>').

    Parameters
    ----------
    name: str, optional
        For verbosity reasons, a string of the property's name that needs thresholding
    borders: tuple, optional
        Specific range the user input needs to fall in. Default is None to not enforce a range
    set_default: int, float, optional
        Minimum/maximum of array (depending on the kind of thresholding to be applied) if
        you do not wish to enforce a threshold

    Returns
    ----------
    float
        thresholds given `borders`

    Example
    ----------
    >>> ecc_val = set_threshold(name="eccentricity", range=(0,15), set_default=min(ecc_array))
    """

    if not name:
        name = "property"

    # set threshold for 'name'
    while True:
        try:
            # check if range is specified
            val = input(f" {name} [def = {set_default}]: ") or set_default
            if isinstance(val, str):
                if "," in val:
                    val = utils.string2list(val, make_float=True)
                else:
                    float(val)
            elif isinstance(val, (list,tuple)):
                pass
            else:
                val = float(val)
        except ValueError:
            print(" Please enter a number")
            continue
        else:
            pass

        if borders and len(borders) == 2:
            if isinstance(val, (list,tuple)):
                if val[0] >= borders[0] and val[1] <= borders[1]:
                    return val
                else:
                    utils.verbose(f" WARNING: specified range is ({borders[0]},{borders[1]}), your value is {val}. Try again..", True)
            else:
                if borders[0] <= float(val) <= borders[1]:
                    return float(val)
                else:
                    utils.verbose(f" WARNING: specified range is ({borders[0]},{borders[1]}), your value is {val}. Try again..", True)
                    continue

        else:
            return val


class SurfaceCalc(object):

    """SurfaceCalc 

    This object does all the surface initialization given a subject and a freesurfer directory. For instance, it reads in the curvature, thickness, and sulcal depth maps from freesurfer, smooths the curvature map, reads by default the V1_exvivo_thresh label in, and creates a boolean mask of this label. So with this one class you will have everything you need from the surface calculations.

    Parameters
    ----------
    subject: str
        subject ID as used in `SUBJECTS_DIR`
    fs_dir: str, optional
        `FreeSurfer` directory (default = SUBJECTS_DIR)
    fs_label: str, optional
        ROI-name to extract the vertex from as per ROIs created with `FreeSurfer`. Default is V1_exvivo.thresh
    aparc: bool, optional
        if True, `fs_label` is an ROI in the ?h.aparc.annot file (FreeSurfer atlas)

    Example
    ----------
    >>> surf_calcs = SurfaceCalc(subj="sub-001")

    Notes
    ----------
    Embedded in :class:`linescanning.optimal.CalcBestVertex`, so if you can also just call that class and you won't have to run the command in "usage"
    """

    def __init__(
        self, 
        subject=None, 
        fs_dir=None, 
        fs_label="V1_exvivo.thresh",
        aparc=False):

        # print(" Perform surface operations")
        self.subject = subject
        self.ctx_path = opj(cortex.database.default_filestore, self.subject)

        # check if we need to reload kernel to activate changes to filestore
        if os.environ.get("PROJECT") not in self.ctx_path:
            os.system('call_ctxfilestore update')
            import importlib
            importlib.reload(cortex)

        # double check:
        self.ctx_path = opj(cortex.database.default_filestore, self.subject)
        if os.environ.get("PROJECT") not in self.ctx_path:
            raise TypeError(f"Project '{os.environ.get('PROJECT')}' not found in '{self.ctx_path}'. This can happen if you changed the filestore, but haven't reloaded the kernel. Use 'call_ctxfilestore' to set the filestore (and reload window if running from VSCode)")
            
        if fs_dir == None:
            self.fs_dir = os.environ.get("SUBJECTS_DIR")
        else:
            self.fs_dir = fs_dir

        # define bunch of conditions when to import subject
        self.import_subj = False
        if not os.path.exists(self.ctx_path):
            self.import_subj = True
        else:
            if not "surfaces" in os.listdir(self.ctx_path):
                self.import_subj = True
            else:
                if len(os.listdir(opj(self.ctx_path, "surfaces")))==0:
                    self.import_subj = True

        if self.import_subj:
            # import subject from freesurfer (will have the same names)
            cortex.freesurfer.import_subj(
                fs_subject=self.subject,
                cx_subject=self.subject,
                freesurfer_subject_dir=self.fs_dir,
                whitematter_surf='smoothwm')
        
        # reload database after import
        cortex.db.reload_subjects()
        
        self.curvature = cortex.db.get_surfinfo(self.subject, type="curvature")
        self.thickness = cortex.db.get_surfinfo(self.subject, type="thickness")
        self.depth = cortex.db.get_surfinfo(self.subject, type="sulcaldepth")

        self.lh_surf_data, self.rh_surf_data = cortex.db.get_surf(self.subject, 'fiducial')
        self.lh_surf,self.rh_surf = cortex.polyutils.Surface(self.lh_surf_data[0], self.lh_surf_data[1]), cortex.polyutils.Surface(self.rh_surf_data[0], self.rh_surf_data[1])

        self.surf_coords = np.vstack([self.lh_surf_data[0],self.rh_surf_data[0]])
        # Normal vector for each vertex (average of normals for neighboring faces)
        self.lh_surf_normals = self.lh_surf.vertex_normals
        self.rh_surf_normals = self.rh_surf.vertex_normals
        self.smooth_surfs(kernel=3,nr_iter=3)

        # concatenate into one array
        self.surf_sm = np.squeeze(np.vstack([self.lh_surf_sm[...,np.newaxis],self.rh_surf_sm[...,np.newaxis]]))

        # try:
        setattr(self, 'roi_label', fs_label.replace('.', '_'))
        make_mask = False
        if not aparc:
            if not fs_label.endswith('.gii'):
                make_mask = True
                tmp = self.read_fs_label(subject=self.subject, fs_dir=self.fs_dir, fs_label=fs_label, hemi="both")
            else:
                tmp = self.read_gii_label()
        else:
            tmp = self.read_aparc_label()

        setattr(self, f'lh_{self.roi_label}', tmp['lh'])
        setattr(self, f'rh_{self.roi_label}', tmp['rh'])

        # this way we can also use read_fs_label for more custom purposes
        if make_mask:
            pp = self.label_to_mask(subject=self.subject, lh_arr=getattr(self, f'lh_{self.roi_label}'), rh_arr=getattr(self, f'rh_{self.roi_label}'), hemi="both")
            self.lh_roi_mask = pp['lh_mask']
            self.rh_roi_mask = pp['rh_mask']
            self.whole_roi   = pp['whole_roi']
            self.whole_roi_v = pp['whole_roi_v']
        else:
            self.lh_roi_mask = getattr(self, f'lh_{self.roi_label}')
            self.rh_roi_mask = getattr(self, f'rh_{self.roi_label}')
            self.whole_roi   = np.concatenate((self.lh_roi_mask, self.rh_roi_mask))

            self.whole_roi_v = pycortex.Vertex2D_fix(
                self.whole_roi,
                subject=subject)

    def read_gii_label(self):
        tmp = {}
        for ii in ['lh', 'rh']:
            gifti = dataset.ParseGiftiFile(opj(self.fs_dir, self.subject, 'label', f"{ii}.{self.fs_label}"), set_tr=1)

            if gifti.data.ndim > 1:
                tmp[ii] = np.squeeze(gifti.data, axis=0)
            else:
                tmp[ii] = gifti.data.copy()

        return tmp

    def read_aparc_label(self):

        # read aparc.annot
        self.aparc_data = self.read_fs_annot(self.subject, fs_annot="aparc")
        
        tmp = {}
        for ii in ['lh', 'rh']:
            
            # get data
            self.aparc_hemi = self.aparc_data[ii][0]

            # initialize empty array
            self.empty = np.full_like(self.aparc_hemi, np.nan)

            # get index of specified ROI in list | hemi doesn't matter here
            for ix,i in enumerate(self.aparc_data[ii][2]):
                if self.fs_label.encode() in i:
                    break

            tmp[ii] = (self.aparc_hemi == ix).astype(int)

        return tmp
        
    def smooth_surfs(self, kernel=10, nr_iter=1):
        """smooth_surfs

        smooth surfaces with a given kernel size. The kernel size does not refer to mm, but to a factor. Therefore it has to be an integer value

        Parameters
        -----------
        kernel: int
            size of kernel to use for smoothing
        nr_iter: int
            number of iterations

        Returns
        ----------
        sets the attributes `self.?h_surf_sm`
        """

        if not isinstance(kernel,int):
            print(f" Rounding smoothing factor '{kernel}' to '{int(kernel)}'")
            kernel = int(kernel)
        setattr(self, 'lh_surf_sm', self.lh_surf.smooth(self.curvature.data[:self.lh_surf_data[0].shape[0]], factor=kernel, iterations=nr_iter))
        setattr(self, 'rh_surf_sm', self.rh_surf.smooth(self.curvature.data[self.lh_surf_data[0].shape[0]:], factor=kernel, iterations=nr_iter))

    @staticmethod
    def read_fs_annot(
        subject, 
        fs_dir=None, 
        fs_annot=None, 
        hemi="both"):
        
        """read_fs_annot

        read a freesurfer annot file (name must match with file in freesurfer directory)
        
        Parameters
        -----------
        subject: str
            subject ID as used in `SUBJECTS_DIR`
        fs_dir: str, optional
            `FreeSurfer` directory (default = SUBJECTS_DIR)
        fs_annot: str, optional
            ROI-name to extract the vertex from as per ROIs created with `FreeSurfer`. Default is V1_exvivo.thresh
        hemi: str, optional
            For which hemisphere to perform the process, `lh`=left hemisphere, `rh`=right hemisphere, `both`=both hemispheres (default = `both`)

        Returns
        ----------
        dict
            Dictionary collecting outputs under the following keys

            * lh: output from `nibabel.freesurfer.io.read_annot`
            * rh: output from `nibabel.freesurfer.io.read_annot`
        """

        if not fs_dir:
            fs_dir = os.environ.get("SUBJECTS_DIR")

        if hemi == "both":
            return {
                'lh': nb.freesurfer.io.read_annot(
                    opj(
                        fs_dir, 
                        subject, 
                        'label', 
                        f'lh.{fs_annot}.annot')
                    ),
                'rh': nb.freesurfer.io.read_annot(
                    opj(
                        fs_dir, 
                        subject, 
                        'label', 
                        f'rh.{fs_annot}.annot'))
                }

        else:
            if hemi.lower() != "lh" and hemi.lower() != "rh":
                raise ValueError(f"Hemisphere should be one of 'both', 'lh', or 'rh'; not {hemi}")
            else:
                annot_file = opj(
                    fs_dir, 
                    subject, 
                    'label', 
                    f'{hemi}.{fs_annot}.annot')

                return {hemi: nb.freesurfer.io.read_annot(annot_file)}

    @staticmethod
    def read_fs_label(
        subject, 
        fs_dir=None, 
        fs_label=None, 
        hemi="both"):

        """read_fs_label

        read a freesurfer label file (name must match with file in freesurfer directory)
        
        Parameters
        -----------
        subject: str
            subject ID as used in `SUBJECTS_DIR`
        fs_dir: str, optional
            `FreeSurfer` directory (default = SUBJECTS_DIR)
        fs_label: str, optional
            ROI-name to extract the vertex from as per ROIs created with `FreeSurfer`. Default is V1_exvivo.thresh
        hemi: str, optional
            For which hemisphere to perform the process, `lh`=left hemisphere, `rh`=right hemisphere, `both`=both hemispheres (default = `both`)

        Returns
        ----------
        dict
            Dictionary collecting outputs under the following keys

            * lh: output from `nibabel.freesurfer.io.read_label`
            * rh: output from `nibabel.freesurfer.io.read_label`
        """

        if not fs_dir:
            fs_dir = os.environ.get("SUBJECTS_DIR")

        if hemi == "both":
            return {
                'lh': nb.freesurfer.io.read_label(
                    opj(
                        fs_dir, 
                        subject, 
                        'label', 
                        f'lh.{fs_label}.label')
                    ),
                'rh': nb.freesurfer.io.read_label(
                    opj(
                        fs_dir, 
                        subject, 
                        'label', 
                        f'rh.{fs_label}.label'))
                }

        else:
            if hemi.lower() != "lh" and hemi.lower() != "rh":
                raise ValueError(f"Hemisphere should be one of 'both', 'lh', or 'rh'; not {hemi}")
            else:
                label_file = opj(
                    fs_dir, 
                    subject, 
                    'label', 
                    f'{hemi}.{fs_label}.annot')

                return {hemi: nb.freesurfer.io.read_annot(label_file)}
    
    def label_to_mask(self, subject=None, lh_arr=None, rh_arr=None, hemi="both"):
        """label_to_mask

        Convert freesurfer label or set of vertices to boolean vector
        
        Parameters
        -----------
        subject: str
            subject ID as used in `SUBJECTS_DIR`
        lh_arr: numpy.ndarray, optional
            array containing the mask in the left hemisphere (can be read from :class:`linescanning.optimal.SurfaceCalc` itself)
        rh_arr: numpy.ndarray, optional
            array containing the mask in the right hemisphere (can be read from :class:`linescanning.optimal.SurfaceCalc` itself)
        hemi: str, optional
            For which hemisphere to perform the process, `lh`=left hemisphere, `rh`=right hemisphere, `both`=both hemispheres (default = `both`)

        Returns
        ----------
        dict
            Dictionary collecting outputs under the following keys

            * lh_mask: boolean numpy.ndarray in the left hemisphere
            * rh_mask: boolean numpy.ndarray in the left hemisphere
            * whole_roi: mask of both hemispheres combined
            * whole_roi_v: cortex.Vertex object of `whole_roi`
        """

        if hemi == "both":
            
            lh_mask = np.zeros(self.lh_surf_data[0].shape[0], dtype=bool)
            lh_mask[lh_arr] = True

            rh_mask = np.zeros(self.rh_surf_data[0].shape[0], dtype=bool)
            rh_mask[rh_arr] = True

        elif hemi == "lh":
            lh_mask = np.zeros(getattr(self, f"lh_surf_data")[0].shape[0], dtype=bool)
            lh_mask[lh_arr] = True
            rh_mask = np.zeros(getattr(self, f"rh_surf_data")[0].shape[0], dtype=bool)

        elif hemi == "rh":
            lh_mask = np.zeros(getattr(self, f"lh_surf_data")[0].shape[0], dtype=bool)
            rh_mask = np.zeros(getattr(self, f"rh_surf_data")[0].shape[0], dtype=bool)    
            rh_mask[rh_arr] = True
        else:
            raise ValueError(f"Invalid option '{hemi}' for hemi. Must be one of 'both', 'lh', or 'rh'")

        whole_roi = np.concatenate((lh_mask, rh_mask))
        whole_roi_v = pycortex.Vertex2D_fix(
            whole_roi,
            subject=subject)
        
        return {
            'lh_mask': lh_mask,
            'rh_mask': rh_mask,
            'whole_roi': whole_roi,
            'whole_roi_v': whole_roi_v}


class Neighbours(SurfaceCalc):

    def __init__(
        self, 
        subject:str=None, 
        fs_dir:str=None, 
        fs_label:str="V1_exvivo.thresh",
        hemi:str="lh",
        verbose:bool=False,
        aparc:bool=False,
        **kwargs):

        self.subject = subject
        self.fs_dir = fs_dir
        self.fs_label = fs_label
        self.hemi = hemi
        self.verbose = verbose
        self.aparc = aparc
        self.__dict__.update(kwargs)

        # pycortex path will be read from 'filestore' key in options.cfg file from pycortex
        if not isinstance(self.fs_dir, str):
            self.fs_dir = os.environ.get("SUBJECTS_DIR")

        allowed_options = ["lh","left","rh","right","both"]
        if self.hemi == "both":
            self.hemi_list = ["lh","rh"]
        elif self.hemi == "lh" or self.hemi == "left":
            self.hemi_list = ["lh"]
        elif self.hemi == "rh" or self.hemi == "right":
            self.hemi_list = ["right"]
        else:
            raise ValueError(f"'hemi' must one of {allowed_options}")

        # set flag for subsurface or not
        self.sub_surface = False
        if isinstance(self.fs_label, str):
            self.sub_surface = True

        # initialize SurfaceCalc
        if not hasattr(self, "lh_surf"):
            utils.verbose("Initializing SurfaceCalc", self.verbose)

            super().__init__(
                subject=self.subject, 
                fs_dir=self.fs_dir,
                fs_label=self.fs_label,
                aparc=self.aparc)

        if isinstance(self.fs_label, str):
            self.create_subsurface()

        if hasattr(self, "target_vert"):
            if isinstance(self.target_vert, int):
                self.distances_to_target(self.target_vert, self.hemi)

    def create_subsurface(self):
        utils.verbose(f"Creating subsurfaces for {self.fs_label}", self.verbose)
                    
        for mask,surf,attr in zip(
            ["lh_roi_mask","rh_roi_mask"],
            [self.lh_surf,self.rh_surf],
            ["lh_subsurf","rh_subsurf"]):

            # create the subsurface
            subsurface = surf.create_subsurface(vertex_mask=getattr(self, mask))
            setattr(self, attr, subsurface)

        # get the vertices belonging to subsurface
        self.lh_subsurf_v = np.where(self.lh_subsurf.subsurface_vertex_map != stats.mode(self.lh_subsurf.subsurface_vertex_map)[0][0])[0]

        self.rh_subsurf_v = np.where(self.rh_subsurf.subsurface_vertex_map != stats.mode(self.rh_subsurf.subsurface_vertex_map)[0][0])[0]+ self.lh_subsurf.subsurface_vertex_map.shape[-1]

        self.leftlim = np.max(self.lh_subsurf_v)
        self.subsurface_verts = np.concatenate([self.lh_subsurf_v, self.rh_subsurf_v])

    def smooth_subsurface(
        self, 
        data, 
        kernel=1, 
        iterations=1):
        
        data_sm = np.full(data.shape[0], np.nan)

        # smooth distance map on V1 subsurfaces
        for vert,surf in zip(
            [self.lh_subsurf_v,self.rh_subsurf_v],
            [self.lh_subsurf,self.rh_subsurf]):

            data_sm[vert] = surf.smooth(
                data[vert], 
                factor=kernel, 
                iterations=iterations)

        return data_sm
    def create_distance(self):
        # Make the distance x distance matrix.
        ldists, rdists = [], []

        utils.verbose('Creating distance by distance matrices', self.verbose)

        for i in range(len(self.lh_subsurf_v)):
            ldists.append(self.lh_subsurf.geodesic_distance([i]))

        self.dists_L = np.array(ldists)

        for i in range(len(self.rh_subsurf_v)):
            rdists.append(self.rh_subsurf.geodesic_distance([i]))

        self.dists_R = np.array(rdists)

        # Pad the right hem with np.inf.
        padL = np.pad(
            self.dists_L, ((0, 0), (0, self.dists_R.shape[-1])), constant_values=np.Inf)
        # pad the left hem with np.inf..
        padR = np.pad(
            self.dists_R, ((0, 0), (self.dists_L.shape[-1], 0)), constant_values=np.Inf)

        self.distance_matrix = np.vstack([padL, padR])  # Now stack.
        self.distance_matrix = (self.distance_matrix + self.distance_matrix.T)/2 # Make symmetrical        

    def distances_to_target(self, target_vert:int=None, hemi:str=None, vert_dict:Union[dict,str]=None):

        txt = "full surface"
        if self.sub_surface:
            txt = self.fs_label

        if not isinstance(vert_dict, (dict,str)):

            if isinstance(self.fs_label, str):
                use_surf = getattr(self, f"{hemi}_subsurf")
                use_vert = getattr(self, f"{hemi}_subsurf_v")

                # find the index of the target vertex within the subsurface
                target_wb = target_vert
                target_vert = np.where(use_vert == target_vert)[0][0]
                
                utils.verbose(f"Target vertex '{target_wb}' is at index '{target_vert}' in subsurface", self.verbose)
            else:
                use_surf = getattr(self, f"{hemi}_surf")

            utils.verbose(f"Finding distances from {txt} to vertex #{target_vert}", self.verbose)
                
            # get distance to target
            dist_to_targ = use_surf.geodesic_distance(target_vert)

            # make into convenient dictionary index on whole-brain
            self.tmp = {}
            self.df = {}
            self.df["idx"] = np.zeros((dist_to_targ.shape[0]))
            self.df["distance"] = np.zeros((dist_to_targ.shape[0]))
            for ii in range(dist_to_targ.shape[0]):
                self.tmp[use_vert[ii]] = dist_to_targ[ii]
                self.df["idx"][ii] = use_vert[ii]
                self.df["distance"][ii] = dist_to_targ[ii]
                
                # make dataframe
                self.df = pd.DataFrame(self.df)

            # store
            setattr(self, f"{hemi}_dist_to_targ_arr", dist_to_targ)
            setattr(self, f"{hemi}_dist_to_targ", self.tmp)
            
        else:
            if isinstance(vert_dict, str):
                utils.verbose(f"Reading distances from {vert_dict}", self.verbose)
                with open(vert_dict) as f:
                    dist_to_targ = json.load(f)

                # store
                setattr(self, f"{hemi}_dist_to_targ", dist_to_targ)
                data = np.array(list(getattr(self, f"{hemi}_dist_to_targ").items()))
                self.df = pd.DataFrame(data, columns=["idx","distance"])

        convert_dict = {
            'idx': int,
            'distance': float
            }

        self.df = self.df.astype(convert_dict)
        setattr(self, f"{hemi}_dist_to_targ_df", self.df)

    def find_distance_range(
        self, 
        hemi:str="lh", 
        vmin:Union[int,float]=None, 
        vmax:Union[int,float]=None):

        """find_distance_range

        Find the vertices given a range of distance using `vmin` and `vmax`. For instance, if you want the vertices within 2mm of your target vertex, specify `vmax=2`. If you want the vertices within 2-4mm, specify `vmin=2,vmax=4`. If vertices are found given your criteria, a dictionary collecting key-value pairings between the vertices and their distances to the target will be returned.

        Parameters
        ----------
        hemi: str, optional
            Which hemisphere to use. Should be one of ['lh','rh'], by default "lh"
        vmin: Union[int,float], optional
            Minimum distance to extract, by default None
        vmax: Union[int,float], optional
            Maximum distance to extract, by default None
        verbose: bool, optional
            Turn on verbosity, by default False

        Returns
        ----------
        dict
            Key-value pairing between the vertices surviving the criteria + their distance to the target vertex

        Raises
        ----------
        ValueError
            If both `vmin` and `vmax` are `None`. One or both should be specified
        ValueError
            If no vertices could be found with the criteria

        Example
        ----------
        >>> from linescanning.pycortex import Neighbours
        >>> # initialize the class, which - given a target vertex - will start to do some
        >>> # initial calculations
        >>> nbr = Neighbours(
        >>>     subject="sub-001",
        >>>     target=1053,
        >>>     verbose=True)
        >>> #
        >>> # call find_distance_range
        >>> nbr.find_distance_range(vmin=2,vmax=4)
        """
        # check if we ran distances_to_target()
        if hasattr(self, f"{hemi}_dist_to_targ"):
            distances = getattr(self, f"{hemi}_dist_to_targ")

            if isinstance(distances, np.ndarray):
                if isinstance(vmin, (int,float)) and not isinstance(vmax, (int,float)):
                    result = np.where(distances >= vmin)
                elif not isinstance(vmin, (int,float)) and isinstance(vmax, (int,float)):
                    result = np.where(distances <= vmax)
                elif isinstance(vmin, (int,float)) and isinstance(vmax, (int,float)):
                    result = np.where((distances >= vmin) & (distances <= vmax))
                else:
                    raise ValueError("Not sure what to do. Please specify 'vmin' and/or 'vmax', or use 'self.<hemi>_dist_to_target to fetch all distances")

                try:
                    verts = result[0]
                    if self.verbose:
                        utils.verbose(f"Found {len(verts)} vertices to target for vmin={vmin} & vmax={vmax}", self.verbose)
                    output = {}
                    for ii in verts:
                        output[ii] = distances[ii]
                    
                    return output

                except:
                    raise ValueError(f"Could not find vertices complying to criteria: vmin = {vmin}; vmax = {vmax}")
            
            elif isinstance(distances, dict):
                if isinstance(vmin, (int,float)) and not isinstance(vmax, (int,float)):
                    result = [int(k) for k, v in distances.items() if v >= vmin]
                elif not isinstance(vmin, (int,float)) and isinstance(vmax, (int,float)):
                    result = [int(k) for k, v in distances.items() if v <= vmax]
                elif isinstance(vmin, (int,float)) and isinstance(vmax, (int,float)):
                    result = [int(k) for k, v in distances.items() if vmin <= v <= vmax]
                else:
                    raise ValueError("Not sure what to do. Please specify 'vmin' and/or 'vmax', or use 'self.<hemi>_dist_to_target to fetch all distances")

                return result

class pRFCalc():

    """pRFCalc

    This short class deals with the population receptive field modeling output from spinoza_fitprfs and/or call_prfpy. Ideally, the output of these scripts is a numpy array containing the 6 pRF-parameters for each voxel. If you have one of those files, specify it in the `prf_file` argument. If, for some reason, you do not have this file, but separate files for each pRF variable (e.g., a file for R2, a file for eccentricitry, and a file for polar angle), make sure they are all in 1 directory and specify that directory in the `prf_dir` parameter. The only function of this class is to collect path names and the data arrays containing information about the pRF parameters. It will actually be used in :class:`linescanning.optimal.CalcBestVertex`.

    Parameters
    ----------
    prf_file: str, optional
        Path to a desc-prf_params.pkl file containing a 6xn dataframe representing the 6 variables from the pRF-analysis times the amount of TRs (time points). You can either specify this file directly or specify the pRF-directory containing separate files for R2, eccentricity, and polar angle if you do not happen to have the prf parameter file
    subject (str): str, optional
        subject ID as used in SUBJECTS_DIR. If `prf_file` is a BIDS-file and contains 'sub', we'll set `subject` to that. Subject ID is required if you want to produce `cortex.Vertex`-objects. In that case, the subject ID must match that of FreeSurfer.
    thr: float, optional
        Threshold the pRF-estimates with variance explained
    save: bool, optional
        Save numpy-arrays of pRF-estimates (eccentricity, size, polar angle, and r2)
    fs_dir: str, optional
        Used if the subject does not exist in pycortex' filestore. If empty, we'll read from *SUBJECTS_DIR*

    Returns
    ----------
    :class:
        Several attributes will be set upon calling the class. These attributes will be necessary to complete :class:`linescanning.optimal.CalcBestVertex`

    Example
    ----------
    >>> prf = pRFCalc("sub-001_desc-prf_params.pkl", thr=0.05)
    """

    # Get stuff from SurfaceCalc
    def __init__(
        self, 
        prf_file, 
        subject=None, 
        save=False, 
        model=None,
        thr=0.1,
        fs_dir=None,
        skip_cortex=False):
        
        # set defaults
        self.prf_file   = prf_file
        self.model      = model
        self.save       = save
        self.subject    = subject
        self.thr        = thr
        self.fs_dir     = fs_dir
        self.skip_cortex = skip_cortex

        # check SUBJECTS_DIR
        if not isinstance(self.fs_dir, str):
            self.fs_dir = os.environ.get("SUBJECTS_DIR")
        
        # do stuff if file exists
        if isinstance(self.prf_file, str):
            if os.path.exists(self.prf_file):
                # read BIDS components from prf-file
                self.comps = utils.split_bids_components(os.path.basename(self.prf_file))

                # set subject
                if not isinstance(self.subject, str):
                    if "sub" in list(self.comps.keys()):
                        self.subject = f"sub-{self.comps['sub']}"

                # set model
                if not isinstance(self.model, str):
                    if "model" in list(self.comps.keys()):
                        self.model = self.comps["model"]
                    else:
                        self.model = "gauss"
                
                # create output string from input file if we found BIDS components
                self.out_ = ""
                for el in list(self.comps.keys()):
                    if el != "desc":
                        self.out_ += f"{el}-{self.comps[el]}_"

                if len(self.out_) != 0:
                    self.out_ += "desc-"

                # read file
                self.prf_params = prf.read_par_file(self.prf_file)
                self.prf_dir = os.path.dirname(self.prf_file)

                # obtained pRF parameters
                self.df_prf = prf.Parameters(self.prf_params, model=self.model).to_df()

            else:
                raise FileNotFoundError(f"Could not find file '{self.prf_file}'")
            
        
            # find max r2
            self.max_r2 = np.amax(self.df_prf["r2"].values)
            self.max_r2_vert = np.where(self.df_prf["r2"].values == self.max_r2)[0][0]

            if not self.skip_cortex:
                if isinstance(self.subject, str):
                    
                    # read pycortex filestore
                    ctx_path = pycortex.set_ctx_path(opt="show_fs")

                    # filestore needs to exist to 'import_subj' to work
                    if not os.path.exists(ctx_path):
                        os.makedirs(ctx_path, exist_ok=True)

                    if not os.path.exists(opj(ctx_path, self.subject)):
                        cortex.freesurfer.import_subj(
                            fs_subject=self.subject,
                            cx_subject=self.subject,
                            freesurfer_subject_dir=self.fs_dir,
                            whitematter_surf='smoothwm')

                    # make object for r2
                    self.r2_v = pycortex.Vertex2D_fix(
                        self.df_prf.r2,
                        self.df_prf.r2,
                        subject=self.subject,
                        cmap="magma",
                        vmax1=round(self.df_prf.r2.max(),2),
                        vmin2=self.thr,
                        vmax2=0.5)

                    # make object for eccentricity
                    self.ecc_v = pycortex.Vertex2D_fix(
                        self.df_prf.ecc,
                        self.df_prf.r2,
                        subject=self.subject,
                        cmap="nipy_spectral",
                        vmax1=5,
                        vmin2=self.thr,
                        vmax2=0.5)

                    # make object for polar angle
                    self.polar_v = pycortex.Vertex2D_fix(
                        self.df_prf.polar,
                        self.df_prf.r2,
                        subject=self.subject,
                        cmap="hsvx2",
                        vmin1=-np.pi,
                        vmax1=np.pi,
                        vmin2=self.thr,                
                        vmax2=0.5)
                    
                    self.prf_data_dict = {}
                    for el in ["r2","ecc","polar"]:
                        self.prf_data_dict[el] = getattr(self, f"{el}_v")
                    
                    # create objects for A,B,C,D
                    if self.model == "norm":
                        for par in ["B","D","ratio (B/D)"]:
                            if par in ["B","D"]:
                                minmax = [0,100]
                                obj_par = par
                            else:
                                minmax = [
                                    np.nanquantile(self.df_prf["ratio (B/D)"].loc[self.df_prf.r2>self.thr].values,0.1),
                                    round(np.nanquantile(self.df_prf["ratio (B/D)"].loc[self.df_prf.r2>self.thr].values,0.9),2)
                                ]
                                obj_par = "ratio_bd"
                            
                            data = self.df_prf[par]
                            # data[data>minmax[1]] = 0
                            obj_ = pycortex.Vertex2D_fix(
                                data,
                                self.df_prf.r2,
                                subject=self.subject,
                                vmin1=minmax[0],
                                vmax1=minmax[1],
                                vmin2=self.thr,
                                vmax2=0.5,
                                cmap="inferno"
                            )

                            setattr(self, f"{obj_par}_v", obj_)
                            self.prf_data_dict[obj_par] = obj_

    def open_pycortex(
        self,
        **kwargs):

        self.pyc = pycortex.SavePycortexViews(
            self.prf_data_dict,
            subject=self.subject,
            **kwargs)

class CalcBestVertex():

    """CalcBestVertex

    This class actually takes in all attributes from pRFCalc and SurfaceCalc and combines the surface information contained in the surface-class with the information from the pRF-class. Specifically, it will call upon :class:`linescanning.optimal.SurfaceCalc` and :class:`linescanning.optimal.pRFCalc` as init function giving you complete access to anatomical and functional parameter estimates with just calling this class.

    Parameters
    ----------
    subject: str
        Subject ID as used in `SUBJECTS_DIR`
    deriv: str, optional
        Path to derivatives folder of the project. Generally should be the path specified with `DIR_DATA_DERIV` in the bash environment. This option overwrite the individual specification of `prf_dir`, `cx_dir`, and `fs_dir`, and will look up defaults.
    prf_file: str, optional
        Path to a desc-prf_params.pkl file containing a 6xn dataframe representing the 6 variables from the pRF-analysis times the amount of TRs (time points). You can either specify this file directly or specify the pRF-directory containing separate files for R2, eccentricity, and polar angle if you do not happen to have the prf parameter file        
    epi_file: str, np.ndarray, bool, optional
        Path to or numpy array containing either time course or averaged BOLD data. Can be used to exclude areas with veins (will have low EPI intensities). Format of time course data should be (time,vertices) to ensure averaging over the correct axis.
    fs_label: str, optional
        ROI-name to extract the vertex from as per ROIs created with `FreeSurfer`. Default is V1_exvivo.thresh
    use_epi: bool, optional
        try to look in the fMRIprep directory for functional data which we can use to exclude vertices with a low intensity. These vertices often denote veins, so we'd like to exclude those when using functional data (e.g., pRF)
        
    Example
    ----------
    >>> BV_ = CalcBestVertex(subject=subject, fs_dir='/path/to/derivatives/freesurfer'), prf_file=prf_params)
    """

    def __init__(
        self, 
        subject=None,
        deriv=None,
        prf_file=None,
        epi_file=None,
        use_epi=False,
        epi_space="fsnative",
        model="gauss",
        fs_label="V1_exvivo.thresh",
        verbose=False,
        aparc=False):

        # read arguments
        self.subject    = subject
        self.deriv      = deriv
        self.prf_file   = prf_file
        self.epi_file   = epi_file
        self.use_epi    = use_epi
        self.epi_space  = epi_space
        self.fs_label   = fs_label
        self.model      = model
        self.verbose    = verbose
        self.aparc      = aparc

        # set EPI=True if a file/array is specified
        if isinstance(self.epi_file, (np.ndarray, str, bool)):
            self.use_epi = True

        # set default derivatives
        if self.deriv == None:
            self.deriv = os.environ.get("DIR_DATA_DERIV")

        # set default freesurfer directory
        self.fs_dir = opj(self.deriv, 'freesurfer')
        self.cx_dir = opj(self.deriv, 'pycortex')

        # Get surface object
        self.surface = Neighbours(
            subject=self.subject, 
            fs_dir=self.fs_dir, 
            fs_label=self.fs_label,
            verbose=self.verbose,
            aparc=self.aparc)

        if self.prf_file != None:
            self.prf_dir = os.path.dirname(self.prf_file)
            self.prf = pRFCalc(
                subject=self.subject, 
                prf_file=self.prf_file, 
                fs_dir=self.fs_dir, 
                model=self.model)

        if self.use_epi:
            
            # try to find fmriprep output
            if not isinstance(self.epi_file, (str,np.ndarray)):
                fprep_dir = opj(self.deriv, "fmriprep", self.subject)
                if not os.path.exists(fprep_dir):
                    raise FileNotFoundError(f"Could not find directory: '{fprep_dir}'")
                
                gii_files = utils.FindFiles(fprep_dir, extension="gii").files

                if len(gii_files) == 0:
                    raise ValueError(f"No files with 'gii' in '{fprep_dir}'")
                
                gii_files_filt = utils.get_file_from_substring([self.epi_space], gii_files)[:2]
                
                tmp = np.hstack([dataset.ParseGiftiFile(ii).data for ii in gii_files_filt])
                self.epi = tmp.mean(axis=0)

            else:
                
                if isinstance(self.epi_file, str):
                    # read string
                    tmp = prf.read_par_file(self.epi_file)
                elif isinstance(self.epi_file, np.ndarray):
                    # copy numpy array
                    tmp = self.epi_file.copy()
                else:
                    # throw error
                    raise ValueError(f"Input must be a string point to (time,vertices) file or a (time,vertices) numpy array, not {type(self.epi_file)}")
                
                if tmp.ndim > 1:
                    # average over first axis (time)
                    self.epi = tmp.mean(axis=0)
                else:
                    self.epi = tmp.copy()

            # save vertex object too
            self.epi_sm = self.epi.copy()
            self.epi_sm[self.surface.whole_roi.astype(int) < 1] = 0
            # self.epi_sm = self.surface.smooth_subsurface(self.epi_sm)

            self.epi_v = pycortex.Vertex2D_fix(
                self.epi,
                subject=self.subject,
                vmin1=0,
                vmax1=self.epi.max(),
                cmap="magma")

            self.epi_sm_v = pycortex.Vertex2D_fix(
                self.epi_sm,
                subject=self.subject,
                vmin1=0,
                vmax1=self.epi_sm.max(),
                cmap="magma")
        
    def apply_thresholds(
        self, 
        x_thresh=None,
        y_thresh=None,
        r2_thresh=None, 
        size_thresh=None, 
        ecc_thresh=None, 
        curv_thresh=None,
        a_thresh=None, 
        b_thresh=None, 
        c_thresh=None, 
        d_thresh=None,         
        epi_thresh=None,
        polar_thresh=None, 
        thick_thresh=None, 
        depth_thresh=None,
        srf=False,
        srf_file=None):

        """apply_thresholds

        Apply thresholds to pRF-parameters and multiply with V1-label mask to obtain only pRFs with certain characteristics in V1. The pRF characteristics are embedded in the prf-class, containing the r2, eccentricity, and polar angle values for all vertices. Additionally, the surface-class contains structural information regarding thickness and sulcal depth. Each of these parameters can be adjusted to find your optimal pRF.

        Parameters
        ----------
        x_thres: int, float, optional
            refers to `x-position` (left-to-right vertical meridian). Usually between -10 and 10. Defaults to 0. A list of values for left and right hemisphere should be given. For the right hemisphere, thresholds are treated as `smaller than` (left hemifield); for left hemisphere, thresholds denote `greater than` (right hemifield). Used to push the vertex away from the fovea
        y_thres: int, float, optional
            refers to y-position in visual space. Usually between -5 and 5 (defaults to the maximum of `y`). Thresholds are defined as a range within which a pRF should fall, e.g,. (-5,5) for screen dimensions in y-direction
        r2_thres: int, float, optional
            refers to amount of variance explained. Usually between 0 and 1 (defaults to the minimum `r2`). Threshold is specified as 'greater than <value>'
        size_thres: int, float, optional
            refers to size of the pRF in visual space. Usually between 0 and 5 (defaults to 0). Thresholds are defined as a range within which a pRF should fall, e.g,. (0.5,2.5)            
        ecc_thresh: int, float, optional
            refers to `size` of pRF (smaller = foveal, larger = peripheral). Usually between 0 and 15. Thresholds are defined as a range within which a pRF should fall, e.g., (2,3)
        curv_thresh: int, float, optional
            refers to curvature of the cortex. Usually between -1 and 1. Used when `selection=="manual"`, otherwise the minimum curvature is specified      
        epi thresh: int, float, optional
            refers to the EPI intensities. Defaults to all values, at the risk of selecting areas that might have veins. Threshold is specified as 'greater than <value>' and siginifies a percentile value.
        polar_thresh: list, float, optional 
            refers to `polar angle` (upper-to-lower vertical meridian). Usually between -pi and pi. Defaults to -pi. Threshold is specified as 'greater than <value>'. A list of values for left and right hemisphere should be given
        thick_thresh: int, float, optional
            refers to cortical thickness as per the `thickness.npz` file created during the importing of a `FreeSurfer` subjects into `Pycortex`. The map is defined as NEGATIVE VALUES!! Thicker cortex is represented by a lower negative value, usually between -5 and 0. Defaults to the minimum value of the thickness array. Threshold is specified as 'lower than <value>'.
        depth_thresh: int, float, optional
            refers to sulcal depth (location of deep/superficial sulci) as per the `sulcaldepth.npz` file created during the importing of a `FreeSurfer` subjects into `Pycortex`. Defaults to the minimum value of the sulcaldepth array. Threshold is specified as 'greater than <value>'.
        srf: bool, optional
            Select vertex based on size-response function (SRF) properties. For now it maximizes suppression
        srf_file: str, optional
            Specify a precomputed dataframe with SRFs

        Returns
        ----------
        attr
            self.lh_prf; boolean mask of where in V1 the specified criteria are met in the left hemisphere
            self.rh_prf; boolean mask of where in V1 the specified criteria are met in the right hemisphere
            self.joint_mask_v; vertex map of the mask created by the specified criteria

        Example
        ----------
        >>> self.apply_thresholds(r2_thresh=0.4, ecc_thresh=3, polar_thresh=5)
        """

        # set cutoff criteria
        utils.verbose("Using these parameters to find vertex with minimal curvature:", True)

        # initialize dictionary for log file
        self.criteria = {}
        self.data_dict = {}
        self.criteria["CreatedOn"] = str(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        self.criteria["Method"] = "Criteria"
        if hasattr(self, 'prf'):
            
            # set thresholds
            self.r2_thresh = r2_thresh or self.prf.df_prf.r2.min()
            self.size_thresh = size_thresh or self.prf.df_prf.prf_size.max()
            self.ecc_thresh = ecc_thresh or self.prf.df_prf.ecc.max()
            self.y_thresh = y_thresh or (self.prf.df_prf.y.min(),self.prf.df_prf.y.max())
            
            # set default size range if 1 value was specified
            if not isinstance(self.size_thresh, (list,tuple)):
                self.size_thresh = [self.size_thresh, self.prf.df_prf.prf_size.max()]

            # parse out polar angle
            self.polar_array    = self.prf.df_prf.polar.values
            self.polar_thresh   = polar_thresh or [max(self.polar_array[:self.surface.lh_surf_data[0].shape[0]]),min(self.polar_array[self.surface.lh_surf_data[0].shape[0]:])]

            lh_polar = self.polar_array[:self.surface.lh_surf_data[0].shape[0]] <= self.polar_thresh[0]
            rh_polar = self.polar_array[self.surface.lh_surf_data[0].shape[0]:] >= self.polar_thresh[1]
            polar = np.concatenate((lh_polar,rh_polar))

            # parse out x-position
            self.x_array = self.prf.df_prf.x.values
            self.x_thresh = x_thresh or [0,0]

            lh_x = self.x_array[:self.surface.lh_surf_data[0].shape[0]] >= self.x_thresh[0]
            rh_x = self.x_array[self.surface.lh_surf_data[0].shape[0]:] <= self.x_thresh[1]
            x_idx = np.concatenate((lh_x,rh_x))

            x_pos = np.full_like(polar, True)
            x_pos[x_idx] = 1
            
            print(f"X-pos: {np.count_nonzero(x_pos)}")
            print(f" lh: {np.count_nonzero(lh_x)}")
            print(f" rh: {np.count_nonzero(rh_x)}")
            
            # compile mask
            df = self.prf.df_prf.copy()
            self.prf_mask = np.zeros(df.shape[0], dtype=bool)

            # check if ecc was list or not
            # prf_idc = list(
            #     utils.multiselect_from_df(
            #         df,
            #         expression=[
            #             f"r2 >= {self.r2_thresh}",
            #             f"prf_size >= {self.size_thresh[0]}",
            #             f"prf_size <= {self.size_thresh[1]}",
            #             f"ecc >= {self.ecc_thresh[0]}",
            #             f"ecc <= {self.ecc_thresh[1]}", 
            #             f"y >= {self.y_thresh[0]}", 
            #             f"y <= {self.y_thresh[1]}",                    
            #         ]
            #     ).index
            # )
            
            prf_idc = list(
                df.loc[
                    (df.y <= self.y_thresh[1]) 
                    & (df.y >= self.y_thresh[0])
                    & (df.r2 >= self.r2_thresh)
                    & (df.ecc >= self.ecc_thresh[0])
                    & (df.ecc <= self.ecc_thresh[1])
                    & (df.prf_size >= self.size_thresh[0])
                    & (df.prf_size <= self.size_thresh[1])                
                ].index
            )
            print(f"pRF: {len(prf_idc)}")

            # sort out polar angle
            self.prf_mask[prf_idc] = True
            self.prf_mask = (self.prf_mask * x_pos)

            print(f"pRF x pos: {np.count_nonzero(self.prf_mask)}")

            utils.verbose(f" x-position:    >={self.x_thresh[0]}/<={self.x_thresh[1]}", True)
            utils.verbose(f" y-position:    {round(self.y_thresh[0],4)}-{round(self.y_thresh[1],4)}", True)
            utils.verbose(f" pRF size:      {round(self.size_thresh[0],4)}-{round(self.size_thresh[1],4)}", True)
            utils.verbose(f" eccentricity:  {round(self.ecc_thresh[0],4)}-{round(self.ecc_thresh[1],4)}", True)
            utils.verbose(f" variance (r2): >= {round(self.r2_thresh,4)}", True)
            utils.verbose(f" polar angle:   {self.polar_thresh}", True)
            
            # append to dictionary
            for par,val in zip(
                ["r2","size","ecc","polar angle"],
                [self.r2_thresh,self.size_thresh,self.ecc_thresh,self.polar_thresh]):
                if isinstance(val, (list,tuple)):
                    self.criteria[par] = [float(i) for i in val]
                else:
                    self.criteria[par] = float(val)

                # store vertex objects in self.data_dict
                if hasattr(self.prf, "prf_data_dict"):
                    for key,val in self.prf.prf_data_dict.items():
                        self.data_dict[key] = val

            # set normalization criteria
            if self.model == "norm":
                self.a_thresh = a_thresh or self.prf.df_prf.A.min()
                self.b_thresh = b_thresh or self.prf.df_prf.B.min()
                self.c_thresh = c_thresh or self.prf.df_prf.C.min()
                self.d_thresh = d_thresh or self.prf.df_prf.D.min()

                for par,val in zip(
                    ["A","B","C","D"],
                    [self.a_thresh,self.b_thresh,self.c_thresh,self.d_thresh]):
                    utils.verbose((f" {par}:             >= {round(val,4)}"), True)
                    self.criteria[par] = float(val)

                # find indices where conditions hold
                norm_idc = list(
                    utils.multiselect_from_df(
                        df,
                        expression=[
                            f"A >= {self.a_thresh}",
                            f"B >= {self.b_thresh}",
                            f"C >= {self.c_thresh}",
                            f"D >= {self.d_thresh}",                  
                        ]
                    ).index
                )

                print(f"DN: {len(norm_idc)}")

                # norm_idc = list(df.loc[
                #     (df.A >= self.a_thresh)
                #     & (df.B >= self.b_thresh)
                #     & (df.C >= self.c_thresh)
                #     & (df.D >= self.d_thresh)
                # ].index)
                
                # make mask
                norm_mask = np.zeros_like(self.prf_mask, dtype=bool)
                norm_mask[norm_idc] = True

                # apply mask to existing prf_mask
                self.prf_mask = self.prf_mask * norm_mask

                print(f"pRF x DN: {np.count_nonzero(self.prf_mask)}")

        # include epi intensity mask
        if hasattr(self, 'epi'):
            
            self.epi_thresh = float(epi_thresh) or 0
            utils.verbose(f" EPI intensity: >= {self.epi_thresh}th percentile", True)

            self.epi_mask = np.zeros_like(self.epi, dtype=bool)
            self.epi_mask[self.epi >= np.percentile(self.epi, self.epi_thresh)] = True
            self.criteria["EPI_intensity"] = float(self.epi_thresh)

            self.data_dict["epi"] = self.epi_v
            self.data_dict["epi_smoothed"] = self.epi_sm_v

        # include curvature mask
        if hasattr(self, 'surface'):
            self.df_struct = pd.DataFrame({
                "thickness": self.surface.thickness.data,
                "depth": self.surface.depth.data,
                "curvature": self.surface.surf_sm
            })

            self.thick_thresh   = thick_thresh or max(self.surface.thickness.data)
            self.depth_thresh   = depth_thresh or min(self.surface.depth.data)
            self.curv_thresh    = curv_thresh or None

            # got a positive value; convert to negative
            if self.thick_thresh:
                if self.thick_thresh > 0:
                    self.thick_thresh = -(self.thick_thresh)

            utils.verbose(f" thickness:     <= {self.thick_thresh}", True)
            utils.verbose(f" depth:         >= {round(self.depth_thresh,4)}", True)
            self.criteria["thickness"] = float(self.thick_thresh)
            self.criteria["depth"] = float(self.depth_thresh)

            struct_idc = list(
                self.df_struct.loc[
                    (self.df_struct.thickness <= self.thick_thresh)
                    & (self.df_struct.depth >= self.depth_thresh)
                ].index)      

            print(f"structural: {len(struct_idc)}")
            if isinstance(self.curv_thresh, list):
                utils.verbose(f" curvature:     {self.curv_thresh}", True)
                struct_idc += list(
                    self.df_struct.loc[
                        (self.df_struct.curvature >= self.curv_thresh[0])
                        & (self.df_struct.curvature <= self.curv_thresh[1])
                    ].index)

            # make mask
            self.struct_mask = np.zeros((self.df_struct.shape[0]), dtype=bool)
            self.struct_mask[struct_idc] = True
            
            # self.struct_mask =  (
            #     (self.surface.thickness.data <= self.thick_thresh) 
            #     * (self.surface.depth.data >= self.depth_thresh))


        ### APPLY MASKS

        # start with full mask
        self.joint_mask = self.surface.whole_roi.copy()

        # and structural mask
        if hasattr(self, 'struct_mask'):
            self.joint_mask = (self.struct_mask * self.joint_mask)
            print(f"V1 x struct: {np.count_nonzero(self.joint_mask)}")
        
            self.struct_mask = pycortex.Vertex2D_fix(
                self.struct_mask,
                self.struct_mask,
                subject=self.subject)
                    
            self.data_dict["struct_mask"] = self.struct_mask

        # and EPI mask
        if hasattr(self, "epi_mask"):
            self.joint_mask = (self.epi_mask * self.joint_mask)
            print(f"V1 x EPI: {np.count_nonzero(self.joint_mask)}")
            self.epi_mask_v = pycortex.Vertex2D_fix(
                self.epi_mask,
                self.epi_mask,
                subject=self.subject)
            
            # self.data_dict["epi_mask"] = self.epi_mask_v

        # apply pRF mask
        if hasattr(self, 'prf_mask'):
            self.joint_mask = (self.prf_mask * self.joint_mask)
            print(f"V1 x pRF: {np.count_nonzero(self.joint_mask)}")
            self.prf_mask_v = pycortex.Vertex2D_fix(
                self.prf_mask,
                self.prf_mask,
                subject=self.subject)
                    
            self.data_dict["prf_mask"] = self.prf_mask_v

        self.surviving_vertices = list(np.where(self.joint_mask > 0)[0])
        utils.verbose(f"Mask contains {len(self.surviving_vertices)} vertices", True)

        # initialize dataframe with whole-brain dimensions, but only with surviving_vertices' pRF estimates
        if hasattr(self, 'prf'):
            if srf:

                if not isinstance(srf_file, str):
                    utils.verbose("Calculating SRFs for surviving vertices", True)
                    tmp_init = np.zeros_like(self.prf.df_prf)
                    self.df_for_srfs = pd.DataFrame(
                        tmp_init, 
                        index=self.prf.df_prf.index, 
                        columns=self.prf.df_prf.columns)

                    self.df_for_srfs.iloc[self.surviving_vertices] = self.prf.df_prf.iloc[self.surviving_vertices]

                    # size response functions
                    self.SR_ = prf.SizeResponse(params=self.df_for_srfs, model="norm")
                    
                    # size-response
                    self.fill_cent, self.fill_cent_sizes = self.SR_.make_stimuli(
                        factor=1,
                        dt="fill"
                    )

                    # hole-response
                    self.hole_cent, self.hole_cent_sizes = self.SR_.make_stimuli(
                        factor=1,
                        dt="hole"
                    )

                    #SRFs for activation & normalization parameters
                    self.sr_fill = self.SR_.batch_sr_function(
                        self.SR_.params_df,
                        stims=self.fill_cent,
                        sizes=self.fill_cent_sizes,
                        center_prf=True
                    )

                    self.sr_hole = self.SR_.batch_sr_function(
                        self.SR_.params_df,
                        stims=self.hole_cent,
                        sizes=self.hole_cent_sizes,
                        center_prf=True
                    )

                    self.sr_fill["type"] = "act"
                    self.sr_hole["type"] = "norm"

                    self.df_sr = pd.concat([self.sr_fill,self.sr_hole])
                    self.df_sr["subject"] = self.subject

                    try:
                        self.df_sr = self.df_sr.set_index(["subject","type","sizes","stim_nr"])
                    except:
                        pass

                else:
                    if isinstance(srf_file, str):
                        utils.verbose(f"Reading SRFs from '{srf_file}'", self.verbose)
                        if srf_file.endswith("pkl"):
                            self.df_sr = pd.read_pickle(srf_file)
                        elif srf_file.endswith("csv"):
                            self.df_sr = pd.read_csv(srf_file)
                        else:
                            raise ValueError(f"File must end with 'pkl' (preferred) or 'csv', not '{srf_file}'")
                        
                        try:
                            self.df_sr = self.df_sr.set_index(["subject","type","sizes","stim_nr"])
                        except:
                            pass
                    else:
                        raise TypeError(f"SRF-file must be a string, not '{srf_file}' of type {type(srf_file)}")
                    
                    # size response functions
                    self.SR_ = prf.SizeResponse(params=self.prf.df_prf, model="norm")

                #     # create max activation/suppression maps
                #     self.df_activation = utils.select_from_df(self.df_sr, expression="type = act")
                #     self.df_suppression = utils.select_from_df(self.df_sr, expression="type = norm")

                #     self.df_max_activation = self.df_activation.max(axis=0)
                #     self.df_max_suppression = self.df_suppression.min(axis=0)

                #     # set values < 0 to 0 in activation; only want real positive activation
                #     self.df_max_activation[self.df_max_activation < 0] = 0                    

                #     # set values > 0 to 0 in suppression; only want real negative suppression
                #     self.df_max_suppression[self.df_max_suppression > 0] = 0

                #     # get ratio, but only divide non-zero numbers
                #     self.df_ratio = np.divide(self.df_max_activation.values, np.abs(self.df_max_suppression.values), out=np.zeros_like(self.df_max_activation.values), where=np.abs(self.df_max_suppression.values)!=0)

                #     # make object for activation
                #     self.act_v = pycortex.Vertex2D_fix(
                #         self.df_max_activation,
                #         subject=self.subject,
                #         cmap="inferno",
                #         vmax1=5)

                #     # make object for suppression
                #     alpha = np.zeros_like(self.df_max_suppression)
                #     alpha[self.df_max_suppression < 0] = 1
                #     self.suppr_v = pycortex.Vertex2D_fix(
                #         self.df_max_suppression,
                #         alpha,
                #         subject=self.subject,
                #         cmap="cool",
                #         vmin1=-3,
                #         vmax1=0)
                    
                #     # make object for ratio
                #     self.ratio_v = pycortex.Vertex2D_fix(
                #         self.df_ratio,
                #         subject=self.subject,
                #         cmap="hot",
                #         vmax1=5)
                    
                #     # make object for ratio
                #     self.ratio_v = pycortex.Vertex2D_fix(
                #         self.df_ratio,
                #         subject=self.subject,
                #         cmap="hot",
                #         vmax1=5)       

                # self.srf_surviving = utils.select_from_df(self.df_sr, expression="ribbon", indices=self.surviving_vertices)
                # self.act_surviving = utils.select_from_df(self.srf_surviving, expression="type = act")             
                # self.suppr_surviving = utils.select_from_df(self.srf_surviving, expression="type = norm")

                # self.final_suppr = np.zeros((self.df_sr.shape[-1]))
                # self.final_suppr[self.surviving_vertices] = self.suppr_surviving.min().values

                # alpha = np.zeros_like(self.final_suppr)
                # alpha[self.final_suppr < 0] = 1
                # self.final_suppr_v = pycortex.Vertex2D_fix(
                #     self.final_suppr,
                #     alpha,
                #     subject=self.subject,
                #     cmap="cool",
                #     vmin1=-3,
                #     vmax1=0)


        # try to find suppression/activation maps based on SRF file
        for tag,obj in zip(
            ["suppression", "activation","abs_ratio"],
            ["suppr_v","act_v","ratio_v"]):
            
            if hasattr(self, obj):
                self.data_dict[tag] = getattr(self, obj)

        # save prf information
        self.lh_prf = self.joint_mask[:self.surface.lh_surf_data[0].shape[0]]
        self.rh_prf = self.joint_mask[self.surface.lh_surf_data[0].shape[0]:]

        print(f"lh: {np.count_nonzero(self.lh_prf)}")
        print(f"rh: {np.count_nonzero(self.rh_prf)}")
        
        # if not "V1_" in self.fs_label:
        #     self.joint_mask[self.joint_mask < 1] = np.nan
        
        self.joint_mask_v = pycortex.Vertex2D_fix(
            self.joint_mask,
            self.joint_mask,
            subject=self.subject)

        self.data_dict["roi"] = self.surface.whole_roi_v
        self.data_dict["final_mask"] = self.joint_mask_v

        # also create a mask scaled by the curvature
        self.mask_by_curv = np.zeros_like(self.joint_mask, dtype=float)
        self.curv_both = []
        for hh in ["lh","rh"]:
            curv = getattr(self.surface, f"{hh}_surf_sm")
            self.curv_both.append(curv)

        self.curv_both = np.concatenate(self.curv_both)
        indices = np.where(self.joint_mask == True)[0]
        self.mask_by_curv[indices] = self.curv_both[indices]

        self.mask_by_curv_v = pycortex.Vertex2D_fix(
            self.mask_by_curv,
            vmin1=self.mask_by_curv.min(),
            vmax1=self.mask_by_curv.max(),
            cmap="viridis_r",
            subject=self.subject,
            curv_type="cortex"
        )
        
        # surviving vertices colored by curvature
        self.data_dict["final_curv"] = self.mask_by_curv_v   

        # surviving vertices colored by suppression
        if hasattr(self, "final_suppr_v"):
            self.data_dict["final_suppr"] = self.final_suppr_v

    def best_vertex(self, open_with="ctx"):

        """Fetch best vertex given pRF-properties and minimal curvature"""

        # check if selection is manual or automatic
        if not isinstance(self.selection, str):
            
            self.srfs_best_vertices = []
            for i in ['lh', 'rh']:
                
                if hasattr(self, f'{i}_prf'):

                    curv = getattr(self.surface, f'{i}_surf_sm') # smoothed curvature from SurfaceCalc
                    mask = getattr(self, f'{i}_prf')
                    surf = getattr(self.surface, f'{i}_surf_data')

                    # get all vertices where mask = True
                    vv = np.where(mask == True)[0]
                    curv_dict = {}
                    for pp in vv:                      
                        curv_dict[pp] = curv[pp]
                    #     print(f"vert {pp} = {curv[pp]}")

                    # sys.exit(1)
                    # get list of curvatures in selected vertices
                    val = list(curv_dict.values())

                    if len(val) == 0:
                        print(f"WARNING [{i}]: Could not find a vertex with these parameters. Try lowering your r2-criteria if used.")
                        setattr(self, f'{i}_best_vert_mask', None)
                    else:
                        if len(val) < 10:
                            add_txt = f": {list(vv)}"
                        else:
                            add_txt = ""
                        print(f"{i}: {len(val)} voxels matched criteria{add_txt}")

                        if self.srf:

                            # get hemi-specific SRFs
                            if i == "rh":
                                idc = vv+self.surface.lh_surf_data[0].shape[0]
                            else:
                                idc = vv.copy()

                            srfs_hemi = utils.select_from_df(self.df_sr, expression="ribbon", indices=idc)
                            setattr(self, f"{i}_srfs", srfs_hemi)

                            # get hole-response functions
                            suppr_hemi = utils.select_from_df(srfs_hemi, expression="type = norm")
                            act_hemi = utils.select_from_df(srfs_hemi, expression="type = act")
                            
                            # get the minimum values
                            suppr_vals = suppr_hemi.values.T.min(axis=-1)
                            act_vals = act_hemi.values.T.max(axis=-1)

                            setattr(self, f"{i}_suppression", suppr_vals)
                            setattr(self, f"{i}_activation", act_vals)

                            # get corresponding curvature values
                            curv_suppr_ix = self.surface.curvature.data[vv]
                            setattr(self, f"{i}_curvature", curv_suppr_ix)

                            # reverse the sign of suppression to that highest value = highest suppression
                            x_1 = utils.reverse_sign(suppr_vals)
                            setattr(self, f"{i}_reversed_suppression", x_1)

                            # take the square of curvature so that all values are positive), then flip sign so that highest value is best curvature
                            x_2 = utils.reverse_sign(curv_suppr_ix**2)
                            setattr(self, f"{i}_reversed_curvature", x_2)

                            # add them together; the result should be maximized
                            optimal_curv_suppr = np.add(x_1,x_2)
                            opt_ix,_ = utils.find_nearest(optimal_curv_suppr, optimal_curv_suppr.max())
                            
                            # index surviving vertex list with opt_ix
                            min_index = vv[opt_ix]
                            setattr(self, f'{i}_best_vertex', min_index)
                            
                            # get maximum suppression
                            suppr_strength = suppr_vals[opt_ix]
                            setattr(self, f"{i}_max_suppression", suppr_strength)

                            # get max positive response of this vertex
                            act_strength = act_vals[opt_ix]
                            setattr(self, f"{i}_max_activation", act_strength)

                            # find associated stimulus ID
                            stim_suppr_ix = np.where(suppr_hemi.values.T[opt_ix] == suppr_strength)[0][0]
                            setattr(self, f"{i}_stim_ix_suppression", stim_suppr_ix)

                            stim_act_ix = np.where(act_hemi.values.T[opt_ix] == act_strength)[0][0]
                            setattr(self, f"{i}_stim_ix_activation", stim_act_ix)

                            # find associated stimulus sizes
                            size_suppr = utils.select_from_df(self.df_sr, expression=("type = norm","&",f"stim_nr = {stim_suppr_ix}")).reset_index()["sizes"][0]
                            setattr(self, f"{i}_stim_size_suppression", size_suppr)

                            size_act = utils.select_from_df(self.df_sr, expression=("type = act","&",f"stim_nr = {stim_act_ix}")).reset_index()["sizes"][0]
                            setattr(self, f"{i}_stim_size_activation", size_act)
                            
                            self.srfs_best_vertices.append(utils.select_from_df(self.df_sr, expression="ribbon", indices=[min_index]))
                        else:
                            # find curvature closest to zero
                            low,_ = utils.find_nearest(val, 0)
                            min_index = vv[low]

                            setattr(self, f'{i}_best_vertex', min_index) 
                                            
                        # 'curv' contains 1e8 for values outside ROI and absolute values for inside ROI (see mask_curv_with_prf)
                        # min_index = find_nearest(curv,0)[0]; setattr(self, f'{i}_best_vertex', min_index) # np.argmin(curv); 
                        min_vertex = surf[0][min_index]; setattr(self, f'{i}_best_vertex_coord', min_vertex)
                        min_vert_mask = (surf[1] == min_vertex).sum(0); setattr(self, f'{i}_best_vert_mask', min_vert_mask)

                        # setattr(self, f'{i}_best_vertex', np.argmin(curv))
                        # setattr(self, f'{i}_best_vertex_coord', surf[0][getattr(self, f'{i}_best_vertex')])
                        # setattr(self, f'{i}_best_vert_mask', (surf[1] == getattr(self, f'{i}_best_vertex_coord')).sum(0))

                else:
                    raise TypeError(f'Attribute {i}_prf does not exist')

            if self.srf:
                self.srfs_best_vertices = pd.concat(self.srfs_best_vertices, axis=1)  
        else:
            # select manually with freeview
            if self.selection == "manual":
                
                utils.verbose(f"Selection method is MANUAL", self.verbose)

                self.webshow = False
                if open_with == "ctx":
                    self.open_pycortex(
                        radius=240,
                        pivot=0)

                    # time.sleep(50)
                else:
                    self.lh_final_mask = self.joint_mask[:self.surface.lh_surf_data[0].shape[0]]
                    self.rh_final_mask = self.joint_mask[self.surface.lh_surf_data[0].shape[0]:]                

                    # write joint_mask to label file that FreeSurfer understands
                    self.label_files = []
                    output_dir = os.path.dirname(self.lh_fid)
                    for hemi in ["lh", "rh"]:
                        out_f = opj(output_dir, f"{hemi}.finalmask")
                        self.label_files.append(out_f)
                        nb.freesurfer.io.write_morph_data(out_f, getattr(self, f"{hemi}_final_mask"))

                    cmd = f"freeview -v {self.orig} -f {self.lh_fid}:edgecolor=green:overlay={self.label_files[0]} {self.rh_fid}:edgecolor=green:overlay={self.label_files[1]} {opj(output_dir, 'lh.inflated')}:edgecolor=blue:overlay={self.label_files[0]} {opj(output_dir, 'rh.inflated')}:edgecolor=blue:overlay={self.label_files[1]} 2>/dev/null &"

                    os.system(cmd)
                
                # fetch vertices
                self.srfs_best_vertices = []
                for tag,hemi,surf in zip(
                    ["left","right"],
                    ["lh","rh"],
                    [self.surface.lh_surf_data,self.surface.rh_surf_data]):

                    while True:
                        try:
                            # check if range is specified
                            min_index = input(f"vertex {tag.upper()} hemi: ")
                            try:
                                min_index = int(min_index)
                                break
                            except ValueError:
                                print(" Please enter a number")
                                continue
                        except ValueError:
                            print(" Please enter a number")
                            continue

                    min_vertex = surf[0][min_index]; setattr(self, f'{hemi}_best_vertex_coord', min_vertex)
                    min_vert_mask = (surf[1] == min_vertex).sum(0); setattr(self, f'{hemi}_best_vert_mask', min_vert_mask)
                    setattr(self, f'{hemi}_best_vertex', min_index) 

                    # set bunch of attributes related to SRFs
                    if self.srf:
                        
                        if tag == "right":
                            min_index += self.surface.lh_surf_data[0].shape[0]
                            
                        srf_idx = utils.select_from_df(self.df_sr, expression="ribbon", indices=[min_index])

                        df_suppr = utils.select_from_df(srf_idx, expression="type = norm")
                        df_act = utils.select_from_df(srf_idx, expression="type = act")

                        self.srfs_best_vertices.append(srf_idx)

                        # get maximum suppression
                        try:
                            suppr_val = df_suppr[str(min_index)].min()
                        except:
                            suppr_val = df_suppr[min_index].min()

                        setattr(self, f"{hemi}_max_suppression", suppr_val)

                        # get max positive response of this vertex
                        try:
                            act_val = df_act[str(min_index)].max()
                        except:
                            act_val = df_act[min_index].max()

                        setattr(self, f"{hemi}_max_activation", act_val)
                        
                        try:
                            stim_suppr_ix = df_suppr[str(min_index)].argmin()
                        except:
                            stim_suppr_ix = df_suppr[min_index].argmin()

                        setattr(self, f"{hemi}_stim_ix_suppression", stim_suppr_ix)

                        try:
                            stim_act_ix = df_act[str(min_index)].argmax()
                        except:
                            stim_act_ix = df_act[min_index].argmax()

                        setattr(self, f"{hemi}_stim_ix_activation", stim_act_ix)

                        # find associated stimulus sizes
                        size_suppr = utils.select_from_df(df_suppr, expression=f"stim_nr = {stim_suppr_ix}").reset_index()["sizes"][0]
                        setattr(self, f"{hemi}_stim_size_suppression", size_suppr)

                        size_act = utils.select_from_df(df_act, expression=f"stim_nr = {stim_act_ix}").reset_index()["sizes"][0]
                        setattr(self, f"{hemi}_stim_size_activation", size_act)                        

                if self.srf:
                    self.srfs_best_vertices = pd.concat(self.srfs_best_vertices, axis=1)  

            else:
                raise ValueError(f"selection must be 'manual', not '{self.method}'")
            
    def fetch_normal(self, method="ctx"):
        """fetch_normal

        Fetch normal vector of best vertex. We can do this by calculating the neighbouring vertices and taking the cross product of these vectorsself (Also see: https://sites.google.com/site/dlampetest/python/calculating-normals-of-a-triangle-mesh-using-numpy) [specify `methods=="cross"`]. Another option (the default one) is to extract the normal vectors as calculated by `Pycortex` itself [specify `method="ctx"`]. Finally, we can apply Newell's calculation of normal vectors (https://stackoverflow.com/questions/27326636/calculate-normal-vector-of-a-polygon-newells-method) [specify `method="newell"`] 

        Parameters
        -----------
        method: str, optional
            Which implementation of normal vector calculation to use ('cross', 'newell', or 'ctx' [default])

        Returns
        ----------
        attr
            Sets the self.?h_normal attributes within the class
        """ 

        for i in ['lh', 'rh']:

            if hasattr(self, f'{i}_best_vert_mask'):

                vertices = getattr(self.surface, f'{i}_surf_data')[0]
                faces = getattr(self.surface, f'{i}_surf_data')[1]

                tris = vertices[faces]

                if method == "cross":
                    n = np.cross(tris[::,1 ]-tris[::,0], tris[::,2 ]-tris[::,0])
                    utils.convert2unit(n)

                    setattr(self, f'{i}_normal', n[getattr(self, f'{i}_best_vertex')])

                elif method == "newell":

                    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
                    n = np.cross(tris[::,1 ]-tris[::,0], tris[::,2 ]-tris[::,0])
                    utils.convert2unit(n, method="mesh")

                    norm[faces[:,0] ] += n
                    norm[faces[:,1] ] += n
                    norm[faces[:,2] ] += n

                    utils.convert2unit(norm, method="mesh")

                    setattr(self, f'{i}_normal', norm[getattr(self, f'{i}_best_vertex')])

                elif method == "orig":

                    vert = getattr(self, f'{i}_best_vert_mask')
                    surf = getattr(self.surface, f'{i}_surf_data')

                    tris = surf[0][surf[1][vert]]
                    normals = np.cross(tris[::,1] - tris[::,0], tris[::,2] - tris[::,0])
                    normals /= np.linalg.norm(normals, axis=1)
                    normal_best_vert = np.average(normals, axis=0)
                    # normal_best_vert[1] = reverse_sign(normal_best_vert[1])
                    setattr(self, f'{i}_normal', normal_best_vert)

                elif method == "ctx":
                    if hasattr(self.surface, f'{i}_surf_normals'):
                        vert = getattr(self, f'{i}_best_vertex')
                        surf = getattr(self.surface, f'{i}_surf_normals')

                        norm = surf[vert]
                        setattr(self, f'{i}_normal', norm)

                else:
                    raise NotImplementedError(f"Unknown method {method} for calculating surface normals")

            else:
                raise TypeError(f'Attribute {i}_best_vertex does not exist')

    def return_normal(self, hemi="both"):

        """return the normal vectors instead of writing them as attributes of the object"""

        if hemi == "both":
            if hasattr(self, "lh_normal") and hasattr(self, "rh_normal"):
                df = {'lh': self.lh_normal,
                      'rh': self.rh_normal}

            return df

        elif hemi == "rh" or hemi == "right" or hemi.lower() == "r":
            if hasattr(self, "rh_normal"):
                return self.rh_normal

        elif hemi == "lh" or hemi == "left" or hemi.lower() == "l":
            if hasattr(self, "lh_normal"):
                return self.lh_normal

        else:
            raise ValueError(f"Unknown option {hemi} specified. Use 'both', or 'lh/rh', 'l/r', 'R/L', 'left/right'")


    def vertex_to_map(self, concat=True, write_files=False):

        """vertex_to_map
        
        Create a vertex object that can be read in by pycortex. We can also combine the left/right hemisphere by setting the concat flag to True. 'Write_files'-flag indicates that we should write the files to .npy files in the pRF-directory.
        
        Parameters
        ----------
        concat: bool
            Boolean that decides whether to concatenate the left and right hemisphere vertex points
        write_files: bool
            Boolean that decides whether or not to write the numpy arrays to files

        Returns
        ----------
        numpy.ndarray
            binarized arrays containing the target vertices in both hemispheres (if `concat==True`), or individual hemispheres (if `concat==False`)
        """

        for i in ['lh', 'rh']:

            if hasattr(self, f'{i}_best_vertex') and hasattr(self.surface, 'curvature') and hasattr(self.surface, f'{i}_surf_data'):

                min_curv_map = np.zeros_like(self.surface.curvature.data)

                # pycortex stacks vertically, left hemi first then right hemi
                if i == "lh":
                    min_curv_map[self.lh_best_vertex] = 1
                elif i == "rh":
                    if self.rh_best_vertex < self.surface.lh_surf_data[0].shape[0]:
                        min_curv_map[self.surface.lh_surf_data[0].shape[0]+self.rh_best_vertex] = 1
                    else:
                        min_curv_map[self.rh_best_vertex] = 1

                min_curv_map_v = pycortex.Vertex2D_fix(
                    min_curv_map, 
                    subject=self.subject)

                setattr(self, f'{i}_best_vertex_map', min_curv_map)
                setattr(self, f'{i}_best_vertex_map_v', min_curv_map_v)

            else:

                raise TypeError("Missing attributes. Need the curvature data and best vertex index")

        if concat == True:

            both = np.copy(self.lh_best_vertex_map)
            both[np.where(self.rh_best_vertex_map == 1)] = 1
            both_v = pycortex.Vertex2D_fix(
                both, 
                subject=self.subject)
        
            self.lr_best_vertex_map = both
            self.lr_best_vertex_map_v = both_v

            if hasattr(self, "data_dict"):
                if isinstance(self.data_dict, dict):
                    self.data_dict["targets"] = self.lr_best_vertex_map_v
            else:
                self.data_dict = {"targets": self.lr_best_vertex_map_v}

        if write_files == True:

            for i in ['lh', 'rh']:
                if hasattr(self, f'{i}_best_vertex_map_sm'):
                    if i == "lh":
                        tag = "hemi-L"
                    elif i == "rh":
                        tag = "hemi-R"
                    else:
                        raise ValueError(f"Unknown value {i}..?")
                    np.save(opj(self.prf_dir, self.subject, f'{self.fname}_desc-vertex_{tag}.npy'), getattr(self, f'{i}_best_vertex_map'))

            if hasattr(self, 'lr_best_vertex_map'):
                np.save(opj(self.prf_dir, self.subject, f'{self.fname}_hemi-LR_desc-vertex.npy'), self.lr_best_vertex_map)


    def smooth_vertexmap(self, kernel=5, concat=True, write_files=True):
        """smooth_vertexmap
        
        Smooth the vertex map with a given kernel size. We can also combine the left/right hemisphere by setting the concat flag to True. 'Write_files'-flag indicates that we should write the files to .npy files in the pRF-directory.
        
        Parameters
        ----------
        kernel: int, optional
            size of kernel to use for smoothing (default = 5)
        concat: bool
            Boolean that decides whether to concatenate the left and right hemisphere vertex points
        write_files: bool
            Boolean that decides whether or not to write the numpy arrays to files

        Returns
        ----------
        numpy.ndarray
            arrays containing the smoothed target vertices in both hemispheres (if `concat==True`), or individual hemispheres (if `concat==False`)
        """

        for i in ['lh', 'rh']:

            if hasattr(self, f'{i}_best_vertex_map') and hasattr(self.surface, f'{i}_surf') and hasattr(self.surface, f'{i}_surf_data'):

                if i == "lh":
                    sm_min_curv_map = self.surface.lh_surf.smooth(self.lh_best_vertex_map[:self.surface.lh_surf_data[0].shape[0]], kernel, kernel)
                elif i == "rh":
                    sm_min_curv_map = self.surface.rh_surf.smooth(self.rh_best_vertex_map[self.surface.lh_surf_data[0].shape[0]:], kernel, kernel)

                sm_min_curv_map /= sm_min_curv_map.max()
                sm_min_curv_map_v = cortex.Vertex(sm_min_curv_map, subject=self.subject, cmap='magma', vmin=-0.5, vmax=1)

                setattr(self, f'{i}_best_vertex_map_sm', sm_min_curv_map)
                setattr(self, f'{i}_best_vertex_map_sm_v', sm_min_curv_map_v)

            else:

                raise TypeError("Missing attributes. Need the curvature data and best vertex index")

        if concat == True:

            sm_bestvertex_LR = np.concatenate((self.lh_best_vertex_map_sm,self.rh_best_vertex_map_sm), axis=0)
            sm_bestvertex_LR_v = cortex.Vertex(sm_bestvertex_LR, subject=self.subject, cmap='magma', vmin=-0.5, vmax=1)

            self.lr_best_vertex_map_sm = sm_bestvertex_LR
            self.lr_best_vertex_map_sm_v = sm_bestvertex_LR_v

        if write_files == True:

            for i in ['lh', 'rh']:
                if hasattr(self, f'{i}_best_vertex_map_sm'):
                    if i == "lh":
                        tag = "hemi-L"
                    elif i == "rh":
                        tag = "hemi-R"
                    else:
                        raise ValueError(f"Unknown value {i}..?")
                    np.save(opj(self.prf_dir, self.subject, f'{self.fname}_desc-smoothvertex_{tag}.npy'), getattr(self, f'{i}_best_vertex_map_sm'))

            if hasattr(self, 'lr_best_vertex_map'):
                np.save(opj(self.prf_dir, self.subject, f'{self.fname}_desc-smoothvertex_hemi-LR.npy'), self.lr_best_vertex_map_sm)

    def write_line_pycortex(self, hemi="both", save_as=None):

        """write_line_pycortex

        This function creates the line_pycortex files containing the angles and translation given the vertex ID, normal vector, and RAS coordinate. It uses :func:`linescanning.planning.create_line_pycortex` to calculate these things. It will return a pandas dataframe containing the relevant information for the vertices in both hemispheres.

        Parameters
        ----------
        hemi: str
            what hemisphere should we process? ("both", "lh", "rh")

        save_as: str, optional
            save dataframe as `csv`-file

        Returns
        ----------
        attr
            sets the `self.?h_trafo_info` attributes
        """

        if hemi == "both":
            # do stuff for both hemispheres
            rot_lh = planning.single_hemi_line_pycortex(self.lh_normal, "L", self.lh_best_vertex, coord=self.lh_best_vertex_coord)
            rot_rh = planning.single_hemi_line_pycortex(self.rh_normal, "R", self.rh_best_vertex, coord=self.rh_best_vertex_coord)
            rot_df = pd.concat([rot_lh, rot_rh]).set_index(['hemi'])

            self.trafo_info = rot_df
            if save_as:
                self.trafo_info.to_csv(save_as)

        else:

            if hemi == "left" or hemi == "lh" or hemi.lower() == "l":
                tag = "lh"
            elif hemi == "right" or hemi == "rh" or hemi.lower() == "r":
                tag = "rh"

            rot = planning.single_hemi_line_pycortex(getattr(self, f'{tag}_normal'), "left", getattr(self, f'{tag}_best_vertex'), coord=getattr(self, f'{tag}_best_vertex_coord'))
            setattr(self, f'{tag}_trafo_info', rot)
            
            if save_as:
                getattr(self, f'{tag}_trafo_info').to_csv(save_as)

    def to_angle(self, hemi="both"):
        """convert normal vector to angle using :func:`linescanning.planning.normal2angle`"""

        if hemi == "both":
            if hasattr(self, "lh_normal") and hasattr(self, "rh_normal"):
                df = {'lh': planning.normal2angle(self.lh_normal),
                      'rh': planning.normal2angle(self.rh_normal)}

            return df


class TargetVertex(CalcBestVertex,utils.VertexInfo):

    """TargetVertex

    Full procedure to extract a target vertex given a set of criteria (as per :func:`linescanning.optimal.set_threshold`) from structural (:class:`linescanning.optimal.SurfaceCalc`) and functional (:class:`linescanning.optimal.pRFCalc`) by combining everything in :class:`linescanning.optimal.CalcBestVert`. The workflow looks as follows:

    * Set thresholds for pRF/structural properties
    * Combine functional/structural properties into :class:`linescanning.optimal.CalcBestVert`
    * Pick out vertex/normal/surface coordinate
    * Verify coordinate using `FreeView`
    * Store pRF/vertex information in `<subject>_desc-prf_params_best_vertices.csv` & `line_pycortex.csv`

    Parameters
    ----------
    subject: str
        Subject ID as used in `SUBJECTS_DIR`
    deriv: str, optional
        Path to derivatives folder of the project. Generally should be the path specified with `DIR_DATA_DERIV` in the bash environment
    prf_file: str, optional
        File containing the pRF-estimates. Required if '--use-prf' is specified
    use_epi: bool
        Allows you to include EPI intensity measures when selecting a vertex; this can be useful to get away from veins. Directly specify inputs to use rather than to look for measures in the fmriprep folder (as is the case with just use_epi=True). Set `use_epi` to `True` by default
    use_epi: bool
        Allows you to include EPI intensity measures when selecting a vertex; this can be useful to get away from veins. Default = False
    webshow: bool
        Start `FreeView` for visual verification. During debugging this is rather cumbersome, so you can turn it off by specifying `webshow=False`
    roi: str, optional
        ROI-name to extract the vertex from as per ROIs created with `FreeSurfer`. Default is V1_exvivo.thresh
    use_prf: bool
        In case you want to base the selection of the target_vertex on both structural and functional properties, set `use_prf=True`. If you only want to include anatomical information such as curvature, set `use_prf=False`. This is relevant for non-visual experiments
    verts: list, optional
        List of manually selected vertices rather than selecting the vertices based on structural/functional properties
    srf: bool, optional
        Select vertices based on size-response function (SRF) characteristics
    selection: str, optional
        Method of selection the best vertex. By default `None`, entailing we'll go for vertex surviving all criteria and has minimum curvature. Can also be 'manual', meaning we'll open FreeView, allowing you to select a vertex manually
    open_with: str, optional
        if `selection="manual"`, we can either open a viewer with FreeView (defualt, as it open a volumetric view too), but can also be `ctx` to open a Pycortex webgl instance
    aparc: bool, optional         
        specified `roi` is part of the aparc.annot file (e.g., "b'lateralorbitofrontal'"). Default = False

    Returns
    ---------
    str
        Creates vertex-information files `<subject>_desc-prf_params_best_vertices.csv` & `line_pycortex.csv` as well as figures showing the timecourses/pRF-location of selected vertices.
    
    class
        :class:`linescanning.CalcBestVertex`

    Example
    ----------
    >>> # use Gaussian iterative parameters to find target vertex for sub-001 in V1_exvivo.thresh by using the default derivatives folders
    >>> optimal.target_vertex(
    >>>     "sub-001", 
    >>>     deriv="/path/to/derivatives",
    >>>     prf_file="gaussian_pars.pkl", 
    >>>     use_prf=True,                         # default
    >>>     out="line_pycortex.csv",              # default
    >>>     roi="V1_exvivo.thresh",               # default
    >>>     webshow=True)                         # default
    """
    def __init__(
        self,
        subject,
        deriv=None,
        prf_file=None,
        epi_file=None,
        srf_file=None,
        use_epi=False,
        webshow=True,
        out=None,
        roi="V1_exvivo.thresh",
        use_prf=True,
        vert=None,
        srf=False,
        verbose=True,
        selection=None,
        open_with="fs",
        skip_prf_info=False,
        aparc=False):

        self.subject = subject
        self.deriv = deriv
        self.prf_file = prf_file
        self.epi_file = epi_file
        self.use_epi = use_epi
        self.webshow = webshow
        self.out = out
        self.roi = roi
        self.use_prf = use_prf
        self.vert = vert
        self.srf_file = srf_file
        self.srf = srf
        self.verbose = verbose
        self.selection = selection
        self.open_with = open_with
        self.aparc = aparc
        self.skip_prf_info = skip_prf_info

        # check if we can read DIR_DATA_DERIV if necessary
        if not self.deriv:
            self.deriv = os.environ.get("DIR_DATA_DERIV")

        # set paths
        if deriv:
            self.prf_dir = opj(self.deriv, 'prf')
            self.fs_dir = opj(self.deriv, 'freesurfer')
            self.cx_dir = opj(self.deriv, 'pycortex')

        # create if necessary
        if not os.path.exists(self.cx_dir):
            os.makedirs(self.cx_dir)

        # update the filestore
        pycortex.set_ctx_path(p=self.cx_dir)

        #----------------------------------------------------------------------------------------------------------------
        # Read in surface and pRF-data

        if isinstance(self.srf_file, str):
            self.srf = True
        
        # set default volumes/surface
        self.orig = opj(self.fs_dir, self.subject, 'mri', 'orig.mgz')
        self.lh_fid = opj(self.fs_dir, self.subject, 'surf', "lh.fiducial")
        self.rh_fid = opj(self.fs_dir, self.subject, 'surf', "rh.fiducial")

        if isinstance(self.out, str) and os.path.isfile(self.out):
            utils.verbose(f"Loading in {self.out}", self.verbose)
            utils.VertexInfo.__init__(
                self,
                self.out, 
                subject=self.subject, 
                hemi="both")
        else:
            if self.use_prf == True:
                if not os.path.exists(self.prf_file):
                    raise FileNotFoundError(f"Could not find with pRF-estimates '{self.prf_file}'")

                self.file_components = utils.split_bids_components(self.prf_file)
                try:
                    self.model = self.file_components['model']
                except:
                    self.model = "gauss"

                if "roi" in list(self.file_components.keys()):
                    v1_data = True
                    v1_flag  = "--v1"
                else:
                    v1_data = False
                    v1_flag = ""
            else:
                self.prf_file = None
                self.model = None
            
            # print some stuff to the terminal
            utils.verbose(f"pRFs = {self.prf_file}", self.verbose)
            utils.verbose(f"ROI = {self.roi}", self.verbose)
            
            if isinstance(epi_file, str):
                utils.verbose(f"EPI = {self.epi_file}", self.verbose)
                self.use_epi = True
            
            # This thing mainly does everything. See the linescanning/optimal.py file for more information
            utils.verbose("Combining surface and pRF-estimates in one object", self.verbose)
            CalcBestVertex.__init__(
                self,
                subject=self.subject, 
                deriv=self.deriv, 
                prf_file=self.prf_file, 
                epi_file=self.epi_file,
                use_epi=self.use_epi,
                fs_label=self.roi,
                model=self.model,
                verbose=self.verbose,
                aparc=self.aparc)

            if self.use_epi or self.use_prf:
                utils.verbose("Also initialize CollectSubject object", self.verbose)
                # load in subject
                self.SI_ = prf.CollectSubject(
                    self.subject, 
                    prf_dir=self.prf_dir, 
                    cx_dir=self.cx_dir, 
                    hemi="lh", 
                    resize_pix=270,
                    best_vertex=False,
                    verbose=True,
                    model=self.model,
                    v1=v1_data)

            # create session directory in pycortex | needs to be after CalcBestVertex so that pycortex get import normally
            if isinstance(self.out, str):
                ctx_ses = os.path.dirname(self.out)
                if not os.path.exists(ctx_ses):
                    os.makedirs(ctx_ses, exist_ok=True)

            #----------------------------------------------------------------------------------------------------------------
            # Set the cutoff criteria based on which you'd like to select a vertex
        
            check = False
            while check == False:
                
                # initialize normalization parameters as 0
                for ii in ["a","b","c","d"]:
                    setattr(self, f"{ii}_val", 0)

                self.manual_vertices = False
                if not isinstance(self.vert, (np.ndarray, list)):
                    utils.verbose("Set thresholds (leave empty and press [ENTER] to not use a particular property):", self.verbose)
                    # get user input with set_threshold > included the possibility to have only pRF or structure only!
                    if hasattr(self, 'prf'):

                        # polar angle left hemi
                        self.x_val_lh = set_threshold(
                            name="x-position (lh)", 
                            borders=(0,10), 
                            set_default=0
                        )
                        
                        # polar angle left hemi
                        self.x_val_rh = set_threshold(
                            name="x-position (rh)", 
                            borders=(-10,0), 
                            set_default=0
                        )

                        # combine polar
                        self.x_val = [self.x_val_lh,self.x_val_rh]

                        # y-position
                        self.y_val = set_threshold(
                            name="y-position", 
                            borders=(-6,6), 
                            set_default=(
                                round(self.prf.df_prf.y.min(),2),
                                round(self.prf.df_prf.y.max(),2)
                            )
                        )           

                        # pRF size
                        self.size_val = set_threshold(
                            name="pRF size (beta)", 
                            borders=(0,self.prf.df_prf.prf_size.max()*1.1), 
                            set_default=(0,round(self.prf.df_prf.prf_size.max(),2)))
                        
                        # r2
                        self.r2_val = set_threshold(
                            name="r2 (variance)", 
                            borders=(0,1), 
                            set_default=0
                        )
                        
                        # eccentricity
                        self.ecc_val = set_threshold(
                            name="ecc band", 
                            borders=(0,15), 
                            set_default=(0,round(self.prf.df_prf.ecc.max(),2))
                        )

                        # polar angle left hemi
                        self.pol_val_lh = set_threshold(
                            name="polar angle lh", 
                            borders=(0,np.pi), 
                            set_default=round(np.pi,2)
                        )
                        
                        # polar angle left hemi
                        self.pol_val_rh = set_threshold(
                            name="polar angle rh", 
                            borders=(-np.pi,0), 
                            set_default=round(-np.pi,2)
                        )

                        # combine polar
                        self.pol_val = [self.pol_val_lh,self.pol_val_rh]

                        if self.model == "norm":

                            for col in ["a","b","c","d"]:
                                thr = set_threshold(
                                    name=f"{col.upper()} value (norm)",
                                    set_default=round(self.prf.df_prf[col.upper()].min(),2))
                                setattr(self, f"{col}_val", thr)

                    else:
                        for ii in ["x","y","size","ecc","r2","pol","a","b","c","d"]:
                            setattr(self, f"{ii}_val", 0)

                    if hasattr(self, 'surface'):
                        self.thick_val = set_threshold(
                            name="thickness (mm)", 
                            borders=(0,5), 
                            set_default=max(self.surface.thickness.data))

                        self.depth_val = set_threshold(
                            name="sulcal depth", 
                            set_default=round(min(self.surface.depth.data)))
                        
                        # specify curvature band
                        self.curv_val = None
                        if isinstance(self.selection, str):
                            self.curv_val = set_threshold(
                                name="curv band", 
                                borders=(-1,1), 
                                set_default=(round(self.surface.surf_sm.min()),round(self.surface.surf_sm.max(),2)))                                

                    else:
                        self.thick_val = 0
                        self.depth_val = 0

                    if hasattr(self, 'epi'):
                        self.epi_val   = set_threshold(name="EPI value (%)", set_default=10)
                    else:
                        self.epi_val = 0

                    # Create mask using selected criteria
                    self.apply_thresholds(
                        x_thresh=self.x_val,
                        y_thresh=self.y_val,
                        ecc_thresh=self.ecc_val,
                        size_thresh=self.size_val,
                        r2_thresh=self.r2_val,
                        polar_thresh=self.pol_val,
                        depth_thresh=self.depth_val,
                        thick_thresh=self.thick_val,
                        epi_thresh=self.epi_val,
                        curv_thresh=self.curv_val,
                        a_thresh=self.a_val,
                        b_thresh=self.b_val,
                        c_thresh=self.c_val,
                        d_thresh=self.d_val,
                        srf=self.srf,
                        srf_file=self.srf_file)

                    # Pick out best vertex
                    self.best_vertex(open_with=self.open_with)
                    
                else:
                    self.manual_vertices = True
                    # set manual vertices in self object
                    for i,r in enumerate(['lh', 'rh']):
                        setattr(self, f"{r}_best_vertex", self.vert[i])
                        setattr(self, f"{r}_best_vertex_coord", getattr(self.surface, f'{r}_surf_data')[0][self.vert[i]])
                        setattr(self, f"{r}_best_vert_mask", (getattr(self.surface, f'{r}_surf_data')[1] == [self.vert[i]]).sum(0))

                    self.data_dict = {}
                    if hasattr(self, "prf"):
                        if hasattr(self.prf, "prf_data_dict"):
                            for key,val in self.prf.prf_data_dict.items():
                                self.data_dict[key] = val                        

                # check if we found a vertex for both hemispheres; if not, go to criteria
                if isinstance(self.lh_best_vert_mask, np.ndarray) and isinstance(self.rh_best_vert_mask, np.ndarray):

                    # Calculate normal using the standard method. Other options are "cross" and "Newell"
                    self.fetch_normal()

                    vert = []
                    for hemi,tag,bids_hemi in zip(['left', 'right'],["lh","rh"],["hemi-L","hemi-R"]):

                        coord = getattr(self, f"{tag}_best_vertex_coord")
                        vertex = getattr(self, f"{tag}_best_vertex")
                        normal = getattr(self.surface, f"{tag}_surf_normals")[vertex]
                        
                        # append vertices to list for log file
                        vert.append(vertex)

                        utils.verbose(f"Found following vertex in {hemi} hemisphere:", self.verbose)
                        for nn,el in zip(["vertex","coord","normal"],[vertex,coord,normal]):
                            utils.verbose(f" {nn}\t= {el}", self.verbose)
                            
                        if self.use_prf == True:

                            if not self.srf:
                                # print parameters and make plot
                                cmd =f"call_prfinfo -s {self.subject} -v {vertex} --{tag} --{self.model} -p {self.prf_file} --plot {v1_flag}"
                                print(cmd)
                                os.system(cmd)
                            else:
                                # print parameters
                                cmd = f"call_prfinfo -s {self.subject} -v {vertex} --{tag} --{self.model} -p {self.prf_file}"

                                print(cmd)
                                os.system(cmd)

                                # compile output name for figures
                                base = f"{self.subject}"

                                if hasattr(self, "file_components"):
                                    for it in ["ses","task","acq"]:
                                        if it in list(self.file_components.keys()):
                                            base += f"_{it}-{self.file_components[it]}"
                            
                                fname = opj(self.prf_dir, f"{base}_{bids_hemi}_vox-{vertex}_model-{self.model}_stage-iter.svg")

                                # create the plot
                                if not self.manual_vertices:
                                    self.make_srf_plot(hemi=tag, save_as=fname)
                                else:
                                    self.make_prf_plot(hemi=tag, save_as=fname)

                    # # Smooth vertex maps
                    # print("Smooth vertex maps for visual verification")
                    self.vertex_to_map()
                    # self.smooth_vertexmap()

                    if isinstance(vert,(list,np.ndarray)):
                        self.webshow = False
                
                    if self.webshow:
                        self.tkr = transform.ctx2tkr(self.subject, coord=[self.lh_best_vertex_coord,self.rh_best_vertex_coord])
                        self.tkr_ = {'lh': self.tkr[0], 'rh': self.tkr[1]}
                        os.system(f"freeview -v {self.orig} -f {self.lh_fid}:edgecolor=green {self.rh_fid}:edgecolor=green  --ras {round(self.tkr_['lh'][0],2)} {round(self.tkr_['lh'][1],2)} {round(self.tkr_['lh'][2],2)} tkreg 2>/dev/null")
                    else:
                        utils.verbose("Verification with FreeView disabled", self.verbose)

                    #----------------------------------------------------------------------------------------------------------------
                    # Write out files if all is OK
                    happy = input("Happy with the position? (y/n): ")
                    if happy.lower() in ['y','yes']:
                        utils.verbose(" Alrighty, continuing with these parameters", self.verbose)

                        if self.manual_vertices:
                            self.write_dict = {}
                            self.write_dict["CreatedOn"] = str(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                            self.write_dict["Method"] = "Manual"
                        else:
                            self.write_dict = self.criteria
                            self.write_dict["Method"] = "Criteria+Manual"

                        # write vertices to dictionary
                        self.write_dict["Vertices"] = {}
                        for ii,hh in enumerate(["lh","rh"]):
                            self.write_dict["Vertices"][hh] = int(vert[ii])
                        
                        if isinstance(self.out, str):

                            # write json file
                            self.json_file = opj(ctx_ses, f"cutoffs_pid-{os.getpid()}.json")
                            self.json_object = json.dumps(self.write_dict, indent=4)
                            with open(self.json_file, "w") as outfile:
                                outfile.write(self.json_object)

                        check = True

                    else:
                        if self.use_prf:
                            svgs = utils.get_file_from_substring(["svg", "vox-", f"model-{self.model}"], os.path.dirname(self.prf_file), return_msg=None)
                            if isinstance(svgs, str):
                                svgs = [svgs]

                            if isinstance(svgs,list):
                                if len(svgs) != 0:
                                    for svg in svgs:
                                        os.remove(svg)

            if isinstance(self.out, str):
                self.write_line_pycortex(save_as=self.out)
                utils.verbose(f" writing {self.out}", self.verbose)

            #----------------------------------------------------------------------------------------------------------------
            # Get pRF-parameters from best vertices
            if not skip_prf_info:
                if isinstance(self.prf_file, str):
                    if os.path.exists(self.prf_file):
                        self.prf_data = prf.read_par_file(self.prf_file)

                        fbase = self.subject
                        ls_comps = utils.split_bids_components(self.out)
                        if "ses" in list(ls_comps.keys()):
                            fbase += f"_ses-{ls_comps['ses']}"

                        if self.model != None:
                            fbase += f'_model-{self.model}'

                        self.prf_bestvertex = opj(ctx_ses, f'{fbase}_desc-best_vertices.csv')

                        # print(prf_right)
                        # print(prf_left)
                        lh_df = self.prf.df_prf.iloc[self.lh_best_vertex,:]
                        rh_df = self.prf.df_prf.iloc[self.surface.lh_surf_data[0].shape[0]+self.rh_best_vertex,:]

                        self.dd_dict = {}
                        # write existing pRF parameters to dictionary
                        for ii in list(lh_df.keys()):
                            self.dd_dict[ii] = [lh_df[ii],rh_df[ii]]

                        # add custom stuff
                        self.dd_dict["index"]    = [lh_df.name,rh_df.name]
                        self.dd_dict["position"] = [self.lh_best_vertex_coord, self.rh_best_vertex_coord]
                        self.dd_dict["normal"]   = [self.lh_normal, self.rh_normal]
                        self.dd_dict["hemi"]     = ["L","R"]

                        # we should have stimulus sizes if srf=True
                        if self.srf:
                            try:
                                self.dd_dict["stim_sizes"] = [np.array([getattr(self, f"{i}_stim_size_activation"), getattr(self, f"{i}_stim_size_suppression")]) for i in ["lh","rh"]]

                                self.dd_dict["stim_betas"] = [np.array([getattr(self, f"{i}_max_suppression"), getattr(self, f"{i}_max_activation")]) for i in ["lh","rh"]]                    
                            except:
                                pass

                        self.final_df = pd.DataFrame(self.dd_dict)

                        if isinstance(self.out, str):
                            self.final_df.to_csv(self.prf_bestvertex, index=False)
                            utils.verbose(f" writing {self.prf_bestvertex}", self.verbose)
                            # utils.verbose(f"Now run 'call_sizeresponse -s {self.subject} --verbose {v1_flag}' to obtain DN-parameters", self.verbose)

            utils.verbose("Done", self.verbose)

    def return_criteria(self):
        return self.criteria

    def make_prf_plot(
        self,
        hemi="lh",
        **kwargs):

        min_index = getattr(self, f"{hemi}_best_vertex")

        # plot pRF and timecourse + prediction of ses-1 paradigm
        _,_,_,_ = self.SI_.plot_vox(
            vox_nr=min_index,
            model=self.model,
            stage="iter",
            title="pars",
            edge_color=None,
            make_figure=True,
            **kwargs)

    def make_srf_plot(
        self,
        hemi="lh",
        save_as=None):

        # get target specific information regarding SRF
        min_index       = getattr(self, f"{hemi}_best_vertex")
        size_suppr      = getattr(self, f"{hemi}_stim_size_suppression")
        size_act        = getattr(self, f"{hemi}_stim_size_activation")
        stim_suppr_ix   = getattr(self, f"{hemi}_stim_ix_suppression")
        stim_act_ix     = getattr(self, f"{hemi}_stim_ix_activation")

        # initialize figure
        fig = plt.figure(constrained_layout=True, figsize=(24,5))
        subfigs = fig.subfigures(ncols=2, width_ratios=[4,1])
        gs00 = subfigs[0].subplots(ncols=3, gridspec_kw={"width_ratios": [10,20,10]})
        gs01 = subfigs[1].subplots(nrows=2)
        cols = ["#1B9E77","#D95F02"]

        hemi_vert = int(min_index)
        if hemi == "rh":
            hemi_vert += self.surface.lh_surf_data[0].shape[0]

        # plot pRF and timecourse + prediction of ses-1 paradigm
        # this stuff is indexed based on whole-brain vertex indices
        _,_,_,_ = self.SI_.plot_vox(
            vox_nr=hemi_vert,
            model=self.model,
            stage="iter",
            title="pars",
            edge_color=None,
            make_figure=True,
            axs=[gs00[0],gs00[1]])
        
        # check if columns are integers or not
        if all([isinstance(i, int) for i in list(utils.select_from_df(self.srfs_best_vertices, expression=f"type = norm").columns)]):
            idx = hemi_vert
        else:
            idx = str(hemi_vert)

        # plot SRFs for suppression/activation; indexed based on hemi-specific indexing
        tc_sizes = [utils.select_from_df(self.srfs_best_vertices, expression=f"type = {ii}")[idx].values for ii in ["act","norm"]]
        sizes = np.unique(self.df_sr.reset_index().sizes.values)
        plotting.LazyPlot(
            tc_sizes,
            axs=gs00[2],
            xx=sizes,
            line_width=2, 
            color=["#1B9E77","#D95F02"],
            labels=[f"act ({round(size_act,2)}dva)",f"suppr ({round(size_suppr,2)}dva)"],
            x_label="stimulus size",
            y_label="response",
            add_vline={
                "pos": [size_act,size_suppr],
                "color": ["#1B9E77","#D95F02"]},    
            # x_lim=x_lim,
            add_hline=0)

        # plot the actual stimuli to use
        for ix,(dt,vx) in enumerate(zip(
            ["fill","hole"],
            [stim_act_ix,stim_suppr_ix])):
                            
            # get parameters
            rf = pd.DataFrame(self.SR_.params_df.iloc[hemi_vert]).T

            # make stimulus
            rf_stims,_ = self.SR_.make_stimuli(
                factor=1, 
                dt=dt,
                loc=(rf.x.values[0],rf.y.values[0]))    
            
            self.SR_.plot_stim_size( 
                rf_stims[...,vx], 
                ax=gs01[ix], 
                clip=False, 
                cmap=cols[ix],
                vf_extent=self.SR_.vf_extent)

        plt.tight_layout()
        fig.savefig(
            save_as,
            facecolor="white",
            dpi=300,
            bbox_inches="tight")
        
    def open_pycortex(
        self,
        **kwargs):

        self.pyc = pycortex.SavePycortexViews(
            self.data_dict,
            subject=self.subject,
            **kwargs)

    def save_all(
        self,
        *args,
        **kwargs):

        if not hasattr(self, "pyc"):
            self.open_pycortex(**kwargs)

        self.pyc.save_all(*args, **kwargs)     

def smooth_scalars(
    surf=None, 
    data=None, 
    subject=None,
    **kwargs):

    if not isinstance(surf, SurfaceCalc):
        surf = SurfaceCalc(subject=subject)

    lh_idc = surf.lh_surf_data[0].shape[0]
    data_sm = []
    for hemi,surface in zip(
        ["lh","rh"],
        [surf.lh_surf,surf.rh_surf]):

        # get hemi-specific data
        if hemi == "lh":
            data_h = data[:lh_idc]
        else:
            data_h = data[lh_idc:]

        sm = surface.smooth(data_h, **kwargs)
        data_sm.append(sm)
    
    data_sm = np.concatenate(data_sm)
    
    return data_sm             
