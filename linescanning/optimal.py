# pylint: disable=no-member,E1130,E1137
import cortex
from datetime import datetime
from linescanning import (
    planning,
    dataset, 
    pycortex, 
    transform, 
    utils,
    prf
    )
import nibabel as nb
import numpy as np
import os
import pandas as pd
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
            val = float(input(f" {name} [def = {int(set_default)}]: \t") or set_default)
        except ValueError:
            print(" Please enter a number")
            continue
        else:
            pass

        if borders and len(borders) == 2:
            if borders[0] <= float(val) <= borders[1]:
                return float(val)
            else:
                print(f" WARNING: specified range is ({borders[0]},{borders[1]}), your value is {val}. Try again..")
                continue

        else:
            return float(val)

def target_vertex(subject,
                  deriv=None,
                  prf_file=None,
                  webshow=True,
                  out=None,
                  roi="V1_exvivo.thresh",
                  use_prf=True,
                  vert=None):

    """target_vertex

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
    webshow: bool
        Start `FreeView` for visual verification. During debugging this is rather cumbersome, so you can turn it off by specifying `webshow=False`
    roi: str, optional
        ROI-name to extract the vertex from as per ROIs created with `FreeSurfer`. Default is V1_exvivo.thresh
    use_prf: bool
        In case you want to base the selection of the target_vertex on both structural and functional properties, set `use_prf=True`. If you only want to include anatomical information such as curvature, set `use_prf=False`. This is relevant for non-visual experiments
    verts: list, optional
        List of manually selected vertices rather than selecting the vertices based on structural/functional properties

    Returns
    ---------
    str
        Creates vertex-information files `<subject>_desc-prf_params_best_vertices.csv` & `line_pycortex.csv` as well as figures showing the timecourses/pRF-location of selected vertices. Doesn't 
    
    class
        :class:`linescanning.CalcBestVertex`

    Example
    ----------
    >>> # use Gaussian iterative parameters to find target vertex for sub-001 in V1_exvivo.thresh by using the default derivatives folders
    >>> optimal.target_vertex("sub-001", 
    >>>                       deriv="/path/to/derivatives",
    >>>                       prf_file="gaussian_pars.npy", 
    >>>                       use_prf=True,                         # default
    >>>                       out="line_pycortex.csv",              # default
    >>>                       roi="V1_exvivo.thresh",               # default
    >>>                       webshow=True)                         # default
    """

    # check if we can read DIR_DATA_DERIV if necessary
    if not deriv:
        deriv = os.environ.get("DIR_DATA_DERIV")

    # set paths
    if deriv:
        prf_dir = opj(deriv, 'prf'),
        fs_dir = opj(deriv, 'freesurfer'),
        cx_dir = opj(deriv, 'pycortex')

    # create if necessary
    if not os.path.exists(cx_dir):
        os.makedirs(cx_dir)

    # update the filestore
    pycortex.set_ctx_path(p=cx_dir)

    if not out:
        out = opj(cx_dir, subject, 'line_pycortex.csv')

    #----------------------------------------------------------------------------------------------------------------
    # Read in surface and pRF-data

    if os.path.isfile(out):
        print(f"Loading in {out}")
        return utils.VertexInfo(out, subject=subject, hemi="both")
    else:
        if use_prf == True:
            if not os.path.exists(prf_file):
                raise FileNotFoundError(f"Could not find with pRF-estimates '{prf_file}'")

            try:
                model = utils.split_bids_components(prf_file)['model']
            except:
                model = "gauss"
        else:
            prf_file = None
            model = None
        
        print(f"pRFs = {prf_file}")
        print(f"ROI = {roi}")

        # This thing mainly does everything. See the linescanning/optimal.py file for more information
        print("Combining surface and pRF-estimates in one object")
        BV_ = CalcBestVertex(subject=subject, 
                             deriv=deriv, 
                             prf_file=prf_file, 
                             fs_label=roi)

        #----------------------------------------------------------------------------------------------------------------
        # Set the cutoff criteria based on which you'd like to select a vertex
    
        check = False
        while check == False:

            if not isinstance(vert, np.ndarray):
                print("Set thresholds (leave empty and press [ENTER] to not use a particular property):")
                # get user input with set_threshold > included the possibility to have only pRF or structure only!
                if hasattr(BV_, 'prf'):
                    size_val    = set_threshold(name="pRF size (beta)", borders=(0,5), set_default=round(min(BV_.prf.size)))
                    r2_val      = set_threshold(name="r2 (variance)", borders=(0,1), set_default=round(min(BV_.prf.r2)))
                    ecc_val     = set_threshold(name="eccentricity", borders=(0,15), set_default=round(min(BV_.prf.ecc)))
                    pol_val_lh  = set_threshold(name="polar angle lh", borders=(0,np.pi), set_default=round(np.pi))
                    pol_val_rh  = set_threshold(name="polar angle rh", borders=(-np.pi,0), set_default=round(-np.pi))
                    pol_val     = [pol_val_lh,pol_val_rh]
                else:
                    size_val    = 0
                    ecc_val     = 0
                    r2_val      = 0
                    pol_val     = 0

                if hasattr(BV_, 'surface'):
                    thick_val   = set_threshold(name="thickness (mm)", borders=(0,5), set_default=max(BV_.surface.thickness.data))
                    depth_val   = set_threshold(name="sulcal depth", set_default=round(min(BV_.surface.depth.data)))
                else:
                    thick_val   = 0
                    depth_val   = 0

                # print out to confirm
                print("Using these parameters to find vertex with minimal curvature:")
                print(f" pRF size:        {round(size_val,2)}")
                print(f" eccentricity:    {round(ecc_val,4)}")
                print(f" r2:              {round(r2_val,4)}")
                print(f" polar angle:     {pol_val}")
                print(f" thickness:       {thick_val}")
                print(f" depth:           {round(depth_val,4)}")

                # Create mask using selected criteria
                BV_.apply_thresholds(ecc_thresh=ecc_val,
                                     size_thresh=size_val,
                                     r2_thresh=r2_val,
                                     polar_thresh=pol_val,
                                     depth_thresh=depth_val,
                                     thick_thresh=thick_val)

                # Pick out best vertex
                BV_.best_vertex()
                
            else:
            
                for i,r in enumerate(['lh', 'rh']):
                    setattr(BV_, f"{r}_best_vertex", vert[i])
                    setattr(BV_, f"{r}_best_vertex_coord", getattr(BV_.surface, f'{r}_surf_data')[0][vert[i]])
                    setattr(BV_, f"{r}_best_vert_mask", (getattr(BV_.surface, f'{r}_surf_data')[1] == [vert[i]]).sum(0))

            # Calculate normal using the standard method. Other options are "cross" and "Newell"
            BV_.fetch_normal()

            for hemi in ['left', 'right']:
                if hemi == "left":
                    tag = "lh"
                else:
                    tag = "rh"

                coord = getattr(BV_, f"{tag}_best_vertex_coord")
                vertex = getattr(BV_, f"{tag}_best_vertex")
                normal = getattr(BV_.surface, f"{tag}_surf_normals")[vertex]
                
                print(f"Found following vertex in {hemi} hemisphere:")
                print(f" vertex = {vertex}")
                print(f" coord  = {coord}")
                print(f" normal = {normal}")

                if use_prf == True:
                    os.system(f"call_prfinfo -s {subject} -v {vertex} --{tag} --{model} -p {prf_file}")

            # # Smooth vertex maps
            # print("Smooth vertex maps for visual verification")
            # BV_.vertex_to_map()
            # BV_.smooth_vertexmap()

            if isinstance(vert,np.ndarray):
                webshow = False
        
            if webshow:
                orig = opj(fs_dir, subject, 'mri', 'orig.mgz')
                tkr = transform.ctx2tkr(subject, coord=[BV_.lh_best_vertex_coord,BV_.rh_best_vertex_coord])
                tkr_ = {'lh': tkr[0], 'rh': tkr[1]}
                lh_fid=opj(fs_dir, subject, 'surf', "lh.fiducial")
                rh_fid=opj(fs_dir, subject, 'surf', "rh.fiducial")
                os.system(f"freeview -v {orig} -f {lh_fid}:edgecolor=green {rh_fid}:edgecolor=green  --ras {round(tkr_['lh'][0],2)} {round(tkr_['lh'][1],2)} {round(tkr_['lh'][2],2)} tkreg 2>/dev/null")
            else:
                print("Verification with FreeView disabled")

            #----------------------------------------------------------------------------------------------------------------
            # Write out files if all is OK
            happy = input("Happy with the position? (y/n): ")
            if happy.lower() == 'y' or happy == 'yes':
                print(" Alrighty, continuing with these parameters")

                if not isinstance(vert, np.ndarray):
                    textList = ["# Created on {date}\n".format(date=datetime.now().strftime("%d/%m/%Y %H:%M:%S")),
                                f"size: {size_val}\n",
                                f"ecc: {ecc_val}\n",
                                f"r2: {r2_val}\n",
                                f"polar: {pol_val}\n",
                                f"thickness: {thick_val}\n",
                                f"depth: {depth_val}\n",
                                ]
                else:
                    textList = ["# Created on {date}\n".format(date=datetime.now().strftime("%d/%m/%Y %H:%M:%S")),
                                "Manually selected following vertices:\n",
                                f"lh: {vert[0]}\n",
                                f"rh: {vert[1]}\n"]

                outF = open(opj(cx_dir, subject, "cutoffs.o{ext}".format(ext=os.getpid())), "w")
                outF.writelines(textList)
                outF.close()
                check = True

        BV_.write_line_pycortex(save_as=out)
        print(" writing {file}".format(file=out))

        #----------------------------------------------------------------------------------------------------------------
        # Get pRF-parameters from best vertices

        if os.path.exists(prf_file):
            prf_data = prf.read_par_file(prf_file)

            if model != None:
                fbase = f'{subject}_model-{model}'
            else:
                fbase = subject

            prf_bestvertex = opj(cx_dir, subject, f'{fbase}_desc-best_vertices.csv')

            prf_right = prf_data[BV_.surface.lh_surf_data[0].shape[0]:][BV_.rh_best_vertex]
            prf_left = prf_data[0:BV_.surface.lh_surf_data[0].shape[0]][BV_.lh_best_vertex]
            
            # print(prf_right)
            # print(prf_left)
            best_vertex = pd.DataFrame({"hemi":     ["L", "R"],
                                        "x":        [prf_left[0], prf_right[0]],
                                        "y":        [prf_left[1], prf_right[1]],
                                        "size":     [prf_left[2], prf_right[2]],
                                        "beta":     [prf_left[3], prf_right[3]],
                                        "baseline": [prf_left[4], prf_right[4]],
                                        "r2":       [prf_left[5], prf_right[5]],
                                        "ecc":      [BV_.prf.ecc[BV_.lh_best_vertex], BV_.prf.ecc[BV_.surface.lh_surf_data[0].shape[0]:][BV_.rh_best_vertex]],
                                        "polar":    [BV_.prf.polar[BV_.lh_best_vertex], BV_.prf.polar[BV_.surface.lh_surf_data[0].shape[0]:][BV_.rh_best_vertex]],
                                        "index":    [BV_.lh_best_vertex, BV_.rh_best_vertex],
                                        "position": [BV_.lh_best_vertex_coord, BV_.rh_best_vertex_coord],
                                        "normal":   [BV_.lh_normal, BV_.rh_normal]})

            best_vertex.to_csv(prf_bestvertex)
            print(" writing {file}".format(file=prf_bestvertex))
            print(f"Now run 'call_sizeresponse -s {subject} --verbose' to obtain DN-parameters (don't forget to use '--adjust' where applicable!")

        print("Done")
        return BV_


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

    Example
    ----------
    >>> surf_calcs = SurfaceCalc(subj="sub-001")

    Notes
    ----------
    Embedded in :class:`linescanning.optimal.CalcBestVertex`, so if you can also just call that class and you won't have to run the command in "usage"
    """

    def __init__(self, subject=None, fs_dir=None, fs_label="V1_exvivo.thresh"):

        """Initialize object"""

        # print(" Perform surface operations")
        self.subject = subject
        self.ctx_path = opj(cortex.database.default_filestore, self.subject)

        if fs_dir == None:
            self.fs_dir = os.environ['SUBJECTS_DIR']
        else:
            self.fs_dir = fs_dir

        if not os.path.exists(self.ctx_path):
            # import subject from freesurfer (will have the same names)
            cortex.freesurfer.import_subj(fs_subject=self.subject,
                                            cx_subject=self.subject,
                                            freesurfer_subject_dir=self.fs_dir,
                                            whitematter_surf='smoothwm')

        self.curvature = cortex.db.get_surfinfo(self.subject, type="curvature")
        self.thickness = cortex.db.get_surfinfo(self.subject, type="thickness")
        self.depth = cortex.db.get_surfinfo(self.subject, type="sulcaldepth")

        self.lh_surf_data, self.rh_surf_data = cortex.db.get_surf(self.subject, 'fiducial')
        self.lh_surf,self.rh_surf = cortex.polyutils.Surface(self.lh_surf_data[0], self.lh_surf_data[1]), cortex.polyutils.Surface(self.rh_surf_data[0], self.rh_surf_data[1])

        # Normal vector for each vertex (average of normals for neighboring faces)
        self.lh_surf_normals = self.lh_surf.vertex_normals
        self.rh_surf_normals = self.rh_surf.vertex_normals
        self.smooth_surfs(kernel=3,nr_iter=3)

        # try:
        setattr(self, 'roi_label', fs_label.replace('.', '_'))
        if not fs_label.endswith('.gii'):
            make_mask = True
            tmp = self.read_fs_label(subject=self.subject, fs_dir=self.fs_dir, fs_label=fs_label, hemi="both")
        else:
            make_mask = False
            tmp = {}
            for ii in ['lh', 'rh']:
                gifti = dataset.ParseGiftiFile(opj(fs_dir, subject, 'label', f"{ii}.{fs_label}"), set_tr=1)

                if gifti.data.ndim > 1:
                    tmp[ii] = np.squeeze(gifti.data, axis=0)
                else:
                    tmp[ii] = gifti.data.copy()

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
            self.whole_roi_v = cortex.Vertex(self.whole_roi, subject=subject, vmin=-0.5, vmax=1)

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
    def read_fs_label(subject, fs_dir=None, fs_label=None, hemi="both"):
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

        # # dots are annoying in attributes, to replace them with underscores; won't fail if there aren't any present
        # setattr(self, 'roi_label', fs_label.replace('.', '_'))

        if not fs_dir:
            fs_dir = os.environ.get("SUBJECTS_DIR")

        if hemi == "both":
            return {'lh': nb.freesurfer.io.read_label(opj(fs_dir, subject, 'label', f'lh.{fs_label}.label'), read_scalars=False),
                    'rh': nb.freesurfer.io.read_label(opj(fs_dir, subject, 'label', f'rh.{fs_label}.label'), read_scalars=False)}
            # [setattr(self, f'{i}_{self.roi_label}', nb.freesurfer.io.read_label(opj(self.fs_dir, self.subject, 'label', f'{i}.{fs_label}.label'), read_scalars=False)) for i in ['lh','rh']]

        else:
            if hemi.lower() != "lh" and hemi.lower() != "rh":
                raise ValueError(f"Hemisphere should be one of 'both', 'lh', or 'rh'; not {hemi}")
            else:
                label_file = opj(fs_dir, subject, 'label', f'{hemi}.{fs_label}.label')
                # setattr(self, f'{hemi}_{self.roi_label}', nb.freesurfer.io.read_label(label_file, read_scalars=False))
                return {hemi: nb.freesurfer.io.read_label(label_file, read_scalars=False)}
    
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
        whole_roi_v = cortex.Vertex(whole_roi, subject=subject, vmin=-0.5, vmax=1)
        return {'lh_mask': lh_mask,
                'rh_mask': rh_mask,
                'whole_roi': whole_roi,
                'whole_roi_v': whole_roi_v}

class pRFCalc():

    """pRFCalc

    This short class deals with the population receptive field modeling output from spinoza_fitprfs and/or call_prfpy. Ideally, the output of these scripts is a numpy array containing the 6 pRF-parameters for each voxel. If you have one of those files, specify it in the `prf_file` argument. If, for some reason, you do not have this file, but separate files for each pRF variable (e.g., a file for R2, a file for eccentricitry, and a file for polar angle), make sure they are all in 1 directory and specify that directory in the `prf_dir` parameter. The only function of this class is to collect path names and the data arrays containing information about the pRF parameters. It will actually be used in :class:`linescanning.optimal.CalcBestVertex`.

    Parameters
    ----------
    prf_file: str, optional
        Path to a desc-prf_params.npy file containing a 6xn dataframe representing the 6 variables from the pRF-analysis times the amount of TRs (time points). You can either specify this file directly or specify the pRF-directory containing separate files for R2, eccentricity, and polar angle if you do not happen to have the prf parameter file
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
    >>> prf = pRFCalc("sub-001_desc-prf_params.npy", thr=0.05)
    """

    # Get stuff from SurfaceCalc
    def __init__(self, prf_file, subject=None, thr=0.1, save=False, fs_dir=None):
        
        # set defaults
        self.prf_file   = prf_file
        self.thr        = thr
        self.save       = save
        self.subject    = subject
        self.fs_dir     = fs_dir

        # check SUBJECTS_DIR
        if not self.fs_dir:
            self.fs_dir = os.environ.get("SUBJECTS_DIR")
        
        # read BIDS components from prf-file
        self.comps = utils.split_bids_components(self.prf_file)

        # set subject
        if not self.subject:
            if "sub" in list(self.comps.keys()):
                self.subject = f"sub-{self.comps['sub']}"
            
        # create output string from input file if we found BIDS components
        self.out_ = ""
        for el in list(self.comps.keys()):
            if el != "desc":
                self.out_ += f"{el}-{self.comps[el]}_"

        if len(self.out_) != 0:
            self.out_ += "desc-"

        # do stuff if file exists
        if os.path.exists(self.prf_file):

            # read file
            self.prf_params = prf.read_par_file(self.prf_file)
            self.prf_dir = os.path.dirname(self.prf_file)

            # obtained pRF parameters
            self.r2     = self.prf_params[:,-1]
            self.size   = self.prf_params[:,2]
            self.ecc    = np.sqrt(self.prf_params[:,0]**2+self.prf_params[:,1]**2)
            self.polar  = np.angle(self.prf_params[:,0]+self.prf_params[:,1]*1j)

            if self.save:
                for est in ['size', 'r2', 'ecc', 'polar']:
                    fname = opj(self.prf_dir, f"{self.out_}{est}.npy")
                    np.save(fname, getattr(self, est))
        else:
            raise FileNotFoundError(f"Could not find file '{self.prf_file}'")
        
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

            self.r2_v       = pycortex.make_r2(self.subject, r2=self.r2)
            self.ecc_v      = pycortex.make_ecc(self.subject, ecc=self.ecc, r2=self.r2, r2_thresh=self.thr)
            self.polar_v    = pycortex.make_polar(self.subject, polar=self.polar, r2=self.r2, r2_thresh=self.thr)


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
        Path to a desc-prf_params.npy file containing a 6xn dataframe representing the 6 variables from the pRF-analysis times the amount of TRs (time points). You can either specify this file directly or specify the pRF-directory containing separate files for R2, eccentricity, and polar angle if you do not happen to have the prf parameter file        
    fs_label: str, optional
        ROI-name to extract the vertex from as per ROIs created with `FreeSurfer`. Default is V1_exvivo.thresh

    Example
    ----------
    >>> BV_ = CalcBestVertex(subject=subject, fs_dir='/path/to/derivatives/freesurfer'), prf_file=prf_params)
    """

    def __init__(self, 
                 subject=None,
                 deriv=None,
                 prf_file=None,
                 fs_label="V1_exvivo.thresh"):

        # read arguments
        self.subject    = subject
        self.deriv      = deriv
        self.prf_file   = prf_file
        self.fs_label   = fs_label

        # set default freesurfer directory
        self.fs_dir = opj(self.deriv, 'freesurfer')
        self.cx_dir = opj(self.deriv, 'pycortex')

        # Get surface object
        self.surface = SurfaceCalc(subject=self.subject, 
                                   fs_dir=self.fs_dir, 
                                   fs_label=self.fs_label)

        if self.prf_file != None:
            self.prf_dir = os.path.dirname(self.prf_file)
            self.prf = pRFCalc(subject=self.subject,
                               prf_file=self.prf_file, 
                               fs_dir=self.fs_dir)

    def apply_thresholds(self, r2_thresh=None, size_thresh=None, ecc_thresh=None, polar_thresh=None, thick_thresh=None, depth_thresh=None):

        """apply_thresholds

        Apply thresholds to pRF-parameters and multiply with V1-label mask to obtain only pRFs with certain characteristics in V1. The pRF characteristics are embedded in the prf-class, containing the r2, eccentricity, and polar angle values for all vertices. Additionally, the surface-class contains structural information regarding thickness and sulcal depth. Each of these parameters can be adjusted to find your optimal pRF.

        Parameters
        ----------
        r2_thres: int, float, optional
            refers to amount of variance explained. Usually between 0 and 1 (defaults to the minimum `r2`). Threshold is specified as 'greater than <value>'
        size_thres: int, float, optional
            refers to size of the pRF in visual space. Usually between 0 and 5 (defaults to 0). Threshold is specified as 'greater than <value>'            
        ecc_thresh: int, float, optional
            refers to `size` of pRF (smaller = foveal, larger = peripheral). Usually between 0 and 15. Defaults to minimum of `r2 array`. Threshold is specified as 'lower than <value>'
        polar_thresh: list, float, optional 
            refers to `polar angle` (upper-to-lower vertical meridian). Usually between -pi and pi. Defaults to -pi. Threshold is specified as 'greater than <value>'. A list of values for left and right hemisphere should be given
        thick_thresh: int, float, optional
            refers to cortical thickness as per the `thickness.npz` file created during the importing of a `FreeSurfer` subjects into `Pycortex`. The map is defined as NEGATIVE VALUES!! Thicker cortex is represented by a lower negative value, usually between -5 and 0. Defaults to the minimum value of the thickness array. Threshold is specified as 'lower than <value>'.
        depth_thresh: int, float, optional
            refers to sulcal depth (location of deep/superficial sulci) as per the `sulcaldepth.npz` file created during the importing of a `FreeSurfer` subjects into `Pycortex`. Defaults to the minimum value of the sulcaldepth array. Threshold is specified as 'greater than <value>'.

        Returns
        ----------
        attr
            self.lh_prf; boolean mask of where in V1 the specified criteria are met in the left hemisphere
            self.rh_prf; boolean mask of where in V1 the specified criteria are met in the right hemisphere
            self.joint_mask_v; vertex map of the mask created by the specified criteria

        Example
        ----------
        >>> BV_.apply_thresholds(r2_thresh=0.4, ecc_thresh=3, polar_thresh=5)
        """

        if r2_thresh:
            if not 0 <= r2_thresh <= 1:
                raise ValueError(f" Hm, this seems to be an odd value. Usually you'll want something between 0-1, your value is {r2_thresh}")

        if ecc_thresh:
            if not 0 <= ecc_thresh <= 15:
                raise ValueError(f" Hm, this seems to be an odd value. Usually you'll want something between 0-15, your value is {ecc_thresh}")

        if thick_thresh:
            if thick_thresh > 0:
                # got a positive value; convert to negative
                thick_thresh = -(thick_thresh)

        # set cutoff criteria

        if hasattr(self, 'prf'):
            
            self.size_thresh    = size_thresh or min(self.prf.size)
            self.r2_thresh      = r2_thresh or min(self.prf.r2)
            self.ecc_thresh     = ecc_thresh or max(self.prf.ecc)
            self.polar_thresh   = polar_thresh or [min(self.prf.polar[:self.surface.lh_surf_data[0].shape[0]]),min(self.prf.polar[self.surface.lh_surf_data[0].shape[0]:])]

            # print(self.polar_thresh)
            lh_polar = self.prf.polar[:self.surface.lh_surf_data[0].shape[0]] <= self.polar_thresh[0]
            rh_polar = self.prf.polar[self.surface.lh_surf_data[0].shape[0]:] >= self.polar_thresh[1]

            # print(lh_polar.shape,rh_polar.shape)
            polar = np.concatenate((lh_polar,rh_polar))
            self.prf_mask = ((self.prf.r2 >= self.r2_thresh) * (self.prf.ecc <= self.ecc_thresh) * (polar) * (self.prf.r2 >= self.r2_thresh) * (self.prf.size >= self.size_thresh))

        if hasattr(self, 'surface'):
            self.thick_thresh   = thick_thresh or max(self.surface.thickness.data)
            self.depth_thresh   = depth_thresh or min(self.surface.depth.data)

            self.struct_mask =  ((self.surface.thickness.data <= self.thick_thresh) * (self.surface.depth.data >= self.depth_thresh))

        # apply to label mask
        if hasattr(self, 'prf_mask') and hasattr(self, 'struct_mask'):
            # we have both prf and surf masks'
            self.joint_mask = (self.prf_mask * self.struct_mask * self.surface.whole_roi)
        elif hasattr(self, 'prf_mask') and not hasattr(self, 'struct_mask'):
            # only pRF-data
            self.joint_mask = (self.prf_mask * self.surface.whole_roi)
        elif hasattr(self, 'struct_mask') and not hasattr(self, 'prf_mask'):
            # only structural
            self.joint_mask = (self.struct_mask * self.surface.whole_roi)
        else:
            # just minimal curvature
            self.joint_mask = self.surface.whole_roi

        # self.joint_mask = ((self.prf.r2 > self.r2_thresh) *
        #                    (self.prf.ecc < self.ecc_thresh) *
        #                    (self.prf.polar > self.polar_thresh) *
        #                    (self.surface.thickness.data < self.thick_thresh) *
        #                    (self.surface.depth.data > self.depth_thresh) *
        #                    (self.surface.whole_roi))

        # save prf information
        self.lh_prf = self.joint_mask[:self.surface.lh_surf_data[0].shape[0]]
        self.rh_prf = self.joint_mask[self.surface.lh_surf_data[0].shape[0]:]
        
        if self.fs_label != "V1_exvivo.thresh":
            self.joint_mask[self.joint_mask < 1] = np.nan
            
        self.joint_mask_v = cortex.Vertex(self.joint_mask, subject=self.subject, vmin=-0.5, vmax=1)
        
        # cortex.webshow({"lh": cortex.Vertex(self.lh_prf.astype(int), subject=self.subject, cmap='magma', vmin=-0.5, vmax=1), "rh": cortex.Vertex(self.rh_prf.astype(int), subject=self.subject, cmap='magma', vmin=-0.5, vmax=1)})

    def best_vertex(self):

        """Fetch best vertex given pRF-properties and minimal curvature"""

        for i in ['lh', 'rh']:

            if hasattr(self, f'{i}_prf'):
                
                curv = np.abs(getattr(self.surface, f'{i}_surf_sm')) # smoothed curvature from SurfaceCalc
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
                    raise ValueError("Could not find a vertex with these parameters")
                # print("list of curvatures: \n",val)

                # find curvature closest to zero
                low = utils.find_nearest(val, 0)

                # print(f"lowest curvature value = {low}")
                # sys.exit(1)
                # look which vertex has this curvature
                for idx,cv in curv_dict.items():
                    if cv == low[-1]:             
                        min_index = idx; setattr(self, f'{i}_best_vertex', min_index)
                
                # 'curv' contains 1e8 for values outside ROI and absolute values for inside ROI (see mask_curv_with_prf)
                # min_index = find_nearest(curv,0)[0]; setattr(self, f'{i}_best_vertex', min_index) # np.argmin(curv); 
                min_vertex = surf[0][min_index]; setattr(self, f'{i}_best_vertex_coord', min_vertex)
                min_vert_mask = (surf[1] == min_vertex).sum(0); setattr(self, f'{i}_best_vert_mask', min_vert_mask)

                # setattr(self, f'{i}_best_vertex', np.argmin(curv))
                # setattr(self, f'{i}_best_vertex_coord', surf[0][getattr(self, f'{i}_best_vertex')])
                # setattr(self, f'{i}_best_vert_mask', (surf[1] == getattr(self, f'{i}_best_vertex_coord')).sum(0))

            else:
                raise TypeError(f'Attribute {i}_prf does not exist')

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


    def vertex_to_map(self, concat=True, write_files=True):

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
                    min_curv_map[self.surface.lh_surf_data[0].shape[0]+self.rh_best_vertex] = 1

                min_curv_map_v = cortex.Vertex(min_curv_map, subject=self.subject, cmap='magma', vmin=-0.5, vmax=1)

                setattr(self, f'{i}_best_vertex_map', min_curv_map)
                setattr(self, f'{i}_best_vertex_map_v', min_curv_map_v)

            else:

                raise TypeError("Missing attributes. Need the curvature data and best vertex index")

        if concat == True:

            both = np.copy(self.lh_best_vertex_map)
            both[np.where(self.rh_best_vertex_map == 1)] = 1
            both_v = cortex.Vertex(both, subject=self.subject, cmap='magma', vmin=-0.5, vmax=1)

            self.lr_best_vertex_map = both
            self.lr_best_vertex_map_v = both_v

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
