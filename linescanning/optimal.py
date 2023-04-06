# pylint: disable=no-member,E1130,E1137
import cortex
from datetime import datetime
import json
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
from scipy import stats
from typing import Union
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
            val = float(input(f" {name} [def = {set_default}]: \t") or set_default)
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

def target_vertex(
    subject,
    deriv=None,
    prf_file=None,
    use_epi=False,
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

    # check if we can read DIR_DATA_DERIV if necessary
    if not deriv:
        deriv = os.environ.get("DIR_DATA_DERIV")

    # set paths
    if deriv:
        prf_dir = opj(deriv, 'prf')
        fs_dir = opj(deriv, 'freesurfer')
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
        BV_ = CalcBestVertex(
            subject=subject, 
            deriv=deriv, 
            prf_file=prf_file, 
            use_epi=use_epi,
            fs_label=roi,
            model=model)

        #----------------------------------------------------------------------------------------------------------------
        # Set the cutoff criteria based on which you'd like to select a vertex
    
        check = False
        while check == False:
            
            a_val = 0
            b_val = 0
            c_val = 0
            d_val = 0
            if not isinstance(vert, np.ndarray):
                print("Set thresholds (leave empty and press [ENTER] to not use a particular property):")
                # get user input with set_threshold > included the possibility to have only pRF or structure only!
                if hasattr(BV_, 'prf'):
                    size_val    = set_threshold(name="pRF size (beta)", borders=(0,5), set_default=round(BV_.prf.df_prf.prf_size.max(),2))
                    r2_val      = set_threshold(name="r2 (variance)", borders=(0,1), set_default=round(BV_.prf.df_prf.r2.min(),2))
                    ecc_val     = set_threshold(name="eccentricity", borders=(0,15), set_default=round(BV_.prf.df_prf.ecc.max(),2))
                    pol_val_lh  = set_threshold(name="polar angle lh", borders=(0,np.pi), set_default=round(np.pi,2))
                    pol_val_rh  = set_threshold(name="polar angle rh", borders=(-np.pi,0), set_default=round(-np.pi,2))
                    pol_val     = [pol_val_lh,pol_val_rh]

                    if model == "norm":
                        a_val = set_threshold(name="A value (norm)", set_default=round(BV_.prf.df_prf.A.min(),2))
                        b_val = set_threshold(name="B value (norm)", set_default=round(BV_.prf.df_prf.B.min(),2))
                        c_val = set_threshold(name="C value (norm)", set_default=round(BV_.prf.df_prf.C.min(),2))
                        d_val = set_threshold(name="D value (norm)", set_default=round(BV_.prf.df_prf.D.min(),2))
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

                if hasattr(BV_, 'epi'):
                    epi_val   = set_threshold(name="EPI value (%)", set_default=10)
                else:
                    epi_val = 0

                # Create mask using selected criteria
                BV_.apply_thresholds(
                    ecc_thresh=ecc_val,
                    size_thresh=size_val,
                    r2_thresh=r2_val,
                    polar_thresh=pol_val,
                    depth_thresh=depth_val,
                    thick_thresh=thick_val,
                    epi_thresh=epi_val,
                    a_thresh=a_val,
                    b_thresh=b_val,
                    c_thresh=c_val,
                    d_thresh=d_val)

                # Pick out best vertex
                BV_.best_vertex()
                
            else:
                
                # set manual vertices in BV_ object
                for i,r in enumerate(['lh', 'rh']):
                    setattr(BV_, f"{r}_best_vertex", vert[i])
                    setattr(BV_, f"{r}_best_vertex_coord", getattr(BV_.surface, f'{r}_surf_data')[0][vert[i]])
                    setattr(BV_, f"{r}_best_vert_mask", (getattr(BV_.surface, f'{r}_surf_data')[1] == [vert[i]]).sum(0))

            # check if we found a vertex for both hemispheres; if not, go to criteria
            if isinstance(getattr(BV_, "lh_best_vert_mask"), np.ndarray) and isinstance(getattr(BV_, "rh_best_vert_mask"), np.ndarray):

                # Calculate normal using the standard method. Other options are "cross" and "Newell"
                BV_.fetch_normal()

                vert = []
                for hemi,tag in zip(['left', 'right'],["lh","rh"]):

                    coord = getattr(BV_, f"{tag}_best_vertex_coord")
                    vertex = getattr(BV_, f"{tag}_best_vertex")
                    normal = getattr(BV_.surface, f"{tag}_surf_normals")[vertex]
                    
                    # append vertices to list for log file
                    vert.append(vertex)

                    print(f"Found following vertex in {hemi} hemisphere:")
                    print(f" vertex = {vertex}")
                    print(f" coord  = {coord}")
                    print(f" normal = {normal}")

                    if use_prf == True:
                        
                        v1_flag = ""
                        if "roi-V1" in os.path.basename(prf_file):
                            v1_flag  = "--v1"

                        os.system(f"call_prfinfo -s {subject} -v {vertex} --{tag} --{model} -p {prf_file} --plot {v1_flag}")

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
                        write_dict = BV_.criteria
                    else:
                        write_dict = {}
                        write_dict["CreatedOn"] = str(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                        write_dict["Method"] = "Manual"
                    
                    # write vertices to dictionary
                    write_dict["Vertices"] = {}
                    for ii,hh in enumerate(["lh","rh"]):
                        write_dict["Vertices"][hh] = int(vert[ii])
                    
                    # write json file
                    json_file = opj(cx_dir, subject, f"cutoffs_pid-{os.getpid()}.json")
                    json_object = json.dumps(write_dict, indent=4)
                    with open(json_file, "w") as outfile:
                        outfile.write(json_object)

                    check = True

                else:
                    if use_prf:
                        svgs = utils.get_file_from_substring(["svg", "vox-", f"model-{model}"], os.path.dirname(prf_file), return_msg=None)
                        if isinstance(svgs, str):
                            svgs = [svgs]

                        if isinstance(svgs,list):
                            if len(svgs) != 0:
                                for svg in svgs:
                                    os.remove(svg)

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

            # print(prf_right)
            # print(prf_left)
            lh_df = BV_.prf.df_prf.iloc[BV_.lh_best_vertex,:]
            rh_df = BV_.prf.df_prf.iloc[BV_.surface.lh_surf_data[0].shape[0]+BV_.rh_best_vertex,:]

            dd_dict = {}
            # write existing pRF parameters to dictionary
            for ii in list(lh_df.keys()):
                dd_dict[ii] = [lh_df[ii],rh_df[ii]]

            # add custom stuff
            dd_dict["index"]    = [lh_df.name,rh_df.name]
            dd_dict["position"] = [BV_.lh_best_vertex_coord, BV_.rh_best_vertex_coord]
            dd_dict["normal"]   = [BV_.lh_normal, BV_.rh_normal]
            dd_dict["hemi"]     = ["L","R"]

            pd.DataFrame(dd_dict).to_csv(prf_bestvertex, index=False)
            print(" writing {file}".format(file=prf_bestvertex))
            print(f"Now run 'call_sizeresponse -s {subject} --verbose {v1_flag}' to obtain DN-parameters")

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

        # check if we need to reload kernel to activate changes to filestore
        if os.environ.get("PROJECT") not in self.ctx_path:
            raise TypeError(f"Project '{os.environ.get('PROJECT')}' not found in '{self.ctx_path}'. This can happen if you changed the filestore, but haven't reloaded the kernel. Please do so to make changes to the filestore available.")
            
        if fs_dir == None:
            self.fs_dir = os.environ.get("SUBJECTS_DIR")
        else:
            self.fs_dir = fs_dir

        if not os.path.exists(self.ctx_path):
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
    def read_fs_annot(subject, fs_dir=None, fs_annot=None, hemi="both"):
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
                    ubject, 
                    'label', 
                    f'{hemi}.{fs_label}.annot')

                return {hemi: nb.freesurfer.io.read_annot(annot_file)}

    @staticmethod
    def read_fs_label(
        subject, fs_dir=None, 
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
                    ubject, 
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
            thr=0.1,
            model="gauss",
            fs_dir=None):
        
        # set defaults
        self.prf_file   = prf_file
        self.thr        = thr
        self.model      = model
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
            self.df_prf = prf.Parameters(self.prf_params, model=self.model).to_df()

            # self.r2     = self.prf_params[:,-1]
            # self.size   = self.prf_params[:,2]
            # self.ecc    = np.sqrt(self.prf_params[:,0]**2+self.prf_params[:,1]**2)
            # self.polar  = np.angle(self.prf_params[:,0]+self.prf_params[:,1]*1j)

            # if self.save:
            #     for est in ['size', 'r2', 'ecc', 'polar']:
            #         fname = opj(self.prf_dir, f"{self.out_}{est}.npy")
            #         np.save(fname, getattr(self, est))
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

            self.r2_v       = pycortex.make_r2(self.subject, r2=self.df_prf.r2)
            self.ecc_v      = pycortex.make_ecc(self.subject, ecc=self.df_prf.ecc, r2=self.df_prf.r2, r2_thresh=self.thr)
            self.polar_v    = pycortex.make_polar(self.subject, polar=self.df_prf.polar, r2=self.df_prf.r2, r2_thresh=self.thr)


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
    epi_file: str, np.ndarray, optional
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
        use_epi=False,
        epi_space="fsnative",
        model="gauss",
        fs_label="V1_exvivo.thresh"):

        # read arguments
        self.subject    = subject
        self.deriv      = deriv
        self.prf_file   = prf_file
        self.use_epi    = use_epi
        self.epi_space  = epi_space
        self.fs_label   = fs_label
        self.model      = model

        # set default derivatives
        if self.deriv == None:
            self.deriv = os.environ.get("DIR_DATA_DERIV")

        # set default freesurfer directory
        self.fs_dir = opj(self.deriv, 'freesurfer')
        self.cx_dir = opj(self.deriv, 'pycortex')

        # Get surface object
        self.surface = SurfaceCalc(subject=self.subject, fs_dir=self.fs_dir, fs_label=self.fs_label)

        if self.prf_file != None:
            self.prf_dir = os.path.dirname(self.prf_file)
            self.prf = pRFCalc(
                subject=self.subject, 
                prf_file=self.prf_file, 
                fs_dir=self.fs_dir, 
                model=self.model)

        if use_epi:
            
            # try to find fmriprep output
            fprep_dir = opj(self.deriv, "fmriprep", self.subject)
            if not os.path.exists(fprep_dir):
                raise FileNotFoundError(f"Could not find directory: '{fprep_dir}'")
            
            gii_files = utils.FindFiles(fprep_dir, extension="gii").files

            if len(gii_files) == 0:
                raise ValueError(f"No files with 'gii' in '{fprep_dir}'")
            
            gii_files_filt = utils.get_file_from_substring([self.epi_space], gii_files)[:2]
            
            tmp = np.hstack([dataset.ParseGiftiFile(ii).data for ii in gii_files_filt])
            self.epi = tmp.mean(axis=0)

            # save vertex object too
            self.epi_v = cortex.Vertex(self.epi, subject=self.subject, cmap="magma")

    def apply_thresholds(
        self, 
        r2_thresh=None, 
        size_thresh=None, 
        ecc_thresh=None, 
        a_thresh=None, 
        b_thresh=None, 
        c_thresh=None, 
        d_thresh=None,         
        epi_thresh=None,
        polar_thresh=None, 
        thick_thresh=None, 
        depth_thresh=None):

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
        a_thresh: int, float, optional
        b_thresh: int, float, optional
        c_thresh: int, float, optional
        d_thresh: int, float, optional        
        epi thresh: int, float, optional
            refers to the EPI intensities. Defaults to all values, at the risk of selecting areas that might have veins. Threshold is specified as 'greater than <value>' and siginifies a percentile value.
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

        # set cutoff criteria
        utils.verbose("Using these parameters to find vertex with minimal curvature:", True)

        # initialize dictionary for log file
        self.criteria = {}
        self.criteria["CreatedOn"] = str(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        self.criteria["Method"] = "Criteria"
        if hasattr(self, 'prf'):
            
            # set thresholds
            self.r2_thresh      = r2_thresh or self.prf.df_prf.r2.min()
            self.size_thresh    = size_thresh or self.prf.df_prf.prf_size.max()
            self.ecc_thresh     = ecc_thresh or self.prf.df_prf.ecc.max()
            
            # parse out polar angle
            self.polar_array    = self.prf.df_prf.polar.values
            self.polar_thresh   = polar_thresh or [max(self.polar_array[:self.surface.lh_surf_data[0].shape[0]]),min(self.polar_array[self.surface.lh_surf_data[0].shape[0]:])]

            lh_polar = self.polar_array[:self.surface.lh_surf_data[0].shape[0]] <= self.polar_thresh[0]
            rh_polar = self.polar_array[self.surface.lh_surf_data[0].shape[0]:] >= self.polar_thresh[1]
            polar = np.concatenate((lh_polar,rh_polar))

            # compile mask
            df = self.prf.df_prf.copy()
            self.prf_mask = np.zeros(df.shape[0], dtype=bool)
            prf_idc = list(
                df.loc[
                    (df.r2 >= self.r2_thresh)
                    & (df.ecc <= self.ecc_thresh)
                    & (df.prf_size <= self.size_thresh)
                ].index)

            # sort out polar angle
            self.prf_mask[prf_idc] = True
            self.prf_mask = (self.prf_mask * polar)
        
            utils.verbose(f" pRF size:      <= {round(self.size_thresh,2)}", True)
            utils.verbose(f" eccentricity:  <= {round(self.ecc_thresh,4)}", True)
            utils.verbose(f" variance (r2): >= {round(self.r2_thresh,4)}", True)
            utils.verbose(f" polar angle:   {self.polar_thresh}", True)
            
            # append to dictionary
            for par,val in zip(
                ["r2","size","ecc","polar angle"],
                [self.r2_thresh,self.size_thresh,self.ecc_thresh,self.polar_thresh]):
                if isinstance(val, list):
                    self.criteria[par] = [float(i) for i in val]
                else:
                    self.criteria[par] = float(val)

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
                norm_idc = list(df.loc[
                    (df.A >= self.a_thresh)
                    & (df.B >= self.b_thresh)
                    & (df.C >= self.c_thresh)
                    & (df.D >= self.d_thresh)
                ].index)
                
                # make mask
                norm_mask = np.zeros_like(self.prf_mask, dtype=bool)
                norm_mask[norm_idc] = True

                # apply mask to existing prf_mask
                self.prf_mask = self.prf_mask * norm_mask

        # include epi intensity mask
        if hasattr(self, 'epi'):
            
            self.epi_thresh = epi_thresh or 0
            utils.verbose(f" EPI intensity: >= {self.epi_thresh}th percentile", True)

            self.epi_mask = np.zeros_like(self.epi, dtype=bool)
            self.epi_mask[self.epi >= np.percentile(self.epi, self.epi_thresh)] = True
            self.criteria["EPI_intensity"] = float(self.epi_thresh)

        # include curvature mask
        if hasattr(self, 'surface'):
            self.thick_thresh   = thick_thresh or max(self.surface.thickness.data)
            self.depth_thresh   = depth_thresh or min(self.surface.depth.data)

            # got a positive value; convert to negative
            if self.thick_thresh:
                if self.thick_thresh > 0:
                    self.thick_thresh = -(self.thick_thresh)

            utils.verbose(f" thickness:     <= {self.thick_thresh}", True)
            utils.verbose(f" depth:         >= {round(self.depth_thresh,4)}", True)
            self.criteria["thickness"] = float(self.thick_thresh)
            self.criteria["depth"] = float(self.depth_thresh)

            self.struct_mask =  ((self.surface.thickness.data <= self.thick_thresh) * (self.surface.depth.data >= self.depth_thresh))

        ### APPLY MASKS

        # start with full mask
        self.joint_mask = self.surface.whole_roi.copy()

        # apply pRF mask
        if hasattr(self, 'prf_mask'):
            self.joint_mask = (self.prf_mask * self.joint_mask)

        # and structural mask
        if hasattr(self, 'struct_mask'):
            self.joint_mask = (self.struct_mask * self.joint_mask)
        
        # and EPI mask
        if hasattr(self, "epi_mask"):
            self.joint_mask = (self.epi_mask * self.joint_mask)

        self.n_vertices = np.count_nonzero(self.joint_mask.astype(int))
        utils.verbose(f"Mask contains {self.n_vertices} vertices", True)

        # save prf information
        self.lh_prf = self.joint_mask[:self.surface.lh_surf_data[0].shape[0]]
        self.rh_prf = self.joint_mask[self.surface.lh_surf_data[0].shape[0]:]
        
        if self.fs_label != "V1_exvivo.thresh":
            self.joint_mask[self.joint_mask < 1] = np.nan
            
        self.joint_mask_v = cortex.Vertex(self.joint_mask, subject=self.subject, vmin=-0.5, vmax=1)

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
                    print(f"WARNING [{i}]: Could not find a vertex with these parameters. Try lowering your r2-criteria if used.")
                    setattr(self, f'{i}_best_vert_mask', None)
                else:
                    if len(val) < 10:
                        add_txt = f": {list(vv)}"
                    else:
                        add_txt = ""
                    print(f"{i}: {len(val)} voxels matched criteria{add_txt}")

                    # find curvature closest to zero
                    low = utils.find_nearest(val, 0)

                    # print(f"lowest curvature value = {low}")
                    # sys.exit(1)
                    # look which vertex has this curvature
                    for idx,cv in curv_dict.items():
                        if cv == low[-1]:             
                            min_index = idx
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
                    if self.rh_best_vertex < self.surface.lh_surf_data[0].shape[0]:
                        min_curv_map[self.surface.lh_surf_data[0].shape[0]+self.rh_best_vertex] = 1
                    else:
                        min_curv_map[self.rh_best_vertex] = 1

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

class Neighbours(SurfaceCalc):

    def __init__(
        self, 
        subject:str=None, 
        fs_dir:str=None, 
        fs_label:str="V1_exvivo.thresh",
        hemi:str="lh",
        verbose:bool=False,
        **kwargs):

        self.subject = subject
        self.fs_dir = fs_dir
        self.fs_label = fs_label
        self.hemi = hemi
        self.verbose = verbose
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
            if self.verbose:
                print("Initializing SurfaceCalc")

            super().__init__(
                subject=self.subject, 
                fs_dir=self.fs_dir,
                fs_label=self.fs_label)

        if isinstance(self.fs_label, str):
            self.create_subsurface()

        if hasattr(self, "target_vert"):
            if isinstance(self.target_vert, int):
                self.distances_to_target(self.target_vert, self.hemi)

    def create_subsurface(self):
        if self.verbose:
            print(f"Creating subsurfaces for {self.fs_label}")        
                    
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

    def create_distance(self):
        # Make the distance x distance matrix.
        ldists, rdists = [], []

        if self.verbose:
            print('Creating distance by distance matrices')

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
