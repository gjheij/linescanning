# pylint: disable=no-member,E1130,E1137
import cortex
from datetime import datetime
from linescanning import (
    glm, 
    info, 
    planning,
    prf, 
    pycortex, 
    transform, 
    utils
    )
import nibabel as nb
import numpy as np
import os
import pandas as pd
from prfpy import stimulus
import random
from scipy.io import loadmat
import warnings

warnings.filterwarnings('ignore')
opj = os.path.join


def set_threshold(name=None, borders=None, set_default=None):

    """
set_threshold

This function is utilized in call_pycortex2 to fetch the thresholds for multiple properties including
pRF-parameters (eccentricity, r2, and polar angle), and structural (surface) properties such as cor-
tical thickness and sulcal depth.

To make verbosity nicer, you can specify a 'name' to print to the terminal what kind of property is
being processed. Then, you can specify a range where the user input should fall in (default is None
to not force a specific range). You can also set a default value if you do not wish to set a certain
threshold (usually this is the minimum/maximum of your array depending on what kind of thresholding
you're about to do, e.g., 'greater/smaller than <value>').

Args:
    name        str:
                For verbosity reasons, a string of the property's name that needs thresholding

    borders     tuple:
                Specific range the user input needs to fall in. Default is None to not enforce a
                range

    set_default int|float:
                Minimum/maximum of array (depending on the kind of thresholding to be applied) if
                you do not wish to enforce a threshold

Usage:
    ecc_val = set_threshold(name="eccentricity", range=(0,15), set_default=min(ecc_array))

"""

    if not name:
        name = "property"

    # set threshold for 'name'
    while True:
        try:
            val = float(input(f" {name} [def = {int(set_default)}]:   ") or set_default)
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

def call_pycortex2(subject,
                   deriv=None,
                   prfdir=os.environ['PRF'],
                   cxdir=os.environ['CTX'],
                   fsdir=os.environ['FS'],
                   task=f"{os.environ['TASK_SES1']}_model-gauss_stage-iter",
                   webshow=True,
                   out=None,
                   roi="V1_exvivo.thresh",
                   use_prf=True,
                   vert=None,
                   ):

    """
call_pycortex2

Python implementation of linescanning/bin/call_pycortex2 with exactly the same features. See that
documentation for more information on the call. It mainly needs a subject ID and a derivatives data.
If you've run everything using this same repository you'll be good to go.

The added advantage of this function is that you get the dictionary (class) back with 1 command:

allstuff = call_pycortex2("sub-001")

If the output file (derivatives/pycortex/sub-xxx/line_pycortex) already exists, it will return the
dataframe contained in that file by means of VertexInfo. This dataframe has the most important info
regarding the position and orientation of the line.

"""

    if deriv:
        # This is mainly for displaying purposes
        dirs = {'prf': opj(deriv, 'prf'),
                'fs': opj(deriv, 'freesurfer'),
                'ctx': opj(deriv, 'pycortex')}

        prfdir, fsdir, cxdir = dirs['prf'], dirs['fs'], dirs['ctx']
    else:
        if not prfdir and not fsdir and not cxdir:
            raise ValueError("Need the paths to pRF/pycortex/FreeSurfer output. Either specify them separately or specify a derivatives folder. See doc!")
        else:
            # This is mainly for displaying purposes
            dirs = {'prf': prfdir,
                    'fs': fsdir,
                    'ctx': cxdir}

    print("Using following directories:")
    [print(f" {i}: {dirs[i]}") for i in dirs]

    if not out:
        out = opj(cxdir, subject, 'line_pycortex.csv')

    #----------------------------------------------------------------------------------------------------------------
    # Read in surface and pRF-data

    if os.path.isfile(out):
        print(f"Loading in {out}")
        return info.VertexInfo(out, subject=subject)
    else:
        if use_prf == True:
            prf_params = utils.get_file_from_substring(f"task-{task}_desc-prf_params.npy", prfdir)
        else:
            prf_params = None
        
        print(f"prf file = {prf_params}")
        print(f"roi      = {roi}")

        # This thing mainly does everything. See the linescanning/optimal.py file for more information
        print("Combining surface and pRF-estimates in one object")
        bv = CalcBestVertex(subject=subject, fs_dir=fsdir, prf_file=prf_params, fs_label=roi)

        #----------------------------------------------------------------------------------------------------------------
        # Set the cutoff criteria based on which you'd like to select a vertex
    
        check = False
        while check == False:

            if not isinstance(vert, np.ndarray):
                print("Set thresholds (leave empty and press [ENTER] to not use a particular property):")
                # get user input with set_threshold > included the possibility to have only pRF or structure only!
                if hasattr(bv, 'prf'):
                    ecc_val     = set_threshold(name="eccentricity", borders=(0,15), set_default=round(min(bv.prf.ecc)))
                    r2_val      = set_threshold(name="r2", borders=(0,1), set_default=round(min(bv.prf.r2)))
                    pol_val_lh  = set_threshold(name="polar angle lh", borders=(0,np.pi), set_default=round(np.pi))
                    pol_val_rh  = set_threshold(name="polar angle rh", borders=(-np.pi,0), set_default=round(-np.pi))
                    pol_val     = [pol_val_lh,pol_val_rh]
                else:
                    ecc_val     = 0
                    r2_val      = 0
                    pol_val     = 0

                if hasattr(bv, 'surface'):
                    thick_val   = set_threshold(name="thickness", borders=(0,5), set_default=max(bv.surface.thickness.data))
                    depth_val   = set_threshold(name="sulcal depth", set_default=round(min(bv.surface.depth.data)))
                else:
                    thick_val   = 0
                    depth_val   = 0

                # print out to confirm
                print("Using these parameters to find vertex with minimal curvature:")
                print(f" eccentricity:    {round(ecc_val,4)}")
                print(f" r2:              {round(r2_val,4)}")
                print(f" polar angle:     {pol_val}")
                print(f" thickness:       {thick_val}")
                print(f" depth:           {round(depth_val,4)}")

                # Create mask using selected criteria
                bv.threshold_prfs(ecc_thresh=ecc_val,
                                 r2_thresh=r2_val,
                                 polar_thresh=pol_val,
                                 depth_thresh=depth_val,
                                 thick_thresh=thick_val)

                # # Look for minimal curvature within that mask
                # bv.mask_curv_with_prf()

                # Pick out best vertex
                bv.best_vertex()
                # check = True
            else:
            
                for i,r in enumerate(['lh', 'rh']):
                    setattr(bv, f"{r}_best_vertex", vert[i])
                    setattr(bv, f"{r}_best_vertex_coord", getattr(bv.surface, f'{r}_surf_data')[0][vert[i]])
                    setattr(bv, f"{r}_best_vert_mask", (getattr(bv.surface, f'{r}_surf_data')[1] == [vert[i]]).sum(0))

            # Calculate normal using the standard method. Other options are "cross" and "Newell"
            bv.normal()

            # Print some stuff to show what's going on
            print("Found following vertex in left hemisphere:")
            print(" coord  = {coord}".format(coord=bv.lh_best_vertex_coord))
            print(" normal = {norm}".format(norm=bv.surface.lh_surf_normals[bv.lh_best_vertex]))
            print(" vertex = {vert}".format(vert=bv.lh_best_vertex))
            
            os.system(f"call_prfinfo -s {subject} -v {bv.lh_best_vertex}")

            print("Found following vertex in right hemisphere:")
            print(" coord  = {coord}".format(coord=bv.rh_best_vertex_coord))
            print(" normal = {norm}".format(norm=bv.surface.lh_surf_normals[bv.rh_best_vertex]))
            print(" vertex = {vert}".format(vert=bv.rh_best_vertex))

            os.system(f"call_prfinfo -s {subject} -v {bv.rh_best_vertex} -h rh")

            # # Smooth vertex maps
            # print("Smooth vertex maps for visual verification")
            # bv.vertex_to_map()
            # bv.smooth_vertexmap()

            # visually check if parameters should be adjusted
            # print(f"  Webshow is set to {webshow}")
            port = random.randint(1024, 65536)

            # place = get_base_dir()[1]
            if isinstance(vert,np.ndarray):
                webshow = False
        
            if webshow:
                orig = opj(fsdir, subject, 'mri', 'orig.mgz')
                tkr = transform.ctx2tkr(subject, coord=[bv.lh_best_vertex_coord,bv.rh_best_vertex_coord])
                tkr_coords = {'lh': tkr[0], 'rh': tkr[1]}
                os.system("freeview -v {orig} -f {lh_fid}:edgecolor=green {rh_fid}:edgecolor=green  --ras {x} {y} {z} tkreg 2>/dev/null".format(orig=orig,
                                                                                                                                                lh_fid=opj(fsdir, subject, 'surf', "lh.fiducial"),
                                                                                                                                                rh_fid=opj(fsdir, subject, 'surf', "rh.fiducial"),
                                                                                                                                                x=round(tkr_coords['lh'][0],2),
                                                                                                                                                y=round(tkr_coords['lh'][1],2),
                                                                                                                                                z=round(tkr_coords['lh'][2],2)))
            else:
                print("Verification with FreeView disabled")

            #----------------------------------------------------------------------------------------------------------------
            # Write out files if all is OK
            happy = input("Happy with the position? (y/n): ")
            if happy.lower() == 'y' or happy == 'yes':
                print(" Alrighty, continuing with these parameters")

                if not isinstance(vert, np.ndarray):
                    textList = ["# Created on {date}\n".format(date=datetime.now().strftime("%d/%m/%Y %H:%M:%S")),
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

                outF = open(opj(cxdir, subject, "cutoffs.o{ext}".format(ext=os.getpid())), "w")
                outF.writelines(textList)
                outF.close()
                check = True

        bv.vertexinfo()
        print(" writing {file}".format(file=out))
        bv.trafo_info.to_csv(out)

        #----------------------------------------------------------------------------------------------------------------
        # Get pRF-parameters from best vertices

        if prf_params and os.path.exists(prf_params):
            prf_data = np.load(prf_params)
            prf_bestvertex = opj(cxdir, subject, f'{subject}_desc-prf_params_best_vertices.csv')

            prf_right = prf_data[bv.surface.lh_surf_data[0].shape[0]:][bv.rh_best_vertex]
            prf_left = prf_data[0:bv.surface.lh_surf_data[0].shape[0]][bv.lh_best_vertex]
            
            # print(prf_right)
            # print(prf_left)
            best_vertex = pd.DataFrame({"hemi":     ["L", "R"],
                                        "x":        [prf_left[0], prf_right[0]],
                                        "y":        [prf_left[1], prf_right[1]],
                                        "size":     [prf_left[2], prf_right[2]],
                                        "beta":     [prf_left[3], prf_right[3]],
                                        "baseline": [prf_left[4], prf_right[4]],
                                        "r2":       [prf_left[5], prf_right[5]],
                                        "ecc":      [bv.prf.ecc[bv.lh_best_vertex], bv.prf.ecc[bv.surface.lh_surf_data[0].shape[0]:][bv.rh_best_vertex]],
                                        "polar":    [bv.prf.polar[bv.lh_best_vertex], bv.prf.polar[bv.surface.lh_surf_data[0].shape[0]:][bv.rh_best_vertex]],
                                        "index":    [bv.lh_best_vertex, bv.rh_best_vertex],
                                        "position": [bv.lh_best_vertex_coord, bv.rh_best_vertex_coord],
                                        "normal":   [bv.lh_normal, bv.rh_normal]})

            best_vertex = best_vertex.set_index(['hemi'])

            best_vertex.to_csv(prf_bestvertex)
            print(" writing {file}".format(file=prf_bestvertex))
            
            # load pRF-experiment details
            vf_extent = [-5, 5]
            design_fn = utils.get_file_from_substring("vis_design.mat", opj(os.environ['DIR_DATA_HOME'], 'code'))
            design_matrix = loadmat(design_fn)

            prf_stim = stimulus.PRFStimulus2D(screen_size_cm=70, screen_distance_cm=225, design_matrix=design_matrix['stim'],TR=1.5)
            prf_lh = prf.make_prf(prf_stim, size=prf_left[2], mu_x=prf_left[0], mu_y=prf_left[1])
            prf_rh = prf.make_prf(prf_stim, size=prf_right[2], mu_x=prf_right[0], mu_y=prf_right[1])

            prf.plot_prf(prf_lh, vf_extent, save_as=opj(cxdir, subject, f'{subject}_hemi-L_desc-prf_position.pdf'))
            prf.plot_prf(prf_rh, vf_extent, save_as=opj(cxdir, subject, f'{subject}_hemi-R_desc-prf_position.pdf'))

        try:
            lh_bold = np.load(utils.get_file_from_substring("avg_bold_hemi-L.npy", os.path.dirname(prf_params)))
            rh_bold = np.load(utils.get_file_from_substring("avg_bold_hemi-R.npy", os.path.dirname(prf_params)))

            glm.plot_array(lh_bold[:,bv.lh_best_vertex], 
                           x_label="volumes", 
                           y_label="amplitude (z-score)", 
                           font_size=14,
                           save_as=opj(cxdir, subject, f'{subject}_hemi-L_desc-prf_timecourse.pdf'))

            glm.plot_array(rh_bold[:,bv.rh_best_vertex], 
                           x_label="volumes", 
                           y_label="amplitude (z-score)", 
                           font_size=14,
                           save_as=opj(cxdir, subject, f'{subject}_hemi-R_desc-prf_timecourse.pdf'))

        except:
            print("Could not create timecourse-plots")

        print("Done")
        return bv


class SurfaceCalc(object):

    """
SurfaceCalc (class)

This object does all the surface initialization given a subject and a freesurfer directory. For instance,
it reads in the curvature, thickness, and sulcal depth maps from freesurfer, smooths the curvature map,
reads by default the V1_exvivo_thresh label in, and creates a boolean mask of this label. So with this
one class you will have everything you need from the surface calculations.

Functions within SurfaceCalc:
    - __init__          (does everything)
    - smooth_surfs      (smooths the curvature map with a set kernel of 5 mm)
    - read_fs_label     (reads in FS-label from 'surf' directory for both hemispheres)
    - label_to_mask     (create boolean mask from label that can be read in by pycortex)

Args:
    subject         str:
                    subject name as defined in the freesurfer/pycortex directories

    fs_dir          str:
                    path to freesurfer directory if not equal to SUBJECTS_DIR

Usage:
    surf_calcs = SurfaceCalc(subj="sub-001")

Note:
    contained within CalcBestVertex, so if you can also just call that class and you won't
    have to run the command in "usage"

    """

    def __init__(self, subject=None, fs_dir=os.environ['SUBJECTS_DIR'], fs_label="V1_exvivo.thresh"):

        """Initialize object"""

        # print(" Perform surface operations")
        self.subject = subject
        self.ctx_path = opj(cortex.database.default_filestore, self.subject)
        self.fs_path = fs_dir

        if not os.path.isdir(self.ctx_path):
            # import subject from freesurfer (will have the same names)
            cortex.freesurfer.import_subj(fs_subject=self.subject,
                                            cx_subject=self.subject,
                                            freesurfer_subject_dir=self.fs_path,
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
        tmp = self.read_fs_label(subject=self.subject, fs_path=self.fs_path, fs_label=fs_label, hemi="both")
        setattr(self, f'lh_{self.roi_label}', tmp['lh'])
        setattr(self, f'rh_{self.roi_label}', tmp['rh'])

        # this way we can also use read_fs_label for more custom purposes
        # self.read_fs_label(fs_label=fs_label)
        pp = self.label_to_mask(subject=self.subject, lh_arr=getattr(self, f'lh_{self.roi_label}'), rh_arr=getattr(self, f'rh_{self.roi_label}'), hemi="both")
        self.lh_roi_mask = pp['lh_mask']
        self.rh_roi_mask = pp['rh_mask']
        self.whole_roi = pp['whole_roi']
        self.whole_roi_v = pp['whole_roi_v']

            # self.lh_surf_sm,self.rh_surf_sm = smooth_surfs
        # except:
        #     print(" WARNING: Pycortex and FS dimensions do not match")

    def smooth_surfs(self, kernel=10, nr_iter=1):

        """smooth surfaces with a given kernel size. The kernel size does not refer to mm, but to a factor. Therefore it has to be an integer value"""

        if not isinstance(kernel,int):
            print(f" Rounding smoothing factor '{kernel}' to '{int(kernel)}'")
            kernel = int(kernel)
        setattr(self, 'lh_surf_sm', self.lh_surf.smooth(self.curvature.data[:self.lh_surf_data[0].shape[0]], factor=kernel, iterations=nr_iter))
        setattr(self, 'rh_surf_sm', self.rh_surf.smooth(self.curvature.data[self.lh_surf_data[0].shape[0]:], factor=kernel, iterations=nr_iter))

    @staticmethod
    def read_fs_label(subject, fs_path=os.environ['SUBJECTS_DIR'], fs_label=None, hemi="both"):

        """read a freesurfer label file (name must match with file in freesurfer directory"""

        # # dots are annoying in attributes, to replace them with underscores; won't fail if there aren't any present
        # setattr(self, 'roi_label', fs_label.replace('.', '_'))

        # print("reading {}".format(opj(self.fs_path, self.subject, 'label', f'{fs_label}.label')))
        if hemi == "both":
            return {'lh': nb.freesurfer.io.read_label(opj(fs_path, subject, 'label', f'lh.{fs_label}.label'), read_scalars=False),
                    'rh': nb.freesurfer.io.read_label(opj(fs_path, subject, 'label', f'rh.{fs_label}.label'), read_scalars=False)}
            # [setattr(self, f'{i}_{self.roi_label}', nb.freesurfer.io.read_label(opj(self.fs_path, self.subject, 'label', f'{i}.{fs_label}.label'), read_scalars=False)) for i in ['lh','rh']]

        else:
            if hemi.lower() != "lh" and hemi.lower() != "rh":
                raise ValueError(f"Hemisphere should be one of 'both', 'lh', or 'rh'; not {hemi}")
            else:
                label_file = opj(fs_path, subject, 'label', f'{hemi}.{fs_label}.label')
                # setattr(self, f'{hemi}_{self.roi_label}', nb.freesurfer.io.read_label(label_file, read_scalars=False))
                return {hemi: nb.freesurfer.io.read_label(label_file, read_scalars=False)}
    
    def label_to_mask(self, subject=None, lh_arr=None, rh_arr=None, hemi="both"):

        """convert freesurfer label or set of vertices to boolean vector"""

        if hemi == "both":

            lh_mask = np.zeros(self.lh_surf_data[0].shape[0], dtype=bool)
            # lh_mask[getattr(self, f'lh_{self.roi_label}')] = True
            lh_mask[lh_arr] = True
            # self.lh_roi_mask = lh_mask

            rh_mask = np.zeros(self.rh_surf_data[0].shape[0], dtype=bool)
            # rh_mask[getattr(self, f'rh_{self.roi_label}')] = True
            rh_mask[rh_arr] = True
            # self.rh_roi_mask = rh_mask

        else:
            if lh_arr and not rh_arr:
                lh_mask = np.zeros(getattr(self, f"lh_surf_data")[0].shape[0], dtype=bool)
                lh_mask[lh_arr] = True
                rh_mask = np.zeros(getattr(self, f"rh_surf_data")[0].shape[0], dtype=bool)

            elif rh_arr and not lh_arr:
                lh_mask = np.zeros(getattr(self, f"lh_surf_data")[0].shape[0], dtype=bool)
                rh_mask = np.zeros(getattr(self, f"rh_surf_data")[0].shape[0], dtype=bool)
                
                rh_mask[rh_arr] = True
            else:
                raise ValueError(f"You entered both rh_arr and lh_arr, but 'hemi' = {hemi}")

        whole_roi = np.concatenate((lh_mask, rh_mask))
        whole_roi_v = cortex.Vertex(whole_roi, subject=subject, vmin=-0.5, vmax=1)
        return {'lh_mask': lh_mask,
            'rh_mask': rh_mask,
            'whole_roi': whole_roi,
            'whole_roi_v': whole_roi_v}

class pRFCalc(object):

    """
pRFCalc

This short class deals with the population receptive field modeling output from spinoza_fitprfs and/or call_prfpy.
Ideally, the output of these scripts is a numpy array containing the 6 pRF-parameters for each voxel. If you have
one of those files, specify it in the prf_file argument. If, for some reason, you do not have this file, but sepa-
rate files for each pRF variable (e.g., a file for R2, a file for eccentricitry, and a file for polar angle), make
sure they are all in 1 directory and specify that directory in the prf_dir parameter.

The only function of this class is to collect path names and the data arrays containing information about the pRF
parameters. It will actually be used in the CalcBestVertex-class.

Args:
    subject         str:
                    subject name as defined in the freesurfer/pycortex directories

    prf_dir         str|optional:
                    path to pRF-output directory possible containing maps for variance explained (desc-R2.npz), ecc-
                    entricity (desc-eccentricity_map.npz), and polar angle (desc-polarangle_map.npz) as created with
                    call_fitprfs and  spinoza_fitprfs

    prf_file        str|optional:
                    path to a desc-prf_params.npy file containing a 6xn dataframe representing the 6 variables from
                    the pRF-analysis times the amount of TRs (time points). You can either specify this file directly
                    or specify the pRF-directory containing separate files for R2, eccentricity, and polar angle if
                    you do not happen to have the prf parameter file


Usage:
    prf = pRFCalc(subject="sub-001", prf_file="sub-001_desc-prf_params.npy")

    """

    # Get stuff from SurfaceCalc
    def __init__(self, subject=None, prf_file=None, prf_dir=None, ses_nr=1, task=os.environ['TASK_SES1']):

        self.subject = subject
        self.session = ses_nr
        self.task_id = task
        self.fname   = f"{self.subject}_ses-{self.session}_task-{self.task_id}"

        if not prf_file or not os.path.exists(prf_file):

            if prf_dir != None:

                try:
                    # print(" Load in pRF-estimates")
                    self.r2_file      = utils.get_file_from_substring("R2", opj(prf_dir, self.subject))
                    self.ecc_file     = utils.get_file_from_substring("eccentricity", opj(prf_dir, self.subject))
                    self.polar_file   = utils.get_file_from_substring("polar", opj(prf_dir, self.subject))
                except:
                    # print(" Set pRF-estimates to none")
                    self.r2      = None
                    self.ecc     = None
                    self.polar   = None

            else:
                raise NameError("Could not find pRF-parameter file and also not loose r2, ecc, polar angle files..")

        else:

            if os.path.exists(prf_file):

                # print(" Extracting pRF-parameters from file")

                self.prf_file = prf_file
                self.prf_params = np.load(self.prf_file)

                self.r2 = self.prf_params[:,-1]; np.save(opj(os.path.dirname(prf_file), f"{self.fname}_desc-R2_map.npy"), self.r2)
                self.ecc = np.sqrt(self.prf_params[:,0]**2+self.prf_params[:,1]**2); np.save(opj(os.path.dirname(prf_file), f"{self.fname}_desc-eccentricity_map.npy"), self.ecc)
                self.polar = np.angle(self.prf_params[:,0]+self.prf_params[:,1]*1j); np.save(opj(os.path.dirname(prf_file), f"{self.fname}_desc-polarangle_map.npy"), self.polar)

            else:
                raise FileNotFoundError("pRF-parameter file does not exist")
        
        # print(self.r2, self.ecc, self.polar)
        if isinstance(self.r2, np.ndarray) and isinstance(self.ecc, np.ndarray) and isinstance(self.polar, np.ndarray):
            # self.r2_v       = cortex.Vertex(self.r2, subject=self.subject, cmap="inferno")
            # self.ecc_v      = cortex.Vertex2D(self.ecc, self.r2, vmin=0, vmax=12, vmin2=0.05, vmax2=0.4, subject=self.subject, cmap="spectral_alpha")
            # self.polar_v    = cortex.Vertex2D(self.polar, self.r2, vmin=-np.pi, vmax=np.pi, vmin2=0.05, vmax2=0.4, subject=self.subject, cmap="hsv_alpha")

            r2_thr = 0.1
            self.r2_v = pycortex.make_r2(self.subject, r2=self.r2)
            self.ecc_v = pycortex.make_ecc(self.subject, ecc=self.ecc, r2=self.r2, r2_thresh=r2_thr)
            self.polar_v = pycortex.make_polar(self.subject, polar=self.polar, r2=self.r2, r2_thresh=r2_thr)


class CalcBestVertex(object):

    """
CalcBestVertex

This class actually takes in all attributes from pRFCalc and SurfaceCalc and combines the surface information contained in the
surface-class with the information from the pRF-class. Specifically, it will call upon SurfaceCalc and pRFCalc as init function
given you complete access by just calling CalcBestVertex.

Args:
    subject         str:
                    subject name as defined in the freesurfer/pycortex directories

    fs_dir          str:
                    path to freesurfer directory if not equal to SUBJECTS_DIR

    ctx_dir         str:
                    path to Pycortex directory (default = filestore)

    prf_dir         str:
                    path to pRF-output directory possible containing maps for variance explained (desc-R2.npz), eccentricity
                    (desc-eccentricity_map.npz), and polar angle (desc-polarangle_map.npz) as created with call_fitprfs and
                    spinoza_fitprfs

    prf_file        str:
                    path to a desc-prf_params.npy file containing a 6xn dataframe representing the 6 variables from the pRF-
                    analysis times the amount of TRs (time points). You can either specify this file directly or specify the
                    pRF-directory containing separate files for R2, eccentricity, and polar angle if you do not happen to
                    have the prf parameter file

    fs_label        str:
                    ROI to be used; must be a label present in <fs_dir>/<subject>/label; default = V1_exvivo.thresh. It
                    should be following the nomenclature of FreeSurfer but omitting the 'lh'/'rh' to include both hemispheres
                    in the analysis

Functions (creates attributes for each hemisphere):
    - __init__              (calls upon SurfaceCalc and pRFCalc using the arguments given to CalcBestVertex)
    - threshold_prfs        (creates a mask to search for the best vertex in using criteria specified by the user)
    - mask_curv_with_prf    (applies the mask created in 'threshold_prfs' to the curvature map)
    - best_vertex           (obtains the best vertex for both hemispheres, by getting the minimal curvature in the map created
                            with mask_curv_with_prf)
    - normal                (calculates the normal vector at the best vertices)
    - vertex_to_map         (create a boolean mask of the vertex to verify its position in pycortex)
    - smooth_vertexmap      (smooths the boolean mask created in vertex_to_map)

Usage:
    bv = CalcBestVertex(subject=subject, fs_dir=opj(os.environ['DIR_DATA_DERIV'], 'freesurfer_orig'), prf_file=prf_params)

Notes:
    - Only 'threshold_prfs' function requires actual arguments (see threshold_prfs.__doc__), the others can be called as is

    """

    def __init__(self, subject=None,
                 fs_dir=os.environ['SUBJECTS_DIR'],
                 ctx_dir=os.environ['CTX'],
                 prf_dir=os.environ['PRF'],
                 prf_file=None,
                 fs_label="V1_exvivo.thresh",
                 ses_nr=1,
                 task=os.environ['TASK_SES1']):

        self.subject = subject
        self.surface = SurfaceCalc(subject=self.subject, fs_dir=fs_dir, fs_label=fs_label)
        self.session = ses_nr
        self.task_id = task
        self.fname   = f"{self.subject}_ses-{self.session}_task-{self.task_id}"

        # try:
        self.prf = pRFCalc(subject=self.subject, prf_file=prf_file)
        # except:
        #     print("No pRF-parameters included; probably a mismatch between vertices of pRF & Pycortex/FreeSurfer")
        #     pass
        self.prf_dir = prf_dir

    def threshold_prfs(self, r2_thresh=None, ecc_thresh=None, polar_thresh=None, thick_thresh=None, depth_thresh=None):

        """
    threshold_prfs

    Apply thresholds to pRF-parameters and multiply with V1-label mask to obtain only pRFs with certain characteristics in V1. The
    pRF characteristics are embedded in the prf-class, containing the r2, eccentricity, and polar angle values for all vertices.
    Additionally, the surface-class contains structural information regarding thickness and sulcal depth. Each of these parameters
    can be adjusted to find your optimal pRF.

    Args:
        r2_thres        int|float;  refers to amount of variance explained. Usually between 0 and 1 (defaults to the minimum r2).
                                    Threshold is specified as 'greater than <value>'

        ecc_thresh      int|float;  refers to size of pRF (smaller = foveal, larger = peripheral). Usually between 0 and 15. Defaults
                                    to minimum of r2 array. Threshold is specified as 'lower than <value>'

        polar_thresh    list|float; refers to polar angle (upper-to-lower vertical meridian). Usually between -pi and pi. Defaults
                                    to -pi. Threshold is specified as 'greater than <value>'. A list of values for left and right
                                    hemisphere should be given

        thick_thresh    int|float;  refers to cortical thickness as per the thickness.npz file created during the importing of a
                                    FreeSurfer subjects into Pycortex. The map is defined as NEGATIVE VALUES!! Thicker cortex is
                                    represented by a lower negative value, usually between -5 and 0. Defaults to the minimum value
                                    of the thickness array. Threshold is specified as 'lower than <value>'.

        depth_thresh    int|float;  refers to sulcal depth (location of deep/superficial sulci) as per the sulcaldepth.npz file
                                    created during the importing of a FreeSurfer subjects into Pycortex. Defaults to the minimum
                                    value of the sulcaldepth array. Threshold is specified as 'greater than <value>'.


    Returns:
        self.lh_prf         attr, boolean mask of where in V1 the specified criteria are met in the left hemisphere
        self.rh_prf         attr, boolean mask of where in V1 the specified criteria are met in the right hemisphere
        self.joint_mask_v   attr, vertex map of the mask created by the specified criteria

    Usage:
        bv.threshold_prfs(r2_thresh=0.4, ecc_thresh=3, polar_thresh=5)

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

            self.r2_thresh      = r2_thresh or min(self.prf.r2)
            self.ecc_thresh     = ecc_thresh or max(self.prf.ecc)
            self.polar_thresh   = polar_thresh or [min(self.prf.polar[:self.surface.lh_surf_data[0].shape[0]]),min(self.prf.polar[self.surface.lh_surf_data[0].shape[0]:])]

            # print(self.polar_thresh)
            lh_polar = self.prf.polar[:self.surface.lh_surf_data[0].shape[0]] < self.polar_thresh[0]
            rh_polar = self.prf.polar[self.surface.lh_surf_data[0].shape[0]:] > self.polar_thresh[1]

            # print(lh_polar.shape,rh_polar.shape)
            polar = np.concatenate((lh_polar,rh_polar))
            self.prf_mask = ((self.prf.r2 > self.r2_thresh) * (self.prf.ecc < self.ecc_thresh) * (polar))

        if hasattr(self, 'surface'):
            self.thick_thresh   = thick_thresh or max(self.surface.thickness.data)
            self.depth_thresh   = depth_thresh or min(self.surface.depth.data)

            self.struct_mask =  ((self.surface.thickness.data < self.thick_thresh) * (self.surface.depth.data > self.depth_thresh))

        # apply to label mask
        if hasattr(self, 'prf_mask') and hasattr(self, 'struct_mask'):
            # we have both prf and surf masks
            self.joint_mask = (self.prf_mask * self.struct_mask * self.surface.whole_roi)
        elif hasattr(self, 'prf_mask') and not hasattr(self, 'struct_mask'):
            # only pRF-data
            self.joint_mask = (self.prf_mask * self.surface.whole_roi)
        elif hasattr(self, 'struct_mask') and not hasattr(self, 'prf_mask'):
            # only pRF-data
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
        self.lh_prf = self.joint_mask[:self.surface.lh_surf_data[0].shape[0]].astype(bool) # values are stacked, left first then right (https://gallantlab.github.io/database.html)
        self.rh_prf = self.joint_mask[self.surface.lh_surf_data[0].shape[0]:].astype(bool)

        self.joint_mask_v = cortex.Vertex(np.nan_to_num(self.joint_mask.astype(int)), subject=self.subject, cmap='magma', vmin=0.5)

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

    def normal(self, method="ctx"):

        """Fetch normal vector of best vertex. We can do this by calculating the neighbouring vertices
        and taking the cross product of these vectorsself.

        Also see: https://sites.google.com/site/dlampetest/python/calculating-normals-of-a-triangle-mesh-using-numpy"""

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

    def ret_normal(self, hemi="both"):

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

        """Create a vertex object that can be read in by pycortex. We can also combine the left/right
        hemisphere by setting the concat flag to True. 'Write_files'-flag indicates that we should write
        the files to .npy files in the pRF-directory."""

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
                np.save(opj(self.prf_dir, self.subject, f'{self.fname}_desc-vertex_hemi-LR.npy'), self.lr_best_vertex_map)


    def smooth_vertexmap(self, kernel=5, concat=True, write_files=True):

        """Smooth the vertex map with a given kernel size. We can also combine the left/right hemisphere by setting
        the concat flag to True. 'Write_files'-flag indicates that we should write the files to .npy files in the
        pRF-directory."""

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

    def vertexinfo(self, hemi="both"):

        """
    vertexinfo

    This function creates the line_pycortex files containing the angles and translation given the vertex ID, normal
    vector, and RAS coordinate. It uses vertex_rotation in utils.py to calculate these things. It will return a pan-
    das dataframe containing the relevant information for the vertices in both hemispheres.

    Args:
        self        inherit from parent class
        hemi        what hemisphere should we process? ("both", "lh", "rh")

    Usage:
        vertexinfo()

        """

        if hemi == "both":
            # do stuff for both hemispheres
            rot_lh = planning.create_line_pycortex(self.lh_normal, "left", self.lh_best_vertex, coord=self.lh_best_vertex_coord)
            rot_rh = planning.create_line_pycortex(self.rh_normal, "right", self.rh_best_vertex, coord=self.rh_best_vertex_coord)
            rot_df = pd.concat([rot_lh, rot_rh]).set_index(['parameter'])

            self.trafo_info = rot_df

        else:

            if hemi == "left" or hemi == "lh" or hemi.lower() == "l":
                tag = "lh"
            elif hemi == "right" or hemi == "rh" or hemi.lower() == "r":
                tag = "rh"

            rot = planning.create_line_pycortex(getattr(self, f'{tag}_normal'), "left", getattr(self, f'{tag}_best_vertex'), coord=getattr(self, f'{tag}_best_vertex_coord'))
            setattr(self, f'{tag}_trafo_info', rot)

    def to_angle(self, hemi="both"):
        
        if hemi == "both":
            if hasattr(self, "lh_normal") and hasattr(self, "rh_normal"):
                df = {'lh': planning.normal2angle(self.lh_normal),
                      'rh': planning.normal2angle(self.rh_normal)}

            return df
