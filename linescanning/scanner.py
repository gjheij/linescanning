# pylint: disable=no-member,E1130,E1137
from datetime import datetime
from linescanning import transform, utils, planning
import nibabel as nb
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
opj = os.path.join

class Scanner(object):
    """Scanner

    This object combines the information from the CalcBestVertex and pRFCalc classes to fetch the best vertex and related information from the surfaces. In fact, it runs these classes internally, so you'd only have to specify this class in order to run the others as well. At the minimum, you need to specify the subject ID as defined in the FreeSurfer/pycortex/BIDS-anat directory, the dictionary resulting from CalcBestVertex, the matrix mapping session 1 to session 2 (the 'fs2ses=' flag), the hemisphere you're interested in, and the FreeSurfer directory.

    Parameters
    ----------
    subject: str
        subject-ID corresponding to the name in the pycortex filestore directory (mandatory!! for looking up how much the image has been moved when imported from FreeSurfer to Pycortex)
    df: dict
        output from optimal.target_vertex; a dictionary containing the result from :class:`linescanning.optimal.CalcBestVertex`
    fs2ses: str, numpy.ndarray
        (4,4) array or path to matrix file mapping FreeSurfer to the new session. It sort of assumes the matrix has been created with ANTs, so make sure the have run `call_antsregistration`. Enter 'identity' to use an identity matrix; this is useful if you want to plan the line in FreeSurfer-space (before session 2)
    new_anat: str
        string pointing to the new anatomy file. Generally this is in `<project>/sourcedata/<subject>/<ses>/planning/nifti`, if you've ran `spinoza_lineplanning`. Should be the **fixed** input for call_antsregistration to create `fs2ses`
    hemisphere: str
        hemisphere you're interested in. Should be 'right' or 'left'. It will do all operations on both hemispheres anyway, but only print the actual values you need to insert in the MR console for the specified hemisphere
    fs_dir: str
        path to FreeSurfer directory; will default to SUBJECTS_DIR if left empty.
    ses: int, optional
        session ID of new session. Generally this will be `ses-2`, pointing to a line-scanning session. Should be >1
    print_to_console: bool
        boolean whether to print the translation/rotations that should be inserted in the console to the terminal. Can be turned off for debugging reasons
    debug: bool
        boolean for debugging mode; if True, we'll print more information about the (converted) angles/normal vectors

    Returns
    ----------
    Attr
        sets attributes in an instantiation of `scanner.Scanner` and will lead to printing of information for the MR-console (if `print_to_console=True`)

    Example
    ----------
    >>> scan = scanner.Scanner(optimal.target_vertex('sub-001'), fs2ses='genaff.mat', hemi='left', new_anat='path/to/sub-001_ses-2_T1w.nii.gz')
    """

    def __init__(
        self, 
        df,
        fs2ses=None,
        new_anat=None,
        hemi=None,
        fs_dir=None,
        ses=2,
        print_to_console=True,
        debug=False):

        if fs_dir == None:
            self.fs_dir = os.environ['SUBJECTS_DIR']
        else:
            self.fs_dir = fs_dir
            
        self.pycortex = df
        self.fs2ses = fs2ses
        self.new_anat = new_anat
        self.subject = self.pycortex.subject
        self.ses = ses

        # set reference anatomy to orig
        self.ref_anat = opj(fs_dir,self.subject,'mri','orig.mgz')

        if hemi == None:
            hemi = "left"

        self.hemi = hemi

        if hasattr(self.pycortex, 'infofile'):
         
            # assuming I got an infofile dataframe from utils.VertexInfo
            self.normals = self.pycortex.get('normal')
            self.ctx_coords = self.pycortex.get('position')
            self.vertices = self.pycortex.get('index')

        else:
            # assuming I got entire dataframe from CalcBestVertex
            self.normals = {
                "lh": self.pycortex.lh_normal,
                "rh": self.pycortex.rh_normal
            }

            self.ctx_coords = {
                "lh": self.pycortex.lh_best_vertex_coord,
                "rh": self.pycortex.rh_best_vertex_coord
            }

            self.vertices   = {
                "lh": self.pycortex.lh_best_vertex,
                "rh": self.pycortex.rh_best_vertex
            }
        
        # set default anatomies
        self.fs_orig = self.ref_anat
        self.fs_raw = opj(fs_dir, self.subject, 'mri', 'rawavg.mgz')

        print("Account for offset induced by pycortex")
        # correct Pycortex coordinate with 'surfmove' to get TKR coordinate
        tkr = transform.ctx2tkr(self.subject,coord=self.ctx_coords.values())
        self.tkr_coords = {
            'lh': tkr[0],
            'rh': tkr[1]
        }

        print("Convert FreeSurfer TKR to Freesurfer")
        # convert TKR coordinate to FS coordinate
        fs = transform.tkr2fs(self.subject,coord=[self.tkr_coords['lh'], self.tkr_coords['rh']], fs_dir=fs_dir)
        self.fs_coords = {
            'lh': fs[0],
            'rh': fs[1]
        }

        # convert FS directly to session 2
        self.fs_chicken = {
            'lh': utils.make_chicken_csv(self.fs_coords['lh'], output_file=opj(os.path.dirname(self.fs_raw), f"{self.subject}_space-fs_hemi-L_vert-{self.vertices['lh']}_desc-lps.csv")),
            'rh': utils.make_chicken_csv(self.fs_coords['rh'], output_file=opj(os.path.dirname(self.fs_raw), f"{self.subject}_space-fs_hemi-R_vert-{self.vertices['rh']}_desc-lps.csv"))
        }
        
        if self.fs2ses != "identity":
            print(f"Convert FreeSurfer straight to session {self.ses}")
            self.ses2_chicken = {
                'lh': transform.ants_applytopoints(self.fs_chicken['lh'], utils.replace_string(self.fs_chicken['lh'], "space-fs", f"space-ses{self.ses}"), self.fs2ses),
                'rh': transform.ants_applytopoints(self.fs_chicken['rh'], utils.replace_string(self.fs_chicken['rh'], "space-fs", f"space-ses{self.ses}"), self.fs2ses)
            }
        else:
            print(f"Identity-matrix was specified, using existing chicken files")
            self.ses2_chicken = self.fs_chicken.copy()

        # get coordinate in ses-2 (RAS/LPS/VOX convention)
        self.ses2_ras = {
            'lh': utils.read_chicken_csv(self.ses2_chicken['lh'], return_type="ras"),
            'rh': utils.read_chicken_csv(self.ses2_chicken['rh'], return_type="ras")
        }

        self.ses2_lps = {
            'lh': utils.read_chicken_csv(self.ses2_chicken['lh']),
            'rh': utils.read_chicken_csv(self.ses2_chicken['rh'])
        } 

        self.ses2_vox = {
            'lh': self.to_vox(self.ses2_ras['lh'], self.new_anat),
            'rh': self.to_vox(self.ses2_ras['rh'], self.new_anat)
        }

        # warp the normal vector (= IN RAS SPACE!!)
        if self.fs2ses != None:
            print("Applying rotation matrix to normal vector")
            self.warp_normals(system="RAS")
        else:
            print("\nWARNING: could not warp normal vector!")

        # get angles and correct
        print("Converting normals to angles and correct for scanner-interpretation of angles")

        # session 1 angles are relative to RAS-axis ([1,0,0],[0,1,0],[0,0,1])
        #
        self.ses1_angles_raw = {
            'lh': planning.normal2angle(self.normals['lh'], system="RAS", unit="rad"),
            'rh': planning.normal2angle(self.normals['rh'], system="RAS", unit="rad")
        }

        if debug:
            print(f"Session 1 normal: {self.normals['lh']}")
            print(f"Session 1 angles: {np.rad2deg(self.ses1_angles_raw['lh'])}")

        # Angles calculated like above are not coplanar, which is defined as: "A group of points 
        # that don't all lie in the same plane are non-coplanar." To make them coplanar, we first 
        # consider the normal vector without Z-component, flattening the vector into the XY plane.
        # We can then recalculate the angles properly.
        #
        # We can project the normal vector to the XY-plane by multiplying the vector with the sin
        # over the angle with the Z-axis
        #
        # See: https://www.youtube.com/watch?v=vVPwQgoSG2g
        for ii in ['lh','rh']:
            # now get the angles for all axes in radians so we can push the vector to the XY-plane with the Z-angle            
            self.ses1_angles_raw[ii][:2] = planning.normal2angle(
                self.normals[ii][:2]*np.sin(self.normals[ii][-1]), 
                system="RAS", 
                return_axis=['x','y'])

            # convert the z-angle to degrees
            self.ses1_angles_raw[ii][-1] = self.ses1_angles_raw[ii][-1] * (180/np.pi)

            if debug:
                if ii == "lh":
                    print(f"Coplanar angles 1: {self.ses1_angles_raw['lh']}")

        if hasattr(self, 'ses1_angles_raw'):
            self.ses1_angles_corr = {
                'lh': planning.correct_angle(self.ses1_angles_raw['lh']),
                'rh': planning.correct_angle(self.ses1_angles_raw['rh'])
            }

        if hasattr(self, 'warped_normals'):

            # convert the vector to LPS too
            self.warped_normals_lps = {
                'lh': transform.ras2lps(self.warped_normals['lh']),
                'rh': transform.ras2lps(self.warped_normals['rh'])
            }

            # now get the angles for all axes in radians so we can push the vector to the XY-plane with the Z-angle
            self.ses2_angles_nonCoPlanar = {
                'lh': planning.normal2angle(self.warped_normals_lps['lh'], system="LPS", unit="rad"),
                'rh': planning.normal2angle(self.warped_normals_lps['rh'], system="LPS", unit="rad")
            }

            if debug:
                print(f"Warped normal LPS {self.warped_normals_lps['lh']}")
                print(f"Warped normal RAS {self.warped_normals['lh']}")
                print(f"Session {self.ses} angles: {np.rad2deg(self.ses2_angles_nonCoPlanar['lh'])}")
                
            # Get component of normal that is in XY-plane and calculate angles with that vector.
            self.ses2_angles_raw = {}
            for ii in ['lh','rh']:

                # initiate dictionary
                self.ses2_angles_raw[ii] = np.zeros((3,))

                # If we have a 90 degree with the Y-axis, this immediately means coronal slice. No need to correct for coplanar
                # angles:
                if int(np.rad2deg(self.ses2_angles_nonCoPlanar[ii][0])) == 90:
                    # angles are coplanar
                    self.ses2_angles_raw[ii] = np.degrees(self.ses2_angles_nonCoPlanar[ii])
                    self.ses2_angles_raw[ii][0] = 0
                elif int(np.rad2deg(self.ses2_angles_nonCoPlanar[ii][1])) == 90:
                    self.ses2_angles_raw[ii] = np.degrees(self.ses2_angles_nonCoPlanar[ii])
                    self.ses2_angles_raw[ii][1] = 0
                else:
                    # calculate coplanar angles
                    vector = self.warped_normals_lps[ii].copy()

                    vector_xy = np.array((*vector[:2],0))*np.sin(self.ses2_angles_nonCoPlanar[ii][-1])
                    self.ses2_angles_raw[ii][:2] = planning.normal2angle(
                        vector_xy, 
                        system="LPS", 
                        return_axis=['x','y'])
                    
                    # vector_yz = np.array((0,*vector[-2:]))*np.sin(self.ses2_angles_nonCoPlanar[ii][1])
                    # self.ses2_angles_raw[ii][-1] = normal2angle(vector_yz, 
                    #                                             system="LPS", 
                    #                                             return_axis=['z'])

                    self.ses2_angles_raw[ii][-1] = np.rad2deg(self.ses2_angles_nonCoPlanar[ii][-1]) #np.rad2deg(vector[-1])

                    if debug:
                        if ii == "lh":
                            print(f"Coplanar angles {self.ses}: {self.ses2_angles_raw['lh']}")

                # self.ses2_angles_raw[ii][-1] = self.ses2_angles_nonCoPlanar[ii][-1] * (180/np.pi)

        if hasattr(self, 'ses2_angles_raw'):
            # turn on verbose for left hemisphere
            # turn OFF 'only_angles' to also know what z-axis angle means
            print(f"Angles in:  {self.ses2_angles_raw['lh']}")
            self.ses2_angles_corr = {
                'lh': planning.correct_angle(
                    self.ses2_angles_raw['lh'], 
                    verbose=True, 
                    only_angles=False
                ),
                'rh': planning.correct_angle(
                    self.ses2_angles_raw['rh'], 
                    only_angles=False
                )
            }
            
            print(f"Angles out: {list(self.ses2_angles_corr['lh'])}")

        if hasattr(self, 'ses2_angles_corr'):
            print("Fetch console settings") 
            self.foldover = {
                'lh': planning.get_console_settings(
                    self.ses2_angles_corr['lh'][0], 
                    "left", 
                    self.vertices['lh'],
                    z_axis_meaning=self.ses2_angles_corr['lh'][1]
                ),
                'rh': planning.get_console_settings(
                    self.ses2_angles_corr['rh'][0],
                    "right", 
                    self.vertices['rh'],
                    z_axis_meaning=self.ses2_angles_corr['rh'][1]
                )
            }
            print("Done")
            
        if hasattr(self, 'hemi') and self.hemi != None:
            if print_to_console:
                try:
                    self.print_to_console(hemi=self.hemi)
                except:
                    raise ValueError("Missing attributes to print MR-console values")
        else:
            raise ValueError("Hemisphere is not specified. Please start again with 'hemi=left' or something alike")

    @staticmethod
    def to_vox(coord,ref):
        """get RAS coordinate from voxel"""
        return transform.native_to_scanner(ref, coord=coord, inv=True)

    @staticmethod
    def to_ras(coord,ref):
        """get voxel coordinate from RAS"""
        return transform.native_to_scanner(ref, coord=coord)

    @staticmethod
    def write_nifti(coord,ref,output=None):
        """create nifti image from voxel coordinate"""

        if isinstance(ref, nb.Nifti1Image) or isinstance(ref, nb.freesurfer.mghformat.MGHImage):
            img = ref
        elif isinstance(ref, str):
            # convert mgz to nifti
            if ref.endswith('mgz'):
                img = nb.freesurfer.load(ref)
            else:
                img = nb.load(ref)

        elif ref == None:
            raise ValueError("'ref' = None..")
        else:
            raise ValueError("Unknown file-type.. Either use an nibabel.Nifti1Image, .nii.gz, or .mgz image (the latter will be converted to nifti)")

        empty_fs = np.zeros_like(img.get_fdata())
        # empty_fs[coord[0]-1,coord[1]-1,coord[2]-1] = 1
        empty_fs[coord[0],coord[1],coord[2]] = 1      
        if output:
            if output.endswith('.nii') or output.endswith('.nii.gz'):
                empty_fs = nb.Nifti1Image(empty_fs, affine=img.affine, header=img.header)
            elif output.endswith('.mgz'):
                empty_fs = nb.freesurfer.MGHImage(empty_fs, img.affine, header=img.header)
            empty_fs.to_filename(output)
        else:
            return empty_fs

    def get_fs_coord(self, hemi="lh"):
        if hemi.lower() != "lh" and hemi.lower() != "rh":
            raise ValueError(f"'hemi' should be either 'lh' or 'rh', not {hemi}")

        if len(self.fs_coords[hemi]) != 3:
            coord = self.fs_coords[hemi][:3]
        else:
            coord = self.fs_coords[hemi]

        return coord

    def get_ses1_coord(self, hemi="lh"):
        if hemi.lower() != "lh" and hemi.lower() != "rh":
            raise ValueError(f"'hemi' should be either 'lh' or 'rh', not {hemi}")

        if len(self.rawavg_coords[hemi]) != 3:
            coord = self.rawavg_coords[hemi][:3]
        else:
            coord = self.rawavg_coords[hemi]

        return coord

    def get_ses2_coord(self, hemi="lh"):
        if hemi.lower() != "lh" and hemi.lower() != "rh":
            raise ValueError(f"'hemi' should be either 'lh' or 'rh', not {hemi}")

        if len(self.ses2_coords[hemi]) != 3:
            coord = self.ses2_coords[hemi][:3]
        else:
            coord = self.ses2_coords[hemi]

        return coord

    def warp_normals(self, hemi="both", system="RAS"):
        """warp_normals
    
        Apply a rigid-body transformation on a vector. Important here is that the coordinate systems used are the same: e.g., only warp an RAS-vector with an RAS-matrix, and a LPS-vector with an LPS-matrix. ANTs outputs per definition an LPS-matrix, so we'd need to convert that with ConvertTransformFile first

        Parameters
        ----------
        hemi: str (default = 'both')
            the dataframe has a normal vector for the left hemisphere and right hemisphere. Easiest to leave this to what it is.
        system: str (default = 'ras')
            if we warp an RAS-vector we need an RAS matrix, if we warp an LPS vector we need an LPS matrix. The function totate_normal will actually deal with this problem

        Example
        ----------
        >>> self.warp_normal = self.warp_normals()
        """

        if hemi == "both":
            if hasattr(self, "normals"):
                self.warped_normals = {}
                for i in ['lh', 'rh']:
                    self.warped_normals[i] = planning.rotate_normal(
                        self.normals[i], 
                        self.fs2ses,
                        system=system
                    )
        else:
            raise NotImplementedError(f"Im lazy, sorry.. Just to both hemi's please")

    def get_all_coords(self,hemi="lh", out_type="both"):
        """Just print all coordinates to the terminal"""

        if out_type.lower() == "vox":
            print(f"FreeSurfer VOX:  {self.to_vox(self.fs_coords[hemi],self.fs_orig)[:3]}")
            print(f"Session 1 VOX:   {self.rawavg_coords[hemi][:3]}")
            print(f"Session 2 VOX:   {self.ses2_coords[hemi][:3]}")
        elif out_type.lower() == "ras":
            print(f"FreeSurfer RAS:  {self.fs_coords[hemi][:3]}")
            print(f"Session 1 RAS:   {self.to_ras(self.rawavg_coords[hemi],self.fs_raw)[:3]}")
            print(f"Session 2 RAS:   {self.to_ras(self.ses2_coords[hemi],self.new_anat)[:3]}")
        elif out_type.lower() == "both":
            print(f"FreeSurfer VOX:  {self.to_vox(self.fs_coords[hemi],self.fs_orig)[:3]}; RAS = {self.fs_coords[hemi][:3]}")
            print(f"Session 1 VOX:   {self.rawavg_coords[hemi][:3]}; RAS = {self.to_ras(self.rawavg_coords[hemi],self.fs_raw)[:3]}")
            print(f"Session 2 VOX:   {self.ses2_coords[hemi][:3]}; RAS = {self.to_ras(self.ses2_coords[hemi],self.new_anat)[:3]}")

    def print_to_console(self, hemi=None):

        if hemi.lower() in ["left","lh","l"]:
            tag = "lh"
        elif hemi.lower() in ["right","rh","r"]:
            tag = "rh"
        else:
            raise ValueError(f"Unknown input for 'hemi': {hemi}. Must be of the format 'lh','left', or 'l' (same for right hemisphere)")

        info = self.foldover[tag]

        textList = ["# Created on {date}\n\n".format(date=datetime.now().strftime("%d/%m/%Y %H:%M:%S")),
            "---------------------------------------------------------------------------------------------------\n",
            "ENTER THE FOLLOWING VALUES IN THE MR-CONSOLE\n",
            "\n",
            "set orientation to " + utils.color.BOLD + utils.color.RED + info['value'][0] + utils.color.END + " and foldover to " + utils.color.BOLD + utils.color.RED + info['value'][1] + utils.color.END + "\n",
            " FH: {angle} deg\n".format(angle=round(info['value'][5],2)),
            " AP: {angle} deg\n".format(angle=round(info['value'][4],2)),
            " RL: {angle} deg\n".format(angle=round(info['value'][3],2)),
            "\n",
            "set translation to:\n",
            " AP: {angle} mm\n".format(angle=round(self.ses2_lps[tag][1],2)),
            " RL: {angle} mm\n".format(angle=round(self.ses2_lps[tag][0],2)),
            " FH: {angle} mm".format(angle=round(self.ses2_lps[tag][2],2)),
            "\n",
            f"Targeted hemisphere: {hemi}\n",
            f"Vertex number:       {self.vertices[tag]}\n",
            f"Isocenter RAS:       {self.ses2_ras[tag][:3]}\n",
            f"Isocenter LPS:       {self.ses2_lps[tag][:3]}"
            ]
        
        try:
            cx_dir = opj(os.environ.get("CTX"), self.subject, f"ses-{self.ses}")
            log_file = opj(cx_dir, "console.o{ext}".format(ext=os.getpid()))
            outF = open(log_file, "w")
            outF.writelines(textList)
            outF.close()
        except:
            fs_dir = opj(self.fs_dir, self.subject, 'mri')
            log_file = opj(fs_dir, "console.o{ext}".format(ext=os.getpid()))
            outF = open(log_file, "w")
            outF.writelines(textList)
            outF.close()

        print("")
        for l in textList[1:]:
            print(l.split("\n")[0])

        print("")
