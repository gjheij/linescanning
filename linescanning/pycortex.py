import cortex
from linescanning import (
    image, 
    utils,
    plotting)
import nibabel as nb
import numpy as np
import os
import pandas as pd
import configparser
import sys
import time
from matplotlib.colors import Normalize
from typing import Union
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
        p = os.environ.get('CTX')

    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

    usercfg = cortex.options.usercfg
    config = configparser.ConfigParser()
    config.read(usercfg)

    # check if filestore exists
    try:
        config.get("basic", "filestore")
    except:
        config.set("basic", "filestore", p)
        with open(usercfg, 'w') as fp:
            config.write(fp)

    if opt == "show_fs":
        return config.get("basic", "filestore")
    elif opt == "show_pn":
        return usercfg
    else:
        if config.get("basic", "filestore") != p:
            config.set("basic", "filestore", p)
            with open(usercfg, 'w') as fp:
                config.write(fp)
            
            if not os.path.exists(p):
                os.makedirs(p, exist_ok=True)

            return config.get("basic", "filestore")
        else:
            return config.get("basic", "filestore")
            

def create_ctx_transform(subject, ctx_dir, in_file, warp_name, binarize=False, ctx_type=None, cmap=None):

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

        if os.path.exists(opj(ctx_dir, subject, 'transforms', warp_name)):
            print("Found existing transform; using reference.nii.gz")
            f = opj(ctx_dir, subject, 'warped', warp_name, 'reference.nii.gz')
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

            if os.path.exists(opj(ctx_dir, subject, warp_name)):
                print(f"Transform directory {warp_name} exist")
                pass
            else:
                print(f"Creating transform directory: {warp_name}")
                cortex.db.save_xfm(subject, warp_name, ident, reference=f)

        if ctx_type:

            if not cmap:
                cmap = 'magma'

            if ctx_type == 'Vertex':
                print(f"Creating vertex ..")
                f_ctx = cortex.Vertex(f, subject=subject, cmap=cmap)

            elif ctx_type == 'Volume':
                print(f"Creating volume ..")
                f_ctx = cortex.Volume(f, subject=subject, cmap=cmap, xfmname=warp_name)

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

def get_ctxsurfmove(subject):

    """get_ctxsurfmove

    Following `cortex.freesurfer` module: "Freesurfer uses FOV/2 for center, let's set the surfaces to use the magnet isocenter", where it adds an offset of [128, 128, 128]*the affine of the files in the 'anatomicals'-folder. This short function fetches the offset added given a subject name, assuming a correct specification of the cortex-directory as defined by 'database.default_filestore, cx_subject'

    Parameters
    ----------
    subject: str
        subject name (e.g., sub-xxx)

    Returns
    ----------
    numpy.ndarray
        (4,4) array representing the inverse of the shift induced when importing a `FreeSurfer` subject into `Pycortex`

    Example
    ----------
    >>> offset = get_ctxsurfmove("sub-001")
    """

    anat = opj(cortex.database.default_filestore, subject, 'anatomicals', 'raw.nii.gz')
    if not os.path.exists(anat):
        raise FileNotFoundError(f'Could not find {anat}')

    trans = nb.load(anat).affine[:3, -1]
    surfmove = trans - np.sign(trans) * [128, 128, 128]

    return surfmove


class SavePycortexViews():
    """SavePycortexViews

    Save the elements of a `dict` containing vertex/volume objects to images given a set of viewing settings. If all goes well, a browser will open. Just wait for your settings to be applied in the viewer. You can then proceed from there. If your selected orientation is not exactly right, you can still manually adapt it before running :func:`linescanning.pycortex.SavePycortexViews.save_all()`, which will save all elements in the dataset to png-files given `fig_dir` and `base_name`. 

    Parameters
    ----------
    data_dict: dict, cortex.dataset.views.Vertex, cortex.dataset.views.Volume, linescanning.pycortex.Vertex2D_fix
        Dictionary collecting objects to be projected on the surface or any object that is compatible with Pycortex plotting. Loose inputs will be automatically converted to dictionary.
    subject: str, optional
        Subject name as per Pycortex' filestore naming, by default None
    fig_dir: str, optional
        Output directory for the figures, by default None
    specularity: int, optional
        Level of 'glow' on the surface; ranges from 0-1, by default 0 (nothing at all)
    unfold: int, optional
        Level of inflation the surface needs to undergo; ranges from 0-1, by default 1 (fully inflated)
    azimuth: int, optional
        Rotation around the top-bottom axis, by default 185
    altitude: int, optional
        Rotation around the left-right axis, by default 90
    radius: int, optional
        _description_, by default 163
    pivot: int, optional
        _description_, by default 0
    size: tuple, optional
        _description_, by default (4000,4000)
    data_name: str, optional
        _description_, by default "occipital_inflated"
    zoom: bool, optional
        _description_, by default False
    base_name: str, optional
        Basename for the images to save from the pycortex viewer. If None, we'll default to `<subject>`; `_desc-<>.png` is appended.
    rois_visible: int, optional
        Show the ROIs as per the 'overlays.svg' file on the FSAverage brain. Default = 0
    rois_labels: int, optional
        Show the ROIs labels as per the 'overlays.svg' file on the FSAverage brain. Default = 0
    sulci_visible: int, optional
        Show the sulci outlines on the FSAverage brain. Default = 1
    sulci_labels: int, optional
        Show the sulci labels on the FSAverage brain. Default = 0
    cm_ext: str, optional
        File extension to save the colormap images with. Default = "pdf"
    save_cm: bool, optional
        If the input is an instance of :class:`linescanning.pycortex.Vertex2D_fix`, we can automatically create colormaps from these inputs. These colormaps will be printed to the terminal, but we can also write them to a file. In that case, `fig_dir` and `base_name` is preferred; `_desc-{input_name}_cm.{cm_ext}` will be appended.
    viewer: bool, optional
        Open the viewer (`viewer=True`, default) or suppress it (`viewer=False`)

    Example
    ----------
    >>> from linescanning import pycortex
    >>> import numpy as np
    >>> # let's say your have statistical maps with the correct dimensions called 'data' for subject 'sub-xx'
    >>> subject = "sub-xx"
    >>> output_dir = "some_directory"
    >>> base_name = "sub-xx_ses-1"
    >>> data_v = pycortex.Vertex2D_fix(
    >>>     data,
    >>>     subject=subject,
    >>>     vmin1=3.1,
    >>>     vmax1=10,
    >>>     cmap="autumn")
    >>> #
    >>> # plop this object in SavePycortexViewer
    >>> pyc = pycortex.SavePycortexViews(
    >>>     {"zstats": data_v},
    >>>     subjects=subject,
    >>>     azimuth=180,            # these settings are focused around V1
    >>>     altitude=120,           # these settings are focused around V1
    >>>     radius=260,             # these settings are focused around V1
    >>>     save_cm=True,
    >>>     fig_dir=output_dir,
    >>>     base_name=base_name)

    >>> # to save "zstats"-object, we can run
    >>> pyc.save_all()

    >>> # to save the object as a static viewer (cortex.make_static()):
    >>> pyc.to_static() 
    """

    def __init__(
        self,
        data_dict: Union[dict, cortex.Vertex,cortex.VertexRGB,cortex.Vertex2D],
        subject: str=None,
        fig_dir: str=None,
        specularity: int=0,
        unfold: int=1,
        azimuth: int=180,
        altitude: int=105,
        radius: int=163,
        pivot: float=0,
        size: tuple=(2400,1200),
        data_name: str="occipital_inflated",
        base_name: str=None,
        rois_visible: int=0,
        rois_labels: int=0,
        sulci_visible: int=1,
        sulci_labels: int=0,
        save_cm: bool=False,
        cm_ext: str="pdf",
        viewer: bool=True,
        **kwargs):

        self.tmp_dict = data_dict
        self.subject = subject
        self.fig_dir = fig_dir
        self.altitude = altitude
        self.radius = radius
        self.pivot = pivot
        self.size = size
        self.azimuth = azimuth
        self.unfold = unfold
        self.specularity = specularity
        self.data_name = data_name
        self.base_name = base_name
        self.rois_visible = rois_visible
        self.rois_labels = rois_labels
        self.sulci_labels = sulci_labels
        self.sulci_visible = sulci_visible
        self.save_cm = save_cm
        self.cm_ext = cm_ext
        self.viewer = viewer

        if not isinstance(self.base_name, str):
            self.base_name = self.subject

        if not isinstance(self.fig_dir, str):
            self.fig_dir = os.getcwd()            

        if not isinstance(self.tmp_dict, dict):
            if isinstance(self.tmp_dict, np.ndarray):
                self.tmp_dict = {"data": cortex.Vertex(self.tmp_dict, subject=self.subject, **kwargs)}
            else:
                self.tmp_dict = {"data": self.tmp_dict}

        # check what kind of object they are; if Vertex2D_fix, make colormaps
        self.data_dict = {}
        for key,val in self.tmp_dict.items():
            if isinstance(val, Vertex2D_fix):
                self.data_dict[key] = val.get_result()
                
                if self.save_cm:
                    if key == "polar":
                        ticks = [-np.pi,0,np.pi]
                    else:
                        ticks = list(np.arange(val.vmin1,val.vmax1+(val.vmax1*0.2), (val.vmax1-val.vmin1)*0.2))
                    
                    ticks = [round(ii,2) for ii in ticks]
                    val.get_colormap(
                        ori="horizontal", 
                        label=key, 
                        flip_label=True, 
                        ticks=ticks,
                        save_as=opj(self.fig_dir, f"{base_name}_desc-{key}_cm.{self.cm_ext}"))
            else:
                self.data_dict[key] = val
        
        # set the views
        self.view = {
            self.data_name: {
                f'surface.{self.subject}.unfold': self.unfold, 
                f'surface.{self.subject}.pivot': self.pivot, 
                f'surface.{self.subject}.specularity': self.specularity,
                # 'camera.target':self.target,
                'camera.azimuth': self.azimuth,
                'camera.altitude': self.altitude, 
                'camera.radius': self.radius}}
                
        if self.subject == "fsaverage":
            for tt in ["rois","sulci"]:
                for tag in ["visible","labels"]:
                    lbl = getattr(self, f"{tt}_{tag}")
                    self.view[self.data_name][f'surface.{self.subject}.overlays.{tt}.{tag}'] = lbl

        self.view[self.data_name]['camera.Save image.Width'] = self.size[0]
        self.view[self.data_name]['camera.Save image.Height'] = self.size[1]

        if self.viewer:
            self.js_handle = cortex.webgl.show(self.data_dict)
            self.params_to_save = list(self.data_dict.keys())
            self.set_view()

    def to_static(self, *args, **kwargs):
        filename = f"{self.base_name}_desc-static"
        cortex.webgl.make_static(
            opj(self.fig_dir, filename),
            self.data_dict,
            *args,
            **kwargs)
        
    def save_all(self):

        for param_to_save in self.js_handle.dataviews.attrs.keys():
            print(f"saving {param_to_save}")
            self.save(
                param_to_save,
                self.base_name)

    def set_view(self):
        # set specified view
        time.sleep(10)
        for _, view_params in self.view.items():
            for param_name, param_value in view_params.items():
                time.sleep(1)
                self.js_handle.ui.set(param_name, param_value)   

    def save(
        self, 
        param_to_save,
        base_name):

        self.js_handle.setData([param_to_save])
        time.sleep(1)
        
        # Save images by iterating over the different views and surfaces
        filename = f"{base_name}_desc-{param_to_save}.png"
        output_path = os.path.join(self.fig_dir, filename)
            
        # Save image           
        self.js_handle.getImage(output_path, size=self.size)
    
        # the block below trims the edges of the image:
        # wait for image to be written
        while not os.path.exists(output_path):
            pass
        time.sleep(1)
        try:
            import subprocess
            subprocess.call(["convert", "-trim", output_path, output_path])
        except:
            pass    


class Vertex2D_fix():

    """Vertex2D_fix

    Wrapper for :func:`cortex.VertexRGB`, where we create a vertex object based on `data1` with `data2` acting as mask or opacity so that the curvature below is shown. If `data2` is not specified, a mask is created based on `data1` with a threshold of >0. This object is compatible with :class:`linescanning.pycortex.SavePycortexViews()`. Because you lose touch with colormaps with this method, we can create one using :class:`linescanning.plotting.LazyColorbar()`, which will take the `vmin1` and `vmax1` from your inputs.

    Parameters
    ----------
    data1: np.ndarray, pd.DataFrame, pd.Series
        Input data to be plotted
    data2: np.ndarray, pd.DataFrame, pd.Series, optional
        Acts as mask or opacity onto `data1`, by default None
    subject: str, optional
        Subject name as defined in the filestore, by default None
    cmap: str, optional
        Colormap for the object, by default "magma"
    vmin1: int, optional
        Min value for `data1`, by default 0
    vmax1: int, optional
        Max value for `data1`, by default 1
    vmin2: int, optional
        Min value for `data2`, by default 0
    vmax2: int, optional
        Max value for `data2`, by default 1
    roi_borders: np.ndarray, optional
        Array describing borders of ROIs, by default None
    curv_type: str, optional
        Type of curvature to be displayed, by default "hcp". Options are:
        - "hcp"; aesthetically the most pleasing; can be made darker or brighter with `fc`
        - "standard"; the more discrete light gray/dark gray contrast version of curvature
        - None; don't display curvature below, sometimes useful to just display masks as they are
    fc: float, optional
        Factor to scale the curvature level, by default -1.25

    Example
    ----------
    >>> from linescanning import pycortex
    >>> import numpy as np
    >>> # let's say your have statistical maps with the correct dimensions called 'data' for subject 'sub-xx'
    >>> subject = "sub-xx"
    >>> output_dir = "some_directory"
    >>> base_name = "sub-xx_ses-1"
    >>> data_v = pycortex.Vertex2D_fix(
    >>>     data,
    >>>     subject=subject,
    >>>     vmin1=3.1,
    >>>     vmax1=10,
    >>>     cmap="autumn")

    >>> # to get the actual vertex object
    >>> data_v.get_result()

    >>> to create a colormap of the object
    >>> data_v.get_colormap(label="zstats")
    """

    def __init__(
        self,
        data1: Union[np.ndarray, pd.DataFrame, pd.Series], 
        data2: Union[np.ndarray, pd.DataFrame, pd.Series]=None, 
        subject: str=None, 
        cmap: str="magma", 
        vmin1: int=0, 
        vmax1: int=1, 
        vmin2: int=0, 
        vmax2: int=1, 
        roi_borders: np.ndarray=None,
        sulci_labels: bool=False,
        sulci_visible: bool=False,
        curv_type: str="hcp",
        fc: float=-1.25):

        self.data1 = data1
        self.data2 = data2
        self.subject = subject
        self.cmap = cmap
        self.vmin1 = vmin1
        self.vmax1 = vmax1
        self.vmin2 = vmin2
        self.vmax2 = vmax2
        self.roi_borders = roi_borders
        self.curv_type = curv_type
        self.fc = fc

        #this provides a nice workaround for pycortex opacity issues, at the cost of interactivity    
        # Get curvature
        self.curv = cortex.db.get_surfinfo(self.subject)

        # Adjust curvature contrast / color. Alternately, you could work
        # with curv.data, maybe threshold it, and apply a color map. 
        
        # check if input is series or dataframe
        if isinstance(self.data1, (pd.DataFrame,pd.Series)):
            self.data1 = self.data1.values
            
        # set alpha if None is specified
        if not isinstance(self.data2, (np.ndarray,pd.Series)):
            alpha = np.zeros_like(self.data1)
            alpha[self.data1>0] = 1
            self.data2 = alpha.copy()
        else:
            if isinstance(self.data2, pd.Series):
                self.data2 = self.data2.values

        #standard
        if self.curv_type == "standard":
            self.curv.data *= (fc+0.1)
        elif self.curv_type == "hcp":
            self.curv.data *= (fc+0.1)
        else:
            # self.curv.data = np.sign(self.curv.data) * .25
            self.curv.data = np.ones_like(self.curv.data)
        
        self.curv = cortex.Vertex(
            self.curv.data, 
            self.subject, 
            vmin=-1,
            vmax=1,
            cmap='gray')
        
        self.norm2 = Normalize(self.vmin2, self.vmax2)   
        self.vx = cortex.Vertex(
            self.data1, 
            self.subject, 
            cmap=self.cmap, 
            vmin=self.vmin1, 
            vmax=self.vmax1)
        
        # Map to RGB
        self.vx_rgb = np.vstack([self.vx.raw.red.data, self.vx.raw.green.data, self.vx.raw.blue.data])
            
        # Pick an arbitrary region to mask out
        # (in your case you could use np.isnan on your data in similar fashion)
        self.alpha = np.clip(self.norm2(self.data2), 0, 1).astype(float)

        # Map to RGB
        self.vx_rgb[:,self.alpha>0] = self.vx_rgb[:,self.alpha>0] * self.alpha[self.alpha>0]
        
        self.curv_rgb = np.vstack([self.curv.raw.red.data, self.curv.raw.green.data, self.curv.raw.blue.data])
        
        # do this to avoid artifacts where curvature gets color of 0 valur of colormap
        self.curv_rgb[:,np.where((self.vx_rgb > 0))[-1]] = self.curv_rgb[:,np.where((self.vx_rgb > 0))[-1]] * (1-self.alpha)[np.where((self.vx_rgb > 0))[-1]]

        # Alpha mask
        self.display_data = self.curv_rgb + self.vx_rgb 

        # display_data = curv_rgb * (1-alpha) + vx_rgb * alpha
        if self.roi_borders is not None:
            self.display_data[:,self.roi_borders.astype(bool)] = 0#255-display_data[:,roi_borders.astype('bool')]#0#255
        
        # Create vertex RGB object out of R, G, B channels
        self.final_result = cortex.VertexRGB(*self.display_data, self.subject)
        
    def get_result(self):
        return self.final_result
    
    def get_curv(self):
        return self.curv
    
    def make_colormap(
        self, 
        label=None, 
        **kwargs):

        self.cm = plotting.LazyColorbar(
            cmap=self.cmap,
            txt=label,
            vmin=self.vmin1,
            vmax=self.vmax1,
            **kwargs)
        
    def get_colormap(self, *args, **kwargs):
        self.make_colormap(*args, **kwargs)
        return self.cm
