import pandas as pd

def get_csv(csv_file, idx_col=0):
    """

get_csv

return the dataframe given an input csv-file.

    """

    df = pd.read_csv(csv_file, index_col=idx_col)

    return df


class VertexInfo:

    """This object reads a .csv file containing relevant information about the angles, vertex position, and normal vector.
    It is a WIP-replacement for get_composite. Beware, this is my first attempt at object oriented programming.."""

    def __init__(self, infofile=None, subject=None):
        self.infofile = infofile
        self.data = get_csv(self.infofile)
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

class pRFInfo(object):

    """
pRFInfo

This class reads in the .csv file containing information about both the position and pRF-properties of 
a selected vertex. The file will be created during call_pycortex(2) and is stored in the derivatves/
pycortex directory. This file contains: 'x','y','size','beta','baseline','r2','index','position' (= RAS
coordinate), 'normal' of a subject. Each of these individual items can be returned.

    """

    def __init__(self, infofile):
        self.infofile = infofile

        try:
            self.data = get_csv(infofile)
        except: 
            raise ValueError(f"Could not read {self.infofile}. Is it a csv-file..?")

    
    def get_prf_x(self):
        """return a dictionary of the property 'x' for left and right hemisphere"""
        return {'lh': self.data['x'][0],
                'rh': self.data['x'][1]}

    def get_prf_y(self):
        """return a dictionary of the property 'y' for left and right hemisphere"""
        return {'lh': self.data['y'][0],
                'rh': self.data['y'][1]}

    def get_prf_size(self):
        """return a dictionary of the property 'size' for left and right hemisphere"""
        return {'lh': self.data['size'][0],
                'rh': self.data['size'][1]}

    def get_prf_beta(self):
        """return a dictionary of the property 'beta' for left and right hemisphere"""
        return {'lh': self.data['beta'][0],
                'rh': self.data['beta'][1]}

    def get_prf_baseline(self):
        """return a dictionary of the property 'baseline' for left and right hemisphere"""
        return {'lh': self.data['baseline'][0],
                'rh': self.data['baseline'][1]}

    def get_prf_r2(self):
        """return a dictionary of the property 'r2' for left and right hemisphere"""
        return {'lh': self.data['r2'][0],
                'rh': self.data['r2'][1]}                                                                

    def get_prf_all(self):
        """return a dictionary of all properties for left and right hemisphere"""
        return {'lh': {'x': self.data['x'][0],
                       'y': self.data['y'][0],
                       'size': self.data['size'][0],
                       'beta': self.data['beta'][0],
                       'baseline': self.data['baseline'][0],
                       'r2': self.data['r2'][0]},
                'rh': {'x': self.data['x'][1],
                       'y': self.data['y'][1],
                       'size': self.data['size'][1],
                       'beta': self.data['beta'][1],
                       'baseline': self.data['baseline'][1],
                       'r2': self.data['r2'][1]}}

    def get_vertex(self):
        """return best vertices"""
        return {'lh': self.data['index'][0],
                'rh': self.data['index'][1]}

    def get_ctx_coordinate(self):
        """return coordinate in pycortex space"""

        from .utils import string2float

        return {'lh': string2float(self.data['position'][0]),
                'rh': string2float(self.data['position'][1])}

    def get_normal(self):
        """return normal vector"""

        from .utils import string2float

        return {'lh': string2float(self.data['normal'][0]),
                'rh': string2float(self.data['normal'][1])}