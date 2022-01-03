import os
import sys
import numpy as np
from linescanning import utils
import pandas as pd

def correct_angle(x, verbose=False, only_angles=True):

    """correct_angle

    This function converts the angles obtained with normal2angle to angles that we can use on the scanner. The scanner doesn't like angles >45 degrees. If inserted, it will flip  all kinds of parameters such as slice orientation and foldover.

    Parameters
    ----------
    x: float, numpy.ndarray
        generally this should be literally the output from normal2angle, a (3,) array containing the angles relative to each axis.
    verbose: bool 
        print messages during the process (default = False)
    only_angles: bool 
        if we are getting the angles for real, we need to decide what the angle with the z-axis means. We do this by returning an additional variable 'z_axis_represents_angle_around' so that we know in :func:`linescanning.utils.get_console_settings` where to place this angle. By default this is false, and it will only return converted angles. When doing the final conversion, the real one, turn this off (default = True).

    Returns
    ----------
    numpy.ndarray
        scanner representation of the input angles

    str
        if <only_angles> is set to False, it additionally returns an "X" or "Y", which specifies around which axis (X or Y) the angle with the z-axis is to be used
    """

    if isinstance(x, np.ndarray) or isinstance(x, list):

        scanner_angles = np.zeros((3,))

        # set boolean for flipping RL angle in case of sagittal slice
        flip = False

        # flag for true coronal slice (i.e., angle with Y is already set to zero in scanner.Scanner())
        true_cor = False
        true_sag = False
        
        # the angle with the z-axis can be interpreted in terms of angle AROUND Y (AP) AND AROUND X (RL)
        # If the slice is in the sweetspot of coronal slices, we can interpret the angle WITH the z-axis
        # as angle AROUND the Y (AP) axis. Otherwise we need to interpret it as the angle AROUND the X
        # axis (RL)
        z_axis_represents_angle_around = "Y"

        # for good coronal slices, there are basically for options:
        # 1) Large X | Large Y > vector = center to top-left        (use angle as is)
        # 2) Small X | small Y > vector = center to bottom-right    (use angle as is)
        # 3) Small X | Large Y > vector = center to top-right       (flip sign)
        # 4) Large X | Small Y > vector = center to bottom-left     (flip sign)

        #-------------------------------------------------------------------------------------------------------------------------------
        # deal with x-axis
        if 0 <= x[0] <= 45: # here we can decide on case 2 and 3 (small X's)
            # 1) angles living between 0 deg and 45 deg can freely do so, nothing to update > most likely coronal slice
            scanner_angles[0] = x[0]

            # case 3) Small X | Large Y >> flip angle
            if x[1] >= 90:
                # in this situation, we can have two lines, mirrored over the RL-axis. The only thing that separates them is the angle with 
                # the Y-axis. If this is a large angle, we should have a negative value for the X-angle. If this is a small angle, this means
                # we should have a positive value for the X-angle
                scanner_angles[0] = utils.reverse_sign(scanner_angles[0])

                if verbose:
                    print(f" Case 3 holds: Small X ({round(x[0],2)}) | Large Y ({round(x[1],2)})")
            else:
                # case 2) Small X | small Y >> use angle as is

                if verbose:
                    print(f" Case 2 holds: Small X ({round(x[0],2)}) | Small Y ({round(x[1],2)})")
                else:
                    pass

        elif 45 <= x[0] <= 90:
            # 2) these angles are a bit tricky. This means we're outside of the range for coronal slices, so we insert 
            #    the corrected value into the second position and set position 1 to 0

            # theoretically we could have a bang on coronal slice. In that case the y-axis angle 
            # has been set to zero and the angle with the x-axis represents the angle around the AP 
            # axis.
            if x[1] != 0:
                scanner_angles[0] = 999 # code for sag slice
                scanner_angles[1] = 90-x[0]
                
                if verbose:
                    print(" No coronal cases hold: we'll get a sagittal slice")
                    
                sag_slice = True
            else:
                # if the y-angle is already set to zero, it means we have a coronal slice. This means the only
                # angle we need it this angle. It will deal with the foldover in utils.get_console_settings
                scanner_angles[0] = x[0]
                scanner_angles[1] = 998 # code for fully 90 deg slice
                true_cor = True

                if verbose:
                    print(" We're dealing with a purely coronal slice. Only X-angle is required")

        elif 90 <= x[0] <= 180: # here we can decide on case 1 and 4 (large X's)
            # 3) such angles would mean we have a vector pointing in the opposite direction of case 1). We simply subtract
            #    it from 180 degrees
            scanner_angles[0] = 180-x[0]

            # case 1) Large X | Large Y >> use angle as is
            if x[1] >= 90:
                if verbose:
                    print(f" Case 1 holds: Large X ({round(x[0],2)}) | Large Y ({round(x[1],2)})")
                else:
                    pass
            else:
                # case 4) Large X | Small Y >> flip angle
                scanner_angles[0] = utils.reverse_sign(scanner_angles[0])

                if verbose:
                    print(f" Case 4 holds: Large X ({round(x[0],2)}) | Small Y ({round(x[1],2)})")

            # in case we have a sagittal slice, the RL-angle decides the direction of the vector. A center-topright
            # vector is created with a positive value. This is decided by the RL-angle: a large RL angle means we have a 
            # center-topleft vector, a small RL angle means we have a center-topright vector.
            flip = True



        # if the above resulted in a sagittal slice, we need to convert the angles relative to the AP-axis. We have the
        # angle with the RL-axis, scanner_angles[0], and the angle with the AP-axis then is 90-angle_RL. Because of the
        # way the scanner interprets this, a center-topleft vector is created with the -(90-angle_RL).
        if 45 <= scanner_angles[0] <= 90:
            # Convert angle with x-axis to be compatible with sagittal slice
            if x[1] != 0:
                scanner_angles[1] = 90-scanner_angles[0]
                scanner_angles[0] = 999 # code for sagittal

                # overwrite the angle around the axis that is represented by the angle with the Z-axis
                z_axis_represents_angle_around = "X"
                
                # decide on the direction of the sagittal slice, center-topleft or center-topright depending on what the 
                # initial angle with the x-axis was. Big angle means center-topleft, small angle means center-topright
                if flip == True:
                    if verbose:
                        print(f" X angle was large ({round(x[0],2)}), inverting {round(scanner_angles[1],2)}")
                        print(f" Z angle = angle around RL-axis")
                    scanner_angles[1] = utils.reverse_sign(scanner_angles[1])
                else:
                    if verbose:
                        print(f" X angle was small ({round(x[0],2)}), using {round(scanner_angles[1],2)} as is")
                    else:
                        pass

        else:
            if verbose:
                print(f" Z angle = angle around AP-axis")
            else:
                pass


        #-------------------------------------------------------------------------------------------------------------------------------
        # now, if we found a suitable angle for X, we can ignore Y because we only need one of the two to get our line
        # we could've obtained the Y-angle above if the X-angle was in the 45-90 range. In that case the first two positions
        # are already filled.
        #
        if x[0] == 0:
            # set straight to true sagittal slice
            scanner_angles[1] = x[1]
            true_sag = True
            scanner_angles[0] = 998 # code for full sagittal

            # overwrite the angle around the axis that is represented by the angle with the Z-axis
            z_axis_represents_angle_around = "X"
            if verbose:
                print(" We're dealing with a purely sagittal slice. Only Y-angle is required")
        elif scanner_angles[0] <= 45:
            scanner_angles[1] = 0
        elif scanner_angles[0] == 999:
            # we've already corrected this angle if we have a sagittal slice
            pass
        else:
            # deal with y-axis; same rules as for the x-axis apply
            # we did not manage to get a good angle for X, so we need to convert the angle of Y relative to the AP axis
            
            if 0 <= x[1] <= 45:
                # 1) angles living between 0 deg and 45 deg can freely do so, nothing to update > most likely coronal slice
                scanner_angles[1] = x[1]
            elif 45 <= x[1] <= 90:
                # 2) these angles are a bit tricky. 
                scanner_angles[1] = 90-x[1]
            elif 90 <= x[1] <= 180:
                # 3) such angles would mean we have a vector pointing in the opposite direction of case 1). We simply subtract
                #    it from 180 degrees
                scanner_angles[1] = 180-x[1]

        #-------------------------------------------------------------------------------------------------------------------------------
        # deal with z-axis > this is a special angle as it can reflect an angle around the Z-axis OR the Y-axis, depending on 
        # the slice orientation. If slice == coronal > z-axis = angle AROUND Y-axis. If slice == sagittal > z-axis is angle
        # around X-axis. Previously we've also flattened this angle to the YZ-plane, so it's now a planar angle.

        # The angle with the z-axis starts at 0, where it points in the superior direction [0,0,1]. It then starts to rotate
        # down towards the inferior axis [0,0,-1]. The start = 0 deg, the y-axis is the 90 deg mark, and inferior axis = 180.
        # The scanner interprets these angles from -45 (DOWN/UP lines) to 45 (UP/DOWN) degrees. Other angles will be wrapped.
        if true_cor == False and true_sag == False:
            # we need to do something with the z-axis
            if -45 <= x[2] <= 45:
                # 1) angles living between 0 deg and 45 deg can freely do so, nothing to update > most likely coronal slice
                scanner_angles[2] = x[2]
            elif 45 <= x[2] <= 90:
                
                # 2) these angles are a bit tricky. Here is means that the foldover direction is changed too
                scanner_angles[2] = x[2]-90
            elif 90 <= x[2] <= 180:
                # 3) such angles would mean we have a vector pointing in the opposite direction of case 1). We simply subtract
                #    it from 180 degrees
                scanner_angles[2] = 180-x[2]
                flip = True

            # check if we should have the opposite angle of the one we got.
            if 45 <= scanner_angles[2] <= 90:
                # this means we got the angle proximal to the vector and Z-axis. We need to opposite one
                scanner_angles[2] = 90-scanner_angles[2]

                if flip == True:
                    # depending on whether the initial angle was big, we need to invert the sign to be compatible
                    # with the scanner
                    scanner_angles[2] = utils.reverse_sign(scanner_angles[2])

        else:
            scanner_angles[2] = x[2]

        # return the result
        if only_angles == True:
            return scanner_angles
        else:
            return scanner_angles, z_axis_represents_angle_around


def normal2angle(normal, unit="deg", system="RAS", return_axis=['x','y','z']):

    """normal2angle

    Convert the normal vector to angles representing the angle with the x,y,z axis. This can be done by taking the arc cosine over the dot product of the normal vector and a vector representing the axis of interest. E.g., the vector for x would be [1,0,0], for y it would be [0,1,0], and for z it would be [0,0,1]. Using these vector representations of the axis we can calculate the angle between these vectors and the normal vector. This results in radians, so we convert it to degrees by multiplying it with 180/pi.
    
    Parameters
    ----------
        normal: numpy.ndarray, list
            array or list-like representation of the normal vector as per output of pycortex or FreeSurfer (they will return the same normals)
        unit: str
            unit of angles: "deg"rees or "rad"ians (default = "deg")
        system: str
            coordinate system used as reference for calculating the angles. A right-handed system is default (RAS)
            see: http://www.grahamwideman.com/gw/brain/orientation/orientterms.html. The scanner works in LPS, so we'd need to define the x/y-axis differently to get correct angles (default = "RAS").
        return_axis: list 
            List of axes to return the angles for. For some functions we only need the first two axes, which we can retrieve by specifying 'return_axes=['x', 'y']' (default = ['x','y','z']).

    Returns
    ----------
    list
        list-like representation of the angles with each axis, first being the x axis, second the y axis, and third the z-axis.

    Notes
    ----------
    Convert angles to sensible plane: https://www.youtube.com/watch?v=vVPwQgoSG2g: angles obtained with this method are not coplanar; they don't live in the same space. So an idea would be to decompose the normal vector into it's components so it lives in the XY-plane, and then calculate the angles.
    """

    vector = np.zeros((3))
    vector[:len(normal)] = normal

    # convert to a unit vector in case we received an array with 2 values
    vector = utils.convert2unit(vector)
    # print(f"Vector = {vector}")
    
    # Define empty 3x3 array to be filled with 1's or -1's depending on requested coordinate system
    COORD_SYS = np.eye(3)
    if system.upper() == "RAS":
        np.fill_diagonal(COORD_SYS, [1,1,1])
    elif system.upper() == "LPS":
        np.fill_diagonal(COORD_SYS, [-1,-1,1])
    else:
        raise NotImplementedError(f"You requested a(n) {system.upper()} system, but I can only deal with 'LPS' or 'RAS' for now")

    # this works if both vectors are normal vectors; otherwise this needs to be scaled by the dot-product of both vector
    # magnitudes
    angles = np.arccos(vector @ COORD_SYS)

    # convert to degree or radian
    if unit == "deg":    
        angles = angles*(180/np.pi)

    # we don't always need all axes to be returned, but we do so by default. 
    # Specify some options below.
    if return_axis == ['x','y','z']:
        return angles
    elif return_axis == ['x','y']:
        return angles[:2]
    elif return_axis == ['y','z']:
        return angles[:-2]        
    elif return_axis == ['x']:
        return angles[0]
    elif return_axis == ['y']:
        return angles[1]
    elif return_axis == ['z']:
        return angles[2]   
    else:
        raise NotImplementedError(f"Requested angles were '{return_axis}', this has not yet been implemented")     


def get_console_settings(angles, hemi, idx, z_axis_meaning="Y"):

    """get_console_settings

    Function that outputs what is to be inserted in the MR console. This function is the biggest source of misery during my PhD so far. Needs thorough investigation. The idea is pretty simple: we have a set of angles obtained from normal2angle, we have converted them to angles that the scanner can understand (i.e., angles <45 degrees), and now we need to derive which ones to use in order to place the line along the normal vector.

    Parameters
    ----------
        angles: np.ndarray
            literally the output from correct_angles, a (3,) numpy array with the 'corrected' angles
        hemi: str
            should be "L" or "R", is mainly for info reason. It's stored in the dataframe so we can use it to index
        idx: int
            this should be the integer representing the selected vertex. This is also only stored in the dataframe. No operations are executed on it
        z_axis: str
            this string specifies how to interpret the angle with the z-axis: as angle around the X (RL) or Y (AP) axis. This can be obtained by turning off <only_angles> in :func:`linescanning.utils.correct_angle`. By default it's set to 'Y', as that means we're dealing with a coronal slice; the most common one. Though we can also get sagittal slices, so make sure to do this dilligently.
        foldover: str
            foldover direction of the OVS bands. Generally this will be FH, but there are instances where that does not apply. It can be returned by `linescanning.utils.correct_angle(foldover=True)`

    Returns
    ----------
    pd.DataFrame
        a dataframe containing the information needed to place the line accordingly. It tells you the foldover direction, slice orientation, and angles
    """

    # Get index of smallest angle
    # min_angle = np.argmin(abs(angles))
    print(f"Dealing with hemisphere: {hemi}")
    foldover = "FH"
    # print(min_angle)
    angle_x,angle_y,angle_z = angles[0],angles[1],angles[2]
    # also see comments in 'correct_angle' for an explanation of what is happening, but in short:
    # - based on the angle between the normal vector and the x-axis [-1,0,0] we decided whether
    #   we have a coronal or sagittal slice. Is the angle between 0-45, then we have a coronal
    #   slice, if it's between 45-90 degrees, we have a sagittal slice and the first angle is 
    #   set to zero
    #
    # - here we will first check if the first angle is zero. As said, if that's the case we have
    #   a sagittal slice.
    #
    # - because the scanner flips the use of (counter)clockwise-ness, we need to flip the angles
    #   for the XY-angles. The Z-angles are interpreted correctedly (negative = UP, positive =
    #   DOWN)
    #
    # Decide on the slice
    if angle_x != 999:

        # coronal slice
        orientation = 'coronal'
        angle_fh = angle_x

        # check if we should do something with the foldover in a coronal slice. This happens when the angle with the
        # z-axis is lower than -45 or larger than 45. We should then flip the angle and change the foldover.
        if angle_fh <= -45:
            print(f" Angle around Y-axis = {angle_fh}; adding 90 deg & setting foldover to LR")
            angle_fh += 90
            foldover = "LR"
        elif angle_fh >= 45:
            print(f" Angle around Y-axis = {angle_fh}; substracting 90 deg & setting foldover to LR")
            angle_fh -= 90
            foldover = "LR"

        # if the angle with the z-axis was large, we need to invert the angle (means right-pointing vector)
        if angle_z >= 90:
            angle_fh = utils.reverse_sign(angle_fh)

    else:
        orientation = 'sagittal'
        angle_fh = angle_y
        # print(f"Angle FH = {round(angle_y,2)}")

        # check if we should do something with the foldover in a sagittal slice. This happens when the angle with the
        # z-axis is lower than -45 or larger than 45. We should then flip the angle and change the foldover.
        if angle_fh <= -45:
            print(f" Angle around X-axis = {angle_fh}; adding 90 deg & setting foldover to AP")
            angle_fh += 90
            foldover = "AP"
        elif angle_fh >= 45:
            print(f" Angle around X-axis = {angle_fh}; adding 90 deg & setting foldover to AP")
            # angle_fh -= 90
            foldover = "AP"

        # if the angle with the z-axis was large, we need to invert the angle (means down-pointing vector)
        if angle_z >= 90:
            angle_fh = utils.reverse_sign(angle_fh)

    # if we have a sagittal slice, the angle with the z-axis represents the angle around the x-axis
    # if we have a coronal slice, the angle with the z-axis represents the angle around the y-axis
    # you can see how this makes sense by using your phone and place is coronally, and sagittally, 
    # then rotate with the angle with the z-axis that invisibly points to your ceiling
    #
    # Sometimes we also give a big angle as z-axis to know what way the vector is pointing. In that
    # case the angle is not actually used for the console.
    if not 998 in angles:
        if z_axis_meaning.upper() == "Y":
            angle_ap = angle_z
            angle_lr = 0
        elif z_axis_meaning.upper() == "X":
            angle_ap = 0
            angle_lr = angle_z
        else:
            raise ValueError(f"Z-axis means an angle around the {z_axis_meaning.upper()} axis?? Needs to be 'X' or 'Y'")
    else:
        angle_lr = 0
        angle_ap = 0


    data = {"parameter": ["orientation", "foldover", "vertex", "LR_rot", "AP_rot", "FH_rot"],
            "value": [orientation, foldover, idx, angle_lr, angle_ap, angle_fh]}    

    df = pd.DataFrame(data)
    df['hemi'] = hemi

    return df


def rotate_normal(norm, xfm, system="RAS"):

    """rotate_normal

    Applies the rotation part of an affine matrix to the normal vectorself.

    Parameters
    ----------
    norm: numpy.ndarray
        (3,) or (4,) array; If (4,) array, the last value should be set to zero to avoid translations
    xfm: numpy.ndarray, str
        (4,4) affine numpy array or string pointing to the matrix-file, can also be 'identity', in which case np.eye(4) will be used. This is handy for planning the line in session 1/FreeSurfer space
    system: str
        use RAS (freesurfer) or LPS (ITK) coordinate system. This is important as we need to apply the matrix in the coordinate system that the vector is living in. e.g., RAS vector = RAS matrix (not ANTs' default), LPS vector = LPS matrix. If LPS, then :func:`linescanning.utils.get_matrixfromants` is used, otherwise the matrix is first converted to ras with `ConvertTransformFile` and then read in with `np.loadtxt`.

    Example
    ----------
    >>> rotate_normal(normal_vector, xfm, system="LPS")

    Notes
    ----------
    The results of LPS_vector @ LPS_matrix is the same as RAS_vector @ RAS_matrix
    """

    import numpy as np

    if isinstance(xfm, str):

        if system.upper() == "RAS":
            if xfm != "identity":
                xfm_tmp = xfm.split('.')[0]+"_ras.txt"
                os.system(f"ConvertTransformFile 3 {xfm} {xfm_tmp} --ras --hm")
                xfm = np.loadtxt(xfm_tmp)
            else:
                xfm = np.eye(4)
            # os.remove(xfm_tmp)
        else:
            xfm = utils.get_matrixfromants(xfm)

    # if len(norm) == 3:
    #     norm =  np.append(norm,[0])
    # elif len(norm) == 4:
    #     if norm[3] != 0:
    #         raise ValueError("The last value of array is not zero; this results in translations in the normal vector. Should be set to 0!")
    # else:
    #     raise ValueError(f"Odd number of elements in array.. Vector = {norm}")

    rot_norm = norm@xfm[:3,:3]

    return rot_norm[:3]

def create_line_pycortex(normal, hemi, idx, coord=None):
    """create the line_pycortex file"""

    angles = normal2angle(normal)

    # Write the rotations describing the orientation of the line in the first session anatomy to a text file
    try:
        data = {"parameter": ["vertex", "LR_rot", "AP_rot", "FH_rot", "normal", "coord"],
                "value": [int(round(idx,0)), angles[0], angles[1], angles[2], normal, coord]}
    except:
        data = {"parameter": ["vertex", "LR_rot", "AP_rot", "FH_rot", "normal"],
                "value": [int(round(idx,0)), angles[0], angles[1], angles[2], normal]}
    data['hemi'] = hemi

    df = pd.DataFrame(data)

    return df    
