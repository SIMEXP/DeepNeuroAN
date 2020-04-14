import re
import numpy as np
import SimpleITK as sitk

def transform_volume(brain, ref_grid, interp=None, rigid=None, def_pix=None):
    """Transform a given a volume and resample it to a grid using rigid transformation [q0, q1, q2, q3, t0, t1, t2]"""
    if interp is None:
        interp = sitk.sitkLinear
    if rigid is None:
        rigid = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rigid = np.float64(rigid)

    rigid_sitk = sitk.VersorRigid3DTransform([rigid[1], rigid[2], rigid[3], rigid[0]])
    translation = sitk.TranslationTransform(3, tuple(rigid[4:]))
    rigid_sitk.SetTranslation(translation.GetOffset())
    if def_pix is None:
        def_pix = np.min(sitk.GetArrayFromImage(brain))
    brain_to_grid = sitk.Resample(brain, ref_grid, rigid_sitk, interp, float(def_pix), sitk.sitkFloat32)

    return brain_to_grid

def get_sitk_from_numpy(np_array, ref_grid):
    sitk_vol = sitk.GetImageFromArray(np_array)
    sitk_vol.SetOrigin(ref_grid.GetOrigin())
    sitk_vol.SetSpacing(ref_grid.GetSpacing())
    sitk_vol.SetDirection(ref_grid.GetDirection())

    return sitk_vol

def load_trf_file(path):
    """load a transformation file into a quaternion + translation (mm) numpy array"""
    q = None
    match_float = "[+-]?[0-9]*[.]?[0-9]+"
    to_match = "(" + match_float + ") \t "\
               + "(" + match_float + ") \t "\
               + "(" + match_float + ") \t "\
               + "(" + match_float + ") \t "\
               + "(" + match_float + ") \t "\
               + "(" + match_float + ") \t "\
               + "(" + match_float + ").*?"
    with open(path, 'r') as fst:
        for line in fst:
            if re.match(to_match, line):
                match = re.match(to_match, line)
                q = np.array([float(match.group(1))
                             , float(match.group(2))
                             , float(match.group(3))
                             , float(match.group(4))
                             , float(match.group(5))
                             , float(match.group(6))
                             , float(match.group(7))])
    return q