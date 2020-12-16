import os
import codecs
import re
import numpy as np
import tensorflow as tf
import SimpleITK as sitk

def transform_volume(vol, ref_grid, interp=None, rigid=None, def_pix=None):
    """Resample the volume into a grid using a quaternion and rigid transformation [q0, q1, q2, q3, t0, t1, t2]"""
    if interp is None:
        interp = sitk.sitkLinear
    if rigid is None:
        rigid = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rigid = np.float64(rigid)

    rigid_sitk = sitk.VersorRigid3DTransform([rigid[1], rigid[2], rigid[3], rigid[0]])
    translation = sitk.TranslationTransform(3, tuple(rigid[4:]))
    rigid_sitk.SetTranslation(translation.GetOffset())
    if def_pix is None:
        def_pix = np.min(sitk.GetArrayFromImage(vol))
    vol_to_grid = sitk.Resample(vol, ref_grid, rigid_sitk, interp, float(def_pix), sitk.sitkFloat32)

    return vol_to_grid

def repeat(x, num_reps):
    num_reps = tf.cast(num_reps, dtype=tf.int32)
    x = tf.expand_dims(x, axis=1)
    return tf.tile(x, multiples=(1,num_reps))

def quat_to_3D_matrix(q):
    size_q = tf.shape(q[...,0])
    R = tf.stack([1 - 2.*(q[...,2]**2 + q[...,3]**2), 2*(q[...,1]*q[...,2] - q[...,0]*q[...,3]), 2*(q[...,0]*q[...,2] + q[...,1]*q[...,3]), tf.zeros(size_q),
                    2.*(q[...,1]*q[...,2] + q[...,0]*q[...,3]), 1 - 2.*(q[...,1]**2 + q[...,3]**2), 2.*(q[...,2]*q[...,3] - q[...,0]*q[...,1]), tf.zeros(size_q),
                    2.*(q[...,1]*q[...,3] - q[...,0]*q[...,2]), 2.*(q[...,0]*q[...,1] + q[...,2]*q[...,3]), 1 - 2.*(q[...,1]**2 + q[...,2]**2), tf.zeros(size_q),
                    tf.zeros(size_q), tf.zeros(size_q), tf.zeros(size_q), tf.ones(size_q)],axis=-1)

    return tf.reshape(R, (-1, 4, 4))

def get_matrix_from_params(transfos, num_elems):
    num_batch = tf.shape(transfos)[0]
    scaling = tf.shape(transfos)[-1] == 10
    trans = tf.shape(transfos)[-1] == 7

    #if the transformations is [q0, q1, q2, q3, tx, ty, tz, sx, sy, sz], then we apply scaling
    def apply_scaling(): return tf.linalg.diag(1./transfos[:, -3:])
    def default_scaling(): return tf.eye(num_rows=3, batch_shape=[num_batch])
    thetas = tf.cond(scaling, apply_scaling, default_scaling)

    #if the transformations is [q0, q1, q2, q3, tx, ty, tz], then we apply translation
    def apply_trans(): return tf.concat(axis=2, values=[thetas, transfos[:, 4:7, tf.newaxis]])
    def default_trans(): return tf.concat(axis=2, values=[thetas, tf.zeros((num_batch, 3, 1))])
    thetas = tf.cond(trans, apply_trans, default_trans)

    #transformations should always be at least [q0, q1, q2, q3]
    R = quat_to_3D_matrix(transfos[:, :4])
    thetas = tf.linalg.matmul(thetas, R)

    return thetas

def get_sitk_from_numpy(np_array, ref_grid):
    """Get a simpleITK image from a numpy array, given a reference grid"""
    sitk_vol = sitk.GetImageFromArray(np_array)
    sitk_vol.SetOrigin(ref_grid.GetOrigin())
    sitk_vol.SetSpacing(ref_grid.GetSpacing())
    sitk_vol.SetDirection(ref_grid.GetDirection())

    return sitk_vol

def load_trf_file(path):
    """Load a transformation file into a quaternion + translation (mm) numpy array"""
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

def read(relative_path):
    """Read the curent file.
    Parameters
    ----------
        relative_path : string, required
            relative path to the file to be read, from the directory of this file
    
    Returns
    -------
        string : content of the file at relative path
    """
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, relative_path), 'r') as fp:
        return fp.read()

def get_version():
    """Get the version of this software, as describe in the __init__.py file from the top module.
    
    Returns
    -------
        string : version of this software
    """
    relative_path = os.path.join("__init__.py")
    for line in read(relative_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")