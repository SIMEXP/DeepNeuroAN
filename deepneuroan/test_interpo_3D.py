import os, sys
import time
sys.path.append('/home/ltetrel/DeepNeuroAN/deepneuroan/')
sys.path.append('/home/ltetrel/DeepNeuroAN/')
sys.path.append('/home/ltetrel/Documents/work/DeepNeuroAN/deepneuroan/')
sys.path.append('/home/ltetrel/Documents/work/DeepNeuroAN/')

import numpy as np
np.set_printoptions(precision=2)
import tensorflow as tf

from preproc import create_ref_grid
import deepneuroan.utils as utils
import SimpleITK as sitk
import matplotlib.pyplot as plt

def mem_size(var):
    return (tf.size(var).numpy() * sys.getsizeof(var.dtype))/(1024**3)

def _repeat(x, num_reps):
    num_reps = tf.cast(num_reps, dtype=tf.int32)
    x = tf.expand_dims(x, axis=1)
    return tf.tile(x, multiples=(1,num_reps))

def _quat_to_3D_matrix(q):
    size_q = tf.shape(q[...,0])
    R = tf.stack([1 - 2.*(q[...,2]**2 + q[...,3]**2), 2*(q[...,1]*q[...,2] - q[...,0]*q[...,3]), 2*(q[...,0]*q[...,2] + q[...,1]*q[...,3]), tf.zeros(size_q),
                  2.*(q[...,1]*q[...,2] + q[...,0]*q[...,3]), 1 - 2.*(q[...,1]**2 + q[...,3]**2), 2.*(q[...,2]*q[...,3] - q[...,0]*q[...,1]), tf.zeros(size_q),
                  2.*(q[...,1]*q[...,3] - q[...,0]*q[...,2]), 2.*(q[...,0]*q[...,1] + q[...,2]*q[...,3]), 1 - 2.*(q[...,1]**2 + q[...,2]**2), tf.zeros(size_q),
                  tf.zeros(size_q), tf.zeros(size_q), tf.zeros(size_q), tf.ones(size_q)],axis=-1)

    return tf.reshape(R, (-1, 4, 4))

def _get_matrix_from_params(transfos, num_elems):
    num_batch = tf.shape(transfos)[0]

    #if the transformations has length > 7, then we apply scaling
    if tf.shape(transfos)[-1] > 7:
        thetas = tf.linalg.diag(1./transfos[:, -3:])
    else:
        thetas = tf.eye(num_rows=3, batch_shape=[num_batch])
        
    #if the transformations has length > 4, then we apply translation
    if tf.shape(transfos)[-1] > 4:  
        thetas = tf.concat(axis=2, values=[thetas, transfos[:, 4:7, tf.newaxis]])
    else:
        thetas = tf.concat(axis=2, values=[thetas, tf.zeros((num_batch, 3, 1))])
    
    R = _quat_to_3D_matrix(transfos[:, :4])
    thetas = tf.linalg.matmul(thetas, R)

    return thetas

# @tf.function
def _transform_grid(ref_size_xyz, transfos, min_ref_grid, max_ref_grid):

    # constants
    num_batch = tf.shape(transfos)[0]
    num_elems = tf.reduce_prod(ref_size_xyz)
    thetas = _get_matrix_from_params(transfos, num_elems)

    # grid creation from volume affine
    mz, my, mx = tf.meshgrid(tf.linspace(min_ref_grid[2], max_ref_grid[2], ref_size_xyz[2])
                            , tf.linspace(min_ref_grid[1], max_ref_grid[1], ref_size_xyz[1])
                            , tf.linspace(min_ref_grid[0], max_ref_grid[0], ref_size_xyz[0])
                            , indexing='ij')

    # preparing grid for quaternion rotation
    grid = tf.concat([tf.reshape(mx, (1, -1)), tf.reshape(my, (1, -1)), tf.reshape(mz, (1, -1))], axis=0)
    grid = tf.expand_dims(grid, axis=0)
    grid = tf.tile(grid, (num_batch, 1, 1))

    # preparing grid for augmented transformation
    grid = tf.concat([grid, tf.ones((num_batch, 1, num_elems))], axis=1)
    
    return tf.linalg.matmul(thetas, grid)

# @tf.function
def _interpolate(im, points, min_ref_grid, max_ref_grid, method="nn", padding_mode="zeros", padding_mode_value=0.):
    
    #constants
    num_batch = tf.shape(im)[0]
    vol_shape_xyz = tf.cast( tf.concat([tf.shape(im)[1:-1][1::-1], tf.shape(im)[1:-1][2:]], axis=0), dtype=tf.float32)
    width = vol_shape_xyz[0]
    height = vol_shape_xyz[1]
    depth = vol_shape_xyz[2]
    width_i = tf.cast(width, dtype=tf.int32)
    height_i = tf.cast(height, dtype=tf.int32)
    depth_i = tf.cast(depth, dtype=tf.int32)
    channels = tf.shape(im)[-1]
    num_row_major = tf.cast(tf.math.cumprod(vol_shape_xyz), dtype=tf.int32)
    zero = tf.zeros([], dtype=tf.float32)
    output = tf.zeros((num_batch, num_row_major[-1] , 1), dtype=tf.float32)
    valid = tf.ones_like(output)
    ibatch = _repeat(num_row_major[-1] * tf.range(num_batch, dtype=tf.int32), num_row_major[-1])

    # scale positions to [0, width/height - 1]
    coeff_x = (width - 1.)/(max_ref_grid[0] - min_ref_grid[0])
    coeff_y = (height - 1.)/(max_ref_grid[1] - min_ref_grid[1])
    coeff_z = (depth - 1.)/(max_ref_grid[2] - min_ref_grid[2])
    ix = (coeff_x * points[:, 0, :]) - (coeff_x *  min_ref_grid[0])
    iy = (coeff_y * points[:, 1, :]) - (coeff_y *  min_ref_grid[1])
    iz = (coeff_z * points[:, 2, :]) - (coeff_z *  min_ref_grid[2])

    # padding mode, for positions outside of refrence grid
    if (padding_mode == "zero") | (padding_mode == "value"):
        valid = tf.expand_dims(tf.cast(tf.less_equal(ix, width - 1.) & tf.greater_equal(ix, zero)
                                        & tf.less_equal(iy, height - 1.) & tf.greater_equal(iy, zero)
                                        & tf.less_equal(iz, depth - 1.) & tf.greater_equal(iz, zero)
                                        , dtype=tf.float32), -1)
    
    # if we use bilinear interpolation, we calculate each area between corners and positions to get the weights for each input pixel
    if method == "bilinear":
        # get north-west-top corner indexes based on the scaled positions
        ix_nwt = tf.clip_by_value(tf.floor(ix), zero, width - 1.)
        iy_nwt = tf.clip_by_value(tf.floor(iy), zero, height - 1.)
        iz_nwt = tf.clip_by_value(tf.floor(iz), zero, depth - 1.)
        ix_nwt_i = tf.cast(ix_nwt, dtype=tf.int32)
        iy_nwt_i = tf.cast(iy_nwt, dtype=tf.int32)
        iz_nwt_i = tf.cast(iz_nwt, dtype=tf.int32)       

        #gettings all offsets to create corners
        offset_corner = tf.constant([ [0., 0., 0.], [0., 0., 1.], [0., 1., 0.], [0., 1., 1.], [1., 0., 0.], [1., 0., 1.], [1., 1., 0.], [1., 1., 1.]], dtype=tf.float32)
        offset_corner_i =  tf.cast(offset_corner, dtype=tf.int32)

        for c in range(8):
            # getting all corner indexes from north-west-top corner
            ix_c = ix_nwt + offset_corner[-c - 1, 0]
            iy_c = iy_nwt + offset_corner[-c - 1, 1]
            iz_c = iz_nwt + offset_corner[-c - 1, 2]

            # area is computed using the opposite corner
            nc = tf.expand_dims(tf.abs((ix - ix_c) * (iy - iy_c) * (iz - iz_c)), -1)

            # current corner position
            ix_c = ix_nwt_i + offset_corner_i[c, 0]
            iy_c = iy_nwt_i + offset_corner_i[c, 1]
            iz_c = iz_nwt_i + offset_corner_i[c, 2]

            # gather input image values from corners idx, and calculate weighted pixel value
            idx_c = ibatch + tf.math.minimum( width_i - 1, ix_c) + num_row_major[0] * tf.math.minimum( height_i - 1, iy_c) + num_row_major[1] * tf.math.minimum( depth_i - 1, iz_c)
            Ic = tf.gather(tf.reshape(im, [-1, channels]), idx_c)

            output += nc * Ic


    # else if method is nearest neighbor, we get the nearest corner
    elif method == "nn":
        # get rounded indice corner based on the scaled positions
        ix_nn = tf.cast(tf.clip_by_value(tf.round(ix), zero, width - 1.), dtype=tf.int32)
        iy_nn = tf.cast(tf.clip_by_value(tf.round(iy), zero, height - 1.), dtype=tf.int32)
        iz_nn = tf.cast(tf.clip_by_value(tf.round(iz), zero, depth - 1.), dtype=tf.int32)

        # gather input pixel values from nn corner indexes
        idx_nn = ibatch + ix_nn + num_row_major[0] * iy_nn + num_row_major[1] * iz_nn

        output = tf.gather(tf.reshape(im, [-1, channels]), idx_nn)

    if padding_mode == "zero":
        output = output * valid
    elif padding_mode == "value":
        output = output * valid + padding_mode_value * (1. - valid)

    return output

n_batch = 4
input_size = (220, 220, 220)
#quaternions (4,) + translations (3,) + scales (3,)
transfos = tf.stack( [tf.zeros(shape=(7), dtype=tf.float32), tf.zeros(shape=(7), dtype=tf.float32), (-20)*tf.ones(shape=(7), dtype=tf.float32)] )[:n_batch]
transfos = tf.stack( [[1., 0., 0., 0., 0., 0., 0.], [0.7071, 0.7071, 0., 0., 0., 0., 0.], [0.7071, 0.7071, 0., 0., -20., -20., -20.]] )[:n_batch]
transfos = tf.reshape(tf.stack( [[0.7071, 0.7071, 0., 0., 20., 20., 20., 1., 1., 1.]*int(n_batch/2), [1., 0., 0., 0., 0., 0., 0., 1.5, 1.5, 1.5]*int(n_batch/2)] ), shape=(n_batch, 10))
transfos = tf.stack( [[-0.66,  0.15,  0.45,  0.59, -0.1 , -0.34, -1.4 ],
                      [-0.47, -0.58,  0.15, -0.64,  0.38,  0.97, -1.18],
                      [-0.3 , -0.59, -0.33, -0.68, -1.24, -0.21, -0.23],
                      [-0.34, -0.32,  0.12, -0.87, -2.99, 28.75, -9.91]])

data_dir="/home/ltetrel/Documents/work/DeepNeuroAN"
file = os.path.join(data_dir, "ses-vid001_task-video_run-01_bold_vol-0001")
sitk_im = sitk.ReadImage(file + ".nii.gz", sitk.sitkFloat32)
U = tf.expand_dims(sitk.GetArrayFromImage(sitk_im), axis=-1)
U = tf.stack( [U]*n_batch, axis=0)
out_size = (220, 220, 220)

ref_size = tf.shape(U)[1:-1]
ref_size_xyz = tf.concat([ref_size[1::-1], ref_size[2:]], axis=0)
# min[d] and max[d] correspond to cartesian coordinate d (d=0 is x, d=1 is y ..)
name='BatchSpatialTransformer3dAffine'
ref_grid = create_ref_grid()
sz_ref = ref_grid.GetSize()
min_ref_grid = ref_grid.GetOrigin()
max_ref_grid = ref_grid.TransformIndexToPhysicalPoint(sz_ref)

interp_method = tf.constant("nn", dtype=tf.string)
padding_mode = tf.constant("border", dtype=tf.string)
padding_mode_value = tf.constant(0., dtype=tf.float32)
min_ref_grid = tf.constant(min_ref_grid, dtype=tf.float32)
max_ref_grid = tf.constant(max_ref_grid, dtype=tf.float32)


#warm-up
input_transformed = _transform_grid(ref_size_xyz, transfos, min_ref_grid, max_ref_grid)
_interpolate(U + 1., input_transformed + 1., min_ref_grid, max_ref_grid, interp_method, padding_mode)

tic = time.time()
    
input_transformed = _transform_grid(ref_size_xyz, transfos, min_ref_grid, max_ref_grid)
input_transformed = _interpolate(U, input_transformed, min_ref_grid, max_ref_grid, interp_method, padding_mode)

output = tf.reshape(input_transformed, tf.stack([tf.shape(U)[0], *ref_size, tf.shape(U)[-1]]))

from vprof import runner
tmp = _transform_grid(ref_size_xyz, transfos, min_ref_grid, max_ref_grid)
runner.run(_interpolate, 'cmhp', args=(U, tmp, min_ref_grid, max_ref_grid, interp_method, padding_mode), host='localhost', port=8001)

ElpsTime = time.time() - tic
print("*** Total %1.3f s ***"%(ElpsTime))

def save_array_to_sitk(data, name, data_dir):
    ref_grid = create_ref_grid()
    sitk_img = utils.get_sitk_from_numpy(data, ref_grid)
    sitk.WriteImage(sitk_img, os.path.join(data_dir, name + ".nii.gz"))
    
    return sitk_img

tic=[]
ElpsTime=[]

for vol in range(output.shape[0]):
    sitk_img = save_array_to_sitk(data=output[vol,], name="vol%02d" %(vol+1), data_dir=data_dir)

    tic += [time.time()]

    num_dims = U.shape.ndims - 2
    rigid_sitk = sitk.ScaleVersor3DTransform(np.ones(num_dims), np.array([*transfos[vol, 1:4], transfos[vol, 0]], dtype=np.float64).flatten())
    if tf.shape(transfos)[-1] > 7:
        rigid_sitk.SetScale(np.array(transfos[vol,7:10], dtype=np.float64).flatten())
    if tf.shape(transfos)[-1] > 4:
        rigid_sitk.SetTranslation(np.array(transfos[vol,4:7], dtype=np.float64).flatten())

    ref_grid = create_ref_grid()
    brain_to_grid = sitk.Resample(sitk_im, ref_grid, rigid_sitk, sitk.sitkLinear, 0.0, sitk.sitkFloat32)

    ElpsTime += [time.time() - tic[vol]]

    sitk.WriteImage(brain_to_grid, os.path.join(data_dir, "vol_sitk%02d" %(vol+1) + ".nii.gz"))

print("*** Total %1.3f s ***"%(tf.reduce_sum(ElpsTime)))

# axis = 1
# plt.subplot(321)
# batch = 0
# plt.imshow(output[batch,:,:,axis,0], cmap="gray", vmin=0, vmax=255)
# plt.subplot(322)
# plt.imshow(U[batch,:,:,axis,0], cmap="gray", vmin=0, vmax=255)

# plt.subplot(323)
# batch = 1
# plt.imshow(output[batch,:,:,axis,0], cmap="gray", vmin=0, vmax=255)
# plt.subplot(324)
# plt.imshow(U[batch,:,:,axis,0], cmap="gray", vmin=0, vmax=255)

# plt.subplot(325)
# batch = 2
# plt.imshow(output[batch,:,:,axis,0], cmap="gray", vmin=0, vmax=255)
# plt.subplot(326)
# plt.imshow(U[batch,:,:,axis,0], cmap="gray", vmin=0, vmax=255)
# plt.show()

# plt.figure()
# plt.subplot(321)
# batch = 0
# plt.imshow(output[batch,:,:,-axis,0], cmap="gray", vmin=0, vmax=255)
# plt.subplot(322)
# plt.imshow(U[batch,:,:,-axis,0], cmap="gray", vmin=0, vmax=255)

# plt.subplot(323)
# batch = 1
# plt.imshow(output[batch,:,:,-axis,0], cmap="gray", vmin=0, vmax=255)
# plt.subplot(324)
# plt.imshow(U[batch,:,:,-axis,0], cmap="gray", vmin=0, vmax=255)

# plt.subplot(325)
# batch = 2
# plt.imshow(output[batch,:,:,-axis,0], cmap="gray", vmin=0, vmax=255)
# plt.subplot(326)
# plt.imshow(U[batch,:,:,-axis,0], cmap="gray", vmin=0, vmax=255)
# plt.show()