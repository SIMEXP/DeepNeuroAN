import deepneuroan.utils as utils
import os, sys
import time

import numpy as np
np.set_printoptions(precision=2)
import tensorflow as tf

from preproc import create_ref_grid
import deepneuroan.utils as utils
import SimpleITK as sitk
import matplotlib.pyplot as plt
print(tf.__version__)

class LinearTransformation():
    def __init__(self, min_ref_grid=[-1.], max_ref_grid=[1.], interp_method="nn", padding_mode="zeros", padding_mode_value=0., **kwargs):
        self.min_ref_grid = tf.constant(min_ref_grid, dtype=tf.float32)
        self.max_ref_grid = tf.constant(max_ref_grid, dtype=tf.float32)
        self.interp_method = tf.constant(interp_method, dtype=tf.string)
        self.padding_mode = tf.constant(padding_mode, dtype=tf.string)
        self.padding_mode_value = tf.constant(padding_mode_value, dtype=tf.float32)

    @tf.function
    def _resample(self, img, transfos):
        input_shape = tf.shape(img)
        ref_size = input_shape[1:-1]
        ref_size_xyz = tf.concat([ref_size[1::-1], ref_size[2:]], axis=0)

        input_transformed = self._transform_grid(ref_size_xyz, transfos=transfos, min_ref_grid=self.min_ref_grid, max_ref_grid=self.max_ref_grid)
        input_transformed = self._interpolate(im=img
                                            , points=input_transformed
                                            , min_ref_grid=self.min_ref_grid
                                            , max_ref_grid=self.max_ref_grid
                                            , method=self.interp_method
                                            , padding_mode=self.padding_mode
                                            , padding_mode_value=self.padding_mode_value)
        output = tf.reshape(input_transformed, shape=input_shape)

        return output

    def _transform_grid(self, ref_size_xyz, transfos, min_ref_grid, max_ref_grid):

        # constants
        num_batch = tf.shape(transfos)[0]
        num_elems = tf.reduce_prod(ref_size_xyz)
        thetas = utils.get_matrix_from_params(transfos, num_elems)

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
    
    def _interpolate(self, im, points, min_ref_grid, max_ref_grid, method="nn", padding_mode="zeros", padding_mode_value=0.):

        #constants
        num_batch = tf.shape(im)[0]
        vol_shape_xyz = tf.cast(tf.concat([tf.shape(im)[1:-1][1::-1], tf.shape(im)[1:-1][2:]], axis=0), dtype=tf.float32)
        width = vol_shape_xyz[0]
        height = vol_shape_xyz[1]
        depth = vol_shape_xyz[2]
        width_i = tf.cast(width, dtype=tf.int32)
        height_i = tf.cast(height, dtype=tf.int32)
        depth_i = tf.cast(depth, dtype=tf.int32)
        channels = tf.shape(im)[-1]
        num_row_major = tf.cast(tf.math.cumprod(vol_shape_xyz), dtype=tf.int32)
        shape_output = tf.stack([num_batch, num_row_major[-1] , 1])
        zero = tf.zeros([], dtype=tf.float32)
        ibatch = utils.repeat(num_row_major[-1] * tf.range(num_batch, dtype=tf.int32), num_row_major[-1])

        # scale positions to [0, width/height - 1]
        coeff_x = (width - 1.)/(max_ref_grid[0] - min_ref_grid[0])
        coeff_y = (height - 1.)/(max_ref_grid[1] - min_ref_grid[1])
        coeff_z = (depth - 1.)/(max_ref_grid[2] - min_ref_grid[2])
        ix = (coeff_x * points[:, 0, :]) - (coeff_x *  min_ref_grid[0])
        iy = (coeff_y * points[:, 1, :]) - (coeff_y *  min_ref_grid[1])
        iz = (coeff_z * points[:, 2, :]) - (coeff_z *  min_ref_grid[2])

        # zeros padding mode, for positions outside of refrence grid
        cond = tf.math.logical_or(tf.math.equal(padding_mode, tf.constant("zeros", dtype=tf.string))
                                  , tf.math.equal(padding_mode, tf.constant("value", dtype=tf.string)))
        def evaluate_valid(): return tf.expand_dims(tf.cast(tf.less_equal(ix, width - 1.) & tf.greater_equal(ix, zero)
                                             & tf.less_equal(iy, height - 1.) & tf.greater_equal(iy, zero)
                                             & tf.less_equal(iz, depth - 1.) & tf.greater_equal(iz, zero)
                                             , dtype=tf.float32), -1)
        def default(): return tf.ones([], dtype=tf.float32)
        valid = tf.cond(cond, evaluate_valid, default)

        # if we use bilinear interpolation, we calculate each area between corners and positions to get the weights for each input pixel
        def bilinear():
            output = tf.zeros(shape_output, dtype=tf.float32)
            
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
            return output
        # else if method is nearest neighbor, we get the nearest corner
        def nearest_neighbor():
            # get rounded indice corner based on the scaled positions
            ix_nn = tf.cast(tf.clip_by_value(tf.round(ix), zero, width - 1.), dtype=tf.int32)
            iy_nn = tf.cast(tf.clip_by_value(tf.round(iy), zero, height - 1.), dtype=tf.int32)
            iz_nn = tf.cast(tf.clip_by_value(tf.round(iz), zero, depth - 1.), dtype=tf.int32)

            # gather input pixel values from nn corner indexes
            idx_nn = ibatch + ix_nn + num_row_major[0] * iy_nn + num_row_major[1] * iz_nn
            
            output = tf.gather(tf.reshape(im, [-1, channels]), idx_nn)
            return output

        cond_bilinear = tf.math.equal(method, tf.constant("bilinear", dtype=tf.string))
        cond_nn = tf.math.equal(method, tf.constant("nn", dtype=tf.string))
        output = tf.case([(cond_bilinear, bilinear), (cond_nn, nearest_neighbor)], exclusive=True)
        
        # padding mode
        cond_border = tf.math.equal(padding_mode, tf.constant("border", dtype=tf.string))
        cond_zero = tf.math.equal(padding_mode, tf.constant("zeros", dtype=tf.string))
        cond_value = tf.math.equal(padding_mode, tf.constant("value", dtype=tf.string))
        def border_padding_mode(): return output
        def zero_padding_mode(): return output * valid
        def value_padding_mode(): return output * valid + padding_mode_value * (1. - valid)
        output = tf.case([(cond_border, border_padding_mode), (cond_zero, zero_padding_mode), (cond_value, value_padding_mode)], exclusive=True)

        return output

n_batch = 4
out_size = (220, 220, 220)
x = np.empty((n_batch, 220, 220, 220, 1), dtype=np.float64)
transfos = np.empty((n_batch, 7), dtype=np.float64)
data_dir="/home/ltetrel/Documents/work/DeepNeuroAN"
sitk_im = []

for i in range(n_batch):
    data_dir="/home/ltetrel/Documents/work/DeepNeuroAN"
    sitk_im += [sitk.ReadImage(data_dir + "/ses-vid001_task-video_run-01_bold_vol-0001_transfo-%06d.nii.gz" %(i+1), sitk.sitkFloat32)]
    x[i, :, :, :, 0] = sitk.GetArrayFromImage(sitk_im[i])

    transfos[i,] = utils.load_trf_file(data_dir + "/ses-vid001_task-video_run-01_bold_vol-0001_transfo-%06d.txt" %(i+1))
    
    # Inversing quaternions to compare volumes with base one
    q = sitk.VersorRigid3DTransform([transfos[i, 1], transfos[i, 2], transfos[i, 3], transfos[i, 0]])
    t = sitk.TranslationTransform(3, tuple(transfos[i, 4:]))
    q.SetTranslation(t.GetOffset())
    q = q.GetInverse().GetParameters()
    transfos[i, 1:4] = [-transfos[i, 1], -transfos[i, 2], -transfos[i, 3]]
    transfos[i, 4:] = q[3:]

x = tf.constant(x, dtype=tf.float32)
transfos = tf.constant(transfos, dtype=tf.float32)

ref_size = tf.shape(x)[1:-1]
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

tic = time.time()
LinearT = LinearTransformation(min_ref_grid=min_ref_grid, max_ref_grid=max_ref_grid, interp_method=interp_method, padding_mode=padding_mode)
output = LinearT._resample(x, transfos)

ElpsTime = time.time() - tic
print("*** Total %1.3f s ***"%(ElpsTime))

def save_array_to_sitk(data, name, data_dir):
    ref_grid = create_ref_grid()
    sitk_img = utils.get_sitk_from_numpy(data, ref_grid)
    sitk.WriteImage(sitk_img, os.path.join(data_dir, name + ".nii.gz"))

tic=[]
ElpsTime=[]

for vol in range(output.shape[0]):
    save_array_to_sitk(data=output[vol,], name="vol%02d" %(vol+1), data_dir=data_dir)

    tic += [time.time()]

    num_dims = x.shape.ndims - 2
    rigid_sitk = sitk.ScaleVersor3DTransform(np.ones(num_dims), np.array([*transfos[vol, 1:4], transfos[vol, 0]], dtype=np.float64).flatten())
    if tf.shape(transfos)[-1] > 7:
        rigid_sitk.SetScale(np.array(transfos[vol,7:10], dtype=np.float64).flatten())
    if tf.shape(transfos)[-1] > 4:
        rigid_sitk.SetTranslation(np.array(transfos[vol,4:7], dtype=np.float64).flatten())

    ref_grid = create_ref_grid()
    brain_to_grid = sitk.Resample(sitk_im[vol], ref_grid, rigid_sitk, sitk.sitkLinear, 0.0, sitk.sitkFloat32)

    ElpsTime += [time.time() - tic[vol]]

    sitk.WriteImage(brain_to_grid, os.path.join(data_dir, "vol_sitk%02d" %(vol+1) + ".nii.gz"))

print("*** Total %1.3f s ***"%(tf.reduce_sum(ElpsTime)))