import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import utils
from preproc import create_ref_grid

class LinearTransformation(tf.keras.layers.Layer):
    def __init__(self, min_ref_grid=[-1.], max_ref_grid=[1.], interp_method="bilinear", padding_mode="zeros", padding_mode_value=0., **kwargs):
        self.min_ref_grid = tf.constant(min_ref_grid, dtype=tf.float32)
        self.max_ref_grid = tf.constant(max_ref_grid, dtype=tf.float32)
        self.interp_method = tf.constant(interp_method, dtype=tf.string)
        self.padding_mode = tf.constant(padding_mode, dtype=tf.string)
        self.padding_mode_value = tf.constant(padding_mode_value, dtype=tf.float32)
        super(self.__class__, self).__init__(**kwargs)

        # if trainable== True & self.interp_method == "nn":
        #     raise Exception("Cannot train with nearest-neighbor interpolator because it is not derivable!") 

    def build(self, input_shape):
        num_dims = input_shape[0].ndims - 2
        shape_grid = tf.shape(self.min_ref_grid)[0]

        def ref_grid():
            self.min_ref_grid = (-1) * tf.ones(num_dims, dtype=tf.float32)
            self.max_ref_grid = tf.ones(num_dims, dtype=tf.float32)
        tf.cond(num_dims != shape_grid, ref_grid, lambda *args: None)
        
        super(self.__class__, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        return {
            'min_ref_grid': self.min_ref_grid,
            'max_ref_grid': self.max_ref_grid,
            'interp_method': self.interp_method,
            'padding_mode': self.padding_mode,
            'padding_mode_value': self.padding_mode_value,}

    def call(self, inputs):
        img, transfos = inputs
        output = self._resample(img, transfos)
        return output

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
    
    def _interpolate(self, im, points, min_ref_grid, max_ref_grid, method="bilinear", padding_mode="zeros", padding_mode_value=0.):
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
        zero_i = tf.zeros([], dtype=tf.int32)
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
                idx_c = ibatch + tf.clip_by_value(ix_c, zero_i, width_i - 1) \
                        + num_row_major[0] * tf.clip_by_value(iy_c, zero_i, height_i - 1) \
                        + num_row_major[1] * tf.clip_by_value(iz_c, zero_i, depth_i - 1)
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
                
class ChannelwiseConv3D(tf.keras.layers.Layer):
    # just working for channel last
    def __init__(self
                 , filters=1
                 , kernel_size=(1, 1, 1)
                 , dilation_rate=(1, 1, 1)
                 , padding="VALID"
                 , strides=(1, 1, 1)
                 , kernel_initializer=tf.keras.initializers.glorot_uniform()
                 , activation="relu"
                 , trainable=True
                 , init_weights=None
                 , **kwargs):
        super(ChannelwiseConv3D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = tuple(kernel_size)
        self.dilation_rate = tuple(dilation_rate)
        self.padding = padding
        self.strides = tuple(strides)
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        self.trainable = trainable
        self.custom_weights = False
        self.init_weights = init_weights

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        # if the input length is > 5, then it has more than 1 feature map (number of kernel filters)
        # this would happen usually for deeper layer but not the input layer
        self.n_input_fmps = 1
        if len(input_shape) > 5:
            self.n_input_fmps = int(input_shape[-2])

        if self.init_weights is None:
            self.kernel = self.add_weight(name='kernel',
                                        shape=self.kernel_size + (self.n_input_fmps,) + (self.filters,),
                                        initializer=self.kernel_initializer,
                                        trainable=self.trainable)
            self.bias = self.add_weight(name='bias',
                                        shape=(self.filters,),
                                        initializer=self.kernel_initializer,
                                        trainable=self.trainable)
        else:
            self.kernel = self.init_weights[0]
            self.bias = self.init_weights[1]

        super(ChannelwiseConv3D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        outputs = []
        split_inputs = tf.split(x, x.shape[-1], axis=-1)
        for split_input in split_inputs:
            if len(x.shape) > 5:
                split_input = tf.squeeze(split_input, axis=-1)
            out = tf.nn.conv3d(split_input
                               , filters=self.kernel
                               , strides=(1,) + self.strides + (1,)
                               , padding=self.padding
                               , dilations=(1,) + self.dilation_rate + (1,))
            out = tf.nn.bias_add(out, self.bias)
            if self.activation == "relu":
                out = tf.nn.relu(out)
            elif self.activation == "softmax":
                out = tf.nn.softmax(out)
            outputs += [out]
        outputs = tf.stack(outputs, axis=-1)
        return outputs

    def get_config(self):
        return {"filters": self.filters
                , "kernel_size": self.kernel_size
                , "dilation_rate": self.dilation_rate
                , "padding": self.padding
                , "strides": self.strides
                , "kernel_initializer": self.kernel_initializer
                , "activation": self.activation
                , "trainable": self.trainable}

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:3] + (self.filters,) + (input_shape[-1],)
        return output_shape


class ChannelwiseMaxpool3D(tf.keras.layers.Layer):
    # just working for channel last
    def __init__(self
                 , pool_size=(1, 1, 1)
                 , padding="VALID"
                 , strides=None
                 , **kwargs):
        super(ChannelwiseMaxpool3D, self).__init__(**kwargs)
        self.pool_size = tuple(pool_size)
        self.padding = padding
        self.strides = strides
        if strides is None:
            self.strides = pool_size

    def build(self, input_shape):
        super(ChannelwiseMaxpool3D, self).build(input_shape)

    def call(self, x):
        outputs = []
        split_inputs = tf.split(x, x.shape[-1], axis=-1)
        for split_input in split_inputs:
            if len(x.shape) > 5:
                split_input = tf.squeeze(split_input, axis=-1)
            out = tf.nn.max_pool3d(split_input
                                   , ksize=self.pool_size
                                   , strides=self.strides
                                   , padding=self.padding)
            outputs += [out]
        outputs = tf.stack(outputs, axis=-1)
        return outputs

    def get_config(self):
        return {"pool_size": self.pool_size
                , "padding": self.padding
                , "strides": self.strides}

    def compute_output_shape(self, input_shape):
        vol_shape = (input_shape[:3] - self.pool_size)/self.strides + 1
        output_shape = (vol_shape,) + input_shape[3:]
        return output_shape

def encode_block_channelwise(x, filters, name, params_conv, params_layer):
    '''
    One encoding block contains:
    - 3x3 convolution on each voxel
    - max pooling for scale-space representation
    - (optional) batch normalization which will increase generalization and learning speed
    - (optional) dropout to force inactive neurons to learn
    
    to max pool over channel
    tf.reduce_max(input_tensor, reduction_indices=[-1], keep_dims=True)
    I think max pool3d is already doing that..
    https://stackoverflow.com/questions/36817868/tensorflow-how-to-pool-over-depth
    '''

    x = ChannelwiseConv3D(name=name + "_conv", filters=filters, **params_conv)(x)
    if params_layer["batch_norm"]:
        x = tf.keras.layers.BatchNormalization(name=name + "_bn")(x)
    if params_layer["dropout"] > 0:
        x = tf.keras.layers.Dropout(rate=params_layer["dropout"], name=name + "_dropout", seed=params_layer["seed"])(x)
    if params_conv["strides"] == (1, 1, 1):
        x = ChannelwiseMaxpool3D(
            pool_size=params_layer["pool_size"], padding=params_layer["padding"], name=name + "_maxpool")(x)

    return x

def encode_block(x, filters, name, params_conv, params_layer):
    '''
    One encoding block contains:
    - 3x3 convolution on each voxel
    - max pooling for scale-space representation
    - (optional) batch normalization which will increase generalization and learning speed
    - (optional) dropout to force inactive neurons to learn
    '''

    conv3d = tf.keras.layers.Conv3D(name=name + "_conv", filters=filters, **params_conv)
    maxpool3d = tf.keras.layers.MaxPool3D(pool_size=params_layer["pool_size"], padding=params_layer["padding"], name=name + "_maxpool")
    batch_norm = tf.keras.layers.BatchNormalization(name=name + "_bn")
    dropout = tf.keras.layers.Dropout(rate=params_layer["dropout"], name=name + "_dropout", seed=params_layer["seed"])

    x_target = x[0]
    x_source = x[1]

    x_target = conv3d(x_target)
    # x_target = tf.keras.layers.LeakyReLU(alpha=0.2)(x_target)
    x_source = conv3d(x_source)
    # x_source = tf.keras.layers.LeakyReLU(alpha=0.2)(x_source)
    if params_layer["batch_norm"]:
        x_target = batch_norm(x_target)
        x_source = batch_norm(x_source)
    if params_layer["dropout"] > 0:
        x_target = dropout(x_target)
        x_source = dropout(x_source)
    if params_conv["strides"] == (1, 1, 1):
        x_target = maxpool3d(x_target)
        x_source = maxpool3d(x_source)

    return [x_target, x_source]

def regression_block(x, n_units, name, params_dense, params_layer):
    '''
    One regression block contains:
    - dense layer
    - (optional) batch normalization which will increase generalization and learning speed
    - (optional) dropout to force inactive neurons to learn
    '''

    x = tf.keras.layers.Dense(name=name + "_dense", units=n_units, **params_dense)(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    if params_layer["batch_norm"]:
        x = tf.keras.layers.BatchNormalization(name=name + "_bn")(x)
    if params_layer["dropout"] > 0:
        x = tf.keras.layers.Dropout(rate=params_layer["dropout"], name=name + "_dropout", seed=params_layer["seed"])(x)

    return x

def gaussian_filtering(x, name):

    if tf.keras.backend.int_shape(x)[-1] == 2:
        x = ChannelwiseConv3D(name=name
                            , filters=1
                            , strides=(2, 2, 2)
                            , kernel_size=(3, 3, 3)
                            , weights=gaussian_kernel_3x3()
                            , trainable=False)(x)
    else:
        x = tf.keras.layers.Conv3D(name=name
                                    , filters=1
                                    , strides=(2, 2, 2)
                                    , kernel_size=(3, 3, 3)
                                    , weights=gaussian_kernel_3x3()
                                    , trainable=False)(x)

    return x

def gaussian_kernel_3x3():
    '''
    Gaussian kernel
    '''

    f = np.zeros((3, 3, 3, 1, 1))
    f[:, :, :, 0, 0] = (1/16)*np.array([[[1, 2, 1]
                                        , [2, 4, 2]
                                        , [1, 2, 1]]

                                        , [[2, 4, 2]
                                        , [4, 4, 4]
                                        , [2, 4, 2]]

                                        , [[1, 2, 1]
                                        , [2, 4, 2]
                                        , [1, 2, 1]]])
    return [f, np.zeros(1)]

def gaussian_kernel_5x5():
    '''
    Gaussian kernel
    '''

    f = np.zeros((5, 5, 5, 1, 1))
    # TO DO
    # f[:, :, :, 0, 0] = (1/256)*np.array([[[1, 4, 6, 4, 1]
    #                                     , [4, 16, 24, 16, 4]
    #                                     , [6, 24, 36, 24, 6]
    #                                     , [4, 16, 24, 16, 4]
    #                                     , [1, 4, 6, 4, 1]]

    #                                     , [[4, 16, 24, 16, 4]
    #                                     , [16, 24, 36, 24, 16]
    #                                     , [24, 36, 36, 36, 24]
    #                                     , [16, 24, 36, 24, 16]
    #                                     , [4, 16, 24, 16, 4]]

    #                                     , [[6, 24, 36, 24, 6]
    #                                     , [24, 24, 36, 24, 24]
    #                                     , [36, 36, 36, 36, 36]
    #                                     , [24, 24, 36, 24, 24]
    #                                     , [6, 24, 36, 24, 6]]

    #                                     , [[4, 16, 24, 16, 4]
    #                                     , [16, 24, 36, 24, 16]
    #                                     , [24, 36, 36, 36, 24]
    #                                     , [16, 24, 36, 24, 16]
    #                                     , [4, 16, 24, 16, 4]]

    #                                     , [[1, 4, 6, 4, 1]
    #                                     , [4, 16, 24, 16, 4]
    #                                     , [6, 24, 36, 24, 6]
    #                                     , [4, 16, 24, 16, 4]
    #                                     , [1, 4, 6, 4, 1]]])
    return [f, np.zeros(1)]


def laplacian_kernel_3x3():
    '''
    Laplacian kernel
    '''

    f = np.zeros((3, 3, 3, 1, 1))
    f[:, :, :, 0, 0] = np.array([[[0, 0, 0]
                                 , [0, -1, 0]
                                 , [0, 0, 0]]

                                 , [[0, -1, 0]
                                 , [-1, 6, -1]
                                 , [0, -1, 0]]

                                 , [[0, 0, 0]
                                 , [0, -1, 0]
                                 , [0, 0, 0]]])
    return [f, np.zeros(1)]

def rigid_concatenated(kernel_size=(3, 3, 3)
                        , pool_size=(2, 2, 2)
                        , dilation=(1, 1, 1)
                        , strides=(2, 2, 2)
                        , activation="relu"
                        , padding="VALID"
                        , batch_norm=True
                        , preproc_layers=0
                        , gaussian_layers=0
                        , dropout=0
                        , seed=0
                        , encode_rate=2
                        , regression_rate=2
                        , filters=4
                        , units=1024
                        , n_encode_layers=5
                        , n_regression_layers=5):
    '''''
    A basic encoder followed by regression.
    The input volume is the concatenation of the 2 volumes to register.
    '''''

    k_init = tf.keras.initializers.glorot_uniform(seed=seed)
    params_conv = dict(
        strides=strides, dilation_rate=dilation, kernel_size=kernel_size, kernel_initializer=k_init, activation=activation, padding=padding)
    params_dense = dict(kernel_initializer=k_init, activation=activation)
    params_layer = dict(pool_size=pool_size, padding=padding, batch_norm=batch_norm, dropout=dropout, seed=seed)

    inp = tf.keras.Input(shape=(220, 220, 220, 2), dtype="float32")

    if True:
        split_inputs = tf.split(inp, inp.shape[-1], axis=-1)
        inp_target = split_inputs[0]
        inp_source = split_inputs[1]

        # encoder part
        for i in range(gaussian_layers):
            inp_target = gaussian_filtering(inp_target, name="gaussian_filter_target_0")
            inp_source = gaussian_filtering(inp_source, name="gaussian_filter_source_0")
        
        features = encode_block([inp_target, inp_source], filters, "encode_%02d" % 0, params_conv, params_layer)
        for i in range(1, n_encode_layers):
            layer_filters = int(filters * encode_rate**i)
            features = encode_block(features, layer_filters, "encode_%02d" % i, params_conv, params_layer)
        features = tf.keras.layers.Concatenate()(features)
    else:
        # encoder part
        if gaussian_layers > 0:
            inp_gauss = gaussian_filtering(inp, name="gaussian_filter_%02d" % 0)
            for i in range(1, gaussian_layers):
                inp_gauss = gaussian_filtering(inp_gauss, "gaussian_filter_%02d" % i)

        features = encode_block_channelwise(inp_gauss if gaussian_layers > 0 else inp, filters, "encode_%02d" % 0, params_conv, params_layer)
        for i in range(1, n_encode_layers):
            layer_filters = int(filters * encode_rate**i)
            features = encode_block_channelwise(features, layer_filters, "encode_%02d" % i, params_conv, params_layer)

    regression = tf.keras.layers.Flatten()(features)

    # regression
    for i in range(n_regression_layers):
        layer_units = int(units // regression_rate**i)
        regression = regression_block(regression, layer_units, "regression_%02d" % i, params_dense, params_layer)
    init_weights=[np.zeros((layer_units, 7), dtype=np.float32), np.array([1, 0, 0, 0, 0, 0, 0])]
    output = tf.keras.layers.Dense(units=7, weights=init_weights, activation=None)(regression)

    return tf.keras.models.Model(inputs=[inp], outputs=[output])

def unsupervised_rigid_concatenated(kernel_size=(3, 3, 3)
                                    , pool_size=(2, 2, 2)
                                    , dilation=(1, 1, 1)
                                    , strides=(2, 2, 2)
                                    , activation="relu"
                                    , padding="VALID"
                                    , batch_norm=True
                                    , preproc_layers=0
                                    , gaussian_layers=0
                                    , dropout=0
                                    , seed=0
                                    , encode_rate=2
                                    , regression_rate=2
                                    , filters=4
                                    , units=1024
                                    , n_encode_layers=5
                                    , n_regression_layers=5):
    '''''
    Unsupervised model with an encoder followed by regression layer.
    The input volume is the concatenation of the 2 volumes to register.
    '''''

    k_init = tf.keras.initializers.glorot_uniform(seed=seed)
    # k_init = tf.keras.initializers.Zeros()
    params_conv = dict(
        strides=strides, dilation_rate=dilation, kernel_size=kernel_size, kernel_initializer=k_init, activation=activation, padding=padding)
    params_dense = dict(kernel_initializer=k_init, activation=activation)
    params_layer = dict(pool_size=pool_size, padding=padding, batch_norm=batch_norm, dropout=dropout, seed=seed)
    ref_grid = create_ref_grid()
    sz_ref = ref_grid.GetSize()
    min_ref_grid = ref_grid.GetOrigin()
    max_ref_grid = ref_grid.TransformIndexToPhysicalPoint(sz_ref)
    params_reg = dict(min_ref_grid=min_ref_grid, max_ref_grid=max_ref_grid, interp_method="bilinear", padding_mode="zeros")

    inp_target = tf.keras.Input(shape=(220, 220, 220, 1), dtype="float32")
    inp_source = tf.keras.Input(shape=(220, 220, 220, 1), dtype="float32")

    # encoder part
    # preprocessing layers
    inp_tgt = inp_target
    inp_src = inp_source
    for i in range(gaussian_layers):
        inp_tgt = gaussian_filtering(inp_tgt, name="gaussian_filter_target_%d" %i)
        inp_src = gaussian_filtering(inp_src, name="gaussian_filter_source_%d" %i)
    for i in range(preproc_layers):
        inp_tgt = tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel_size, name="preproc_conv_target_%d" %i)(inp_tgt)
        inp_src = tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel_size, name="preproc_conv_source_%d" %i)(inp_src)
    
    features = encode_block([inp_tgt, inp_src], filters, "encode_%02d" % 0, params_conv, params_layer)
    for i in range(1, n_encode_layers):
        layer_filters = int(filters * encode_rate**i)
        features = encode_block(features, layer_filters, "encode_%02d" % i, params_conv, params_layer)
    features = tf.keras.layers.Concatenate()(features)

    regression = tf.keras.layers.Flatten()(features)

    # regression
    for i in range(n_regression_layers):
        layer_units = int(units // regression_rate**i)
        regression = regression_block(regression, layer_units, "regression_%02d" % i, params_dense, params_layer)
    
    # #initial weights with identity quaternion
    init_weights_quaternion = [np.zeros((layer_units, 4), dtype=np.float32), np.array([1., 0., 0., 0.])]
    quaternion = tf.keras.layers.Dense(units=4, weights=init_weights_quaternion, activation=None)(regression)
    #init_weights_displacement = [np.zeros((layer_units, 3), dtype=np.float32), np.array([0., 0., 0.])]
    #displacement = tf.keras.layers.Dense(units=3, weights=init_weights_displacement, activation=None)(regression)
    displacement = tf.zeros((tf.shape(quaternion)[0], 3))
    transformation = tf.keras.layers.Concatenate(axis=-1)([quaternion, displacement])

    # spatial transformer layer
    output = LinearTransformation(**params_reg)([inp_source, transformation])

    return tf.keras.models.Model(inputs=[inp_target, inp_source], outputs=[output, transformation])
