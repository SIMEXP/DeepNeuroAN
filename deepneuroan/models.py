import tensorflow as tf
import numpy as np


class ChannelwiseConv3D(tf.keras.layers.Layer):
    # just working for channel last
    def __init__(self
                 , filters=1
                 , kernel_size=(1, 1, 1)
                 , dilation_rate=(1, 1, 1)
                 , padding="SAME"
                 , strides=(1, 1, 1)
                 , kernel_initializer=tf.keras.initializers.glorot_uniform()
                 , activation="relu"
                 , **kwargs):
        super(ChannelwiseConv3D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = tuple(kernel_size)
        self.dilation_rate = tuple(dilation_rate)
        self.padding = padding
        self.strides = tuple(strides)
        self.kernel_initializer = kernel_initializer
        self.activation = activation

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        # if the input length is > 5, then it has more than 1 feature map (number of kernel filters)
        # this would happen usually for deeper layer but not the input layer
        self.n_input_fmps = 1
        if len(input_shape) > 5:
            self.n_input_fmps = int(input_shape[-2])

        self.kernel = self.add_weight(name='kernel',
                                      shape=self.kernel_size + (self.n_input_fmps,) + (self.filters,),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.filters,),
                                    initializer=self.kernel_initializer,
                                    trainable=True)

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
                , "activation": self.activation}

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:3] + (self.filters,) + (input_shape[-1],)
        return output_shape


class ChannelwiseMaxpool3D(tf.keras.layers.Layer):
    # just working for channel last
    def __init__(self
                 , pool_size=(1, 1, 1)
                 , padding="SAME"
                 , strides=None
                 , **kwargs):
        super(ChannelwiseMaxpool3D, self).__init__(**kwargs)
        self.pool_size = tuple(pool_size)
        self.padding = padding
        self.strides = strides
        if strides is None:
            self.strides = pool_size

    def build(self, input_shape):
        super(ChannelwiseMaxpool3D, self).build(input_shape)  # Be sure to call this at the end

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
    maxpool3d = tf.keras.layers.MaxPool3D(
        pool_size=params_layer["pool_size"], padding=params_layer["padding"], name=name + "_maxpool")
    batch_norm = tf.keras.layers.BatchNormalization(name=name + "_bn")
    dropout = tf.keras.layers.Dropout(rate=params_layer["dropout"], name=name + "_dropout", seed=params_layer["seed"])

    x_target = x[0]
    x_source = x[1]

    x_target = conv3d(x_target)
    x_source = conv3d(x_source)
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
    if params_layer["batch_norm"]:
        x = tf.keras.layers.BatchNormalization(name=name + "_bn")(x)
    if params_layer["dropout"] > 0:
        x = tf.keras.layers.Dropout(rate=params_layer["dropout"], name=name + "_dropout", seed=params_layer["seed"])(x)

    return x

def gaussian_filtering(x, name):

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
                       , gauss_filt=False
                       , dropout=0
                       , seed=0
                       , growth_rate=2
                       , filters=4
                       , units=1024
                       , n_encode_layers=6
                       , n_regression_layers=4):
    '''''
    A basic encoder followed by regression.
    The input volume is the concatenation of the 2 volumes to register.
    '''''

    k_init = tf.keras.initializers.glorot_uniform(seed=seed)
    params_conv = dict(
        strides=strides, dilation_rate=dilation, kernel_size=kernel_size, kernel_initializer=k_init, activation=activation, padding=padding)
    params_dense = dict(kernel_initializer=k_init, activation=activation)
    params_layer = dict(pool_size=pool_size, padding=padding, batch_norm=batch_norm, dropout=dropout, seed=seed)

    '''''
    calculation of output size
    we want to reduce the input to have less trainable parameters for the regression.
    but not too much to avoid over compressed data
    
    Input 256**3 x2x4=128 Mo

    conv1 : 256**3 x2x4 x4 = 512 Mo
    -- Max pool 2x2x2 --
    conv2 : 128**3 x2x4 x8 = 128 Mo
    -- Max pool 2x2x2 --
    conv3 : 64x64x64x2x4 x16 = 32 Mo
    -- Max pool 2x2x2 --
    
    adding more to have less trainable parameters for dense blocks
    
    conv4 : 32x32x32x2x4 x32 = 8 Mo
    -- Max pool 2x2x2 --
    conv5 : 16x16x16x2x4 x 64 = 2 Mo
    -- Max pool 2x2x2 --
    conv5 : 8x8x8x2x4 x 128 = 0.5 Mo
    -- Max pool 2x2x2 --
    conv6 : 4x4x4x2x4 x 256 = 0.125 Mo
    -- Max pool 2x2x2 --
    output : 2x2x2x2 x 256 = 4096 params
    '''''

    inp = tf.keras.Input(shape=(220, 220, 220, 2), dtype="float32")
    split_inputs = tf.split(inp, inp.shape[-1], axis=-1)
    inp_target = split_inputs[0]
    inp_source = split_inputs[1]

    # encoder part
    if gauss_filt:
        inp_target = gaussian_filtering(inp_target, name="gaussian_filter_target_0")
        inp_source = gaussian_filtering(inp_source, name="gaussian_filter_source_0")
    
    features = encode_block([inp_target, inp_source], filters, "encode%02d" % 0, params_conv, params_layer)
    for i in range(1, n_encode_layers):
        layer_filters = int(filters * growth_rate**i)
        features = encode_block(features, layer_filters, "encode%02d" % i, params_conv, params_layer)
    features = tf.keras.layers.Concatenate()(features)
    regression = tf.keras.layers.Flatten()(features)

    # regression
    for i in range(n_regression_layers):
        layer_units = int(units // growth_rate**i)
        regression = regression_block(regression, layer_units, "regress%02d" % i, params_dense, params_layer)
    output = tf.keras.layers.Dense(
        units=7, kernel_initializer=params_dense["kernel_initializer"], activation=None)(regression)

    model = tf.keras.models.Model(inputs=[inp], outputs=[output])

    return model
