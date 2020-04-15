import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import utils
from preproc import create_ref_grid

class LinearTransformation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        # self.trainable = False
        super(self.__class__, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        if len(input_shape) > 2:
            raise Exception("LinearRegistration must be called on a list of length 2. "
                            "First argument are the volumes, second are the quaternions transform.")

        super(self.__class__, self).build(input_shape)

    def call(self, inputs):
        assert isinstance(inputs, list)
        
        # map transform across batch
        return tf.map_fn(self._tf_single_transform, inputs, dtype=tf.float32)
            
    def compute_output_shape(self, input_shape):
        return (input_shape[0],)
    
    def _single_transform(self, src, trf):
        #removing channels dim to make array compatible with sitk registration
        src = np.squeeze(src, axis=-1)
        ref_grid = create_ref_grid()

        sitk_src = utils.get_sitk_from_numpy(src, ref_grid)
        sitk_tgt = utils.transform_volume(sitk_src, ref_grid, rigid=trf)

        #converting back with channel dim for tf compatibility
        return np.expand_dims(sitk.GetArrayFromImage(sitk_tgt), axis=-1)
    
    @tf.function
    def _tf_single_transform(self, inputs): 
        out = tf.numpy_function(self._single_transform, inp=[inputs[0], inputs[1]], Tout=tf.float32)
        out.set_shape(inputs[0].get_shape())
        return out

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

    inp = tf.keras.Input(shape=(220, 220, 220, 2), dtype="float32")

    if False:
        split_inputs = tf.split(inp, inp.shape[-1], axis=-1)
        inp_target = split_inputs[0]
        inp_source = split_inputs[1]

        # encoder part
        for i in range(preproc_layers):
            inp_target = gaussian_filtering(inp_target, name="gaussian_filter_target_0")
            inp_source = gaussian_filtering(inp_source, name="gaussian_filter_source_0")
        
        features = encode_block([inp_target, inp_source], filters, "encode_%02d" % 0, params_conv, params_layer)
        for i in range(1, n_encode_layers):
            layer_filters = int(filters * growth_rate**i)
            features = encode_block(features, layer_filters, "encode_%02d" % i, params_conv, params_layer)
        features = tf.keras.layers.Concatenate()(features)
    else:
        # encoder part
        if preproc_layers > 0:
            inp_gauss = gaussian_filtering(inp, name="gaussian_filter_%02d" % 0)
            for i in range(1, preproc_layers):
                inp_gauss = gaussian_filtering(inp_gauss, "gaussian_filter_%02d" % i)

        features = encode_block_channelwise(inp_gauss if preproc_layers > 0 else inp, filters, "encode_%02d" % 0, params_conv, params_layer)
        for i in range(1, n_encode_layers):
            layer_filters = int(filters * growth_rate**i)
            features = encode_block_channelwise(features, layer_filters, "encode_%02d" % i, params_conv, params_layer)

    regression = tf.keras.layers.Flatten()(features)

    # regression
    for i in range(n_regression_layers):
        layer_units = int(units // growth_rate**i)
        regression = regression_block(regression, layer_units, "regression_%02d" % i, params_dense, params_layer)
    output = tf.keras.layers.Dense(
        units=7, kernel_initializer=params_dense["kernel_initializer"], activation=None)(regression)

    return tf.keras.models.Model(inputs=[inp], outputs=[output])

def unsupervised_rigid_concatenated(kernel_size=(3, 3, 3)
                                    , pool_size=(2, 2, 2)
                                    , dilation=(1, 1, 1)
                                    , strides=(2, 2, 2)
                                    , activation="relu"
                                    , padding="VALID"
                                    , batch_norm=True
                                    , preproc_layers=0
                                    , dropout=0
                                    , seed=0
                                    , growth_rate=2
                                    , filters=4
                                    , units=1024
                                    , n_encode_layers=6
                                    , n_regression_layers=4):
    '''''
    Unsupervised model with an encoder followed by regression layer.
    The input volume is the concatenation of the 2 volumes to register.
    '''''

    k_init = tf.keras.initializers.glorot_uniform(seed=seed)
    params_conv = dict(
        strides=strides, dilation_rate=dilation, kernel_size=kernel_size, kernel_initializer=k_init, activation=activation, padding=padding)
    params_dense = dict(kernel_initializer=k_init, activation=activation)
    params_layer = dict(pool_size=pool_size, padding=padding, batch_norm=batch_norm, dropout=dropout, seed=seed)

    inp = tf.keras.Input(shape=(220, 220, 220, 2), dtype="float32")

    if False:
        split_inputs = tf.split(inp, inp.shape[-1], axis=-1)
        inp_target = split_inputs[0]
        inp_source = split_inputs[1]

        # encoder part
        for i in range(preproc_layers):
            inp_target = gaussian_filtering(inp_target, name="gaussian_filter_target_0")
            inp_source = gaussian_filtering(inp_source, name="gaussian_filter_source_0")
        
        features = encode_block([inp_target, inp_source], filters, "encode_%02d" % 0, params_conv, params_layer)
        for i in range(1, n_encode_layers):
            layer_filters = int(filters * growth_rate**i)
            features = encode_block(features, layer_filters, "encode_%02d" % i, params_conv, params_layer)
        features = tf.keras.layers.Concatenate()(features)
    else:
        # encoder part
        if preproc_layers > 0:
            inp_gauss = gaussian_filtering(inp, name="gaussian_filter_%02d" % 0)
            for i in range(1, preproc_layers):
                inp_gauss = gaussian_filtering(inp_gauss, "gaussian_filter_%02d" % i)

        features = encode_block_channelwise(inp_gauss if preproc_layers > 0 else inp, filters, "encode_%02d" % 0, params_conv, params_layer)
        for i in range(1, n_encode_layers):
            layer_filters = int(filters * growth_rate**i)
            features = encode_block_channelwise(features, layer_filters, "encode_%02d" % i, params_conv, params_layer)

    regression = tf.keras.layers.Flatten()(features)

    # regression
    for i in range(n_regression_layers):
        layer_units = int(units // growth_rate**i)
        regression = regression_block(regression, layer_units, "regression_%02d" % i, params_dense, params_layer)
    quaternion = tf.keras.layers.Dense(
        units=7, kernel_initializer=params_dense["kernel_initializer"], activation=None)(regression)

    # transform layer
    split_inputs = tf.split(inp, inp.shape[-1], axis=-1)
    inp_target = split_inputs[0]
    output = LinearTransformation()([inp_target, quaternion])

    return tf.keras.models.Model(inputs=[inp], outputs=[output])
