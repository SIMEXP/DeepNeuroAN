import tensorflow as tf


class ChannelwiseConv3D(tf.keras.layers.Layer):
    # just working for channel last
    def __init__(self
                 , filters=1
                 , kernel_size=(1, 1, 1)
                 , dilation_rate=(1, 1, 1)
                 , padding='SAME'
                 , strides=(1, 1, 1)
                 , kernel_initializer='glorot_uniform'
                 , activation='relu'
                 , **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = (1,) + dilation_rate + (1,)
        self.padding = padding
        self.strides = (1,) + strides + (1,)
        self.kernel_initializer = kernel_initializer

        super(ChannelwiseConv3D, self).__init__(**kwargs)

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
            split_input = tf.squeeze(split_input, axis=-1)
            out = tf.nn.conv3d(split_input
                               , filter=self.kernel
                               , strides=self.strides
                               , padding=self.padding
                               , dilations=self.dilation_rate)
            out = tf.nn.bias_add(out, self.bias)
            outputs += [out]
        outputs = tf.stack(outputs, axis=-1)
        return outputs

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
        self.pool_size = pool_size
        self.padding = padding
        self.strides = strides
        if strides is None:
            self.strides = pool_size
        super(ChannelwiseMaxpool3D, self).__init__(**kwargs)

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

    def compute_output_shape(self, input_shape):
        vol_shape = (input_shape[:3] - self.pool_size)/self.strides + 1
        output_shape = (vol_shape,) + input_shape[3:]
        return output_shape


def rigid_metric(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred), axis=0)


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
    x = tf.keras.layers.Activation(params_conv["activation"])(x)
    if params_layer["batch_norm"]:
        x = tf.keras.layers.BatchNormalization(name=name + "_bn")(x)
    if params_layer["dropout"] > 0:
        x = tf.keras.layers.Dropout(rate=params_layer["dropout"], name=name + "_dropout", seed=params_layer["seed"])(x)
    x = ChannelwiseMaxpool3D(
        pool_size=params_layer["pool_size"], padding=params_layer["padding"], name=name + "_maxpool")(x)

    return x


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

def gaussian_filter(x):
    '''
    Gaussian filtering block
    '''

    f = tf.constant([[[1, 2, 1],
                      [2, 4, 2],
                      [1, 2, 1]],
                     [[2, 4, 2],
                      [4, 4, 4],
                      [2, 4, 2]],
                     [[1, 2, 1],
                      [2, 4, 2],
                      [1, 2, 1]]])
    f = tf.expand_dims(f, 3)
    f = tf.expand_dims(f, 4)

    x = ChannelwiseConv3D(name="gaussian_filter_0"
                          , filters=1
                          , strides=(3, 3, 3)
                          , kernel_size=(3, 3, 3)
                          , weights=[f, tf.zeros(1)]
                          , trainable=False)(x)

    return x


def rigid_concatenated(kernel_size=(3, 3, 3)
                       , pool_size=(2, 2, 2)
                       , activation="relu"
                       , padding="VALID"
                       , batch_norm=True
                       , dropout=0.1
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

    k_init = tf.glorot_uniform_initializer(seed=seed)
    params_conv = dict(kernel_size=kernel_size, kernel_initializer=k_init, activation=activation, padding=padding)
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

    inp = tf.keras.Input(shape=(220, 220, 220, 2), dtype=tf.float32)

    # preprocessing
    filtered_inp = gaussian_filter(inp)

    # encoder part
    for i in range(n_encode_layers):
        features = filtered_inp if i == 0 else features
        layer_filters = int(filters * growth_rate**i)
        features = encode_block_channelwise(features, layer_filters, "encode%02d" % i, params_conv, params_layer)
    regression = tf.keras.layers.Flatten()(features)

    # regression
    for i in range(n_regression_layers):
        layer_units = int(units // growth_rate**i)
        regression = regression_block(regression, layer_units, "regress%02d" % i, params_dense, params_layer)
    output = tf.keras.layers.Dense(
        units=7, kernel_initializer=params_dense["kernel_initializer"], activation=None)(regression)

    model = tf.keras.models.Model(inputs=[inp], outputs=[output])

    return model
