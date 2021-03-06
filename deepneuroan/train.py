import os
import argparse
import datetime
import platform
import random as rn
import numpy as np
import tensorflow as tf
from data_generator import DataGenerator
from utils import get_version
import models
import metrics

class Training:
    def __init__(self
                 , data_dir=None
                 , ckpt_dir=None
                 , model_path=None
                 , output_model_path=None
                 , model_name="rigid_concatenated"
                 , weights_dir=None
                 , seed=None
                 , epochs=50
                 , batch_size=8
                 , kernel_size=[3, 3, 3]
                 , pool_size=[2, 2, 2]
                 , dilation=[1, 1, 1]
                 , strides=[2, 2, 2]
                 , activation="relu"
                 , padding="VALID"
                 , no_batch_norm=False
                 , preproc_layers=0
                 , gaussian_layers=0
                 , motion_correction = False
                 , unsupervised = False
                 , dropout=0
                 , encode_rate=2
                 , regression_rate=2
                 , filters=4
                 , units=1024
                 , encode_layers=7
                 , regression_layers=4
                 , lr=1e-4
                 , gpu=-1
                 , ncpu=-1):
        self._model_path = model_path
        self._model_name = model_name
        self._weights_dir = weights_dir
        self._epochs = epochs
        self._kernel_size = tuple(kernel_size)
        self._pool_size = tuple(pool_size)
        self._dilation = tuple(dilation)
        self._strides = tuple(strides)
        self._batch_size = int(batch_size)
        self._activation = activation
        self._padding = padding
        self._batch_norm = not no_batch_norm
        self._preproc_layers = preproc_layers
        self._gaussian_layers = gaussian_layers
        self._use_template = not motion_correction
        self._unsupervised = unsupervised
        self._dropout = float(dropout)
        self._encode_rate = float(encode_rate)
        self._regression_rate = float(regression_rate)
        self._filters = int(filters)
        self._units = int(units)
        self._encode_layers = int(encode_layers)
        self._regression_layers = int(regression_layers)
        self._lr = lr
        self._gpu = gpu
        self._ncpu = ncpu

        self._data_dir = None
        self._ckpt_dir = None
        self._ckpt_path = None
        self._output_model_path = None
        self._list_files = None
        self._seed = None

        self._set_data_dir(data_dir)
        self._set_seed(seed)
        self._set_list_files()
        self._set_ckpt_dir(ckpt_dir)
        self._set_output_model_path(output_model_path)
        self._set_ncpu()

        self.train_gen = None
        self.valid_gen = None
        self.test_gen = None

    def __repr__(self):
        return str(__file__) \
               + "\n" + str(datetime.datetime.now()) \
               + "\n" + str(platform.platform()) \
               + "\nDeepNeuroAN - {}".format(get_version()) \
               + "\n" + "class Training()" \
               + "\n\t input data dir : %s" % self._data_dir \
               + "\n\t checkpoint dir : %s" % self._ckpt_dir \
               + "\n\t model name : %s" % self._model_name \
               + "\n\t weights dir : %s" % self._weights_dir \
               + "\n\t seed : %s" % self._seed \
               + "\n\t number of epochs : %s" % (self._epochs,) \
               + "\n\t batch size : %s" % self._batch_size \
               + "\n\t kernel size : %s" % (self._kernel_size,) \
               + "\n\t pool size : %s" % (self._pool_size,) \
               + "\n\t dilation rate : %s" % (self._dilation,) \
               + "\n\t strides : %s" % (self._strides,) \
               + "\n\t padding : %s" % self._padding \
               + "\n\t activation : %s" % self._activation \
               + "\n\t batch norm : %s" % self._batch_norm \
               + "\n\t preprocessing (convolution) layers : %s" % self._preproc_layers \
               + "\n\t preprocessing (gaussian) layers : %s" % self._gaussian_layers \
               + "\n\t motion correction : %s" % (not self._use_template) \
               + "\n\t unsupervised learning : %s" % (self._unsupervised) \
               + "\n\t dropout : %f" % self._dropout \
               + "\n\t encode rate : %f" % self._encode_rate \
               + "\n\t regression rate : %f" % self._regression_rate \
               + "\n\t filters : %d" % self._filters \
               + "\n\t units : %d" % self._units \
               + "\n\t number of encoding layer : %d" % self._encode_layers \
               + "\n\t number of regression layer : %d" % self._regression_layers \
               + "\n\t learning rate : %f" % self._lr \
               + "\n\t number of cpus : %d" % self._ncpu \
               + "\n\t gpu : %d" % self._gpu

    def _set_data_dir(self, data_dir=None):
        if data_dir is None:
            self._data_dir = os.getcwd()
        else:
            self._data_dir = data_dir

    def _set_ckpt_dir(self, ckpt_dir=None):
        if (ckpt_dir is None) & (self._data_dir is not None):
            self._ckpt_dir = os.path.join(self._data_dir, "../", "checkpoints")
        else:
            self._ckpt_dir = ckpt_dir
        self._ckpt_path = os.path.join(self._ckpt_dir, "%s" % self._model_name, "%s_cp-{epoch:04d}.ckpt" % self._model_name)

    def _set_ncpu(self):
        ncpu = self._ncpu
        if ncpu < 0:
            ncpu = os.cpu_count()
        elif ncpu == 0:
            ncpu = 1
        self._ncpu = ncpu

    def _set_output_model_path(self, output_model_path=None):
        if (output_model_path is None) & (self._data_dir is not None):
            self._output_model_path = os.path.join(
                self._data_dir, "../", "%s_{end_time:s}" % self._model_name)
        else:
            self._output_model_path = output_model_path

    def _set_seed(self, seed=None):
        if seed is not None:
            self._seed = int(seed)

    def _set_list_files(self):
        self._list_files = []
        list_files_tmp = set([])
        for root, _, files in os.walk(self._data_dir):
            for file in files:
                filepath = os.path.join(root, file).split('.')[0]
                if os.path.exists(filepath + ".txt"):
                    list_files_tmp.add(filepath)
        self._list_files = list(list_files_tmp)
        self._list_files.sort()

    def _build_model(self):
        if self._model_path is not None:
            if self._model_path.split(".")[-1] == "json":
                with open(self._model_path, "r") as json_file:
                    model = tf.keras.models.model_from_json(json_file.read(), custom_objects={'ChannelwiseConv3D': models.ChannelwiseConv3D})
            elif self._model_path.split(".")[-1] == "h5":
                model = tf.keras.models.load_model(self._model_path, custom_objects={'ChannelwiseConv3D': models.ChannelwiseConv3D})
            else:
                print("Warning: incompatible input model type (is not .json nor .h5)")
        else:
            params_model = dict(kernel_size=self._kernel_size
                            , pool_size=self._pool_size
                            , dilation=self._dilation
                            , strides=self._strides
                            , activation=self._activation
                            , padding=self._padding
                            , batch_norm=self._batch_norm
                            , preproc_layers=self._preproc_layers
                            , gaussian_layers=self._gaussian_layers
                            , dropout=self._dropout
                            , seed=self._seed
                            , encode_rate=self._encode_rate
                            , regression_rate=self._regression_rate
                            , filters=self._filters
                            , units=self._units
                            , n_encode_layers=self._encode_layers
                            , n_regression_layers=self._regression_layers)
            if self._unsupervised:
                model = models.unsupervised_rigid_concatenated(**params_model)
            else:
                model = models.rigid_concatenated(**params_model)
        return model

    def _load_weights(self, model):
        if self._weights_dir is not None:
            latest_checkpoint = tf.train.latest_checkpoint(self._weights_dir)
            model.load_weights(latest_checkpoint)
        return model

    def _build_data_generators(self):
        #TODO: we should use it under preproc
        template_filepath = None

        # if template is undefined, target is the same volume but not moved (motion correction)
        if self._use_template:
            template_filepath = os.path.join(self._data_dir, "template_on_grid")
        # if unsupervised, network output is not a transformation but target itself
        params_gen = dict(list_files=self._list_files
                          , template_file=template_filepath
                          , is_unsupervised=self._unsupervised
                          , batch_size=self._batch_size
                          , avail_cores=self._ncpu)
        self.train_gen = DataGenerator(partition="train", **params_gen)
        self.valid_gen = DataGenerator(partition="valid", **params_gen)
        self.test_gen = DataGenerator(partition="test", **params_gen)

    def create_callbacks(self):
        """callbacks to optimize lr, tensorboard and checkpoints"""
        model_ckpt = tf.keras.callbacks.ModelCheckpoint(
            self._ckpt_path, verbose=0, save_weights_only=True, save_freq="epoch")
        # reduce_lr_logs = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-10)
        tensorboard_dir = os.path.join(self._data_dir
                                       , "../"
                                       , "tensorboard_logs"
                                       , self._model_name
                                       , datetime.datetime.now().strftime("%Y/%m/%d/%H:%M:%S"))
        tensorboard_logs = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir
                                                          , update_freq="epoch"
                                                          , histogram_freq=1
                                                          , write_graph=False
                                                          , write_images=True)
        # train_dice_logs = DiceCallback(data_gen=self.train_gen, logs_dir=tensorboard_dir + "/train_diff")
        # valid_dice_logs = DiceCallback(data_gen=self.valid_gen, logs_dir=tensorboard_dir + "/validation_diff")
        q_callback = metrics.QuaternionCallback(data_gen=self.valid_gen, logs_dir=tensorboard_dir + "/valid_quaternion")
        return [model_ckpt, tensorboard_logs, q_callback]

    def add_custom_callbacks(self, calls, train_gen, valid_gen):
        """custom callbacks using metrics.py"""

    def run(self):
        print(self.__repr__())

        #configuration for cpu
        tf.config.threading.set_inter_op_parallelism_threads(self._ncpu)
        tf.config.threading.set_intra_op_parallelism_threads(self._ncpu)

        #configuration for gpu
        if self._gpu > -1:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpu)
            # physical_devices = tf.config.experimental.list_physical_devices('GPU')
            # tf.config.experimental.set_memory_growth(physical_devices[0], True)

        if self._seed is not None:
            os.environ['PYTHONHASHSEED'] = str(self._seed)
            rn.seed(self._seed)
            np.random.seed(self._seed)
            tf.random.set_seed(self._seed)

        # generator creation
        self._build_data_generators()

        # model building
        model = self._build_model()

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self._lr)
        #, loss=[metrics.NCC().loss  if self._unsupervised else metrics.quaternion_mse_loss, metrics.quaternion_penalty])
        , loss=[metrics.ncc  if self._unsupervised else metrics.quaternion_mse_loss])
        # , loss=[metrics.dice_loss if self._unsupervised else metrics.quaternion_mse_loss, metrics.quaternion_penalty])

        #by default, if weights_dir is given, the model use them
        model = self._load_weights(model)
        model.summary(positions=[.30, .65, .80, 1.])
        # model.summary(positions=[.30, .70, 1.])
        # tf.keras.utils.plot_model(model, show_shapes=True, to_file=os.path.join(self._data_dir, "../", "model.png")
        calls = self.create_callbacks()
        # if reduce lr callback
        # calls[-1].set_model(model)
        
        # ### Check grad
        # def train(model, x, target):
        #     with tf.GradientTape() as tape:
        #         loss_value = metrics.dice_loss(model(x)[0], target)
        #     return loss_value, tape.gradient(loss_value, model.trainable_variables)

        # # learning parameters (weight, offset, non-rigid bias) and optimizers
        # opt = tf.keras.optimizers.Adam(learning_rate=self._lr)

        # for i in range(self._epochs):
        #     x = self.train_gen.__getitem__(i)[0]
        #     y = self.train_gen.__getitem__(i)[1]
        #     loss_value, grads = train(model, x, y)
        #     if i % 1 == 0:
        #         print("Step: {} - loss: {}".format(i, loss_value.numpy()))
        #         print("\tParams: {}".format([np.mean(t.numpy()) for t in model.trainable_variables]))
        #         print("\tGrads: {}".format([np.mean(g) for g in grads]))
        #     opt.apply_gradients(zip(grads, model.trainable_variables))

        # training
        model.save_weights(self._ckpt_path.format(epoch=0))

        model.fit(x=self.train_gen
                  , epochs=self._epochs
                  , callbacks=calls
                  , validation_data=self.valid_gen
                  , verbose=1
                  , shuffle=False
                  , use_multiprocessing=False)

        #TODO: to optimize (via gradient descent) motion parameters after first alignment (from CNN)
        # https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/6dof_alignment.ipynb#scrollTo=vw9RMRPVz0YL

        # saving both model (with weights), and model architecture (.json)
        model.save(self._output_model_path.format(
            end_time=datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")) + ".h5")
        with open(self._output_model_path.format(
                end_time=datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")) + ".json", "w") as json:
            json.write(model.to_json())

        # test
        tf.print("Test")
        model.evaluate_generator(generator=self.test_gen, use_multiprocessing=False, verbose=1)
        tf.print("Done !")

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
        , description="DeepNeuroAN - {}\nDocumentation at https://github.com/SIMEXP/DeepNeuroAN".format(get_version()))

    parser.add_argument(
        "--activation"
        , required=False
        , help="Activation function for non-linearity, Default: \"relu\"",
    )

    parser.add_argument(
        "--batch_size"
        , type=int
        , required=False
        , help="Size of one batch to generate, Default: 8",
    )
    parser.add_argument(
        "--ckpt_dir"
        , required=False
        , help="Directory containing the tensorflow/keras checkpoints, "
               "Default: data_dir/derivatives/deepneuroan/training/checkpoints",
    )

    parser.add_argument(
        "-d"
        , "--data_dir"
        , required=False
        , help="Directory containing all fmri data, Default: current directory",
    )

    parser.add_argument(
        "--dilation"
        , nargs='+'
        , type=int
        , required=False
        , help="Dilation rate, Default: (1, 1, 1)",
    )

    parser.add_argument(
        "--dropout"
        , type=float
        , required=False
        , help="Neuron dropout rate, Default: 0",
    )

    parser.add_argument(
        "--encode_layers"
        , type=int
        , required=False
        , help="Number of encoding layers, Default: 6",
    )

    parser.add_argument(
        "--epochs"
        , type=int
        , required=False
        , help="Number of epochs for training, Default: 50",
    )

    parser.add_argument(
        "--filters"
        , type=int
        , required=False
        , help="Number of filters for the first encoding layer, Default: 4",
    )

    parser.add_argument(
        "--gpu"
        , type=int
        , required=False
        , help="Which gpu to use (sorted by PCI_BUS_ID), if -1 CPU will be used , Default: -1",
    )

    parser.add_argument(
        "--encode_rate"
        , type=float
        , required=False
        , help="Growth rate for encode layers, Default: 2",
    )

    parser.add_argument(
        "--regression_rate"
        , type=float
        , required=False
        , help="Decrease rate for regression layers, Default: 2",
    )

    parser.add_argument(
        "--kernel_size"
        , nargs='+'
        , type=int
        , required=False
        , help="Kernel size for 3D convolution, Default: (3, 3, 3)",
    )

    parser.add_argument(
        "--lr"
        , type=float
        , required=False
        , help="Learning rate, Default: 1e-4",
    )

    parser.add_argument(
        "--model_name"
        , required=False
        , help="Model name to use, Default: \"rigid_concatenated\"",
    )

    parser.add_argument(
        "--model_path"
        , required=False
        , help="Input path to keras model (*.json, *.h5)",
    )

    parser.add_argument(
        "--motion_correction"
        , required=False
        , action="store_true"
        , help="Use this for motion correction, when the source and target are the same volume, Default: template is the registration target",
    )

    parser.add_argument(
        "--ncpu"
        , type=int
        , required=False
        , help="Number of cpus for multiprocessing; -1: all, 0: disabling parallel loading; Default: -1",
    )

    parser.add_argument(
        "--no_batch_norm"
        , required=False
        , action="store_true"
        , help="To disable batch normalisation, Default: batch_norm enabled",
    )

    parser.add_argument(
        "--output_model_path"
        , required=False
        , help="Output path to keras model (.h5), Default: data_dir/derivatives/deepneuroan/model_name_end_date.h5",
    )

    parser.add_argument(
        "--padding"
        , required=False
        , help="Padding for conv and maxpool, Default: \"VALID\"",
    )

    parser.add_argument(
        "--pool_size"
        , nargs='+'
        , type=int
        , required=False
        , help="Pool size for 3D maxpool, Default: (2, 2, 2)",
    )

    parser.add_argument(
        "--preproc_layers"
        , type=int
        , required=False
        , help="Number of preprocessing layers (convolution filtering) before first layer, if 0 no preprocessing layers, Default: 0",
    )

    parser.add_argument(
        "--gaussian_layers"
        , type=int
        , required=False
        , help="Number of preprocessing layers (gaussian filtering) before first layer, if 0 no preprocessing layers, Default: 0",
    )

    parser.add_argument(
        "--regression_layers"
        , type=int
        , required=False
        , help="Number of regression layers, Default: 4",
    )

    parser.add_argument(
        "-s"
        , "--seed"
        , type=int
        , required=False
        , help="Random seed to use for data generation, Default: None",
    )

    parser.add_argument(
        "--strides"
        , nargs='+'
        , type=int
        , required=False
        , help="Strides for conv layers, Default: (2, 2, 2)",
    )

    parser.add_argument(
        "--units"
        , type=int
        , required=False
        , help="Number of units for the first regression layer (optimal if even), Default: 1024",
    )

    parser.add_argument(
        "--unsupervised"
        , required=False
        , action="store_true"
        , help="Unsupervised learning, Default: supervised learning",
    )

    parser.add_argument(
        "--weights_dir"
        , required=False
        , help="Checkpoint tensorflow dir for weights",
    )

    return parser


def main():
    args = get_parser().parse_args()
    print(args)
    # deletion of none attributes, to use default arg from class
    attributes = []
    for k in args.__dict__:
        if args.__dict__[k] is None:
            attributes += [k]
    for attribute in attributes:
        args.__delattr__(attribute)
    train = Training(**vars(args))
    train.run()


if __name__ == '__main__':
    main()
