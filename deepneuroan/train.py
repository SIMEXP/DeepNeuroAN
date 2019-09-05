#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 11:55:12 2018

@author: ltetrel
"""
import os
import argparse
import tensorflow as tf
import preproc
import datetime
import platform
from data_generator import DataGenerator
from models import rigid_concatenated, rigid_metric
import numpy as np


class Training:
    def __init__(self
                 , data_dir=None
                 , ckpt_dir=None
                 , model_path=None
                 , output_model_path=None
                 , model_name="rigid_concatenated"
                 , weights_dir=None
                 , seed=0
                 , epochs=50
                 , kernel_size=[3, 3, 3]
                 , pool_size=[2, 2, 2]
                 , batch_size=8
                 , activation="relu"
                 , padding="SAME"
                 , no_batch_norm=False
                 , dropout=0.1
                 , growth_rate=2
                 , filters=4
                 , units=1024
                 , encode_layers=7
                 , regression_layers=4
                 , lr=0.01):
        self._model_path = model_path
        self._model_name = model_name
        self._weights_dir = weights_dir
        self._epochs = epochs
        self._kernel_size = tuple(kernel_size)
        self._pool_size = tuple(pool_size)
        self._batch_size = int(batch_size)
        self._activation = activation
        self._padding = padding
        self._batch_norm = not no_batch_norm
        self._dropout = float(dropout)
        self._growth_rate = float(growth_rate)
        self._filters = int(filters)
        self._units = int(units)
        self._encode_layers = int(encode_layers)
        self._regression_layers = int(regression_layers)
        self._lr = lr

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

    def __repr__(self):
        return str(__file__) \
               + "\n" + str(datetime.datetime.now()) \
               + "\n" + str(platform.platform()) \
               + "\n" + "class Training()" \
               + "\n\t input data dir : %s" % self._data_dir \
               + "\n\t checkpoint dir : %s" % self._ckpt_dir \
               + "\n\t model name : %s" % self._model_name \
               + "\n\t weights dir : %s" % self._weights_dir \
               + "\n\t seed : %s" % self._seed \
               + "\n\t number of epochs : %s" % (self._epochs,) \
               + "\n\t kernel size : %s" % (self._kernel_size,) \
               + "\n\t pool size : %s" % (self._pool_size,) \
               + "\n\t padding : %s" % self._padding \
               + "\n\t batch norm : %s" % self._batch_norm \
               + "\n\t dropout : %f" % self._dropout \
               + "\n\t growth rate : %d" % self._growth_rate \
               + "\n\t filters : %d" % self._filters \
               + "\n\t units : %d" % self._units \
               + "\n\t number of encoding layer : %d" % self._encode_layers \
               + "\n\t number of regression layer : %d" % self._regression_layers \
               + "\n\t learning rate : %f" % self._lr

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
        self._ckpt_path = os.path.join(self._ckpt_dir, "%s_cp-{epoch:04d}.ckpt" % self._model_name)

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

    def _build_model(self):
        if self._model_path is not None:
            if self._model_path.split(".")[-1] == "json":
                with open("model.json", "r") as json_file:
                    model = tf.keras.models.model_from_json(json_file.read())
            else:
                print("Warning: incompatible input model type (is not .json)")
        else:
            model = rigid_concatenated(kernel_size=self._kernel_size
                                       , pool_size=self._pool_size
                                       , activation=self._activation
                                       , padding=self._padding
                                       , batch_norm=self._batch_norm
                                       , dropout=self._dropout
                                       , seed=self._seed
                                       , growth_rate=self._growth_rate
                                       , filters=self._filters
                                       , units=self._units
                                       , n_encode_layers=self._encode_layers
                                       , n_regression_layers=self._regression_layers)
        return model

    def _set_weights(self, model):
        if self._weights_dir is not None:
            latest_checkpoint = tf.train.latest_checkpoint(self._weights_dir)
            model.load_weights(latest_checkpoint)
        return model

    def create_callbacks(self):
        """callbacks to optimize lr, tensorboard and checkpoints"""
        checkpoint_path = os.path.join(self._ckpt_dir, "cp_%s_{epoch:04d}.ckpt" % self._model_name)
        model_ckpt = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, verbose=0, save_weights_only=True, save_freq="epoch")
        reduce_lr_logs = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.0001)
        tensorboard_dir = os.path.join(self._data_dir
                                       , "../"
                                       , "tensorboard_logs"
                                       , datetime.datetime.now().strftime("%Y/%m/%d-%H:%M:%S"))
        tensorboard_logs = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir
                                                          , update_freq="batch"
                                                          , histogram_freq=1
                                                          , write_graph=False
                                                          , write_grads=True
                                                          , write_images=True)

        return [model_ckpt, reduce_lr_logs, tensorboard_logs]

    def run(self):
        print(self.__repr__())

        if self._seed is not None:
            np.random.seed(self._seed)
            tf.set_random_seed(self._seed)
            os.environ['PYTHONHASHSEED'] = str(self._seed)

        # generator creation
        ### again, we should use it under preproc
        template_filepath = os.path.join(self._data_dir, "template_on_grid")
        params_gen = dict(
            list_files=self._list_files, template_file=template_filepath, batch_size=self._batch_size, seed=self._seed)
        train_gen = DataGenerator(partition="train", **params_gen)
        valid_gen = DataGenerator(partition="valid", **params_gen)
        test_gen = DataGenerator(partition="test", **params_gen)

        # model building
        model = self._build_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self._lr, clipnorm=1)
                      , loss=tf.keras.losses.mean_squared_error
                      , metrics=["mae"])

        #by default, if weights_dir is given, the model use them
        model = self._set_weights(model)
        model.summary()
        calls = self.create_callbacks()
        calls[-1].set_model(model)

        # inference_gen = DataGenerator(partition="train", is_inference=False, **params_gen)
        # model.predict_generator(inference_gen, steps=1, use_multiprocessing=True, verbose=1)

        # training
        model.save_weights(self._ckpt_path.format(epoch=0))
        model.fit_generator(generator=train_gen
                            , epochs=self._epochs
                            , callbacks=calls
                            , validation_data=valid_gen
                            , verbose=1
                            , shuffle=False
                            , use_multiprocessing=False)

        # saving both model (with weights), and model architecture (.json)
        model.save(self._output_model_path.format(end_time=datetime.datetime.now()) + ".h5")
        with open(self._output_model_path.format(end_time=datetime.datetime.now()) + ".json", "w") as json:
            json.write(model.to_json())

        model.evaluate_generator(generator=test_gen, use_multiprocessing=False)

        # # Optimizer
        # opt = tf.keras.optimizers.SGD(lr=0.01)
        # leNet5.compile(optimizer=opt,
        #                loss='sparse_categorical_crossentropy',
        #                metrics=['accuracy'])

        # # Saving the model
        # leNet5.save("model.h5")

        # sess = tf.keras.backend.get_session()
        # tf.train.Saver().save(sess, "/notebooks/yu_gpu_cpu_profile/LeNetVisu/LeNet5")


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
        , description=""
        , epilog="""
            Documentation at https://github.com/SIMEXP/DeepNeuroAN
            """)

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
        "--dropout"
        , type=float
        , required=False
        , help="Neuron dropout rate, if <=0 no dropout is applied, Default: 0.1",
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
        "--growth_rate"
        , type=int
        , required=False
        , help="Growth/decrease rate for encoder/regression layers, Default: 2",
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
        , help="Learning rate, Default: 0.01",
    )

    parser.add_argument(
        "--model_name"
        , required=False
        , help="Model name to use, Default: \"rigid_concatenated\"",
    )

    parser.add_argument(
        "--model_path"
        , required=False
        , help="Input path to keras model (*.json)",
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
        , help="Padding for conv and maxpool, Default: \"valid\"",
    )

    parser.add_argument(
        "--pool_size"
        , nargs='+'
        , type=int
        , required=False
        , help="Pool size for 3D maxpool, Default: (2, 2, 2)",
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
        , help="Random seed to use for data generation, Default: 0",
    )

    parser.add_argument(
        "--units"
        , type=int
        , required=False
        , help="Number of units for the first regression layer (optimal if even), Default: 1024",
    )

    parser.add_argument(
        "--weights_dir"
        , required=False
        , help="Checkpoint tensorflow dir for weights",
    )

    return parser


def main():
    args = get_parser().parse_args()
    # deletion of none attributes
    attributes = []
    for k in args.__dict__:
        if args.__dict__[k] is None:
            attributes += [k]
    for attribute in attributes:
        args.__delattr__(attribute)
    train_gen = Training(**vars(args))
    train_gen.run()


if __name__ == '__main__':
    main()
