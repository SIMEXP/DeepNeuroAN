import tensorflow as tf
import numpy as np
import SimpleITK as sitk

# https://stackoverflow.com/questions/43784921/how-to-display-custom-images-in-tensorboard-using-keras
# def make_image(tensor):
#     """
#     Convert an numpy representation image to Image protobuf.
#     Copied from https://github.com/lanpa/tensorboard-pytorch/
#     """
#     from PIL import Image
#     height, width, channel = tensor.shape
#     image = Image.fromarray(tensor)
#     import io
#     output = io.BytesIO()
#     image.save(output, format='PNG')
#     image_string = output.getvalue()
#     output.close()
#     return tf.summary.image(height=height,
#                          width=width,
#                          colorspace=channel,
#                          encoded_image_string=image_string)

# class TensorBoardImage(tf.keras.callbacks.Callback):
#     def __init__(self, tag):
#         super().__init__() 
#         self.tag = tag

#     def on_epoch_end(self, epoch, logs={}):
#         # Load image
#         img = data.astronaut()
#         # Do something to the image
#         img = (255 * skimage.util.random_noise(img)).astype('uint8')

#         image = make_image(img)
#         summary = tf.summary(value=[tf.summary.value(tag=self.tag, image=image)])
#         writer = tf.summary.FileWriter('./logs')
#         writer.add_summary(summary, epoch)
#         writer.close()
#         return
# tbi_callback = TensorBoardImage('Title in tensorboard')

# def generate_png(scan,nslices=20):
#     from nilearn import plotting
#     import os
#     nifti_filename = os.path.basename(scan)
#     nifti_filename_noext = os.path.splitext(os.path.splitext(nifti_filename)[0])[0]
#     workfolder = os.getcwd()
#     imagefile1 = f"{workfolder}/{nifti_filename_noext}.png"
#     plotting.plot_anat(scan, display_mode='ortho', annotate=True, output_file=imagefile1)
#     imagefile2 = f"{workfolder}/{nifti_filename_noext}_slices.png"
#     plotting.plot_anat(scan, display_mode='y', cut_coords=nslices, annotate=True, output_file=imagefile2)
#     return imagefile1

#https://chadrick-kwag.net/how-to-manually-write-to-tensorboard-from-tf-keras-callback-useful-trick-when-writing-a-handful-of-validation-metrics-at-once/
class DiceCallback(tf.keras.callbacks.Callback):
    def __init__(self, data_gen, logs_dir):
        self.logsdir = logs_dir
        self.writer = tf.summary.create_file_writer(self.logsdir)
        self.data_gen = data_gen
        self.step_number = 0

    def on_epoch_end(self, epoch, logs=None):
        errors = []
        for i in range(self.data_gen.__len__()):
            pred = self.model.predict(x=self.data_gen.__getitem__(i)[0], use_multiprocessing=False, verbose=0)
            truth = self.data_gen.__getitem__(i)[1]
            errors += [truth - pred]
        errors = np.reshape(errors, (self.data_gen.__len__()*self.data_gen.batch_size, self.data_gen.n_regressors))
        diff = np.mean(errors, axis=0)
        names = ["diff_q0", "diff_q1", "diff_q2", "diff_q3", "diff_x", "diff_y", "diff_z"]
        '''
        Add also image metric to check how well the volume is superposed:
        for image check nilearn: test_transfo.ipynb
        '''
        with self.writer.as_default():
            for i in range(self.data_gen.n_regressors):
                tf.summary.scalar(name=names[i], data=diff[i], step=epoch)
            self.writer.flush()

def quaternion_mse_loss(y_true, y_pred):

    lag_mult = 2
    penalty_norm = lag_mult * tf.abs(tf.linalg.norm(y_pred[:4]) - 1)
    # penalty_negative to add
    return tf.keras.losses.MeanSquaredError(y_true, y_pred) + penalty_norm

def dice_loss(y_true, y_pred):
  
  numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
  denominator = tf.reduce_sum(y_true + y_pred, axis=-1)

  return 1 - (numerator + 1) / (denominator + 1)

def rigid_metric(y_true, y_pred):

    return tf.reduce_mean(tf.abs(y_true - y_pred), axis=0)