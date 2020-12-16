import io
import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from nibabel import Nifti1Image
from nilearn import image, plotting

class QuaternionCallback(tf.keras.callbacks.Callback):
    def __init__(self, data_gen, logs_dir):
        self.logsdir = logs_dir
        self.writer_quaternion = tf.summary.create_file_writer(self.logsdir)
        self.writer_registration = tf.summary.create_file_writer(self.logsdir)
        self.data_gen = data_gen
        self.step_number = 0

    def on_epoch_end(self, epoch, logs=None):
        data = self.data_gen.__getitem__(0)
        x = data[0]
        y = data[1]
        predict = self.model.predict(x, use_multiprocessing=False, verbose=0)
        output = predict[0]
        quaternions = predict[1]
        names = self.data_gen.get_files_batch(0)
        # write quaternion
        with self.writer_quaternion.as_default():
            for i in range(len(names)):
                for j in range(len(quaternions[i])):
                    name_q = "quaternion/" + names[i].split('/')[-1] + "_q{}".format(j)
                    tf.summary.scalar(name=name_q, data=quaternions[i][j], step=epoch)
        self.writer_quaternion.flush()
        # capture registration result every n epochs
        if (epoch%1 == 0):
            with self.writer_registration.as_default():
                for i in range(len(names)):
                    img = self.get_registration_result(fixed=y, moving_registered=output)
                    name_reg = "registration_results/" + names[i].split('/')[-1]
                    tf.summary.image(name_reg, img, step=epoch)
        self.writer_registration.flush()

    def get_registration_result(self, fixed, moving_registered):
        fig = plt.figure()
        background = Nifti1Image(tf.squeeze(fixed).numpy(), np.eye(4))
        edges = Nifti1Image(tf.squeeze(moving_registered).numpy(), np.eye(4))
        display = plotting.plot_anat(background, figure=fig)
        display.add_edges(edges)
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    # def get_volume(self, volume):
    #     fig = plt.figure()
    #     background = Nifti1Image(tf.squeeze(volume).numpy(), np.eye(4))
    #     display = plotting.plot_anat(background, figure=fig)
    #     """Converts the matplotlib plot specified by 'figure' to a PNG image and
    #     returns it. The supplied figure is closed and inaccessible after this call."""
    #     # Save the plot to a PNG in memory.
    #     buf = io.BytesIO()
    #     plt.savefig(buf, format='png')
    #     plt.close(fig)
    #     buf.seek(0)
    #     # Convert PNG buffer to TF image
    #     image = tf.image.decode_png(buf.getvalue(), channels=4)
    #     # Add the batch dimension
    #     image = tf.expand_dims(image, 0)
    #     return image

def quaternion_mse_loss(y_true, y_pred):
    lag_mult = 1
    
    #mse loss
    mse_loss = tf.math.reduce_mean(tf.math.squared_difference(y_pred, y_true))
    
    # penalty for non-unit quaternions
    penalty_norm = lag_mult * tf.math.reduce_sum(tf.math.abs(tf.linalg.norm(y_pred[:, :4], axis=1) - 1.))

    # penalty for negative quaternion
    penalty_negative = lag_mult * tf.math.reduce_sum( tf.cast(y_pred[:, :4]<0, dtype=tf.float32) )

    return mse_loss + penalty_norm + penalty_negative

def quaternion_penalty(_, y_pred, use_penalty_norm=True, use_penalty_negative=False, use_penalty_positive=False): 
    penalty_norm = 0.
    penalty_negative = 0.
    penalty_one = 0.
    L = 5000

    # penalty for non-unit quaternions
    if use_penalty_norm:
        penalty_norm = tf.reduce_sum(tf.exp(tf.math.abs(tf.linalg.norm(y_pred[:, :4], axis=1) - 1.)) -1.)
    # penalty for negative quaternion values
    if use_penalty_negative:
        negative_y = tf.cast(tf.less(y_pred[:, :4], 0.), dtype=tf.float32)
        penalty_negative = tf.reduce_sum( L/(tf.exp( y_pred[:, :4]*negative_y)) - L )
    # penalty for quaternion values superior than 1.
    if use_penalty_positive:
        greater_y = tf.cast(tf.greater(y_pred[:, :4], 1.), dtype=tf.float32)
        penalty_one = tf.reduce_sum( L*(tf.exp(y_pred[:, :4] * greater_y - 1. * greater_y)) - L )

    return penalty_norm + penalty_negative + penalty_one

def ncc(I, J):
    mean_I = tf.reduce_mean(I)
    mean_J = tf.reduce_mean(J)
    IJ_cross = tf.reduce_sum((I - mean_I)*(J - mean_J))
    IJ_norm = tf.sqrt(tf.reduce_sum(tf.pow(I - mean_I, 2))) * tf.sqrt(tf.reduce_sum(tf.pow(J - mean_J, 2)))
    ncc = IJ_cross / IJ_norm
    
    return (-1)*ncc

# https://github.com/voxelmorph/voxelmorph/blob/legacy/src/losses.py
class NCC():
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=None, eps=1e-5):
        self.win = win
        self.eps = eps


    def ncc(self, I, J):
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(I.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            self.win = [9] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        sum_filt = tf.ones([*self.win, 1, 1])
        strides = 1
        if ndims > 1:
            strides = [1] * (ndims + 2)
        padding = 'SAME'

        # compute local sums via convolution
        I_sum = conv_fn(I, sum_filt, strides, padding)
        J_sum = conv_fn(J, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var + self.eps)

        # return negative cc.
        return tf.reduce_mean(cc)

    def loss(self, I, J):
        return - self.ncc(I, J)

def dice_loss(y_true, y_pred):
  
    num = 2 * tf.reduce_sum(y_true * y_pred, -1)
    den = tf.maximum(tf.reduce_sum(y_true + y_pred, -1), 1e-6)

    return (tf.reduce_mean(num/den))

# def dice_loss(y_true, y_pred):
    
#     true_masked = tf.clip_by_value(1e9*y_true, 0, 1)
#     pred_masked = tf.clip_by_value(1e9*y_pred, 0, 1)

#     numerator = 2 * tf.reduce_sum(true_masked * pred_masked)
#     denominator = tf.reduce_sum(true_masked) + tf.reduce_sum(pred_masked)    
#     return 1 - (numerator + 1) / (denominator + 1)