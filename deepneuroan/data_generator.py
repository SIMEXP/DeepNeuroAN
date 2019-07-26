import re
import tensorflow as tf
import numpy as np
import SimpleITK as sitk


def load_file(path):
    'load a transformation file into a quaternion + translation (mm) numpy array'
    q = None
    match_float = "[+-]?[0-9]*[.]?[0-9]+"
    to_match = "(" + match_float + ") \t "\
               + "(" + match_float + ") \t "\
               + "(" + match_float + ") \t "\
               + "(" + match_float + ") \t "\
               + "(" + match_float + ") \t "\
               + "(" + match_float + ") \t "\
               + "(" + match_float + ").*?"
    with open(path, 'r') as fst:
        for line in fst:
            if re.match(to_match, line):
                match = re.match(to_match, line)
                q = np.array([ float(match.group(1))
                                 , float(match.group(2))
                                 , float(match.group(3))
                                 , float(match.group(4))
                                 , float(match.group(5))
                                 , float(match.group(6))
                                 , float(match.group(7))])
    return np.array(q)


# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self
                 , list_ids
                 , template_id
                 , batch_size=32
                 , dim=(256, 256, 256, 2)
                 , n_channels=1
                 , n_regressors=7
                 , seed=None
                 , shuffle=True
                 , is_inference=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_ids = list_ids
        self.template_id = template_id
        self.n_channels = n_channels
        self.n_regressors = n_regressors
        self.seed = seed
        self.shuffle = shuffle
        self.is_inference = is_inference
        if not self.is_inference:
            self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        data = self.__data_generation(list_ids_temp)

        return data

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_ids))
        # needs s static variable to automatically increase the seed so the shuffle is not the same
        if self.shuffle == True:
            if self.seed is not None:
                np.random.seed(self.seed)
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_regressors), dtype=int)

        # Generate data
        template = sitk.GetArrayFromImage(sitk.ReadImage(self.template_id + ".nii.gz", sitk.sitkFloat32))
        for i, ID in enumerate(list_ids_temp):
            # Store sample
            x[i, :, :, :, 0, ] = template
            x[i, :, :, :, 1, ] = sitk.GetArrayFromImage(sitk.ReadImage(ID + ".nii.gz", sitk.sitkFloat32))
            y[i, ] = load_file(ID + ".txt")
            if y[i, ] is None:
                self.is_inference = True

        '''''
        def z_normalize_img(self, img):
            """
            Normalize the image so that the mean value for each image
            is 0 and the standard deviation is 1.
            """
            for channel in range(img.shape[-1]):
                img_temp = img[..., channel]
                img_temp = (img_temp - np.mean(img_temp)) / np.std(img_temp)

                img[..., channel] = img_temp

            return img
        '''''

        data = tuple([x, y])
        if not self.is_inference:
            data = tuple([x])

        return data