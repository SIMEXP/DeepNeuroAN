import re
import tensorflow as tf
import numpy as np
import SimpleITK as sitk


def load_file(path):
    """load a transformation file into a quaternion + translation (mm) numpy array"""
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
                q = np.array([float(match.group(1))
                             , float(match.group(2))
                             , float(match.group(3))
                             , float(match.group(4))
                             , float(match.group(5))
                             , float(match.group(6))
                             , float(match.group(7))])
    return q


# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self
                 , list_files
                 , template_file
                 , partition
                 , batch_size=8
                 , dim=(220, 220, 220)
                 , n_channels=2
                 , n_regressors=7
                 , seed=None
                 , shuffle=True
                 , is_inference=False):
        self.dim = dim
        self.batch_size = batch_size
        self.list_files = list_files
        self.partition = partition
        self.template_file = template_file
        self.n_samples = len(self.list_files)
        self.indexes = np.arange(self.n_samples)
        self.n_channels = n_channels
        self.n_regressors = n_regressors
        self.seed = seed
        self.shuffle = shuffle
        self.is_inference = is_inference

        if not self.is_inference:
            self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.n_samples // self.batch_size

    def __getitem__(self, index):
        """Generate one batch of data"""
        list_files_batch = self.get_files_batch(index)
        # Generate data
        data = self.__data_generation(list_files_batch)

        return data

    def get_files_batch(self, index):
        # Generate indexes of the batch
        indexes = self.indexes_partition[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of files for this batch
        list_files_batch = [self.list_files[k] for k in indexes]

        return list_files_batch

    def _set_partition_idx(self):
        """partition the indexes into train/valid/test data"""
        if self.partition == "train":
            range_idx = (0, int(0.7 * self.n_samples))
        elif self.partition == "valid":
            range_idx = (int(0.7 * self.n_samples), int(0.85 * self.n_samples))
        elif self.partition == "test":
            range_idx = (int(0.85 * self.n_samples), self.n_samples)
        else:
            print("Error: partition %s is not valid" % self.partition)
        self.indexes_partition = self.indexes[range_idx[0]:range_idx[1]]

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)
        self._set_partition_idx()

    def normalize_img(self, img):
        """
        Normalize the image so that the mean value for each image
        is 0 and the standard deviation is 1.
        """
        return (img - np.mean(img)) / np.std(img)

    def __data_generation(self, list_files_batch):
        """Generates data containing batch_size samples"""
        # Initialization
        x = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float64)
        y = np.empty((self.batch_size, self.n_regressors), dtype=np.float64)

        # Generate data
        template = sitk.GetArrayFromImage(sitk.ReadImage(self.template_file + ".nii.gz", sitk.sitkFloat64))
        template = self.normalize_img(template)
        for i, file in enumerate(list_files_batch):
            # Store sample
            img = sitk.GetArrayFromImage(sitk.ReadImage(file + ".nii.gz", sitk.sitkFloat64))
            img = self.normalize_img(img)
            x[i, :, :, :, 0] = template
            x[i, :, :, :, 1] = img
            if self.is_inference is False:
                y[i, ] = load_file(file + ".txt")

        '''
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
        '''

        data = tuple([x, y])
        if self.is_inference:
            data = x

        return data
