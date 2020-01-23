import re
import traceback
import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import multiprocessing as mp

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
                 , template_file=None
                 , partition="train"
                 , batch_size=8
                 , dim=(220, 220, 220)
                 , n_channels=2
                 , n_regressors=7
                 , shuffle=True
                 , is_inference=False
                 , avail_cores=-1):
        self.dim = dim
        self.batch_size = batch_size
        list_files.sort()
        self.list_files = list_files
        self.partition = partition
        self.template_file = template_file
        self.n_samples = len(self.list_files)
        self.indexes = np.arange(self.n_samples)
        self.n_channels = n_channels
        self.n_regressors = n_regressors
        self.shuffle = shuffle
        self.is_inference = is_inference
        self.template = None
        if self.template_file is not None:
            self.template = self.load_img(self.template_file)
        self.avail_cores = avail_cores

        if not self.is_inference:
            self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch, few samples will not be seen if n samples is not even with bsize"""
        return len(self.indexes_partition) // self.batch_size

    def __getitem__(self, index):
        """Generate one batch of data"""
        list_files_batch = self.get_files_batch(index)
        # Generate data
        if self.avail_cores > 1:
            data = self.__mp_data_generation(list_files_batch)
        else:
            data = self.__data_generation(list_files_batch)
        return data

    def create_shared_mem(self):
        # shared memory pointers
        s_data_x, self.data_x = self.create_shared_array(
            shape=(self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        self.s_mem = (s_data_x,)
        if not self.is_inference:
            s_data_y, self.data_y = self.create_shared_array(shape=(self.batch_size, self.n_regressors), dtype=np.float32)
            self.s_mem = (s_data_x, s_data_y)

    def get_files_batch(self, index):
        # Generate indexes of the batch
        indexes = self.indexes_partition[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of files for this batch
        list_files_batch = [self.list_files[k] for k in indexes]
        return list_files_batch

    def get_target_files_batch(self, index):
        # Generate indexes of the batch
        indexes = self.indexes_partition[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of files for this batch
        if self.template_file is not None:
            list_target_files_batch = [self.template_file for k in indexes]
        else:
            list_target_files_batch = [re.match("(.*?_vol-[+-]?[0-9]*[.]?[0-9]+).*?", self.list_files[k]).group(1) for k in indexes]
        return list_target_files_batch

    def _set_indexes_partition(self):
        """partition the indexes into train/valid/test data"""
        if self.partition == "train":
            range_idx = (0, int(0.7 * self.n_samples))
        elif self.partition == "valid":
            range_idx = (int(0.7 * self.n_samples), int(0.85 * self.n_samples))
        elif self.partition == "test":
            range_idx = (int(0.85 * self.n_samples), self.n_samples)
        elif self.partition == "all":
            range_idx = (0, self.n_samples)
        else:
            print("Error: partition %s is not valid" % self.partition)
        return self.indexes[range_idx[0]:range_idx[1]]

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)
        self.indexes_partition = self._set_indexes_partition()

    def normalize_img(self, img):
        """
        Normalize the image so that the mean value for each image
        is 0 and the standard deviation is 1.
        """
        return (img - np.mean(img)) / np.std(img)

    def shared_to_numpy(self, shared_arr, shape, dtype):
        """Get a NumPy array from a shared memory buffer, with a given dtype and shape.
        No copy is involved, the array reflects the underlying shared buffer."""
        sz = int(np.product(shape))
        dtype = np.dtype(dtype)
        return np.frombuffer(shared_arr, dtype=dtype, count=sz).reshape(shape)

    def create_shared_array(self, shape, dtype):
        """Create a new shared array. Return a tuple of (shared array pointer, shape, npdtype),
        and a NumPy array view to it. Note that the buffer values are not initialized.
        """
        # Get a ctype type from the NumPy dtype.
        cdtype = np.ctypeslib.as_ctypes_type(dtype)
        # Create the RawArray instance.
        shared_arr = mp.RawArray(cdtype, int(np.prod(shape)))
        # Get a NumPy array view.
        arr = self.shared_to_numpy(shared_arr, shape, dtype)
        return (shared_arr, shape, dtype,), arr

    def load_img(self, file):
        img = sitk.GetArrayFromImage(sitk.ReadImage(file + ".nii.gz", sitk.sitkFloat32))
        img = self.normalize_img(img)
        return img

    def worker_load_data_train(self, i, file, s_data_x, s_data_y):

        data_x = self.shared_to_numpy(*s_data_x)
        data_y = self.shared_to_numpy(*s_data_y)

        if self.template is not None:
            data_x[i, :, :, :, 0] = self.template
        else:
            file_target = re.match("(.*?_vol-[+-]?[0-9]*[.]?[0-9]+).*?", file).group(1)
            data_x[i, :, :, :, 0] = self.load_img(file_target)
        img = self.load_img(file)
        data_x[i, :, :, :, 1] = img
        data_y[i, ] = load_file(file + ".txt")

    def worker_load_data_infer(self, i, file, s_data_x):
        data_x = self.shared_to_numpy(*s_data_x)

        if self.template is not None:
            data_x[i, :, :, :, 0] = self.template
        else:
            file_target = re.match("(.*?_vol-[+-]?[0-9]*[.]?[0-9]+).*?", file).group(1)
            data_x[i, :, :, :, 0] = self.load_img(file_target)
        img = self.load_img(file)
        data_x[i, :, :, :, 1] = img

    def create_processes(self, nb_proc, files, shared_mem, curr_proc_batch=0):
        # will not create more processes than available cores
        nb_proc = min(mp.cpu_count(), nb_proc)

        #processes creation
        processes = []
        for i in range(nb_proc):
            k = curr_proc_batch * nb_proc + i
            if not self.is_inference:
                process = mp.Process(target=self.worker_load_data_train, args=(k, files[k], *shared_mem))
            else:
                process = mp.Process(target=self.worker_load_data_infer, args=(k, files[k], *shared_mem))
            processes.append(process)
            process.start()

        # waiting for processes to end
        for process in processes:
            process.join(timeout=5)
            process.terminate()

    def __mp_data_generation(self, list_files_batch):
        """Generates data containing batch_size samples"""

        #first, we nitialize the shared memory to zero
        self.data_x, self.data_y, self.s_mem = (None, None, None)
        self.create_shared_mem()
        self.data_x[:] = 0
        self.data_y[:] = 0

        # If the batch size is bigger than available cores (proc_batches > 0),
        # then we need to manage the batch loading among the available cores.
        # Otherwise, there is enough cpus for the batch size, so each sample in the batch can be loaded by one cpu
        if self.avail_cores > 0:
            cores = self.avail_cores
        else:
            cores = mp.cpu_count()

        p_batches = self.batch_size // cores
        remainder = self.batch_size % cores
        if p_batches > 0:
            for i in range(p_batches):
                # print("Loading %d/%d mini-batch with %d cpus" % (i, p_batches if remainder > 0 else p_batches-1, cores))
                self.create_processes(nb_proc=cores, files=list_files_batch, shared_mem=self.s_mem, curr_proc_batch=i)
            # and now the remaining processes
            if remainder > 0:
                # print("Loading %d/%d mini-batch with %d cpus" % (p_batches, p_batches, cores))
                self.create_processes(
                    nb_proc=remainder, files=list_files_batch, shared_mem=self.s_mem, curr_proc_batch=p_batches)
        else:
            # print("Loading batch with %d cpus" % self.batch_size)
            self.create_processes(nb_proc=self.batch_size, files=list_files_batch, shared_mem=self.s_mem)

        # finally, we return the shared array
        data = tuple([self.data_x, self.data_y])
        if self.is_inference:
            data = (self.data_x, )

        return data

    def __data_generation(self, list_files_batch):
        # Initialization
        x = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size, self.n_regressors), dtype=np.float32)

        # Generate data
        for i, file in enumerate(list_files_batch):
            # Store sample
            if self.template is not None:
                x[i, :, :, :, 0] = self.template
            else:
                # we use the not transformed volume as target
                file_target = re.match("(.*?_vol-[+-]?[0-9]*[.]?[0-9]+).*?", file).group(1)
                x[i, :, :, :, 0] = self.load_img(file_target)
            x[i, :, :, :, 1] = self.load_img(file)
            if self.is_inference is False:
                y[i, ] = load_file(file + ".txt")

        data = tuple([x, y])

        return data
