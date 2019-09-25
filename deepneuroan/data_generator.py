import re
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
                 , template_file
                 , partition="infer"
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
        self._set_indexes_partition()
        self.template = self.load_img(self.template_file)

        if not self.is_inference:
            self.on_epoch_end()
        self.indexes_partition = self._set_indexes_partition()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes_partition) // self.batch_size

    def __getitem__(self, index):
        """Generate one batch of data"""
        list_files_batch = self.get_files_batch(index)
        # Generate data
        data = self.__mp_data_generation(list_files_batch)

        return data

    def get_files_batch(self, index):
        # Generate indexes of the batch
        indexes = self.indexes_partition[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of files for this batch
        list_files_batch = [self.list_files[k] for k in indexes]

        return list_files_batch

    def _set_indexes_partition(self):
        """partition the indexes into train/valid/test data"""
        if self.partition == "train":
            range_idx = (0, int(0.7 * self.n_samples))
        elif self.partition == "valid":
            range_idx = (int(0.7 * self.n_samples), int(0.85 * self.n_samples))
        elif self.partition == "test":
            range_idx = (int(0.85 * self.n_samples), self.n_samples)
        elif self.partition == "infer":
            range_idx = (0, self.n_samples)
        else:
            print("Error: partition %s is not valid" % self.partition)

        return self.indexes[range_idx[0]:range_idx[1]]

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

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
        arr = self.__shared_to_numpy(shared_arr, shape, dtype)
        return (shared_arr, shape, dtype,), arr

    def worker_load_data_train(self, i, file, s_data_x, s_data_y):
        data_x = self.shared_to_numpy(*s_data_x)
        data_y = self.shared_to_numpy(*s_data_y)

        data_x[i, :, :, :, 0] = self.template
        img = sitk.GetArrayFromImage(sitk.ReadImage(file + ".nii.gz", sitk.sitkFloat32))
        img = self.normalize_img(img)
        data_x[i, :, :, :, 1] = img
        data_y[i, ] = load_file(file + ".txt")

    def worker_load_data_infer(self, i, file, s_data_x):
        data_x = self.shared_to_numpy(*s_data_x)

        img = sitk.GetArrayFromImage(sitk.ReadImage(file + ".nii.gz", sitk.sitkFloat32))
        img = self.normalize_img(img)
        data_x[i, :, :, :, 0] = img

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
            process.join()

    def __mp_data_generation(self, list_files_batch):
        """Generates data containing batch_size samples"""

        # shared memory pointers
        s_data_x = None
        s_data_y = None
        s_data_x, data_x = self.create_shared_array(shape=(self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        if not self.is_inference:
            s_data_y, data_y = self.create_shared_array(shape=(self.batch_size, self.n_regressors), dtype=np.float32)
        s_mem = (s_data_x, s_data_y)

        # If there batch size is bigger than available cores (proc_batches > 0),
        # then we need to manage the batch loading among the available cores.
        # Otherwise, there is enough cpus for the batch size, so each sample in the batch can be loaded by one cpu
        cores = mp.cpu_count()
        p_batches = self.batch_size // cores
        remainder = self.batch_size % cores
        if p_batches > 0:
            for i in range(p_batches):
                print("Loading %d/%d mini-batch with %d cpus" % (i, p_batches if remainder > 0 else p_batches-1, cores))
                self.create_processes(nb_proc=cores, files=list_files_batch, shared_mem=s_mem, curr_proc_batch=i)
            # and now the remaining processes
            if remainder > 0:
                print("Loading %d/%d mini-batch with %d cpus" % (p_batches, p_batches, cores))
                self.create_processes(
                    nb_proc=remainder, files=list_files_batch, shared_mem=s_mem, curr_proc_batch=p_batches)
        else:
            print("Loading batch with %d cpus" % self.batch_size)
            self.create_processes(nb_proc=self.batch_size, files=list_files_batch, shared_mem=s_mem)

        data = tuple([data_x, data_y])
        if self.is_inference:
            data = (data_x, )

        return data
