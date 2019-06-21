import math
import os
import argparse
import numpy as np
import SimpleITK as sitk
import preproc

class TrainingGeneration():
    def __init__(self
                 , data_dir=None
                 , out_dir=None
                 , n_transfs=None
                 , seed=None):
        self._data_dir = None
        self._out_dir = None
        self._n_transfs = None
        self._source_paths = None
        self._seed = None

        self._set_data_dir(data_dir)
        self._set_out_dir(out_dir)
        self._set_source_paths()
        self._set_n_transfs(n_transfs)
        self._set_seed(seed)

    def _set_data_dir(self, data_dir=None):

        if data_dir is None:
            self._data_dir = os.getcwd()
        else:
            self._data_dir = data_dir

    def _set_out_dir(self, out_dir=None):

        if out_dir is None:
            self._out_dir = os.path.join(self._data_dir, "training")
        else:
            self._out_dir = out_dir

    # paths to all the moving brains data
    def _set_source_paths(self):

        self._source_paths = []
        for root, _, files in os.walk(self._data_dir):
            for file in files:
                self._source_paths += [os.path.join(root, file)]

    def _set_n_transfs(self, n):

        if n is None:
            self._n_transfs = 1000
        else:
            self._n_transfs = n

    def _set_seed(self, seed=None):

        self._seed = seed
        if seed is None:
            np.random.seed()
        else:
            np.random.seed(int(seed))

    def transform_volume(self, transf, ref_grid, interp, brain):

        def_pix = 0.0
        brain_to_grid = sitk.Resample(
            brain, ref_grid, transf, interp, def_pix, sitk.sitkFloat32)

        return brain_to_grid

    def run(self):

        print("---- deepneuroan starting ----")
        print(os.path.dirname(__file__))
        print("---- training data generation ----")
        print("Input data dir :")
        print(self._data_dir)
        print("Dest dir :")
        print(self._out_dir)
        if self._seed is not None:
            print("seed :")
            print(self._seed)

        if not os.path.exists(self._out_dir):
            os.makedirs(self._out_dir)

        # creating reference grid
        ref_grid = preproc.create_ref_grid()

        # we iterate through all the files
        for ii, source_path in enumerate(self._source_paths):
            print(source_path)
            # source_brain = sitk.ReadImage(source_path, sitk.sitkFloat32)
            # # if the modality is fmri, then we take the middle EPI scan for the registration
            # # this is done with filter method because slicing not working
            # if self._modality == "bold":
            #     source_brain = self._get_middle_epi(self, source_brain)
            #
            # # Resample the source to reference grid, with the translation given by centroid
            # source_brain_to_grid = self.resample_to_grid(source_brain, ref_grid)
            #
            # # Writing source to reference grid
            # name = os.path.basename(source_path).split('.')[0] + "_transf%d.nii.gz" %
            # path = os.path.join(self._dest_dir, name)
            # sitk.WriteImage(source_brain_to_grid, path)
            # print("#### %d/%d - %s" % (ii + 1, len(self._source_paths), path))

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
        , description=""
        , epilog="""
            Documentation at https://github.com/SIMEXP/DeepNeuroAN
            """)

    parser.add_argument(
        "-d"
        , "--data_dir"
        , required=False
        , default=None
        , help="Directory containing all fmri data, Default: current directory",
    )

    parser.add_argument(
        "-o"
        , "--out_dir"
        , required=False
        , default=None
        , help="Output directory, Default: ./training",
    )

    parser.add_argument(
        "-n"
        , "--n_transfs"
        , required=False
        , default=None
        , help="Number of tranformations to generate, Default: 1000",
    )

    parser.add_argument(
        "-s"
        , "--seed"
        , required=False
        , default=None
        , help="Random seed to use for data generation, Default: None",
    )

    return parser

def main():
    args = get_parser().parse_args()
    train_gen = TrainingGeneration(**vars(args))
    train_gen.run()

if __name__ == '__main__':
    main()