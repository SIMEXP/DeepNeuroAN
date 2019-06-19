import math
import os
import argparse
import numpy as np
import SimpleITK as sitk
from bids import BIDSLayout
from deepneuroan import DataPreprocessing

class TrainingGeneration():
    def __init__(self
                 , data_dir=None
                 , dest_dir=None
                 , seed=None):
        self._data_dir = None
        self._dest_dir = None
        self._source_paths = None
        self._seed = None

        self._set_data_dir(data_dir)
        self._set_dest_dir(dest_dir)
        self._set_source_paths()
        self._set_seed(seed)

    def _set_data_dir(self, data_dir=None):

        if data_dir is None:
            self._data_dir = os.getcwd()
        else:
            self._data_dir = data_dir

    def _set_dest_dir(self, dest_dir=None):

        if dest_dir is None:
            self._dest_dir = os.path.join(self._data_dir, "derivatives", "deepneuroan", "preprocess")
        else:
            self._dest_dir = dest_dir

    # paths to all the moving brains data
    def _set_source_paths(self, source_paths):

        layout = BIDSLayout(self._data_dir)
        if self._modality == "all":
            self._source_paths = layout.get(
                    scope='raw', extensions=[".nii", ".nii.gz"], return_type='file')
        else:
            self._source_paths = layout.get(
                    scope='raw', suffix=self._modality, return_type='file')