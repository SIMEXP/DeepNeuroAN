#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:45:28 2019

@author: ltetrel
"""

import math
import os
import argparse
import datetime
import shutil
import platform
import numpy as np
import SimpleITK as sitk
from scipy import stats
from pyquaternion import Quaternion
from preproc import create_ref_grid, DataPreprocessing, get_epi
import deepneuroan.utils as utils

def create_empty_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


def extract_path(dir):
# paths to all the moving brains data

    source_paths = []
    for root, _, files in os.walk(dir):
        for file in files:
            if os.path.join(root, file)[-7:] == ".nii.gz" or os.path.join(root, file)[-4:] == ".nii":
                source_paths += [os.path.join(root, file)]
    return source_paths

def generate_random_quaternions(rnd, range_rad, p_outliers=-1, method="gauss"):
    q = np.zeros((rnd.shape[0], rnd.shape[1], 4))

    if method == "gauss":
        # gaussian sampling for little angles : sampling in tangent around space unit quaternion exponential
        # https://math.stackexchange.com/questions/473736/small-angular-displacements-with-a-quaternion-representation
        # p of the samples (outliers) will be over angle range, multiplied by a factor to correct the assymetry
        if p_outliers < 0.0:
            p_outliers = 1e-3
        sigma_outliers = stats.norm.ppf(1 - p_outliers / 2)
        sigma = (range_rad / sigma_outliers)

        # assym_factor = 0.615
        # sigma = sigma * assym_factor
        r = rnd * sigma
        theta = np.linalg.norm(r, axis=2)
        q[:, :, 0] = np.cos(theta)
        q[:, :, 1:] = r * np.dstack([(1 / theta) * np.sin(theta)] * 3)
    elif method == "uniform":
        # randomly sampling p outliers quaternions using uniform law
        # http://planning.cs.uiuc.edu/node198.html
        # Trying also with Shoemake method http://refbase.cvc.uab.es/files/PIE2012.pdf
        # q = np.dstack((np.sqrt(1.0 - rnd[:, :, 0]) * (np.sin(2 * math.pi * rnd[:, :, 1]))
        #                , np.sqrt(1.0 - rnd[:, :, 0]) * (np.cos(2 * math.pi * rnd[:, :, 1]))
        #                , np.sqrt(rnd[:, :, 0]) * (np.sin(2 * math.pi * rnd[:, :, 2]))
        #                , np.sqrt(rnd[:, :, 0]) * (np.cos(2 * math.pi * rnd[:, :, 2]))))
        q = np.dstack((np.sqrt(1.0 - rnd[:, :, 0]) * (np.sin(2 * math.pi * rnd[:, :, 1]))
                       , np.sqrt(1.0 - rnd[:, :, 0]) * (np.cos(2 * math.pi * rnd[:, :, 1]))
                       , np.sqrt(rnd[:, :, 0]) * (np.sin(2 * math.pi * rnd[:, :, 2]))
                       , np.sqrt(rnd[:, :, 0]) * (np.cos(2 * math.pi * rnd[:, :, 2]))))
    else:
        raise Exception("method is unknown, available methods are : 'gauss', 'uniform'.")

    return q


def generate_random_transformations(n_transfs, n_vol, p_outliers, range_rad, range_mm, seed=None):
    # generation of random rotation on axis x y z, and 3 translations (mm) in the given range for each EPI

    if seed is not None:
        np.random.seed(seed)
    if p_outliers < 0.0:
        p_outliers = 1e-3

    # random gaussian for the inliers
    rnd = np.random.rand(n_vol, n_transfs, 6)
    q = generate_random_quaternions(rnd[:, :, :3], range_rad, p_outliers, "uniform")
    t = rnd[:, :, 3:] * (range_mm / stats.norm.ppf(1 - p_outliers / 2))

    if p_outliers > 1e-3:
        # random uniform for the outliers
        n_outliers = int(np.ceil(p_outliers * n_transfs))
        rnd_uniform = np.random.rand(n_vol, n_outliers, 6)
        # q_uniform = generate_random_quaternions(rnd_uniform[:, :, :3], range_rad, p_outliers, "uniform")
        # maximum translations allowed is +-25mm
        t_uniform = (rnd_uniform[:, :, 3:] * (25 - range_mm) + range_mm) * np.sign(rnd_uniform[:, :, :3] - 0.5)

        r = rnd_uniform[:, :, 3:] * (100 - range_mm) + range_mm
        r = r * np.sign(rnd_uniform[:, :, :3] - 0.5)

        # now we can replace the outliers on the original matrices
        # logic = np.zeros((n_vol, n_transfs), dtype=bool)
        # angles = 2 * np.arccos(q[:, :, 0])
        # logic_Q = np.argsort(angles, axis=1)[:, -n_outliers:]
        # for ii in range(logic_Q.shape[0]):
        #     logic[ii, :] = np.isin(range(n_transfs), logic_Q[ii, :])
        # logic = np.dstack([logic] * 4)
        # q[logic] = q_uniform.flatten()

        logic = np.zeros((n_vol, n_transfs), dtype=bool)
        norm = np.linalg.norm(t, axis=2)
        logic_T = np.argsort(norm, axis=1)[:, -n_outliers:]
        for ii in range(logic_T.shape[0]):
            logic[ii, :] = np.isin(range(n_transfs), logic_T[ii, :])
        logic = np.dstack([logic] * 3)
        t[logic] = t_uniform.flatten()

    return q, t


class TrainingGeneration:
    def __init__(self
                 , data_dir=None
                 , out_dir=None
                 , number=None
                 , seed=None
                 , rot=None
                 , trans=None
                 , p_outliers=None):
        self._data_dir = None
        self._out_dir = None
        self._nb_transfs = None
        self._seed = None
        self._range_rad = None
        self._range_mm = None
        self._p_outliers = None

        self._set_data_dir(data_dir)
        self._set_out_dir(out_dir)
        self._set_num_transfs(number)
        self._set_seed(seed)
        self._set_range_rad(rot)
        self._set_range_mm(trans)
        self._set_p_outliers(p_outliers)

    def __repr__(self):
        return str(__file__) \
               + "\n" + str(datetime.datetime.now()) \
               + "\n" + str(platform.platform()) \
               + "\n" + "class TrainingGeneration()" \
               + "\n\t input data dir : %s" % self._data_dir \
               + "\n\t dest dir : %s" % self._out_dir \
               + "\n\t n transformations : %d" % self._nb_transfs \
               + "\n\t maximum rotation : %.2f deg" % (self._range_rad * 180 / math.pi) \
               + "\n\t maximum translation : %.2f mm" % self._range_mm \
               + "\n\t p outliers : %.2f %%" % (100.0 * self._p_outliers) \
               + "\n\t seed : %d \n" % self._seed if self._seed is not None else "\n\t no seed \n"

    def _set_data_dir(self, data_dir=None):

        if data_dir is None:
            self._data_dir = os.getcwd()
        else:
            self._data_dir = data_dir

    def _set_out_dir(self, out_dir=None):

        if out_dir is None:
            self._out_dir = os.path.join(self._data_dir, "derivatives", "deepneuroan", "training", "generated_data")
        else:
            self._out_dir = out_dir

    def _set_num_transfs(self, number):

        if number is None:
            self._nb_transfs = int(10000)
        else:
            self._nb_transfs = int(number)

    def _set_seed(self, seed=None):

        if seed is not None:
            self._seed = int(seed)

    def _set_range_rad(self, rot=None):

        if rot is None:
            self._range_rad = 5.0 * math.pi / 180
        elif rot > 180:
            self._range_rad = math.pi
        else:
            self._range_rad = float(rot) * math.pi / 180

    def _set_range_mm(self, trans=None):

        if trans is None:
            self._range_mm = 3.0
        else:
            self._range_mm = float(trans)

    def _set_p_outliers(self, p_outliers=None):

        if p_outliers is None:
            self._p_outliers = 0.05
        elif p_outliers <= 0:
            self._p_outliers = -1
        else:
            self._p_outliers = float(p_outliers)

    def _rigid_to_file(self, output_txt_path, rigid):
        q = rigid[:4]
        t = rigid[4:]
        rigid_matrix = np.eye(4)
        rigid_matrix[:3, :3] = Quaternion(q).rotation_matrix
        rigid_matrix[:3, 3] = t
        angles = np.array(Quaternion(q).yaw_pitch_roll[::-1]) * 180 / math.pi
        with open(output_txt_path, "w") as fst:
            fst.write(self.__repr__())
            fst.write("\n\nquaternion in scalar-first format")
            fst.write("\n\nq0 \t\t q1 \t\t q2 \t\t q3 \t\t t0 (mm) \t t1 (mm) \t t2 (mm)")
            fst.write("\n%.6f \t %.6f \t %.6f \t %.6f \t %.6f \t %.6f \t %.6f"
                      % (rigid[0], rigid[1], rigid[2], rigid[3], rigid[4], rigid[5], rigid[6]))
            fst.write("\n\nEuler angle (ZYX)")
            fst.write("\n\ntheta_x (deg) \t theta_y (deg) \t theta_z (deg)")
            fst.write("\n %.2f \t\t %.2f \t\t %.2f" % (angles[0], angles[1], angles[2]))
            fst.write("\n\nrigid transformation matrix (ZYX)")
            fst.write("\n" + str(rigid_matrix))

    def run(self):
        print(self.__repr__())

        create_empty_dir(self._out_dir)
        source_paths = extract_path(self._data_dir)

        # creating reference grid
        ref_grid = create_ref_grid()
        # creation of the template to the fixed grid
        ### this should be done under preproc...
        mni_template_path = DataPreprocessing(source_paths="").target_path
        template_brain = sitk.ReadImage(mni_template_path, sitk.sitkFloat32)
        template_brain_on_grid = utils.transform_volume(template_brain, ref_grid)
        sitk.WriteImage(template_brain_on_grid, os.path.join(self._out_dir, "template_on_grid.nii.gz"))

        # iteration through all the files
        for ii, source_path in enumerate(source_paths):
            print("## file %d/%d" % (ii + 1, len(source_paths)))
            try:
                source_brain = sitk.ReadImage(source_path, sitk.sitkFloat32)
            except Exception:
                print("Incompatible type with SimpleITK, ignoring %s" % source_path)
                continue

            output_filename = os.path.basename(source_path.split(".", maxsplit=1)[0]) \
                                               + "_vol-%04d" \
                                               + "_transfo-%06d" \
                                               + "." + source_path.split(".", maxsplit=1)[1]
            output_epi_name = os.path.basename(source_path.split(".", maxsplit=1)[0]) \
                                               + "_vol-%04d" \
                                               + "." + source_path.split(".", maxsplit=1)[1]

            is_fmri = False
            nb_vol = 1
            size = np.array(source_brain.GetSize())
            if len(size) == 4:
                is_fmri = True
                nb_vol = size[3]
            q, t = generate_random_transformations(
                self._nb_transfs, nb_vol, self._p_outliers, self._range_rad, self._range_mm, self._seed)
            fixed_brain = source_brain
            print("## nb volumes %d" % (nb_vol))
            for i in range(nb_vol):
                if is_fmri:
                    # we take the corresponding EPI
                    fixed_brain = get_epi(source_brain, i)

                for j in range(self._nb_transfs):
                    output_path = os.path.join(self._out_dir, output_filename % (i + 1, j + 1))
                    output_path_fixed = os.path.join(self._out_dir, output_epi_name % (i + 1))
                    output_txt_path = output_path.split(".")[0] + ".txt"

                    # transforming and resampling the fixed brain
                    rigid = np.concatenate([q[i, j, :], t[i, j, :]])
                    moving_brain = utils.transform_volume(fixed_brain, ref_grid, sitk.sitkBSplineResampler, rigid)
                    sitk.WriteImage(moving_brain, output_path)
                    fixed_brain_on_grid = utils.transform_volume(fixed_brain, ref_grid, sitk.sitkBSplineResampler)
                    sitk.WriteImage(fixed_brain_on_grid, output_path_fixed)

                    # writing the transformations into a file
                    self._rigid_to_file(output_txt_path, rigid)

                    print("#### transfo %d/%d - %s" % (i * self._nb_transfs + j + 1
                                                       , self._nb_transfs * nb_vol
                                                       , output_path))


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
        , "--number"
        , type=int
        , required=False
        , default=None
        , help="Number of tranformations to generate, Default: 1000",
    )

    parser.add_argument(
        "-s"
        , "--seed"
        , type=int
        , required=False
        , default=None
        , help="Random seed to use for data generation, Default: None",
    )

    parser.add_argument(
        "-r"
        , "--rot"
        , type=float
        , required=False
        , default=None
        , help="95% range in degree for random rotations [-r, r], Default: 5",
    )

    parser.add_argument(
        "-t"
        , "--trans"
        , type=float
        , required=False
        , default=None
        , help="95% range in mm for random translations [-t, t], Default: 3",
    )

    parser.add_argument(
        "-p"
        , "--p_outliers"
        , type=float
        , required=False
        , default=None
        , help="probability of the outliers, -1 for no outliers, Default: 0.05",
    )

    return parser


def main():
    args = get_parser().parse_args()
    train_gen = TrainingGeneration(**vars(args))
    train_gen.run()


if __name__ == '__main__':
    main()
