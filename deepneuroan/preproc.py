#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:44:16 2019

@author: ltetrel
"""

import math
import os
import numpy as np
from nilearn import datasets#tmp to have mni template
import SimpleITK as sitk
from bids import BIDSLayout

class DataPreprocessing():
    def __init__(self
                 , data_dir=None
                 , dest_dir=None
                 , source_paths=None
                 , target_path=None
                 , modality=None
                 , interpolator=None):
        self._data_dir = None
        self._dest_dir = None
        self._source_paths = None
        self._target_path = None
        self._modality = None
        self._interpolator = None

        self._set_data_dir(data_dir)
        self._set_dest_dir(dest_dir)
        self._set_modality(modality)
        self._set_source_paths(source_paths)
        self._set_target_path(target_path)
        self._set_interpolator(interpolator)
        
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
        if not os.path.exists(self._dest_dir):
            os.makedirs(self._dest_dir)

    # paths to all the moving brains data
    def _set_source_paths(self, source_paths):
        if source_paths is None:
            layout = BIDSLayout(self._data_dir)
            if self._modality == "all":
                self._source_paths = layout.get(
                        scope='raw', extensions=[".nii", ".nii.gz"], return_type='file')
            else:
                self._source_paths = layout.get(
                        scope='raw', suffix=self._modality, return_type='file')
        else:
            self._source_paths = source_paths
            
    # paths to the target brain data (default to MNI template)
    def _set_target_path(self, target_path):
        if target_path is None:
            self._target_path = os.path.join(
                    os.path.dirname(__file__), "..", "data", "avg152T1_brain.nii.gz")
        else:
            self._target_path = target_path

    # which modality to preprocess ? (T1w, bold, all)
    def _set_modality(self, modality=None):
        if modality is None:
            modality = "all"
        else:
            self._modality = modality
    
    def _set_interpolator(self, interpolator=None):
        if interpolator is None:
            self._interpolator = sitk.sitkBSplineResampler
        elif interpolator == "nn":
            self._interpolator = sitk.sitkNearestNeighbor
        elif interpolator == "lin":
            self._interpolator = sitk.sitkLinear
        elif interpolator == "bspline":
            self._interpolator = sitk.sitkBSplineResampler
        elif interpolator == "lanczos":
            self._interpolator = sitk.sitkLanczosWindowedSinc
    
    def _get_middle_EPI(self, source_brain):
        extract = sitk.ExtractImageFilter()
        extract_size = np.array(source_brain.GetSize())
        extract.SetIndex([0, 0, 0, math.floor(extract_size[3]/2)])
        extract_size[3] = 0
        extract.SetSize(extract_size.tolist())
        
        return extract.Execute(source_brain)
    
    def _create_ref_grid(self, target_brain=None):
        
        if target_brain is None:
            #Then, it will be a grid near the MNI152 template
            spacing = (1, 1, 1)
            direction = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
            size = (250, 250, 250)
            pixel_type = 8
            origin = np.array([-90,  126, -72])
            t = np.array([182., 218., 182.])
        else:
            # pixel spacing (mm), lower space to have more pixel in a given area
            spacing = tuple( np.array(target_brain.GetSpacing())/2)
            direction = target_brain.GetDirection()
            size = tuple(3*[int(max(target_brain.GetSize())*2.4)])
            pixel_type = target_brain.GetPixelIDValue()
            origin = target_brain.GetOrigin()
            t = np.array(target_brain.GetSpacing())*np.array(target_brain.GetSize())

        # we want the grid to have the same center as target
        t = (t - np.array(spacing)*np.array(size))/2
        # pixel (0,0,0) is at top left
        origin = origin + t*np.array([1., -1., 1.])

        # construction of the reference
        ref_grid = sitk.Image(size, pixel_type)
        ref_grid.SetOrigin(origin)
        ref_grid.SetSpacing(spacing)
        ref_grid.SetDirection(direction)
        
        return ref_grid
    
    def _resample_to_grid(self, brain, ref_grid):
        center_grid = np.array(ref_grid.TransformContinuousIndexToPhysicalPoint(
            np.array(ref_grid.GetSize())/2.0))
        center_brain = np.array(brain.TransformContinuousIndexToPhysicalPoint(
            np.array(brain.GetSize())/2.0))
        source_to_target_lin = center_brain - center_grid
        if sum(source_to_target_lin) == 0:
            t_lin = sitk.Transform(3, sitk.sitkIdentity)
        else:
            t_lin = sitk.TranslationTransform(3)
            t_lin.SetOffset(source_to_target_lin)
            
        # Now we translate and resample the source to the grid
        # And the target to the grid
        def_pix = 0.0
        source_brain_to_grid = sitk.Resample(
            brain, ref_grid, t_lin, self._interpolator, def_pix, sitk.sitkFloat32)
        
        return source_brain_to_grid
    
    def run(self):
        print("---- deepneuroan starting ----")
        print(os.path.dirname(__file__))
        print("---- 1. preprocessing ----")
        print("Input data dir :")
        print(self._data_dir)
        print("Dest dir :")
        print(self._dest_dir)
        print("Target brain path :")
        print(self._target_path)
        print("Modalities :")
        print(self._modality)
        
        # reading target
        target_brain = sitk.ReadImage(self._target_path, sitk.sitkFloat32)
        
        # reference grid, based on the target
        if os.path.basename(self._target_path) == "avg152T1_brain.nii.gz":
            ref_grid = self._create_ref_grid()
        else:
            ref_grid = self._create_ref_grid(target_brain)
        
        # Resample target brain to reference grid
        target_brain_to_grid = self._resample_to_grid(target_brain, ref_grid)
        
        # Writing target to reference grid
        name = os.path.basename(self._target_path).split('.')[0] + "_to_ref_grid.nii.gz"
        path = os.path.join(self._dest_dir, name)
        sitk.WriteImage(target_brain_to_grid, path)
        
        # we iterate through all the files
        for ii, source_path in enumerate(self._source_paths):
            
            source_brain = sitk.ReadImage(source_path, sitk.sitkFloat32)
            # if the modality is fmri, then we take the middle EPI scan for the registration
            # this is done with filter method because slicing not working
            if self._modality == "bold":
                source_brain = self._get_middle_EPI(self, source_brain)
            
            # Resample the source to reference grid, with the translation given by centroid
            source_brain_to_grid = self._resample_to_grid(source_brain, ref_grid)
            
            # Writing source to reference grid
            name = os.path.basename(source_path).split('.')[0] + "_to_ref_grid.nii.gz"
            path = os.path.join(self._dest_dir, name)
            sitk.WriteImage(source_brain_to_grid, path)
            print("#### %d/%d - %s" %(ii+1, len(self._source_paths), path))
        
def main():
    
    data_dir = "/home/ltetrel/Documents/data/preventad_prep"
    data_prep = DataPreprocessing(data_dir=data_dir, modality="T1w")
    data_prep.run()
#    print("BIDS checking...")
#    data_dir = "/home/ltetrel/Documents/data/preventad_prep"
#    dest_dir = os.path.join(data_dir, "derivatives", "deepneuroan")
#    structural = True
#    
#    layout = BIDSLayout(data_dir)
#    if structural:
#        file_paths = layout.get(suffix="T1w", return_type='file')
#    else:
#        file_paths = layout.get(suffix="bold", return_type='file')
#    
#    print("Preparing the data...")
#    target_brain = sitk.ReadImage(datasets.MNI152_FILE_PATH, sitk.sitkFloat32)
#    
#    if not os.path.exists(dest_dir):
#        os.mkdir(dest_dir)
#    
#    # we iterate through all the files
#    for ii, source_brain_path in enumerate(file_paths):
#        
#        source_brain = sitk.ReadImage(source_brain_path, sitk.sitkFloat32)
#        # if the volume is fmri, then we take the middle scan for the registration
#        # this is done with filter method because slicing not working
#        if not structural:
#            extract = sitk.ExtractImageFilter()
#            extract_size = np.array(source_brain.GetSize())
#            extract.SetIndex([0, 0, 0, math.floor(extract_size[3]/2)])
#            extract_size[3] = 0
#            extract.SetSize(extract_size.tolist())
#            source_brain = extract.Execute(source_brain)
#        
#        # This part here will create a grid automatically based on MNI152
#        # the reference grid is on the space location as the MNI template, 
#        # but just with a slightly larger grid (more pixels)
#        
#        # pixel spacing (mm), lower space to have more pixel in a given area (default = MNI152 spacing)
#        spacing = tuple( np.array(target_brain.GetSpacing())/2)
#        
#        # cosine orientation (default = MNI152 template direction)
#        direction = target_brain.GetDirection()
#        
#        # ref size (default = MNI152 template size @ 120%)
#        size = tuple(3*[int(max(target_brain.GetSize())*2.4)])
#        
#        # pixel type (default = MNI152 template float32)
#        pixel_type = target_brain.GetPixelIDValue()
#        
#        # origin of the reference (mm) (default = MNI152 template origin)
#        origin = target_brain.GetOrigin()
#        
#        # construction of the reference
#        ref_grid = sitk.Image(size, pixel_type)
#        ref_grid.SetOrigin(origin)
#        ref_grid.SetSpacing(spacing)
#        ref_grid.SetDirection(direction)
#        
#        # Here, we recompute the origin, so the center for the grid is the same as target (MNI152)
#        center_target = np.array( target_brain.TransformContinuousIndexToPhysicalPoint(
#                np.array(target_brain.GetSize())/2.0))
#        center_ref = np.array( ref_grid.TransformContinuousIndexToPhysicalPoint(
#                np.array(ref_grid.GetSize())/2.0))
#        ref_grid.SetOrigin(origin - (center_ref - center_target))
#        
#        # Calculate the translation from source brain to target
#
#        center_target = np.array(target_brain.TransformContinuousIndexToPhysicalPoint(
#            np.array(target_brain.GetSize())/2.0))
#        center_source = np.array(source_brain.TransformContinuousIndexToPhysicalPoint(
#            np.array(source_brain.GetSize())/2.0))
#        source_to_target_lin = center_source - center_target
#        t_lin = sitk.TranslationTransform(3)
#        t_lin.SetOffset(source_to_target_lin)
#        
#        # Now we translate and resample the source to the grid
#        # And the target to the grid
#        interp = sitk.sitkBSplineResampler
#        def_pix = 0.0
#        source_brain_to_ref = sitk.Resample(
#            source_brain, ref_grid, t_lin, interp, def_pix, sitk.sitkFloat32)
#        
#        # Writing images
#        source_name = os.path.basename(source_brain_path).split('.')[0] + "_to_ref_grid.nii.gz"
#        source_brain_to_ref_path = os.path.join(dest_dir, source_name)
#        sitk.WriteImage(source_brain_to_ref, source_brain_to_ref_path)
#        print("#### %d/%d - %s" %(ii+1, len(file_paths), source_brain_to_ref_path))

if __name__ == '__main__':
    main()