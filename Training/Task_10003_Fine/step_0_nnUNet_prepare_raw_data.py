import random

import SimpleITK as sitk
import os
import numpy as np
import json
import math

import skimage.morphology
import torch

import task_setting
import config

from Utils.utils_simpleitk import *
from Utils.bbox import *
from Utils.connected_components import *


def get_dataset():
    image_dir = config.data_path["image"]
    gt_dir = config.data_path["gt_plaque"]
    data_split = json.load(open("../../data_split.json", 'r'))

    save_dir = f'{task_setting.path_nnunet_raw_data}/{task_setting.task_name}'
    save_dir_image = f"{save_dir}/imagesTr"
    save_dir_label = f"{save_dir}/labelsTr"
    os.makedirs(save_dir_image, exist_ok=True)
    os.makedirs(save_dir_label, exist_ok=True)

    list_train_case_ids = data_split["train"]
    print(f"==> Total {len(list_train_case_ids)} train cases")

    count = 0
    for case_id in list_train_case_ids:

        image_nii = sitk.ReadImage(f"{image_dir}/{case_id}.nii.gz")
        gt_nii = sitk.ReadImage(f"{gt_dir}/{case_id}.nii.gz")

        # To LPS
        image_nii = to_orientation(image_nii)
        gt_nii = to_orientation(gt_nii)

        # Check geometry
        same_geometry = compare_geometry_multiple([image_nii, gt_nii])
        if not same_geometry:
            raise Exception(f"{case_id} has not same geometry !")

        count += 1
        print(f"==> {count}: {case_id}")

        # Crop ROI
        roi_mask = sitk.GetArrayFromImage(gt_nii) >= 2
        roi_mask = remove_small_objects_binary_3D(roi_mask, min_size=100. / np.prod(gt_nii.GetSpacing()))
        bbox = get_bbox(roi_mask)
        if bbox is None:
            continue
        bbox = extend_bbox_physical(bbox, list_extend_physical=(10., 10., 10., 10., 10., 10.),
                                    max_shape=roi_mask.shape, spacing=gt_nii.GetSpacing()[::-1])
        bz, ez, by, ey, bx, ex = bbox
        image_nii = image_nii[bx:ex + 1, by:ey + 1, bz:ez + 1]
        gt_nii = gt_nii[bx:ex + 1, by:ey + 1, bz:ez + 1]

        # Split lumen with aorta
        gt = sitk.GetArrayFromImage(gt_nii)
        lumen = gt == 2
        plaque = gt == 3
        lumen_nii = sitk.GetImageFromArray(np.uint8(lumen))
        lumen_nii = copy_nii_info(image_nii, lumen_nii)
        plaque_nii = sitk.GetImageFromArray(np.uint8(plaque))
        plaque_nii = copy_nii_info(image_nii, plaque_nii)

        # Resampling
        image_nii = resample(image_nii, config.spacing["Task_10003_Fine"][::-1], interp=sitk.sitkLinear)
        lumen_nii = resample(
            lumen_nii,
            new_spacing=image_nii.GetSpacing(),
            new_origin=image_nii.GetOrigin(),
            new_size=image_nii.GetSize(),
            new_direction=image_nii.GetDirection(),
            interp=sitk.sitkLinear,
            dtype=sitk.sitkFloat32,
            constant_value=0,
            UseNearestNeighborExtrapolator=False
        )
        plaque_nii = resample(
            plaque_nii,
            new_spacing=image_nii.GetSpacing(),
            new_origin=image_nii.GetOrigin(),
            new_size=image_nii.GetSize(),
            new_direction=image_nii.GetDirection(),
            interp=sitk.sitkLinear,
            dtype=sitk.sitkFloat32,
            constant_value=0,
            UseNearestNeighborExtrapolator=False
        )
        lumen_nii = sitk.BinaryThreshold(lumen_nii, lowerThreshold=0.5, upperThreshold=1.0, insideValue=1,
                                         outsideValue=0)
        plaque_nii = sitk.BinaryThreshold(plaque_nii, lowerThreshold=0.5, upperThreshold=1.0, insideValue=1,
                                         outsideValue=0)

        # Center line
        lumen = sitk.GetArrayFromImage(lumen_nii)
        plaque = sitk.GetArrayFromImage(plaque_nii)
        center_line = skimage.morphology.skeletonize(lumen)
        kernel = np.ones((3, 3, 3), np.uint8)
        center_line = skimage.morphology.binary_dilation(center_line, kernel)
        center_line = skimage.morphology.binary_dilation(center_line, kernel)
        center_line_nii = sitk.GetImageFromArray(np.uint8(center_line))
        center_line_nii = copy_nii_info(image_nii, center_line_nii)

        # New GT
        new_gt = np.zeros(lumen.shape, np.uint8)
        new_gt[lumen > 0] = 1
        new_gt[plaque > 0] = 2
        new_gt_nii = sitk.GetImageFromArray(new_gt)
        new_gt_nii = copy_nii_info(image_nii, new_gt_nii)

        # Saving
        sitk.WriteImage(image_nii, f"{save_dir_image}/{case_id}_0000.nii.gz")
        sitk.WriteImage(center_line_nii, f"{save_dir_image}/{case_id}_0001.nii.gz")
        sitk.WriteImage(new_gt_nii, f"{save_dir_label}/{case_id}.nii.gz")

        print(f"        ----> Done ")


def get_dataset_info():
    save_dir = f"{task_setting.path_nnunet_raw_data}/{task_setting.task_name}"
    save_dir_label = f"{task_setting.path_nnunet_raw_data}/{task_setting.task_name}/labelsTr"

    # Train case ids
    numTraining = 0
    for file in os.listdir(save_dir_label):
        if file.find('.nii.gz') > -1:
            numTraining += 1
    print(f"==> Total {numTraining} train cases")

    # Dataset json
    dataset_info = {
        "channel_names": {  # must belong to ['CT', 'noNorm', 'zscore', 'rescale_to_0_1', 'rgb_to_0_1']
            "0": "CT",
            "1": "noNorm",
        },
        "labels": {
            "background": 0,
            "1": 1,
            "2": 2,
        },
        "numTraining": numTraining,
        "file_ending": ".nii.gz",
        # "regions_class_order": [0, 1, 2],
    }

    with open(os.path.join(save_dir, "dataset.json"), 'w') as f:
        json.dump(dataset_info, f, indent=4, sort_keys=False)


def main():
    get_dataset()
    get_dataset_info()


if __name__ == '__main__':
    main()
