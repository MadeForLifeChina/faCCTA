import SimpleITK as sitk
import os
import numpy as np
import json
import math

import task_setting
import config

from Utils.utils_simpleitk import to_orientation, resample, compare_geometry, resample_to_template


def get_dataset():
    image_dir = config.data_path["image"]
    gt_dir = config.data_path["gt_heart"]
    data_split = json.load(open("../../data_split.json", 'r'))

    save_dir = f'{task_setting.path_nnunet_raw_data}/{task_setting.task_name}'
    save_dir_image = f"{save_dir}/imagesTr"
    save_dir_label = f"{save_dir}/labelsTr"
    os.makedirs(save_dir_image, exist_ok=True)
    os.makedirs(save_dir_label, exist_ok=True)

    target_size = config.path_size["Task_10001_Heart"]  # (Z, Y, X)
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
        same_geometry = compare_geometry(image_nii, gt_nii, return_each=False)
        if not same_geometry:
            raise Exception(f"{case_id} has not same geometry !")

        count += 1
        print(f"==> {count}: {case_id}")

        # Resampling
        ori_spacing = image_nii.GetSpacing()[::-1]
        ori_size = image_nii.GetSize()[::-1]
        new_spacing = [ori_spacing[k] * ori_size[k] / target_size[k] for k in range(3)]
        image_nii = resample(image_nii, new_spacing=new_spacing[::-1], new_size=target_size[::-1], interp=sitk.sitkLinear)
        gt_nii = resample_to_template(gt_nii, template_nii=image_nii, dtype=sitk.sitkUInt8, constant_value=0)

        # Pseudo spacing
        image_nii.SetSpacing((1.234, 1.234, 1.234))
        gt_nii.SetSpacing((1.234, 1.234, 1.234))

        # Saving
        sitk.WriteImage(image_nii, f"{save_dir_image}/{case_id}_0000.nii.gz")
        sitk.WriteImage(gt_nii, f"{save_dir_label}/{case_id}.nii.gz")

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
        },
        "labels": {
            "background": 0,
            "1": 1,
            "2": 2,
        },
        "numTraining": numTraining,
        "file_ending": ".nii.gz",
        # "regions_class_order": [1, 2],
    }

    with open(os.path.join(save_dir, "dataset.json"), 'w') as f:
        json.dump(dataset_info, f, indent=4, sort_keys=False)


def main():
    get_dataset()
    get_dataset_info()


if __name__ == '__main__':
    main()
