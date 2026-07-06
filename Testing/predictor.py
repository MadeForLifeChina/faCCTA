import time
import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm
import skimage.morphology
import skimage.measure
import logging
from IO import read_DICOM, save_DICOM, read_nii_DICOM_dir

from Utils.nnUNet_base import NNUNetV2Predictor
from Utils.utils_simpleitk import get_nii_info, copy_nii_info, set_nii_info, resample, get_orientation_str, \
    to_orientation
from Utils.bbox import get_bbox, extend_bbox, extend_bbox_to_size, get_CT_non_air_bbox, clip_bbox, extend_bbox_physical
from Utils.connected_components import remove_small_objects_binary_3D

import config


class CoarseToFinePredictor:
    def __init__(self, dict_path_models, device, fast_mode=False):
        if fast_mode:
            list_TTA_axis = [[0, 0, 0]]
            stride = 1.0
        else:
            list_TTA_axis = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
            stride = 0.5

        # init models
        self.predictors = {}
        for task_name in dict_path_models.keys():
            base_predictor = NNUNetV2Predictor(
                list_model_pth=dict_path_models[task_name]["list_model_pth"],
                plan_file=dict_path_models[task_name]["plan_file"],
                dataset_file=dict_path_models[task_name]["dataset_file"],
                device=device,

                params={
                    'list_TTA_axis': list_TTA_axis,
                    'stride': stride,
                    'use_gaussian_for_sliding_window': True,
                    'gaussian_sigma_scale': 0.25,
                    'input_ch': dict_path_models[task_name]["input_ch"],
                },
            )
            self.predictors[task_name] = base_predictor

        # Simple test
        with torch.no_grad():
            input_ = torch.zeros((1, 1, 64, 64, 64))
            input_ = input_.to(device)
            output_ = self.predictors["coarse"].list_model[0].forward(input_)

    @staticmethod
    def pre_processing(predictor, image):
        norm_v = predictor.infer_params['norm_v']

        image = np.float32(image)
        pre_precess_channels = range(image.shape[0])
        for chan_i in pre_precess_channels:
            if predictor.infer_params['normalization_schemes'][chan_i] != 'NoNormalization':
                p_005 = norm_v[chan_i]['percentile_00_5']
                p_995 = norm_v[chan_i]['percentile_99_5']
                mean_ = norm_v[chan_i]['mean_v']
                std_ = norm_v[chan_i]['std_v']

                image[chan_i] = np.clip(image[chan_i], a_min=p_005, a_max=p_995)
                image[chan_i] = (image[chan_i] - mean_) / (std_ + 1e-7)

        return image

    def predict_heart(self, image_nii):
        time_start = time.time()

        task_name = "heart"
        patch_size = config.path_size["Task_10001_Heart"]  # (Z, Y, X)

        ori_image_info = get_nii_info(image_nii)

        # Resampling
        ori_spacing = ori_image_info['spacing'][::-1]
        ori_size = ori_image_info['size'][::-1]
        new_spacing = [ori_spacing[k] * ori_size[k] / patch_size[k] for k in range(3)]
        resampled_image_nii = resample(
            image_nii,
            new_spacing=new_spacing[::-1],
            new_size=patch_size[::-1],
            interp=sitk.sitkLinear,
            UseNearestNeighborExtrapolator=True
        )
        resampled_image = sitk.GetArrayFromImage(resampled_image_nii)

        # Pre-processing
        input_ = resampled_image[np.newaxis].astype(np.float32)
        input_ = self.pre_processing(self.predictors[task_name], input_)

        # Sliding-window
        pred_prob = self.predictors[task_name].sliding_window_inference(input_, argmax=True)

        # Post-processing
        pred = pred_prob  # Already argmax

        final_pred_nii = sitk.GetImageFromArray(np.uint8(pred))
        final_pred_nii = copy_nii_info(resampled_image_nii, final_pred_nii)

        output = {
            "heart_nii": final_pred_nii
        }

        return output

    def predict_coarse(self, image_nii, pred_heart_nii):
        time_start = time.time()

        task_name = "coarse"
        new_spacing = config.spacing["Task_10002_Coarse"]  # (Z, Y, X)

        ori_image_info = get_nii_info(image_nii)

        # Crop ROI
        pred_heart = sitk.GetArrayFromImage(pred_heart_nii)
        bbox = get_bbox(pred_heart > 0)
        bz, ez, by, ey, bx, ex = bbox
        start_ = [bx - 1, by - 1, bz - 1]
        end_ = [ex + 1, ey + 1, ez + 1]

        start_ = pred_heart_nii.TransformIndexToPhysicalPoint(start_)
        end_ = pred_heart_nii.TransformIndexToPhysicalPoint(end_)
        start_ = image_nii.TransformPhysicalPointToIndex(start_)
        end_ = image_nii.TransformPhysicalPointToIndex(end_)

        bbox = [start_[2], end_[2], start_[1], end_[1], start_[0], end_[0]]
        bbox = clip_bbox(bbox, max_shape=ori_image_info["size"][::-1])
        bz, ez, by, ey, bx, ex = bbox

        roi_image_nii = image_nii[bx:ex + 1, by:ey + 1, bz:ez + 1]

        # Resampling
        resampled_roi_image_nii = resample(roi_image_nii, new_spacing=new_spacing[::-1])
        resampled_roi_image = sitk.GetArrayFromImage(resampled_roi_image_nii)

        # Pre-processing
        input_ = resampled_roi_image[np.newaxis].astype(np.float32)
        input_ = self.pre_processing(self.predictors[task_name], input_)

        # Sliding-window
        pred_prob = self.predictors[task_name].sliding_window_inference(input_, argmax=False)

        # Post
        pred_prob = np.transpose(pred_prob, [1, 2, 3, 0])
        final_pred_nii = sitk.GetImageFromArray(pred_prob, isVector=True)
        final_pred_nii = copy_nii_info(resampled_roi_image_nii, final_pred_nii)

        output = {
            "coarse_prob_nii": final_pred_nii
        }

        return output

    def predict_fine(self, image_nii, pred_coarse_prob_nii):
        time_start = time.time()

        task_name = "fine"
        new_spacing = config.spacing["Task_10003_Fine"]  # (Z, Y, X)
        list_extend_physical = config.extend["Task_10003_Fine"]  # extend 10mm
        th_wall = 0.5

        ori_image_info = get_nii_info(image_nii)

        # Crop ROI, only image_nii cropped here, pred_wall_prob_nii has different spacing
        pred_wall_prob_nii = sitk.VectorIndexSelectionCast(pred_coarse_prob_nii, 2)
        pred_wall = sitk.GetArrayFromImage(pred_wall_prob_nii) >= int(255 * th_wall)
        bbox = get_bbox(pred_wall > 0)
        bz, ez, by, ey, bx, ex = bbox
        start_ = [bx - 1, by - 1, bz - 1]
        end_ = [ex + 1, ey + 1, ez + 1]

        start_ = pred_wall_prob_nii.TransformIndexToPhysicalPoint(start_)
        end_ = pred_wall_prob_nii.TransformIndexToPhysicalPoint(end_)
        start_ = image_nii.TransformPhysicalPointToIndex(start_)
        end_ = image_nii.TransformPhysicalPointToIndex(end_)

        bbox = [start_[2], end_[2], start_[1], end_[1], start_[0], end_[0]]
        bbox = clip_bbox(bbox, max_shape=ori_image_info["size"][::-1])

        bbox = extend_bbox_physical(
            bbox,
            list_extend_physical=list_extend_physical,
            max_shape=ori_image_info['size'][::-1],
            spacing=ori_image_info['spacing'][::-1]
        )

        bz, ez, by, ey, bx, ex = bbox

        roi_image_nii = image_nii[bx:ex + 1, by:ey + 1, bz:ez + 1]

        # Resampling
        resampled_roi_image_nii = resample(roi_image_nii, new_spacing=new_spacing[::-1])
        pred_wall_prob_nii = resample(
            pred_wall_prob_nii,
            new_spacing=resampled_roi_image_nii.GetSpacing(),
            new_origin=resampled_roi_image_nii.GetOrigin(),
            new_size=resampled_roi_image_nii.GetSize(),
            new_direction=resampled_roi_image_nii.GetDirection(),
            interp=sitk.sitkLinear,
            dtype=sitk.sitkUInt8,
            constant_value=0,
            UseNearestNeighborExtrapolator=False
        )

        # Pre-processing
        resampled_roi_image = sitk.GetArrayFromImage(resampled_roi_image_nii)
        pred_coarse_wall = sitk.GetArrayFromImage(pred_wall_prob_nii) >= int(255 * th_wall)

        kernel = np.ones((3, 3, 3), np.uint8)
        center_line = skimage.morphology.skeletonize(pred_coarse_wall)
        center_line = skimage.morphology.binary_dilation(center_line, kernel)
        center_line = skimage.morphology.binary_dilation(center_line, kernel)

        input_ = np.concatenate((resampled_roi_image[np.newaxis], center_line[np.newaxis]), axis=0)
        input_ = self.pre_processing(self.predictors[task_name], input_)

        # Sliding-window
        pred_prob = self.predictors[task_name].sliding_window_inference(input_, argmax=False)

        # Post
        pred_prob = np.transpose(pred_prob, [1, 2, 3, 0])
        final_pred_nii = sitk.GetImageFromArray(pred_prob)
        final_pred_nii = copy_nii_info(resampled_roi_image_nii, final_pred_nii)

        output = {
            "fine_prob_nii": final_pred_nii
        }
        return output

    def predict_from_nii(self, image_nii):
        with torch.no_grad():
            time_start = time.time()
            image_nii = to_orientation(image_nii, "LPS")

            # Heart
            output_heart = self.predict_heart(image_nii)

            # Coarse
            output_coarse = self.predict_coarse(image_nii, output_heart["heart_nii"])

            # Boundary refine
            output_fine = self.predict_fine(image_nii, output_coarse["coarse_prob_nii"])

            output = {
                "output_heart": output_heart,
                "output_coarse": output_coarse,
                "output_fine": output_fine,
            }

            total_time = time.time() - time_start
            return output, total_time


def init_predictor(model_dir, device, better_connectivity_mode=False):
    dict_path_models = {
        "heart": {
            "list_model_pth": [
                f"{model_dir}/Task_10001_Heart/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_final.pth"],
            "plan_file": f'{model_dir}/Task_10001_Heart/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json',
            "dataset_file": f'{model_dir}/Task_10001_Heart/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json',
            "input_ch": 1
        },

        "coarse": {
            "list_model_pth": [
                f"{model_dir}/Task_10002_Coarse/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_final.pth"],
            "plan_file": f'{model_dir}/Task_10002_Coarse/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json',
            "dataset_file": f'{model_dir}/Task_10002_Coarse/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json',
            "input_ch": 1
        },

        "fine": {
            "list_model_pth": [
                f"{model_dir}/Task_10003_Fine/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_final.pth"],
            "plan_file": f'{model_dir}/Task_10003_Fine/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json',
            "dataset_file": f'{model_dir}/Task_10003_Fine/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json',
            "input_ch": 2
        },
    }

    predictor = CoarseToFinePredictor(dict_path_models=dict_path_models, device=device)

    return predictor


def predict_from_DICOM_dir(predictor, input_DICOM_dir, output_DICOM_dir):
    # Reading
    time_start = time.time()
    image_nii, list_dcm_info = read_DICOM(input_DICOM_dir)
    print(f"==> Reading DICOM use {time.time() - time_start}")

    # Predicting
    time_start = time.time()
    all_output, total_time = predictor.predict_from_nii(image_nii)
    print(f"==> Model predicting use {time.time() - time_start}")

    # ----------------------- Saving to DICOM --------------------------- #
    time_start = time.time()
    pred = np.zeros(image_nii.GetSize()[::-1], np.int16)
    aorta_prob_nii = sitk.VectorIndexSelectionCast(all_output["output_coarse"]["coarse_prob_nii"], 1)
    aorta_prob_nii = resample(
        aorta_prob_nii,
        image_nii.GetSpacing(),
        new_origin=image_nii.GetOrigin(),
        new_size=image_nii.GetSize(),
        new_direction=image_nii.GetDirection(),
        interp=sitk.sitkLinear,
        dtype=sitk.sitkUInt8,
        constant_value=0,
        UseNearestNeighborExtrapolator=False
    )
    aorta_prob = sitk.GetArrayFromImage(aorta_prob_nii)
    aorta = aorta_prob >= int(255 * 0.5)

    lumen_prob_nii = sitk.VectorIndexSelectionCast(all_output["output_fine"]["fine_prob_nii"], 1)
    lumen_prob_nii = resample(
        lumen_prob_nii,
        image_nii.GetSpacing(),
        new_origin=image_nii.GetOrigin(),
        new_size=image_nii.GetSize(),
        new_direction=image_nii.GetDirection(),
        interp=sitk.sitkLinear,
        dtype=sitk.sitkUInt8,
        constant_value=0,
        UseNearestNeighborExtrapolator=False
    )

    lumen_prob = sitk.GetArrayFromImage(lumen_prob_nii)
    lumen = lumen_prob >= int(255 * 0.5)
    lumen = remove_small_objects_binary_3D(lumen, min_size=50. / np.prod(image_nii.GetSpacing()))

    plaque_prob_nii = sitk.VectorIndexSelectionCast(all_output["output_fine"]["fine_prob_nii"], 2)
    plaque_prob_nii = resample(
        plaque_prob_nii,
        image_nii.GetSpacing(),
        new_origin=image_nii.GetOrigin(),
        new_size=image_nii.GetSize(),
        new_direction=image_nii.GetDirection(),
        interp=sitk.sitkLinear,
        dtype=sitk.sitkUInt8,
        constant_value=0,
        UseNearestNeighborExtrapolator=False
    )

    plaque_prob = sitk.GetArrayFromImage(plaque_prob_nii)
    plaque = plaque_prob >= int(255 * 0.5)

    # Saving DICOM
    aorta_dilated = skimage.morphology.binary_dilation(aorta)
    lumen_dilated = skimage.morphology.binary_dilation(lumen)

    pred[aorta > 0] = 1000
    pred[np.logical_and(aorta_dilated > 0, lumen_dilated > 0)] = 1000
    pred[lumen > 0] = 2000
    pred[plaque > 0] = 3000

    save_DICOM(pred, list_dcm_info, output_DICOM_dir)
    print(f"==> Saving use {time.time() - time_start}")

    # Saving Nii
    pred_nii = sitk.GetImageFromArray(pred)
    pred_nii = copy_nii_info(image_nii, pred_nii)
    sitk.WriteImage(pred_nii, f"{output_DICOM_dir}/pred_nii.nii.gz")
