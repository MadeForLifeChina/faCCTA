import json
from predictor import *
import SimpleITK as sitk
import numpy as np
import torch
import os
from Utils.utils_simpleitk import *
from Utils.connected_components import *


if __name__ == '__main__':
    # ------------------ Need modification -------------------- #
    MODEL_DIR = f"xxx/Models"
    INPUT_DIR = f"xxx/Input_DICOM"
    OUTPUT_DIR = f"xxx/Prediction"
    DEVICE = torch.device("cuda:0")

    # Init predictor
    predictor = init_predictor(MODEL_DIR, DEVICE)

    # Do Inference
    predict_from_DICOM_dir(predictor, INPUT_DIR, OUTPUT_DIR)



