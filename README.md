# Fully Automated Coronary Artery Analysis Algorithm
Coronary Artery Analysis,  nn-UNet  

## Overview
This repository provides the implementation of the faCCTA algorithm, a fully automated solution for coronary analysis. 
It encompasses a comprehensive suite of features, including aorta and coronary artery extraction, plaque analysis, 
stenosis assessment, and perivascular fat analysis. The algorithm is built upon [nnUnet](https://github.com/MIC-DKFZ/nnUNet) 
and [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators).


## Requirements
- PyTorch 2.0.0
- Python 3.10
- At least 32 GB GPU memory
- Following packages
    - MedPy==0.4.0
    - nibabel==3.2.1
    - numpy==1.24.1
    - opencv-python==4.4.0.46
    - pandas==2.0.3
    - pydicom==2.1.2
    - scikit-image==0.21.0
    - scikit-learn==0.24.1
    - scipy==1.10.1
    - seaborn==0.13.0
    - SimpleITK==2.2.1
    - sklearn==0.0
    - threadpoolctl==3.1.0
    - tifffile==2023.7.10
    - tqdm==4.53.0
    - typing-extensions==4.3.0
    - connected-components-3d==3.10.5
    - openpyxl==3.1.3

## Code structure

The training code consists of two main parts: the training framework related to nnU-Net, 
and the model design and optimization methods specific to faCCTA. The detailed code structure 
is as follows:

- acvl_utils: nnUnet-related codes,  https://github.com/MIC-DKFZ/acvl_utils
- batchgenerators: nnUnet-related codes,  https://github.com/MIC-DKFZ/batchgenerators
- dynamic_network_architectures: nnUnet-related codes,  https://github.com/MIC-DKFZ/dynamic-network-architectures
- nnunet/nnunetv2: old/new version of nnUnet,  https://github.com/MIC-DKFZ/nnUNet
- Training: codes for faCCTA training
- Testing: codes for faCCTA testing
- Utils: some usage of nnUnet, SimpleITK and scikit-image.

## Training
Training is divided into three tasks, each following the same training pipeline. 
These three tasks are Task_10001_Heart (heart segmentation), Task_10002_Coarse 
(aorta segmentation and coarse coronary artery region extraction), and Task_10003_Fine 
(fine-grained coronary artery and plaque extraction).

- Step 0: Set the training data and model saving paths in the configs file.
All training data must save as NIFTI format as nn-Unet required.


- Step 1: All  tasks share similar training procedures
	- cd Training/Tasks/Task_XXX/
	- python step_0_nnUNet_prepare_raw_data.py
	- python step_1_nnUNet_planning_preprocessing.py
	- python step_3_nnUNet_change_plan.py
	- python step_2_nnUNet_run_training.py

## Testing

- After training, you can run Testing/test.py to test your own cases. 
- You can also download our models [trained models](TBD) for testing. 


 A Simple usage is: (more details please refer to test.py)

    # Init predictor
    predictor = init_predictor(MODEL_DIR, DEVICE)

    # Do Inference
    predict_from_DICOM_dir(predictor, INPUT_DIR, OUTPUT_DIR)

## Acknowledgement
-Thank [nnUnet](https://github.com/MIC-DKFZ/batchgenerators), [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators)

