<img width="1965" height="1595" alt="{FA46DAE1-E626-4681-B4D1-DEE0502E4788}" src="https://github.com/user-attachments/assets/cf7d8409-0e9f-4810-a263-f494dee54bb5" /># Fully Automated Coronary Artery Analysis Algorithm

**Keywords:** Coronary Artery Analysis, nnU-Net, faCCTA

---

## Overview

This repository provides the implementation of the **faCCTA** algorithm, a fully automated solution for coronary analysis. It encompasses a comprehensive suite of features, including:
* Aorta and coronary artery extraction
* Plaque analysis
* Stenosis assessment
* Perivascular fat analysis

To demonstate the above analysis results, we deployed the application on AWS accessed via: [The QUANTUM Study Demo Platform](http://52.83.98.61:8000/thequantumstudy/index.html). The sample data in the demo is selected from the dataset in QUANTUM study. 

<br>
<img width="1912" height="1145" alt="CCTADemoPreview" src="https://github.com/user-attachments/assets/75ac3a52-add9-4501-99c9-04068ac8688b" />
<br><br>

The algorithm is built upon [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators).

---

## Requirements

### System & Environment
* **OS / Environment:** Linux / Windows
* **Python:** 3.10
* **PyTorch:** 2.0.0
* **Hardware:** At least 32 GB GPU memory

### Python Packages
| Package | Version | Package | Version |
| :--- | :--- | :--- | :--- |
| `connected-components-3d` | 3.10.5 | `scikit-image` | 0.21.0 |
| `MedPy` | 0.4.0 | `scikit-learn` | 0.24.1 |
| `nibabel` | 3.2.1 | `scipy` | 1.10.1 |
| `numpy` | 1.24.1 | `seaborn` | 0.13.0 |
| `opencv-python` | 4.4.0.46 | `SimpleITK` | 2.2.1 |
| `openpyxl` | 3.1.3 | `sklearn` | 0.0 |
| `pandas` | 2.0.3 | `threadpoolctl` | 3.1.0 |
| `pydicom` | 2.1.2 | `tifffile` | 2023.7.10 |
| `tqdm` | 4.53.0 | `typing-extensions`| 4.3.0 |

---

## Code Structure

The repository consists of two main parts: the core training framework powered by nnU-Net, and the custom model design/optimization methods specific to faCCTA.

* 📂 **Third-party / nnU-Net Dependencies**
  * `acvl_utils`: nnU-Net related utilities — [GitHub](https://github.com/MIC-DKFZ/acvl_utils)
  * `batchgenerators`: Data augmentation framework — [GitHub](https://github.com/MIC-DKFZ/batchgenerators)
  * `dynamic_network_architectures`: Dynamic network topologies — [GitHub](https://github.com/MIC-DKFZ/dynamic-network-architectures)
  * `nnunet/nnunetv2`: Submodules for v1 and v2 versions of nnU-Net — [GitHub](https://github.com/MIC-DKFZ/nnUNet)
* 📂 **Core Algorithm**
  * `Training`: Source code for faCCTA model training.
  * `Testing`: Source code for faCCTA model inference and evaluation.
  * `Utils`: Utility scripts leveraging nnU-Net, SimpleITK, and scikit-image.

---

## Training

### Data Preparation
1. Before training, configure your dataset paths and model saving directories in the `configs` file.
2. All training data **must** be saved in **NIFTI** format (`.nii.gz`), as required by nnU-Net.

### Training Pipeline
The training pipeline is divided into three sequential tasks. Each task shares a similar execution procedure:

```bash
# Navigate to the specific task directory
cd Training/Tasks/Task_XXX/

# Run the pipeline step by step
python step_0_nnUNet_prepare_raw_data.py
python step_1_nnUNet_planning_preprocessing.py
python step_3_nnUNet_change_plan.py
python step_2_nnUNet_run_training.py
```

## Testing

* **Evaluate Custom Cases:** After completing the training phase, you can run `Testing/test.py` to test your own data.
* **Pre-trained Models:** Alternatively, you can download our [Trained Models (TBD)](TBD) directly for quick testing.

> 💡 **Quick Start:** Below is a simple usage example. For more detailed configurations and advanced options, please refer to `test.py`.

```python
# Initialize the predictor with model path and target device
predictor = init_predictor(MODEL_DIR, DEVICE)

# Perform end-to-end inference directly from a DICOM directory
predict_from_DICOM_dir(predictor, INPUT_DIR, OUTPUT_DIR)
```
## ️ License

> **Strictly for Research Use Only**

## Acknowledgement

We sincerely thank the authors and contributors of the following open-source frameworks, which served as foundational building blocks for the development of this project:

* 🚀 **[nnU-Net](https://github.com/MIC-DKFZ/nnUNet)** — A self-configuring method for deep learning-based biomedical image segmentation.
* 📦 **[batchgenerators](https://github.com/MIC-DKFZ/batchgenerators)** — A powerful framework for 2D and 3D real-time medical image data augmentation.
