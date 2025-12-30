# DGCNet: Dynamic Global Convolution Network for DOA Estimation

This repository contains the official implementation of **DGCNet**, a deep learning framework designed for high-precision Direction of Arrival (DOA) estimation.

The method leverages **Dynamic Global Convolutions** to capture both local details and global context from covariance matrices. It employs a novel **Pre-training & Fine-tuning** strategy coupled with **Weighted Interpolation** to achieve robust, off-grid DOA estimation, even under challenging conditions (low SNR, few snapshots).

## 🌟 Key Features

* **DGCNet Architecture**: Incorporates `ContMixBlock` with dynamic large-kernel convolutions and `DilatedReparamBlock` for efficient feature extraction.
* **Two-Stage Training Strategy**:
    * **Pre-training**: Trained on ideal **Expected Covariance Matrices (ECM)** to learn pure feature mappings.
    * **Fine-tuning**: Trained on **Sampled Covariance Matrices (SCM)** to adapt to real-world noise and imperfections.
* **Asymmetric Loss (ASL)**: Addresses the extreme class imbalance problem in spatial spectrum prediction.
* **Off-Grid Estimation**: Uses weighted interpolation post-processing to convert grid-based classification into continuous angle predictions.

## 📂 Directory Structure

```text
├── model/
│   ├── Dynamic_Global_ConvolutionNet_B.py  # DGCNet Model Architecture (Backbone & Head)
│   ├── ASL.py                              # Asymmetric Loss Function
│   └── regularizer.py                      # L1/L2/Group Lasso Regularization Tools
├── data/
│   └── dataProcess.py                      # PyTorch Dataset & DataLoader
├── generation/                             # MATLAB Scripts for Data Generation
│   ├── generate_train_data.m               # Base training data (K=2 sources)
│   ├── generate_TL_data.m                  # Transfer Learning data (K=4 sources)
│   └── generate_test_data.m                # Test data for various scenarios (K=1~10)
├── train.py                                # Main Training Script (Pre-train + Fine-tune)
├── test_tune.py                            # Inference Script (Evaluation & Visualization)
```

## 🛠️ Requirements

### Python Environment

* **eniops**==0.8.1
* h5py==3.14.0
* matplotlib==3.10.7
* mmengine==0.2.0
* **natten**==0.17.5+torch260cu124
* **numpy**==2.1.2
* pandas==2.3.0
* **python**==3.12.11
* **scikit-learn**==1.7.1
* scipy==1.16.2
* tensorboard==2.20.0
* **timm**==1.0.20
* **torch**==2.6.0+cu124
* tqdm==4.67.1
* triton==3.2.0



### MATLAB

* Required for running scripts in the `generation/` folder to produce `.h5` datasets.
* *Note: Core signal processing formulas are implemented manually, so specific toolboxes may not be strictly required, but Phased Array System Toolbox is recommended.*

## 🚀 Usage Guide

### Step 1: Data Generation

Before training, you must generate the datasets using MATLAB.
*Ensure you update the `filename` paths in the `.m` scripts to your local storage directory.*

1. **Generate Training Data**: Run `generate_train_data.m`.
* Output: `ECM_k=2.h5` (Pre-training).

1. **Generate Transfer Learning Data**: Run `generate_TL_data.m`.
* Output: `SCM_k=4_coverage.h5` (Fine-tuning).

1. **Generate Test Data**: Run `generate_test_data.m`.
* Output: `DOA_test_K_7.h5`, etc. (Generates scenarios with 1 to 7 sources).

### Step 2: Training

Run the `train.py` script. This script automatically handles the two-stage training process.

**Configuration**:
Open `train.py` and modify the paths at the top of the file:

```python
# Change these to your actual data paths
filename_data_Pre_train = '/your/path/to/ECM_k=2.h5'
filename_data_Fine_tuning = '/your/path/to/SCM_k=2.h5'
filename_save_root = '/your/path/to/save/models/'
filename_logs_root = '/your/path/to/save/logs/'
```

**Run Command**:

```bash
python train.py
```

* **Stage 1 (Pre-train)**: Trains on ECM data. Best model saved as `model_best.pth` in the pre-train folder.
* **Stage 2 (Fine-tune)**: Loads the pre-trained weights and continues training on SCM data.

### Step 3: Evaluation & Inference

Use `test_tune.py` to evaluate the model on test data and visualize the results.

**Configuration**:
Open `test_tune.py` and modify the model/data paths:

```python
# Path to your trained model
model = torch.load('/your/path/to/Fine_tuning_cnn_k=2/model_best.pth')
# Path to test data
f_data = '/your/path/to/DOA_test_K_7.h5'
```

**Run Command**:

```bash
python test_tune.py
```

**Output**:

1. **Console**: Prints Ground Truth angles, Original (Grid) Estimates, and Refined (Off-Grid) Estimates.
2. **Visualization**: Saves a spatial spectrum plot (e.g., `DGCNet_k=7.svg`).
3. **Result File**: Saves predicted spectrums to an `.h5` file for further analysis.


## 📊 Results Example

The framework demonstrates superior resolution capability. For example, in a 7-source scenario:

```
GT:       [-48.5, -34.1, -14.0, -0.9, 10.9, 27.1, 44.1]
Estimate: [-48.48, -34.12, -13.98, -0.89, 10.92, 27.05, 44.11]
```

## 📝 License

[MIT License]
