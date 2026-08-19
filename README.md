# 🫁 Pneumonia Detection Using CNN

> A deep learning project for classifying chest X-ray images into **Normal** and **Pneumonia** using a custom Convolutional Neural Network (CNN) built with TensorFlow/Keras.

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)](https://keras.io/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-yellow?logo=googlecolab)](https://colab.research.google.com/)

---

## 📌 Overview

Pneumonia is a respiratory infection that can affect one or both lungs. Chest X-rays are commonly used during the diagnostic process, but interpreting medical images can be challenging and time-consuming.

This project explores a **Convolutional Neural Network (CNN)** for automated binary classification of chest X-ray images.

The model classifies images into:

- 🟢 **NORMAL**
- 🔴 **PNEUMONIA**

The notebook demonstrates an end-to-end deep learning workflow covering dataset acquisition, dataset verification, exploratory data analysis, image preprocessing, data augmentation, class-imbalance handling, CNN development, model training, evaluation, ROC-AUC analysis, and single-image prediction.

> **Important:** This is an educational/research project. It is not a clinically validated diagnostic system and must not be used as a substitute for professional medical diagnosis.

---

## 🎯 Objectives

- Build a custom CNN for binary chest X-ray classification.
- Classify chest X-ray images as Normal or Pneumonia.
- Resize and normalize grayscale X-ray images.
- Apply image augmentation to improve model generalization.
- Analyze dataset structure and class distribution.
- Handle training-set class imbalance using balanced class weights.
- Train and validate a custom CNN architecture.
- Use callbacks to improve training.
- Evaluate the model using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.
- Demonstrate single-image inference.

---

## 📊 Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset available through Kaggle.

**Dataset Source:**

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

### Dataset Structure

```text
chest_xray/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### Dataset Distribution Used in the Notebook

| Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Train | 1,341 | 3,875 | 5,216 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |
| **Total** | **1,583** | **4,273** | **5,856** |

The training data is imbalanced toward the Pneumonia class.

### Class Mapping

| Class | Label | Description |
|---|---:|---|
| NORMAL | 0 | Chest X-ray classified as Normal |
| PNEUMONIA | 1 | Chest X-ray classified as Pneumonia |

---

## 🔄 Project Workflow

```text
Kaggle Chest X-Ray Dataset
          ↓
Dataset Download & Extraction
          ↓
Dataset Verification
          ↓
Exploratory Data Analysis
          ↓
Class Distribution Analysis
          ↓
Image Visualization
          ↓
Image Resizing & Normalization
          ↓
Data Augmentation
          ↓
Class Weight Calculation
          ↓
Custom CNN Architecture
          ↓
Model Compilation
          ↓
Model Training
          ↓
Early Stopping / Learning-Rate Adjustment
          ↓
Best Model Checkpoint
          ↓
Test Evaluation
          ↓
Classification Report
          ↓
Confusion Matrix
          ↓
ROC Curve / AUC
          ↓
Single Image Prediction
```

---

## 🖼️ Image Preprocessing

The notebook uses `ImageDataGenerator` to load images directly from the directory structure in batches.

### Preprocessing Configuration

| Parameter | Configuration |
|---|---|
| Image Size | **150 × 150** |
| Color Mode | **Grayscale** |
| Input Shape | **150 × 150 × 1** |
| Batch Size | **32** |
| Pixel Scaling | `1./255` |
| Classification | Binary |
| Class Mode | `binary` |
| Training Shuffle | Yes |
| Validation/Test Shuffle | No |

Pixel values are normalized using `rescale=1./255`.

---

## 🔁 Data Augmentation

Training images are augmented to introduce controlled variation and improve generalization.

The training augmentation pipeline includes:

- Rotation range: **15°**
- Width shift: **0.1**
- Height shift: **0.1**
- Shear: **0.1**
- Zoom: **0.15**
- Horizontal flip: **Enabled**
- Fill mode: **nearest**

Validation and test images use normalization only without augmentation.

---

## ⚖️ Handling Class Imbalance

The training set contains substantially more Pneumonia images than Normal images.

To reduce bias toward the majority class, the notebook calculates balanced class weights using `compute_class_weight`.

The calculated weights are:

| Class | Label | Weight |
|---|---:|---:|
| NORMAL | 0 | **1.9448** |
| PNEUMONIA | 1 | **0.6730** |

These weights are passed to `model.fit()` during training.

---

## 🧠 CNN Architecture

A custom CNN is built from scratch using TensorFlow/Keras.

### Architecture

```text
Input
150 × 150 × 1
       │
       ▼
Conv2D - 16 Filters
3 × 3 Kernel
ReLU
       │
Batch Normalization
       │
Max Pooling 2 × 2
       │
       ▼
Conv2D - 32 Filters
3 × 3 Kernel
ReLU
       │
Batch Normalization
       │
Max Pooling 2 × 2
       │
       ▼
Conv2D - 64 Filters
3 × 3 Kernel
ReLU
       │
Batch Normalization
       │
Max Pooling 2 × 2
       │
       ▼
Conv2D - 128 Filters
3 × 3 Kernel
ReLU
       │
Batch Normalization
       │
Max Pooling 2 × 2
       │
       ▼
Dropout - 0.3
       │
Flatten
       │
Dense - 128
ReLU
       │
Dropout - 0.3
       │
Dense - 1
Sigmoid
       │
       ▼
NORMAL / PNEUMONIA
```

### Model Characteristics

- **4 convolutional blocks**
- Filters: **16 → 32 → 64 → 128**
- Kernel size: **3 × 3**
- Activation: **ReLU**
- Batch Normalization
- Max Pooling
- Dropout: **0.3**
- Dense layer: **128 units**
- Output layer: **1 unit**
- Output activation: **Sigmoid**

---

## ⚙️ Model Configuration

| Component | Configuration |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | TensorFlow / Keras |
| TensorFlow Version | 2.20.0 |
| Model Type | Sequential CNN |
| Input Shape | 150 × 150 × 1 |
| Classification | Binary |
| Output Activation | Sigmoid |
| Loss Function | Binary Crossentropy |
| Optimizer | Adam |
| Learning Rate | 0.0005 |
| Metrics | Accuracy, AUC |
| Dropout | 0.3 |
| Maximum Epochs | 25 |
| Batch Size | 32 |
| Class Weighting | Balanced |
| Model Checkpoint | Best `val_loss` |
| Early Stopping | Patience = 5 |
| ReduceLROnPlateau | Factor = 0.5 |
| Minimum Learning Rate | 1e-6 |

---

## 🚦 Training Strategy

The model is trained for up to **25 epochs** using the augmented training generator and validation generator.

### ModelCheckpoint

Saves the best-performing model based on validation loss:

```text
saved_model/pneumonia_model.best.keras
```

### EarlyStopping

Configuration:

```text
patience = 5
restore_best_weights = True
```

### ReduceLROnPlateau

Configuration:

```text
factor = 0.5
patience = 2
min_lr = 1e-6
```

Class weights are supplied to `model.fit()` to address training-set imbalance.

---

## 📊 Model Performance

The trained CNN was evaluated on the held-out **624-image test set**.

### Test Results

| Metric | Score |
|---|---:|
| **Test Accuracy** | **91.67%** |
| **Test AUC** | **0.9646** |
| **ROC-AUC** | **0.9661** |
| Test Loss | **0.2559** |

### Training Results

| Metric | Score |
|---|---:|
| Training Accuracy | **97.60%** |
| Training AUC | **0.9962** |
| Training Loss | **0.0667** |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| NORMAL | 0.90 | 0.88 | 0.89 | 234 |
| PNEUMONIA | 0.93 | 0.94 | 0.93 | 390 |
| **Accuracy** | | | **0.92** | **624** |
| Macro Average | 0.91 | 0.91 | 0.91 | 624 |
| Weighted Average | 0.92 | 0.92 | 0.92 | 624 |

### 📈 ROC-AUC

The model achieved:

```text
ROC-AUC = 0.9661
```

This indicates strong discrimination between the Normal and Pneumonia classes on this test set.

### 🔍 Performance Observations

- Achieved **91.67% test accuracy**.
- Achieved **0.9646 test AUC**.
- Achieved **0.9661 ROC-AUC**.
- Pneumonia recall reached **94%**.
- Pneumonia F1-score reached **0.93**.
- Normal recall reached **88%**.
- Normal F1-score reached **0.89**.
- Training accuracy was **97.60%**, compared with **91.67%** on the test set.
- Class weighting was used to address the training-set imbalance.
- Data augmentation and Batch Normalization were used during model development.

> **Important:** These results are specific to this project's dataset and test split. They do not represent clinical validation or real-world diagnostic performance.

---

## 📉 Training & Evaluation

The notebook generates visualizations for:

- Training vs Validation Accuracy
- Training vs Validation Loss
- Training vs Testing Accuracy
- Training vs Testing Loss
- Confusion Matrix
- ROC Curve
- Classification Report

These visualizations help analyze model learning, generalization, classification errors, and discrimination performance.

---

## 🧮 Confusion Matrix

The notebook generates a confusion matrix using the test-set predictions.

It compares:

```text
Actual Classes
      ↓
NORMAL / PNEUMONIA

        vs

Predicted Classes
      ↓
NORMAL / PNEUMONIA
```

This provides visibility into:

- True Negatives
- True Positives
- False Positives
- False Negatives

---

## 📈 ROC Curve

The ROC curve is generated using:

- False Positive Rate (FPR)
- True Positive Rate (TPR)
- Prediction probabilities from the CNN

The notebook reports:

```text
ROC-AUC = 0.9661
```

The ROC curve evaluates model performance across different classification thresholds.

---

## 🧪 Single Image Prediction

The notebook includes a single-image prediction workflow.

### Prediction Pipeline

```text
Input Chest X-Ray
       ↓
Open Image
       ↓
Convert to Grayscale
       ↓
Resize to 150 × 150
       ↓
Normalize Pixel Values
       ↓
CNN Model
       ↓
Sigmoid Probability
       ↓
Threshold > 0.5
       ↓
NORMAL / PNEUMONIA
```

### Example Prediction

```text
Prediction: PNEUMONIA
Probability: 0.xx
```

The notebook contains examples for predicting both Pneumonia and Normal test images.

> Individual prediction probabilities are model outputs for this dataset and should not be interpreted as clinical confidence scores.

---

## 🛠️ Technologies Used

### Programming

- Python

### Deep Learning

- TensorFlow
- Keras
- Convolutional Neural Networks

### Data Processing

- NumPy
- Pandas
- Pillow
- Scikit-learn

### Visualization

- Matplotlib
- Seaborn

### Development Environment

- Google Colab
- GPU-based training

### Dataset

- Kaggle Chest X-Ray Images (Pneumonia)

---

## 📁 Repository Structure

```text
Pneumonia-Detection-Using-CNN/
│
├── Pneumonia_Detection_Using_CNN.ipynb
│
├── README.md
│
└── saved_model/
    └── pneumonia_model.best.keras
```

> The trained model file may not be included in the repository unless it has been uploaded separately.

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Sreeharistack/Pneumonia-Detection-Using-CNN.git
cd Pneumonia-Detection-Using-CNN
```

### 2. Open the Notebook

Open:

```text
Pneumonia_Detection_Using_CNN.ipynb
```

The notebook can be run using **Google Colab** with GPU acceleration.

### 3. Install Dependencies

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn pillow scikit-learn kaggle
```

### 4. Configure Kaggle API

The notebook uses the Kaggle API to download the dataset.

Configure your Kaggle API credentials before running the dataset download section.

### 5. Run the Notebook

Run the notebook cells sequentially:

```text
Dataset Setup
      ↓
Dataset Verification
      ↓
Exploratory Data Analysis
      ↓
Preprocessing
      ↓
Data Augmentation
      ↓
Class Weight Calculation
      ↓
CNN Construction
      ↓
Model Compilation
      ↓
Training
      ↓
Evaluation
      ↓
Visualization
      ↓
Single Image Prediction
```

---

## ▶️ Open in Google Colab

Open the notebook directly in Google Colab:

https://colab.research.google.com/github/Sreeharistack/Pneumonia-Detection-Using-CNN/blob/main/Pneumonia_Detection_Using_CNN.ipynb

---

## 💡 Key Highlights

- ✅ Custom CNN built from scratch
- ✅ Binary chest X-ray classification
- ✅ Grayscale image processing
- ✅ 150 × 150 image input
- ✅ Four convolutional blocks
- ✅ Batch Normalization
- ✅ Max Pooling
- ✅ Dropout regularization
- ✅ Image augmentation
- ✅ Balanced class weighting
- ✅ Adam optimizer
- ✅ Binary Crossentropy
- ✅ Accuracy and AUC monitoring
- ✅ ModelCheckpoint
- ✅ EarlyStopping
- ✅ ReduceLROnPlateau
- ✅ Confusion Matrix
- ✅ Classification Report
- ✅ ROC Curve
- ✅ ROC-AUC evaluation
- ✅ Single-image inference
- ✅ Google Colab workflow
- ✅ Kaggle API dataset integration

---

## 🔍 Skills Demonstrated

This project demonstrates practical experience with:

- Python
- Machine Learning
- Deep Learning
- Computer Vision
- Medical Image Classification
- Convolutional Neural Networks
- TensorFlow
- Keras
- Image Preprocessing
- Image Augmentation
- Exploratory Data Analysis
- Class Imbalance Handling
- Model Training
- Model Evaluation
- Hyperparameter Configuration
- Model Regularization
- Callbacks
- ROC-AUC Analysis
- Confusion Matrix
- Classification Report
- Single Image Prediction
- Google Colab
- Kaggle API

---

## ⚠️ Limitations

- The training dataset is imbalanced toward Pneumonia.
- The validation set contains only **16 images**, which is very small.
- The model is trained and evaluated on a specific dataset and test split.
- Performance may vary on images from different hospitals, devices, populations, and imaging protocols.
- The model has not been clinically validated.
- External validation has not been performed.
- The model should not be used as a replacement for a radiologist or qualified healthcare professional.
- The reported metrics should not be interpreted as clinical diagnostic performance.

---

## 🔮 Future Improvements

Potential improvements include:

- Use a larger and more representative validation set.
- Perform systematic hyperparameter tuning.
- Compare class weighting with other imbalance-handling approaches.
- Experiment with transfer learning.
- Compare architectures such as ResNet, DenseNet, EfficientNet, and MobileNet.
- Implement Grad-CAM or other explainability methods.
- Perform external validation on an independent dataset.
- Add experiment tracking.
- Implement robust cross-validation strategies where appropriate.
- Build a Streamlit prediction interface.
- Develop an inference API.
- Deploy the model as an educational web application.
- Add automated testing and reproducible environment configuration.

---

## 📚 Project Information

| Item | Details |
|---|---|
| Project | Pneumonia Detection Using CNN |
| Domain | Deep Learning / Computer Vision |
| Task | Binary Image Classification |
| Input | Chest X-Ray Image |
| Image Format | Grayscale |
| Input Size | 150 × 150 × 1 |
| Classes | NORMAL / PNEUMONIA |
| Model | Custom Convolutional Neural Network |
| Framework | TensorFlow / Keras |
| Environment | Google Colab |
| Test Accuracy | **91.67%** |
| Test AUC | **0.9646** |
| ROC-AUC | **0.9661** |

---

## 👨‍💻 Author

### Sreehari.P

**Aspiring Data Scientist | Machine Learning | Deep Learning | Computer Vision**

**GitHub:**  
https://github.com/Sreeharistack

## ⚕️ Medical Disclaimer

This project is intended strictly for **educational and research purposes**.

It is **not a medical diagnostic tool** and has not been clinically validated.

The model should not be used for:

- Clinical diagnosis
- Treatment decisions
- Patient care
- Emergency medical decisions

Any medical decision should be made by a qualified healthcare professional.

---

## ⭐ Support

If you found this project useful for learning about:

- Machine Learning
- Deep Learning
- CNN
- Computer Vision
- Medical Image Classification
- TensorFlow/Keras

please consider giving this repository a ⭐ on GitHub.

---

## 📌 Repository

**Pneumonia Detection Using CNN**

https://github.com/Sreeharistack/Pneumonia-Detection-Using-CNN
