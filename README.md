# 🫁 Pneumonia Detection Using CNN

> A deep learning project for classifying chest X-ray images into **Normal** and **Pneumonia** using a Convolutional Neural Network (CNN).

---

## 📌 Overview

Pneumonia is a serious respiratory infection that can affect one or both lungs. Chest X-rays are commonly used during the diagnostic process, but interpreting medical images can be challenging and time-consuming.

This project explores the use of **Convolutional Neural Networks (CNNs)** for automated classification of chest X-ray images.

The model classifies chest X-ray images into two categories:

- 🟢 **NORMAL**
- 🔴 **PNEUMONIA**

The project covers the complete deep learning workflow, including dataset preparation, exploratory data analysis, image preprocessing, data augmentation, CNN model development, training, and evaluation.

---

## 🎯 Objectives

- Build a CNN-based binary image classification model.
- Classify chest X-ray images as Normal or Pneumonia.
- Preprocess and normalize medical images.
- Apply image augmentation to improve model generalization.
- Analyze dataset structure and class distribution.
- Train a custom CNN architecture.
- Evaluate the model using multiple performance metrics.
- Explore the application of deep learning in medical image classification.

---

## 📊 Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset.

The dataset contains approximately **5,863 chest X-ray images** organized into training, validation, and test sets.

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

### Classes

| Class | Description |
|---|---|
| NORMAL | Chest X-ray without pneumonia |
| PNEUMONIA | Chest X-ray showing pneumonia |

The dataset contains an imbalanced distribution between the two classes, which is considered during the model development and evaluation process.

---

## 🔄 Project Workflow

```text
Chest X-Ray Dataset
        ↓
Dataset Download
        ↓
Dataset Verification
        ↓
Exploratory Data Analysis
        ↓
Class Distribution Analysis
        ↓
Image Visualization
        ↓
Image Preprocessing
        ↓
Data Augmentation
        ↓
CNN Architecture
        ↓
Model Compilation
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Normal / Pneumonia Prediction
```

---

## 🖼️ Image Preprocessing

The chest X-ray images are prepared before being provided to the CNN model.

The preprocessing workflow includes:

- Image resizing
- Image normalization
- Dataset verification
- Image visualization
- Training and validation data preparation

### Input Image Size

```text
150 × 150 × 1
```

The model uses grayscale chest X-ray images as input.

---

## 🔁 Data Augmentation

Data augmentation is applied to increase the diversity of training images and improve model generalization.

The augmentation pipeline includes:

- Rotation
- Zoom
- Shearing
- Horizontal flipping

These transformations create variations of training images and can help reduce overfitting.

---

## 🧠 CNN Architecture

A custom **Convolutional Neural Network** is implemented using TensorFlow/Keras.

### Architecture

```text
Input
150 × 150 × 1
        │
        ▼
Conv2D
16 Filters
        │
Batch Normalization
        │
Max Pooling
        │
        ▼
Conv2D
32 Filters
        │
Batch Normalization
        │
Max Pooling
        │
        ▼
Conv2D
64 Filters
        │
Batch Normalization
        │
Max Pooling
        │
        ▼
Conv2D
128 Filters
        │
Batch Normalization
        │
Max Pooling
        │
        ▼
Flatten
        │
Dense - 128
ReLU Activation
        │
Dropout - 0.3
        │
        ▼
Dense - 1
Sigmoid Activation
        │
        ▼
NORMAL / PNEUMONIA
```

---

## ⚙️ Model Configuration

| Component | Configuration |
|---|---|
| Programming Language | Python |
| Framework | TensorFlow / Keras |
| Model | Sequential CNN |
| Input Shape | 150 × 150 × 1 |
| Classification | Binary |
| Output Activation | Sigmoid |
| Loss Function | Binary Crossentropy |
| Optimizer | Adam |
| Learning Rate | 0.0005 |
| Main Metrics | Accuracy, AUC |
| Dropout | 0.3 |
| Training Epochs | 25 |

---

## 📈 Model Training

The CNN model is trained for up to **25 epochs**.

During training, the following metrics are monitored:

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy
- Training AUC
- Validation AUC

A learning-rate adjustment strategy is also used during the training process.

---

## 📊 Model Evaluation

The model is evaluated using multiple performance metrics.

### Accuracy

Measures the percentage of correctly classified images.

### Precision

Measures how many images predicted as Pneumonia are actually Pneumonia.

### Recall

Measures how many actual Pneumonia cases are correctly identified.

### F1-Score

Provides a balance between Precision and Recall.

### Confusion Matrix

Provides a detailed view of:

- True Positives
- True Negatives
- False Positives
- False Negatives

### ROC-AUC

Measures the model's ability to distinguish between Normal and Pneumonia cases across different classification thresholds.

> **Note:** Final numerical performance results are intentionally not stated here unless verified directly from the notebook's final evaluation output.

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

### Visualization

- Matplotlib
- Seaborn

### Development Environment

- Google Colab
- GPU-based training

### Dataset

- Chest X-Ray Images (Pneumonia)

---

## 📁 Repository Structure

```text
Pneumonia-Detection-Using-CNN/
│
├── Pneumonia_Detection_Using_CNN.ipynb
│
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Sreeharistack/Pneumonia-Detection-Using-CNN.git
```

### 2. Open the Project

Open:

```text
Pneumonia_Detection_Using_CNN.ipynb
```

The notebook can be run using **Google Colab** with GPU acceleration.

### 3. Install Dependencies

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn pillow kaggle
```

### 4. Configure Kaggle

The notebook uses the Kaggle API to download the dataset.

You need to configure your Kaggle API credentials before running the dataset download section.

### 5. Run the Notebook

Run the notebook cells sequentially:

```text
Dataset Setup
      ↓
Data Exploration
      ↓
Preprocessing
      ↓
Data Augmentation
      ↓
CNN Construction
      ↓
Model Compilation
      ↓
Training
      ↓
Evaluation
```

---

## 💡 Key Highlights

- ✅ Binary chest X-ray classification
- ✅ Custom CNN architecture
- ✅ Four convolutional blocks
- ✅ Batch Normalization
- ✅ Max Pooling
- ✅ Dropout regularization
- ✅ Image augmentation
- ✅ Adam optimizer
- ✅ Binary Crossentropy loss
- ✅ Accuracy and AUC monitoring
- ✅ Confusion Matrix evaluation
- ✅ ROC-AUC evaluation
- ✅ Kaggle dataset integration
- ✅ GPU-based deep learning workflow

---

## 🔍 Skills Demonstrated

This project demonstrates practical experience with:

- Python
- Machine Learning
- Deep Learning
- Computer Vision
- CNN Architecture
- TensorFlow
- Keras
- Image Classification
- Data Preprocessing
- Data Augmentation
- Exploratory Data Analysis
- Model Evaluation
- Hyperparameter Configuration
- Model Regularization
- Google Colab
- Kaggle API

---

## ⚠️ Limitations

This project is an educational deep learning project and has several limitations:

- The dataset contains class imbalance.
- The validation dataset is relatively small.
- Model performance may vary on images from different hospitals or imaging systems.
- The model has not been clinically validated.
- Performance on external datasets may differ from the training/test dataset.
- The model should not be considered a replacement for professional medical diagnosis.

---

## 🔮 Future Improvements

Potential improvements include:

- Address class imbalance using suitable techniques.
- Improve validation strategy.
- Perform systematic hyperparameter tuning.
- Experiment with transfer learning.
- Compare architectures such as ResNet, DenseNet, EfficientNet, and MobileNet.
- Implement Grad-CAM for model explainability.
- Perform external validation using an independent dataset.
- Add model checkpointing.
- Add experiment tracking.
- Build a Streamlit prediction interface.
- Develop an API for model inference.
- Deploy the trained model as a web application.

---

## 📚 Project Information

| Item | Details |
|---|---|
| Project | Pneumonia Detection Using CNN |
| Domain | Deep Learning / Computer Vision |
| Task | Binary Image Classification |
| Input | Chest X-Ray Image |
| Classes | NORMAL / PNEUMONIA |
| Model | Convolutional Neural Network |
| Framework | TensorFlow / Keras |
| Environment | Google Colab |

---

## 👨‍💻 Author

### Sreehari.P

**Aspiring Data Scientist | Machine Learning | Deep Learning | Computer Vision**

**GitHub:**  
https://github.com/Sreeharistack

## ⚕️ Medical Disclaimer

This project is intended **strictly for educational and research purposes**.

It is **not a medical diagnostic tool** and should not be used for clinical diagnosis, treatment decisions, or patient care.

Any medical decision should be made by a qualified healthcare professional.

---

## ⭐ Support

If you found this project useful for learning about:

- Machine Learning
- Deep Learning
- CNN
- Computer Vision
- Medical Image Classification

please consider giving this repository a ⭐ on GitHub.
