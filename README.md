# 🫁 Pneumonia Detection Using CNN

A deep learning project for classifying chest X-ray images into **NORMAL** and **PNEUMONIA** using a custom **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras**.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)
![CNN](https://img.shields.io/badge/Model-CNN-purple)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Medical%20Imaging-green)
![Google Colab](https://img.shields.io/badge/Google%20Colab-GPU-yellow?logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

Pneumonia is a respiratory infection that can affect one or both lungs. Chest X-ray imaging is commonly used as part of the diagnostic process.

This project demonstrates how **deep learning and computer vision** can be applied to classify chest X-ray images into two categories:

* 🟢 **NORMAL**
* 🔴 **PNEUMONIA**

A custom CNN architecture was developed using **TensorFlow/Keras** and trained on grayscale chest X-ray images.

The project covers the complete machine learning workflow:

**Dataset Preparation → EDA → Preprocessing → Augmentation → Class Weighting → CNN Training → Evaluation → Visualization → Single-Image Prediction**

> ⚠️ **Important:** This project is intended for educational and research purposes only. It is not a clinically validated medical diagnostic system.

---

## 🎯 Project Objectives

* Build a custom CNN for binary image classification.
* Classify chest X-ray images as NORMAL or PNEUMONIA.
* Preprocess and normalize medical images.
* Apply image augmentation to improve generalization.
* Analyze class distribution and imbalance.
* Apply class weighting during training.
* Train and evaluate a custom CNN.
* Monitor Accuracy and AUC during training.
* Analyze performance using a Confusion Matrix.
* Evaluate classification using ROC-AUC.
* Generate a Classification Report.
* Perform single-image prediction.
* Visualize training, validation, and testing performance.

---

# 📊 Dataset

The project uses the **Chest X-Ray Images (Pneumonia)** dataset.

The dataset contains approximately **5,863 chest X-ray images** organized into training, validation, and testing sets.

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

| Class     | Description                         |
| --------- | ----------------------------------- |
| NORMAL    | Chest X-ray classified as normal    |
| PNEUMONIA | Chest X-ray classified as pneumonia |

The dataset contains class imbalance, so **class weighting** was incorporated during model training.

---

# 📊 Dataset Class Distribution

The class distribution was analyzed before training to understand the balance between NORMAL and PNEUMONIA images.

![Dataset Class Distribution](images/dataset_class_distribution.png)

---

# 🖼️ Sample Chest X-Ray Images

Sample images were visualized to understand the appearance, variation, and characteristics of the dataset.

![Sample Chest X-Ray Images](images/sample_pneumonia_xray_images.png)

---

# 🔄 Machine Learning Workflow

```text
Chest X-Ray Dataset
        │
        ▼
Dataset Download
        │
        ▼
Dataset Verification
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Class Distribution Analysis
        │
        ▼
Image Visualization
        │
        ▼
Image Preprocessing
        │
        ▼
Image Normalization
        │
        ▼
Data Augmentation
        │
        ▼
Class Weight Calculation
        │
        ▼
CNN Architecture
        │
        ▼
Model Compilation
        │
        ▼
Model Training
        │
        ▼
Early Stopping
        │
        ▼
Learning Rate Reduction
        │
        ▼
Model Checkpointing
        │
        ▼
Model Evaluation
        │
        ▼
Confusion Matrix
        │
        ▼
ROC-AUC Analysis
        │
        ▼
Classification Report
        │
        ▼
Single Image Prediction
```

---

# 🖼️ Image Preprocessing

The chest X-ray images are processed before being passed to the CNN.

### Preprocessing Steps

* Image resizing
* Conversion to grayscale
* Pixel normalization
* Dataset verification
* Image visualization
* Training data preparation
* Validation data preparation
* Test data preparation

### Input Shape

```text
150 × 150 × 1
```

The model uses grayscale chest X-ray images with normalized pixel values.

---

# 🔁 Data Augmentation

Data augmentation was applied to increase training-image diversity and improve model generalization.

The augmentation pipeline includes:

* Rotation
* Zoom
* Shearing
* Horizontal flipping

These transformations introduce variations in the training data and help reduce overfitting.

---

# 🧠 CNN Architecture

A custom **Convolutional Neural Network** was developed using TensorFlow/Keras.

### Architecture

```text
Input
150 × 150 × 1
        │
        ▼
Conv2D - 16 Filters
        │
Batch Normalization
        │
Max Pooling
        │
        ▼
Conv2D - 32 Filters
        │
Batch Normalization
        │
Max Pooling
        │
        ▼
Conv2D - 64 Filters
        │
Batch Normalization
        │
Max Pooling
        │
        ▼
Conv2D - 128 Filters
        │
Batch Normalization
        │
Max Pooling
        │
        ▼
Flatten
        │
        ▼
Dense - 128 Neurons
        │
ReLU
        │
        ▼
Dropout - 0.3
        │
        ▼
Dense - 1 Neuron
        │
Sigmoid
        │
        ▼
NORMAL / PNEUMONIA
```

---

# ⚙️ Model Configuration

| Component               | Configuration         |
| ----------------------- | --------------------- |
| Language                | Python                |
| Framework               | TensorFlow / Keras    |
| Model                   | Sequential CNN        |
| Input Shape             | 150 × 150 × 1         |
| Task                    | Binary Classification |
| Output Activation       | Sigmoid               |
| Loss Function           | Binary Crossentropy   |
| Optimizer               | Adam                  |
| Learning Rate           | 0.0005                |
| Metrics                 | Accuracy, AUC         |
| Dropout                 | 0.3                   |
| Maximum Epochs          | 25                    |
| Batch Normalization     | Yes                   |
| Data Augmentation       | Yes                   |
| Class Weighting         | Yes                   |
| Early Stopping          | Yes                   |
| Learning Rate Reduction | Yes                   |
| Model Checkpointing     | Yes                   |

---

# 🏋️ Model Training

The CNN was trained for up to **25 epochs**.

The following metrics were monitored:

* Training Accuracy
* Validation Accuracy
* Training Loss
* Validation Loss
* Training AUC
* Validation AUC

Training incorporated:

* Class weighting
* Data augmentation
* Batch normalization
* Dropout
* Early stopping
* Learning-rate reduction
* Model checkpointing

---

# 📈 Training & Validation Curves

Training and validation curves were generated to analyze the learning behavior of the CNN.

![Training and Validation Curves](images/training_validation_curves.png)

---

# 📊 Training, Validation & Testing Performance

Training, validation, and testing performance were compared to analyze model learning and generalization.

![Training Validation Test Curves](images/training_validation_test_curves.png)

---

# 📊 Model Performance

The trained CNN was evaluated using a held-out test dataset containing **624 chest X-ray images**.

## Test Performance

| Metric        |      Score |
| ------------- | ---------: |
| Test Accuracy | **91.67%** |
| Test AUC      | **0.9646** |
| ROC-AUC       | **0.9661** |
| Test Loss     | **0.2559** |

### Training Performance

| Metric            |      Score |
| ----------------- | ---------: |
| Training Accuracy | **97.60%** |
| Training AUC      | **0.9962** |
| Training Loss     | **0.0667** |

The independently calculated ROC-AUC was **0.9661**, while the AUC reported during model evaluation was **0.9646**.

---

# 📋 Classification Report

Performance on the held-out test dataset:

| Class                | Precision |   Recall | F1-Score | Support |
| -------------------- | --------: | -------: | -------: | ------: |
| NORMAL               |      0.90 |     0.88 |     0.89 |     234 |
| PNEUMONIA            |      0.93 |     0.94 |     0.93 |     390 |
| **Accuracy**         |           |          | **0.92** | **624** |
| **Macro Average**    |  **0.91** | **0.91** | **0.91** | **624** |
| **Weighted Average** |  **0.92** | **0.92** | **0.92** | **624** |

---

# 📈 ROC Curve

The model achieved a **ROC-AUC score of 0.9661** on the test dataset, indicating strong discrimination between the two classes.

![ROC Curve](images/roc_curve.png)

---

# 🔲 Confusion Matrix

The confusion matrix provides a detailed view of the model's classification performance, including:

* True Positives
* True Negatives
* False Positives
* False Negatives

![Confusion Matrix](images/confusion_matrix.png)

---

# 📉 Training vs Testing Metrics

Training and testing metrics were compared to evaluate the model's generalization performance.

![Training vs Testing Metrics](images/training_vs_testing_metrics.png)

---

# 🔍 Performance Summary

| Metric              |     Result |
| ------------------- | ---------: |
| Test Accuracy       | **91.67%** |
| Test AUC            | **0.9646** |
| ROC-AUC             | **0.9661** |
| Training Accuracy   | **97.60%** |
| Training AUC        | **0.9962** |
| Pneumonia Precision |   **0.93** |
| Pneumonia Recall    |   **0.94** |
| Pneumonia F1-Score  |   **0.93** |
| Normal Precision    |   **0.90** |
| Normal Recall       |   **0.88** |
| Normal F1-Score     |   **0.89** |
| Test Images         |    **624** |

---

# 🔍 Key Findings

* ✅ Test accuracy reached **91.67%**.
* ✅ ROC-AUC reached **0.9661**.
* ✅ Pneumonia recall reached **94%**.
* ✅ Pneumonia F1-score reached **0.93**.
* ✅ Normal recall reached **88%**.
* ✅ Normal F1-score reached **0.89**.
* ✅ Class weighting was applied.
* ✅ Data augmentation was applied.
* ✅ Batch normalization was used.
* ✅ Dropout regularization was used.
* ✅ Early stopping was implemented.
* ✅ Learning-rate reduction was implemented.
* ✅ Model checkpointing was implemented.

The training accuracy of **97.60%** compared with test accuracy of **91.67%** indicates a generalization gap that could be investigated through additional regularization, external validation, and transfer-learning approaches.

> **Note:** These results are based on the project's held-out test dataset and should not be interpreted as clinical performance.

---

# 🧪 Single Image Prediction

The trained model can classify an individual chest X-ray image.

### Prediction Pipeline

```text
Input Chest X-Ray
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
Threshold = 0.5
        ↓
NORMAL / PNEUMONIA
```

### Example

```text
Prediction: PNEUMONIA
Probability: 0.xx
```

The notebook contains examples demonstrating individual image prediction.

---

# 🔬 Technical Highlights

* Custom CNN architecture
* Four convolutional blocks
* 16, 32, 64, and 128 filters
* Batch Normalization
* Max Pooling
* Flatten layer
* Dense layer with 128 neurons
* ReLU activation
* Dropout regularization
* Sigmoid output
* Binary Crossentropy
* Adam optimizer
* Data augmentation
* Class weighting
* Early stopping
* Learning-rate reduction
* Model checkpointing
* ROC-AUC evaluation
* Confusion Matrix
* Classification Report
* Single-image inference

---

# 🛠️ Technologies Used

### Programming

* Python

### Deep Learning

* TensorFlow
* Keras
* Convolutional Neural Networks

### Data Processing

* NumPy
* Pandas
* Pillow

### Visualization

* Matplotlib
* Seaborn

### Development

* Google Colab
* GPU-based training

### Dataset Access

* Kaggle API

---

# 📁 Repository Structure

```text
Pneumonia-Detection-Using-CNN/
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
│
├── Pneumonia_Detection_Using_CNN.ipynb
│
└── images/
    │
    ├── dataset_class_distribution.png
    ├── sample_pneumonia_xray_images.png
    ├── training_validation_curves.png
    ├── training_validation_test_curves.png
    ├── training_vs_testing_metrics.png
    ├── confusion_matrix.png
    └── roc_curve.png
```

---

# 🚀 How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/Sreeharistack/Pneumonia-Detection-Using-CNN.git
```

## 2. Navigate to the Project

```bash
cd Pneumonia-Detection-Using-CNN
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install the required packages manually:

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn pillow kaggle
```

## 4. Open the Notebook

Open:

```text
Pneumonia_Detection_Using_CNN.ipynb
```

The notebook can be executed using **Google Colab** with GPU acceleration.

## 5. Configure Kaggle

The notebook uses the Kaggle API to download the dataset.

Configure your Kaggle API credentials before running the dataset download section.

## 6. Execute the Notebook

Run the notebook sequentially:

```text
Dataset Setup
      ↓
Data Exploration
      ↓
Dataset Verification
      ↓
Visualization
      ↓
Preprocessing
      ↓
Augmentation
      ↓
Class Weight Calculation
      ↓
CNN Construction
      ↓
Model Compilation
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Performance Analysis
      ↓
Single Image Prediction
```

---

# 📚 Skills Demonstrated

This project demonstrates practical experience in:

* Python
* Machine Learning
* Deep Learning
* Computer Vision
* TensorFlow
* Keras
* CNN
* Image Classification
* Medical Image Analysis
* Image Preprocessing
* Image Normalization
* Data Augmentation
* Exploratory Data Analysis
* Class Imbalance Handling
* Class Weighting
* Model Evaluation
* Model Regularization
* Hyperparameter Configuration
* ROC-AUC Analysis
* Confusion Matrix
* Classification Report
* Google Colab
* Kaggle API

---

# ⚠️ Limitations

This is an **educational and research project**.

Limitations include:

* Dataset class imbalance.
* Relatively small validation dataset.
* Model performance may vary on external datasets.
* Images from different hospitals or imaging systems may have different characteristics.
* The model has not been clinically validated.
* External generalization has not been established.
* The model should not replace professional medical evaluation.

---

# 🔮 Future Improvements

Potential improvements include:

* Transfer learning using pretrained CNN architectures.
* Experiment with ResNet.
* Experiment with DenseNet.
* Experiment with EfficientNet.
* Experiment with MobileNet.
* Hyperparameter tuning.
* Improved validation strategies.
* External dataset validation.
* Grad-CAM explainability.
* Model calibration.
* More extensive error analysis.
* Experiment tracking.
* Streamlit deployment.
* REST API deployment.
* Web-based prediction interface.
* Model monitoring.

---

# 📦 Project Files

### 📓 Notebook

```text
Pneumonia_Detection_Using_CNN.ipynb
```

Contains:

* Dataset preparation
* Data exploration
* Dataset verification
* Preprocessing
* Data augmentation
* CNN construction
* Model training
* Model evaluation
* Visualization
* Single-image prediction

### 📋 Requirements

```text
requirements.txt
```

Contains the Python dependencies required for the project.

### 🖼️ Visualizations

The `images/` folder contains:

```text
dataset_class_distribution.png
sample_pneumonia_xray_images.png
training_validation_curves.png
training_validation_test_curves.png
training_vs_testing_metrics.png
confusion_matrix.png
roc_curve.png
```

---

# 👨‍💻 Author

## Shaam Shhanu

**Aspiring Data Scientist | Machine Learning | Deep Learning | Computer Vision**

### GitHub

https://github.com/Sreeharistack

### LinkedIn

Add your LinkedIn profile URL here.

---

# 📄 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for details.

---

# ⚕️ Medical Disclaimer

This project is intended **strictly for educational and research purposes**.

It is **not a medical diagnostic tool** and should not be used for:

* Clinical diagnosis
* Treatment decisions
* Patient care
* Medical decision-making

The model has not been clinically validated.

Medical decisions should always be made by qualified healthcare professionals.

---

# ⭐ Support

If you found this project useful, consider giving the repository a ⭐ on GitHub.

Your support is appreciated!

---

# 🙌 Thank You

Thank you for visiting this project!

Feel free to explore the notebook, review the visualizations, and contribute improvements.

**Built with Python, TensorFlow/Keras, and Deep Learning.**
