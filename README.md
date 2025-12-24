# Cats vs Dogs Image Classification using CNN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Keras](https://img.shields.io/badge/Keras-CNN-red)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)
![Notebook](https://img.shields.io/badge/Made%20with-Jupyter-orange)
![Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-yellow)

This project implements a **Convolutional Neural Network (CNN)** to classify images of **cats and dogs**.  
The model is trained on labeled image data and learns visual patterns such as edges, textures, and shapes to make accurate predictions.

---

## Project Overview

- **Task:** Binary image classification (Cat vs Dog)
- **Approach:** Deep Learning using CNN
- **Framework:** TensorFlow / Keras
- **Platform:** Google Colab

---

## Model Architecture

The CNN architecture consists of:
- Convolutional layers for feature extraction
- MaxPooling layers for spatial downsampling
- Fully connected (Dense) layers for classification
- Sigmoid activation for binary output


---

## 📊 Dataset

- The dataset contains images of cats and dogs
- Images are resized and normalized before training
- Data augmentation is applied to improve generalization

> Dataset source can be Kaggle or any standard Cats vs Dogs dataset. I used this [Dataset](https://www.kaggle.com/datasets/d4rklucif3r/cat-and-dogs)

---

## How It Works

- Load and preprocess image data
- Build CNN model
- Train the model on training data
- Evaluate performance on validation data

## Results

- Achieves **73.55%** accuracy on validation data

#### Accuracy
![Accuracy and Validation Accuracy History](./images/accuracy.png)
#### Loss
![Loss and Validation Loss History](./images/loss.png)

> Model performance depends on dataset size and training epochs