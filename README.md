# 🚗 Vehicle Classification using EfficientNet (7 Classes)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Project Overview

This project focuses on multi-class image classification of vehicles using deep learning and transfer learning techniques.

The goal is to correctly classify images into one of seven categories:

* Cars
* Bikes
* Motorcycles
* Planes
* Ships
* Trains
* Auto Rickshaws

The model is built using a pretrained EfficientNet architecture and fine-tuned for the task.

---

## 📊 Dataset

Dataset from Kaggle:

https://www.kaggle.com/datasets/mohamedmaher5/vehicle-classification

### Key Characteristics:

* ~800 images per class
* ~5500+ total images
* Balanced dataset
* Size: ~1GB

⚠️ Dataset is not included in this repository due to size limitations.

---

## 🧠 Model Architecture

### Base Model:

* EfficientNetB0 (pretrained on ImageNet)

### Custom Head:

* GlobalAveragePooling
* Dropout (0.3)
* Dense layer (Softmax)

### Training Strategy:

1. Freeze base model → train classifier
2. Partial fine-tuning (last layers)

---

## 📈 Results

| Metric              | Value  |
| ------------------- | ------ |
| Training Accuracy   | 99.9%  |
| Validation Accuracy | 99.36% |

---

## ⚠️ Critical Observations

High accuracy may be influenced by:

* Dataset simplicity
* Visual differences between classes
* Potential similarity across splits

This highlights the importance of evaluating models on more diverse datasets.

---

## ⚙️ Project Structure

```
project/
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
│
├── data/              # not included
├── models/            # ignored
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Download dataset

Place it in:

```
data/
```

### 3. Train model

```
python src/train.py
```

### 4. Run inference

```
python src/predict.py
```

---

## 🔮 Example Predictions

| Input Image | Prediction  |
| ----------- | ----------- |
| Train       | Trains      |
| Car         | Cars        |
| Motorcycle  | Motorcycles |

---

## 💡 Key Features

* Transfer learning with EfficientNet
* Clean modular pipeline
* High classification accuracy
* Robust preprocessing

---

## 🚧 Future Improvements

* Data augmentation
* Cross-validation
* Larger and more diverse dataset
* EfficientNetB3/B4 upgrade
* Model deployment (API / UI)

---

## 📌 Notes

* Model weights are not included due to size limitations
* The project is fully reproducible via training script

---

## 👨‍💻 Author

https://github.com/Zh09-hak
