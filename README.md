# 🐱🐶 Cats vs. Dogs Image Classification using ResNet-50

This project uses **deep learning and transfer learning** to classify images of cats and dogs with high accuracy. Leveraging a pre-trained **ResNet-50** model, the solution achieves strong performance on the **Kaggle Dogs vs. Cats** dataset.

> 📎 **Explore the full notebook on Kaggle**
> [Cats & Dogs Image Classification | ResNet-50](https://www.kaggle.com/code/lykin22/cats-dogs-image-classification-resnet-50)

---

## 📁 Project Structure

```
cats-dogs-resnet50/
├── data/                   # Training and validation images
├── notebooks/
│   └── cats_dogs_resnet50.ipynb   # Main training + evaluation notebook
├── models/                 # (Optional) Saved model weights
├── outputs/                # Logs, plots, and predictions
├── requirements.txt        # Project dependencies
└── README.md               # Project overview
```

---

## 🧠 Model Summary

* **Architecture:** ResNet-50 (pre-trained on ImageNet)
* **Customizations:** Replaced top classification layer with a dense layer for binary classification
* **Loss Function:** Binary Crossentropy
* **Optimizer:** Adam
* **Primary Metric:** Accuracy

---

## 🧪 Workflow Overview

### 1. 📦 Data Preparation

* Downloaded the dataset from Kaggle: [Dogs vs. Cats](https://www.kaggle.com/competitions/dogs-vs-cats/data)
* Resized images to `224x224` to fit ResNet-50 input requirements
* Applied data augmentation and created training/validation splits

### 2. 🏗️ Model Construction

* Loaded `ResNet50` with `include_top=False` using Keras
* Added custom dense layers for binary classification
* Compiled with the Adam optimizer and binary crossentropy loss

### 3. 🚂 Training

* Used callbacks for early stopping and model checkpointing
* Trained over several epochs with real-time augmentation

### 4. 📈 Evaluation

* Evaluated on the validation set
* Generated confusion matrix and classification report
* Visualized sample predictions

---

## 📊 Results

* ✅ **Achieved \~98% validation accuracy**
* 🔍 Model generalizes well across a variety of images
* 📸 Visualizations show strong performance across both classes

---

## 📦 Requirements

To run this project, install the following dependencies:

* TensorFlow / Keras
* NumPy
* Matplotlib
* scikit-learn
* Pillow (PIL)
* tqdm
* Jupyter Notebook

You can install them using:

```bash
pip install -r requirements.txt
```

---

## 🚀 Future Enhancements

* Fine-tune hyperparameters for further optimization
* Try advanced architectures like **EfficientNet** or **DenseNet**
* Deploy using **Flask** or **Streamlit** for interactive demos

---

## 📚 References

* 🐕 [Kaggle Dogs vs. Cats Dataset](https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition)
* 📄 [ResNet-50 Original Paper](https://arxiv.org/abs/1512.03385)

