# Neural Network & Deep Learning Projects

Course projects from NNDL (Neural Networks and Deep Learning), Amrita Vishwa Vidyapeetham. Both models built and trained from scratch using TensorFlow/Keras — no pretrained weights.

---

## Projects

### 1. Butterfly Species CNN Classifier
**File:** `butterfly-species-classification-simple-cnn.ipynb`

Multi-class image classification across 75 butterfly species using a convolutional neural network built from scratch.

**Architecture**
- Input: 150×150×3 RGB images
- Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool → Flatten → Dense(512) → Softmax(75)
- Optimizer: Adam | Loss: Categorical Crossentropy

**Data Pipeline**
- Augmentation: rotation (±40°), width/height shift, shear, zoom, horizontal flip via `ImageDataGenerator`
- 80/20 train-validation split
- Batch size: 32 | Epochs: 40

**Results**

| Metric | Value |
|--------|-------|
| Validation Accuracy | **80%** |
| Training Epochs | 40 |
| Classes | 75 |
| Image Size | 150×150 |

**Stack:** Python, TensorFlow, Keras, NumPy, Pandas, Scikit-Learn, Matplotlib

---

### 2. Cardiotocography (CTG) Clinical MLP Classifier
**File:** `CB.EN.U4CSE22026_NNDL_E1_S1.ipynb`

3-class fetal health outcome classifier (Normal / Suspect / Pathological) trained on the UCI Cardiotocography dataset. Demonstrates end-to-end MLP design: preprocessing, architecture search, dropout regularization, and evaluation.

**Architecture**
- Dense(30, ReLU) → Dropout(0.2) → Dense(15, ReLU) → Dropout(0.2) → Dense(7, ReLU) → Dropout(0.2) → Dense(3, Softmax)
- Optimizer: Adam | Loss: Sparse Categorical Crossentropy

**Preprocessing**
- Dropped non-predictive administrative columns
- StandardScaler normalization
- LabelEncoder for target classes
- 80/20 train-test split

**Results**

| Metric | Value |
|--------|-------|
| Train Accuracy | **96.3%** |
| Validation Accuracy | **97.1%** |
| Training Epochs | 100 |
| Input Features | 22 |
| Classes | 3 |

**Stack:** Python, TensorFlow, Keras, Scikit-Learn, Pandas, NumPy, Matplotlib

---

## Key Concepts Demonstrated

- CNN architecture design (Conv → Pool → Dense stacks)
- Data augmentation for image generalization
- MLP regularization via Dropout
- Multi-class classification (categorical and sparse)
- Feature preprocessing: scaling, encoding, null handling
- Training history visualization (loss/accuracy curves)
- Activation functions from scratch: ReLU, Leaky ReLU (with derivatives)

---

## Setup

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
jupyter notebook
```

Open either `.ipynb` file. The butterfly notebook expects the dataset in `./butterfly/` (train and test subdirectories with image files + `Training_set.csv`). The CTG notebook expects `CTG.csv` in the working directory.

---

## Environment

- Python 3.10+
- TensorFlow 2.x
- Trained on Google Colab (GPU)
