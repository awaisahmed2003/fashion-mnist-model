# Fashion-MNIST CNN Classifier

This repository contains a high-accuracy Convolutional Neural Network (CNN) implemented using TensorFlow and Keras for classifying images from the Fashion-MNIST dataset into 10 clothing categories.

Fashion-MNIST is significantly more challenging than the original MNIST digit dataset due to visually similar classes such as shirts, pullovers, and coats. This project addresses those challenges through a carefully designed CNN architecture combined with modern deep learning techniques.

The model was developed as part of a course competition and achieves **~96%+ accuracy**, placing it near the top of the leaderboard while strictly following the competition’s data usage rules.

---

## Key Features

- Convolutional Neural Network optimized for 28×28 grayscale images  
- Data augmentation to improve generalization  
- Regularization using dropout and batch normalization  
- Learning rate scheduling with `ReduceLROnPlateau`  
- Test-Time Augmentation (TTA) for more stable predictions  
- Trained strictly on the provided dataset (no external data used)

---

## Dataset

- **Training set:** 40,000 labeled images  
- **Test set:** 30,000 unlabeled images  
- Image size: 28×28 grayscale  
- Number of classes: 10  

Dataset files (`train.csv`, `test.csv`) are **not included** in this repository due to competition rules and must be provided separately.

### Class Labels

| Label | Description |
|------|-------------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## Performance

- Training accuracy on full dataset: **~96%**
- Leaderboard accuracy: **~96%+**
- Exceeds the competition passing threshold of 91.5%

---

## Project Structure

├── MNIST-Classifier.py # End-to-end training and inference script
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── .gitignore # Excluded files and folders


---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/awaisahmed2003/fashion-mnist-model.git
cd fashion-mnist-model
