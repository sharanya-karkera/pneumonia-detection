# Pneumonia Detection from Chest X-rays (CNN + MobileNetV2)

A deep learning project that classifies chest X-ray images as NORMAL or PNEUMONIA
using a custom CNN and a MobileNetV2 transfer-learning model.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sharanya-karkera/pneumonia-detection/blob/main/pneumonia_project.ipynb)

## Dataset
Chest X-Ray Images (Pneumonia) from Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The dataset is not included in this repo. The notebook downloads it with the Kaggle API.

## Models
- Custom CNN: 4 convolution blocks with BatchNorm, GlobalAveragePooling and Dropout
- MobileNetV2: ImageNet pretrained base (frozen) with a custom classification head
- Fine-tuned MobileNetV2: top layers unfrozen with a low learning rate

## Steps in the notebook
1. Download and extract the dataset
2. Class distribution and sample images
3. Data augmentation
4. Train the custom CNN
5. Train MobileNetV2 with transfer learning
6. Fine-tune MobileNetV2
7. Evaluate with accuracy, precision, recall, classification report and confusion matrix
8. Predict a new X-ray (single model, ensemble, adjustable threshold)
9. Save and load the final model


## Notes
During fine-tuning, validation loss increased while training loss decreased (overfitting).
Early stopping restored the best weights from the first epoch.

## How to run
1. Open the notebook in Google Colab using the badge above.
2. Select a GPU: Runtime, Change runtime type, T4 GPU.
3. Upload your own kaggle.json when asked.
4. Run all cells.

## Tools used
Python, TensorFlow/Keras, NumPy, pandas, Matplotlib, scikit-learn, Pillow

