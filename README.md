# -Transfer-Learning-with-ResNet18-on-Caltech101-Full-Training-Pipeline-Grad-CAM-Model-Visualization-
A complete PyTorch Transfer Learning project using ResNet18 trained on the Caltech101 dataset with full training pipeline, early stopping, LR scheduling, Grad-CAM visualization, Grad-CAM++ heatmaps, and classification metrics.
Caltech-101 Image Classification using Transfer Learning (ResNet-18)

This project demonstrates a full Transfer Learning pipeline for classifying images from the Caltech-101 dataset using a pretrained ResNet-18 model.
The workflow includes training, validation, prediction, model saving, and visual explanations using Grad-CAM.

This project is designed to highlight practical deep learning skills suitable for Data Scientist, Computer Vision, and Machine Learning roles.

🚀 Project Highlights

Transfer Learning with pretrained ResNet-18

Caltech-101 multi-class image classification

Custom data augmentation

Full training + validation loop

Early Stopping to prevent overfitting

Learning Rate Scheduler

Grad-CAM / Grad-CAM++ visualizations

Prediction function for inference

Model saving & loading (state_dict)

📊 Results (Summary)

Best Validation Accuracy: ~ 92–93%

Final Training Accuracy: > 99%

Consistent improvement with augmentation & scheduler

Strong visual interpretability using Grad-CAM

🗂 Dataset

Caltech-101

101 object categories

9,000+ images

High intra-class variation

Dataset automatically downloaded using torchvision.datasets.Caltech101.

🏗 Model Architecture

Base model: ResNet-18 pretrained on ImageNet

Replaced final layer with a custom 101-class classifier

Frozen early layers → trained deeper layers + classifier

Optimizer: Adam

Loss: CrossEntropyLoss

🔧 Tech Stack

Python

PyTorch

Torchvision

Matplotlib

OpenCV

NumPy

✨ Grad-CAM Interpretability

Used Grad-CAM and Grad-CAM++ to highlight the regions the model focuses on while predicting.

This helps validate:

Model trustworthiness

How to run

git clone <your-repo-url>
cd caltech101-resnet
pip install -r requirements.txt
python train.py

📁 Folder Structure (Recommended)

├── data/
├── models/
│   └── best_model.pth
├── notebooks/
├── results/
│   ├── training_curves.png
│   ├── gradcam_examples.png
├── train.py
├── predict.py
├── utils.py
└── README.md

⭐ Key Skills Demonstrated

Transfer Learning

CNN fine-tuning

Large-scale image preprocessing

Early stopping

LRScheduler

Model evaluation

Grad-CAM interpretability

Professional ML project structuring

🤝 Open for Collaboration

I am open to ML/CV research collaborations, internships, and full-time opportunities in:

Computer Vision

Deep Learning

Machine Learning Engineering

Model heatmap focus on object of interest

Interpretability for real-world use
