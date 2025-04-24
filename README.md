# 9517-GroupWork

## Project Structure

```
9517-GroupWork/
├── ML_Methods/                    # Machine Learning Methods
│   ├── sift/                      # Scale-Invariant Feature Transform implementation
│   │   ├── result.ipynb          # Results and analysis notebook
│   │   ├── sift.py               # Main SIFT implementation
│   │   ├── sift_config.py        # Configuration settings
│   │   ├── sift_evaluation.py    # Evaluation metrics and functions
│   │   ├── sift_feature_bow.py   # Bag of Words feature extraction
│   │   ├── sift_knn.py           # K-Nearest Neighbors implementation
│   │   └── sift_preprocess.py    # Data preprocessing utilities
│   └── lbp.ipynb                 # Local Binary Patterns implementation
│
└── DL_Methods/                    # Deep Learning Methods
    ├── Efficientnet-B3.ipynb     # EfficientNet-B3 implementation
    ├── EfficientNet-B0.ipynb     # EfficientNet-B0 implementation
    ├── ViT.ipynb                 # Vision Transformer implementation
    └── Resnet18/                 # ResNet-18 implementation
        ├── main.py               # Main execution script
        ├── requirements.txt      # Python dependencies
        ├── results/              # Training and evaluation results
        └── src/                  # Source code
            ├── data_loader.py    # Data loading and preprocessing
            ├── traditional_ml.py # Traditional ML methods
            ├── evaluate.py       # Model evaluation utilities
            ├── train_test.py     # Training and testing functions
            └── deep_learning.py  # Deep learning model implementation
```

## Description

This project implements various machine learning and deep learning methods for image classification and feature extraction. The project is organized into two main directories:

1. **ML_Methods**: Contains traditional machine learning approaches including:
   - SIFT (Scale-Invariant Feature Transform) with KNN classification
   - Local Binary Patterns (LBP)

2. **DL_Methods**: Contains deep learning approaches including:
   - EfficientNet (B0 and B3 variants)
   - Vision Transformer (ViT)
   - ResNet-18

Each implementation includes necessary preprocessing, training, evaluation, and visualization components.
