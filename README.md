# 9517-GroupWork

## Project Structure

```
9517-GroupWork/
├── ML_Methods/                    # Machine Learning Methods
│   ├── sift/                      # Scale-Invariant Feature Transform implementation
│   │   ├── result.ipynb          # Results and analysis notebook
│   │   ├── sift.py               # Main SIFT implementation with SVM
│   │   ├── sift_config.py        # Configuration settings
│   │   ├── sift_evaluation.py    # Evaluation metrics and functions
│   │   ├── sift_feature_bow.py   # Bag of Words feature extraction
│   │   ├── sift_knn.py           # SIFT with KNN implementation
│   │   └── sift_preprocess.py    # Data preprocessing utilities
│   └── lbp.ipynb                 # Local Binary Patterns with multiple classifiers
│
└── DL_Methods/                    # Deep Learning Methods
    ├── Efficientnet-B3.ipynb     # EfficientNet-B3 implementation
    ├── EfficientNet-B0.ipynb     # EfficientNet-B0 implementation
    ├── efficientnei_m.ipynb      # EfficientNet-M implementation
    ├── ViT.ipynb                 # Vision Transformer implementation
    └── Resnet18/                 # ResNet-18 implementation
        ├── main.py               # Main execution script
        ├── requirements.txt      # Python dependencies
        ├── results/              # Training and evaluation results
        └── src/                  # Source code
            ├── data_loader.py    # Data loading and preprocessing
            ├── traditional_ml.py # Traditional ML methods (SVM, KNN)
            ├── evaluate.py       # Model evaluation utilities
            ├── train_test.py     # Training and testing functions
            └── deep_learning.py  # Deep learning model implementation
```

## Description

This project implements various machine learning and deep learning methods for aerial scene image classification. The project is organized into two main directories:

1. **ML_Methods**: Contains traditional machine learning approaches
   - **SIFT-based Methods**:
     - SIFT + SVM: Using Scale-Invariant Feature Transform with Support Vector Machine
     - SIFT + KNN: Using Scale-Invariant Feature Transform with K-Nearest Neighbors
   - **LBP-based Methods**:
     - LBP + KNN: Local Binary Patterns with K-Nearest Neighbors
     - LBP + SVM: Local Binary Patterns with Support Vector Machine
     - LBP + Random Forest: Local Binary Patterns with Random Forest Classifier
     - LBP + XGBoost: Local Binary Patterns with XGBoost Classifier

2. **DL_Methods**: Contains deep learning approaches
   - **EfficientNet Variants**:
     - EfficientNet-B0: Lightweight version optimized for speed
     - EfficientNet-B3: Larger version optimized for accuracy
     - EfficientNet-M: Medium-sized version with balanced performance
   - **Vision Transformer (ViT)**: Implementation of the Vision Transformer architecture
   - **ResNet-18**: Implementation of ResNet-18 architecture with:
     - Full training pipeline
     - Data augmentation
     - Model evaluation
     - Integration with traditional ML methods

Each implementation includes:
- Data preprocessing and augmentation
- Feature extraction (for traditional ML methods)
- Model training and validation
- Performance evaluation metrics
- Visualization tools for results
- Confusion matrix generation
- Detailed classification reports

The project supports multiple classifiers and feature extractors that can be combined in various ways to achieve optimal performance on the aerial scene classification task.

## Experimental Results

### Traditional Machine Learning Methods Results

| Feature | Dimensions | Classifier | Accuracy | Precision | Recall | F1 Score |
|---------|------------|------------|----------|-----------|---------|-----------|
| SIFT | - | SVM | 0.5875 | 0.5974 | 0.5875 | 0.5905 |
| SIFT | - | KNN | 0.5333 | 0.5596 | 0.5333 | 0.528 |
| Grayscale-LBP | 100 | KNN | 0.43 | 0.4328 | 0.43 | 0.4241 |
| Grayscale-LBP | 100 | SVM | 0.5133 | 0.5499 | 0.5133 | 0.5097 |
| Grayscale-LBP | 100 | Random Forest | 0.46 | 0.4713 | 0.46 | 0.4547 |
| Grayscale-LBP | 100 | XGBoost | 0.4567 | 0.4705 | 0.4567 | 0.4559 |
| Color-LBP | 100 | KNN | 0.6567 | 0.6694 | 0.6567 | 0.6545 |
| Color-LBP | 100 | SVM | 0.6833 | 0.681 | 0.6833 | 0.678 |
| Color-LBP | 100 | Random Forest | 0.7333 | 0.7327 | 0.7333 | 0.725 |
| Color-LBP | 100 | XGBoost | 0.7133 | 0.7189 | 0.7133 | 0.7116 |
| Color-LBP | 600 | KNN | 0.7233 | - | - | - |
| Color-LBP | 600 | SVM | 0.7456 | - | - | - |
| Color-LBP | 600 | Random Forest | 0.7717 | - | - | - |
| Color-LBP | 600 | XGBoost | 0.7528 | - | - | - |

### Deep Learning Methods Results

| Method | Accuracy | F1 Score | Precision | Recall |
|--------|----------|-----------|------------|---------|
| EfficientNet | 0.9113 | 0.9101 | 0.9112 | 0.9113 |
| ResNet | 0.8908 | 0.8928 | 0.8908 | 0.8908 |
| ViT | 0.8501 | 0.8509 | 0.8502 | 0.8491 |
