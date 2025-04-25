# 9517-GroupWork

## Project Structure

```
9517-GroupWork/
├── ML_Methods/                   # Machine Learning Methods
│   ├── sift/                     # Scale-Invariant Feature Transform implementation
│   │   ├── result.ipynb          # Results and analysis notebook
│   │   ├── sift.py               # Main SIFT implementation with SVM
│   │   ├── sift_config.py        # Configuration settings
│   │   ├── sift_evaluation.py    # Evaluation metrics and functions
│   │   ├── sift_feature_bow.py   # Bag of Words feature extraction
│   │   ├── sift_knn.py           # SIFT with KNN implementation
│   │   └── sift_preprocess.py    # Data preprocessing utilities
│   └── LBP/                      # Local Binary Patterns implementation
│       ├── lbp.ipynb             # LBP with multiple classifiers notebook
│       └── requirements.txt      # Python dependencies for LBP
│
└── DL_Methods/                   # Deep Learning Methods
    ├── EfficientNet/             # EfficientNet implementation
    │   ├── Efficientnet-B3.ipynb # EfficientNet-B3 notebook
    │   ├── EfficientNet-B0.ipynb # EfficientNet-B0 notebook
    │   └── requirements.txt      # Python dependencies for EfficientNet
    ├── Efficientnet-fine_tune/   # Fine-tuned EfficientNet implementation
    │   └── efficientnet-B0-fine.ipynb # Fine-tuned EfficientNet-B0 notebook
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

### Detailed Results

#### SIFT Results

|Metric|SIFT-SVM|SIFT-KNN|
|---|---|---|
|Accuracy|0.5875|0.5333|
|Precision|0.5974|0.5596|
|Recall|0.5875|0.5333|
|F1 Score|0.5905|0.5280|

#### Grayscale-LBP (100 of 800) Results

|Classifier|Accuracy|F1 Score|Precision|Recall|
|---|---|---|---|---|
|KNN|43.00%|0.4241|0.4328|0.4300|
|SVM|51.33%|0.5097|0.5499|0.5133|
|Random Forest|46.00%|0.4547|0.4713|0.4600|
|XGBoost|45.67%|0.4559|0.4705|0.4567|

#### Color-LBP (100 of 800) Results

|Classifier|Accuracy|F1 Score|Precision|Recall|
|---|---|---|---|---|
|K-Nearest Neighbors|0.6567 (65.67%)|0.6545|0.6694|0.6567|
|Support Vector Machine|0.6833 (68.33%)|0.6780|0.6810|0.6833|
|Random Forest|0.7333 (73.33%)|0.7250|0.7327|0.7333|
|XGBoost|0.7133 (71.33%)|0.7116|0.7189|0.7133|

#### LBP Parameters Accuracy Comparison

|Classifier|KNN|Random Forest|SVM|XGBoost|
|---|---|---|---|---|
|Parameters|||||
|P=16, R=1|0.637|0.747|0.693|0.723|
|P=16, R=3|0.667|0.730|0.683|0.713|
|P=16, R=8|0.673|0.760|0.727|0.720|
|P=24, R=1|0.647|0.720|0.693|0.740|
|P=24, R=3|0.667|0.720|0.697|0.747|
|P=24, R=8|0.657|0.703|0.710|0.710|
|P=8, R=1|0.653|0.747|0.670|0.707|
|P=8, R=3|0.623|0.727|0.667|0.713|
|P=8, R=8|0.670|0.747|0.677|0.720|

#### Computation Time Comparison (seconds)

|Classifier|KNN|Random Forest|SVM|XGBoost|
|---|---|---|---|---|
|Parameters|||||
|P=16, R=1|0.08|0.81|0.14|1.58|
|P=16, R=3|0.08|0.90|0.17|1.55|
|P=16, R=8|0.07|0.87|0.16|1.57|
|P=24, R=1|0.09|1.09|0.16|1.78|
|P=24, R=3|0.07|1.07|0.16|1.77|
|P=24, R=8|0.07|0.98|0.17|1.75|
|P=8, R=1|0.08|0.62|0.14|1.41|
|P=8, R=3|0.07|0.62|0.16|1.44|
|P=8, R=8|0.07|0.72|0.15|1.43|

#### Color-LBP (600 of 800) with Different Parameters

|Parameter Set|Classifier|CV Mean|CV Std|Test Acc|Time (s)|
|---|---|---|---|---|---|
|P=8, R=1|KNN|0.7277|0.0091|0.7233|0.36|
||SVM|0.7508|0.0091|0.7456|17.34|
||Random Forest|0.7748|0.0068|0.7717|28.83|
||XGBoost|0.7554|0.0072|0.7528|12.97|
|P=8, R=3|KNN|0.7298|0.0081|0.7167|0.34|
||SVM|0.7518|0.0085|0.7483|18.14|
||Random Forest|0.7644|0.0045|0.7689|28.66|
||XGBoost|0.7476|0.0068|0.7556|12.72|
|P=8, R=8|KNN|0.7256|0.0128|0.7056|0.34|
||SVM|0.7530|0.0030|0.7444|17.99|
||Random Forest|0.7598|0.0044|0.7667|28.59|
||XGBoost|0.7526|0.0051|0.7622|12.58|
|P=16, R=1|KNN|0.7410|0.0052|0.7317|0.36|
||SVM|0.7652|0.0079|0.7628|17.77|
||Random Forest|0.7794|0.0080|0.7783|36.88|
||XGBoost|0.7666|0.0061|0.7594|15.91|
|P=16, R=3|KNN|0.7333|0.0039|0.7222|0.36|
||SVM|0.7724|0.0043|0.7800|18.49|
||Random Forest|0.7726|0.0050|0.7661|36.49|
||XGBoost|0.7623|0.0068|0.7611|14.91|
|P=16, R=8|KNN|0.7276|0.0101|0.7161|0.31|
||SVM|0.7666|0.0066|0.7594|18.03|
||Random Forest|0.7634|0.0058|0.7583|36.09|
||XGBoost|0.7648|0.0038|0.7717|15.04|
|P=24, R=1|KNN|0.7341|0.0076|0.7206|0.38|
||SVM|0.7678|0.0079|0.7667|18.73|
||Random Forest|0.7747|0.0088|0.7722|40.45|
||XGBoost|0.7701|0.0073|0.7589|18.12|
|P=24, R=3|KNN|0.7313|0.0013|0.7228|0.38|
||SVM|0.7779|0.0053|0.7811|20.22|
||Random Forest|0.7692|0.0059|0.7700|42.20|
||XGBoost|0.7666|0.0067|0.7689|18.29|
|P=24, R=8|KNN|0.7247|0.0142|0.7106|0.34|
||SVM|0.7762|0.0071|0.7694|20.64|
||Random Forest|0.7591|0.0069|0.7589|41.84|
||XGBoost|0.7697|0.0064|0.7722|17.57|

## Installation and Setup

### Requirements

To run this project, you'll need Python 3.8+ and the following dependencies:

```bash
# For ML methods
pip install numpy pandas scikit-learn matplotlib opencv-python seaborn tqdm 

# For DL methods
pip install torch torchvision tqdm pandas matplotlib seaborn scikit-learn pillow

# For ResNet18 implementation
cd DL_Methods/Resnet18
pip install -r requirements.txt
```

### Dataset

The project uses the Aerial Landscapes dataset, which is organized as follows:

```
Aerial_Landscapes/
├── Agriculture/    # Agricultural scenes
├── Airport/        # Airport scenes
├── Beach/          # Beach scenes
├── City/           # Urban/city scenes
├── Desert/         # Desert landscapes
├── Forest/         # Forest scenes
├── Grassland/      # Grassland scenes
├── Highway/        # Highway scenes
├── Lake/           # Lake scenes
├── Mountain/       # Mountain scenes
├── Parking/        # Parking lot scenes
├── Port/           # Port/harbor scenes
├── Railway/        # Railway scenes
├── Residential/    # Residential area scenes
└── River/          # River scenes
```

Each directory contains a series of JPEG images of the respective scene category.

## Running the Code

### Traditional Machine Learning Methods

#### SIFT Features

To run the SIFT-based classification:

```bash
cd ML_Methods/sift
python sift.py --data_path '../../Aerial_Landscapes/' --classifier 'svm'
```

Options for `--classifier`: `svm`, `knn`

To evaluate the SIFT results:

```bash
python sift_evaluation.py --results_path './results/'
```

#### LBP Features

For LBP-based classification, run the Jupyter notebook:

```bash
cd ML_Methods
jupyter notebook lbp.ipynb
```

Inside the notebook, you can configure:
- Feature type (Grayscale-LBP, Color-LBP)
- Feature dimensions (100, 600)
- Classifier type (KNN, SVM, Random Forest, XGBoost)

### Deep Learning Methods

#### ResNet-18

To train the ResNet-18 model:

```bash
cd DL_Methods/Resnet18
python main.py --mode train --data_path '../../Aerial_Landscapes/' --batch_size 32 --epochs 50
```

To evaluate the model:

```bash
python main.py --mode test --data_path '../../Aerial_Landscapes/' --model_path './models/resnet18_best.pth'
```

To use traditional ML methods with ResNet-18 features:

```bash
python main.py --mode extract_features --data_path '../../Aerial_Landscapes/' --model_path './models/resnet18_best.pth'
python src/traditional_ml.py --features_path './features/resnet18_features.pkl' --classifier 'svm'
```

Options for `--classifier`: `svm`, `knn`, `random_forest`, `xgboost`

#### EfficientNet and ViT

For EfficientNet and Vision Transformer implementations, run the respective Jupyter notebooks:

```bash
cd DL_Methods
jupyter notebook EfficientNet-B3.ipynb
# or
jupyter notebook ViT.ipynb
```

Each notebook contains detailed instructions on:
- Loading and preprocessing the data
- Training the model
- Evaluating the results
- Visualizing performance

## Code Documentation

Each module in the project has been documented with docstrings and comments explaining:
- Function purpose and parameters
- Implementation details
- References to papers or external resources
- Usage examples

For detailed API documentation, refer to the docstrings in individual files.
