import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import os
import time
from PIL import Image
from .data_loader import IMG_SIZE # Use the same size for consistency

def extract_hog_features(image):
    """Extracts HOG features from a single image."""
    # Convert to grayscale and resize if needed (ensure consistent input)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[0] != IMG_SIZE or image.shape[1] != IMG_SIZE:
         image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    # Parameters might need tuning based on dataset/image characteristics
    features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return features

def extract_lbp_features(image, radius=1, n_points=8):
    """Extracts LBP features (histogram) from a single image."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[0] != IMG_SIZE or image.shape[1] != IMG_SIZE:
         image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    # Normalize histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6) # Add epsilon for numerical stability
    return hist


def load_data_for_sklearn(data_dir, feature_extractor='hog'):
    """Loads images and extracts features for scikit-learn."""
    X, y = [], []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    label_map = {name: i for i, name in enumerate(class_names)}

    print(f"Loading data from {data_dir} for feature extraction ({feature_extractor})...")
    for class_name in tqdm(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            # Load with OpenCV for easy color/grayscale handling
            img = cv2.imread(img_path)
            if img is not None:
                 # Resize here OR ensure consistent size in extractor
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                if feature_extractor == 'hog':
                    features = extract_hog_features(img_resized)
                elif feature_extractor == 'lbp':
                    features = extract_lbp_features(img_resized)
                # Add more feature extractors here (e.g., SIFT/ORB with BoVW)
                else:
                    raise ValueError("Unsupported feature extractor")

                X.append(features)
                y.append(label_map[class_name])

    return np.array(X), np.array(y), class_names

def train_evaluate_traditional_ml(X_train, y_train, X_test, y_test, model_type='svm'):
    """Trains and evaluates SVM or kNN."""
    start_time = time.time()
    print(f"\nTraining {model_type.upper()} model...")

    # Create a pipeline with scaling and the classifier
    if model_type == 'svm':
        print("Initializing SVM with RBF kernel...")
        classifier = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
        # Example GridSearch (can be computationally expensive)
        # params = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
        # classifier = GridSearchCV(SVC(probability=True, random_state=42), params, cv=3, n_jobs=-1)
    elif model_type == 'knn':
        print("Initializing k-NN classifier...")
        # k can be tuned
        classifier = KNeighborsClassifier(n_neighbors=5, weights='distance')
        # Example GridSearch
        # params = {'n_neighbors': [3, 5, 7, 9]}
        # classifier = GridSearchCV(KNeighborsClassifier(), params, cv=3, n_jobs=-1)
    else:
        raise ValueError("Unsupported model type")

    print("Creating preprocessing pipeline...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()), # Feature scaling is important for SVM and kNN
        ('classifier', classifier)
    ])

    print("Starting model training...")
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    # Evaluate
    print(f"\nEvaluating {model_type.upper()} model...")
    print("Making predictions on test set...")
    y_pred = pipeline.predict(X_test)
    print("Evaluation complete.")

    return pipeline, y_pred # Return the trained model and predictions

# --- Example Usage ---
if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    # Adjust path as needed
    DATA_PATH = '../Aerial_Landscapes'
    FEATURE_TYPE = 'hog' # or 'lbp'

    try:
        X, y, class_names_trad = load_data_for_sklearn(DATA_PATH, feature_extractor=FEATURE_TYPE)
        print(f"Feature matrix shape: {X.shape}")
        print(f"Labels shape: {y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

        # Train and evaluate SVM
        svm_model, svm_preds = train_evaluate_traditional_ml(X_train, y_train, X_test, y_test, model_type='svm')

        # Train and evaluate kNN
        knn_model, knn_preds = train_evaluate_traditional_ml(X_train, y_train, X_test, y_test, model_type='knn')

        # Add calls to src.evaluate functions here to show results
        # from src.evaluate import calculate_metrics, plot_confusion_matrix
        # svm_metrics = calculate_metrics(y_test, svm_preds, class_names_trad)
        # knn_metrics = calculate_metrics(y_test, knn_preds, class_names_trad)
        # print("\nSVM Metrics:", svm_metrics)
        # print("kNN Metrics:", knn_metrics)
        # plot_confusion_matrix(y_test, svm_preds, class_names_trad, title=f'SVM ({FEATURE_TYPE}) Confusion Matrix')
        # plot_confusion_matrix(y_test, knn_preds, class_names_trad, title=f'kNN ({FEATURE_TYPE}) Confusion Matrix')

    except FileNotFoundError as e:
        print(e)
        print("Please ensure the 'Aerial_Landscapes' directory exists.")
    except Exception as e:
        print(f"An error occurred: {e}")