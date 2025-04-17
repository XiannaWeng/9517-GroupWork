import torch
import argparse
import os
import pandas as pd
from src.data_loader import get_dataloaders
from src.deep_learning import get_model
from src.train_test import run_training_loop
from src.evaluate import calculate_metrics, plot_confusion_matrix, plot_training_history
from src.traditional_ml import load_data_for_sklearn, train_evaluate_traditional_ml
from sklearn.model_selection import train_test_split

def main(args):
    print(f"Arguments: {args}")
    RESULTS_DIR = args.results_dir
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    if args.method_type == 'dl':
        # --- Deep Learning Workflow ---
        print("\nRunning Deep Learning Workflow...")
        # Load data
        train_loader, test_loader, class_names = get_dataloaders(
            args.data_dir, args.batch_size, args.test_split, args.num_workers, args.seed
        )
        num_classes = len(class_names)

        # Get model
        model = get_model(args.model_name, num_classes, args.pretrained, args.freeze_layers)
        model.to(device)

        # Run training
        trained_model, history, y_true, y_pred = run_training_loop(
            model, train_loader, test_loader, args.epochs, args.lr, device, model_name=args.model_name
        )

        # Evaluate
        print("\n--- Final Deep Learning Results ---")
        dl_metrics = calculate_metrics(y_true, y_pred, class_names)
        print(pd.DataFrame(dl_metrics['classification_report']).transpose())
        print(f"\nOverall Accuracy: {dl_metrics['accuracy']:.4f}")
        print(f"Weighted Precision: {dl_metrics['precision_weighted']:.4f}")
        print(f"Weighted Recall: {dl_metrics['recall_weighted']:.4f}")
        print(f"Weighted F1-Score: {dl_metrics['f1_weighted']:.4f}")


        # Plot results
        history_save_path = os.path.join(RESULTS_DIR, f'{args.model_name}_history.png')
        cm_save_path = os.path.join(RESULTS_DIR, f'{args.model_name}_cm.png')
        plot_training_history(history, title=f'{args.model_name.capitalize()} Training History', save_path=history_save_path)
        plot_confusion_matrix(y_true, y_pred, class_names, title=f'{args.model_name.capitalize()} Confusion Matrix', save_path=cm_save_path)

        # Save metrics summary
        metrics_summary = {k: v for k, v in dl_metrics.items() if k != 'classification_report'}
        metrics_df = pd.DataFrame([metrics_summary])
        metrics_df.to_csv(os.path.join(RESULTS_DIR, f'{args.model_name}_metrics.csv'), index=False)
        print(f"Metrics summary saved to {os.path.join(RESULTS_DIR, f'{args.model_name}_metrics.csv')}")

    elif args.method_type == 'ml':
        # --- Traditional ML Workflow ---
        print("\nRunning Traditional Machine Learning Workflow...")
        # Load data and extract features
        X, y, class_names = load_data_for_sklearn(args.data_dir, feature_extractor=args.feature_extractor)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_split, random_state=args.seed, stratify=y
        )
        print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

        # Train and evaluate the selected model
        ml_model, y_pred = train_evaluate_traditional_ml(
            X_train, y_train, X_test, y_test, model_type=args.model_name
        )

        # Evaluate
        print(f"\n--- Final Traditional ML Results ({args.model_name.upper()} with {args.feature_extractor}) ---")
        ml_metrics = calculate_metrics(y_test, y_pred, class_names)
        print(pd.DataFrame(ml_metrics['classification_report']).transpose())
        print(f"\nOverall Accuracy: {ml_metrics['accuracy']:.4f}")
        print(f"Weighted Precision: {ml_metrics['precision_weighted']:.4f}")
        print(f"Weighted Recall: {ml_metrics['recall_weighted']:.4f}")
        print(f"Weighted F1-Score: {ml_metrics['f1_weighted']:.4f}")

        # Plot confusion matrix
        cm_save_path = os.path.join(RESULTS_DIR, f'{args.model_name}_{args.feature_extractor}_cm.png')
        plot_confusion_matrix(y_test, y_pred, class_names,
                              title=f'{args.model_name.upper()} ({args.feature_extractor}) Confusion Matrix',
                              save_path=cm_save_path)

        # Save metrics summary
        metrics_summary = {k: v for k, v in ml_metrics.items() if k != 'classification_report'}
        metrics_df = pd.DataFrame([metrics_summary])
        metrics_df.to_csv(os.path.join(RESULTS_DIR, f'{args.model_name}_{args.feature_extractor}_metrics.csv'), index=False)
        print(f"Metrics summary saved to {os.path.join(RESULTS_DIR, f'{args.model_name}_{args.feature_extractor}_metrics.csv')}")

    else:
        print("Error: Invalid method_type specified. Choose 'dl' or 'ml'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COMP9517 Group Project - Aerial Scene Classification')

    # Common arguments
    parser.add_argument('--data_dir', type=str, default='../Aerial_Landscapes', help='Path to the dataset directory')
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save results (plots, metrics)')
    parser.add_argument('--method_type', type=str, required=True, choices=['dl', 'ml'], help='Type of method: dl (Deep Learning) or ml (Traditional ML)')
    parser.add_argument('--test_split', type=float, default=0.2, help='Fraction of data to use for testing (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--cpu', action='store_true', help='Force use CPU even if CUDA is available')

    # Deep Learning specific arguments
    parser.add_argument('--model_name', type=str, default='resnet18', help='Model name (dl: resnet18, resnet50, efficientnet_b0; ml: svm, knn)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs (DL only)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (DL only)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (DL only)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers (DL only)')
    parser.add_argument('--pretrained', action='store_false', help='Use non-pretrained model (DL only)') # Default is True
    parser.add_argument('--freeze_layers', action='store_false', help='Fine-tune all layers instead of just the classifier (DL only)') # Default is True

    # Traditional ML specific arguments
    parser.add_argument('--feature_extractor', type=str, default='hog', choices=['hog', 'lbp'], help='Feature extractor for ML (ML only)')

    args = parser.parse_args()

    # Make sure model name matches method type
    if args.method_type == 'dl' and args.model_name in ['svm', 'knn']:
        parser.error("Model name must be a DL architecture (e.g., resnet18) when method_type is 'dl'.")
    if args.method_type == 'ml' and args.model_name not in ['svm', 'knn']:
         parser.error("Model name must be 'svm' or 'knn' when method_type is 'ml'.")


    main(args)