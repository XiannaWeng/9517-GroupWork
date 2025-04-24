import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Define standard transforms
IMG_SIZE = 224 # Standard for many pre-trained models
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def get_dataloaders(data_dir, batch_size=32, test_split=0.2, num_workers=4, seed=42):
    """Creates training and testing dataloaders."""
    if not os.path.isdir(data_dir):
         raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    dataset = datasets.ImageFolder(data_dir) # Automatically finds classes from folder names
    class_names = dataset.classes
    print(f"Found {len(dataset)} images in {len(class_names)} categories.")
    print(f"Categories: {class_names}")

    # Split data per category if needed for strict balance, or use sklearn for overall split
    # Using sklearn's train_test_split for simplicity here on indices
    targets = dataset.targets
    train_idx, test_idx = train_test_split(
        np.arange(len(targets)),
        test_size=test_split,
        random_state=seed,
        shuffle=True,
        stratify=targets # Ensure split respects class distribution
    )

    # Debug: Print split statistics
    print("\nDataset Split Statistics:")
    print(f"Total samples: {len(dataset)}")
    print(f"Train samples: {len(train_idx)} ({len(train_idx)/len(dataset)*100:.1f}%)")
    print(f"Test samples: {len(test_idx)} ({len(test_idx)/len(dataset)*100:.1f}%)")
    
    # Debug: Check class distribution in splits
    train_targets = [targets[i] for i in train_idx]
    test_targets = [targets[i] for i in test_idx]
    print("\nClass distribution in splits:")
    for i, class_name in enumerate(class_names):
        train_count = train_targets.count(i)
        test_count = test_targets.count(i)
        print(f"{class_name}: Train={train_count}, Test={test_count} (Train%={train_count/(train_count+test_count)*100:.1f}%)")

    # Create dataset subsets with respective transforms
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    train_transformed_dataset = TransformedDataset(train_dataset, transform=train_transform)
    test_transformed_dataset = TransformedDataset(test_dataset, transform=test_transform)

    # Debug: Verify no overlap between train and test indices
    train_set = set(train_idx)
    test_set = set(test_idx)
    overlap = train_set.intersection(test_set)
    if overlap:
        print(f"\nWARNING: Found {len(overlap)} overlapping samples between train and test sets!")
    else:
        print("\nTrain-test split verification: No overlapping samples found.")

    # Create dataloaders
    train_loader = DataLoader(train_transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_transformed_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, class_names

# --- Add functions for K-Fold Cross-Validation setup if needed ---
# --- Add functions for creating imbalanced datasets if needed (src/imbalance.py) ---

if __name__ == '__main__':
    # Example Usage (replace with your actual data path)
    DATA_PATH = '../Aerial_Landscapes' # Adjust this path relative to where you run this script
    try:
        train_loader, test_loader, class_names = get_dataloaders(DATA_PATH)
        print(f"\nTrain batches: {len(train_loader)}, Test batches: {len(test_loader)}")
        # Check a batch
        dataiter = iter(train_loader)
        images, labels = next(dataiter)
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        print(f"Class names: {class_names}")
    except FileNotFoundError as e:
        print(e)
        print("Please ensure the 'Aerial_Landscapes' directory exists at the expected location.")