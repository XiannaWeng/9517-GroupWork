
def load_and_split_dataset(dataset_path, test_size=0.2, sample_ratio=1.0):
    import os
    import numpy as np
    import cv2
    from glob import glob

    train_images, train_labels, test_images, test_labels = [], [], [], []
    classes = sorted(os.listdir(dataset_path))

    for class_id, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_path, class_name)
        img_paths = glob(os.path.join(class_dir, '*.jpg'))
        if len(img_paths) == 0:
            continue

        np.random.seed(42)
        np.random.shuffle(img_paths)
        n_samples = int(len(img_paths) * sample_ratio)
        sampled_paths = img_paths[:max(n_samples, 1)]
        split_idx = int(len(sampled_paths) * (1 - test_size))
        train_paths = sampled_paths[:split_idx]
        test_paths = sampled_paths[split_idx:] if split_idx < len(sampled_paths) else []

        for path in train_paths:
            img = cv2.imread(path)
            if img is not None:
                train_images.append(img)
                train_labels.append(class_id)

        for path in test_paths:
            img = cv2.imread(path)
            if img is not None:
                test_images.append(img)
                test_labels.append(class_id)

    return (train_images, train_labels), (test_images, test_labels), classes