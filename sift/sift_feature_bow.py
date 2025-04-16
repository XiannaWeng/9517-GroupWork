
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sift_config import *

def extract_color_sift_features(images):
    sift = cv2.SIFT_create()
    all_descriptors = []

    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        channels = cv2.split(rgb)
        img_descriptors = []

        for channel in channels:
            _, des = sift.detectAndCompute(channel, None)
            if des is not None:
                img_descriptors.append(des)

        if len(img_descriptors) > 0:
            descriptors = np.vstack(img_descriptors)
            all_descriptors.append(descriptors)
        else:
            all_descriptors.append(np.array([]))

    return all_descriptors


def create_visual_vocabulary(descriptors_list, n_clusters=N_CLUSTERS, max_samples=MAX_SAMPLES):
    all_descriptors = np.vstack([d for d in descriptors_list if len(d) > 0])
    if len(all_descriptors) > max_samples:
        np.random.seed(RANDOM_STATE)
        all_descriptors = all_descriptors[np.random.choice(len(all_descriptors), max_samples, replace=False)]

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    kmeans.fit(all_descriptors)
    return kmeans


def extract_bow_features(descriptors_list, kmeans):
    n_clusters = kmeans.n_clusters
    features = []
    for descriptors in descriptors_list:
        hist = np.zeros(n_clusters)
        if len(descriptors) > 0:
            labels = kmeans.predict(descriptors)
            hist = np.bincount(labels, minlength=n_clusters)
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-7
        features.append(hist)
    return np.array(features)

