#!/usr/bin/env python3
"""
Version information for 9517-GroupWork project.
"""

import os

# Get the version from VERSION file
def get_version():
    """
    Read and return the version from the VERSION file.
    
    Returns:
        str: The current version of the project
    """
    version_file = os.path.join(os.path.dirname(__file__), 'VERSION')
    try:
        with open(version_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        return "unknown"

__version__ = get_version()

if __name__ == "__main__":
    print(f"9517-GroupWork Version: {__version__}")
    print("\nThis project implements various machine learning and deep learning methods")
    print("for aerial scene image classification.")
    print("\nIncluded methods:")
    print("  - Machine Learning: SIFT, LBP with various classifiers")
    print("  - Deep Learning: ResNet-18, EfficientNet, Vision Transformer")
