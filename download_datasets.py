#!/usr/bin/env python3
"""
Dataset Download Script for SignXpress
This script helps users download the required ASL datasets
"""

import os
import requests
import zipfile

def download_file(url, save_path):
    """Download a file from URL to save_path"""
    print(f"Downloading {os.path.basename(save_path)}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Download the file
    response = requests.get(url, stream=True)
    
    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
    
    print(f"Downloaded: {save_path}")

def main():
    print("=== SignXpress Dataset Downloader ===")
    print()
    print("This script will help you download ASL datasets.")
    print("Due to large file sizes, datasets are not included in the repository.")
    print()
    
    print("Please download datasets from these sources:")
    print()
    print("1. ASL Alphabet Dataset:")
    print("   https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
    print("   - Download from Kaggle")
    print("   - Extract to 'asl_alphabet_dataset/' folder")
    print()
    print("2. ASL Number Dataset:") 
    print("   https://www.kaggle.com/datasets/ayuraj/asl-dataset")
    print("   - Download from Kaggle")
    print("   - Extract to 'asl_number_dataset/' folder")
    print()
    
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()