import os
import zipfile
import requests
from pathlib import Path

# =========================
# DOWNLOAD DATASET
# =========================

DATA_URL = "https://shorturl.at/9Zkkn"

DOWNLOAD_DIR = "/Data/divya/Part1/dataset"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

zip_path = os.path.join(DOWNLOAD_DIR, "dataset.zip")

print("Downloading dataset...")

response = requests.get(DATA_URL, allow_redirects=True)

with open(zip_path, "wb") as f:
    f.write(response.content)

print("Dataset downloaded.")

# =========================
# EXTRACT ZIP
# =========================

print("Extracting dataset...")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(DOWNLOAD_DIR)

print("Dataset extracted.")

# =========================
# FIND DATA DIRECTORY
# =========================

for root, dirs, files in os.walk(DOWNLOAD_DIR):
    if "train.hi" in files:
        DATA_DIR = root
        break

print("Dataset directory:", DATA_DIR)
