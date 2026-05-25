import os
import zipfile

ZIP_PATH = "dataset/dataset.zip"

EXTRACT_DIR = "dataset/extracted"

os.makedirs(EXTRACT_DIR, exist_ok=True)

print("Checking ZIP...")

assert os.path.exists(ZIP_PATH), f"ZIP not found: {ZIP_PATH}"

print("Extracting dataset...")

with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

print("Extraction completed.")

# Find dataset directory automatically
DATA_DIR = None

for root, dirs, files in os.walk(EXTRACT_DIR):
    if "train.hi" in files:
        DATA_DIR = root
        break

if DATA_DIR is None:
    raise ValueError("Could not find train.hi")

print("Dataset directory:", DATA_DIR)

# Dataset files
hi_file = os.path.join(DATA_DIR, "train.hi")
mr_file = os.path.join(DATA_DIR, "train.mr")

test_hi_file = os.path.join(DATA_DIR, "test.hi")
test_mr_file = os.path.join(DATA_DIR, "test.mr")

print("Checking dataset files...")

assert os.path.exists(hi_file)
assert os.path.exists(mr_file)
assert os.path.exists(test_hi_file)
assert os.path.exists(test_mr_file)

print("All dataset files found.")
