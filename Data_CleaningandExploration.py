
import os
from pathlib import Path
import pandas as pd
import sentencepiece as spm
import subprocess




# =========================
# CONFIGURATION
# =========================

DATA_DIR ="/mnt/storage/divya/exam/"

#r"C:\Users\USER\Downloads\IIT-Delhi-MISN-Lab-Adivaani-Hindi-Marathi-Dataset-Hiring-2026-20260511T103428Z-3-001\IIT-Delhi-MISN-Lab-Adivaani-Hindi-Marathi-Dataset-Hiring-2026"

PREP_DIR = "prepared_data"

VOCAB_SIZE = 16000
VALID_SIZE = 1000

# =========================
# CREATE OUTPUT DIRECTORY
# =========================

prep_dir = Path(PREP_DIR)
prep_dir.mkdir(exist_ok=True)

# =========================
# FILE PATHS
# =========================

hi_file = os.path.join(DATA_DIR, "train.hi")
mr_file = os.path.join(DATA_DIR, "train.mr")

test_hi_file = os.path.join(DATA_DIR, "test.hi")
test_mr_file = os.path.join(DATA_DIR, "test.mr")

print("Checking dataset files...")

assert os.path.exists(hi_file)
assert os.path.exists(mr_file)
print(test_hi_file)
assert os.path.exists(test_hi_file)
assert os.path.exists(test_mr_file)

print("All files found.")

# =========================
# LOAD TRAIN DATA
# =========================

with open(hi_file, encoding="utf-8") as f:
    hi_lines = [line.strip() for line in f]

with open(mr_file, encoding="utf-8") as f:
    mr_lines = [line.strip() for line in f]

train_df = pd.DataFrame({
    "hi": hi_lines,
    "mr": mr_lines
})

print("Training samples:", len(train_df))

# =========================
# LOAD TEST DATA
# =========================

with open(test_hi_file, encoding="utf-8") as f:
    test_hi = [line.strip() for line in f]

with open(test_mr_file, encoding="utf-8") as f:
    test_mr = [line.strip() for line in f]

test_df = pd.DataFrame({
    "hi": test_hi,
    "mr": test_mr
})

print("Test samples:", len(test_df))

# =========================
# SAVE CLEAN TRAIN FILES
# =========================

train_hi_clean = prep_dir / "clean.train.hi"
train_mr_clean = prep_dir / "clean.train.mr"

with open(train_hi_clean, "w", encoding="utf-8") as f:
    f.write("\n".join(train_df["hi"]))

with open(train_mr_clean, "w", encoding="utf-8") as f:
    f.write("\n".join(train_df["mr"]))

print("Saved cleaned training files.")

# =========================
# SAVE CLEAN TEST FILES
# =========================

test_hi_clean = prep_dir / "clean.test.hi"
test_mr_clean = prep_dir / "clean.test.mr"

with open(test_hi_clean, "w", encoding="utf-8") as f:
    f.write("\n".join(test_df["hi"]))

with open(test_mr_clean, "w", encoding="utf-8") as f:
    f.write("\n".join(test_df["mr"]))

print("Saved cleaned test files.")

# =========================
# TRAIN SENTENCEPIECE
# =========================

combined_text = prep_dir / "combined.txt"

with open(combined_text, "w", encoding="utf-8") as out_f:
    for line in train_df["hi"]:
        out_f.write(line + "\n")

    for line in train_df["mr"]:
        out_f.write(line + "\n")

spm.SentencePieceTrainer.Train(
    input=str(combined_text),
    model_prefix=str(prep_dir / "spm"),
    vocab_size=VOCAB_SIZE,
    character_coverage=1.0,
    model_type="unigram"
)

print("SentencePiece model trained.")

# =========================
# LOAD TOKENIZER
# =========================

sp = spm.SentencePieceProcessor()
sp.load(str(prep_dir / "spm.model"))

# =========================
# TOKENIZATION FUNCTION
# =========================

def encode_file(input_path, output_path):
    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            pieces = sp.encode(line.strip(), out_type=str)
            fout.write(" ".join(pieces) + "\n")

# =========================
# TOKENIZE TRAIN FILES
# =========================

encode_file(train_hi_clean, prep_dir / "spm.train.hi")
encode_file(train_mr_clean, prep_dir / "spm.train.mr")

print("Tokenized training files.")

# =========================
# TOKENIZE TEST FILES
# =========================

encode_file(test_hi_clean, prep_dir / "spm.test.hi")
encode_file(test_mr_clean, prep_dir / "spm.test.mr")

print("Tokenized test files.")

# =========================
# TRAIN / VALID SPLIT
# =========================

train_hi_lines = open(prep_dir / "spm.train.hi", encoding="utf-8").readlines()
train_mr_lines = open(prep_dir / "spm.train.mr", encoding="utf-8").readlines()

valid_hi = train_hi_lines[:VALID_SIZE]
valid_mr = train_mr_lines[:VALID_SIZE]

train_hi = train_hi_lines[VALID_SIZE:]
train_mr = train_mr_lines[VALID_SIZE:]

with open(prep_dir / "valid.spm.hi", "w", encoding="utf-8") as f:
    f.writelines(valid_hi)

with open(prep_dir / "valid.spm.mr", "w", encoding="utf-8") as f:
    f.writelines(valid_mr)

with open(prep_dir / "train_final.spm.hi", "w", encoding="utf-8") as f:
    f.writelines(train_hi)

with open(prep_dir / "train_final.spm.mr", "w", encoding="utf-8") as f:
    f.writelines(train_mr)

print("Train/Validation split completed.")

# =========================
# FAIRSEQ PREPROCESS COMMAND
# =========================

print("\nRun this command manually:\n")

print(
    "fairseq-preprocess "
    "--source-lang mr "
    "--target-lang hi "
    "--trainpref prepared_data/train_final.spm "
    "--validpref prepared_data/valid.spm "
    "--testpref prepared_data/spm.test "
    "--destdir prepared_data/data-bin "
    "--workers 8"
)

print("\nPreprocessing setup completed.")

