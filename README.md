# RNN Neural Machine Translation (NMT) Pipeline

This directory contains the codebase for training and evaluating Neural Machine Translation models (with random and pretrained embeddings ). It also supports Actor Critic based reinforcement fine-tuning of trained Model.

## Directory Structure

```text
.
├── data/               # Raw and extracted dataset configurations
├── lib/                # Core architecture modules
│   ├── data/           # Dataset, vocabulary, and dictionary utilities
│   ├── eval/           # Evaluation metrics and loop managers
│   ├── metric/         # Reward shaping, BLEU score computation
│   ├── model/          # Encoder-Decoder and Attention architectures
│   └── train/          # Trainers (Standard and Reinforce)
├── scripts/            # Bash execution scripts for the pipeline
│   ├── random_preprocess.sh(random) or pretrained_preprocess.sh(pretrained)  # Data preprocessing entrypoint 
│   └── pretrain.sh     # Model training execution script
    └── test.sh         # Model evaluation execution script
├── preprocess.py       # Preprocessing Python entrypoint
└── train.py            # Main training Python entrypoint



Getting Started
1. Prerequisites & Environment Setup
Ensure your environment meets the dependency requirements. You can install them using your project's root requirements.txt:

Bash
pip install -r requirements.txt
2. Data Preprocessing
Before starting the training loop, you must process the raw source and target corpus into tokenized data tensors. The preprocessing step sets up vocabulary limits, truncation, and validation splits.

Run the preprocessing script located in the scripts folder:

Bash
bash scripts/random_preprocess.sh (random embeddings)
bash scripts/pretrained_preprocess.sh (pretrained embeddings)


## Configuration & Embeddings

The pipeline supports both standard learned embeddings and pretrained contextual embeddings (BERT) for cross-lingual configurations (e.g., Hindi to Marathi).

### Embedding Parameters

You can configure the text representation layer using the following command-line arguments:

| Argument | Type / Choices | Default | Description |
| :--- | :--- | :--- | :--- |
| `-embedding_type` | `random`, `bert` | `random` | **`random`**: Initializes an embedding matrix from scratch to be trained with the model.<br>**`bert`**: Leverages pretrained contextual BERT models for token representations. |
| `-src_bert_model` | `str` (Path) | `"/mnt/storage/divya/exam/embeddings/hindi-bert-v2"` | The local file path or model identifier for the **source language** BERT model. |
| `-tgt_bert_model` | `str` (Path) | `"/mnt/storage/divya/exam/embeddings/marathi-bert-v2"` | The local file path or model identifier for the **target language** BERT model. |

### Usage Examples

#### 1. Running with Default Random Embeddings
If you do not specify an embedding type, the model defaults to training standard embeddings from scratch:
```bash
python train.py -embedding_type random

3. Training the Model
Once the data is processed, you can launch the training sequence. The pipeline supports standard sequence-to-sequence pretraining as well as Reinforcement Learning (AC) tuning.

To run the training with or without pre-trained embeddings set appropriate flag in training script file (epochs, batch sizes, learning rates, and optimization parameters), execute:

Bash
bash scripts/pretrain.sh
