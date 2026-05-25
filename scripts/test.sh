export CUDA_VISIBLE_DEVICES=0
#!/bin/bash

# ============================================================
# Usage:
#
# RANDOM embeddings:
# bash run_train.sh en-de save_random
#
# BERT embeddings:
# bash run_train.sh en-de save_bert bert
#
# ============================================================

SAVE_DIR="/mnt/storage/divya/exam/bfreezmodel" #$2
EMB_TYPE="bert" #${3:-random}

DATA_DIR="/mnt/storage/divya/exam/rnndata" #DATA/$lang

# ============================================================
# BERT MODELS
# ============================================================

SRC_BERT="/mnt/storage/divya/exam/embeddings/hindi-bert-v2"
TGT_BERT="/mnt/storage/divya/exam/embeddings/marathi-bert-v2"

# ============================================================
# CHECK DATA
# ============================================================

if [ ! -d "$DATA_DIR" ]; then
    echo "Can't find data dir $DATA_DIR!"
    exit 1
fi

# ============================================================
# CHECK SAVE DIR
# ============================================================

#if [ -d "$SAVE_DIR" ]; then
#    echo "$SAVE_DIR already exists!"
#    exit 1
#fi

mkdir -p $SAVE_DIR

# ============================================================
# DATA FILE
# ============================================================

DATA_FILE=$DATA_DIR/bprocessed_all-train.pt

# ============================================================
# RANDOM EMBEDDINGS
# ============================================================

if [ "$EMB_TYPE" = "random" ]; then

    echo "=================================================="
    echo "Training with RANDOM embeddings"
    echo "=================================================="

    python -u ../train.py \
        -data $DATA_FILE \
        -save_dir $SAVE_DIR \
        -eval \
	-load_from $SAVE_DIR/model_10.pt 
# ============================================================
# BERT EMBEDDINGS
# ============================================================

elif [ "$EMB_TYPE" = "bert" ]; then

    echo "=================================================="
    echo "Training with BERT embeddings"
    echo "=================================================="

    python -u ../train.py \
        -data $DATA_FILE \
        -load_from $SAVE_DIR/model_10.pt \
	-save_dir $SAVE_DIR \
        -use_bert_src \
        -use_bert_tgt \
        -src_bert_model $SRC_BERT \
        -tgt_bert_model $TGT_BERT \
	-eval  
# ============================================================
# INVALID OPTION
# ============================================================

else

    echo "Invalid embedding type: $EMB_TYPE"
    echo ""
    echo "Supported:"
    echo "  random"
    echo "  bert"
    exit 1

fi
