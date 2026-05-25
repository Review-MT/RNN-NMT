#!/bin/bash

# Usage:
# bash eval.sh reference.txt prediction.txt

REF=$1
HYP=$2

echo "=============================="
echo "Evaluating Translation Output"
echo "=============================="

# BLEU-100
echo ""
echo "BLEU-100:"
sacrebleu "$REF" -i "$HYP" -m bleu -b -w 4

# chrF++-100
echo ""
echo "chrF++-100:"
sacrebleu "$REF" -i "$HYP" -m chrf --chrf-word-order 2 -b -w 4
