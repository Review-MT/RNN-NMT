# ============================================================
# BertDict.py
# ============================================================
#
# Pruned BERT vocabulary dictionary
#
# Features:
# 1. Keeps ONLY tokens appearing in corpus
# 2. Optional top-k pruning
# 3. Maintains mapping:
#       pruned_idx -> original_bert_idx
# 4. Compatible with torch.save / torch.load
# 5. Compatible with existing NMT pipeline
#
# ============================================================

import torch
from collections import Counter


class BertDict(object):

    def __init__(
        self,
        tokenizer,
        corpus_files=None,
        vocab_size=None
    ):

        self.tokenizer = tokenizer

        # ----------------------------------------------------
        # Core mappings
        # ----------------------------------------------------

        self.idxToLabel = {}
        self.labelToIdx = {}

        # NEW:
        # pruned_vocab_idx -> original_bert_vocab_idx
        self.idxToOrigIdx = {}

        self.frequencies = {}

        self.special = []

        # ----------------------------------------------------
        # Special tokens
        # ----------------------------------------------------

        self.pad_token = tokenizer.pad_token
        self.unk_token = tokenizer.unk_token

        self.bos_token = (
            tokenizer.bos_token
            if tokenizer.bos_token is not None
            else tokenizer.cls_token
        )

        self.eos_token = (
            tokenizer.eos_token
            if tokenizer.eos_token is not None
            else tokenizer.sep_token
        )

        self.pad_id = tokenizer.pad_token_id
        self.unk_id = tokenizer.unk_token_id

        self.bos_id = (
            tokenizer.bos_token_id
            if tokenizer.bos_token_id is not None
            else tokenizer.cls_token_id
        )

        self.eos_id = (
            tokenizer.eos_token_id
            if tokenizer.eos_token_id is not None
            else tokenizer.sep_token_id
        )

        # ----------------------------------------------------
        # Build vocabulary
        # ----------------------------------------------------

        if corpus_files is not None:

            self.build_vocab(
                corpus_files,
                vocab_size
            )

    # ========================================================
    # Vocabulary Building
    # ========================================================

    def build_vocab(
        self,
        corpus_files,
        vocab_size=None
    ):

        print("Building pruned BERT vocabulary...")

        counter = Counter()

        # ----------------------------------------------------
        # Count tokens from corpus
        # ----------------------------------------------------

        for path in corpus_files:

            print(f"Reading: {path}")

            with open(path, encoding="utf-8") as f:

                for line in f:

                    line = line.strip()

                    if len(line) == 0:
                        continue

                    tokens = self.tokenizer.tokenize(line)

                    counter.update(tokens)

        # ----------------------------------------------------
        # Add special tokens FIRST
        # ----------------------------------------------------

        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]

        bert_vocab = self.tokenizer.get_vocab()

        for token in special_tokens:

            if token is None:
                continue

            orig_idx = bert_vocab[token]

            self.addSpecial(
                token,
                orig_idx
            )

        # ----------------------------------------------------
        # Sort by frequency
        # ----------------------------------------------------

        sorted_tokens = sorted(
            counter.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if vocab_size is not None:

            sorted_tokens = sorted_tokens[:vocab_size]

        # ----------------------------------------------------
        # Add tokens
        # ----------------------------------------------------

        for token, freq in sorted_tokens:

            if token in self.labelToIdx:
                continue

            orig_idx = bert_vocab[token]

            idx = self.add(
                token,
                orig_idx
            )

            self.frequencies[idx] = freq

        print(
            f"Final pruned vocabulary size: "
            f"{self.size()}"
        )

    # ========================================================
    # Add token
    # ========================================================

    def add(
        self,
        token,
        orig_idx=None
    ):

        if token in self.labelToIdx:

            idx = self.labelToIdx[token]

            if idx not in self.frequencies:
                self.frequencies[idx] = 1
            else:
                self.frequencies[idx] += 1

            return idx

        idx = len(self.idxToLabel)

        self.idxToLabel[idx] = token
        self.labelToIdx[token] = idx

        # IMPORTANT
        self.idxToOrigIdx[idx] = orig_idx

        self.frequencies[idx] = 1

        return idx

    # ========================================================
    # Add special token
    # ========================================================

    def addSpecial(
        self,
        token,
        orig_idx=None
    ):

        idx = self.add(
            token,
            orig_idx
        )

        self.special.append(idx)

    # ========================================================
    # Basic utilities
    # ========================================================

    def size(self):

        return len(self.idxToLabel)

    def lookup(
        self,
        key,
        default=None
    ):

        return self.labelToIdx.get(
            key,
            default
        )

    def getLabel(
        self,
        idx
    ):

        return self.idxToLabel[idx]

    # ========================================================
    # Sentence -> indices
    # ========================================================

    def convertToIdx(
        self,
        sentence,
        unkWord=None,
        bosWord=False,
        eosWord=False
    ):

        tokens = self.tokenizer.tokenize(
            sentence
        )

        vec = []

        if bosWord:
            vec.append(self.bos_id)

        unk = self.lookup(
            self.unk_token
        )

        for tok in tokens:

            idx = self.lookup(
                tok,
                unk
            )

            vec.append(idx)

        if eosWord:
            vec.append(self.eos_id)

        return torch.LongTensor(vec)

    # ========================================================
    # Indices -> tokens
    # ========================================================

    def convertToLabels(
        self,
        idx,
        stop=None
    ):

        labels = []

        for i in idx:

            if isinstance(i, torch.Tensor):
                i = i.item()

            labels.append(
                self.idxToLabel[i]
            )

            if stop is not None and i == stop:
                break

        return labels

    # ========================================================
    # Save vocabulary
    # ========================================================

    def writeFile(
        self,
        filename
    ):

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as f:

            for i in range(self.size()):

                token = self.idxToLabel[i]

                orig_idx = self.idxToOrigIdx[i]

                f.write(
                    f"{i}\t{token}\t{orig_idx}\n"
                )

    # ========================================================
    # Optional pruning
    # ========================================================

    def prune(
        self,
        size
    ):

        if size >= self.size():
            return self

        freq = torch.Tensor([
            self.frequencies[i]
            for i in range(len(self.frequencies))
        ])

        _, idx = torch.sort(
            freq,
            0,
            True
        )

        newDict = BertDict(
            self.tokenizer
        )

        # preserve specials

        for i in self.special:

            token = self.idxToLabel[i]

            orig_idx = self.idxToOrigIdx[i]

            newDict.addSpecial(
                token,
                orig_idx
            )

        for i in idx[:size]:

            i = i.item()

            token = self.idxToLabel[i]

            if token in newDict.labelToIdx:
                continue

            orig_idx = self.idxToOrigIdx[i]

            newDict.add(
                token,
                orig_idx
            )

        return newDict
