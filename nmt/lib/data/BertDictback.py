# ============================================================
# lib/BertDict.py
# ============================================================

import torch


class BertDict(object):

    def __init__(self, tokenizer):

        self.tokenizer = tokenizer

        self.idxToLabel = {}
        self.labelToIdx = {}
        self.frequencies = {}

        # ----------------------------------------------------
        # Load tokenizer vocabulary
        # ----------------------------------------------------

        vocab = tokenizer.get_vocab()

        for token, idx in vocab.items():

            self.idxToLabel[idx] = token
            self.labelToIdx[token] = idx

            self.frequencies[idx] = 1

        # ----------------------------------------------------
        # Special tokens
        # ----------------------------------------------------

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

    # ========================================================
    # Size
    # ========================================================

    def size(self):

        return len(self.idxToLabel)

    # ========================================================
    # Lookup token -> id
    # ========================================================

    def lookup(self, key, default=None):

        try:

            return self.labelToIdx[key]

        except KeyError:

            return default

    # ========================================================
    # Lookup id -> token
    # ========================================================

    def getLabel(self, idx, default=None):

        try:

            return self.idxToLabel[idx]

        except KeyError:

            return default

    # ========================================================
    # Save vocabulary
    # ========================================================

    def writeFile(self, filename):

        with open(filename, "w", encoding="utf-8") as file:

            for i in range(self.size()):

                label = self.idxToLabel[i]

                file.write(
                    "%s %d\n" % (label, i)
                )

    # ========================================================
    # Convert sentence -> ids
    # SAME API as Dict.convertToIdx()
    # ========================================================

    def convertToIdx(
        self,
        labels,
        unkWord=None,
        bosWord=None,
        eosWord=None
    ):

        # ----------------------------------------------------
        # Input may already be string
        # ----------------------------------------------------

        if isinstance(labels, list):

            sentence = " ".join(labels)

        else:

            sentence = labels

        vec = []

        # ----------------------------------------------------
        # BOS
        # ----------------------------------------------------

        if bosWord is not None:

            vec += [self.bos_id]

        # ----------------------------------------------------
        # Tokenize
        # ----------------------------------------------------

        token_ids = self.tokenizer.encode(
            sentence,
            add_special_tokens=False
        )

        vec += token_ids

        # ----------------------------------------------------
        # EOS
        # ----------------------------------------------------

        if eosWord is not None:

            vec += [self.eos_id]

        return torch.LongTensor(vec)

    # ========================================================
    # Convert ids -> labels
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

            labels += [self.getLabel(i)]

            if stop is not None and i == stop:

                break

        return labels

    # ========================================================
    # Dummy prune (BERT vocab is fixed)
    # ========================================================

    def prune(self, size):

        print(
            "WARNING: prune() ignored for BertDict "
            "(fixed pretrained vocabulary)"
        )

        return self
