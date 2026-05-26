# ============================================================
# preprocess.py
# ============================================================
#
# Supports TWO preprocessing modes:
#
# 1. Random Embeddings
#    - custom vocabulary
#    - word-level tokenization
#
# 2. BERT Embeddings
#    - BERT tokenizer vocabulary
#    - WordPiece tokenization
#
# ============================================================

import argparse
import torch

from transformers import AutoTokenizer

import lib
#from lib.BertDict import BertDict
from lib.data.BertDict import BertDict
# ============================================================
# Argument Parser
# ============================================================

parser = argparse.ArgumentParser(description="preprocess.py")

# ------------------------------------------------------------
# Dataset paths
# ------------------------------------------------------------

parser.add_argument("-train_src", required=True)
parser.add_argument("-train_tgt", required=True)

parser.add_argument("-train_xe_src", required=True)
parser.add_argument("-train_xe_tgt", required=True)

parser.add_argument("-train_pg_src", required=True)
parser.add_argument("-train_pg_tgt", required=True)

parser.add_argument("-valid_src", required=True)
parser.add_argument("-valid_tgt", required=True)

parser.add_argument("-test_src", required=True)
parser.add_argument("-test_tgt", required=True)

# ------------------------------------------------------------
# Save path
# ------------------------------------------------------------

parser.add_argument("-save_data", required=True)

# ------------------------------------------------------------
# Vocabulary size (random mode only)
# ------------------------------------------------------------

parser.add_argument("-src_vocab_size", type=int, default=50000)
parser.add_argument("-tgt_vocab_size", type=int, default=50000)

# ------------------------------------------------------------
# Sequence length
# ------------------------------------------------------------

parser.add_argument("-seq_length", type=int, default=100)

# ------------------------------------------------------------
# Embedding type
# ------------------------------------------------------------

parser.add_argument(
    "-embedding_type",
    choices=["random", "bert"],
    default="random"
)

# ------------------------------------------------------------
# BERT models
# ------------------------------------------------------------

parser.add_argument(
    "-src_bert_model",
    type=str,
    default="/mnt/storage/divya/exam/embeddings/hindi-bert-v2"
)

parser.add_argument(
    "-tgt_bert_model",
    type=str,
    default="/mnt/storage/divya/exam/embeddings/marathi-bert-v2"
)

# ------------------------------------------------------------
# Misc
# ------------------------------------------------------------

parser.add_argument("-seed", type=int, default=3435)

parser.add_argument(
    "-report_every",
    type=int,
    default=100000
)

opt = parser.parse_args()

torch.manual_seed(opt.seed)


# ============================================================
# RANDOM VOCABULARY
# ============================================================

def makeVocabulary(filename, size):

    vocab = lib.Dict([
        lib.Constants.PAD_WORD,
        lib.Constants.UNK_WORD,
        lib.Constants.BOS_WORD,
        lib.Constants.EOS_WORD
    ])

    with open(filename, encoding="utf-8") as f:

        for sent in f.readlines():

            for word in sent.split():

                vocab.add(word.lower())

    originalSize = vocab.size()

    vocab = vocab.prune(size)

    print(
        "Created dictionary of size %d "
        "(pruned from %d)"
        % (vocab.size(), originalSize)
    )

    return vocab


def initVocabulary(
    name,
    dataFile,
    vocabSize,
    saveFile
):

    print(f"Building {name} vocabulary...")

    vocab = makeVocabulary(
        dataFile,
        vocabSize
    )

    print(
        f"Saving {name} vocabulary to "
        f"\"{saveFile}\"..."
    )

    vocab.writeFile(saveFile)

    return vocab


# ============================================================
# BERT VOCABULARY
# ============================================================

#class BertDict(object):

#    def __init__(self, tokenizer):

#        self.tokenizer = tokenizer

#        self.idxToLabel = {}
#        self.labelToIdx = {}

#        vocab = tokenizer.get_vocab()

#        for token, idx in vocab.items():
#
#            self.labelToIdx[token] = idx
#            self.idxToLabel[idx] = token

#        self.pad_id = tokenizer.pad_token_id
#        self.unk_id = tokenizer.unk_token_id
#        self.bos_id = tokenizer.cls_token_id
#        self.eos_id = tokenizer.sep_token_id

#    def size(self):

#        return len(self.labelToIdx)

#    def lookup(self, key, default=None):

#        return self.labelToIdx.get(key, default)

 #   def getLabel(self, idx):

 #       return self.idxToLabel[idx]

    # --------------------------------------------------------
    # SAME FORMAT AS RANDOM convertToIdx
    # --------------------------------------------------------

#    def convertToIdx(
#        self,
#        sentence,
#        unkWord=None,
#        bosWord=False,
#        eosWord=False
#    ):

#        vec = []

#        if bosWord:
#            vec += [self.bos_id]

#        token_ids = self.tokenizer.encode(
#            sentence,
#            add_special_tokens=False
#        )

#        vec += token_ids

#        if eosWord:
#            vec += [self.eos_id]

#        return torch.LongTensor(vec)

#    def convertToLabels(self, idx):

#        tokens = []

#        for i in idx:

#            if isinstance(i, torch.Tensor):
#                i = i.item()

#            tokens.append(self.idxToLabel[i])

#        return tokens

#    def writeFile(self, filename):

#        with open(filename, "w", encoding="utf-8") as f:

#            for idx in range(len(self.idxToLabel)):

#                token = self.idxToLabel[idx]

#                f.write(f"{token}\n")


#def initBertVocabulary(
#    name,
#    model_name,
#    saveFile
#):

#    print(
#        f"Loading BERT tokenizer for {name}: "
#        f"{model_name}"
#    )

#    tokenizer = AutoTokenizer.from_pretrained(
#        model_name,
#        use_fast=False
#    )

#    vocab = BertDict(tokenizer)

#    print(
#        f"{name} BERT vocabulary size: "
#        f"{vocab.size()}"
#    )

#    print(
#        f"Saving {name} BERT vocabulary to "
#        f"\"{saveFile}\"..."
#    )

#    vocab.writeFile(saveFile)

#    return vocab



def initBertVocabulary(
    name,
    model_name,
    saveFile,
    corpus_files,
    vocab_size
):

    print(
        f"Loading BERT tokenizer for {name}: "
        f"{model_name}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False
    )

    vocab = BertDict(
        tokenizer=tokenizer,
        corpus_files=corpus_files,
        vocab_size=vocab_size
    )

    print(
        f"{name} pruned BERT vocabulary size: "
        f"{vocab.size()}"
    )

    print(
        f"Saving {name} vocabulary to "
        f"\"{saveFile}\"..."
    )

    vocab.writeFile(saveFile)

    return vocab

# ============================================================
# RANDOM DATA PROCESSING
# ============================================================

def makeDataRandom(
    which,
    srcFile,
    tgtFile,
    srcDicts,
    tgtDicts
):

    src = []
    tgt = []

    sizes = []

    count = 0
    ignored = 0

    print(f"Processing RANDOM mode: {which}")

    srcF = open(srcFile, encoding="utf-8")
    tgtF = open(tgtFile, encoding="utf-8")

    while True:

        srcWords = srcF.readline().strip().split()
        tgtWords = tgtF.readline().strip().split()

        if not srcWords or not tgtWords:

            if (srcWords and not tgtWords) or \
               (not srcWords and tgtWords):

                print(
                    "WARNING: source and target "
                    "have different number of lines"
                )

            break

        if (
            len(srcWords) <= opt.seq_length and
            len(tgtWords) <= opt.seq_length
        ):

            src_ids = srcDicts.convertToIdx(
                srcWords,
                lib.Constants.UNK_WORD
            )

            tgt_ids = tgtDicts.convertToIdx(
                tgtWords,
                lib.Constants.UNK_WORD,
                eosWord=lib.Constants.EOS_WORD
            )

            src.append(src_ids)
            tgt.append(tgt_ids)

            sizes.append(len(srcWords))

        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:

            print(f"... {count} sentences processed")

    srcF.close()
    tgtF.close()

    assert len(src) == len(tgt)

    print(
        f"Prepared {len(src)} sentence pairs "
        f"({ignored} ignored)"
    )

    return src, tgt, range(len(src))


# ============================================================
# BERT DATA PROCESSING
# ============================================================

def makeDataBERT(
    which,
    srcFile,
    tgtFile,
    srcDicts,
    tgtDicts
):

    src = []
    tgt = []

    sizes = []

    count = 0
    ignored = 0

    print(f"Processing BERT mode: {which}")

    srcF = open(srcFile, encoding="utf-8")
    tgtF = open(tgtFile, encoding="utf-8")

    while True:

        src_sent = srcF.readline().strip()
        tgt_sent = tgtF.readline().strip()

        if not src_sent or not tgt_sent:

            if (src_sent and not tgt_sent) or \
               (not src_sent and tgt_sent):

                print(
                    "WARNING: source and target "
                    "have different number of lines"
                )

            break

        src_ids = srcDicts.convertToIdx(
            src_sent
        )

        tgt_ids = tgtDicts.convertToIdx(
            tgt_sent
        )

        if (
            len(src_ids) <= opt.seq_length and
            len(tgt_ids) <= opt.seq_length
        ):

            src.append(src_ids)
            tgt.append(tgt_ids)

            sizes.append(len(src_ids))

        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:

            print(f"... {count} sentences processed")

    srcF.close()
    tgtF.close()

    assert len(src) == len(tgt)

    print(
        f"Prepared {len(src)} sentence pairs "
        f"({ignored} ignored)"
    )

    return src, tgt, range(len(src))


# ============================================================
# GENERAL DATA WRAPPER
# ============================================================

def makeDataGeneral(
    which,
    src_path,
    tgt_path,
    dicts,
    embedding_type="random"
):

    print(f"Preparing {which} ({embedding_type})")

    res = {}

    # --------------------------------------------------------
    # RANDOM
    # --------------------------------------------------------

    if embedding_type == "random":

        src, tgt, pos = makeDataRandom(
            which,
            src_path,
            tgt_path,
            dicts["src"],
            dicts["tgt"]
        )

    # --------------------------------------------------------
    # BERT
    # --------------------------------------------------------

    elif embedding_type == "bert":

        src, tgt, pos = makeDataBERT(
            which,
            src_path,
            tgt_path,
            dicts["src"],
            dicts["tgt"]
        )

    else:

        raise ValueError(
            f"Unknown embedding type: "
            f"{embedding_type}"
        )

    res["src"] = src
    res["tgt"] = tgt
    res["pos"] = pos

    return res


# ============================================================
# MAIN
# ============================================================

def main():

    dicts = {}

    # ========================================================
    # RANDOM MODE
    # ========================================================

    if opt.embedding_type == "random":

        print("\nUsing RANDOM embeddings\n")

        dicts["src"] = initVocabulary(
            "source",
            opt.train_src,
            opt.src_vocab_size,
            opt.save_data + ".src.dict"
        )

        dicts["tgt"] = initVocabulary(
            "target",
            opt.train_tgt,
            opt.tgt_vocab_size,
            opt.save_data + ".tgt.dict"
        )

    # ========================================================
    # BERT MODE
    # ========================================================

    elif opt.embedding_type == "bert":

        print("\nUsing BERT embeddings\n")

#        dicts["src"] = initBertVocabulary(
#            "source",
#            opt.src_bert_model,
#            opt.save_data + ".src.dict"
#        )

        dicts["src"] = initBertVocabulary(
            "source",
            opt.src_bert_model,
            opt.save_data + ".src.dict",
            corpus_files=[
                opt.train_xe_src
            ],
            vocab_size=opt.src_vocab_size
        )

#        dicts["tgt"] = initBertVocabulary(
#            "target",
#            opt.tgt_bert_model,
#            opt.save_data + ".tgt.dict"
#        )



        dicts["tgt"] = initBertVocabulary(
            "target",
            opt.tgt_bert_model,
            opt.save_data + ".tgt.dict",
            corpus_files=[
                opt.train_xe_tgt
            ],
            vocab_size=opt.tgt_vocab_size
        )
    # ========================================================
    # SAVE DATA
    # ========================================================

    save_data = {}

    save_data["dicts"] = dicts

    save_data["embedding_type"] = (
        opt.embedding_type
    )

    # --------------------------------------------------------
    # TRAIN XE
    # --------------------------------------------------------

    save_data["train_xe"] = makeDataGeneral(
        "train_xe",
        opt.train_xe_src,
        opt.train_xe_tgt,
        dicts,
        opt.embedding_type
    )

    # --------------------------------------------------------
    # TRAIN PG
    # --------------------------------------------------------

    save_data["train_pg"] = makeDataGeneral(
        "train_pg",
        opt.train_pg_src,
        opt.train_pg_tgt,
        dicts,
        opt.embedding_type
    )

    # --------------------------------------------------------
    # VALID
    # --------------------------------------------------------

    save_data["valid"] = makeDataGeneral(
        "valid",
        opt.valid_src,
        opt.valid_tgt,
        dicts,
        opt.embedding_type
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    save_data["test"] = makeDataGeneral(
        "test",
        opt.test_src,
        opt.test_tgt,
        dicts,
        opt.embedding_type
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_path = opt.save_data + "-train.pt"

    print(f"\nSaving data to: {save_path}")

    torch.save(save_data, save_path)

    print("\nFinished preprocessing.\n")


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
