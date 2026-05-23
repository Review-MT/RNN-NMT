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
# 2. Static BERT Embeddings
#    - BERT tokenizer
#    - WordPiece tokenization
#
# ============================================================

import argparse
import torch

from transformers import BertTokenizer

import lib


# ============================================================
# Argument Parser
# ============================================================

parser = argparse.ArgumentParser(
    description="preprocess.py"
)

# ------------------------------------------------------------
# Dataset paths
# ------------------------------------------------------------

parser.add_argument(
    "-train_src",
    required=True
)

parser.add_argument(
    "-train_tgt",
    required=True
)

parser.add_argument(
    "-train_xe_src",
    required=True
)

parser.add_argument(
    "-train_xe_tgt",
    required=True
)

parser.add_argument(
    "-train_pg_src",
    required=True
)

parser.add_argument(
    "-train_pg_tgt",
    required=True
)

parser.add_argument(
    "-valid_src",
    required=True
)

parser.add_argument(
    "-valid_tgt",
    required=True
)

parser.add_argument(
    "-test_src",
    required=True
)

parser.add_argument(
    "-test_tgt",
    required=True
)

# ------------------------------------------------------------
# Save path
# ------------------------------------------------------------

parser.add_argument(
    "-save_data",
    required=True
)

# ------------------------------------------------------------
# Vocabulary sizes
# (used ONLY for random embedding mode)
# ------------------------------------------------------------

parser.add_argument(
    "-src_vocab_size",
    type=int,
    default=50000
)

parser.add_argument(
    "-tgt_vocab_size",
    type=int,
    default=50000
)

# ------------------------------------------------------------
# Sequence length
# ------------------------------------------------------------

parser.add_argument(
    "-seq_length",
    type=int,
    default=100
)

# ------------------------------------------------------------
# Embedding type
# ------------------------------------------------------------

parser.add_argument(
    "-embedding_type",
    choices=["random", "bert"],
    default="bert"
)

# ------------------------------------------------------------
# BERT model
# ------------------------------------------------------------

parser.add_argument(
    "-bert_model",
    type=str,
    default="bert-base-uncased"
)

# ------------------------------------------------------------
# Misc
# ------------------------------------------------------------

parser.add_argument(
    "-seed",
    type=int,
    default=3435
)

parser.add_argument(
    "-report_every",
    type=int,
    default=100000
)

opt = parser.parse_args()

torch.manual_seed(opt.seed)


# ============================================================
# RANDOM VOCABULARY CREATION
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

                # lowercasing only for random embeddings
                vocab.add(word.lower())

    originalSize = vocab.size()

    vocab = vocab.prune(size)

    print(
        "Created dictionary of size %d "
        "(pruned from %d)"
        % (vocab.size(), originalSize)
    )

    return vocab


def initVocabulary(name, dataFile,
                   vocabSize, saveFile):

    print(f"Building {name} vocabulary...")

    vocab = makeVocabulary(
        dataFile,
        vocabSize
    )

    print(
        f"Saving {name} vocabulary "
        f"to \"{saveFile}\"..."
    )

    vocab.writeFile(saveFile)

    return vocab


# ============================================================
# RANDOM EMBEDDING DATA PROCESSING
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

    print(
        f"Processing RANDOM mode: "
        f"{srcFile} & {tgtFile}"
    )

    srcF = open(srcFile, encoding="utf-8")
    tgtF = open(tgtFile, encoding="utf-8")

    while True:

        srcWords = srcF.readline().strip().split()
        tgtWords = tgtF.readline().strip().split()

        # ----------------------------------------------------
        # End of file
        # ----------------------------------------------------
        if not srcWords or not tgtWords:

            if (srcWords and not tgtWords) or \
               (not srcWords and tgtWords):

                print(
                    "WARNING: source and target "
                    "do not have same number of lines"
                )

            break

        # ----------------------------------------------------
        # Length filtering
        # ----------------------------------------------------
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

            print(
                f"... {count} sentences processed"
            )

    srcF.close()
    tgtF.close()

    assert len(src) == len(tgt)

    print(
        f"Prepared {len(src)} sentence pairs "
        f"({ignored} ignored)"
    )

    return src, tgt, range(len(src))


# ============================================================
# BERT EMBEDDING DATA PROCESSING
# ============================================================

def makeDataBERT(
    which,
    srcFile,
    tgtFile,
    srcTokenizer,
    tgtTokenizer
):

    src = []
    tgt = []

    sizes = []

    count = 0
    ignored = 0

    print(
        f"Processing BERT mode: "
        f"{srcFile} & {tgtFile}"
    )

    srcF = open(srcFile, encoding="utf-8")
    tgtF = open(tgtFile, encoding="utf-8")

    while True:

        src_sent = srcF.readline().strip()
        tgt_sent = tgtF.readline().strip()

        # ----------------------------------------------------
        # End of file
        # ----------------------------------------------------
        if not src_sent or not tgt_sent:

            if (src_sent and not tgt_sent) or \
               (not src_sent and tgt_sent):

                print(
                    "WARNING: source and target "
                    "do not have same number of lines"
                )

            break

        # ----------------------------------------------------
        # BERT tokenization
        # ----------------------------------------------------
        src_ids = srcTokenizer.encode(
            src_sent,
            add_special_tokens=True,
            truncation=True,
            max_length=opt.seq_length
        )

        tgt_ids = tgtTokenizer.encode(
            tgt_sent,
            add_special_tokens=True,
            truncation=True,
            max_length=opt.seq_length
        )

        # ----------------------------------------------------
        # Length filtering
        # ----------------------------------------------------
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

            print(
                f"... {count} sentences processed"
            )

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

    print(
        f"Preparing {which} "
        f"({embedding_type})"
    )

    res = {}

    # --------------------------------------------------------
    # RANDOM EMBEDDINGS
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
    # BERT EMBEDDINGS
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
    # RANDOM EMBEDDING MODE
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
    # BERT EMBEDDING MODE
    # ========================================================
    elif opt.embedding_type == "bert":

        print("\nUsing STATIC BERT embeddings\n")

        src_tokenizer = BertTokenizer.from_pretrained(
            opt.bert_model
        )

        tgt_tokenizer = BertTokenizer.from_pretrained(
            opt.bert_model
        )

        dicts["src"] = src_tokenizer
        dicts["tgt"] = tgt_tokenizer

        # ----------------------------------------------------
        # Save tokenizer special token ids
        # ----------------------------------------------------
        dicts["special"] = {
            "PAD": src_tokenizer.pad_token_id,
            "UNK": src_tokenizer.unk_token_id,
            "BOS": src_tokenizer.cls_token_id,
            "EOS": src_tokenizer.sep_token_id
        }

    # ========================================================
    # SAVE DATA
    # ========================================================
    save_data = {}

    save_data["dicts"] = dicts

    save_data["embedding_type"] = (
        opt.embedding_type
    )

    # --------------------------------------------------------
    # Train XE
    # --------------------------------------------------------
    save_data["train_xe"] = makeDataGeneral(
        "train_xe",
        opt.train_xe_src,
        opt.train_xe_tgt,
        dicts,
        opt.embedding_type
    )

    # --------------------------------------------------------
    # Train PG
    # --------------------------------------------------------
    save_data["train_pg"] = makeDataGeneral(
        "train_pg",
        opt.train_pg_src,
        opt.train_pg_tgt,
        dicts,
        opt.embedding_type
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------
    save_data["valid"] = makeDataGeneral(
        "valid",
        opt.valid_src,
        opt.valid_tgt,
        dicts,
        opt.embedding_type
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------
    save_data["test"] = makeDataGeneral(
        "test",
        opt.test_src,
        opt.test_tgt,
        dicts,
        opt.embedding_type
    )

    # ========================================================
    # Save preprocessing output
    # ========================================================
    save_path = opt.save_data + "-train.pt"

    print(f"\nSaving data to: {save_path}")

    torch.save(save_data, save_path)

    print("\nFinished preprocessing.\n")


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()
