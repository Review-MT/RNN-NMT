import argparse
import os
import numpy as np
import random
import time

import torch
import torch.nn as nn

from torch import cuda
from torch.autograd import Variable

import lib


# ============================================================
# Argument Parser
# ============================================================

parser = argparse.ArgumentParser(
    description="train.py"
)

# ============================================================
# DATA
# ============================================================

parser.add_argument(
    "-data",
    required=True,
    help="Path to *-train.pt"
)

parser.add_argument(
    "-save_dir",
    required=True,
    help="Directory to save models"
)

parser.add_argument(
    "-load_from",
    default=None,
    help="Load pretrained checkpoint"
)

# ============================================================
# MODEL
# ============================================================

parser.add_argument(
    "-layers",
    type=int,
    default=1
)

parser.add_argument(
    "-rnn_size",
    type=int,
    default=500
)

parser.add_argument(
    "-word_vec_size",
    type=int,
    default=500
)

parser.add_argument(
    "-input_feed",
    type=int,
    default=1
)

parser.add_argument(
    "-brnn",
    action="store_true"
)

parser.add_argument(
    "-brnn_merge",
    default="concat"
)

# ============================================================
# BERT OPTIONS
# ============================================================

# ------------------------------------------------------------
# Source embedding
# ------------------------------------------------------------

parser.add_argument(
    "-use_bert_src",
    action="store_true",
    help="Use BERT initialized source embeddings"
)

parser.add_argument(
    "-src_bert_model",
    type=str,
    default="bert-base-uncased"
)

parser.add_argument(
    "-freeze_src_bert",
    action="store_true",
    help="Freeze source BERT embeddings"
)

# ------------------------------------------------------------
# Target embedding
# ------------------------------------------------------------

parser.add_argument(
    "-use_bert_tgt",
    action="store_true",
    help="Use BERT initialized target embeddings"
)

parser.add_argument(
    "-tgt_bert_model",
    type=str,
    default="bert-base-uncased"
)

parser.add_argument(
    "-freeze_tgt_bert",
    action="store_true",
    help="Freeze target BERT embeddings"
)

# ============================================================
# OPTIMIZATION
# ============================================================

parser.add_argument(
    "-batch_size",
    type=int,
    default=32
)

parser.add_argument(
    "-max_generator_batches",
    type=int,
    default=32
)

parser.add_argument(
    "-end_epoch",
    type=int,
    default=50
)

parser.add_argument(
    "-start_epoch",
    type=int,
    default=1
)

parser.add_argument(
    "-param_init",
    type=float,
    default=0.1
)

parser.add_argument(
    "-optim",
    default="adam"
)

parser.add_argument(
    "-lr",
    type=float,
    default=1e-3
)

parser.add_argument(
    "-max_grad_norm",
    type=float,
    default=5
)

parser.add_argument(
    "-dropout",
    type=float,
    default=0
)

parser.add_argument(
    "-learning_rate_decay",
    type=float,
    default=0.5
)

parser.add_argument(
    "-start_decay_at",
    type=int,
    default=5
)

# ============================================================
# GPU
# ============================================================

parser.add_argument(
    "-gpus",
    default=[0],
    nargs="+",
    type=int
)

parser.add_argument(
    "-log_interval",
    type=int,
    default=100
)

parser.add_argument(
    "-seed",
    type=int,
    default=3435
)

# ============================================================
# CRITIC
# ============================================================

parser.add_argument(
    "-start_reinforce",
    type=int,
    default=None
)

parser.add_argument(
    "-critic_pretrain_epochs",
    type=int,
    default=0
)

parser.add_argument(
    "-reinforce_lr",
    type=float,
    default=1e-4
)

# ============================================================
# EVALUATION
# ============================================================

parser.add_argument(
    "-eval",
    action="store_true"
)

parser.add_argument(
    "-eval_sample",
    action="store_true",
    default=False
)

parser.add_argument(
    "-max_predict_length",
    type=int,
    default=50
)

# ============================================================
# REWARD SHAPING
# ============================================================

parser.add_argument(
    "-pert_func",
    type=str,
    default=None
)

parser.add_argument(
    "-pert_param",
    type=float,
    default=None
)

# ============================================================
# OTHER
# ============================================================

parser.add_argument(
    "-no_update",
    action="store_true",
    default=False
)

parser.add_argument(
    "-sup_train_on_bandit",
    action="store_true",
    default=False
)

# ============================================================
# Parse
# ============================================================

opt = parser.parse_args()

print(opt)

# ============================================================
# Seeds
# ============================================================

torch.manual_seed(opt.seed)

np.random.seed(opt.seed)

random.seed(opt.seed)

# ============================================================
# CUDA
# ============================================================

opt.cuda = len(opt.gpus)

if opt.save_dir and not os.path.exists(opt.save_dir):

    os.makedirs(opt.save_dir)

if torch.cuda.is_available() and not opt.cuda:

    print(
        "WARNING: CUDA device exists "
        "but -gpus not provided"
    )

if opt.cuda:

    cuda.set_device(opt.gpus[0])

    torch.cuda.manual_seed(opt.seed)

# ============================================================
# Initialize Parameters
# ============================================================

def init(model):

    # --------------------------------------------------------
    # Do NOT reinitialize pretrained BERT embeddings
    # --------------------------------------------------------

    for name, p in model.named_parameters():

        if "word_lut.weight" in name:

            print(
                f"Skipping initialization for: "
                f"{name}"
            )

            continue

        p.data.uniform_(
            -opt.param_init,
            opt.param_init
        )

# ============================================================
# Optimizer
# ============================================================

#def create_optim(model):

#    optim = lib.Optim(
#        model.parameters(),
#        opt.optim,
#        opt.lr,
#        opt.max_grad_norm,
#        lr_decay=opt.learning_rate_decay,
#        start_decay_at=opt.start_decay_at
#    )

#    return optim

def create_optim(model):
    optim = lib.Optim(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    opt.optim,
    opt.lr,
    opt.max_grad_norm,
    lr_decay=opt.learning_rate_decay,
    start_decay_at=opt.start_decay_at)

    return optim
# ============================================================
# Create Model
# ============================================================

def create_model(
    model_class,
    dicts,
    gen_out_size
):

    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    encoder = lib.Encoder(
        opt,
        dicts["src"]
    )

    # --------------------------------------------------------
    # Decoder
    # --------------------------------------------------------

    decoder = lib.Decoder(
        opt,
        dicts["tgt"]
    )

    # --------------------------------------------------------
    # Automatically update embedding dimension
    # when using BERT
    # --------------------------------------------------------

    if opt.use_bert_src:

        opt.word_vec_size = (
            encoder.word_lut.embedding_dim
        )

        print(
            f"Using source BERT embedding size: "
            f"{opt.word_vec_size}"
        )

    if opt.use_bert_tgt:

        opt.word_vec_size = (
            decoder.word_lut.embedding_dim
        )

        print(
            f"Using target BERT embedding size: "
            f"{opt.word_vec_size}"
        )

    # --------------------------------------------------------
    # Generator
    # --------------------------------------------------------

    if (
        opt.max_generator_batches <
        opt.batch_size
        and gen_out_size > 1
    ):

        generator = lib.MemEfficientGenerator(
            nn.Linear(
                opt.rnn_size,
                gen_out_size
            ),
            opt
        )

    else:

        generator = lib.BaseGenerator(
            nn.Linear(
                opt.rnn_size,
                gen_out_size
            ),
            opt
        )

    # --------------------------------------------------------
    # Full model
    # --------------------------------------------------------

    model = model_class(
        encoder,
        decoder,
        generator,
        opt
    )

    # --------------------------------------------------------
    # Initialize parameters
    # --------------------------------------------------------

    init(model)

    optim = create_optim(model)

    return model, optim

# ============================================================
# Create Critic
# ============================================================

def create_critic(
    checkpoint,
    dicts,
    opt
):

    if (
        opt.load_from is not None
        and "critic" in checkpoint
    ):

        critic = checkpoint["critic"]

        critic_optim = checkpoint["critic_optim"]

    else:

        critic, critic_optim = create_model(
            lib.NMTModel,
            dicts,
            1
        )

    if opt.cuda:

        critic.cuda(opt.gpus[0])

    return critic, critic_optim

# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # Load dataset
    # ========================================================

    print(
        'Loading data from "%s"' %
        opt.data
    )

    dataset = torch.load(opt.data)

    # ========================================================
    # Automatically restore embedding config
    # ========================================================

    if "src_embedding_type" in dataset:

        opt.use_bert_src = (
            dataset["src_embedding_type"]
            == "bert"
        )

    if "tgt_embedding_type" in dataset:

        opt.use_bert_tgt = (
            dataset["tgt_embedding_type"]
            == "bert"
        )

    if "src_bert_model" in dataset:

        opt.src_bert_model = (
            dataset["src_bert_model"]
        )

    if "tgt_bert_model" in dataset:

        opt.tgt_bert_model = (
            dataset["tgt_bert_model"]
        )

    print("\n================================================")
    print("Embedding Configuration")
    print("================================================")

    print(
        f"Source embeddings : "
        f"{'BERT' if opt.use_bert_src else 'RANDOM'}"
    )

    print(
        f"Target embeddings : "
        f"{'BERT' if opt.use_bert_tgt else 'RANDOM'}"
    )

    if opt.use_bert_src:

        print(
            f"Source model      : "
            f"{opt.src_bert_model}"
        )

    if opt.use_bert_tgt:

        print(
            f"Target model      : "
            f"{opt.tgt_bert_model}"
        )

    print("================================================\n")

    # ========================================================
    # Datasets
    # ========================================================

    supervised_data = lib.Dataset(
        dataset["train_xe"],
        opt.batch_size,
        opt.cuda,
        eval=False
    )

    bandit_data = lib.Dataset(
        dataset["train_pg"],
        opt.batch_size,
        opt.cuda,
        eval=False
    )

    valid_data = lib.Dataset(
        dataset["valid"],
        opt.batch_size,
        opt.cuda,
        eval=True
    )

    test_data = lib.Dataset(
        dataset["test"],
        opt.batch_size,
        opt.cuda,
        eval=True
    )

    dicts = dataset["dicts"]

    # ========================================================
    # Dataset statistics
    # ========================================================

    print(
        " * vocabulary size. "
        "source = %d; target = %d"
        % (
            dicts["src"].size(),
            dicts["tgt"].size()
        )
    )

    print(
        " * number of XENT "
        "training sentences. %d"
        % len(dataset["train_xe"]["src"])
    )

    print(
        " * number of PG "
        "training sentences. %d"
        % len(dataset["train_pg"]["src"])
    )

    print(
        " * maximum batch size. %d"
        % opt.batch_size
    )

    print("\nBuilding model...\n")

    # ========================================================
    # Reinforcement learning?
    # ========================================================

    use_critic = (
        opt.start_reinforce is not None
    )

    # ========================================================
    # Create model
    # ========================================================

    if opt.load_from is None:

        model, optim = create_model(
            lib.NMTModel,
            dicts,
            dicts["tgt"].size()
        )

        checkpoint = None

    else:

        print(
            f"Loading checkpoint: "
            f"{opt.load_from}"
        )

        checkpoint = torch.load(
            opt.load_from
        )

        model = checkpoint["model"]

        optim = checkpoint["optim"]

        opt.start_epoch = (
            checkpoint["epoch"] + 1
        )

    # ========================================================
    # GPU
    # ========================================================

    if opt.cuda:

        model.cuda(opt.gpus[0])

    # ========================================================
    # Start RL immediately
    # ========================================================

    if opt.start_reinforce == -1:

        opt.start_decay_at = (
            opt.start_epoch
        )

        opt.start_reinforce = (
            opt.start_epoch
        )

    # ========================================================
    # Epoch validation
    # ========================================================

    if use_critic:

        assert (
            opt.start_epoch +
            opt.critic_pretrain_epochs - 1
            <= opt.end_epoch
        ), (
            "Increase -end_epoch "
            "for critic pretraining"
        )

    # ========================================================
    # Parameters
    # ========================================================

    nParams = sum([
        p.nelement()
        for p in model.parameters()
    ])

    print(
        f"* number of parameters: "
        f"{nParams}"
    )

    # ========================================================
    # Metrics
    # ========================================================

    metrics = {}

    metrics["nmt_loss"] = (
        lib.Loss.weighted_xent_loss
    )

    metrics["critic_loss"] = (
        lib.Loss.weighted_mse
    )

    metrics["sent_reward"] = (
        lib.Reward.sentence_bleu
    )

    metrics["corp_reward"] = (
        lib.Reward.corpus_bleu
    )

    if opt.pert_func is not None:

        opt.pert_func = lib.PertFunction(
            opt.pert_func,
            opt.pert_param
        )

    # ========================================================
    # Evaluation only
    # ========================================================

    if opt.eval:

        evaluator = lib.Evaluator(
            model,
            metrics,
            dicts,
            opt
        )

        pred_file = (
            opt.load_from.replace(
                ".pt",
                ".valid.pred"
            )
        )

        evaluator.eval(
            valid_data,
            pred_file
        )

        pred_file = (
            opt.load_from.replace(
                ".pt",
                ".test.pred"
            )
        )

        evaluator.eval(
            test_data,
            pred_file
        )

    # ========================================================
    # Eval sample
    # ========================================================

    elif opt.eval_sample:

        opt.no_update = True

        critic, critic_optim = create_critic(
            checkpoint,
            dicts,
            opt
        )

        reinforce_trainer = (
            lib.ReinforceTrainer(
                model,
                critic,
                bandit_data,
                test_data,
                metrics,
                dicts,
                optim,
                critic_optim,
                opt
            )
        )

        reinforce_trainer.train(
            opt.start_epoch,
            opt.start_epoch,
            False
        )

    # ========================================================
    # Supervised training on bandit
    # ========================================================

    elif opt.sup_train_on_bandit:

        optim.set_lr(
            opt.reinforce_lr
        )

        xent_trainer = lib.Trainer(
            model,
            bandit_data,
            test_data,
            metrics,
            dicts,
            optim,
            opt
        )

        xent_trainer.train(
            opt.start_epoch,
            opt.start_epoch
        )

    # ========================================================
    # Main Training
    # ========================================================

    else:

        xent_trainer = lib.Trainer(
            model,
            supervised_data,
            valid_data,
            metrics,
            dicts,
            optim,
            opt
        )

        # ----------------------------------------------------
        # Reinforcement learning
        # ----------------------------------------------------

        if use_critic:

            start_time = time.time()

            xent_trainer.train(
                opt.start_epoch,
                opt.start_reinforce - 1,
                start_time
            )

            critic, critic_optim = create_critic(
                checkpoint,
                dicts,
                opt
            )

            if opt.critic_pretrain_epochs > 0:

                reinforce_trainer = (
                    lib.ReinforceTrainer(
                        model,
                        critic,
                        supervised_data,
                        test_data,
                        metrics,
                        dicts,
                        optim,
                        critic_optim,
                        opt
                    )
                )

                reinforce_trainer.train(
                    opt.start_reinforce,
                    opt.start_reinforce +
                    opt.critic_pretrain_epochs - 1,
                    True,
                    start_time
                )

            reinforce_trainer = (
                lib.ReinforceTrainer(
                    model,
                    critic,
                    bandit_data,
                    test_data,
                    metrics,
                    dicts,
                    optim,
                    critic_optim,
                    opt
                )
            )

            reinforce_trainer.train(
                opt.start_reinforce +
                opt.critic_pretrain_epochs,
                opt.end_epoch,
                False,
                start_time
            )

        # ----------------------------------------------------
        # Pure supervised training
        # ----------------------------------------------------

        else:

            xent_trainer.train(
                opt.start_epoch,
                opt.end_epoch
            )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()
