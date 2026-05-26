# ============================================================
# NMT MODEL
# Supports:
#
# 1. Random embeddings
# 2. BERT initialized embeddings
# 3. Different source/target vocabularies
# 4. Proper BOS/EOS handling for:
#       - custom vocab
#       - BERT vocab
#
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from torch.nn.utils.rnn import (
    pad_packed_sequence as unpack
)

from torch.nn.utils.rnn import (
    pack_padded_sequence as pack
)

from transformers import (
    AutoModel,
    AutoTokenizer
)

import lib


# ============================================================
# Embedding Builder
# ============================================================

def build_embedding(
    vocab_size,
    word_vec_size,
    padding_idx,
    use_bert=False,
    bert_model_name=None,
    freeze=False
):

    # --------------------------------------------------------
    # RANDOM EMBEDDINGS
    # --------------------------------------------------------

    if not use_bert:

        emb = nn.Embedding(
            vocab_size,
            word_vec_size,
            padding_idx=padding_idx
        )

        return emb

    # --------------------------------------------------------
    # BERT EMBEDDINGS
    # --------------------------------------------------------

    print(
        f"Loading pretrained embeddings: "
        f"{bert_model_name}"
    )

    bert_model = AutoModel.from_pretrained(
        bert_model_name
    )

    bert_embeddings = (
        bert_model.embeddings.word_embeddings
    )

    pretrained_weight = (
        bert_embeddings.weight.data.clone()
    )

    emb = nn.Embedding.from_pretrained(
        pretrained_weight,
        freeze=freeze,
        padding_idx=padding_idx
    )

    print(
        f"Loaded embedding matrix: "
        f"{pretrained_weight.size()}"
    )

    return emb


# ============================================================
# Encoder
# ============================================================

class Encoder(nn.Module):

    def __init__(self, opt, dicts):

        super(Encoder, self).__init__()

        self.layers = opt.layers

        self.num_directions = (
            2 if opt.brnn else 1
        )

        assert (
            opt.rnn_size %
            self.num_directions == 0
        )

        self.hidden_size = (
            opt.rnn_size //
            self.num_directions
        )

        # ====================================================
        # EMBEDDING TYPE
        # ====================================================

        self.use_bert = getattr(
            opt,
            "use_bert_src",
            False
        )

        # ====================================================
        # SPECIAL TOKENS
        # ====================================================

        if self.use_bert:

            tokenizer = AutoTokenizer.from_pretrained(
                opt.src_bert_model
            )

            self.pad_id = tokenizer.pad_token_id

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

        else:

            self.pad_id = lib.Constants.PAD
            self.bos_id = lib.Constants.BOS
            self.eos_id = lib.Constants.EOS

        # ====================================================
        # EMBEDDING LAYER
        # ====================================================

        self.word_lut = build_embedding(
            vocab_size=dicts.size(),
            word_vec_size=opt.word_vec_size,
            padding_idx=self.pad_id,
            use_bert=self.use_bert,
            bert_model_name=getattr(
                opt,
                "src_bert_model",
                None
            ),
            freeze=getattr(
                opt,
                "freeze_src_bert",
                False
            )
        )

        embedding_dim = (
            self.word_lut.embedding_dim
        )

        print(
            f"Encoder embedding dim: "
            f"{embedding_dim}"
        )

        # ====================================================
        # RNN
        # ====================================================

        self.rnn = nn.LSTM(
            embedding_dim,
            self.hidden_size,
            num_layers=opt.layers,
            dropout=opt.dropout,
            bidirectional=opt.brnn
        )

    def forward(
        self,
        inputs,
        hidden=None
    ):

        emb = self.word_lut(inputs[0])

        emb = pack(
            emb,
            inputs[1]
        )

        outputs, hidden_t = self.rnn(
            emb,
            hidden
        )

        outputs = unpack(outputs)[0]

        return hidden_t, outputs


# ============================================================
# STACKED LSTM
# ============================================================

class StackedLSTM(nn.Module):

    def __init__(
        self,
        num_layers,
        input_size,
        rnn_size,
        dropout
    ):

        super(StackedLSTM, self).__init__()

        self.dropout = nn.Dropout(
            dropout
        )

        self.num_layers = num_layers

        self.layers = nn.ModuleList()

        for i in range(num_layers):

            self.layers.append(
                nn.LSTMCell(
                    input_size,
                    rnn_size
                )
            )

            input_size = rnn_size

    def forward(
        self,
        inputs,
        hidden
    ):

        h_0, c_0 = hidden

        h_1, c_1 = [], []

        for i, layer in enumerate(self.layers):

            h_1_i, c_1_i = layer(
                inputs,
                (h_0[i], c_0[i])
            )

            inputs = h_1_i

            if i != self.num_layers - 1:

                inputs = self.dropout(inputs)

            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return inputs, (h_1, c_1)


# ============================================================
# Decoder
# ============================================================

class Decoder(nn.Module):

    def __init__(self, opt, dicts):

        super(Decoder, self).__init__()

        self.layers = opt.layers

        self.input_feed = opt.input_feed

        # ====================================================
        # EMBEDDING TYPE
        # ====================================================

        self.use_bert = getattr(
            opt,
            "use_bert_tgt",
            False
        )

        # ====================================================
        # SPECIAL TOKENS
        # ====================================================

        if self.use_bert:

            tokenizer = AutoTokenizer.from_pretrained(
                opt.tgt_bert_model
            )

            self.pad_id = tokenizer.pad_token_id

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

        else:

            self.pad_id = lib.Constants.PAD
            self.bos_id = lib.Constants.BOS
            self.eos_id = lib.Constants.EOS

        # ====================================================
        # EMBEDDING LAYER
        # ====================================================

        self.word_lut = build_embedding(
            vocab_size=dicts.size(),
            word_vec_size=opt.word_vec_size,
            padding_idx=self.pad_id,
            use_bert=self.use_bert,
            bert_model_name=getattr(
                opt,
                "tgt_bert_model",
                None
            ),
            freeze=getattr(
                opt,
                "freeze_tgt_bert",
                False
            )
        )

        embedding_dim = (
            self.word_lut.embedding_dim
        )

        print(
            f"Decoder embedding dim: "
            f"{embedding_dim}"
        )

        # ====================================================
        # INPUT SIZE
        # ====================================================

        input_size = embedding_dim

        if self.input_feed:

            input_size += opt.rnn_size

        # ====================================================
        # DECODER RNN
        # ====================================================

        self.rnn = StackedLSTM(
            opt.layers,
            input_size,
            opt.rnn_size,
            opt.dropout
        )

        self.attn = lib.GlobalAttention(
            opt.rnn_size
        )

        self.dropout = nn.Dropout(
            opt.dropout
        )

        self.hidden_size = opt.rnn_size

    def step(
        self,
        emb,
        output,
        hidden,
        context
    ):

        if self.input_feed:

            emb = torch.cat(
                [emb, output],
                1
            )

        output, hidden = self.rnn(
            emb,
            hidden
        )

        output, attn = self.attn(
            output,
            context
        )

        output = self.dropout(output)

        return output, hidden

    def forward(
        self,
        inputs,
        init_states
    ):

        emb, output, hidden, context = (
            init_states
        )

        embs = self.word_lut(inputs)

        outputs = []

        for i in range(inputs.size(0)):

            output, hidden = self.step(
                emb,
                output,
                hidden,
                context
            )

            outputs.append(output)

            emb = embs[i]

        outputs = torch.stack(outputs)

        return outputs


# ============================================================
# NMT MODEL
# ============================================================

class NMTModel(nn.Module):

    def __init__(
        self,
        encoder,
        decoder,
        generator,
        opt
    ):

        super(NMTModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.opt = opt

    def make_init_decoder_output(
        self,
        context
    ):

        batch_size = context.size(1)

        h_size = (
            batch_size,
            self.decoder.hidden_size
        )

        return Variable(
            context.data.new(*h_size).zero_(),
            requires_grad=False
        )

    def _fix_enc_hidden(self, h):

        if self.encoder.num_directions == 2:

            return h.view(
                h.size(0) // 2,
                2,
                h.size(1),
                h.size(2)
            ).transpose(1, 2).contiguous().view(
                h.size(0) // 2,
                h.size(1),
                h.size(2) * 2
            )

        else:

            return h

    # ========================================================
    # Initialize Decoder
    # ========================================================

    def initialize(
        self,
        inputs,
        eval
    ):

        src = inputs[0]
        tgt = inputs[1]

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        enc_hidden, context = self.encoder(
            src
        )

        init_output = (
            self.make_init_decoder_output(
                context
            )
        )

        # ----------------------------------------------------
        # Fix hidden shape
        # ----------------------------------------------------

        enc_hidden = (
            self._fix_enc_hidden(
                enc_hidden[0]
            ),
            self._fix_enc_hidden(
                enc_hidden[1]
            )
        )

        # ----------------------------------------------------
        # BOS TOKEN
        # ----------------------------------------------------

        batch_size = init_output.size(0)

        bos_id = self.decoder.bos_id

        init_token = Variable(
            torch.LongTensor(
                [bos_id] * batch_size
            ),
            volatile=eval
        )

        if self.opt.cuda:

            init_token = init_token.cuda()

        # ----------------------------------------------------
        # Initial embedding
        # ----------------------------------------------------

        emb = self.decoder.word_lut(
            init_token
        )

        return tgt, (
            emb,
            init_output,
            enc_hidden,
            context.transpose(0, 1)
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        inputs,
        eval,
        regression=False
    ):

        targets, init_states = (
            self.initialize(
                inputs,
                eval
            )
        )

        outputs = self.decoder(
            targets,
            init_states
        )

        if regression:

            logits = self.generator(outputs)

            return logits.view_as(targets)

        return outputs

    # ========================================================
    # Backward
    # ========================================================

    def backward(
        self,
        outputs,
        targets,
        weights,
        normalizer,
        criterion,
        regression=False
    ):

        grad_output, loss = (
            self.generator.backward(
                outputs,
                targets,
                weights,
                normalizer,
                criterion,
                regression
            )
        )

        outputs.backward(grad_output)

        return loss

    # ========================================================
    # Predict
    # ========================================================

    def predict(
        self,
        outputs,
        targets,
        weights,
        criterion
    ):

        return self.generator.predict(
            outputs,
            targets,
            weights,
            criterion
        )

    # ========================================================
    # Greedy Translation
    # ========================================================

    def translate(
        self,
        inputs,
        max_length
    ):

        targets, init_states = (
            self.initialize(
                inputs,
                eval=True
            )
        )

        emb, output, hidden, context = (
            init_states
        )

        preds = []

        batch_size = targets.size(1)

        num_eos = targets[0].data.byte().new(
            batch_size
        ).zero_()

        for i in range(max_length):

            output, hidden = self.decoder.step(
                emb,
                output,
                hidden,
                context
            )

            logit = self.generator(output)

            pred = logit.max(1)[1].view(-1).data

            preds.append(pred)

            # ------------------------------------------------
            # Stop if EOS
            # ------------------------------------------------

            num_eos |= (
                pred == self.decoder.eos_id
            )

            if num_eos.sum() == batch_size:

                break

            emb = self.decoder.word_lut(
                Variable(pred)
            )

        preds = torch.stack(preds)

        return preds

    # ========================================================
    # Sampling
    # ========================================================

    def sample(
        self,
        inputs,
        max_length
    ):

        targets, init_states = (
            self.initialize(
                inputs,
                eval=False
            )
        )

        emb, output, hidden, context = (
            init_states
        )

        outputs = []
        samples = []

        batch_size = targets.size(1)

        num_eos = targets[0].data.byte().new(
            batch_size
        ).zero_()

        for i in range(max_length):

            output, hidden = self.decoder.step(
                emb,
                output,
                hidden,
                context
            )

            outputs.append(output)

            dist = F.softmax(
                self.generator(output),
                dim=-1
            )

            sample = dist.multinomial(
                1,
                replacement=False
            ).view(-1).data

            samples.append(sample)

            num_eos |= (
                sample == self.decoder.eos_id
            )

            if num_eos.sum() == batch_size:

                break

            emb = self.decoder.word_lut(
                Variable(sample)
            )

        outputs = torch.stack(outputs)
        samples = torch.stack(samples)

        return samples, outputs
