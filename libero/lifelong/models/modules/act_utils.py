import math
import numpy as np
from torch import nn
import torch
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from positional_encodings.torch_encodings import PositionalEncoding1D

###############################################################################
#
# MLP projection module
#
###############################################################################


class MLP_Proj(nn.Module):
    """
    Encode any embedding

    h = f(e), where
        e: embedding from some model
        h: latent embedding (B, H)
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        assert num_layers >= 1, "[error] num_layers < 1"
        sizes = [input_size] + [hidden_size] * (num_layers - 1) + [output_size]
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.projection = nn.Sequential(*layers)

    def forward(self, data):
        """
        data:
            task_emb: (B, E)
        """
        h = self.projection(data)  # (B, H)
        return h

class ACTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.transformer_encoder = GPT_Encoder(GPT_Encoder_Config(cfg.gpt_encoder))
        self.transformer_decoder = GPT_Decoder(GPT_Decoder_Config(cfg.gpt_decoder))
    
    def forward(self, init_obs):
        """
        actions: (B, At, A)
        init_obs: (B, Ot, O)
        """
        x = self.transformer_encoder(init_obs)
        x = self.transformer_decoder(x)
        return x

###############################################################################
#
# GPT Encoder
#
###############################################################################

class GPT_Encoder_Config:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    discrete_input = False
    causal_attention = False
    vocab_size = 0
    n_embd = 256
    n_layer = 2
    n_head = 4
    block_size = 128
    def __init__(self, kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class GPT_Encoder(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(self, config: GPT_Encoder_Config):
        super().__init__()
        self.discrete_input = config.discrete_input
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[EncoderBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.block_size = config.block_size
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, GPT_Encoder):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    def forward(self, idx):
        b, t, dim = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        x = self.drop(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        return x

###############################################################################
#
# GPT Decoder
#
###############################################################################

class GPT_Decoder_Config:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    discrete_input = False
    causal_attention = True
    vocab_size = 0
    n_embd = 256
    n_layer = 2
    n_head = 4
    output_dim = 7
    block_size = 32
    def __init__(self, kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class GPT_Decoder(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(self, config: GPT_Decoder_Config):
        super().__init__()
        self.discrete_input = config.discrete_input
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.fixed_pos_emb = PositionalEncoding1D(config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[DecoderBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.output_dim, bias=False)
        self.block_size = config.block_size
        self.n_emb = config.n_embd
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, GPT_Decoder):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    def forward(self, context):
        x = self.fixed_pos_emb(torch.zeros((context.size(0), self.block_size, self.n_emb), dtype=context.dtype, device=context.device))
        x,_ = self.blocks((x, context))
        x = self.ln_f(x)
        x = self.head(x)
        return x

###############################################################################
#
# GPT submodules
#
###############################################################################

class EncoderBlock(nn.Module):
    """an unassuming Transformer block"""
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class DecoderBlock(nn.Module):
    """an unassuming Transformer block"""
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.masked_attn = SelfAttention(config)
        self.cross_attn = CrossAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, inputs):
        x, context = inputs
        x = x + self.masked_attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), context)
        x = x + self.mlp(self.ln2(x))
        return x, context

class CrossAttention(nn.Module):
    """
    A vanilla multi-head masked cross-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head

    def forward(self, x, context):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        (
            B,
            Tc,
            C,
        ) = context.size()  # batch size, encoder sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(context).view(B, Tc, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, Tc, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(context).view(B, Tc, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, Tc, hs)

        # cross-attention; Attend: (B, nh, T, hs) x (B, nh, hs, Tc) -> (B, nh, T, Tc)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, Tc) x (B, nh, Tc, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
    

class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.use_causal_attention = config.causal_attention
        # causal mask to ensure that attention is only applied to the left in the input sequence
        if self.use_causal_attention:
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )
        self.n_head = config.n_head

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.use_causal_attention:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        # att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

