import math
import numpy as np
from torch import nn
import torch
from libero.lifelong.models.modules.skill_utils import ResidualTemporalBlock, ResidualTemporalDeConvBlock
from vector_quantize_pytorch import VectorQuantize, FSQ
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer

class SkillVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.vq_type == 'vq':
            self.vq = VectorQuantize(dim=cfg.encoder_dim, codebook_dim=cfg.codebook_dim, codebook_size=cfg.codebook_size)
        elif cfg.vq_type == 'fsq':
            self.vq = FSQ(dim=cfg.encoder_dim, levels=cfg.fsq_level)
        else:
            raise NotImplementedError('Unknown vq_type')
        self.action_proj = nn.Linear(cfg.action_dim, cfg.encoder_dim)
        self.action_head = nn.Linear(cfg.decoder_dim, cfg.action_dim)
        self.conv_block = ResidualTemporalBlock(
            cfg.encoder_dim, cfg.encoder_dim, kernel_size=cfg.kernel_sizes, stride=cfg.strides, causal=cfg.use_causal_encoder)
        self.deconv_block = ResidualTemporalDeConvBlock(
            cfg.decoder_dim, cfg.decoder_dim, kernel_size=cfg.kernel_sizes, stride=cfg.strides, causal=cfg.use_causal_decoder)

        encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.encoder_dim, nhead=cfg.encoder_heads, dim_feedforward=4*cfg.encoder_dim, dropout=cfg.attn_pdrop, activation='gelu', batch_first=True, norm_first=True)
        self.encoder =  nn.TransformerEncoder(encoder_layer, num_layers=cfg.encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=cfg.decoder_dim, nhead=cfg.decoder_heads, dim_feedforward=4*cfg.decoder_dim, dropout=cfg.attn_pdrop, activation='gelu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.decoder_layers)
        self.add_positional_emb = Summer(PositionalEncoding1D(cfg.encoder_dim))
        self.fixed_positional_emb = PositionalEncoding1D(cfg.decoder_dim)
    
    def encode(self, act):
        x = self.action_proj(act)
        x = self.conv_block(x)
        x = self.add_positional_emb(x)
        if self.cfg.use_causal_encoder:
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
            x = self.encoder(x, mask=mask, is_causal=True)
        else:
            x = self.encoder(x)
        return x

    def quantize(self, z):
        if self.cfg.vq_type == 'vq':
            codes, indices, commitment_loss = self.vq(z)
            pp = torch.tensor(torch.unique(indices).shape[0] / self.vq.codebook_size).to(z.device)
        else:
            codes, indices = self.vq(z)
            commitment_loss = torch.tensor([0.0]).to(z.device)
            pp = torch.tensor(torch.unique(indices).shape[0] / self.vq.codebook_size).to(z.device)
        pp_sample = torch.tensor(np.mean([len(torch.unique(index_seq)) for index_seq in indices])/z.shape[1]).to(z.device)
        return codes, indices, pp, pp_sample, commitment_loss

    def decode(self, codes):
        x = self.fixed_positional_emb(torch.zeros((codes.shape[0], self.cfg.skill_block_size, self.cfg.decoder_dim), dtype=codes.dtype, device=codes.device))
        if self.cfg.use_causal_decoder:
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
            x = self.decoder(x, codes, tgt_mask=mask, tgt_is_causal=True)
        else:
            x = self.decoder(x, codes)
        x = self.action_head(x)
        return x

    def forward(self, act):
        z = self.encode(act)
        codes, _, pp, pp_sample, commitment_loss = self.quantize(z)
        x = self.decode(codes)
        return x, pp, pp_sample, commitment_loss

    def get_indices(self, act):
        z = self.encode(act)
        _, indices, _, _, _ = self.quantize(z)
        return indices
    
    def decode_actions(self, indices):
        if self.cfg.vq_type == 'fsq':
            codes = self.vq.indices_to_codes(indices)
        else:
            codes = self.vq.get_output_from_indices(indices)
        x = self.decode(codes)
        return x

    @property
    def device(self):
        return next(self.parameters()).device