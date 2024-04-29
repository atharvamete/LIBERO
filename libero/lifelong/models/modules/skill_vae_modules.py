import math
import numpy as np
from torch import nn
import torch
from libero.lifelong.models.modules.skill_utils import ResidualTemporalBlock, ResidualTemporalDeConvBlock, GPTConfig, GPT
from vector_quantize_pytorch import VectorQuantize, FSQ
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer

class SkillVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if cfg.vq_type == 'vq':
            latent_dim = cfg.codebook_dim
            self.vq = VectorQuantize(dim=cfg.codebook_dim, codebook_size=cfg.codebook_size) # codebook size
        elif cfg.vq_type == 'fsq':
            latent_dim = len(cfg.fsq_level)
            self.vq = FSQ(cfg.fsq_level)
        else:
            raise NotImplementedError('Unknown vq_type')
        self.obs_proj = nn.Linear(cfg.cat_obs_dim, cfg.decoder_dim)
        self.action_proj = nn.Linear(cfg.action_dim, cfg.encoder_dim)
        self.encoder_latent_proj = nn.Linear(cfg.encoder_dim, latent_dim)
        self.latent_decoder_proj = nn.Linear(latent_dim, cfg.decoder_dim)
        self.action_head = nn.Linear(cfg.decoder_dim, cfg.action_dim)
        self.conv_block = ResidualTemporalBlock(
            cfg.encoder_dim, cfg.encoder_dim, kernel_size=cfg.kernel_sizes, stride=cfg.strides, causal=cfg.use_causal_encoder)
        self.deconv_block = ResidualTemporalDeConvBlock(
            cfg.decoder_dim, cfg.decoder_dim, kernel_size=cfg.kernel_sizes, stride=cfg.strides, causal=cfg.use_causal_decoder)

        encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.encoder_dim, nhead=cfg.encoder_heads, dim_feedforward=4*cfg.encoder_dim, dropout=cfg.attn_pdrop, activation='gelu', batch_first=True)
        self.encoder =  nn.TransformerEncoder(encoder_layer, num_layers=cfg.encoder_layers)
        self.add_positional_emb = Summer(PositionalEncoding1D(cfg.encoder_dim))
        self.fixed_positional_emb = PositionalEncoding1D(cfg.decoder_dim)

        decoder_input_dim = cfg.obs_emb_dim if cfg.cross_z else latent_dim
        gpt_block_size = cfg.skill_block_size if cfg.use_m4 else cfg.skill_block_size//(2**(cfg.strides.count(2)))
        gpt_config = GPTConfig(vocab_size=0, 
                                block_size=gpt_block_size, 
                                input_size=decoder_input_dim, 
                                discrete_input=False,
                                n_layer=cfg.decoder_layers,
                                n_head=cfg.decoder_heads,
                                n_embd=cfg.decoder_dim,
                                causal_attention=cfg.use_causal_decoder,)
        self.decoder = GPT(gpt_config)
    
    def encode(self, act):
        x = self.action_proj(act)
        x = self.conv_block(x)
        x = self.add_positional_emb(x)
        if self.cfg.use_causal_encoder:
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
            x = self.encoder(x,mask=mask)
        else:
            x = self.encoder(x)
        x = self.encoder_latent_proj(x)
        return x

    def quantize(self, z):
        if self.cfg.vq_type == 'vq':
            codes, indices, commitment_loss = self.vq(z)
            pp = torch.tensor(torch.unique(indices).shape[0] / self.vq.codebook_size).to(z.device)
        elif self.cfg.vq_type == 'fsq':
            codes, indices = self.vq(z)
            commitment_loss = torch.tensor(1.).to(z.device)
            pp = torch.tensor(torch.unique(indices).shape[0] / self.vq.n_codes).to(z.device)
        else:
            raise NotImplementedError(f"VQ type {self.cfg.vq_type} not implemented")
        pp_sample = torch.tensor(np.mean([len(torch.unique(index_seq)) for index_seq in indices])/z.shape[1]).to(z.device)
        return codes, indices, pp, pp_sample, commitment_loss

    def decode(self, codes, init_obs):
        init_obs = self.obs_proj(init_obs)
        init_obs = init_obs.unsqueeze(1)
        if self.cfg.cross_z:
            if self.cfg.use_m4==2:
                x = init_obs.repeat(1, self.cfg.skill_block_size, 1)
                cross_cond = self.latent_decoder_proj(codes)
            elif self.cfg.use_m4==1:
                # x = torch.ones(codes.shape[0], self.cfg.skill_block_size, self.cfg.decoder_dim).to(codes.device)
                x = self.fixed_positional_emb(torch.zeros((codes.shape[0], self.cfg.skill_block_size, self.cfg.decoder_dim), dtype=codes.dtype, device=codes.device))
                codes = self.latent_decoder_proj(codes)
                cross_cond = torch.cat([init_obs,codes], dim=1)
            else:
                x = init_obs.repeat(1, codes.shape[1], 1)
                cross_cond = self.latent_decoder_proj(codes)  
        else:
            x = codes
            cross_cond = init_obs
        x = self.decoder(x, cross_cond)
        if not self.cfg.use_m4:
            x = self.deconv_block(x)
        x = self.action_head(x)
        return x

    def forward(self, act, init_obs):
        z = self.encode(act)
        codes, _, pp, pp_sample, commitment_loss = self.quantize(z)
        x = self.decode(codes, init_obs)
        return x, pp, pp_sample, commitment_loss

    def get_indices(self, act):
        z = self.encode(act)
        _, indices, _, _, _ = self.quantize(z)
        return indices
    
    def decode_actions(self, indices, init_obs):
        if self.cfg.vq_type == 'fsq':
            codes = self.vq.indices_to_codes(indices)
        else:
            codes = self.vq.get_codes_from_indices(indices)
            codes = codes.view(codes.shape[0], -1, self.cfg.codebook_dim)
        x = self.decode(codes, init_obs)
        return x

    @property
    def device(self):
        return next(self.parameters()).device