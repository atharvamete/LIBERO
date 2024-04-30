import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn

from libero.lifelong.models.modules.rgb_modules import *
from libero.lifelong.models.modules.language_modules import *
# from libero.lifelong.models.modules.transformer_modules import *
from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.modules.encodec_utils import *
# from libero.lifelong.models.bc_transformer_policy import ExtraModalityTokens
# from libero.lifelong.models.policy_head import *

class SkillVAE_Model(BasePolicy):
    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy
        self.encoder = Encoder(in_channels=policy_cfg.action_dim, channels=policy_cfg.channel, codebook_dim=policy_cfg.codebook_dim, kernels=policy_cfg.kernels, strides=policy_cfg.strides)
        self.decoder = Decoder(out_channels=policy_cfg.action_dim, channels=policy_cfg.channel, codebook_dim=policy_cfg.codebook_dim, kernels=policy_cfg.kernels, strides=policy_cfg.strides)
        self.quantizer = Quantizer(**policy_cfg.quantizer_args)

        if cfg.train.loss_type == "mse":
            self.loss = torch.nn.MSELoss()
        elif cfg.train.loss_type == "l1":
            self.loss = torch.nn.L1Loss()
        else:
            raise NotImplementedError(f"Unknown loss type {cfg.train.loss_type}")

    def forward(self, data):
        out = self.encoder(data['actions'])
        codes, indices, aux_loss = self.quantizer(out)
        out = self.decoder(codes)
        pp = torch.tensor(torch.unique(indices).shape[0] / self.quantizer.total_codes).to(out.device)
        return out, {'aux_loss':aux_loss, 'pp':pp}

    def compute_loss(self, data):
        pred, info = self.forward(data)
        recon_loss = self.loss(pred, data['actions'])
        total_loss = recon_loss + info['aux_loss']
        return total_loss, info

    def get_indices(self, actions):
        out = self.encoder(actions)
        _, indices, _ = self.quantizer(out)
        return indices

    def decode_actions(self, indices):
        codes = self.quantizer.indices_to_codes(indices)
        return self.decoder(codes)

    def configure_optimizers(self, lr, betas, weight_decay):
        learning_rate = lr
        # Get the optimizer configured for the decoder
        optimizer = self.skill_vae.decoder.configure_optimizers(weight_decay, learning_rate, betas)
        # Get all parameters of the current module
        all_params = {id(p): p for p in self.parameters()}
        # Get the parameters of the decoder
        decoder_params = {id(p): p for p in self.skill_vae.decoder.parameters()}
        # Exclude the parameters of the decoder from all_params
        other_params = [p for p_id, p in all_params.items() if p_id not in decoder_params]
        # Add other_params to the optimizer as a separate group
        optimizer.add_param_group({
            'params': other_params,
            'weight_decay': weight_decay,
            'lr': learning_rate,
            'betas': betas
        })
        return optimizer