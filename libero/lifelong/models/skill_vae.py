import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn

from libero.lifelong.models.modules.rgb_modules import *
from libero.lifelong.models.modules.language_modules import *
# from libero.lifelong.models.modules.transformer_modules import *
from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.modules.skill_vae_modules import *
# from libero.lifelong.models.bc_transformer_policy import ExtraModalityTokens
# from libero.lifelong.models.policy_head import *

class SkillVAE_Model(BasePolicy):
    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy
        self.skill_vae = SkillVAE(policy_cfg)
        self.using_vq = True if policy_cfg.vq_type == "vq" else False

        if cfg.train.loss_type == "mse":
            self.loss = torch.nn.MSELoss()
        elif cfg.train.loss_type == "l1":
            self.loss = torch.nn.L1Loss()
        else:
            raise NotImplementedError(f"Unknown loss type {cfg.train.loss_type}")

    def forward(self, data):
        pred, pp, pp_sample, aux_loss = self.skill_vae(data["actions"])
        info = {'pp': pp, 'pp_sample': pp_sample, 'aux_loss': aux_loss.sum()}
        return pred, info

    def compute_loss(self, data):
        pred, info = self.forward(data)
        loss = self.loss(pred, data["actions"])
        if self.using_vq:
            loss += info['aux_loss']
        return loss, info

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