import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn

from libero.lifelong.models.modules.rgb_modules import *
from libero.lifelong.models.modules.language_modules import *
# from libero.lifelong.models.modules.transformer_modules import *
from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.skill_vae import SkillVAE_Model
from libero.lifelong.models.modules.skill_vae_modules import *
from libero.lifelong.models.modules.skill_utils import SkillGPT_Config, SkillGPT, MLP_Proj, beam_search, top_k_sampling
from libero.lifelong.models.skill_vae import ExtraModalityTokens
from libero.lifelong.utils import torch_load_model
from collections import deque

class SkillGPT_Model(BasePolicy):
    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy
        self.prior_cfg = policy_cfg.prior
        self.batch_size = cfg.train.batch_size
        self.start_token = policy_cfg.prior.start_token
        self.offset_loss_scale = policy_cfg.offset_loss_scale
        self.mpc_horizon = policy_cfg.mpc_horizon
        self.action_queue = deque(maxlen=self.mpc_horizon)
        self.act_dim = policy_cfg.action_dim
        self.decoder_block_size = policy_cfg.skill_block_size
        
        self.skill_vae_policy = SkillVAE_Model(cfg, shape_meta)
        if cfg.pretrain_skillVAE_path is not None:
            self.skill_vae_policy.load_state_dict(torch_load_model(cfg.pretrain_skillVAE_path)[0])
        self.skill_vae_policy = self.skill_vae_policy.to(self.device)
        if not cfg.tune_decoder:
            self.skill_vae_policy.eval()
            for param in self.skill_vae_policy.parameters():
                param.requires_grad = False
        else:
            self.skill_vae_policy.train()

        offset_dim = self.act_dim*self.decoder_block_size
        skillgpt_config = SkillGPT_Config(policy_cfg.prior, offset_dim)
        self.skill_gpt = SkillGPT(skillgpt_config).to(self.device)
        self.lang_proj = MLP_Proj(policy_cfg.lang_emb_dim, skillgpt_config.n_embd, skillgpt_config.n_embd)
        self.obs_proj = MLP_Proj(policy_cfg.cat_obs_dim, skillgpt_config.n_embd, skillgpt_config.n_embd)

        if cfg.train.loss_type == "mse":
            self.loss = torch.nn.MSELoss()
        elif cfg.train.loss_type == "l1":
            self.loss = torch.nn.L1Loss()
        else:
            raise NotImplementedError(f"Unknown loss type {cfg.train.loss_type}")

    def print_data(self, data):
        for key in data.keys():
            if isinstance(data[key], dict):
                for sub_key in data[key].keys():
                    print(key, sub_key, data[key][sub_key].shape)
            else:
                print(key, data[key].shape)

    def forward(self, data):
        with torch.no_grad():
            indices = self.skill_vae_policy.skill_vae.get_indices(data["actions"]).long()
        init_obs = self.skill_vae_policy.obs_encode(data)
        start_tokens = (torch.ones((data["actions"].shape[0], 1))*self.start_token).long().to(self.device)
        x = torch.cat([start_tokens, indices[:,:-1]], dim=1)
        targets = indices.clone()
        init_obs_emb = self.obs_proj(init_obs)
        lang_emb = self.lang_proj(data["task_emb"])
        context = torch.cat([lang_emb.unsqueeze(1), init_obs_emb.unsqueeze(1)], dim=1)
        logits, prior_loss, offset = self.skill_gpt(x, context, targets, return_offset=True)
        offset = offset.view(-1, self.decoder_block_size, self.act_dim)
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            sampled_indices = torch.multinomial(probs.view(-1,logits.shape[-1]),1)
            sampled_indices = sampled_indices.view(-1,logits.shape[1])
        pred_actions = self.skill_vae_policy.skill_vae.decode_actions(sampled_indices, init_obs)
        pred_actions_with_offset = pred_actions + offset
        return pred_actions_with_offset, prior_loss

    def loss_fn(self, pred, target):
        return self.loss(pred, target)

    def compute_loss(self, data):
        data = self.skill_vae_policy.preprocess_input(data, train_mode=True)
        pred, prior_loss = self.forward(data)
        loss = self.loss_fn(pred, data["actions"])
        total_loss = prior_loss + self.offset_loss_scale*loss
        return total_loss, {'offset_loss': loss}
    
    def get_action(self, data):
        self.eval()
        if len(self.action_queue) == 0:
            with torch.no_grad():
                actions = self.sample_actions(data)
                self.action_queue.extend(actions[:self.mpc_horizon])
        action = self.action_queue.popleft()
        return action
    
    def sample_actions(self, data):
        data = self.skill_vae_policy.preprocess_input(data, train_mode=False)
        init_obs = self.skill_vae_policy.obs_encode(data)
        init_obs_emb = self.obs_proj(init_obs)
        lang_emb = self.lang_proj(data["task_emb"])
        context = torch.cat([lang_emb.unsqueeze(1), init_obs_emb.unsqueeze(1)], dim=1)
        sampled_indices, offset = self.get_indices_top_k(context)
        # print(sampled_indices, 'sampled_indices')
        # print('offset max min', offset.max(), offset.min())
        pred_actions = self.skill_vae_policy.skill_vae.decode_actions(sampled_indices, init_obs)
        pred_actions_with_offset = pred_actions + offset
        pred_actions_with_offset = pred_actions_with_offset.permute(1,0,2)
        return pred_actions_with_offset.detach().cpu().numpy()

    def get_indices(self, context):
        x = torch.ones((context.shape[0], 1)).long().to(self.device)*self.start_token
        for i in range(self.prior_cfg.block_size):
            if i == self.prior_cfg.block_size-1:
                logits, offset = self.skill_gpt(x, context, None, return_offset=True)
            else:
                logits = self.skill_gpt(x, context)
            next_indices = torch.multinomial(torch.softmax(logits[:,-1,:], dim=-1), 1)
            x = torch.cat([x, next_indices], dim=1)
        offset = offset.view(-1, self.decoder_block_size, self.act_dim)
        return x[:,1:], offset

    def get_indices_top_k(self, context):
        x = torch.ones((context.shape[0], 1)).long().to(self.device)*self.start_token
        for i in range(self.prior_cfg.block_size):
            if i == self.prior_cfg.block_size-1:
                logits, offset = self.skill_gpt(x, context, None, return_offset=True)
            else:
                logits = self.skill_gpt(x, context)
            next_indices = top_k_sampling(logits[:,-1,:], self.prior_cfg.beam_size, self.prior_cfg.temperature)
            x = torch.cat([x, next_indices], dim=1)
        offset = offset.view(-1, self.decoder_block_size, self.act_dim)
        return x[:,1:], offset

    def get_indices_beam(self, context):
        outputs = beam_search(self.start_token, self.skill_gpt, context, self.prior_cfg.block_size, self.device, self.prior_cfg.beam_size, self.prior_cfg.temperature)
        final_indices = torch.tensor(outputs).unsqueeze(0).to(self.device)
        _, offset = self.skill_gpt(final_indices, context, None, return_offset=True)
        offset = offset.view(-1, self.decoder_block_size, self.act_dim)
        return final_indices, offset

    def reset(self):
        self.action_queue = deque(maxlen=self.mpc_horizon)
    
    def configure_optimizer(self, lr, betas, weight_decay):
        learning_rate = lr
        # Get the optimizer configured for the decoder
        decoder_optimizer = self.skill_vae_policy.configure_optimizers(weight_decay, learning_rate, betas)
        # Get the optimizer configured for GPT
        gpt_optimizer = self.skill_gpt.configure_optimizers(weight_decay, learning_rate, betas)
        # Combine the two optimizers
        combined_param_groups = decoder_optimizer.param_groups + gpt_optimizer.param_groups
        optimizer = torch.optim.AdamW(combined_param_groups)
        # add remaining parameters to optimizer
        optimizer.add_param_group({'params': self.lang_proj.parameters(), 'weight_decay': weight_decay, 'lr': learning_rate, 'betas': betas})
        optimizer.add_param_group({'params': self.obs_proj.parameters(), 'weight_decay': weight_decay, 'lr': learning_rate, 'betas': betas})
        # all_params = [p for p in self.parameters()]
        # decoder_params = [p for p in self.skill_vae_policy.parameters()]
        # gpt_params = [p for p in self.skill_gpt.parameters()]
        # other_params = [p for p in all_params if p not in decoder_params and p not in gpt_params]
        # optimizer.add_param_group({
        #     'params': other_params,
        #     'weight_decay': weight_decay,
        #     'lr': learning_rate,
        #     'betas': betas
        # })
        return optimizer
