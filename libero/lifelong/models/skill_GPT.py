import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn

from libero.lifelong.models.modules.rgb_modules import *
from libero.lifelong.models.modules.language_modules import *
# from libero.lifelong.models.modules.transformer_modules import *
from libero.lifelong.models.base_policy import BasePolicy
# from libero.lifelong.models.skill_vae import SkillVAE_Model
from libero.lifelong.models.modules.skill_vae_modules import *
from libero.lifelong.models.modules.skill_vae_modules import *
from libero.lifelong.models.modules.skill_utils import Transformer_Prior, MLP_Proj, beam_search, top_k_sampling
from libero.lifelong.utils import torch_load_model
from collections import deque

class ExtraModalityTokens(nn.Module):
    def __init__(
        self,
        use_joint=False,
        use_gripper=False,
        use_ee=False,
        extra_num_layers=0,
        extra_hidden_size=64,
        extra_embedding_size=32,
    ):
        super().__init__()
        self.use_joint = use_joint
        self.use_gripper = use_gripper
        self.use_ee = use_ee
        self.extra_embedding_size = extra_embedding_size
        joint_states_dim = 7
        gripper_states_dim = 2
        ee_dim = 3
        self.num_extra = int(use_joint) + int(use_gripper) + int(use_ee)
        extra_low_level_feature_dim = (
            int(use_joint) * joint_states_dim
            + int(use_gripper) * gripper_states_dim
            + int(use_ee) * ee_dim
        )
        assert extra_low_level_feature_dim > 0, "[error] no extra information"
        self.extra_encoders = {}
        def generate_proprio_mlp_fn(modality_name, extra_low_level_feature_dim):
            assert extra_low_level_feature_dim > 0  # we indeed have extra information
            if extra_num_layers > 0:
                layers = [nn.Linear(extra_low_level_feature_dim, extra_hidden_size)]
                for i in range(1, extra_num_layers):
                    layers += [
                        nn.Linear(extra_hidden_size, extra_hidden_size),
                        nn.ReLU(inplace=True),
                    ]
                if modality_name == "joint_states":
                    layers += [nn.Linear(extra_low_level_feature_dim, extra_embedding_size*2)]
                else:
                    layers += [nn.Linear(extra_low_level_feature_dim, extra_embedding_size)]
            else:
                if modality_name == "joint_states":
                    layers = [nn.Linear(extra_low_level_feature_dim, extra_embedding_size*2)]
                else:
                    layers = [nn.Linear(extra_low_level_feature_dim, extra_embedding_size)]
            self.proprio_mlp = nn.Sequential(*layers)
            self.extra_encoders[modality_name] = {"encoder": self.proprio_mlp}
        for (proprio_dim, use_modality, modality_name) in [
            (joint_states_dim, self.use_joint, "joint_states"),
            (gripper_states_dim, self.use_gripper, "gripper_states"),
            (ee_dim, self.use_ee, "ee_pos"),
        ]:
            if use_modality:
                generate_proprio_mlp_fn(modality_name, proprio_dim)
        self.encoders = nn.ModuleList(
            [x["encoder"] for x in self.extra_encoders.values()]
        )

    def forward(self, obs_dict):
        tensor_list = []
        for (use_modality, modality_name) in [
            (self.use_joint, "joint_states"),
            (self.use_gripper, "gripper_states"),
            (self.use_ee, "ee_pos"),
        ]:
            if use_modality:
                tensor_list.append(
                    self.extra_encoders[modality_name]["encoder"](
                        obs_dict[modality_name]
                    )
                )
        x = torch.cat(tensor_list, dim=-1)
        return x

def load_vae(cfg, tune_decoder, device):
    skill_vae = SkillVAE(cfg)
    state_dict, _, _ = torch_load_model(cfg.path)
    vae_state_dict = {key.replace('skill_vae.', ''): value for key, value in state_dict.items()}
    skill_vae.load_state_dict(vae_state_dict, strict=True)
    # print number of matching keys in skill_vae and state_dict
    print(sum([1 for key in skill_vae.state_dict().keys() if key in vae_state_dict.keys()]), 'matching keys')
    skill_vae = skill_vae
    if not tune_decoder:
        skill_vae.eval()
        for param in skill_vae.parameters():
            param.requires_grad = False
    else:
        skill_vae.train()
    return skill_vae

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
        self.vae_1_block_size = policy_cfg.skill_vae_1.skill_block_size
        self.return_offset = True if policy_cfg.prior.offset_layers > 0 else False
        offset_dim = self.act_dim*self.vae_1_block_size
        self.prior_cfg.offset_dim = offset_dim
        
        self.skill_vae_1 = load_vae(policy_cfg.skill_vae_1, tune_decoder=cfg.tune_decoder, device=self.device).to(self.device)
        print(next(self.skill_vae_1.parameters()).requires_grad, 'skill_vae_1 grad')
        self.skill_gpt = Transformer_Prior(self.prior_cfg).to(self.device)

        self.lang_proj = MLP_Proj(policy_cfg.lang_emb_dim, self.prior_cfg.n_embd, self.prior_cfg.n_embd)
        self.obs_proj = MLP_Proj(policy_cfg.cat_obs_dim, self.prior_cfg.n_embd, self.prior_cfg.n_embd)

        self.image_encoders = {}
        for name in shape_meta["all_shapes"].keys():
            if "rgb" in name or "depth" in name:
                kwargs = policy_cfg.image_encoder.network_kwargs
                kwargs.input_shape = shape_meta["all_shapes"][name]
                kwargs.output_size = policy_cfg.obs_emb_dim
                self.image_encoders[name] = {
                    "input_shape": shape_meta["all_shapes"][name],
                    "encoder": eval(policy_cfg.image_encoder.network)(**kwargs),
                }
        self.encoders = nn.ModuleList(
            [x["encoder"] for x in self.image_encoders.values()]
        )
        self.extra_encoder = ExtraModalityTokens(
            use_joint=cfg.data.use_joint,
            use_gripper=cfg.data.use_gripper,
            use_ee=cfg.data.use_ee,
            extra_num_layers=policy_cfg.extra_num_layers,
            extra_hidden_size=policy_cfg.extra_hidden_size,
            extra_embedding_size=policy_cfg.extra_embedding_size,
        )
        if cfg.train.loss_type == "mse":
            self.loss = torch.nn.MSELoss()
        elif cfg.train.loss_type == "l1":
            self.loss = torch.nn.L1Loss()
        else:
            raise NotImplementedError(f"Unknown loss type {cfg.train.loss_type}")
    
    def obs_encode(self, data):
        ### 1. encode image
        encoded = []
        for img_name in self.image_encoders.keys():
            x = data["obs"][img_name]
            B, T, C, H, W = x.shape
            e = self.image_encoders[img_name]["encoder"](
                x.reshape(B * T, C, H, W),
                langs=data["task_emb"]
                .reshape(B, 1, -1)
                .repeat(1, T, 1)
                .reshape(B * T, -1),
            ).view(B, T, -1)
            encoded.append(e)
        # 2. add gripper info
        encoded.append(self.extra_encoder(data["obs"]))  # add (B, T, H_extra)
        encoded = torch.cat(encoded, -1)  # (B, T, H_all)
        init_obs_emb = self.obs_proj(encoded)
        lang_emb = self.lang_proj(data["task_emb"]).unsqueeze(1)
        context = torch.cat([lang_emb, init_obs_emb], dim=1)
        return context

    def forward(self, data):
        indices = self.skill_vae_1.get_indices(data["actions"]).long()
        context = self.obs_encode(data)
        start_tokens = (torch.ones((context.shape[0], 1))*self.start_token).long().to(self.device)
        x = torch.cat([start_tokens, indices[:,:-1]], dim=1)
        targets = indices.clone()
        logits, prior_loss, offset = self.skill_gpt(x, context, targets, return_offset=self.return_offset)
        if self.return_offset:
            offset = offset.view(-1, self.vae_1_block_size, self.act_dim)
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                sampled_indices = torch.multinomial(probs.view(-1,logits.shape[-1]),1)
                sampled_indices = sampled_indices.view(-1,logits.shape[1])
            pred_actions = self.skill_vae_1.decode_actions(sampled_indices)
            pred_actions_with_offset = pred_actions + offset
            offset_loss = self.loss(pred_actions_with_offset, data["actions"])
            total_loss = prior_loss + self.offset_loss_scale*offset_loss
            return total_loss, {'offset_loss': offset_loss}
        else:
            return prior_loss, {}

    def compute_loss(self, data):
        data = self.preprocess_input(data, train_mode=True)
        loss, info = self.forward(data)
        return loss, info
    
    def get_action(self, data):
        self.eval()
        if len(self.action_queue) == 0:
            with torch.no_grad():
                actions = self.sample_actions(data)
                self.action_queue.extend(actions[:self.mpc_horizon])
        action = self.action_queue.popleft()
        return action
    
    def sample_actions(self, data):
        data = self.preprocess_input(data, train_mode=False)
        context = self.obs_encode(data)
        sampled_indices, offset = self.get_indices_top_k(context)
        pred_actions = self.skill_vae_1.decode_actions(sampled_indices)
        pred_actions_with_offset = pred_actions + offset if offset is not None else pred_actions
        pred_actions_with_offset = pred_actions_with_offset.permute(1,0,2)
        return pred_actions_with_offset.detach().cpu().numpy()

    def get_indices_top_k(self, context):
        x = torch.ones((context.shape[0], 1)).long().to(self.device)*self.start_token
        for i in range(self.prior_cfg.block_size):
            if i == self.prior_cfg.block_size-1:
                logits,offset = self.skill_gpt(x, context, return_offset=self.return_offset)
                offset = offset.view(-1, self.vae_1_block_size, self.act_dim) if self.return_offset else None
            else:
                logits,_ = self.skill_gpt(x, context)
            next_indices = top_k_sampling(logits[:,-1,:], self.prior_cfg.beam_size, self.prior_cfg.temperature)
            x = torch.cat([x, next_indices], dim=1)
        return x[:,1:], offset

    def reset(self):
        self.action_queue = deque(maxlen=self.mpc_horizon)