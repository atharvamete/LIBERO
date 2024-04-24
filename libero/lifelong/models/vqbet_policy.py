import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn

from libero.lifelong.models.modules.rgb_modules import *
from libero.lifelong.models.modules.language_modules import *
# from libero.lifelong.models.modules.transformer_modules import *
from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.vqbet_vae import VQVAE_Model
from libero.lifelong.models.modules.vq_behavior_transformer import *

from libero.lifelong.utils import torch_load_model
from collections import deque

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

class VQBet_Model(BasePolicy):
    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy
        self.mpc_horizon = policy_cfg.mpc_horizon
        self.obs_window_size = policy_cfg.obs_window_size
        self.action_queue = deque(maxlen=self.mpc_horizon)
        
        vq_vae = VQVAE_Model(cfg, shape_meta)
        if cfg.pretrain_vqvae_path is not None:
            vq_vae.load_state_dict(torch_load_model(cfg.pretrain_vqvae_path)[0])
        vq_vae = vq_vae.to(self.device)
        if not cfg.tune_decoder:
            vq_vae.eval()
            for param in vq_vae.parameters():
                param.requires_grad = False
        else:
            vq_vae.train()
        
        gpt = GPT(GPTConfig(
            block_size=policy_cfg.gpt_block_size,
            input_dim=policy_cfg.gpt_n_embd,
            n_layer=policy_cfg.gpt_n_layer,
            n_head=policy_cfg.gpt_n_head,
            n_embd=policy_cfg.gpt_n_embd,
        )).to(self.device)

        self.Bet = BehaviorTransformer(
            gpt_model=gpt,
            vqvae_model=vq_vae.vq_vae,
            obs_dim=policy_cfg.gpt_n_embd,
            act_dim=policy_cfg.action_dim,
            goal_dim=policy_cfg.gpt_n_embd,
            obs_window_size=policy_cfg.obs_window_size,
            act_window_size=policy_cfg.skill_block_size,
            offset_loss_multiplier=policy_cfg.offset_loss_multiplier,
        ).to(self.device)

        self.lang_proj = MLP_Proj(policy_cfg.lang_emb_dim, policy_cfg.gpt_n_embd, policy_cfg.gpt_n_embd)
        self.obs_proj = MLP_Proj(policy_cfg.cat_obs_dim, policy_cfg.gpt_n_embd, policy_cfg.gpt_n_embd)
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
        # self.encoders.append(self.extra_encoder)

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
        return encoded

    def forward(self, data):
        obs = self.obs_proj(self.obs_encode(data))
        obs = obs[:,:self.obs_window_size,:]
        goal = self.lang_proj(data["task_emb"]).unsqueeze(1)
        predicted_act, loss, loss_dict = self.Bet(obs, goal, data["actions"])
        return predicted_act, loss, loss_dict

    def compute_loss(self, data):
        data = self.preprocess_input(data, train_mode=True)
        _, loss, loss_dict = self.forward(data)
        return loss, loss_dict
    
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
        obs = self.obs_proj(self.obs_encode(data))[:,:self.obs_window_size,:]
        goal = self.lang_proj(data["task_emb"])
        predicted_act, _, _ = self.Bet(obs, goal, None)
        predicted_act = predicted_act.permute(1,0,2)
        return predicted_act.detach().cpu().numpy()

    def reset(self):
        self.action_queue = deque(maxlen=self.mpc_horizon)
    
    def configure_optimizers(self, lr, betas, weight_decay):
        bet_optimizers = self.Bet.configure_optimizers(weight_decay=weight_decay, learning_rate=lr, betas=betas)
        bet_optimizers['optimizer1'].add_param_group({'params': self.lang_proj.parameters()})
        bet_optimizers['optimizer1'].add_param_group({'params': self.obs_proj.parameters()})
        bet_optimizers['optimizer1'].add_param_group({'params': self.encoders.parameters(), 'lr': lr*0.1})
        bet_optimizers['optimizer1'].add_param_group({'params': self.extra_encoder.parameters()})
        return bet_optimizers