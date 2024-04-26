import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn

from libero.lifelong.models.modules.rgb_modules import *
from libero.lifelong.models.modules.language_modules import *
# from libero.lifelong.models.modules.transformer_modules import *
from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.modules.diffusion_modules import *
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

class Diffusion_Policy(BasePolicy):
    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy
        self.mpc_horizon = policy_cfg.mpc_horizon
        self.action_queue = deque(maxlen=self.mpc_horizon)
        
        self.diff_model = Diffusion_Model(policy_cfg, self.device)
        self.diff_model = self.diff_model.to(self.device)
        # self.input_proj = MLP_Proj(policy_cfg.cat_obs_dim+policy_cfg.lang_emb_dim, policy_cfg.cond_dim, policy_cfg.cond_dim)

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
    
    def obs_encode(self, data):
        ### 1. encode image
        encoded = []
        for img_name in self.image_encoders.keys():
            x = data["obs"][img_name]
            B, T, C, H, W = x.shape
            e = self.image_encoders[img_name]["encoder"](
                x.reshape(B * T, C, H, W),
                langs=None,
            ).view(B, T, -1)
            encoded.append(e)
        # 2. add gripper info
        encoded.append(self.extra_encoder(data["obs"]))  # add (B, T, H_extra)
        encoded = torch.cat(encoded, -1)  # (B, T, H_all)
        return encoded.squeeze(1)

    def forward(self, data):
        init_obs = self.obs_encode(data)
        lang_emb = data["task_emb"]
        cond = torch.cat([init_obs, lang_emb], dim=-1)
        loss = self.diff_model(cond,data["actions"])
        return loss

    def compute_loss(self, data):
        data = self.preprocess_input(data, train_mode=True)
        loss = self.forward(data)
        return loss, {}
    
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
        init_obs = self.obs_encode(data)
        lang_emb = data["task_emb"]
        cond = torch.cat([init_obs, lang_emb], dim=-1)
        actions = self.diff_model.get_action(cond)
        actions = actions.permute(1,0,2)
        return actions.detach().cpu().numpy()

    def reset(self):
        self.action_queue = deque(maxlen=self.mpc_horizon)