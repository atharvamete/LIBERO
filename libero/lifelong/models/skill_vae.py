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

class SkillVAE_Model(BasePolicy):
    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy
        self.skill_vae = SkillVAE(policy_cfg)
        self.skill_vae = self.skill_vae.to(self.device)
        embed_size = policy_cfg.obs_emb_dim
        self.using_vq = True if policy_cfg.vq_type == "vq" else False

        self.image_encoders = {}

        for name in shape_meta["all_shapes"].keys():
            if "rgb" in name or "depth" in name:
                kwargs = policy_cfg.image_encoder.network_kwargs
                kwargs.input_shape = shape_meta["all_shapes"][name]
                kwargs.output_size = embed_size
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
        return encoded.squeeze(1)

    def forward(self, data):
        init_obs = self.obs_encode(data)
        pred, pp, pp_sample, commitment_loss = self.skill_vae(data["actions"], init_obs)
        info = {'pp': pp, 'pp_sample': pp_sample, 'commitment_loss': commitment_loss}
        return pred, info

    def loss_fn(self, pred, target):
        return self.loss(pred, target)

    def compute_loss(self, data):
        data = self.preprocess_input(data, train_mode=True)
        pred, info = self.forward(data)
        loss = self.loss_fn(pred, data["actions"])
        if self.using_vq:
            loss += info['commitment_loss'][0]
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