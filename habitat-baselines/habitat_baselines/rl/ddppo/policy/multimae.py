import sys
sys.path.append("/private/home/sergioarnaud/habitat/MultiMAE")
from utils_mae import create_model
from functools import partial
from multimae.input_adapters import PatchedInputAdapter
import torch
import torchvision
from utils_mae.pos_embed import interpolate_pos_embed_multimae
import torch.nn.functional as F
from torch import nn as nn
import numpy as np

from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
    HeadingSensor,
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
    ProximitySensor,
)

def masked_l1_loss(preds, target, mask_valid=None):
    if mask_valid is None:
        mask_valid = torch.ones_like(preds).bool()
    if preds.shape[1] != mask_valid.shape[1]:
        mask_valid = mask_valid.repeat_interleave(preds.shape[1], 1)
    element_wise_loss = abs(preds - target)
    element_wise_loss[~mask_valid] = 0
    return element_wise_loss.sum() / mask_valid.sum()


def get_model():
    in_domains = ["rgb", "depth"]

    patch_size = 16
    input_size = 128
    model_name = "multivit_base"
    drop_path_encoder = 0.1

    DOMAIN_CONF = {
        "rgb": {
            "channels": 3,
            "stride_level": 1,
            "input_adapter": partial(PatchedInputAdapter, num_channels=3),
            "loss": masked_l1_loss,
        },
        "depth": {
            "channels": 1,
            "stride_level": 1,
            "input_adapter": partial(PatchedInputAdapter, num_channels=1),
            "loss": masked_l1_loss,
        },
    }

    input_adapters = {
        domain: DOMAIN_CONF[domain]["input_adapter"](
            stride_level=DOMAIN_CONF[domain]["stride_level"],
            patch_size_full=patch_size,
            image_size=input_size,
        )
        for domain in in_domains
    }

    output_adapters = None
    model = create_model(
        model_name,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        drop_path_rate=drop_path_encoder,
    )

    checkpoint = torch.load(
        "/private/home/sergioarnaud/habitat/MultiMAE/multimae-b_98_rgb+-depth-semseg_1600e_multivit-afff3f8c.pth"
    )
    checkpoint_model = checkpoint["model"]

    class_emb_key = "input_adapters.semseg.class_emb.weight"
    if class_emb_key in checkpoint_model:
        checkpoint_model[class_emb_key] = F.pad(
            checkpoint_model[class_emb_key], (0, 0, 0, 1)
        )

    # Remove output adapters
    for k in list(checkpoint_model.keys()):
        if "output_adapters" in k:
            del checkpoint_model[k]

    # Interpolate position embedding
    interpolate_pos_embed_multimae(model, checkpoint_model)

    # Load pre-trained model
    model.load_state_dict(checkpoint_model, strict=False)

    return model


class MMAE(nn.Module):
    def __init__(self, observation_space):
        super().__init__()
        self.model = get_model()
        self.norm = nn.LayerNorm(self.output_shape()[1], eps=1e-6)
        
        self.visual_keys = [
            k
            for k, v in observation_space.spaces.items()
            if len(v.shape) > 1
            and k != ImageGoalSensor.cls_uuid
            and "debug" not in k
        ]

        self.key_needs_rescaling = {k: None for k in self.visual_keys}
        for k, v in observation_space.spaces.items():
            if v.dtype == np.uint8:
                self.key_needs_rescaling[k] = 1.0 / v.high.max()

    def forward(self, x):
        rgb = x['robot_head_rgb']
        rgb = rgb.permute(0, 3, 1, 2)
        if self.key_needs_rescaling['robot_head_rgb'] is not None:
            rgb = (
                rgb.float() * self.key_needs_rescaling['robot_head_rgb']
            )

        depth = x['robot_head_depth']
        depth = depth.permute(0, 3, 1, 2)
        if self.key_needs_rescaling['robot_head_depth'] is not None:
            depth = (
                depth.float() * self.key_needs_rescaling['robot_head_depth']
            )

        x = {
            'rgb': rgb,
            'depth': depth,
        }

        x = self.model(x)
        x = x.mean(1)
        x = self.norm(x)
        return x

    def is_blind(self):
        return False

    def output_shape(self):
        return (1, 768)

