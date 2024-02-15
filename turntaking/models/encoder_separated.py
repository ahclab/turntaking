import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from turntaking.models.cpc_base_model import load_CPC
from turntaking.models.cnn import CConv1d, LayerNorm
from turntaking.models.autoregressive import AR


def get_cnn_layer(dim, kernel, stride, dilation, activation):
    layers = [Rearrange("b t d -> b d t")]
    for k, s, d in zip(kernel, stride, dilation):
        layers.append(CConv1d(dim, dim, kernel_size=k, stride=s, dilation=d))
        layers.append(LayerNorm(dim))
        layers.append(getattr(torch.nn, activation)())
    layers.append(Rearrange("b d t -> b t d"))
    return nn.Sequential(*layers)


class Encoder_Separated(nn.Module):
    def __init__(self, conf, freeze=True):
        super().__init__()
        self.conf = conf
        self.name = conf["name"]
        # self.frame_hz = conf["frame_hz"]
        # self.sample_rate = conf["sample_rate"]
        self.encoder_layer = conf["output_layer"]
        self.encoder = load_CPC()
        self.output_dim = self.encoder.gEncoder.conv4.out_channels

        if self.conf["user1_input"]:
            self._initialize_module("user_1")
        elif self.conf["user2_input"]:
            self._initialize_module("user_2")

        if not (self.conf["user1_input"] or self.conf["user2_input"]):
            print("Error: Encoder_Separated() must be true for either user1 or user2.")
            exit(1)

        if freeze:
            self.freeze()

    def _initialize_module(self, user_key):
        module_conf = self.conf[user_key]["module"]
        self._set_module(user_key, module_conf)
        self._set_downsample(user_key, self.conf)

    def _set_module(self, user_key, module_conf):
        if module_conf["use_module"]:
            setattr(self, f"module_{user_key[-1]}", AR(module_conf))
        else:
            setattr(self, f"module_{user_key[-1]}", nn.Identity())

    def _set_downsample(self, user_key, user_conf):
        if user_conf.get("downsample", False):
            down = user_conf["downsample"]
            downsample = get_cnn_layer(
                dim=self.output_dim,
                kernel=down["kernel"],
                stride=down["stride"],
                dilation=down["dilation"],
                activation=down["activation"],
            )
            setattr(self, f"downsample_{user_key[-1]}", downsample)
        else:
            setattr(self, f"downsample_{user_key[-1]}", nn.Identity())

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def unfreeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(True)

    def encode(self, waveform):
        if waveform.ndim < 3:
            waveform = waveform.unsqueeze(1)
        z = self.encoder.gEncoder(waveform)
        z = einops.rearrange(z, "b c n -> b n c")

        if self.encoder_layer > 0:
            z = self.encoder.gAR(z)
        return z

    def _process_waveform(self, waveform, user_key):
        z = self.encode(waveform)
        downsample = getattr(self, f"downsample_{user_key[-1]}")
        z = downsample(z)
        module = getattr(self, f"module_{user_key[-1]}")
        if not isinstance(module, nn.Identity):
            z = module(z)["z"]
        return z

    def forward(self, waveform_1, waveform_2):
        if self.conf["user1_input"]:
            z_1 = self._process_waveform(waveform_1, "user_1")
        if self.conf["user2_input"]:
            z_2 = self._process_waveform(waveform_2, "user_2")

        z = (
            z_1 + z_2
            if self.conf["user1_input"] and self.conf["user2_input"]
            else z_1
            if self.conf["user1_input"]
            else z_2
        )
        return {"z": z}
