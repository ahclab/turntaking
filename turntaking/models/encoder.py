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


def Out_to_Linear():
    layers = []
    layers = [Rearrange("b t d -> b d t")]
    layers.append(nn.Linear(250, 1))
    layers.append(Rearrange("b d t -> b t d"))
    return nn.Sequential(*layers)


def DownSample(dim, norm_dim, kernel, stride, dilation, activation):
    layers = []
    # layers = [Rearrange("b t d -> b d t")]
    for k, s, d in zip(kernel, stride, dilation):
        layers.append(CConv1d(dim, dim, kernel_size=k, stride=s, dilation=d))
        layers.append(nn.LayerNorm(norm_dim))
        layers.append(getattr(torch.nn, activation)())
    # layers.append(Rearrange("b d t -> b t d"))
    return nn.Sequential(*layers)


def Pool(dim, kernel, stride, dilation, activation):
    layers = []
    for k, s, d in zip(kernel, stride, dilation):
        layers.append(nn.MaxPool1d(kernel_size=k, stride=s, dilation=d))
        layers.append(LayerNorm(dim))
        layers.append(getattr(torch.nn, activation)())
    return nn.Sequential(*layers)


def UpSample(dim, kernel, stride, dilation, activation):
    # layers = []
    layers = [Rearrange("b t d -> b d t")]
    for k, s, d in zip(kernel, stride, dilation):
        layers.append(nn.ConvTranspose1d(dim, dim, kernel_size=k, stride=s, dilation=d))
        layers.append(LayerNorm(dim))
        layers.append(getattr(torch.nn, activation)())
    layers.append(Rearrange("b d t -> b t d"))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """
    Encoder: waveform -> h
    pretrained: default='cpc'

    A simpler version of the Encoder
    check paper (branch) version to see other encoders...
    """

    def __init__(self, conf, freeze=True):
        super().__init__()
        self.conf = conf
        self.name = conf["name"]
        self.frame_hz = conf["frame_hz"]
        self.sample_rate = conf["sample_rate"]
        self.encoder_layer = conf["output_layer"]
        self.encoder = load_CPC()
        self.output_dim = self.encoder.gEncoder.conv4.out_channels

        if conf["module"]["use_module"]:
            self.module = AR(conf["module"])
        else:
            self.module = nn.Identity()

        if conf.get("downsample", False):
            down = conf["downsample"]
            self.downsample = get_cnn_layer(
                dim=self.output_dim,
                kernel=down["kernel"],
                stride=down["stride"],
                dilation=down["dilation"],
                activation=down["activation"],
            )
        else:
            self.downsample = nn.Identity()

        if freeze:
            self.freeze()

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        # print(f"Froze {self.__class__.__name__}!")

    def unfreeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(True)
        # print(f"Trainable {self.__class__.__name__}!")

    def encode(self, waveform):
        # pprint(waveform)
        if waveform.ndim < 3:
            waveform = waveform.unsqueeze(1)  # channel dim

        # Backwards using only the encoder encounters:
        # ---------------------------------------------------
        # RuntimeError: one of the variables needed for gradient computation
        # has been modified by an inplace operation:
        # [torch.FloatTensor [4, 256, 1000]], which is output 0 of ReluBackward0, is at version 1;
        # expected version 0 instead. Hint: enable anomaly detection to find
        # the operation that failed to compute its gradient, with
        # torch.autograd.set_detect_anomaly(True).
        z = self.encoder.gEncoder(waveform)  # .permute(0, 2, 1)
        z = einops.rearrange(z, "b c n -> b n c")

        # However, if we feed through gAR we do not encounter that problem...
        if self.encoder_layer > 0:
            z = self.encoder.gAR(z)
        return z

    def forward(self, waveform):
        z = self.encode(waveform)
        z = self.downsample(z)
        z = (
            self.module(z)["z"]
            if not isinstance(self.module, nn.Identity)
            else self.module(z)
        )
        return {"z": z}


def _test_encoder(config_name):
    from turntaking.utils import load_hydra_conf

    conf = load_hydra_conf(config_name=config_name)
    econf = conf["model"]["encoder"]
    enc = Encoder(econf, freeze=econf["freeze"])
    x = torch.rand((4, econf["sample_rate"]))
    out = enc(x)
    z = out["z"]
    print("Config: ", config_name)
    print("x: ", tuple(x.shape))
    print("z: ", tuple(z.shape))


if __name__ == "__main__":
    _test_encoder("model/discrete")
    _test_encoder("model/discrete_20hz")
    _test_encoder("model/discrete_50hz")
