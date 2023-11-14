import torch
import torch.nn as nn

from turntaking.models.transformer import Transformer
from turntaking.models.conformer import Conformer
from turntaking.models.cnn import CNN


class AR(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.dim = conf["input_size"]
        self.num_layers = conf["num_layers"]
        self.dropout = conf["dropout"]
        self.ar_type = conf["type"]
        self.ar = self._ar(conf["type"], conf["model_kwargs"])

    def _ar(self, ar, model_kwargs):
        ar = ar.lower()
        ret = nn.Identity()

        if ar == "gru":
            ret = nn.GRU(
                input_size=self.dim,
                hidden_size=model_kwargs["GRU"]["dff_k"] * self.dim,
                num_layers=self.num_layers,
                batch_first=model_kwargs["GRU"]["batch_first"],
                bias=model_kwargs["GRU"]["bias"],
                dropout=self.dropout,
            )
        elif ar == "lstm":
            ret = nn.LSTM(
                input_size=self.dim,
                hidden_size=model_kwargs["LSTM"]["dff_k"] * self.dim,
                num_layers=self.num_layers,
                batch_first=model_kwargs["LSTM"]["batch_first"],
                dropout=self.dropout,
            )
        elif ar == "cnn":
            ret = CNN(
                input_size=self.dim,
                kernel=model_kwargs["CNN"]["kernel"],
                stride=model_kwargs["CNN"]["stride"],
                padding=model_kwargs["CNN"]["padding"],
            )
        elif ar == "transformer":
            ret = Transformer(
                input_size=self.dim,
                dff_k=model_kwargs["Transformer"]["dff_k"],
                num_layers=self.num_layers,
                num_heads=model_kwargs["Transformer"]["num_heads"],
                activation=model_kwargs["Transformer"]["activation"],
                dropout=self.dropout,
                use_pos_emb=model_kwargs["Transformer"]["use_pos_emb"],
                max_context=model_kwargs["Transformer"]["max_context"],
                use_pre_ln=model_kwargs["Transformer"]["use_pre_ln"],
            )
        elif ar == "conformer":
            ret = Conformer(
                input_size=self.dim,
                ffn_dim=self.dim * model_kwargs["Conformer"]["dff_k"],
                num_layers=self.num_layers,
                num_heads=model_kwargs["Conformer"]["num_heads"],
                depthwise_conv_kernel_size=model_kwargs["Conformer"][
                    "conv_kernel_size"
                ],
                dropout=self.dropout,
                convolution_first=model_kwargs["Conformer"]["convolution_first"],
                use_pos_emb=model_kwargs["Conformer"]["use_pos_emb"],
            )
        else:
            print(f"Autoregressive Error: {ret}")
            exit(1)

        return ret

    def forward(self, x, lengths=None, attention=False):
        ret = {}
        if self.ar_type == "transformer":
            x = self.ar(x, attention=attention)
            if attention:
                x, attn = x
                ret["attn"] = attn
            ret["z"] = x
        elif self.ar_type == ["conformer", "squeezeformer"]:
            x, lengths = self.ar(x, lengths)
            ret["z"] = x
            ret["lengths"] = lengths
        else:
            x, h = self.ar(x)
            ret["z"] = x
            ret["h"] = x
        return ret


def _test_ar(config_name):
    from turntaking.utils import load_hydra_conf
    from omegaconf import OmegaConf

    conf = load_hydra_conf(config_name=config_name)
    conf = conf["model"]
    print(OmegaConf.to_yaml(conf))
    B = 4
    N = 100
    D = 256
    # Autoregressive
    model = AR(
        input_dim=D,
        dim=conf["ar"]["dim"],
        num_layers=conf["ar"]["num_layers"],
        dropout=conf["ar"]["dropout"],
        ar=conf["ar"]["type"],
        transfomer_kwargs=dict(
            num_heads=conf["ar"]["num_heads"],
            dff_k=conf["ar"]["dff_k"],
            use_pos_emb=conf["ar"]["use_pos_emb"],
            max_context=conf["ar"].get("max_context", None),
            abspos=conf["ar"].get("abspos", None),
            sizeSeq=conf["ar"].get("sizeSeq", None),
        ),
    )
    # print(model)
    x = torch.rand((B, N, D))
    print("x: ", x.shape)
    o = model(x)
    print(o["z"].shape)


if __name__ == "__main__":
    _test_ar("model/discrete")
    _test_ar("model/discrete_20hz")
