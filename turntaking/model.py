import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import OmegaConf
import einops
from einops.layers.torch import Rearrange

from turntaking.models import Encoder, Encoder_Separated, AR
from turntaking.utils import to_device
from turntaking.vap_to_turntaking import VAP, TurnTakingMetrics
from torchinfo import summary


class VadCondition(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.output_size = conf["output_size"]
        self.use_va_history = conf["use_history"]

        self.va_condition = nn.Linear(2, self.output_size)

        if self.use_va_history:
            self.va_history = nn.Linear(conf["history_bins"], self.output_size)

        self.ln = nn.LayerNorm(self.output_size)

    def forward(self, vad, va_history=None):
        v_cond = self.va_condition(vad)

        if self.use_va_history:
            v_cond += self.va_history(va_history)

        return self.ln(v_cond)


class AudioCondition(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        self.encoder = self._init_module("encoder", Encoder)
        self.encoder_separated = self._init_module(
            "encoder_separated_audio", Encoder_Separated
        )
        self.vad_condition = self._init_module("va_cond", VadCondition)
        self.audio_module = self._init_module("audio_module", AR)

    def _init_module(self, conf_key, module_class):
        if self.conf[conf_key]["use_module"]:
            return module_class(self.conf[conf_key])
        return nn.Identity()

    def forward(self, **kwargs):
        wc = vc = torch.tensor([])
        encoder = False

        if self.conf["encoder"]["use_module"]:
            wc = self.encoder(kwargs["waveform"])["z"][:, : kwargs["va"].shape[1]]
            encoder = True
        elif self.conf["encoder_separated_audio"]["use_module"]:
            wc = self.encoder_separated(
                kwargs["waveform_user1"], kwargs["waveform_user2"]
            )["z"][:, : kwargs["va"].shape[1]]
            encoder = True

        if self.conf["va_cond"]["use_module"]:
            vc = self.vad_condition(kwargs["va"], kwargs["va_history"])

        z = None
        if encoder and self.conf["va_cond"]["use_module"]:
            z = wc + vc[:, : wc.shape[1]]
        elif encoder:
            z = wc
        elif self.conf["va_cond"]["use_module"]:
            z = vc

        if z is not None:
            z = (
                self.audio_module(z)["z"]
                if not isinstance(self.audio_module, nn.Identity)
                else self.audio_module(z)
            )

        return z


class VAPHead(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.type = conf["type"]

        self.rea_t = Rearrange("b t d -> b d t")
        self.rea_d = Rearrange("b d t -> b t d")
        self.one_frame_head = nn.Linear(conf["t_dim"], 1)

        if self.type == "comparative":
            self.projection_head = nn.Linear(conf["d_dim"], 1)
            self.output_dim = 1
        else:
            self.total_bins = 2 * len(conf["bin_times"])
            if self.type == "independent":
                self.projection_head = nn.Sequential(
                    nn.Linear(conf["d_dim"], self.total_bins),
                    Rearrange("... (c f) -> ... c f", c=2, f=self.total_bins // 2),
                )
                self.output_dim = (2, conf["bin_times"])
            else:
                self.n_classes = 2**self.total_bins
                self.projection_head = nn.Linear(conf["d_dim"], self.n_classes)
                self.output_dim = self.n_classes

    def __repr__(self):
        s = "VAPHead\n"
        s += f"  type: {self.type}"
        s += f"  output: {self.output_dim}"
        return super().__repr__()

    def forward(self, x):
        # x = x.contiguous().view(x.size(0), -1).unsqueeze(1) # old
        x = self.rea_t(x)
        x = self.one_frame_head(x)
        x = self.rea_d(x)
        x = self.projection_head(x)
        return x


class NonVerbalCondition(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        num_features = {
            "gaze": 3,
            "au": 18,
            "head": 4,
            "pose": 21,
        }
        flags = ["gaze", "au", "head", "pose"]
        self.flags = {flag: conf[flag]["use_module"] for flag in flags}

        for flag in flags:
            if self.flags[flag]:
                setattr(
                    self,
                    f"{flag}_linear",
                    nn.Linear(
                        num_features[flag], conf[flag][f"{flag}_module"]["input_size"]
                    ),
                )
                setattr(
                    self,
                    f"{flag}_ln",
                    nn.LayerNorm(conf[flag][f"{flag}_module"]["input_size"]),
                )

                module = (
                    AR(conf[flag][f"{flag}_module"])
                    if conf[flag][f"{flag}_module"]["use_module"]
                    else nn.Identity()
                )
                if self.conf["user1_input"] and self.conf["user2_input"]:
                    for user in ["user1", "user2"]:
                        setattr(self, f"{flag}_module_{user}", module)
                elif self.conf["user1_input"]:
                    setattr(self, f"{flag}_module_user1", module)
                elif self.conf["user2_input"]:
                    setattr(self, f"{flag}_module_user2", module)
                else:
                    print(
                        "Error: NonVerbalCondition must be true for either user1_input or user2_input"
                    )
                    exit(1)

        self.conf["non_verbal_module"]["input_size"] = sum(
            conf[flag][f"{flag}_module"]["input_size"]
            for flag in flags
            if self.flags[flag]
        )
        self.batch_norm = nn.BatchNorm1d(self.conf["t_dim"])
        self.non_verbal_module = self._init_module("non_verbal_module", AR)

        self.linear = nn.Linear(
            sum(
                conf[flag][f"{flag}_module"]["input_size"]
                for flag in flags
                if self.flags[flag]
            ),
            conf["linear"]["output_size"],
        )
        self.ln = nn.LayerNorm(conf["linear"]["output_size"])

    def _init_module(self, conf_key, module_class):
        if self.conf[conf_key]["use_module"]:
            return module_class(self.conf[conf_key])
        return nn.Identity()

    def get_module_output(self, module, input):
        return (
            module(input)["z"] if not isinstance(module, nn.Identity) else module(input)
        )

    def forward(self, **kwargs):
        conditions = []
        for flag in self.flags:
            if self.flags[flag]:
                linear = getattr(self, f"{flag}_linear")
                ln = getattr(self, f"{flag}_ln")

                if self.conf["user1_input"] and self.conf["user2_input"]:
                    user1 = kwargs[f"{flag}_user1"]
                    user2 = kwargs[f"{flag}_user2"]
                    ln = getattr(self, f"{flag}_ln")
                    module1 = getattr(self, f"{flag}_module_user1")
                    module2 = getattr(self, f"{flag}_module_user2")
                    z_1 = self.get_module_output(module1, user1)
                    z_2 = self.get_module_output(module2, user2)
                    z = ln(linear(z_1 + z_2))
                elif self.conf["user1_input"]:
                    user1 = kwargs[f"{flag}_user1"]
                    module1 = getattr(self, f"{flag}_module_user1")
                    z_1 = self.get_module_output(module1, user1)
                    z = ln(linear(z_1))
                else:
                    user2 = kwargs[f"{flag}_user2"]
                    module2 = getattr(self, f"{flag}_module_user2")
                    z_2 = self.get_module_output(module2, user2)
                    z = ln(linear(z_2))
                # z = linear(z_1 + z_2)
                conditions.append(z)

        z = torch.cat(conditions, dim=2)
        z = self.get_module_output(self.non_verbal_module, self.batch_norm(z))
        nc = self.ln(self.linear(z))

        return nc


class Net(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        self.audio_cond = self._init_module("audio_cond", AudioCondition)
        self.non_verbal_cond = self._init_module("non_verbal_cond", NonVerbalCondition)
        self.main_module = self._init_module("main_module", AR)
        self.vap_head = VAPHead(conf["vap"])

    def _init_module(self, conf_key, module_class):
        if self.conf[conf_key]["use_module"]:
            return module_class(self.conf[conf_key])
        return nn.Identity()

    def loss_vad_projection(self, logits, labels, reduction="mean"):
        loss = F.cross_entropy(
            einops.rearrange(logits, "b n d -> (b n) d"),
            einops.rearrange(labels, "b n -> (b n)"),
            reduction=reduction,
        )
        if reduction == "none":
            n = logits.shape[1]
            loss = einops.rearrange(loss, "(b n) -> b n", n=n)
        return loss

    def forward(self, **kwargs):
        ac = nc = torch.tensor([])

        if self.conf["audio_cond"]["use_module"]:
            ac = self.audio_cond(
                waveform=kwargs["waveform"],
                va=kwargs["va"],
                waveform_user1=kwargs.get("waveform_user1", None),
                waveform_user2=kwargs.get("waveform_user2", None),
                va_history=kwargs.get("va_history", None),
            )
            ac = ac[:, : kwargs["va"].shape[1]]

        if self.conf["non_verbal_cond"]["use_module"]:
            nc = self.non_verbal_cond(
                gaze_user1=kwargs.get("gaze_user1", None),
                au_user1=kwargs.get("au_user1", None),
                head_user1=kwargs.get("head_user1", None),
                pose_user1=kwargs.get("pose_user1", None),
                gaze_user2=kwargs.get("gaze_user2", None),
                au_user2=kwargs.get("au_user2", None),
                head_user2=kwargs.get("head_user2", None),
                pose_user2=kwargs.get("pose_user2", None),
            )
            nc = nc[:, : kwargs["va"].shape[1]]

        z = None
        if (
            self.conf["audio_cond"]["use_module"]
            & self.conf["non_verbal_cond"]["use_module"]
        ):
            z = torch.cat((ac, nc), 2)
        elif self.conf["audio_cond"]["use_module"]:
            z = ac
        elif self.conf["non_verbal_cond"]["use_module"]:
            z = nc
        else:
            print("Error: No Audio Condition or Non Verbal Conditon")
            exit(1)

        z = (
            self.main_module(z)["z"]
            if not isinstance(self.main_module, nn.Identity)
            else self.main_module(z)
        )
        out = {"logits_vp": self.vap_head(z)}

        return out


class Model(pl.LightningModule):
    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf
        self.conf["model"]["vap"]["t_dim"] = (
            self.conf["data"]["vad_hz"] * self.conf["data"]["audio_duration"]
        )
        self.conf["model"]["non_verbal_cond"]["t_dim"] = (
            self.conf["data"]["vad_hz"] * self.conf["data"]["audio_duration"]
        )
        if (
            self.conf["model"]["audio_cond"]["use_module"]
            & self.conf["model"]["non_verbal_cond"]["use_module"]
        ):
            self.conf["model"]["vap"]["d_dim"] = (
                self.conf["model"]["audio_cond"]["audio_module"]["input_size"]
                + self.conf["model"]["non_verbal_cond"]["linear"]["output_size"]
            )
        elif self.conf["model"]["audio_cond"]["use_module"]:
            self.conf["model"]["vap"]["d_dim"] = self.conf["model"]["audio_cond"][
                "audio_module"
            ]["input_size"]
        elif self.conf["model"]["non_verbal_cond"]["use_module"]:
            self.conf["model"]["vap"]["d_dim"] = self.conf["model"]["non_verbal_cond"][
                "linear"
            ]["output_size"]
        self.conf["model"]["main_module"]["input_size"] = self.conf["model"]["vap"][
            "d_dim"
        ]
        self.frame_hz = self.conf["model"]["encoder"]["frame_hz"]

        # Network
        self.net = Net(self.conf["model"]).to(
            conf["train"]["device"]
        )  # x, vf, vh -> logits
        self.vap_type = conf["model"]["vap"]["type"]

        # VAP: labels, logits -> zero-shot probs
        self.VAP = VAP(
            type=conf["model"]["vap"]["type"],
            bin_times=conf["model"]["vap"]["bin_times"],
            frame_hz=conf["model"]["encoder"]["frame_hz"],
            pre_frames=conf["model"]["vap"]["pre_frames"],
            threshold_ratio=conf["model"]["vap"]["bin_threshold"],
        )

        # Metrics
        self.val_metric = self.init_metric()  # self.init_metric()
        self.test_metric = None  # set in test if necessary

        # Training params
        self.learning_rate = conf["train"]["learning_rate"]
        self.save_hyperparameters()

    @property
    def model_summary(self):
        frame_audio = (
            self.conf["data"]["sample_rate"] * self.conf["data"]["audio_duration"]
        )
        frame_vad = self.conf["data"]["vad_hz"] * self.conf["data"]["audio_duration"]
        inputs = {
            "waveform": torch.randn(self.conf["data"]["batch_size"], frame_audio),
            "va": torch.randn(self.conf["data"]["batch_size"], frame_vad, 2),
            "waveform_user1": torch.randn(
                self.conf["data"]["batch_size"], frame_audio
            ),
            "waveform_user2": torch.randn(
                self.conf["data"]["batch_size"], frame_audio
            ),
            "va_history": torch.randn(self.conf["data"]["batch_size"], frame_vad, 5),
            "gaze_user1": torch.randn(self.conf["data"]["batch_size"], frame_vad, 3),
            "au_user1": torch.randn(self.conf["data"]["batch_size"], frame_vad, 18),
            "pose_user1": torch.randn(self.conf["data"]["batch_size"], frame_vad, 21),
            "head_user1": torch.randn(self.conf["data"]["batch_size"], frame_vad, 4),
            "gaze_user2": torch.randn(self.conf["data"]["batch_size"], frame_vad, 3),
            "au_user2": torch.randn(self.conf["data"]["batch_size"], frame_vad, 18),
            "pose_user2": torch.randn(self.conf["data"]["batch_size"], frame_vad, 21),
            "head_user2": torch.randn(self.conf["data"]["batch_size"], frame_vad, 4),
        }
        inputs = to_device(inputs, self.conf["train"]["device"])

        return summary(
            model=self.net,
            device=self.conf["train"]["device"],
            input_data=inputs,
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            depth=10,
            row_settings=["var_names"],
        )

    @property
    def inference_speed(self):
        inputs = {
            "waveform": torch.randn(1, 160000),
            "va": torch.randn(1, 250, 2),
            "waveform_user1": torch.randn(1, 160000),
            "waveform_user2": torch.randn(1, 160000),
            "va_history": torch.randn(1, 250, 5),
            "gaze_user1": torch.randn(1, 250, 3),
            "au_user1": torch.randn(1, 250, 18),
            "pose_user1": torch.randn(1, 250, 21),
            "head_user1": torch.randn(1, 250, 4),
            "gaze_user2": torch.randn(1, 250, 3),
            "au_user2": torch.randn(1, 250, 18),
            "pose_user2": torch.randn(1, 250, 21),
            "head_user2": torch.randn(1, 250, 4),
        }
        inputs = to_device(inputs, self.conf["train"]["device"])

        execution_times = []

        for _ in range(1000):
            time_start = time.perf_counter()
            self.net(**inputs)
            time_end = time.perf_counter()

            execution_times.append(time_end - time_start)

        mean_time = sum(execution_times) / len(execution_times)
        variance = sum([(x - mean_time) ** 2 for x in execution_times]) / len(
            execution_times
        )

        return mean_time, variance

    def init_metric(
        self,
        conf=None,
        threshold_shift_hold=None,
        threshold_pred_shift=None,
        threshold_pred_ov=None,
        threshold_short_long=None,
        threshold_bc_pred=None,
        shift_hold_pr_curve=False,
        bc_pred_pr_curve=False,
        shift_pred_pr_curve=False,
        ov_pred_pr_curve=False,
        long_short_pr_curve=False,
    ):
        if conf is None:
            conf = self.conf

        if threshold_pred_shift is None:
            threshold_shift_hold = conf["events"]["threshold"]["SH"]

        if threshold_pred_shift is None:
            threshold_pred_shift = conf["events"]["threshold"]["S_pred"]
        
        if threshold_pred_ov is None:
            threshold_pred_ov = conf["events"]["threshold"]["OV_pred"]

        if threshold_bc_pred is None:
            threshold_bc_pred = conf["events"]["threshold"]["BC_pred"]

        if threshold_short_long is None:
            threshold_short_long = conf["events"]["threshold"]["SL"]

        metric = TurnTakingMetrics(
            hs_kwargs=conf["events"]["SH"],
            bc_kwargs=conf["events"]["BC"],
            metric_kwargs=conf["events"]["metric"],
            threshold_shift_hold=threshold_shift_hold,
            threshold_pred_shift=threshold_pred_shift,
            threshold_pred_ov=threshold_pred_ov,
            threshold_short_long=threshold_short_long,
            threshold_bc_pred=threshold_bc_pred,
            shift_hold_pr_curve=shift_hold_pr_curve,
            shift_pred_pr_curve=shift_pred_pr_curve,
            ov_pred_pr_curve=ov_pred_pr_curve,
            bc_pred_pr_curve=bc_pred_pr_curve,
            long_short_pr_curve=long_short_pr_curve,
            frame_hz=self.frame_hz,
        )
        metric = metric.to(self.device)
        return metric

    def summary(self):
        s = "Model\n"
        s += f"{self.net}"
        s += f"{self.VAP}"
        return s

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def calc_losses(self, logits, va_labels, reduction="mean"):
        if self.vap_type == "comparative":
            loss = F.binary_cross_entropy_with_logits(logits, va_labels.unsqueeze(-1))
        elif self.vap_type == "independent":
            loss = F.binary_cross_entropy_with_logits(
                logits, va_labels, reduction=reduction
            )
        else:
            loss = self.net.loss_vad_projection(
                logits, labels=va_labels, reduction=reduction
            )
        return loss

    def shared_step(self, batch, reduction="mean"):
        # Forward pass
        batch = to_device(batch, self.conf["train"]["device"])
        # for k, b in batch.items():
        #     if isinstance(b, list):
        #         print(f"{k}: {np.array(b).shape}")
        #     else:
        #         print(f"{k}: {b.shape}")
        out = self(
            waveform=batch["waveform"],
            va=batch["vad"],
            waveform_user1=batch.get("waveform_user1", None),
            waveform_user2=batch.get("waveform_user2", None),
            va_history=batch.get("vad_history", None),
            gaze_user1=batch.get("gaze_user1", None),
            au_user1=batch.get("au_user1", None),
            head_user1=batch.get("head_user1", None),
            pose_user1=batch.get("pose_user1", None),
            gaze_user2=batch.get("gaze_user2", None),
            au_user2=batch.get("au_user2", None),
            head_user2=batch.get("head_user2", None),
            pose_user2=batch.get("pose_user2", None),
        )
        out["va_labels"] = batch["label"]

        # Calculate Loss
        loss = self.calc_losses(
            logits=out["logits_vp"],
            va_labels=out["va_labels"],
            reduction=reduction,
        )
        out_loss = {"vp": loss.mean(), "total": loss.mean()}
        if reduction == "none":
            out_loss["frames"] = loss

        return out_loss, out, batch

    def get_event_max_frames(self, batch):
        total_frames = batch["vad"].shape[1]
        return total_frames - self.VAP.horizon_frames

    @torch.no_grad()
    def output(self, batch, out_device="cpu"):
        batch = to_device(batch, self.conf["train"]["device"])
        out = self(
            waveform=batch.get("waveform", None),
            va=batch.get("vad", None),
            waveform_user1=batch.get("waveform_user1", None),
            waveform_user2=batch.get("waveform_user2", None),
            va_history=batch.get("vad_history", None),
            gaze_user1=batch.get("gaze_user1", None),
            au_user1=batch.get("au_user1", None),
            head_user1=batch.get("head_user1", None),
            pose_user1=batch.get("pose_user1", None),
            gaze_user2=batch.get("gaze_user2", None),
            au_user2=batch.get("au_user2", None),
            head_user2=batch.get("head_user2", None),
            pose_user2=batch.get("pose_user2", None),
        )
        out = to_device(out, out_device)
        return out


def _test_model():
    from turntaking.utils import load_hydra_conf

    conf = load_hydra_conf()
    cfg_dict = dict(OmegaConf.to_object(conf))
    print(type(cfg_dict))
    model = Model(cfg_dict)
    print(model)
    model.val_metric = model.init_metric()


if __name__ == "__main__":
    _test_model()
