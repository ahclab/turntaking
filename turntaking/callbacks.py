import torch
import pytorch_lightning as pl
import wandb
import numpy as np
import copy

from turntaking.augmentations import (
    flatten_pitch_batch,
    shift_pitch_batch,
    low_pass_filter_resample,
    IntensityNeutralizer,
)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.model = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # DataParallelを使用している場合はmodel.moduleを取り出す。
        torch.save(model_to_save.state_dict(), self.path)
        torch.save(model_to_save, self.path.replace("pt", "pth"))
        self.model = copy.deepcopy(model_to_save)
        self.val_loss_min = val_loss


class SymmetricSpeakersCallback(pl.Callback):
    """
    This callback "flips" the speakers such that we get a fair evaluation not dependent on the
    biased speaker-order / speaker-activity

    The audio is mono which requires no change.

    The only change we apply is to flip the channels in the VAD-tensor and get the corresponding VAD-history
    which is defined as the ratio of speaker 0 (i.e. vad_history_flipped = 1 - vad_history)
    """

    def get_symmetric_batch(self, batch):
        """Appends a flipped version of the batch-samples"""
        for k, v in batch.items():
            if k == "vad":
                flipped = torch.stack((v[..., 1], v[..., 0]), dim=-1)
            elif k == "vad_history":
                flipped = 1.0 - v
            else:
                flipped = v
            if isinstance(v, torch.Tensor):
                batch[k] = torch.cat((v, flipped))
            else:
                batch[k] = v + flipped
        return batch

    def on_train_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = self.get_symmetric_batch(batch)

    def on_test_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = self.get_symmetric_batch(batch)

    def on_val_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = self.get_symmetric_batch(batch)


class FlattenPitchCallback(pl.Callback):
    """ """

    def __init__(
        self,
        target_f0: int = -1,
        statistic: str = "mean",
        stats_frame_length: int = 800,
        stats_hop_length: int = 320,
        sample_rate: int = 16000,
        to_mono: bool = True,
    ):
        super().__init__()
        self.statistic = statistic
        self.stats_frame_length = stats_frame_length
        self.stats_hop_length = stats_hop_length
        self.target_f0 = target_f0
        self.sample_rate = sample_rate
        self.to_mono = to_mono

    def flatten_pitch(self, batch, device):
        """Appends a flipped version of the batch-samples"""
        flat_waveform = flatten_pitch_batch(
            waveform=batch["waveform"].cpu(),
            vad=batch["vad"],
            target_f0=self.target_f0,
            statistic=self.statistic,
            stats_frame_length=self.stats_frame_length,
            stats_hop_length=self.stats_hop_length,
            sample_rate=self.sample_rate,
            to_mono=self.to_mono,
        )
        batch["waveform"] = flat_waveform.to(device)
        return batch

    def on_test_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = self.flatten_pitch(batch, device=pl_module.device)

    def on_val_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = self.flatten_pitch(batch, device=pl_module.device)

    def on_train_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = self.flatten_pitch(batch, device=pl_module.device)


class NeutralIntensityCallback(pl.Callback):
    """ """

    def __init__(
        self,
        vad_hz,
        vad_cutoff: float = 0.2,
        hop_time: float = 0.01,
        f0_min: int = 60,
        statistic: str = "mean",
        sample_rate: int = 16000,
        to_mono: bool = True,
    ):
        super().__init__()
        self.hop_time = hop_time
        self.vad_hz = vad_hz
        self.f0_min = f0_min
        self.vad_cutoff = vad_cutoff
        self.statistic = statistic
        self.sample_rate = sample_rate
        self.to_mono = to_mono
        self.neutralizer = IntensityNeutralizer(
            hop_time=hop_time,
            vad_hz=vad_hz,
            f0_min=f0_min,
            vad_cutoff=vad_cutoff,
            scale_stat=statistic,
            sample_rate=sample_rate,
            to_mono=to_mono,
        )

    def neutral_batch(self, batch):
        batch_size = batch["waveform"].shape[0]
        n_frames = batch["vad_history"].shape[1]

        combine = False

        if batch["waveform"].ndim == 3:
            combine = True

        new_waveform = []
        for b in range(batch_size):
            vad = batch["vad"][b, :n_frames]
            if combine:
                y_tmp = batch["waveform"][b].mean(0, keepdim=True)
            else:
                y_tmp = batch["waveform"][b]
            y, _ = self.neutralizer(y_tmp, vad=vad)
            new_waveform.append(y)
        batch["waveform"] = torch.cat(new_waveform)
        return batch

    def on_test_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = self.neutral_batch(batch)

    def on_val_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = self.neutral_batch(batch)

    def on_train_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = self.neutral_batch(batch)


class LowPassFilterCallback(pl.Callback):
    """
    Applies a low-pass filter by downsampling and upsampling the signal based on Nyquist theorem.
    """

    def __init__(
        self,
        cutoff_freq: int = 300,
        sample_rate: int = 16000,
        norm: bool = True,
        to_mono: bool = True,
    ):
        super().__init__()
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        self.norm = norm
        self.to_mono = to_mono

    def normalize(self, x):
        assert x.ndim == 2, f"normalization expects (B, n_samples) got {x.shape}"
        xx = x - x.min(-1, keepdim=True).values
        xx = 2 * xx / xx.max()
        xx = xx - 1.0
        return xx

    def low_pass(self, waveform):
        waveform = low_pass_filter_resample(
            waveform, self.cutoff_freq, self.sample_rate
        )
        if self.to_mono:
            waveform = waveform.mean(1)

        if self.norm:
            waveform = self.normalize(waveform)

        return waveform

    def on_test_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch["waveform"] = self.low_pass(batch["waveform"])

    def on_val_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch["waveform"] = self.low_pass(batch["waveform"])

    def on_train_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch["waveform"] = self.low_pass(batch["waveform"])


class ShiftPitchCallback(pl.Callback):
    def __init__(
        self, factor: float = 0.9, sample_rate: int = 16000, to_mono: bool = True
    ):
        super().__init__()
        self.factor = factor
        self.sample_rate = sample_rate
        self.to_mono = to_mono

    def shift_pitch(self, batch, device):
        flat_waveform = shift_pitch_batch(
            waveform=batch["waveform"].cpu(),
            factor=self.factor,
            vad=batch["vad"],
            sample_rate=self.sample_rate,
            to_mono=self.to_mono,
        )
        batch["waveform"] = flat_waveform.to(device)
        return batch

    def on_test_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = self.shift_pitch(batch, device=pl_module.device)

    def on_val_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = self.shift_pitch(batch, device=pl_module.device)

    def on_train_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = self.shift_pitch(batch, device=pl_module.device)


class WandbArtifactCallback(pl.Callback):
    def upload(self, trainer):
        run = trainer.logger.experiment
        print(f"Ending run: {run.id}")
        artifact = wandb.Artifact(f"{str(run.id)}_model", type="model")
        for path, val_loss in trainer.checkpoint_callback.best_k_models.items():
            print(f"Adding artifact: {path}")
            artifact.add_file(path)
        run.log_artifact(artifact)

    def on_train_end(self, trainer, pl_module):
        if trainer.global_rank > 0:  # add
            return
        print("Training End ---------------- Custom Upload")
        self.upload(trainer)

    def on_exception(self, trainer, pl_module, exception):
        if isinstance(exception, KeyboardInterrupt):
            print("Keyboard Interruption ------- Custom Upload")
            self.upload(trainer)


if __name__ == "__main__":
    from os.path import join, basename
    from turntaking.evaluation.evaluation_phrases import load_model_dset
    import sounddevice as sd

    ch_root = "assets/PaperB/checkpoints"
    checkpoint = join(ch_root, "cpc_48_50hz_15gqq5s5.ckpt")
    checkpoint = join(ch_root, "cpc_48_50hz_15gqq5s5.ckpt")
    model, dset = load_model_dset(checkpoint)
    checkpoint_name = basename(checkpoint)

    batch = dset.get_sample("student", "long", "female", 0)
    batch["waveform"] = batch["waveform"].unsqueeze(1)
    waveform = shift_pitch_batch(batch["waveform"].cpu(), factor=0.8)
    sd.play(waveform[0].cpu(), samplerate=16000)

    # test Callbacks
    batch = dset.get_sample("student", "long", "female", 0)
    batch["waveform"] = batch["waveform"].unsqueeze(1)
    # augmentation = "flat_f0"
    # augmentation = 'only_f0'
    augmentation = "shift_f0"
    clb = []
    if augmentation == "flat_f0":
        clb.append(FlattenPitchCallback())
    elif augmentation == "only_f0":
        clb.append(LowPassFilterCallback(cutoff_freq=300))
    elif augmentation == "shift_f0":
        clb.append(ShiftPitchCallback(factor=0.9))
    elif augmentation == "flat_intensity":
        pass
    clb[0].on_test_batch_start(trainer=None, pl_module=model, batch=batch)
    print(batch["waveform"])
    sd.play(batch["waveform"][0].cpu(), samplerate=16000)
