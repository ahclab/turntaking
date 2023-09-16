import random
from os import cpu_count, environ

# omit verbose `datasets` info
# WARNING: Setting verbosity lev del by hand...
environ["DATASETS_VERBOSITY"] = "error"

from os.path import join
from typing import Optional, Dict
from datasets import concatenate_datasets

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm

from turntaking.dataload.dialog_audio_dataset import DialogAudioDataset
from turntaking.dataload.dataset.switchboard import load_switchboard
from turntaking.dataload.dataset.noxi import load_noxi
from turntaking.dataload.utils import repo_root, OmegaConfArgs, load_config


def get_dialog_audio_datasets(
    datasets, split, train_files=None, val_files=None, test_files=None
):
    """
    Load multiple dataset (of Huggingface `datasets` type) and concatenate to
    a single dataset.
    """
    dsets = []
    if datasets == "switchboard":
        dsets.append(
            load_switchboard(
                split=split,
                train_files=train_files,
                val_files=val_files,
                test_files=test_files,
            )
        )
    elif datasets == "noxi":
        dsets.append(
            load_noxi(
                split=split,
                train_files=train_files,
                val_files=val_files,
                test_files=test_files,
            )
        )
    else:
        raise NotImplementedError(f"{d} is not yet implemented")
    dsets = concatenate_datasets(dsets)
    return dsets


DEFAULT_CONFIG = join(repo_root(), "config/dset_dialog_audio.yaml")


class DialogAudioDM(pl.LightningDataModule):
    def __init__(
        self,
        datasets,
        type="sliding",  # ipu
        audio_mono=True,
        audio_duration=10,
        audio_normalize=True,
        audio_overlap=2,
        audio_include_ratio=0.4,
        audio_context_duration=8,
        ipu_min_time=1,
        ipu_pause_time=0.2,
        sample_rate=16000,
        vad=True,
        vad_hz=100,
        vad_horizon=2,
        vad_history=False,
        vad_history_times=[60, 30, 10, 5],
        flip_channels=True,
        train_files=None,
        val_files=None,
        test_files=None,
        batch_size=4,
        num_workers=0,
        pin_memory=True,
        transforms=None,
        label_type="discrete",
        bin_times=[0.20, 0.40, 0.60, 0.80],
        pre_frames=2,
        threshold_ratio=0.5,
        undersampling=False,
        oversampling=False,
    ):
        super().__init__()
        self.datasets = datasets  # names of datasets
        self.type = type
        self.transforms = transforms

        # IterableDataset
        # Audio (waveforms)
        self.audio_mono = audio_mono
        self.audio_duration = audio_duration
        self.sample_rate = sample_rate
        # Sliding Window Dataset
        self.audio_overlap = audio_overlap
        self.audio_normalize = audio_normalize
        self.audio_include_ratio = audio_include_ratio

        # IPU Dataset
        self.audio_context_duration = audio_context_duration
        self.ipu_min_time = ipu_min_time
        self.ipu_pause_time = ipu_pause_time

        # VAD
        self.vad = vad
        self.vad_hz = vad_hz
        self.vad_horizon = vad_horizon
        self.vad_history = vad_history
        self.vad_history_times = vad_history_times
        self.flip_channels = flip_channels

        # Dataset Files
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files

        # DataLoder
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        # Lable
        self.label_type = label_type
        self.bin_times = bin_times
        self.pre_frames = pre_frames
        self.threshold_ratio = threshold_ratio
        self.undersampling = undersampling
        self.oversampling = oversampling

    def prepare_data(self):
        """
        loads the data over all splits.
        Using huggingface datasets which may process/download and cache the data or used cache versions.

        Doing this here to make sure that all the data is accessable before any training, evaluation etc.
        However, uses the same call as is done in `self.setup()`

        So this might be unnecessary given we use Huggingface `datasets` ...

        To avoid the `datasets` logging warnings set `DATASETS_VERBOSITY=error` in your terminal ENV.
        """
        for split in ["train", "validation", "test"]:
            _ = get_dialog_audio_datasets(
                datasets=self.datasets,
                split=split,
            )

    def _dataset(self, dset, split="train"):
        # Only flip during training...
        if split == "train":
            flip = self.flip_channels
            undersampling = self.undersampling
            oversampling = self.oversampling
        elif split == "val":
            flip = False
            undersampling = False
            oversampling = False
        elif split == "test":
            flip = False
            undersampling = False
            oversampling = False
        else:
            print("SPLIT ERROR")
            exit(1)

        return DialogAudioDataset(
            dataset=dset,
            transforms=self.transforms,
            type=self.type,
            audio_mono=self.audio_mono,
            audio_duration=self.audio_duration,
            audio_overlap=self.audio_overlap,
            audio_normalize=self.audio_normalize,
            sample_rate=self.sample_rate,
            vad=self.vad,
            vad_hz=self.vad_hz,
            vad_horizon=self.vad_horizon,
            vad_history=self.vad_history,
            vad_history_times=self.vad_history_times,
            flip_channels=flip,
            flip_probability=0.5,
            label_type=self.label_type,
            bin_times=self.bin_times,
            pre_frames=self.pre_frames,
            threshold_ratio=self.threshold_ratio,
            undersampling=undersampling,
            oversampling=oversampling
        )

    def setup(self, stage: Optional[str] = "fit"):
        """Loads the datasets"""
        
        # Define a helper function to avoid repetitive code
        def get_dataset(split: str):
            return get_dialog_audio_datasets(
                datasets=self.datasets,
                split=split,
                train_files=self.train_files,
                val_files=self.val_files,
                test_files=self.test_files
            )
        
        if stage == "fit":
            self.train_dset = self._dataset(get_dataset("train"), split="train")
            self.val_dset = self._dataset(get_dataset("val"), split="val")
            self.test_dset = None

        elif stage == "test":
            self.train_dset = None
            self.val_dset = None
            self.test_dset = self._dataset(get_dataset("test"), split="test")

        else:
            self.train_dset = self._dataset(get_dataset("train"), split="train")
            self.val_dset = self._dataset(get_dataset("val"), split="val")
            self.test_dset = self._dataset(get_dataset("test"), split="test")


    def collate_fn(self, batch):
        def pad_tensor(tensor, target_size, dim=-1):
            if tensor.size(dim) < target_size:
                padding_size = target_size - tensor.size(dim)
                padding = torch.zeros(
                    *tensor.size()[:dim], padding_size, *tensor.size()[dim + 1 :]
                )
                tensor = torch.cat((tensor, padding), dim)
            elif tensor.size(dim) > target_size:
                tensor = tensor.narrow(dim, 0, target_size)
            return tensor

        keys = [
            "waveform",
            "waveform_expert",
            "waveform_novice",
            "vad",
            "vad_history",
            "gaze_expert",
            "au_expert",
            "pose_expert",
            "head_expert",
            "gaze_novice",
            "au_novice",
            "pose_novice",
            "head_novice",
            "label",
        ]
        target_sizes = {
            "waveform": self.audio_duration * self.sample_rate,
            "waveform_expert": self.audio_duration * self.sample_rate,
            "waveform_novice": self.audio_duration * self.sample_rate,
            "vad": self.audio_duration * self.vad_hz,
            "vad_history": self.audio_duration * self.vad_hz,
            "gaze_expert": self.vad_hz * self.audio_duration,
            "au_expert": self.vad_hz * self.audio_duration,
            "head_expert": self.vad_hz * self.audio_duration,
            "pose_expert": self.vad_hz * self.audio_duration,
            "gaze_novice": self.vad_hz * self.audio_duration,
            "au_novice": self.vad_hz * self.audio_duration,
            "head_novice": self.vad_hz * self.audio_duration,
            "pose_novice": self.vad_hz * self.audio_duration,
            "label": len(self.bin_times) if self.label_type == "independent" else 1,
        }
        dimensions = {
            "vad": -2,
            "vad_history": -2,
            "gaze_expert": -2,
            "au_expert": -2,
            "head_expert": -2,
            "pose_expert": -2,
            "gaze_novice": -2,
            "au_novice": -2,
            "head_novice": -2,
            "pose_novice": -2,
            "label": -1,
        }
        ret = {key: [] for key in keys}

        for b in batch:
            for key in keys:
                if key in b:
                    ret[key].append(
                        pad_tensor(b[key], target_sizes[key], dimensions.get(key, -1))
                    )

        ret = {key: torch.cat(value) for key, value in ret.items() if len(value) > 0}
        ret["dset_name"] = [b["dataset_name"] for b in batch]
        ret["session"] = [b["session"] for b in batch]

        return ret

    def get_full_sample(self, split="val"):
        if split == "train":
            return self.train_dset.get_full_sample()
        elif split == "val":
            return self.val_dset.get_full_sample()
        elif split == "test":
            return self.test_dset.get_full_sample()
        else:
            return None

    def change_frame_mode(self, mode="False"):
        if self.train_dset is not None:
            self.train_dset.change_frame_mode(mode)
        if self.val_dset is not None:
            self.val_dset.change_frame_mode(mode)
        if self.test_dset is not None:
            self.test_dset.change_frame_mode(mode)

    def seed_worker(self, worker_id):
        np.random.seed(worker_id)
        random.seed(worker_id)

    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=True,
            worker_init_fn=self.seed_worker,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=False,
            worker_init_fn=self.seed_worker,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=False,
            worker_init_fn=self.seed_worker,
        )

    def __repr__(self):
        s = "DialogAudioDM"
        s += f"\n\tbatch_size: {self.batch_size}"
        s += f"\n\tpin_memory: {self.pin_memory}"
        s += f"\n\tnum_workers: {self.num_workers}"

        if hasattr(self, "train_dset"):
            s += "\n\t" + ("-" * 10) + "\n"
            s += str(self.train_dset)
        elif hasattr(self, "test_dset"):
            s += "\n\t" + ("-" * 10) + "\n"
            s += str(self.train_dset)
        return s

    @staticmethod
    def print_dm(data_conf, args=None):
        print("-" * 60)
        print("Dataloader")
        for k, v in data_conf["dataset"].items():
            print(f"  {k}: {v}")
        if args is not None:
            print("  batch_size: ", args.batch_size)
            print("  num_workers: ", args.num_workers)
        print()

    @staticmethod
    def default_config_path():
        return DEFAULT_CONFIG

    @staticmethod
    def load_config(path=None, args=None, format="dict") -> Dict:
        if path is None:
            path = DialogAudioDM.default_config_path()
        return load_config(path, args=args, format=format)

    @staticmethod
    def add_data_specific_args(parent_parser):
        """argparse arguments for SoSIModel (based on yaml-config)"""
        parser = parent_parser.add_argument_group("ULMProjection")
        parser.add_argument("--data_conf", default=None, type=str)
        parser.add_argument("--batch_size", default=4, type=int)
        parser.add_argument("--num_workers", default=cpu_count(), type=int)
        parser.add_argument("--train_files", default=None, type=str)
        parser.add_argument("--val_files", default=None, type=str)
        parser.add_argument("--test_files", default=None, type=str)

        # A workaround for OmegaConf + WandB-Sweeps
        conf = DialogAudioDM.load_config()
        parser = OmegaConfArgs.add_argparse_args(parser, conf)
        return parent_parser


if __name__ == "__main__":
    data_conf = DialogAudioDM.load_config()

    dm = DialogAudioDM(
        datasets="noxi",
        audio_duration=10,
        audio_overlap=9.5,
        vad_hz=25,
        num_workers=0,
    )

    dm.setup(None)
    print(dm)

    print("\nBATCH DATASET")
    d = dm.val_dset[0]
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")

    print("FULL SAMPLE")
    d = dm.get_full_sample("val")
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")

    for i in range(len(dm.val_dset)):
        d = dm.val_dset[i]

    print("DATALOADER TEST")
    pbar_val = tqdm(
        enumerate(dm.train_dataloader()),
        total=len(dm.train_dataloader()),
    )
    for ii, batch in pbar_val:
        pass
    pbar_val = tqdm(
        enumerate(dm.val_dataloader()),
        total=len(dm.val_dataloader()),
    )
    for ii, batch in pbar_val:
        pass

    print("Frame Mode ON")
    dm.change_frame_mode(True)
    pbar_val = tqdm(
        enumerate(dm.val_dataloader()),
        total=len(dm.val_dataloader()),
    )
    for ii, batch in pbar_val:
        pass
    pbar_val = tqdm(
        enumerate(dm.test_dataloader()),
        total=len(dm.test_dataloader()),
    )
    for ii, batch in pbar_val:
        pass
