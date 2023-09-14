from os.path import basename, dirname
from os import remove
from omegaconf import OmegaConf
import json
import subprocess
import pandas as pd
from os.path import join

import torch
import torchaudio
import torchaudio.functional as AF
from torchaudio.backend.sox_io_backend import info as info_sox

from decimal import Decimal, ROUND_HALF_UP


def repo_root():
    """
    Returns the absolute path to the git repository
    """
    root = dirname(__file__)
    root = dirname(root)
    return root


NORM_JSON = join(repo_root(), "dataload/dataset/noxi/files/normalize.json")


def samples_to_frames(s, hop_len):
    return round(s / hop_len)


def sample_to_time(n_samples, sample_rate):
    return n_samples / sample_rate


def frames_to_time(f, hop_time):
    return f * hop_time


def time_to_frames(t, hop_time):
    return int(
        Decimal(str(t / hop_time)).quantize(Decimal("0"), rounding=ROUND_HALF_UP)
    )  # ugly code


def time_to_frames_samples(t, sample_rate, hop_length):
    return int(
        Decimal(str(t * sample_rate / hop_length)).quantize(
            Decimal("0"), rounding=ROUND_HALF_UP
        )
    )


def time_to_samples(t, sample_rate):
    return int(
        Decimal(str(t * sample_rate)).quantize(Decimal("0"), rounding=ROUND_HALF_UP)
    )


def get_audio_info(audio_path):
    info = info_sox(audio_path)
    return {
        "name": basename(audio_path),
        "duration": sample_to_time(info.num_frames, info.sample_rate),
        "sample_rate": info.sample_rate,
        "num_frames": info.num_frames,
        "bits_per_sample": info.bits_per_sample,
        "num_channels": info.bits_per_sample,
    }


def load_waveform(
    path,
    sample_rate=None,
    start_time=None,
    end_time=None,
    normalize=False,
    mono=False,
    audio_normalize_threshold=0.05,
):
    if start_time is not None:
        info = get_audio_info(path)
        frame_offset = time_to_samples(start_time, info["sample_rate"])
        num_frames = info["num_frames"]
        if end_time is not None:
            num_frames = time_to_samples(end_time, info["sample_rate"]) - frame_offset
        else:
            num_frames = num_frames - frame_offset
        x, sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
    else:
        x, sr = torchaudio.load(path)

    if normalize:
        if x.shape[0] > 1:
            if x[0].abs().max() > audio_normalize_threshold:
                x[0] /= x[0].abs().max()
            if x[1].abs().max() > audio_normalize_threshold:
                x[1] /= x[1].abs().max()
        else:
            if x.abs().max() > audio_normalize_threshold:
                x /= x.abs().max()

    if mono and x.shape[0] > 1:
        x = x.mean(dim=0).unsqueeze(0)
        if normalize:
            if x.abs().max() > audio_normalize_threshold:
                x /= x.abs().max()

    if sample_rate:
        if sr != sample_rate:
            x = AF.resample(x, orig_freq=sr, new_freq=sample_rate)
            sr = sample_rate
    return x, sr


def load_multimodal_features(path, feature_name, normalize="batch_normalization"):
    df = pd.read_csv(path)
    feature_columns = []

    if feature_name == "gaze":
        feature_columns = ["gaze_x", "gaze_y", "gaze_confidence"]
    elif feature_name == "au":
        feature_columns = [
            f"AU{code:02d}"
            for code in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
        ] + ["gaze_confidence"]
    elif feature_name == "pose":
        feature_columns = [
            f"pose_{i}_{axis}" for i in range(1, 8) for axis in ["x", "y", "confidence"]
        ]
    elif feature_name == "head":
        feature_columns = ["head_x", "head_y", "head_z", "gaze_confidence"]
    else:
        print("ERROR: load_multimodal_features()")
        exit(1)

    features = torch.tensor(df[feature_columns].to_numpy())
    features = torch.nan_to_num(features, nan=0, posinf=0, neginf=0)

    return features


# def load_multimodal_features(
#     path,
#     feature_name,
#     normalize = "batch_normalization"
# ):
#     df = pd.read_csv(path)

#     if feature_name == "gaze":
#         confidence = torch.tensor(df["gaze_confidence"]).permute(*torch.arange(torch.tensor(df["gaze_confidence"]).ndim - 1, -1, -1))
#         gaze_x = torch.tensor(df["gaze_x"]).permute(*torch.arange(torch.tensor(df["gaze_x"]).ndim - 1, -1, -1))
#         gaze_y = torch.tensor(df["gaze_y"]).permute(*torch.arange(torch.tensor(df["gaze_y"]).ndim - 1, -1, -1))
#         gaze = torch.stack([gaze_x,gaze_y,confidence]).view(3,-1).permute(*torch.arange(torch.stack([gaze_x,gaze_y,confidence]).view(3,-1).ndim - 1, -1, -1))

#         ### normalization ###
#         if normalize == "batch_normalization":
#             batch_norm = nn.BatchNorm1d(3)
#             gaze = batch_norm(gaze.float())
#             gaze = torch.nan_to_num(gaze, nan=0, posinf=0, neginf=0)
#             gaze = gaze.clone().detach().requires_grad_(False)
#         elif normalize == "min-max":
#             gaze = torch.nan_to_num(gaze, nan=0, posinf=1, neginf=0)
#         else:
#             print(f"normalize error: {normalize}")
#             exit(1)

#         return gaze

#     elif feature_name == "au":
#         confidence = torch.tensor(df["gaze_confidence"]).permute(*torch.arange(torch.tensor(df["gaze_confidence"]).ndim - 1, -1, -1)).unsqueeze(1)
#         au_01 = torch.tensor(df["AU01"]).permute(*torch.arange(torch.tensor(df["AU01"]).ndim - 1, -1, -1)).unsqueeze(1)
#         au_02 = torch.tensor(df["AU02"]).permute(*torch.arange(torch.tensor(df["AU02"]).ndim - 1, -1, -1)).unsqueeze(1)
#         au_04 = torch.tensor(df["AU04"]).permute(*torch.arange(torch.tensor(df["AU04"]).ndim - 1, -1, -1)).unsqueeze(1)
#         au_05 = torch.tensor(df["AU01"]).permute(*torch.arange(torch.tensor(df["AU05"]).ndim - 1, -1, -1)).unsqueeze(1)
#         au_06 = torch.tensor(df["AU02"]).permute(*torch.arange(torch.tensor(df["AU06"]).ndim - 1, -1, -1)).unsqueeze(1)
#         au_07 = torch.tensor(df["AU04"]).permute(*torch.arange(torch.tensor(df["AU07"]).ndim - 1, -1, -1)).unsqueeze(1)
#         au_09 = torch.tensor(df["AU01"]).permute(*torch.arange(torch.tensor(df["AU09"]).ndim - 1, -1, -1)).unsqueeze(1)
#         au_10 = torch.tensor(df["AU02"]).permute(*torch.arange(torch.tensor(df["AU10"]).ndim - 1, -1, -1)).unsqueeze(1)
#         au_12 = torch.tensor(df["AU04"]).permute(*torch.arange(torch.tensor(df["AU12"]).ndim - 1, -1, -1)).unsqueeze(1)
#         au_14 = torch.tensor(df["AU01"]).permute(*torch.arange(torch.tensor(df["AU14"]).ndim - 1, -1, -1)).unsqueeze(1)
#         au_15 = torch.tensor(df["AU02"]).permute(*torch.arange(torch.tensor(df["AU15"]).ndim - 1, -1, -1)).unsqueeze(1)
#         au_17 = torch.tensor(df["AU04"]).permute(*torch.arange(torch.tensor(df["AU17"]).ndim - 1, -1, -1)).unsqueeze(1)
#         au_20 = torch.tensor(df["AU01"]).permute(*torch.arange(torch.tensor(df["AU20"]).ndim - 1, -1, -1)).unsqueeze(1)
#         au_23 = torch.tensor(df["AU02"]).permute(*torch.arange(torch.tensor(df["AU23"]).ndim - 1, -1, -1)).unsqueeze(1)
#         au_25 = torch.tensor(df["AU04"]).permute(*torch.arange(torch.tensor(df["AU25"]).ndim - 1, -1, -1)).unsqueeze(1)
#         au_26 = torch.tensor(df["AU02"]).permute(*torch.arange(torch.tensor(df["AU26"]).ndim - 1, -1, -1)).unsqueeze(1)
#         au_45 = torch.tensor(df["AU04"]).permute(*torch.arange(torch.tensor(df["AU45"]).ndim - 1, -1, -1)).unsqueeze(1)
#         au = torch.stack([au_01,au_02,au_04,au_05,au_06,au_07,au_09,au_10,au_12,au_14,au_15,au_17,au_20,au_23,au_25,au_26,au_45,confidence]
#                          ).view(18,-1
#                                 ).permute(*torch.arange(torch.stack([au_01,au_02,au_04,au_05,au_06,au_07,au_09,au_10,au_12,au_14,au_15,au_17,au_20,au_23,au_25,au_26,au_45,confidence]).view(18,-1).ndim - 1, -1, -1))

#         if normalize == "batch_normalization":
#             # batch_norm=nn.BatchNorm1d(3)
#             # au = batch_norm(au.float())
#             # au = torch.nan_to_num(au, nan=0, posinf=0, neginf=0)
#             # au = au.clone().detach().requires_grad_(False)
#             batch_norm = nn.BatchNorm1d(18)
#             au = batch_norm(au.float())
#             au = torch.nan_to_num(au, nan=0, posinf=0, neginf=0)
#             au = au.clone().detach().requires_grad_(False)
#             # au[torch.where(au==0)] = -1
#             # au = torch.nan_to_num(au, nan=0, posinf=1, neginf=-1) # -1 or 1
#         else:
#             print(f"normalize error: {normalize}")
#             exit(1)

#         return au

#     elif feature_name == "pose":
#         pose_1_x = torch.tensor(df["pose_1_x"]).permute(*torch.arange(torch.tensor(df["pose_1_x"]).ndim - 1, -1, -1)).unsqueeze(1)
#         pose_1_y = torch.tensor(df["pose_1_y"]).permute(*torch.arange(torch.tensor(df["pose_1_y"]).ndim - 1, -1, -1)).unsqueeze(1)
#         pose_1_confidence = torch.tensor(df["pose_1_confidence"]).permute(*torch.arange(torch.tensor(df["pose_1_confidence"]).ndim - 1, -1, -1)).unsqueeze(1)
#         pose_2_x = torch.tensor(df["pose_2_x"]).permute(*torch.arange(torch.tensor(df["pose_2_x"]).ndim - 1, -1, -1)).unsqueeze(1)
#         pose_2_y = torch.tensor(df["pose_2_y"]).permute(*torch.arange(torch.tensor(df["pose_2_y"]).ndim - 1, -1, -1)).unsqueeze(1)
#         pose_2_confidence = torch.tensor(df["pose_2_confidence"]).permute(*torch.arange(torch.tensor(df["pose_2_confidence"]).ndim - 1, -1, -1)).unsqueeze(1)
#         pose_3_x = torch.tensor(df["pose_3_x"]).permute(*torch.arange(torch.tensor(df["pose_3_x"]).ndim - 1, -1, -1)).unsqueeze(1)
#         pose_3_y = torch.tensor(df["pose_3_y"]).permute(*torch.arange(torch.tensor(df["pose_3_y"]).ndim - 1, -1, -1)).unsqueeze(1)
#         pose_3_confidence = torch.tensor(df["pose_3_confidence"]).permute(*torch.arange(torch.tensor(df["pose_3_confidence"]).ndim - 1, -1, -1)).unsqueeze(1)
#         pose_4_x = torch.tensor(df["pose_4_x"]).permute(*torch.arange(torch.tensor(df["pose_4_x"]).ndim - 1, -1, -1)).unsqueeze(1)
#         pose_4_y = torch.tensor(df["pose_4_y"]).permute(*torch.arange(torch.tensor(df["pose_4_y"]).ndim - 1, -1, -1)).unsqueeze(1)
#         pose_4_confidence = torch.tensor(df["pose_4_confidence"]).permute(*torch.arange(torch.tensor(df["pose_4_confidence"]).ndim - 1, -1, -1)).unsqueeze(1)
#         pose_5_x = torch.tensor(df["pose_5_x"]).permute(*torch.arange(torch.tensor(df["pose_5_x"]).ndim - 1, -1, -1)).unsqueeze(1)
#         pose_5_y = torch.tensor(df["pose_5_y"]).permute(*torch.arange(torch.tensor(df["pose_5_y"]).ndim - 1, -1, -1)).unsqueeze(1)
#         pose_5_confidence = torch.tensor(df["pose_5_confidence"]).permute(*torch.arange(torch.tensor(df["pose_5_confidence"]).ndim - 1, -1, -1)).unsqueeze(1)
#         pose_6_x = torch.tensor(df["pose_6_x"]).permute(*torch.arange(torch.tensor(df["pose_6_x"]).ndim - 1, -1, -1)).unsqueeze(1)
#         pose_6_y = torch.tensor(df["pose_6_y"]).permute(*torch.arange(torch.tensor(df["pose_6_y"]).ndim - 1, -1, -1)).unsqueeze(1)
#         pose_6_confidence = torch.tensor(df["pose_6_confidence"]).permute(*torch.arange(torch.tensor(df["pose_6_confidence"]).ndim - 1, -1, -1)).unsqueeze(1)
#         pose_7_x = torch.tensor(df["pose_7_x"]).permute(*torch.arange(torch.tensor(df["pose_7_x"]).ndim - 1, -1, -1)).unsqueeze(1)
#         pose_7_y = torch.tensor(df["pose_7_y"]).permute(*torch.arange(torch.tensor(df["pose_7_y"]).ndim - 1, -1, -1)).unsqueeze(1)
#         pose_7_confidence = torch.tensor(df["pose_7_confidence"]).permute(*torch.arange(torch.tensor(df["pose_7_confidence"]).ndim - 1, -1, -1)).unsqueeze(1)
#         pose = torch.stack([pose_1_x,pose_1_y,pose_1_confidence,
#                             pose_2_x,pose_2_y,pose_2_confidence,
#                             pose_3_x,pose_3_y,pose_3_confidence,
#                             pose_4_x,pose_4_y,pose_4_confidence,
#                             pose_5_x,pose_5_y,pose_5_confidence,
#                             pose_6_x,pose_6_y,pose_6_confidence,
#                             pose_7_x,pose_7_y,pose_7_confidence]
#                          ).view(21,-1
#                                 ).permute(*torch.arange(torch.stack([pose_1_x,pose_1_y,pose_1_confidence,
#                                                                      pose_2_x,pose_2_y,pose_2_confidence,
#                                                                     pose_3_x,pose_3_y,pose_3_confidence,
#                                                                     pose_4_x,pose_4_y,pose_4_confidence,
#                                                                     pose_5_x,pose_5_y,pose_5_confidence,
#                                                                     pose_6_x,pose_6_y,pose_6_confidence,
#                                                                     pose_7_x,pose_7_y,pose_7_confidence]).view(21,-1).ndim - 1, -1, -1))

#         if normalize == "batch_normalization":
#             batch_norm=nn.BatchNorm1d(21)
#             pose = batch_norm(pose.float())
#             pose = torch.nan_to_num(pose, nan=0, posinf=0, neginf=0)
#             pose = pose.clone().detach().requires_grad_(False)
#         else:
#             print(f"normalize error: {normalize}")
#             exit(1)

#         return pose

#     elif feature_name == "head":
#         head_x = torch.tensor(df["head_x"]).permute(*torch.arange(torch.tensor(df["head_x"]).ndim - 1, -1, -1)).unsqueeze(1)
#         head_y = torch.tensor(df["head_y"]).permute(*torch.arange(torch.tensor(df["head_y"]).ndim - 1, -1, -1)).unsqueeze(1)
#         head_z = torch.tensor(df["head_z"]).permute(*torch.arange(torch.tensor(df["head_z"]).ndim - 1, -1, -1)).unsqueeze(1)
#         head = torch.stack([head_x,head_y,head_z]).view(3,-1).permute(*torch.arange(torch.stack([head_x,head_y,head_z]).view(3,-1).ndim - 1, -1, -1))

#         if normalize == "batch_normalization":
#             batch_norm=nn.BatchNorm1d(3)
#             head = batch_norm(head.float())
#             head = torch.nan_to_num(head, nan=0, posinf=1, neginf=-1)
#             head = head.clone().detach().requires_grad_(False)
#         else:
#             print(f"normalize error: {normalize}")
#             exit(1)

#         return head

#     else:
#         print(f"ERROR: load_mutimodal_features()")
#         exit(1)


def write_json(data, filename):
    with open(filename, "w", encoding="utf-8") as jsonfile:
        json.dump(data, jsonfile, ensure_ascii=False)


def read_json(path, encoding="utf8"):
    with open(path, "r", encoding=encoding) as f:
        data = json.loads(f.read())
    return data


def write_txt(txt, name):
    """
    Argument:
        txt:    list of strings
        name:   filename
    """
    with open(name, "w") as f:
        f.write("\n".join(txt))


def read_txt(path, encoding="utf-8"):
    data = []
    with open(path, "r", encoding=encoding) as f:
        for line in f.readlines():
            data.append(line.strip())
    return data


def find_island_idx_len(x):
    """
    Finds patches of the same value.

    starts_idx, duration, values = find_island_idx_len(x)

    e.g:
        ends = starts_idx + duration

        s_n = starts_idx[values==n]
        ends_n = s_n + duration[values==n]  # find all patches with N value

    """
    assert x.ndim == 1
    n = len(x)
    y = x[1:] != x[:-1]  # pairwise unequal (string safe)
    i = torch.cat(
        (torch.where(y)[0], torch.tensor(n - 1, device=x.device).unsqueeze(0))
    ).long()
    it = torch.cat((torch.tensor(-1, device=x.device).unsqueeze(0), i))
    dur = it[1:] - it[:-1]
    idx = torch.cumsum(
        torch.cat((torch.tensor([0], device=x.device, dtype=torch.long), dur)), dim=0
    )[
        :-1
    ]  # positions
    return idx, dur, x[i]


def load_config(path=None, args=None, format="dict"):
    conf = OmegaConf.load(path)
    if args is not None:
        conf = OmegaConfArgs.update_conf_with_args(conf, args)

    if format == "dict":
        conf = OmegaConf.to_object(conf)
    return conf


class OmegaConfArgs:
    """
    This is annoying... And there is probably a SUPER easy way to do this... But...

    Desiderata:
        * Define the model completely by an OmegaConf (yaml file)
            - OmegaConf argument syntax  ( '+segments.c1=10' )
        * run `sweeps` with WandB
            - requires "normal" argparse arguments (i.e. '--batch_size' etc)

    This class is a helper to define
    - argparse from config (yaml)
    - update config (loaded yaml) with argparse arguments


    See ./config/sosi.yaml for reference yaml
    """

    @staticmethod
    def add_argparse_args(parser, conf, omit_fields=None):
        for field, settings in conf.items():
            if omit_fields is None:
                for setting, value in settings.items():
                    name = f"--{field}.{setting}"
                    parser.add_argument(name, default=None, type=type(value))
            else:
                if not any([field == f for f in omit_fields]):
                    for setting, value in settings.items():
                        name = f"--{field}.{setting}"
                        parser.add_argument(name, default=None, type=type(value))
        return parser

    @staticmethod
    def update_conf_with_args(conf, args, omit_fields=None):
        if not isinstance(args, dict):
            args = vars(args)

        for field, settings in conf.items():
            if omit_fields is None:
                for setting in settings:
                    argname = f"{field}.{setting}"
                    if argname in args and args[argname] is not None:
                        conf[field][setting] = args[argname]
            else:
                if not any([field == f for f in omit_fields]):
                    for setting in settings:
                        argname = f"{field}.{setting}"
                        if argname in args:
                            conf[field][setting] = args[argname]
        return conf


def delete_path(filepath):
    remove(filepath)


def sph2pipe_to_wav(sph_file):
    wav_file = sph_file.replace(".sph", ".wav")
    subprocess.check_call(["sph2pipe", sph_file, wav_file])
    return wav_file
