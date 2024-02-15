import torch
import os
import json
from os.path import splitext, join, dirname as dr
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from argparse import ArgumentParser
from turntaking.augmentations import torch_to_praat_sound
import librosa

from turntaking.dataload.utils import (
    load_waveform,
    get_audio_info,
    time_to_frames,
)
from turntaking.model import Model
from turntaking.vap_to_turntaking.utils import vad_list_to_onehot, get_activity_history
import numpy as np

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf 
from turntaking.utils import to_device, everything_deterministic, read_json

everything_deterministic()

AUDIO_MIX = "/ahc/work2/kazuyo-oni/SST_demo/mix.wav"
AUDIO_USER1 = "/ahc/work2/kazuyo-oni/SST_demo/trainer.wav"
AUDIO_USER2 = "/ahc/work2/kazuyo-oni/SST_demo/trainee.wav"
TEXT_JSON = "/ahc/work2/kazuyo-oni/SST_demo/text.json"
VA_LIST = "/ahc/work2/kazuyo-oni/SST_demo/voice_activity.json"
MODEL_DIR = "/ahc/work2/kazuyo-oni/turntaking/output/noxi_unimodal_single-speaker_in1/"

def load_sample(
    cfg_dict, audio_mix, audio_user1, audio_user2, vad_list, normalize=True, mono_channel=True
):
    vad_hop_time = 1.0 / cfg_dict["data"]["vad_hz"]
    vad_history_frames = (
        (torch.tensor([60, 30, 10, 5]) / vad_hop_time).long().tolist()
    )

    ret = {}
    ret["waveform"] = load_waveform(
        audio_mix,
        sample_rate=cfg_dict["data"]["sample_rate"],
        normalize=normalize,
        mono=mono_channel,
    )[0]
    ret["waveform_user1"] = load_waveform(
        audio_user1,
        sample_rate=cfg_dict["data"]["sample_rate"],
        normalize=normalize,
        mono=mono_channel,
    )[0]
    ret["waveform_user2"] = load_waveform(
        audio_user2,
        sample_rate=cfg_dict["data"]["sample_rate"],
        normalize=normalize,
        mono=mono_channel,
    )[0]
    duration = get_audio_info(audio_mix)["duration"]

    ##############################################
    # VAD-frame of relevant part
    ##############################################
    end_frame = time_to_frames(duration, vad_hop_time)
    all_vad_frames = vad_list_to_onehot(
        vad_list,
        hop_time=vad_hop_time,
        duration=duration,
        channel_last=True,
    )
    ret["vad"] = all_vad_frames[:end_frame].unsqueeze(0)

    ##############################################
    # History
    ##############################################
    vad_history, _ = get_activity_history(
        all_vad_frames,
        bin_end_frames=vad_history_frames,
        channel_last=True,
    )
    # vad history is always defined as speaker 0 activity
    ret["vad_history"] = vad_history[:end_frame][..., 0].unsqueeze(0)
    return ret


def plot_next_speaker(p_ns, ax, color=["b", "orange"], alpha=0.6, fontsize=12):
    x = torch.arange(len(p_ns))
    ax.fill_between(
        x,
        y1=0.5,
        y2=p_ns,
        where=p_ns > 0.5,
        alpha=alpha,
        color=color[0],
        label="A turn",
    )
    ax.fill_between(
        x,
        y1=p_ns,
        y2=0.5,
        where=p_ns < 0.5,
        alpha=alpha,
        color=color[1],
        label="B turn",
    )
    ax.set_xlim([0, len(p_ns)])
    ax.set_xticks([])
    ax.set_yticks([0.25, 0.75], ["Turn_Trainee", "Turn_Trainer"], fontsize=fontsize)
    ax.set_ylim([0, 1])
    ax.hlines(y=0.5, xmin=0, xmax=len(p_ns), linestyle="dashed", color="k")
    return ax


def plot_bc(p_bc, ax, color=["b", "orange"], alpha=0.6, fontsize=12):
    x = torch.arange(len(p_bc[0, :, 0]))
    ax.fill_between(
        x,
        y1=0,
        y2=p_bc[0, :, 0],
        alpha=alpha,
        color=color[0],
        label="Trainer BC",
    )
    ax.fill_between(
        x,
        y1=-p_bc[0, :, 1],
        y2=0,
        alpha=alpha,
        color=color[1],
        label="Trainee BC",
    )
    ax.set_xlim([0, len(p_bc[0, :, 0])])
    ax.set_xticks([])
    ax.set_yticks([-0.25, 0.25], ["BC_Trainee", "BC_Trainer"], fontsize=fontsize)
    ax.set_ylim([-0.5, 0.5])
    ax.hlines(y=0, xmin=0, xmax=len(p_bc[0, :, 0]), linestyle="dashed", color="k")
    ax.hlines(
        y=0.1,
        xmin=0,
        xmax=len(p_bc[0, :, 0]),
        linestyle="dashed",
        color="k",
        linewidth=1,
    )
    ax.hlines(
        y=-0.1,
        xmin=0,
        xmax=len(p_bc[0, :, 0]),
        linestyle="dashed",
        color="k",
        linewidth=1,
    )
    return ax

def plot_waveform(
    sample,
    va=None,
    words=None,
    sample_rate=16000,
    ax=None,
    vad_color="coral",
    fontsize=12,
    plot=False,
):
    snd = torch_to_praat_sound(sample["waveform"], sample_rate)
    xmin, xmax = snd.xs().min(), snd.xs().max()
    melspec = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=sample["waveform"].numpy(), sr=sample_rate, n_fft=800, hop_length=160
        ),
        ref=np.max,
    )[0]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(3, 1, figsize=(6, 5))

    # Waveform
    ax.plot(snd.xs(), snd.values.T, alpha=0.4)
    ax.set_xlim([xmin, xmax])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("waveform", fontsize=fontsize)

    # Voice Activity
    if va is not None:
        for value in va:
            ax.axvspan(value[0], value[1], alpha=0.4, color=vad_color)

    # Plot text on top of waveform
    if words is not None:
        y_min = -0.8
        y_max = 0.8
        diff = y_max - y_min
        steps = 4
        for ii, (start_time, end_time, word) in enumerate(words):
            yy = y_min + diff * (ii % steps) / steps
            mid = start_time + 0.5 * (end_time - start_time)

            ax.text(
                x=mid,
                y=yy,
                s=word,
                fontsize=5,
                horizontalalignment="center",
            )
            # ax.vlines(
            #     start_time,
            #     ymin=-1,
            #     ymax=1,
            #     linestyle="dashed",
            #     linewidth=1,
            #     color="k",
            #     alpha=0.8,
            # )
            # ax.vlines(
            #     end_time,
            #     ymin=-1,
            #     ymax=1,
            #     linestyle="dashed",
            #     linewidth=1,
            #     color="k",
            #     alpha=0.8,
            # )

    return fig, ax


def plot_origin(
    p_ns,
    p_bc,
    sample,
    sample_rate=16000,
    fontsize=12,
    plot=False,
):
    fig, ax = plt.subplots(4, 1, figsize=(8, 5))

    waveform = {"waveform": sample["waveform_trainer"]}
    _, ax[0] = plot_waveform(
        waveform,
        va=list(sample["vad"])[0],
        words=list(sample["words"])[0],
        sample_rate=sample_rate,
        ax=ax[0],
        vad_color="b",
        fontsize=fontsize,
    )
    ax[0].set_ylabel("Trainer")

    waveform = {"waveform": sample["waveform_trainee"]}
    _, ax[1] = plot_waveform(
        waveform,
        va=list(sample["vad"])[1],
        words=list(sample["words"])[1],
        sample_rate=sample_rate,
        ax=ax[1],
        vad_color="orange",
        fontsize=fontsize,
    )
    ax[1].set_ylabel("Trainee")

    # Next speaker probs
    ax[2] = plot_next_speaker(p_ns[0, :, 0], ax=ax[2], fontsize=fontsize)

    # BC probs
    ax[3] = plot_bc(p_bc, ax=ax[3], fontsize=fontsize)

    plt.subplots_adjust(
        left=0.2, bottom=0.05, right=0.9, top=0.95, wspace=None, hspace=0.09
    )
    if plot:
        plt.pause(0.1)
    return fig, ax


def words_to_vad(words):
    vad_list = []
    for i in words:
        tmp_vad = []
        for s, e, w in i:
            tmp_vad.append([s, e])
        vad_list.append(tmp_vad)
    return vad_list

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg_dict = dict(OmegaConf.to_object(cfg))

    vad_list = None
    text_list = None
    if TEXT_JSON is not None:
        text_list = read_json(TEXT_JSON)
        vad_list = words_to_vad(text_list)
    else:
        vad_list = read_json(VA_LIST)

    device = cfg_dict["train"]["device"]
    with open(join(MODEL_DIR, "log.json")) as f:
        cfg_dict = json.load(f)
        cfg_dict["train"]["device"] = device
    
    model_path = join(MODEL_DIR, "00", "model.pt")
    model = Model(cfg_dict).to(cfg_dict["train"]["device"])
    model.load_state_dict(
        torch.load(model_path, map_location=cfg_dict["train"]["device"])
    )

    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # get sample and process
    sample = load_sample(cfg_dict, AUDIO_MIX, AUDIO_USER1, AUDIO_USER2, vad_list)  # vad, va-history

    vad_frame_num = int(cfg_dict["data"]["vad_hz"] * cfg_dict["data"]["audio_duration"])
    audio_frame_num = int(cfg_dict["data"]["sample_rate"] * cfg_dict["data"]["audio_duration"])
    wav_len = sample["vad"].size(-2)

    padding = torch.zeros(1, vad_frame_num, 2)
    sample["vad"] = torch.cat((sample["vad"], padding), -2)
    padding = torch.zeros(1, vad_frame_num, 5)
    sample["vad_history"] = torch.cat((sample["vad_history"], padding), -2)
    padding = torch.zeros(1, audio_frame_num)
    sample["waveform"] = torch.cat((sample["waveform"], padding), -1)
    sample["waveform_user1"] = torch.cat((sample["waveform_user1"], padding), -1)
    sample["waveform_user2"] = torch.cat((sample["waveform_user2"], padding), -1)

    sample = to_device(sample, cfg_dict["train"]["device"])

    ratio = audio_frame_num / vad_frame_num

    probs_p_list = []
    probs_bc_p_list = []
    ret = {}
    probs = []
    for i in range(0, sample["vad"].shape[1] - vad_frame_num):
        ret["vad"] = sample["vad"][:, i:i + vad_frame_num, :]
        ret["vad_history"] = sample["vad_history"][:, i:i + vad_frame_num, :]

        waveform_start = int(i * ratio)
        waveform_end = int((i + vad_frame_num) * ratio)
        ret["waveform"] = sample["waveform"][:, waveform_start:waveform_end]
        ret["waveform_user1"] = sample["waveform_user1"][:, waveform_start:waveform_end]
        ret["waveform_user2"] = sample["waveform_user2"][:, waveform_start:waveform_end]
        
        out = model.output(ret)
        probs.append(out["logits_vp"])

    probs = torch.cat(probs).permute(1, 0, 2)
    turn_taking_probs = model.VAP(logits=probs, va=sample["vad"][:,vad_frame_num:,:].to("cpu"))


    # Plot
    inp = {
        "waveform": load_waveform(
            AUDIO_MIX,
            sample_rate=cfg_dict["data"]["sample_rate"],
            normalize=True,
            mono=True,
        )[0],
        "waveform_trainer": load_waveform(
            AUDIO_USER1,
            sample_rate=cfg_dict["data"]["sample_rate"],
            normalize=True,
            mono=True,
        )[0],
        "waveform_trainee": load_waveform(
            AUDIO_USER2,
            sample_rate=cfg_dict["data"]["sample_rate"],
            normalize=True,
            mono=True,
        )[0],
        "vad": vad_list,
        "words": text_list,
    }

    ### Figure ###
    fig, ax = plot_origin(
        turn_taking_probs["p"],
        turn_taking_probs["bc_prediction"],
        sample=inp,
        sample_rate=cfg_dict["data"]["sample_rate"],
    )
    output_file = join(MODEL_DIR, "viz.png")
    plt.savefig(output_file, dpi=300)
 


if __name__ == "__main__":
    main()

    ### Video ###
    # def anime(frames):
    #     ax[0].cla()
    #     ax[1].cla()
    #     ax[2].cla()
    #     ax[3].cla()
    #     ax[4].cla()
    #     _, _ = plot_origin(
    #         probs["p"],
    #         probs["bc_prediction"],
    #         sample=inp,
    #         ax=ax,
    #         sample_rate=model.sample_rate,
    #         frame_hz=model.frame_hz,
    #     )
    #     ax[0].vlines(
    #         frames / 10,
    #         ymin=-1,
    #         ymax=1,
    #         linewidth=2,
    #         color="r",
    #     )

    # anim = FuncAnimation(
    #     fig,
    #     anime,
    #     frames=tqdm(range(round(len(probs["p"][0, :, 0]) / 10))),
    #     interval=100,
    # )
    # mp4_filepath = os.path.splitext(args.output)[0] + ".mp4"
    # anim.save(mp4_filepath, writer="ffmpeg", dpi=300)

    ### Video ###
    # fig, ax = plt.subplots(5, 1, figsize=(5, 5))
    # fig, ax = plot_origin(
    #     probs["p"],
    #     probs["bc_prediction"],
    #     sample=inp,
    #     ax=ax,
    #     sample_rate=model.sample_rate,
    #     frame_hz=model.frame_hz,
    # )
    # # ax[0].set_xlim(0, 10)
    # ax[0].axvspan(10, 100, alpha=1, color="w", zorder=2)
    # plt.savefig(args.output, dpi=300)

    # def anime(frames):
    #     ax[0].cla()
    #     ax[1].cla()
    #     ax[2].cla()
    #     ax[3].cla()
    #     ax[4].cla()
    #     _, _ = plot_origin(
    #         probs["p"],
    #         probs["bc_prediction"],
    #         sample=inp,
    #         ax=ax,
    #         sample_rate=model.sample_rate,
    #         frame_hz=model.frame_hz,
    #     )
    #     ax[0].vlines(
    #         frames / 10,
    #         ymin=-1,
    #         ymax=1,
    #         linewidth=2,
    #         color="r",
    #     )

    # anim = FuncAnimation(
    #     fig,
    #     anime,
    #     frames=tqdm(range(round(len(probs["p"][0, :, 0]) / 10))),
    #     interval=100,
    # )
    # mp4_filepath = os.path.splitext(args.output)[0] + ".mp4"
    # anim.save(mp4_filepath, writer="ffmpeg", dpi=300)
