from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from collections import defaultdict
import hydra
from omegaconf import DictConfig, OmegaConf 
from tqdm import tqdm

from turntaking.vap_to_turntaking.backchannel import Backchannel
from turntaking.vap_to_turntaking.hold_shifts import HoldShift
from turntaking.vap_to_turntaking.events import TurnTakingEvents
from turntaking.vap_to_turntaking.utils import (find_island_idx_len,
                                                get_dialog_states,
                                                get_last_speaker,
                                                time_to_frames)
from turntaking.dataload import DialogAudioDM
from turntaking.dataload.utils import (
    load_waveform,
    get_audio_info,
    time_to_frames,
    load_multimodal_features,
)
from turntaking.vap_to_turntaking.utils import vad_list_to_onehot, get_activity_history

from decimal import Decimal
from os.path import join, exists
import re
import wave

from turntaking.dataload.utils import read_txt

FRAME_HZ = 100
HOT_TIME = 1/FRAME_HZ

HIST = False
DATA = 0

def count_occurances(x):
    n = 0
    for b in range(x.shape[0]):
        for sp in [0, 1]:
            _, _, v = find_island_idx_len(x[b, :, sp])
            n += (v == 1).sum().item()
    return n

def find_continuous_ones(tensor):
    def calculate(x):
        differences = {}
        for key in x:
            differences[key] = [round((x[key][i][1] - x[key][i][0] + 1) * HOT_TIME, 3) for i in range(len(x[key]))]
            # for i in range(len(x[key])):
            #     if round((x[key][i][1] - x[key][i][0] + 1) * HOT_TIME, 3) > 10:
            #         print(f"{round((x[key][i][0] + 1) * HOT_TIME, 3)} to {round((x[key][i][1])* HOT_TIME, 3)} ({round((x[key][i][1] - x[key][i][0] + 1) * HOT_TIME, 3)})")
        return differences
    
    continuous_segments = {0: [], 1: []}
    for dim in range(2):
        start = None
        for i, value in enumerate(tensor[0, :, dim]):
            if value == 1 and start is None:
                start = i
            elif value == 0 and start is not None:
                continuous_segments[dim].append((start, i - 1))
                start = None
        if start is not None:
            continuous_segments[dim].append((start, len(tensor[0, :, dim]) - 1))

    return calculate(continuous_segments)

def calculate_statistics(array):
    return {
        'num': len(array),
        'mean': round(np.mean(array), 3),
        'median': round(np.median(array), 3),
        'max': round(np.max(array), 3),
        'min': round(np.min(array), 3),
        'variance': round(np.var(array), 3)
    }

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg_dict = dict(OmegaConf.to_object(cfg))
    cfg_dict["data"]["vad_hz"] = FRAME_HZ
    cfg_dict["data"]["oversampling"] = False
    cfg_dict["data"]["flip_channels"] = False
    cfg_dict["data"]["vad_history"] = False

    if DATA is not None:
        cfg_dict["data"]["train_files"] = f"turntaking/dataload/dataset/{cfg_dict['data']['datasets']}/files/train_{DATA}.txt"
        cfg_dict["data"]["val_files"] = f"turntaking/dataload/dataset/{cfg_dict['data']['datasets']}/files/val_{DATA}.txt"
        cfg_dict["data"]["test_files"] = f"turntaking/dataload/dataset/{cfg_dict['data']['datasets']}/files/test_{DATA}.txt"  

    cfg_dict["events"]["SH"]["metric_pad"] = 0.0
    cfg_dict["events"]["SH"]["metric_dur"] = 0.0

    dm = DialogAudioDM(**cfg_dict["data"])
    dm.setup(None)

    eventer = TurnTakingEvents(
        hs_kwargs=cfg_dict["events"]["SH"],
        bc_kwargs=cfg_dict["events"]["BC"],
        metric_kwargs=cfg_dict["events"]["metric"],
        frame_hz=FRAME_HZ,
    )

    results = []
    all_shift0 = []
    all_shift1 = []
    all_hold0 = []
    all_hold1 = []
    all_bc0 = []
    all_bc1 = []

    # vad = dm.test_dset.data["vad"]
    # sessions = dm.test_dset.data["session"]

    vad = dm.train_dset.data["vad"] + dm.val_dset.data["vad"] + dm.test_dset.data["vad"]
    sessions = dm.train_dset.data["session"] + dm.val_dset.data["session"] + dm.test_dset.data["session"]

    all_ov_count = {0: 0, 1: 0}

    for d, s in tqdm(zip(vad, sessions), desc="Processing"):
        # print(f"{s}: {d.shape}")
        e = eventer(d, max_frame=None)
        # print(s)
        # print("shift")
        shift = find_continuous_ones(eventer.tt["shift_dur"])
        # print("hold")
        hold = find_continuous_ones(eventer.tt["hold_dur"])
        ov = find_continuous_ones(eventer.tt["ov_dur"])
        bc = find_continuous_ones(eventer.bcs["bc_dur"])


        for key, values in ov.items():
            neg_values = [-x for x in values]
            shift.setdefault(key, []).extend(neg_values)

        all_shift0 += shift[0]
        all_shift1 += shift[1]
        all_hold0 += hold[0]
        all_hold1 += hold[1]
        all_bc0 += bc[0]
        all_bc1 += bc[1]

        all_ov_count[0] += len(ov[0])
        all_ov_count[1] += len(ov[1])

        results.append([s, shift[0], shift[1], shift[0] + shift[1], hold[0], hold[1], hold[0] + hold[1], bc[0], bc[1], bc[0] + bc[1]])

    shift_ov_ratio_0 = all_ov_count[0] / len(all_shift0)
    shift_ov_ratio_1 = all_ov_count[1] / len(all_shift1)
    print(f"Shift to OV Ratio for 0: {shift_ov_ratio_0}")
    print(f"Shift to OV Ratio for 1: {shift_ov_ratio_1}")
    exit(1)

    df = pd.DataFrame(results, columns=['session', 'shift0', 'shift1', 'shift', 'hold0', 'hold1', 'hold', 'bc0', 'bc1', "bc"])

    all_session_data = ['all', all_shift0, all_shift1, all_shift0 + all_shift1, 
                        all_hold0, all_hold1, all_hold0 + all_hold1, 
                        all_bc0, all_bc1, all_bc0 + all_bc1]
    df.loc[len(df)] = all_session_data

    all_row = df[df['session'] == 'all']

    statistics = {}
    categories = ['shift0', 'shift1', 'shift', 'hold0', 'hold1', 'hold',  'bc0', 'bc1', 'bc']
    for category in categories:
        statistics[category] = calculate_statistics(all_row[category].iloc[0])

    stats_df = pd.DataFrame(statistics).transpose()
    print(stats_df)

    if HIST:
        for index, row in df.iterrows():
            session = row['session']
            for column in df.columns[1:]:
                if DATA is not None:
                    plot_histogram(row[column], f'{session}{DATA}', column, f"/ahc/work2/kazuyo-oni/turntaking/output/{cfg_dict['data']['datasets']}")
                else:
                    plot_histogram(row[column], f'{session}', column, f"/ahc/work2/kazuyo-oni/turntaking/output/{cfg_dict['data']['datasets']}")
    else:
        for index, row in df.iterrows():
            session = row['session']
            if session == 'all':
                for column in df.columns[1:]:
                    if DATA is not None:
                        plot_histogram(row[column], f'{session}{DATA}', column, f"/ahc/work2/kazuyo-oni/turntaking/output/{cfg_dict['data']['datasets']}")
                    else:
                        plot_histogram(row[column], f'{session}', column, f"/ahc/work2/kazuyo-oni/turntaking/output/{cfg_dict['data']['datasets']}") 

def plot_histogram(data, session, column, output_dir):
    if not data:
        return

    plt.figure()

    if column in ["shift", "shift0", "shift1"]:
        plt.hist(data, bins=[i/20 for i in range(-40, 61)], edgecolor='black', range=(-2, 3), color='#035894')
    elif column in ["hold", "hold0", "hold1"]:
        plt.hist(data, bins=[i/20 for i in range(0, 61)], edgecolor='black', range=(0, 3), color='#035894')
    elif column in ["bc", "bc0", "bc1"]:
        plt.hist(data, bins=[i/20 for i in range(5, 21)], edgecolor='black', range=(0.2, 1), color='#035894')
    else:
        plt.hist(data, bins=[i/20 for i in range(int(min(data)*20), int(max(data)*20)+1)], edgecolor='black', color='#035894')

    # plt.title(f'{session} - {column}')

    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    filename = f'{output_dir}/{session}_{column}.pdf'
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    main()