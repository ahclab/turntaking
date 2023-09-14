import torch
from torch.utils.data import Dataset
from turntaking.dataload.utils import (
    load_waveform,
    get_audio_info,
    time_to_frames,
    find_island_idx_len,
    load_multimodal_features,
)

from torch.utils.data import DataLoader


from turntaking.vap_to_turntaking.utils import vad_list_to_onehot, get_activity_history
from turntaking.vap_to_turntaking import VAPLabel, ActivityEmb
from decimal import Decimal, ROUND_HALF_UP
from concurrent.futures import ProcessPoolExecutor

import torch.nn.functional as F
import concurrent.futures
from tqdm import tqdm
import time
import math

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

class DialogAudioDataset(Dataset):
    def __init__(
        self,
        dataset,
        data=None,
        feature_extractor=None,
        type="sliding",
        # AUDIO #################################
        sample_rate:int =16000,
        audio_mono=True,
        audio_duration=10,
        audio_normalize=True,
        expert_and_novice=False,
        # VAD #################################
        vad=True,
        vad_hz:int =100,
        vad_horizon=2,
        vad_history=False,
        vad_history_times=[60, 30, 10, 5],
        # Multimodal Features #################################
        multimodal=False,
        normalize="batch_normalization",
        # Sliding #################################
        audio_overlap=2,  # Sliding Window
        # IPU #################################
        ipu_pause_time=0.1,
        ipu_min_time=0.4,
        audio_context_time=5,
        # DSET #################################
        flip_channels=True,
        flip_probability=0.5,
        transforms=None,
        # LABEL #################################
        label_type="discrete",
        # label_type="independent",
        bin_times=[0.20, 0.40, 0.60, 0.80],
        # bin_times=[.5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5,  .5],
        pre_frames=2,
        threshold_ratio=0.5,
        undersampling=True,
    ):
        super().__init__()
        self.dataset = dataset  # Hugginface datasets
        self.feature_extractor = feature_extractor
        self.transforms = transforms
        self.frame_mode = False

        # Audio (waveforms)
        self.sample_rate = sample_rate
        self.audio_mono = audio_mono
        self.audio_duration = audio_duration
        self.audio_overlap_default = audio_overlap
        self.audio_overlap = self.audio_overlap_default 
        self.audio_step_time = round(audio_duration - audio_overlap, 3)
        self.audio_normalize = audio_normalize
        self.audio_normalize_threshold = 0.05
        self.expert_and_novice = expert_and_novice

        # VAD parameters
        self.vad = vad  # use vad or not
        self.vad_hz = vad_hz
        self.vad_hop_time = 1 / vad_hz

        self.audio_overlap_frame = audio_duration - self.vad_hop_time

        # Multimodal parameters
        if self.vad_hz == 25:
            self.multimodal = multimodal  # use multimodal features or not
        else:
            self.multimodal = False  # use multimodal features or not
        self.normalize = normalize

        # Vad prediction labels
        self.horizon_time = vad_horizon
        self.vad_horizon = time_to_frames(vad_horizon, hop_time=self.vad_hop_time)

        # Vad history
        self.vad_history = vad_history
        self.vad_history_times = vad_history_times
        self.vad_history_frames = (
            (torch.tensor(vad_history_times) / self.vad_hop_time).long().tolist()
        )

        # IPU
        self.ipu_pause_time = ipu_pause_time
        self.ipu_min_time = ipu_min_time
        self.audio_context_time = audio_context_time

        # Dset
        self.flip_channels = flip_channels
        self.flip_probability = flip_probability

        # Lable
        self.bin_times = bin_times
        self.threshold_ratio = threshold_ratio
        self.pre_frames = pre_frames
        self.type = label_type
        self.undersampling = undersampling
        self.emb = ActivityEmb(self.bin_times, self.vad_hz)
        self.vap_label = VAPLabel(self.bin_times, self.vad_hz, self.threshold_ratio)

        if data is not None:
            self.data = data
        else:
            self._get_all()

        self.map_to_dset_idx, self.map_to_vad_idx, self.map_to_audio_idx = self._get_sliding_window_indices()

    def change_frame_mode(self, mode=False):
        self.frame_mode = mode
        self.audio_overlap = self.audio_overlap_frame if self.frame_mode else self.audio_overlap_default
        self.audio_step_time = round(self.audio_duration - self.audio_overlap, 3)

        self.map_to_dset_idx, self.map_to_vad_idx, self.map_to_audio_idx = self._get_sliding_window_indices(mode)
    
    def _undersampling(self, map_to_dset_idx, map_to_vad_idx, map_to_audio_idx):
        from collections import Counter        
        new_map_to_dset_idx = []
        new_map_to_vad_idx = []
        new_map_to_audio_idx = []
        
        for dset_idx, vad_idx, audio_idx in zip(map_to_dset_idx, map_to_vad_idx, map_to_audio_idx):
            label = self.data["label"][dset_idx][0, int(vad_idx + self.vad_hz*self.audio_duration)-1].item()
            if not ((label == 15) and (torch.rand(1) > 0.2)):
                new_map_to_dset_idx.append(dset_idx)
                new_map_to_vad_idx.append(vad_idx)
                new_map_to_audio_idx.append(audio_idx)

        return new_map_to_dset_idx, new_map_to_vad_idx, new_map_to_audio_idx

    def _get_sliding_window_indices(self, frame_mode=False):
        map_to_dset_idx = []
        map_to_vad_idx = []
        map_to_audio_idx = []

        vad_skip_frame = 1 if self.frame_mode else int(self.audio_step_time * self.vad_hz)
        audio_skip_frame = int(self.audio_step_time * self.sample_rate) if self.frame_mode else int(vad_skip_frame / self.vad_hz * self.sample_rate)

        for i in range(len(self.data["vad"])):
            vad_max_idx = self.data["vad"][i].size(1) - int(self.audio_duration * self.vad_hz)
            audio_max_idx = self.data["waveform"][i].size(1) - int(self.audio_duration * self.sample_rate)

            map_to_start_vad = list(range(0, vad_max_idx + 1, vad_skip_frame))
            map_to_start_audio = list(range(0, audio_max_idx + 1, audio_skip_frame))
            map_to_vad_idx.extend(map_to_start_vad)
            map_to_audio_idx.extend(map_to_start_audio)
            map_to_dset_idx.extend([i] * len(map_to_start_vad))
        
        if self.undersampling and not frame_mode:
            map_to_dset_idx, map_to_vad_idx, map_to_audio_idx = self._undersampling(map_to_dset_idx, map_to_vad_idx, map_to_audio_idx)

        return map_to_dset_idx, map_to_vad_idx, map_to_audio_idx

    def __repr__(self):
        s = "DialogSlidingWindow"
        s += f"\n\tsample_rate: {self.sample_rate}"
        s += f"\n\taudio_mono: {self.audio_mono}"
        s += f"\n\taudio_duration: {self.audio_duration}"
        s += f"\n\taudio_overlap: {self.audio_overlap}"
        s += f"\n\taudio_step_time: {self.audio_step_time}"
        s += f"\n\taudio_normalize: {self.audio_normalize}"
        s += f"\n\taudio_normalize_threshold: {self.audio_normalize_threshold}"

        # VAD parameters
        s += f"\n\tvad_hz: {self.vad_hz}"
        s += f"\n\tvad_hop_time: {self.vad_hop_time}"

        # Vad prediction labels
        s += f"\n\tvad_horizon: {self.vad_horizon}"

        # Vad history
        s += f"\n\tvad_history: {self.vad_history}"
        s += f"\n\tvad_history_times: {self.vad_history_times}"
        s += f"\n\tvad_history_frames: {self.vad_history_frames}"

        # Dset
        s += f"\n\tflip_channels: {self.flip_channels}"
        s += f"\n\tflip_probability: {self.flip_probability}"
        s += "\n" + "-" * 40
        return s

    def __len__(self):
        return len(self.map_to_dset_idx)
    
    def _load_waveform(self, b):
        waveform, _ = load_waveform(
            b["audio_path"],
            sample_rate=self.sample_rate,
            normalize=self.audio_normalize,
            mono=self.audio_mono,
        )
        if b["dataset_name"] == "noxi":
            waveform_expert, _ = load_waveform(
                b["expert_audio_path"],
                sample_rate=self.sample_rate,
                normalize=self.audio_normalize,
                mono=self.audio_mono,
            )
            waveform_novice, _ = load_waveform(
                b["novice_audio_path"],
                sample_rate=self.sample_rate,
                normalize=self.audio_normalize,
                mono=self.audio_mono,
            )
        else:
            waveform_expert, waveform_novice = None, None
        return waveform, waveform_expert, waveform_novice

    def _load_vad(self, b):
        vad = vad_list_to_onehot(
                b["vad"],
                hop_time=self.vad_hop_time,
                duration=get_audio_info(b["audio_path"])["duration"],
                channel_last=True,
            )
        vad_history, _ = get_activity_history(
                    vad,
                    bin_end_frames=self.vad_history_frames,
                    channel_last=True,
                )
        
        return vad.unsqueeze(0), vad_history[..., 0].unsqueeze(0)

    def _load_multimodal(self, b):
        if b["dataset_name"] == "noxi":
            gaze_expert, au_expert, pose_expert, head_expert = load_multimodal_features(
                b["multimodal_expert_path"], self.normalize
                )
            gaze_novice, au_novice, pose_novice, head_novice = load_multimodal_features(
                b["multimodal_novice_path"], self.normalize
                )
        else:
            gaze_expert, au_expert, pose_expert, head_expert, gaze_novice, au_novice, pose_novice, head_novice = None, None, None, None, None, None, None, None
        
        
        return gaze_expert, au_expert, pose_expert, head_expert, gaze_novice, au_novice, pose_novice, head_novice
    
    def _extract_label(self, va: torch.Tensor) -> torch.Tensor:
        if self.type == "comparative":
            return self.vap_label(va, type="comparative")

        vap_bins = self.vap_label(va, type="binary")

        if self.type == "independent":
            return vap_bins
        else:
            return self.emb(vap_bins)  # discrete

    def _adjust_waveform_size(self, waveform, all_vad_frames_num):
        audio_frame_num = int(self.sample_rate * all_vad_frames_num / self.vad_hz) - waveform.size(-1)
        if audio_frame_num < 0:
            return waveform[:, : int(self.sample_rate * all_vad_frames_num / self.vad_hz)]
        elif audio_frame_num > 0:
            padding = torch.zeros(1, audio_frame_num)
            return torch.cat((waveform, padding), -1)
        else:
            return waveform

    def _adjust_multimodal_size(self, multimodal, all_vad_frames_num):
        padding_shape = multimodal.size(-1)
        multimodal_frame_num = all_vad_frames_num - multimodal.size(-2)
        if multimodal_frame_num < 0:
            return multimodal[:, : all_vad_frames_num, :]
        elif multimodal_frame_num > 0:
            padding = torch.zeros(1, multimodal_frame_num, padding_shape)
            return torch.cat((multimodal, padding), -2)
        else:
            return multimodal

    def _process_data(self, b):
        data_dict = {}

        # Load the audio file and corresponding multimodal data
        waveform, waveform_expert, waveform_novice = self._load_waveform(b)
        vad, vad_history = self._load_vad(b)
        lookahead = torch.zeros((1, self.vad_horizon, 2))
        label = self._extract_label(torch.cat((vad, lookahead), -2))
        gaze_expert, au_expert, pose_expert, head_expert, gaze_novice, au_novice, pose_novice, head_novice = self._load_multimodal(b)

        # Shape the audio waveform and multimodal data
        all_vad_frames_num = vad.size(1)
        waveform = self._adjust_waveform_size(waveform, all_vad_frames_num)

        is_noxi = b["dataset_name"] == "noxi"
        if is_noxi:
            waveform_expert = self._adjust_waveform_size(waveform_expert, all_vad_frames_num)
            waveform_novice = self._adjust_waveform_size(waveform_novice, all_vad_frames_num)

            gaze_expert = self._adjust_multimodal_size(gaze_expert, all_vad_frames_num)
            au_expert = self._adjust_multimodal_size(au_expert, all_vad_frames_num)
            pose_expert = self._adjust_multimodal_size(pose_expert, all_vad_frames_num)
            head_expert = self._adjust_multimodal_size(head_expert, all_vad_frames_num)
            gaze_novice = self._adjust_multimodal_size(gaze_novice, all_vad_frames_num)
            au_novice = self._adjust_multimodal_size(au_novice, all_vad_frames_num)
            pose_novice = self._adjust_multimodal_size(pose_novice, all_vad_frames_num)
            head_novice = self._adjust_multimodal_size(head_novice, all_vad_frames_num)

        # Append the data to the dictionary
        data_dict["dataset_name"] = b["dataset_name"]
        data_dict["session"] = b["session"]
        data_dict["waveform"] = waveform
        data_dict["vad"] = vad
        data_dict["label"] = label
        data_dict["vad_history"] = vad_history

        if is_noxi:
            data_dict["waveform_expert"] = waveform_expert
            data_dict["waveform_novice"] = waveform_novice
            data_dict["gaze_expert"] = gaze_expert
            data_dict["au_expert"] = au_expert
            data_dict["pose_expert"] = pose_expert
            data_dict["head_expert"] = head_expert
            data_dict["gaze_novice"] = gaze_novice
            data_dict["au_novice"] = au_novice
            data_dict["pose_novice"] = pose_novice
            data_dict["head_novice"] = head_novice
    

        return data_dict

    def _get_all(self):
        # Initialize the dictionary to store data
        self.data = {
            "dataset_name": [],
            "session": [],
            "waveform": [],
            "waveform_expert": [],
            "waveform_novice": [],
            "vad": [],
            "label": [],
            "vad_history": [],
            "gaze_expert": [],
            "au_expert": [],
            "pose_expert": [],
            "head_expert": [],
            "gaze_novice": [],
            "au_novice": [],
            "pose_novice": [],
            "head_novice": [],
        }

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(self._process_data, [self.dataset[i] for i in range(len(self.dataset["audio_path"]))]), total=len(self.dataset["audio_path"])))

        for result in results:
            for key, value in result.items():
                self.data[key].append(value)
        
        empty_keys = [key for key, value in self.data.items() if value == [] or (isinstance(value, list) and all(v is None for v in value))]
        for key in empty_keys:
            del self.data[key]


    # def _get_all(self):
    #     # Initialize the dictionary to store data
    #     self.data = {
    #         "dataset_name": [],
    #         "session": [],
    #         "waveform": [],
    #         "waveform_expert": [],
    #         "waveform_novice": [],
    #         "vad": [],
    #         "label": [],
    #         "vad_history": [],
    #         "gaze_expert": [],
    #         "au_expert": [],
    #         "pose_expert": [],
    #         "head_expert": [],
    #         "gaze_novice": [],
    #         "au_novice": [],
    #         "pose_novice": [],
    #         "head_novice": [],
    #     }

    #     def adjust_waveform_size(waveform):
    #         audio_frame_num = int(self.sample_rate * all_vad_frames_num / self.vad_hz) - waveform.size(-1)
    #         if audio_frame_num < 0:
    #             return waveform[:, : int(self.sample_rate * all_vad_frames_num / self.vad_hz)]
    #         elif audio_frame_num > 0:
    #             padding = torch.zeros(1, audio_frame_num)
    #             return torch.cat((waveform, padding), -1)
    #         else:
    #             return waveform

    #     def adjust_multimodal_size(multimodal):
    #         padding_shape = multimodal.size(-1)
    #         multimodal_frame_num = all_vad_frames_num - multimodal.size(-2)
    #         if multimodal_frame_num < 0:
    #             return multimodal[:, : all_vad_frames_num, :]
    #         elif multimodal_frame_num > 0:
    #             padding = torch.zeros(1, multimodal_frame_num, padding_shape)
    #             return torch.cat((multimodal, padding), -2)
    #         else:
    #             return multimodal

    #     for i in tqdm(range(len(self.dataset["audio_path"]))):
    #         # Load the audio file and corresponding multimodal data
    #         b = self.dataset[i]
    #         waveform, waveform_expert, waveform_novice = self._load_waveform(b)
    #         vad, vad_history = self._load_vad(b)
    #         lookahead = torch.zeros((1, self.vad_horizon, 2))
    #         label = self._extract_label(torch.cat((vad, lookahead), -2))
    #         gaze_expert, au_expert, pose_expert, head_expert, gaze_novice, au_novice, pose_novice, head_novice = self._load_multimodal(b)

    #         # Shape the audio waveform and multimodal data
    #         all_vad_frames_num = vad.size(1)

    #         waveform = adjust_waveform_size(waveform)

    #         is_noxi = b["dataset_name"] == "noxi"
    #         if is_noxi:
    #             waveform_expert = adjust_waveform_size(waveform_expert)
    #             waveform_novice = adjust_waveform_size(waveform_novice)

    #             gaze_expert = adjust_multimodal_size(gaze_expert)
    #             au_expert = adjust_multimodal_size(au_expert)
    #             pose_expert = adjust_multimodal_size(pose_expert)
    #             head_expert = adjust_multimodal_size(head_expert)
    #             gaze_novice = adjust_multimodal_size(gaze_novice)
    #             au_novice = adjust_multimodal_size(au_novice)
    #             pose_novice = adjust_multimodal_size(pose_novice)
    #             head_novice = adjust_multimodal_size(head_novice)

    #         # Append the data to the dictionary
    #         self.data["dataset_name"].append(b["dataset_name"])
    #         self.data["session"].append(b["session"])
    #         self.data["waveform"].append(waveform)
    #         self.data["vad"].append(vad)
    #         self.data["label"].append(label)
    #         self.data["vad_history"].append(vad_history)

    #         if is_noxi:
    #             self.data["waveform_expert"].append(waveform_expert)
    #             self.data["waveform_novice"].append(waveform_novice)
    #             self.data["gaze_expert"].append(gaze_expert)
    #             self.data["au_expert"].append(au_expert)
    #             self.data["pose_expert"].append(pose_expert)
    #             self.data["head_expert"].append(head_expert)
    #             self.data["gaze_novice"].append(gaze_novice)
    #             self.data["au_novice"].append(au_novice)
    #             self.data["pose_novice"].append(pose_novice)
    #             self.data["head_novice"].append(head_novice)
            
    #     empty_keys = [key for key, value in self.data.items() if value == [] or (isinstance(value, list) and all(v is None for v in value))]
    #     for key in empty_keys:
    #         del self.data[key]


    def get_full_sample(self):
        ret = {}

        def process_tensor(k, tensor):
            if k in ["waveform", "waveform_expert", "waveform_novice"]:
                return tensor[:, int((self.audio_duration - self.audio_step_time) * self.sample_rate):]
            elif k in ["label"]:
                return tensor[:, int(self.audio_duration * self.vad_hz - 1):]
            else:
                return tensor[:, int(self.audio_duration * self.vad_hz - 1):, :]

        for k, v in self.data.items():
            for idx, item in enumerate(v):
                if isinstance(item, torch.Tensor):
                    processed_item = process_tensor(k, item)
                    if idx == 0:
                        ret[k] = processed_item
                    else:
                        ret[k] = torch.cat((ret[k], processed_item), dim=1)
                else:
                    if idx == 0:
                        ret[k] = [item]
                    else:
                        ret[k] += [item]

        if not self.vad_history:
            del ret["vad_history"]

        return ret

    def get_sample(
        self,
        idx: int,
        dset_idx: int,
        start_vad_idx: int,
        end_vad_idx: int,
        start_audio_idx: int,
        end_audio_idx: int,
    ):

        all_vad_frames = self.data["vad"][dset_idx]

        if self.type == "independent":
            target_label_value = self.data["label"][dset_idx][0, end_vad_idx-1]
            label_tensor = torch.tensor(target_label_value).unsqueeze(0).unsqueeze(0)
        else:
            target_label_value = self.data["label"][dset_idx][0, end_vad_idx-1].item()
            label_tensor = torch.tensor([target_label_value]).unsqueeze(1)

        ret = {
            "waveform": self.data["waveform"][dset_idx][:, start_audio_idx:end_audio_idx],
            "dataset_name": [self.data["dataset_name"][dset_idx]],
            "session": [self.data["session"][dset_idx]],
            "label": label_tensor
        }

        if self.expert_and_novice:
            for key in ["waveform_expert", "waveform_novice"]:
                if key in self.data:
                    ret[key] = self.data[key][dset_idx][:, start_audio_idx:end_audio_idx]

        if self.vad:
            # if end_vad_idx + self.vad_horizon > all_vad_frames.size(1):
            #     lookahead = torch.zeros((1, self.vad_horizon, 2))
            #     all_vad_frames = torch.cat((all_vad_frames, lookahead), -2)

            # ret["vad"] = all_vad_frames[:, start_vad_idx:end_vad_idx + self.vad_horizon, :]
            ret["vad"] = all_vad_frames[:, start_vad_idx:end_vad_idx, :]

            if self.vad_history:
                ret["vad_history"] = self.data["vad_history"][dset_idx][:, start_vad_idx:end_vad_idx, :]

        if self.multimodal:
            for key in ["gaze_expert", "au_expert", "pose_expert", "head_expert", "gaze_novice", "au_novice", "pose_novice", "head_novice"]:
                if key in self.data:
                    ret[key] = self.data[key][dset_idx][:, start_vad_idx:end_vad_idx, :]
        
        if self.flip_channels and idx%2:
            ret["vad"] = torch.stack(
                (ret["vad"][:, :, 1], ret["vad"][:, :, 0]), dim=-1
            )
            if self.vad and self.vad_history:
                ret["vad_history"] = 1 - ret["vad_history"]
            if self.multimodal:
                for key1, key2 in zip(["gaze_expert", "au_expert", "pose_expert", "head_expert"], ["gaze_novice", "au_novice", "pose_novice", "head_novice"]):
                    ret[key1], ret[key2] = ret[key2], ret[key1]

            if self.expert_and_novice:
                ret["waveform_expert"], ret["waveform_novice"] = ret["waveform_novice"], ret["waveform_expert"]
            
            if self.type == "discrete":
                binary_str = '{:08b}'.format(ret["label"].item())
                swapped_binary_str = binary_str[4:] + binary_str[:4]
                decimal = int(swapped_binary_str, 2)
                ret["label"] = torch.tensor([[decimal]])
            elif self.type == "comparative":
                ret["label"] = 1 - ret["label"]
            else:
                ret["label"] = torch.stack(
                                (ret["label"][:, :, 1, :], ret["label"][:, :, 0, :]), dim=2
                            )
        return ret

    def __getitem__(self, idx):
        dset_idx = self.map_to_dset_idx[idx]
        start_vad_idx = self.map_to_vad_idx[idx]
        end_vad_idx = int(start_vad_idx + self.vad_hz*self.audio_duration)
        start_audio_idx = self.map_to_audio_idx[idx]
        end_audio_idx = int(start_audio_idx + self.sample_rate*self.audio_duration)

        d = self.get_sample(
            idx,
            dset_idx, 
            start_vad_idx, 
            end_vad_idx,
            start_audio_idx,
            end_audio_idx
            )

        if self.transforms is not None:
            n_frames = d["vad_history"].shape[1]
            vad = d["vad"][:, :n_frames]
            d["waveform"] = self.transforms(d["waveform"], vad=vad)
        return d

def events_plot(
    batch,
    key = None,
    value = None,
    sample_rate = 16000
    ):
    import matplotlib.pyplot as plt
    from conv_ssl.augmentations import torch_to_praat_sound
    import numpy as np

    vad_expert = batch["vad"][0][:,:,0].squeeze()
    vad_novice = batch["vad"][0][:,:,1].squeeze()

    indices_expert = np.where(vad_expert == 1)
    indices_novice = np.where(vad_novice == 1)

    waveform_expert = torch_to_praat_sound(batch["waveform_expert"][0].detach().numpy().copy(), sample_rate) + 1
    waveform_novice = torch_to_praat_sound(batch["waveform_novice"][0].detach().numpy().copy(), sample_rate) - 1

    fig = plt.figure(figsize=(100, 2), dpi=300)
    ax = fig.add_subplot(111)
    

    x = np.arange(len(vad_expert))
    snd_x = np.linspace(0,len(vad_expert),len(waveform_expert))

    ax.set_xlim([0, len(vad_expert)])
    ax.set_xticks(range(0, len(vad_expert)+1, 250), fontsize=3)

    ax.plot(snd_x, waveform_expert.values.T, alpha=0.4, linewidth=0.01)
    ax.plot(snd_x, waveform_novice.values.T, alpha=0.4, linewidth=0.01)

    # ax.plot(x,vad_expert, linewidth=0.01)
    # ax.plot(x,vad_novice, linewidth=0.01)

    for index in indices_expert[0]:
        plt.hlines(y=1, xmin=index-0.5, xmax=index+0.5, linewidth=1)
    for index in indices_novice[0]:
        plt.hlines(y=-1, xmin=index-0.5, xmax=index+0.5, linewidth=1)
    
    if value is not None:
        events_expert = value[:,:,0].squeeze()
        events_novice = value[:,:,1].squeeze()

        indices_events_expert = np.where(events_expert == 1)
        indices_events_novice = np.where(events_novice == 1)

        for index in indices_events_expert[0]:
            plt.vlines(x=index, ymin=-2, ymax=2, linewidth=1, color="k")
        for index in indices_events_novice[0]:
            plt.vlines(x=index, ymin=-2, ymax=2, linewidth=1, color="k")


    if key is not None:
        fig.savefig(f"output/img/{batch['session'][0]}_{key}.pdf", format="pdf")
    else:
        fig.savefig(f"output/img/{batch['session'][0]}.pdf", format="pdf")
    plt.cla()

    


if __name__ == "__main__":
    from turntaking.dataload.dialog_audio_dm import get_dialog_audio_datasets

    # # Switchboard Debug
    # # default
    # print(f"###SWITCHBOARD###")
    # dset_hf = get_dialog_audio_datasets(datasets=["switchboard"], split="val")
    # dset = DialogAudioDataset(
    #     dataset=dset_hf,
    #     type="sliding",
    #     vad_history=True,
    #     expert_and_novice=False,
    #     multimodal=False,
    # )
    # print(dset)
    # print(f"Datasets Size: {len(dset)}")

    # for i in range(len(dset)):
    #     batch = dset[i]
    #     for k, v in batch.items():
    #         if isinstance(v, torch.Tensor):
    #             print(f"{k}: {tuple(v.shape)}")
    #         else:
    #             print(f"{k}: {v}")
    
    # # optional
    # dset = DialogAudioDataset(
    #     dataset=dset_hf,
    #     type="sliding",
    #     vad_history=False,
    #     audio_overlap=0.5,
    #     audio_duration=1.0,
    #     vad_hz=25,
    #     expert_and_novice=False,
    #     multimodal=False,
    # )
    # print(dset)
    # print(f"Datasets Size: {len(dset)}")

    # batch = dset[0]
    # for k, v in batch.items():
    #     if isinstance(v, torch.Tensor):
    #         print(f"{k}: {tuple(v.shape)}")
    #     else:
    #         print(f"{k}: {v}")
    
    # # full sample
    # print(f"Full Sample")
    # d = dset.get_full_sample()
    # for k, v in d.items():
    #     if isinstance(v, torch.Tensor):
    #         print(f"{k}: {tuple(v.shape)}")
    #     else:
    #         print(f"{k}: {v}")
    
    # # frame mode
    # print(f"Frame Mode ON")
    # dset.change_frame_mode(True)
    # print(dset)
    # print(f"Datasets Size: {len(dset)}")
    # batch = dset[0]
    # for k, v in batch.items():
    #     if isinstance(v, torch.Tensor):
    #         print(f"{k}: {tuple(v.shape)}")
    #     else:
    #         print(f"{k}: {v}")
    
    # print(f"Frame Mode OFF")
    # dset.change_frame_mode(False)
    # print(dset)
    # print(f"Datasets Size: {len(dset)}")
    # batch = dset[0]
    # for k, v in batch.items():
    #     if isinstance(v, torch.Tensor):
    #         print(f"{k}: {tuple(v.shape)}")
    #     else:
    #         print(f"{k}: {v}")

    ##############################################
    # NoXi Database
    ##############################################
    print(f"### NoXi ###")
    dset_hf = get_dialog_audio_datasets(datasets=["noxi"], split="test")
    dset = DialogAudioDataset(
        dataset=dset_hf,
        type="sliding",
        expert_and_novice=False,
        multimodal=False,
        audio_overlap=9.5,
        vad_hz=25
    )
    print(dset)
    print(f"Datasets Size: {len(dset)}")

    # from turntaking.vap_to_turntaking import VAP, TurnTakingMetrics
    # metric = TurnTakingMetrics(
    #         hs_kwargs=  dict(
    #             post_onset_shift=1,
    #             pre_offset_shift=1,
    #             post_onset_hold=1,
    #             pre_offset_hold=1,
    #             non_shift_horizon=2,
    #             metric_pad=0.05,
    #             metric_dur=0.1,
    #             metric_pre_label_dur=0.2,
    #             metric_onset_dur=0.2),
    #         bc_kwargs=dict(
    #             max_duration_frames=1.0,
    #             pre_silence_frames=1.0,
    #             post_silence_frames=2.0,
    #             min_duration_frames=0.2,
    #             metric_dur_frames=0.2,
    #             metric_pre_label_dur=0.5
    #         ),
    #         metric_kwargs=dict(
    #             pad=0.05,
    #             dur=0.1,
    #             pre_label_dur=0.5,
    #             onset_dur=0.2,
    #             min_context=3.0
    #         ),
    #         threshold_pred_shift=0.5,
    #         threshold_short_long=0.3,
    #         threshold_bc_pred=0.1,
    #         shift_pred_pr_curve=False,
    #         bc_pred_pr_curve=False,
    #         long_short_pr_curve=False,
    #         frame_hz=25,
    #     )

    # split_dicts = [{key: [dset.data[key][i]] for key in dset.data} for i in range(len(dset.data["dataset_name"]))]

    # for i in range(len(dset.data["dataset_name"])):
    #     print(f"{split_dicts[i]['session'][0]}")
    #     # events = metric.extract_events(va=split_dicts[i]["vad"][0])
    #     events_plot(split_dicts[i])
    #     # for key, value in events.items():
    #     #     print(f"{key}")
    #     #     events_plot(split_dicts[i], key=key, value=value)
    # exit(1)


    ### Dataloader ###
    my_dataloader = DataLoader(dset, batch_size=1)
    pbar_val = tqdm(
                enumerate(my_dataloader),
                total=len(my_dataloader),
            )
    batch = dset[0]
    print(batch["waveform"].shape)
    print(batch["vad"].shape)
    print(batch["label"].shape)
    for ii, batch in pbar_val:
        print(batch["waveform"].shape)
        print(batch["vad"].shape)
        print(batch["label"].shape)
        exit(1)
    
    print(f"### Frame Mode ON ###")
    dset.change_frame_mode(True)
    print(f"Datasets Size: {len(dset)}")

    my_dataloader = DataLoader(dset, batch_size=1)
    pbar_val = tqdm(
                enumerate(my_dataloader),
                total=len(my_dataloader),
            )
    for ii, batch in pbar_val:
        pass
    
    # optional
    dset = DialogAudioDataset(
        dataset=dset_hf,
        type="sliding",
        vad_history=False,
        audio_overlap=0.5,
        audio_duration=1.0,
        vad_hz=25,
        expert_and_novice=False,
        multimodal=False,
    )
    print(dset)
    print(f"Datasets Size: {len(dset)}")

    batch = dset[0]
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")

    # full sample
    print(f"Full Sample")
    d = dset.get_full_sample()
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")

    # frame mode
    print(f"Frame Mode ON")
    dset.change_frame_mode(True)
    print(dset)
    print(f"Datasets Size: {len(dset)}")
    batch = dset[0]
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")
    
    print(f"Frame Mode OFF")
    dset.change_frame_mode(False)
    print(dset)
    print(f"Datasets Size: {len(dset)}")
    batch = dset[0]
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")