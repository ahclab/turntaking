# coding: UTF-8
import datetime
import json
import os
import random
import time
import warnings
from os.path import join, dirname
from pprint import pprint
import torch.nn as nn
from einops.layers.torch import Rearrange

import hydra
import pandas as pd
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf 
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from timm.scheduler import CosineLRScheduler
from tqdm import tqdm

from turntaking.callbacks import EarlyStopping
from turntaking.evaluation import roc_to_threshold
from turntaking.model import Model
from turntaking.test import Test
from turntaking.utils import to_device, repo_root, everything_deterministic, write_json, set_seed, set_debug_mode
from turntaking.dataload import DialogAudioDM

everything_deterministic()
warnings.simplefilter("ignore")

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

class Finetune():
    def __init__(self, conf, dm, output_path, model_path, verbose=True) -> None:
        super().__init__()
        self.conf = conf
        self.model_path = model_path
        self.model = Model(self.conf).to(self.conf["train"]["device"])
        self.model.load_state_dict(
            torch.load(model_path, map_location=conf["train"]["device"])
        )
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.net.vap_head.parameters():
            param.requires_grad = True
        for param in self.model.net.main_module.parameters():
            param.requires_grad = True
        # vap_head = VAPHead(conf["model"]["vap"]).to(
        #     conf["train"]["device"]
        # )
        # self.model.net.vap_head = vap_head

        self.dm = dm
        self.dm.change_frame_mode(False)
        self.output_path = output_path

        self.optimizer = self._create_optimizer()
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.95**epoch)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.conf["train"]["max_epochs"], eta_min=0)
        # self.scheduler = CosineLRScheduler(self.optimizer, t_initial=self.conf["train"]["max_epochs"], lr_min=0, warmup_t=3, warmup_lr_init=5e-5, warmup_prefix=True)
        

        self.early_stopping = EarlyStopping(patience=self.conf["train"]["patience"], verbose=self.conf["train"]["verbose"], path=self.output_path)
        self.checkpoint = self._create_checkpoints()

        if verbose == True:
            self.model.model_summary
            # mean, var = self.model.inference_speed
            # print(f"inference speed: {mean}({var})")
            # print(self.model)
            # exit(1)

    def train(self):
        self.model.net.train()
        # initial_weights = {name: weight.clone() for name, weight in self.model.named_parameters()}
        for i in range(self.conf["train"]["max_epochs"]):
            pbar = tqdm(enumerate(self.dm.train_dataloader()), total=len(self.dm.train_dataloader()), dynamic_ncols=True, leave=False)
            
            for ii, batch in pbar:
                self.optimizer.zero_grad()
                loss, _, _ = self.model.shared_step(batch)
                loss["total"].backward()
                self.optimizer.step()
                postfix = f"epoch={i}, loss={loss['total']:>3f}"
                pbar.set_postfix_str(postfix)

                if ii in self.checkpoint:
                    val_loss = self._run_validation()
                    self.model.net.train()
                    print(f"val_loss: {val_loss}")

                    self.early_stopping(val_loss, self.model)
                    if self.early_stopping.early_stop:
                        print("Early stopping triggered.")
                        break

            self.scheduler.step(i+1)

            if self.early_stopping.early_stop:
                break

        # for name, weight in self.model.named_parameters():
        #     if torch.equal(weight, initial_weights[name]):
        #         print(f"Weight {name} did not change!")

        checkpoint = {
            "model": self.early_stopping.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "random": random.getstate(),
            "torch": torch.get_rng_state(),
            "torch_random": torch.random.get_rng_state(),
            "cuda_random": torch.cuda.get_rng_state(),
            "cuda_random_all": torch.cuda.get_rng_state_all(),
        }
        torch.save(checkpoint, join(dirname(self.output_path), "checkpoint.bin"))
        print(f"### END ###")

        return self.early_stopping.model


    def _create_optimizer(self):
        if self.conf["train"]["optimizer"] == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), lr=self.conf["train"]["learning_rate"])
        elif self.conf["train"]["optimizer"] == "SGD":
            return torch.optim.SGD(self.model.parameters(), lr=self.conf["train"]["learning_rate"])
        else:
            print(f"Error optimizer")
            exit(1)


    def _create_checkpoints(self):
        def frange(start, end, step):
            list = [start]
            n = start
            while n + step < end:
                n = n + step
                list.append(n)
            return list
        
        if 0.1 <= self.conf["train"]["checkpoint"] <= 1.0:
            return [int(i * len(self.dm.train_dataloader())) - 1 for i in frange(0.0, 1.1, self.conf["train"]["checkpoint"]) if i != 0.0]
        else:
            print(f"checkpoint must be greater than 0 and less than 1")
            exit(1)

    def _run_validation(self):
        self.model.net.eval()
        pbar_val = tqdm(enumerate(self.dm.val_dataloader()), total=len(self.dm.val_dataloader()), dynamic_ncols=True, leave=False)

        val_loss = 0
        for ii, batch in pbar_val:
            if ii == 1000:
                break 
            with torch.no_grad():
                loss, _, _ = self.model.shared_step(batch)
                val_loss += loss["total"]

        val_loss /= len(self.dm.val_dataloader())

        return val_loss

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    def compile_scores(score_json_path, output_dir):
        df = pd.DataFrame()
        for i, path in enumerate(score_json_path):
            with open(path, 'r') as f:
                data = json.load(f)
            temp_df = pd.json_normalize(data)
            temp_df['model'] = f'model{i:02}'
            temp_df['score_json_path'] = path
            df = pd.concat([df, temp_df], ignore_index=True)
        
        avg_row = df.select_dtypes(include=[np.number]).mean()
        avg_row['model'] = 'Average'
        avg_row['score_json_path'] = f'{join(output_dir, "final_score.csv")}'
        avg_df = pd.DataFrame([avg_row])
        df = pd.concat([df, avg_df], ignore_index=True)
        df = df[['model', 'score_json_path'] + [col for col in df.columns if col not in ['model', 'score_json_path']]]
        return df
    
    cfg_dict = dict(OmegaConf.to_object(cfg))
    debug = cfg_dict["info"]["debug"]
    device = cfg_dict["train"]["device"]

    model_dir_path = input("Please enter the path of the model file dir.: ")
    with open(join(model_dir_path, "log.json")) as f:
        cfg_dict = json.load(f)
        # cfg_dict["train"]["max_epochs"] = 1
        cfg_dict["train"]["learning_rate"] = 1e-4
        # cfg_dict["train"]["optimizer"] = "SGD"
        if debug:
            set_debug_mode(cfg_dict)
    
    datasets = input("Please enter the model datasets: ")
    if datasets not in ["noxi", "eald", "switchboard"]:
        print(f"Error: {datasets} is not defined.")
        exit(1)
    cfg_dict['data']['datasets'] = datasets

    if datasets == "switchboard":
        cfg_dict["data"]["train_files"] = f"turntaking/dataload/dataset/{datasets}/files/train.txt"
        cfg_dict["data"]["val_files"] = f"turntaking/dataload/dataset/{datasets}/files/val.txt"
        cfg_dict["data"]["test_files"] = f"turntaking/dataload/dataset/{datasets}/files/test.txt"
    else:
        data_id = input(
            "[NoXi Database]\n"
            "0: Nottingham\n"
            "1: Augsburg\n"
            "2: Paris\n"
            "3: Nara\n"
            "[eald]\n"
            "0: Elderly - Caretaker\n"
            "1: Elderly - Psychologist\n"
            "2: Elderly - Student\n"
            "Select the data set number to be used for fine tuning.: "
            )

    
        cfg_dict["data"]["train_files"] = f"turntaking/dataload/dataset/{datasets}/files/train_{data_id}.txt"
        cfg_dict["data"]["val_files"] = f"turntaking/dataload/dataset/{datasets}/files/val_{data_id}.txt"
        cfg_dict["data"]["test_files"] = f"turntaking/dataload/dataset/{datasets}/files/test_{data_id}.txt"

    dm = DialogAudioDM(**cfg_dict["data"])
    dm.setup(None)

    score_json_path = []
    id = datetime.datetime.now().strftime("%H%M%S")
    d = datetime.datetime.now().strftime("%Y_%m_%d")

    # Run
    for i in range(cfg_dict["train"]["trial_count"]):
        ### Preparation ###
        output_dir = os.path.join(repo_root(), "output", d, id, str(i).zfill(2))
        model_path = os.path.join(model_dir_path, str(i).zfill(2), "model.pt")
        output_path = os.path.join(output_dir, "model.pt")
        os.makedirs(output_dir, exist_ok=True)
        set_seed(i)

        ### Train ###
        finetune = Finetune(cfg_dict, dm, output_path, model_path, cfg_dict["train"]["verbose"])
        model = finetune.train()

        ### Test ###
        test = Test(cfg_dict, dm, output_path, output_dir, True)
        score, turn_taking_probs, probs, events = test.test()
        write_json(score, join(output_dir, "score.json"))

        print(f"Output Model -> {output_path}")
        print("Saved score -> ", join(output_dir, "score.json"))

        score_json_path.append(join(output_dir, "score.json"))

    output_dir = os.path.join(repo_root(), "output", d, id)
    df = compile_scores(score_json_path, output_dir)
    print("-" * 60)
    print(f"Output Final Score -> {join(output_dir, 'final_score.csv')}")
    print(df)
    print("-" * 60)
    write_json(cfg_dict, os.path.join(output_dir, "log.json")) # log file
    df.to_csv(join(output_dir, "final_score.csv"), index=False)


if __name__ == "__main__":
    main()
