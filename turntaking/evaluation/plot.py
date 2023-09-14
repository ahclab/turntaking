import numpy as np
from os.path import join, isfile, basename, splitext, dirname as dr
import matplotlib.pyplot as plt
import torch

def create_and_save_plot(batch, key, output_path, time=30*25):
    fig, ax = plt.subplots(batch[f"{key}_expert"].size(-1), 1, figsize=(16,4*batch[f"{key}_expert"].size(-1)))
    for i in range(batch[f"{key}_expert"].size(-1)):
        ax[i].set_xlim([0, time])
        ax[i].set_xticks(range(0, time+1, 50), fontsize=5)
        ax[i].set_xticklabels(range(0, int(time/25+1), 2))
        # ax[i].set_ylim([0, 1])
        ax[i].set_yticks([])
        x = np.arange(time)

        expert = batch[f"{key}_expert"][:,:,i].squeeze()
        novice = batch[f"{key}_novice"][:,:,i].squeeze()
        ax[i].plot(x, expert, color="#00a3d9", linewidth=1)
        ax[i].plot(x, novice, color="#f5003d", linewidth=1)

    fig.savefig(f"{join(dr(output_path), f'{splitext(basename(output_path))[0]}_{key}.pdf')}", format="pdf")
    plt.cla()

def events_plot(
        batch,
        turn_taking_probs, 
        events, 
        output_path, 
        audio_duration = 10,
        frame_hz = 25,
        sample_rate = 16000,
        multimodal = False,
        ):
    import matplotlib.pyplot as plt
    from turntaking.augmentations import torch_to_praat_sound
    import numpy as np

    vad_expert = batch["vad"][:,:,0].squeeze()
    vad_novice = batch["vad"][:,:,1].squeeze()

    indices_expert = np.where(vad_expert == 1)
    indices_novice = np.where(vad_novice == 1)

    waveform_expert = torch_to_praat_sound(batch["waveform_expert"][0].detach().numpy().copy(), sample_rate) + 1
    waveform_novice = torch_to_praat_sound(batch["waveform_novice"][0].detach().numpy().copy(), sample_rate) - 1

    fig = plt.figure(figsize=(16, 4), dpi=300)
    ax = fig.add_subplot(111)
    

    x = np.arange(len(vad_expert))
    snd_x = np.linspace(0,len(vad_expert),len(waveform_expert))

    ax.set_xlim([0, len(vad_expert)])
    ax.set_xticks(range(0, len(vad_expert)+1, 50), fontsize=5)
    ax.set_xticklabels(range(0, int(len(vad_expert)/25+1), 2))
    ax.set_ylim([-2, 2])
    ax.set_yticks([])

    ax.axhline(y=0, linewidth=0.1, color="k")

    ax.plot(snd_x, waveform_expert.values.T, alpha=0.4, linewidth=0.01)
    ax.plot(snd_x, waveform_novice.values.T, alpha=0.4, linewidth=0.01)

    # ax.plot(x,vad_expert, linewidth=0.01)
    # ax.plot(x,vad_novice, linewidth=0.01)

    for index in indices_expert[0]:
        plt.hlines(y=1, xmin=index-0.5, xmax=index+0.5, linewidth=1, color="#0f79d6")
    for index in indices_novice[0]:
        plt.hlines(y=-1, xmin=index-0.5, xmax=index+0.5, linewidth=1, color="#ba4704")
    
    p_expert = 2 * turn_taking_probs["p"][:,:,0].squeeze().to("cpu")
    p_novice = 2 * turn_taking_probs["p"][:,:,1].squeeze().to("cpu") - 2
    p_bc_expert = 3 * turn_taking_probs["bc_prediction"][:,:,0].squeeze().to("cpu")
    p_bc_novice = 3 * turn_taking_probs["bc_prediction"][:,:,1].squeeze().to("cpu") - 2

    # ax.plot(x, p_expert, color="#00a3d9", linewidth=1)
    # ax.plot(x, p_novice, color="#f5003d", linewidth=1)

    ax.plot(x, p_bc_expert, color="#00a3d9", linewidth=1)
    ax.plot(x, p_bc_novice, color="#f5003d", linewidth=1)

    fig.savefig(f"{output_path}", format="pdf")
    plt.cla()

    if multimodal:
        create_and_save_plot(batch, "gaze", output_path, time=len(vad_expert))
        create_and_save_plot(batch, "au", output_path, time=len(vad_expert))
        create_and_save_plot(batch, "head", output_path, time=len(vad_expert))
        create_and_save_plot(batch, "pose", output_path, time=len(vad_expert))

