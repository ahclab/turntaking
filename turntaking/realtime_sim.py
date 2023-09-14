import pyaudio
import numpy as np
import time
import json
from os.path import join, dirname, dirname
from functools import partial

from turntaking.model import Model

RATE = 16000 
CHUNK = int(RATE / 25)
SECONDS = 10 
BUFFER_SIZE = RATE * SECONDS
DEVICE = "cuda:0"

def callback(audio_buffer, model, in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    audio_buffer = np.append(audio_buffer, audio_data)[-BUFFER_SIZE:]

    # print(audio_buffer)
    
    out = model.output(audio_buffer)

    return (in_data, pyaudio.paContinue)


def main():
    model_path = "/ahc/work2/kazuyo-oni/turntaking/output/2023_09_12/170141/00/model.pt"

    with open(join(dirname(dirname(model_path)), "log.json")) as f:
        cfg_dict = json.load(f)
        cfg_dict["train"]["device"] = DEVICE
    
    model = Model(cfg_dict).to(cfg_dict["train"]["device"])

    audio_buffer = np.zeros(BUFFER_SIZE)
    p = pyaudio.PyAudio()

    stream_callback = partial(callback, audio_buffer, model)

    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=stream_callback)

    stream.start_stream()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    stream.stop_stream()
    stream.close()
    p.terminate()
        

if __name__ == "__main__":
    main()