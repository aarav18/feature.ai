import torch
import torchaudio
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import os
from openunmix import predict
from pathlib import Path

input_file = "beat_and_vocals.wav"
output_dir = f"separated_{input_file.split(".")[0]}"
os.makedirs(output_dir, exist_ok=True)

waveform, sr = torchaudio.load(input_file)

if waveform.shape[0] == 1:
    waveform = waveform.repeat(2, 1)

estimates = predict.separate(
    audio=waveform,
    rate=sr,
    targets=["vocals"],
    residual=True,
    niter=1
)

torchaudio.save(f"{output_dir}/vocals.wav", estimates["vocals"].squeeze(0), sample_rate=44100)
torchaudio.save(f"{output_dir}/residual.wav", estimates["residual"].squeeze(0), sample_rate=44100)

print(f"Separation complete. Stems saved in {output_dir}.")