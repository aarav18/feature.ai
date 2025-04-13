import soundfile as sf
import numpy as np
import torchaudio
import torch

# Load the vocals and music audio
vocals, sr_vocals = torchaudio.load("mic_jack_vocals.wav")
music, sr_music = torchaudio.load("beat.wav")

if vocals.shape[0] > 1:
    vocals = vocals.mean(dim=0, keepdim=True)
if music.shape[0] > 1:
    music = music.mean(dim=0, keepdim=True)

# Check if sampling rates match
if sr_vocals < sr_music:
    # downsample music
    resampler = torchaudio.transforms.Resample(orig_freq=sr_music, new_freq=sr_vocals)
    music = resampler(music)
    sr = sr_vocals
elif sr_music < sr_vocals:
    # downsample vocals
    resampler = torchaudio.transforms.Resample(orig_freq=sr_vocals, new_freq=sr_music)
    vocals = resampler(vocals)
    sr = sr_music
else:
    sr = sr_vocals

def pad_to_match(a, b):
    diff = b.shape[1] - a.shape[1]
    if diff > 0:
        pad = torch.zeros((1, diff))
        return torch.cat([a, pad], dim=1)
    return a

vocals = pad_to_match(vocals, music)
music = pad_to_match(music, vocals)

print(vocals.shape, music.shape)

# Make sure they have the same shape (e.g., both stereo or mono)
if vocals.shape != music.shape:
    raise ValueError("Shape mismatch: vocals and music must have the same shape")

# Mix the tracks (add signals)
merged = vocals + music

# Normalize to prevent clipping
merged = merged / merged.abs().max()

# Save the merged audio
sf.write("mj_verse.wav", merged.squeeze().numpy(), sr)
print("Merged file saved as 'mj_verse.wav'")