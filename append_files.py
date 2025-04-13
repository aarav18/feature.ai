import torchaudio
import torch
from torchaudio.transforms import Resample

# Load the first and second WAV files
waveform1, sr1 = torchaudio.load("bm_first_part.wav")   # PUT FIRST AUDIO FILE HERE
waveform2, sr2 = torchaudio.load("mj_verse.wav")        # PUT SECOND AUDIO FILE HERE

sr = sr1
# Sanity check: sample rates and channel counts must match
if sr1 < sr2:
    # resample waveform2
    resampler = Resample(orig_freq=sr2, new_freq=sr1)
    waveform2 = resampler(waveform2)
elif sr2 < sr1:
    # resample sr1
    resampler = Resample(orig_freq=sr1, new_freq=sr2)
    waveform1 = resampler(waveform1)
    sr = sr2

if waveform1.shape[0] > 1:
    waveform1 = waveform1.mean(dim=0, keepdim=True)
if waveform2.shape[0] > 1:
    waveform2 = waveform2.mean(dim=0, keepdim=True)    

# Concatenate along the time axis (dim=1)
combined_waveform = torch.cat((waveform1, waveform2), dim=1)

# Save the combined waveform
torchaudio.save("DEMO_2.wav", combined_waveform, sr)