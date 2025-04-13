import torchaudio

def extract(path, start_time, end_time):
    waveform, sample_rate = torchaudio.load(path)
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    clip = waveform[:, start_sample:end_sample]
    return clip

def merge(vocals, sr_vocals, music, sr_music):
    vocals, sr_vocals = torchaudio.load("testing1_vocals_AI.wav")
    music, sr_music = torchaudio.load("testing1_music.wav")

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

    # Make sure they have the same shape (e.g., both stereo or mono)
    if vocals.shape != music.shape:
        raise ValueError("Shape mismatch: vocals and music must have the same shape")

    # Mix the tracks (add signals)
    merged = vocals + music

    # Normalize to prevent clipping
    merged = merged / merged.abs().max()

    # Save the merged audio
    sf.write("merged.wav", merged.squeeze().numpy(), sr)
    print("Merged file saved as 'merged.wav'")