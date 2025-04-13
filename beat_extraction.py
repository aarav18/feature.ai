import torchaudio

# Load full audio
waveform, sample_rate = torchaudio.load("Bruno Mars - That's What I Like (Lyrics).mp3")

# Set start and end time in seconds
# start_sec = 3.58 # e.g., 10 seconds
# end_sec = 35.82  # e.g., 20 seconds
start_sec = 0
end_sec = 204

# Convert to sample indices
start_sample = int(start_sec * sample_rate)
end_sample = int(end_sec * sample_rate)

# Slice the waveform
clip = waveform[:, start_sample:end_sample]

# Save the clipped audio (optional)
torchaudio.save("bm_first_part.wav", clip, sample_rate=sample_rate)