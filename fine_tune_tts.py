from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
import torch
import soundfile as sf
from datasets import load_dataset
from torchaudio.transforms import GriffinLim

# Loading the processor and model
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

# Loading the speaker embeddings from the validation split
speaker_embeddings = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")["xvector"][:1]  # Load one embedding
speaker_embeddings = torch.tensor(speaker_embeddings)

# Text that we want to speak
text = "Your technical text here, e.g., API, CUDA, TTS."
inputs = processor(text=text, return_tensors="pt")

# Generating the speech with speaker embeddings
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings=speaker_embeddings)


spectrogram = speech  # Assuming output from model is a spectrogram

print("Spectrogram shape before processing:", spectrogram.shape)

# Adjusting the shape
if spectrogram.dim() == 2:
    spectrogram = spectrogram.unsqueeze(0)  # Add a batch dimension if it's 2D

# Checking the number of frequency bins
expected_bins = 513
actual_bins = spectrogram.shape[1]

# Adjust the n_fft value
if actual_bins != expected_bins:
    print(f"Warning: Expected {expected_bins} frequency bins, but got {actual_bins}. Adjusting `n_fft`.")
    n_fft = (actual_bins - 1) * 2  # Calculating the new n_fft
else:
    n_fft = 1024  # default

# Griffin-Lim parameters
hop_length = 256
win_length = n_fft

# Apply Griffin-Lim to convert spectrogram back to waveform
griffin_lim = GriffinLim(n_fft=n_fft, win_length=win_length, hop_length=hop_length)
waveform = griffin_lim(spectrogram)

# Ensure waveform is 1D for mono audio
if waveform.dim() > 1:
    waveform = waveform.squeeze(0)  # Remove batch dimensions

# Check waveform shape and dtype
print("Waveform shape:", waveform.shape)
print("Waveform dtype:", waveform.dtype)

# Ensure the waveform is a floating-point tensor
if not waveform.dtype.is_floating_point:
    waveform = waveform.float()

# Normalize the waveform to be in the range [-1, 1]
waveform = waveform / waveform.abs().max()

# Save the output audio
try:
    sf.write("output.wav", waveform.numpy(), samplerate=22050)
    print("Audio saved successfully as 'output.wav'")
except Exception as e:
    print(f"Error saving audio: {e}")
