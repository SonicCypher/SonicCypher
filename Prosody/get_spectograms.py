import os
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from hyperparams import Hyperparams as hp

import torch
import torchaudio

def convert_flac_to_wav(flac_path, wav_path):
    """Convert a .flac file to .wav format"""
    audio = AudioSegment.from_file(flac_path, format="flac")
    audio.export(wav_path, format="wav")

# def get_spectrograms(fpath):
#     """Extract spectrograms from audio files"""
#     y, sr = librosa.load(fpath, sr=hp.sr)

#     y, _ = librosa.effects.trim(y)
#     y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

#     linear = librosa.stft(y=y, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)

#     mag = np.abs(linear)
#     mel_basis = librosa.filters.mel(sr=hp.sr, n_fft=hp.n_fft, n_mels=hp.n_mels)
#     mel = np.dot(mel_basis, mag)

#     mel = 20 * np.log10(np.maximum(1e-5, mel))
#     mag = 20 * np.log10(np.maximum(1e-5, mag))

#     mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
#     mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

#     mel = mel.T.astype(np.float32)
#     mag = mag.T.astype(np.float32)

#     return mel, mag

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_spectrograms(wav_path):
    waveform, sr = torchaudio.load(wav_path)
    waveform = waveform.to(device)

    # Optional: resample to target sample rate
    if sr != hp.sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=hp.sr).to(device)
        waveform = resampler(waveform)

    # Pre-emphasis (simple filter)
    waveform = torch.nn.functional.pad(waveform, (1, 0))
    waveform = waveform[:, 1:] - hp.preemphasis * waveform[:, :-1]

    # STFT
    stft = torch.stft(
        waveform,
        n_fft=hp.n_fft,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        return_complex=True
    )

    mag = torch.abs(stft)

    # Mel filter
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=hp.sr,
        n_fft=hp.n_fft,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        n_mels=hp.n_mels
    ).to(device)

    mel = mel_spectrogram(waveform)

    # dB scale
    mel = 20 * torch.log10(torch.clamp(mel, min=1e-5))
    mag = 20 * torch.log10(torch.clamp(mag, min=1e-5))

    # Normalization
    mel = torch.clamp((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = torch.clamp((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    return mel.squeeze().T.cpu().numpy(), mag.squeeze().T.cpu().numpy()

def preprocess_audio(input_dir, output_mel_dir, output_mag_dir, temp_wav_dir="./temp_wav"):
    os.makedirs(output_mel_dir, exist_ok=True)
    os.makedirs(output_mag_dir, exist_ok=True)
    os.makedirs(temp_wav_dir, exist_ok=True)  # Temporary folder for .wav files

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".flac"):
            print(f"Processing {file_name}...")
            flac_path = os.path.join(input_dir, file_name)
            wav_path = os.path.join(temp_wav_dir, file_name.replace(".flac", ".wav"))

            # Convert .flac to .wav
            convert_flac_to_wav(flac_path, wav_path)

            # Process the converted .wav file
            mel, mag = get_spectrograms(wav_path)

            mel_path = os.path.join(output_mel_dir, file_name.replace(".flac", "_mel.npy"))
            mag_path = os.path.join(output_mag_dir, file_name.replace(".flac", "_mag.npy"))
            np.save(mel_path, mel)
            np.save(mag_path, mag)

            # Remove temporary wav file after processing
            os.remove(wav_path)
            print(f"Finished processing {file_name}.")

# Update paths
input_audio_folder = r"D:\FYP\SonicCypher\SASVC2022_Baseline\LA\ASVspoof2019_LA_dev\flac"
output_mel_folder = "./mels_eval"
output_mag_folder = "./mags_eval"

preprocess_audio(input_audio_folder, output_mel_folder, output_mag_folder)
