import os
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # For progress bars
from hyperparams import Hyperparams as hp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_flac_to_wav_torchaudio(flac_path, wav_path):
    """Convert a .flac file to .wav format using torchaudio"""
    waveform, sample_rate = torchaudio.load(flac_path)  # Load FLAC
    torchaudio.save(wav_path, waveform, sample_rate)    # Save as WAV

def get_spectrograms_torchaudio(wav_path):
    """Compute mel spectrogram and magnitude spectrogram using torchaudio"""
    # Load audio file
    waveform, sr = torchaudio.load(wav_path)
    waveform = waveform.to(device)

    # Resample if needed
    if sr != hp.sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=hp.sr).to(device)
        waveform = resampler(waveform)

    # Pre-emphasis (simple filter)
    waveform = torch.nn.functional.pad(waveform, (1, 0))
    waveform = waveform[:, 1:] - hp.preemphasis * waveform[:, :-1]

    # Compute STFT
    stft = torch.stft(
        waveform,
        n_fft=hp.n_fft,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        return_complex=True
    )
    mag = torch.abs(stft)

    # Compute mel spectrogram
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=hp.sr,
        n_fft=hp.n_fft,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        n_mels=hp.n_mels
    ).to(device)
    mel = mel_spectrogram_transform(waveform)

    # Convert to dB scale
    mel = 20 * torch.log10(torch.clamp(mel, min=1e-5))
    mag = 20 * torch.log10(torch.clamp(mag, min=1e-5))

    # Normalize
    mel = torch.clamp((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = torch.clamp((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    return mel.squeeze().T.cpu().numpy(), mag.squeeze().T.cpu().numpy()

class AudioDataset(Dataset):
    """Custom Dataset for loading audio files"""
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.file_names = [f for f in os.listdir(input_dir) if f.endswith(".flac")]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        flac_path = os.path.join(self.input_dir, file_name)
        return file_name, flac_path

def preprocess_audio_gpu(input_dir, output_mel_dir, output_mag_dir, batch_size=4):
    os.makedirs(output_mel_dir, exist_ok=True)
    os.makedirs(output_mag_dir, exist_ok=True)

    # Create dataset and dataloader
    dataset = AudioDataset(input_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Process files in batches with a progress bar
    with tqdm(total=len(dataset), desc="Processing Audio Files", unit="file") as pbar:
        for batch in dataloader:
            file_names, flac_paths = batch  # Unpack batch
            for file_name, flac_path in zip(file_names, flac_paths):
                # Convert FLAC to WAV in memory
                wav_path = flac_path.replace(".flac", ".wav")
                convert_flac_to_wav_torchaudio(flac_path, wav_path)

                # Compute spectrograms
                mel, mag = get_spectrograms_torchaudio(wav_path)

                # Save mel and mag spectrograms
                mel_path = os.path.join(output_mel_dir, file_name.replace(".flac", "_mel.npy"))
                mag_path = os.path.join(output_mag_dir, file_name.replace(".flac", "_mag.npy"))
                np.save(mel_path, mel)
                np.save(mag_path, mag)

                # Remove temporary WAV file
                os.remove(wav_path)

                # Update progress bar
                pbar.update(1)

# Update paths
input_audio_folder = r"D:\FYP\SonicCypher\SASVC2022_Baseline\LA\ASVspoof2019_LA_dev\flac"
output_mel_folder = "./mels_dev"
output_mag_folder = "./mags_dev"

preprocess_audio_gpu(input_audio_folder, output_mel_folder, output_mag_folder)