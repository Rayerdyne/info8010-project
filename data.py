# This file contains all the stuff to pre-compute the spectrograms and load/save them.

from typing import List
import torch
import torch.utils.data as data

from os import makedirs
from os.path import exists, isfile, join
from torchaudio import datasets as audiodatasets
from torchaudio import transforms as audiotransforms
from torchvision import transforms as imgtransforms

from librosa.feature.inverse import mel_to_audio

SPEC_DIR = "spectrograms"
SPEC_NAME = "spec_"

GTZAN_SAMPLING_RATE: int = 22050
SAMPLING_RATE: int = 12000      # 12 kHz

SEQUENCE_DURATION: int = 20      # s

# Spectrogram hyper-parameters
N_FFT: int = 2048                      # samples
WIN_LENGTHS: List[int] = [25, 50, 100] # ms
HOP_LENGTHS: List[int] = [10, 25, 50]  # ms
N_MELS: int = 128
NORMALIZE: bool = True

WIN_LENGTHS = [int(round(x * SAMPLING_RATE / 1000)) for x in WIN_LENGTHS]
HOP_LENGTHS = [int(round(x * SAMPLING_RATE / 1000)) for x in HOP_LENGTHS]

# lenght the spectrogram are resized to (height is n_mels)
# untreated output lengths are 401, 801 and 2001
SPEC_LENGTH: int = 801

class MelSpectrogram(object):
    def __init__(
        self,
        sample_rate: int = SAMPLING_RATE,
        n_fft: int = N_FFT,
        win_lengths: List[int] = WIN_LENGTHS,
        hop_lengths: List[int] = HOP_LENGTHS,
        n_mels: int = N_MELS,
        spec_length: int = SPEC_LENGTH,
        normalized: bool = True,
    ):
        self.resize = imgtransforms.Resize((n_mels, spec_length))
        self.spectrograms = [
            audiotransforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mels=n_mels,
                normalized=normalized
            )
            for win_length, hop_length in zip(win_lengths, hop_lengths)
        ]
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # squeeze needed because MelSpectrogram returns: (channel, n_mels, time)
        # tensor, with channel dimension being always 1 -> squeeze it
        out = []
        for spec in self.spectrograms:
            print(f"1 -> {x.shape}")
            y = spec(x)
            print(f"2 -> {y.shape}")
            y = self.resize(y)
            print(f"3 -> {y.shape}")
            y = y.squeeze(0)
            print(f"4 -> {y.shape}")
            out.append(y)
        return torch.stack(out)

class InverseMelSpectrogramSlow(object):
    """Frickin' slow"""
    def __init__(
        self,
        sequence_duration: int = SEQUENCE_DURATION,
        sample_rate: int = SAMPLING_RATE,
        n_fft: int = N_FFT,
        win_lengths: List[int] = WIN_LENGTHS,
        hop_lengths: List[int] = HOP_LENGTHS,
        n_mels: int = N_MELS,
        # normalized: bool
    ):
        sizes = [int((sequence_duration * sample_rate / x) + 1) for x in hop_lengths]
        self.resizes = [
            imgtransforms.Resize((n_mels, size)) for size in sizes
        ]
        self.inverses = [
            imgtransforms.Compose([
                # imgtransforms.Lambda(lambda y: y.unsqueeze(0)),
                audiotransforms.InverseMelScale(
                    sample_rate=sample_rate,
                    n_stft=(n_fft // 2 + 1),
                    # n_mels=n_mels,
                ),
                audiotransforms.GriffinLim(
                    n_fft=n_fft,
                    hop_length=hop_length,
                ),
            ]) for hop_length in hop_lengths
        ]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for resize, inverse, spec in zip(self.resizes, self.inverses, x):
            print(resize)
            print(f" A -> {spec.shape}")
            y = spec.unsqueeze(0)
            print(f" B -> {y.shape}")
            y = resize(y)
            print(f" C -> {y.shape}")
            y = inverse(y)
            print(f" D -> {y.shape}")

            out.append(y)
        
        out = torch.stack(out)
        out = out.mean(0)

        return out

class InverseMelSpectrogram(object):
    def __init__(
        self,
        sequence_duration: int = SEQUENCE_DURATION,
        sample_rate: int = SAMPLING_RATE,
        n_fft: int = N_FFT,
        win_lengths: List[int] = WIN_LENGTHS,
        hop_lengths: List[int] = HOP_LENGTHS,
        n_mels: int = N_MELS,
    ):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.win_lengths = win_lengths
        self.hop_lengths = hop_lengths
        self.n_mels = n_mels

        sizes = [int((sequence_duration * sample_rate / x) + 1) for x in hop_lengths]
        self.resizes = [
            imgtransforms.Resize((n_mels, size)) for size in sizes
        ]
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        recs = []
        for i in range(3):
            y = self.resizes[i](x[i].unsqueeze(0))
            print(f"shape of librosa input: {y.shape}")
            rec = mel_to_audio(
                y.numpy(),
                sr = self.sr,
                n_fft = self.n_fft,
                hop_length = self.hop_lengths[i],
                win_length = self.win_lengths[i],
            )
            recs.append(torch.from_numpy(rec))
        
        recs = torch.stack(recs)
        return recs.mean(0)

class AudioDataset(data.Dataset):
    def __init__(
        self,
        root_dir: str,
        sequence_duration: int = SEQUENCE_DURATION,
        sampling_rate: int = SAMPLING_RATE,
        subset=None,
        download: bool = False
    ):
        super().__init__()

        self.gtzan = audiodatasets.GTZAN(root_dir, subset=subset, download=download)
        self.transform = imgtransforms.Compose([
            imgtransforms.RandomCrop((1, sequence_duration * GTZAN_SAMPLING_RATE)),
            audiotransforms.Resample(GTZAN_SAMPLING_RATE, sampling_rate),
        ])
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.gtzan)
    
    def __getitem__(self, index):
        waveform, _sr, genre = self.gtzan[index]
        waveform = self.transform(waveform)

        return waveform, self.sampling_rate, genre

class SpecDataset(data.Dataset):
    def __init__(
        self,
        root_dir: str,
        sequence_duration: int = SEQUENCE_DURATION,
        sampling_rate: int = SAMPLING_RATE,
        transform=None,
        subset=None,
        download: bool = False,
        save: bool = True,
    ):
        super().__init__()
        self.location = root_dir
        self.spec_dir = join(root_dir, SPEC_DIR)
        if not exists(self.spec_dir):
            makedirs(self.spec_dir)

        self.audio_ds = AudioDataset(
            root_dir,
            sequence_duration,
            sampling_rate,
            subset,
            download
        )

        self.spectrogram = MelSpectrogram(
            sampling_rate, 
            N_FFT,
            WIN_LENGTHS,
            HOP_LENGTHS,
            N_MELS,
            SPEC_LENGTH,
            NORMALIZE
        )

        self.transform = transform
        self.save = save
    
    def __len__(self):
        return len(self.audio_ds)
    
    def __getitem__(self, index):
        name = join(self.spec_dir, SPEC_NAME) + str(index) + ".pt"

        if isfile(name):
            out = torch.load(name)
        else:
            waveform, _sr, genre = self.audio_ds[index]
            out = self.spectrogram(waveform)
            if self.save:
                torch.save(out, name)
        
        if self.transform is not None:
            out = self.transform(out)
        
        return out, genre


class OldDataset(data.Dataset):
    """for info"""
    def __init__(self, root_dir, transformer, sr=SAMPLING_RATE, subset=None, download=False):
        super().__init__()

        self.gtzan = audiodatasets.GTZAN(root_dir, subset=subset, download=download)
        self.transform = transformer
        self.sr = sr

    def __len__(self):
        return len(self.gtzan)
    
    def __getitem__(self, index):
        waveform, _sr, genre = self.gtzan[index]
        spec = self.transform(waveform)
        return waveform[0, :self.sr * 28], spec, self.sr, genre


def main():
    audio_ds = SpecDataset("./data/gtzan", save=False)
    audio_loader = data.DataLoader(
        audio_ds,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    length = len(audio_loader)
    for i, spec in enumerate(audio_loader):
        print(f"{i} / {length}")


def test():
    spec_ds = SpecDataset("./data/gtzan", save=False)
    spec, genre = spec_ds[0]

    inverse = InverseMelSpectrogram()
    rec = inverse(spec)
    print(f"rect: {rec.shape}")

if __name__ == "__main__":
    main()