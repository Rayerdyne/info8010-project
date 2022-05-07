# This file contains all the stuff to pre-compute the spectrograms and load/save them.

from typing import List, Tuple
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

SEQUENCE_DURATIONS: Tuple[int, int] = (20, 28)      # s
MAX_AUDIO_LENGTH: int = 345600                      # s
MAX_AUDIO_LENGTH_GTZAN: int = int(MAX_AUDIO_LENGTH * GTZAN_SAMPLING_RATE / SAMPLING_RATE)

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
        normalized: bool = True,
    ):
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
        self.n_mels = n_mels
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # squeeze needed because MelSpectrogram returns: (channel, n_mels, time)
        # tensor, with channel dimension being always 1 -> squeeze it
        out = []
        mid_size = int(x.shape[-1] / HOP_LENGTHS[1] + 1)
        print(f"x.shape: {x.shape} -> {mid_size}")
        resize = imgtransforms.Resize((self.n_mels, mid_size))
        for spec in self.spectrograms:
            y = spec(x)
            y = resize(y)
            y = y.squeeze(0)
            out.append(y)
        return torch.stack(out)

class InverseMelSpectrogramSlow(object):
    """Frickin' slow"""
    def __init__(
        self,
        sequence_duration: int = SEQUENCE_DURATIONS[0],
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
            y = spec.unsqueeze(0)
            y = resize(y)
            y = inverse(y)

            out.append(y)
        
        out = torch.stack(out)
        out = out.mean(0)

        return out

class InverseMelSpectrogram(object):
    def __init__(
        self,
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
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        spec_length = x.shape[-1]
        print(f"spec length: {spec_length}")
        sizes = [int(((spec_length - 1) * HOP_LENGTHS[1]) / HOP_LENGTHS[i] + 1) for i in range(3)]
        resizes = [ imgtransforms.Resize((self.n_mels, size)) for size in sizes ]
        recs = []
        for i in range(3):
            y = resizes[i](x[i].unsqueeze(0))
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
        sampling_rate: int = SAMPLING_RATE,
        subset=None,
        download: bool = False
    ):
        super().__init__()

        self.gtzan = audiodatasets.GTZAN(root_dir, subset=subset, download=download)
        self.transform = imgtransforms.Compose([
            imgtransforms.CenterCrop((1, MAX_AUDIO_LENGTH_GTZAN)),
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
            NORMALIZE
        )

        self.transform = transform
        self.save = save
    
    def __len__(self):
        return len(self.audio_ds)
    
    def __getitem__(self, index):
        name = join(self.spec_dir, SPEC_NAME) + str(index) + ".spec.pt"

        if isfile(name):
            (out, genre) = torch.load(name)
        else:
            waveform, _sr, genre = self.audio_ds[index]
            out = self.spectrogram(waveform)
            if self.save:
                torch.save((out, genre), name)
        
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
    ds = SpecDataset("./data/gtzan", save=False)
    loader = data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    length = len(loader)
    for i, spec in enumerate(loader):
        print(f"{i} / {length}")


def test_spectrogram_inverse():
    spec_ds = SpecDataset("./data/gtzan", save=False)
    spec, genre = spec_ds[0]

    inverse = InverseMelSpectrogram()
    rec = inverse(spec)
    print(f"rect: {rec.shape}")

def test_variable_length():
    ds = SpecDataset("./data/gtzan", save=False)
    u = ds[0]
    print(f"u.shape: {u[0].shape}")
    spec = audiotransforms.MelSpectrogram(
        sample_rate=SAMPLING_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTHS[0],
        win_length=WIN_LENGTHS[0],
        n_mels=N_MELS,
        normalized=NORMALIZE
    )
    for i in range(1000 + HOP_LENGTHS[0], 1000 + 3*HOP_LENGTHS[0]+1):
        xt = torch.randn(1, i)
        xf = spec(xt)
        print(f"{xt.shape[-1]} -> {xf.shape[-1]} ({xt.shape[-1] // HOP_LENGTHS[0] + 1})")
        # audio length -> spec length
        # 1199         -> 10
        # 1200         -> 11
        # 1319         -> 11
        # 1320         -> 12

def test_cropping_lengths():
    a = 3
    b = 10
    for i in range(100):
        print(f"{i}) {i // b + 1}, {(i // b + 1) / a}", end="")
        if i % (a * b) < b:
            print(f" OK ({i - i % (a*b)})")
        else:
            print(f" crop at {(a*b) * (i // (a*b))}")

if __name__ == "__main__":
    test_spectrogram_inverse()
    # main()