# This file contains all the stuff to pre-compute the spectrograms and load/save them.

import torch
import ipynb.fs.defs.notebook as nb
import torch.utils.data as data

from os import makedirs
from os.path import exists, isfile, join
from torchaudio import datasets as audiodatasets
from torchaudio import transforms as audiotransforms
from torchvision import transforms as imgtransforms

SPEC_DIR = "spectrograms"
SPEC_NAME = "spec_"

GTZAN_SAMPLING_RATE = 22050
SAMPLING_RATE = 12000      # 12 kHz

# Spectrogram hyper-parameters
N_FFT = 2048                # samples
WIN_LENGTHS = [25, 50, 100] # ms
HOP_LENGTHS = [10, 25, 50]  # samples
N_MELS = 128
NORMALIZE = True

class AudioDataset(data.Dataset):
    def __init__(
        self,
        root_dir,
        sequence_duration,
        sampling_rate=22050,
        subset=None,
        download=False
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
        root_dir,
        sequence_duration,
        sampling_rate=22050,
        subset=None,
        download=False
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

        self.transform = nb.MelSpectrogram(
            sampling_rate, 
            N_FFT,
            WIN_LENGTHS,
            HOP_LENGTHS,
            N_MELS,
            NORMALIZE
        )
    
    def __len__(self):
        return len(self.audio_ds)
    
    def __getitem__(self, index):
        name = join(self.spec_dir, SPEC_NAME) + str(index) + ".pt"

        if isfile(name):
            return torch.load(name)
        else:
            ret = self.transform(self.audio_ds[index])
            ret.save(name)
            return ret

class FooDataset(data.Dataset):
    def __init__(self, x):
        super().__init__()

        self.x = x
    
    def __len__(self):
        return 10
    
    def __getitem__(self, index):
        return self.x * torch.Tensor([index, 100*index])
    

class BarDataset(data.Dataset):
    def __init__(self, x):
        super().__init__()
        self.parent = FooDataset(x)
        self.location = "./foo"
        if not exists(self.location):
            makedirs(self.location)
    
    def __len__(self):
        return len(self.parent)

    def __getitem__(self, index):
        name = join(self.location, SPEC_NAME) + str(index) + ".pt"
        if isfile(name):
            print(f"loaded '{name}'")
            return torch.load(name)
        else:
            print(f"computed '{name}'")
            ret = self.parent[index]
            torch.save(ret, name)
            return ret

def main():
    # audio_ds = SpecDataset("./data/gtzan")
    audio_ds = BarDataset(3)
    audio_loader = data.DataLoader(
        audio_ds,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    for i, waveform in enumerate(audio_loader):
        waveform = waveform.squeeze(0)

        



if __name__ == "__main__":
    main()