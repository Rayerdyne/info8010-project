# This file contains all the stuff to pre-compute the spectrograms and load/save them.

import torch
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
HOP_LENGTHS = [10, 25, 50]  # ms
N_MELS = 128
NORMALIZE = True

WIN_LENGTHS = [int(round(x * SAMPLING_RATE / 1000)) for x in WIN_LENGTHS]
HOP_LENGTHS = [int(round(x * SAMPLING_RATE / 1000)) for x in HOP_LENGTHS]

# lenght the spectrogram are resized to (height is n_mels)
# untreated output lengths are 401, 801 and 2001
SPEC_LENGTH = 801

class MelSpectrogram(object):
    def __init__(self, sample_rate, n_fft, win_lengths, hop_lengths, n_mels, normalized):
        self.resize = imgtransforms.Resize((N_MELS, SPEC_LENGTH))
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
    
    def __call__(self, tensor):
        # squeeze needed because MelSpectrogram returns: (channel, n_mels, time)
        # tensor, with channel dimension being always 1 -> squeeze it
        out = []
        for spec in self.spectrograms:
            x = spec(tensor)
            x = self.resize(x)
            x = x.squeeze(0)
            out.append(x)
        return torch.stack(out)

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
        transform=None,
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

        self.spectrogram = MelSpectrogram(
            sampling_rate, 
            N_FFT,
            WIN_LENGTHS,
            HOP_LENGTHS,
            N_MELS,
            NORMALIZE
        )

        self.transform = transform
    
    def __len__(self):
        return len(self.audio_ds)
    
    def __getitem__(self, index):
        name = join(self.spec_dir, SPEC_NAME) + str(index) + ".pt"

        if isfile(name):
            out = torch.load(name)
        else:
            waveform, _sr, genre = self.audio_ds[index]
            out = self.spectrogram(waveform)
            # torch.save(out, name)
        
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



# recovering music from spectrograms still to be done
# # actually, very slow due to InverseMelScale, so that one wants to train
# # only using spectrograms... phew
# # it sucks still a bit, maybe recover using librosa mel_to_audio
# inverse_transformer = imgtransforms.Compose([
#     # pick one of the recovered spectrograms
#     imgtransforms.Lambda(lambda y: y[0].unsqueeze(0)),
#     audiotransforms.InverseMelScale(
#         sample_rate=sampling_rate,
#         n_stft=(n_fft // 2 + 1),
#         # n_mels=n_mels,
#     ),
#     audiotransforms.GriffinLim(
#         n_fft=n_fft,
#         hop_length=hop_lengths[0],
#     ),
# ])

def main():
    audio_ds = SpecDataset("./data/gtzan")
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