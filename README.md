# INFO8010: Deep learning Project
## Latent space manipulation on music sequences

### Objectives
- Build a latent space representation for short sequences of music
- Use that latent space to:
  - interpolate between two existing sequences to create new ones
  - tweak existing sequence to max out some dimension to understand what is does

### General architecture

**Pre-processing**:

1. Mel-spectrogram, following the example of [1] and [2].
   - Input: waveform
   - Output: tensor (image)

**Encoder**:

The music signal will go through the following steps:
1. CNN layers, as proposed in [1] and [2] as well. 
   To be precised and tuned for best results.
   - Input: tensor (image)
   - Output: tensor, time-dependent and high-dimensional representation of input-sequence
1. RNN / transformers / attention layers, that is, deep learning tools for efficient manipulation of sequential data.
   To be precised and tuned for best results.
   - Output: tensor, time-independent and low(er)-dimentional representation
1. Fully connected layers "finish the work" of the previous layers.
   Needed ? To be precised and toned for best results.
   - Ouput: the latent space representation

**Decoder**:

By simplicity, we will simply consider the inverse steps of the encoder.

### Means

- From [7], we know that we can use pre-trained models from ImageNet as a starting point for ours, even if they originally came from standard image processing, not (mel) spectrograms. Actually in [7], their model is tested on the GTZAN dataset, used as audio input here
- 

### References
- [1] [Automatic Tagging using Deep Convolutional Neural Networks](https://scholar.google.co.kr/citations?view_op=view_citation&hl=en&user=ZrqdSu4AAAAJ&citation_for_view=ZrqdSu4AAAAJ:3fE2CSJIrl8C)
- [2] [Convolutional Recurrent Neural Networks for Music Classification](https://scholar.google.co.kr/citations?view_op=view_citation&hl=en&user=ZrqdSu4AAAAJ&sortby=pubdate&citation_for_view=ZrqdSu4AAAAJ:ULOm3_A8WrAC)
- [3] [Magenta Home page](https://github.com/magenta/magenta)
- [4] [Magenta Music VAE directory](https://github.com/magenta/magenta/tree/main/magenta/models/music_vae)
- [5] [MusicVAE on GitHub](https://github.com/Variational-Autoencoder/MusicVAE)
- [6] [GTZAN dataset](http://marsyas.info/downloads/datasets.html)
- [7] [Rethinking CNN Models for Audio Classification](https://arxiv.org/pdf/2007.11154.pdf)