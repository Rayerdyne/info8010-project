# INFO8010: Deep learning Project
## Latent space manipulation on music sequences

### Objectives
- Build a latent space representation for short sequences of music
- Use that latent space to:
  - interpolate between two existing sequences to create new ones
  - tweak existing sequence to max out some dimension to understand what is does

### General architecture

**Pre-processing**:

1. Cropping, to start of a sequence of a compatible lenght for the computation. (zero-padding is avoided, as it would add artificial abrupt, unwanted ending to the music)

1. Mel-spectrograms, following the example of [1] and [2]. To benefit from [7], three are performed and stacked.
   - Input: waveform
   - Output: tensor (3-channel image)

1. Data augmentation: spectrogram being kind of images, we can simply take advantage of image transforms !

We will randomly apply following transformations:
   - Random color jitter
   - Gaussian Blur
   - Random erasing (cf [8])

**Encoder**:

The music signal will go through the following steps:
1. CNN layers, as proposed in [1] and [2] as well. 
   ResNet34 pre-trained model, but not the last layers (last convolution and dense layer)
   - Input: tensor (image)
   - Output: tensor, time-dependent and high-dimensional representation of input-sequence
1. LSTM, that is, a deep learning tools for efficient manipulation of sequential data.
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
- [8] [Random Erasing Data Augmentation](https://doi.org/10.48550/arXiv.1708.04896)