# ENFify

Welcome to **ENFify**, an open-source tool to detect digital audio tampering using the Electric Network Frequency signal in the background noise.

## Table of Contents
- [Description](#description)
- [Status](#status)
- [Installation](#installation-of-the-repository)
- [Example Files](#example-files)
- [Testing](#testing)
- [Configuration](#configuration)
- [Training](#training)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Description
The **ENFify** project develops an open-source tool for analyzing audio recordings using the Electric Network Frequency (ENF) signal contained in the audio. This tool enables users to distinguish between authentic (uncut) and tampered (cut) audio files. By analysing the phase of the ENF signal, the tool helps detecting audio manipulations based on the background noise of the ENF.

## Status
The first version of **ENFify** has been completed. The tool can classify audio files with little noise. The goal of the first version has thus been achieved, as it focused on AI classification using a neural network.

The tool includes functions for better denoising using Variational Modal Decomposition (VMD) and a Robust Filtering Algorithm (RFA). However, these are only at an experimental stage and still require individual configuration.

## Installation of the Repository
Step-by-step instructions on how to get the development environment running.

- **Clone the repository**
```bash
git clone https://github.com/Licates/ENFify.git
```

- **Navigate to the project directory**
```bash
cd ENFify
```

- **Install Poetry if missing**

Look [here](https://python-poetry.org/docs/) for Poetry installation details.

- **Create and activate the Poetry environment**
```bash
poetry install
poetry shell
```

- **Now you should be able to run**
```bash
which enfify
enfify --help
```

## Example Files
The tool contains commands for generating sample audio files. When the corresponding command is executed, an authentic and a tampered audio file are created in the current directory, as well as a file with information about the cut in the tampered file, in case this information is of interest. If you like, you can create a new directory for the files.
```bash
mkdir ~/example_files_enfify
cd ~/example_files_enfify
enfify example-whuref
enfify example-synthetic
```
The `whuref-*.wav` files there are sourced from the [ENF-WHU-Dataset](https://github.com/ghua-ac/ENF-WHU-Dataset/tree/78ed7f3784949f769f291fc1cb94acd10da6322f/ENF-WHU-Dataset/H1_ref). These files include power grid data and therefore have minimal noise.

The `synthetic-*.wav` files are generated by sampling a signal in the frequence band of the nominal ENF.

The tool was also tested on a real-world audio dataset named [Carioca](https://doi.org/10.1109/TIFS.2010.2051270) but we can not publish this data.

## Testing

To test the tool on a file, enter the following with the path to the audio file, e.g. `whuref_tamp.wav`:

```bash
enfify detect whuref-auth.wav
enfify detect whuref-tamp.wav
enfify detect synthetic-auth.wav
enfify detect synthetic-tamp.wav
```

This classifies the file in the command line and in the default configuration also generates a report with the audio features for more insight in the current directory.

Feel free to experiment by adding manual cuts on the authentic file.

## Training
The models were trained on Kaggle. Copies of the notebooks used are located in the [training](training) folder. Additional exploratory notebooks can be found in the [notebooks](notebooks) folder, but some of the older ones are partially dependent on an earlier version of the repository.

## Configuration

There are two ways of configure the prompt for the tool.

1. The first version with higher priority is to use the CLI arguments. They can be listed by running:

```bash
enfify detect --help
```

This is only for a limited set of configuration options.

2. All options can be configured by using a config file:

```bash
enfify detect <path/to/autio_file> --config-file <path/to/config_file>
```

You can generate an example config file in the current directory with:

```bash
enfify example-config
```

Any missing line of the config file will default to the settings in [default.yml](config/default.yml).

## Contributing
We welcome contributions from the community! Whether you're looking to fix bugs, add features, or improve documentation, your help is greatly appreciated in making **ENFify** even better.

One exciting area for potential contributions is the denoising functionality. There are opportunities to further enhance the implementation of the Variational Modal Decomposition (VMD) and Robust Filtering Algorithm (RFA) methods for even better results on audio recordings with noise. If you have expertise in audio processing your insights could help take this aspect of the tool to the next level.

**How to contribute:**
1. Fork the repository and create your own branch for changes.
2. Submit a pull request with a description of your changes.
3. Make sure your code is well-documented.

We're excited to collaborate with you!

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute the code as long as you include the original copyright and license notice in any copies or substantial portions of the software. For more details, please refer to the [LICENSE](LICENSE) file.

## References
Several academic papers and resources have been considered in the development of this project. Here are some of the key references:

1. **1D-CNN-based audio tampering detection using ENF signals** \
*Author(s): Haifeng Zhao, Yanming Ye, Xingfa Shen, Lili Liu \
Journal: Scientific Reports \
DOI: https://doi.org/10.1038/s41598-024-60813-0*

2. **Audio Authenticity: Detecting ENF Discontinuity With High Precision Phase Analysis** \
*Author(s): D. P. Nicolalde Rodriguez, J. A. Apolinario and L. W. P. Biscainho \
Journal: IEEE Transactions on Information Forensics and Security \
DOI: https://doi.org/10.1109/TIFS.2010.2051270*

We would like to thank the authors of these papers for their valuable contributions, which served as a foundation for our implementation.

--------
