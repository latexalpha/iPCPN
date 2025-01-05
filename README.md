# README

Source code for the paper "Integral-form physics constrained parallel network for identifying nonlinear dynamical systems from only noisy displacement measurements".

## Requirements

The code is written in Python 3.10.15, and the required packages are listed in `requirements.txt`. You can install them using the following command:

```bash
pip install -r requirements.txt
```

All the hyperparameters used in the experiments are stored in the `config.py` file.

## Data

The data used in the experiments are stored in the `data` folder, including two noise types and three different noise levels. 

- 25 dB (approximately) pink Gaussian noise: `data/nonautonomous_duffing_0.11.mat`
- 20 dB (approximately) pink Gaussian noise: `data/nonautonomous_duffing_0.19.mat`
- 25 dB white Gaussian noise: `data/nonautonomous_duffing_25_white.mat`
- 20 dB white Gaussian noise: `data/nonautonomous_duffing_20_white.mat`
- 15 dB white Gaussian noise: `data/nonautonomous_duffing_15_white.mat`

## Training

To train the model, you can run the following command:

```bash
python run.py
```

The trained model will be saved in the `models` subfolder of `outputs` folder.

The visualization of the training process can be found in the `figures` subfolder of the `outputs` folder.

The output of the model will be saved in the `results` subfolder of the `outputs` folder.
