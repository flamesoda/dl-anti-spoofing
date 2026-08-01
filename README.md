# LightCNN for ASVspoof 2019

This repository contains my solution for the voice anti-spoofing homework. The
task is to distinguish bona fide speech from spoofed speech on the Logical
Access partition of ASVspoof 2019.

The project is based on the course PyTorch template. I kept its Hydra
configuration, trainer, checkpointing and WandB logging, and added the dataset,
STFT front-end, LightCNN model, EER metric and submission code needed for this
task.

## Model

Audio is converted to a log-power spectrogram before it is passed to the
network. The front-end uses the settings from the STC paper:

- 16 kHz mono audio;
- a 1724-sample periodic Blackman window;
- 1724-point FFT;
- 130-sample hop;
- 863 frequency bins and at most 600 frames.

Recordings that are shorter than the required input length are repeated instead
of being padded with zeros. Longer recordings are cropped to the first 600
frames.

The classifier is an LCNN with Max-Feature-Map activations. Its convolutional
part is followed by a 160-unit fully connected layer, MFM, dropout, BatchNorm
and a two-class output layer. The dropout layer is placed before the final
BatchNorm as required by the assignment. The complete model has 10,198,818
trainable parameters.

I used cross-entropy loss rather than A-Softmax. Training batches are balanced:
half of every batch is bona fide and half is spoof. Development and evaluation
data are not resampled.

Labels are encoded as `spoof = 0` and `bonafide = 1`. The score written to the
submission is:

```text
bonafide_logit - spoof_logit
```

## Project structure

```text
src/datasets/asvspoof2019.py   ASVspoof protocol parsing and audio loading
src/transforms/stft.py         log-power STFT front-end
src/model/lcnn.py              LightCNN and MFM layers
src/metrics/eer.py             pooled EER calculation
src/trainer/                   training and inference loops
src/configs/lcnn.yaml          training configuration
src/configs/lcnn_inference.yaml
scripts/validate_submission.py CSV format and EER check
train.py                       training entry point
inference.py                   evaluation entry point
```

## Installation

Python 3.10 is recommended.

```bash
python3 -m venv project_env
source project_env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

WandB authentication can be done interactively with `wandb login` or through a
`WANDB_API_KEY` environment variable. API keys and GitHub tokens should not be
stored in the repository.

## Dataset

Set `ASVSPOOF_LA_ROOT` to the directory that contains the three LA partitions
and the protocol directory:

```text
ASVspoof2019_LA/
├── ASVspoof2019_LA_train/flac/
├── ASVspoof2019_LA_dev/flac/
├── ASVspoof2019_LA_eval/flac/
└── ASVspoof2019_LA_cm_protocols/
```

For example:

```bash
export ASVSPOOF_LA_ROOT=/absolute/path/to/ASVspoof2019_LA
```

The code reads file names from the official protocols. The expected split
sizes are 25,380 train files, 24,844 development files and 71,237 evaluation
files.

## Training

This is the command used for the final 20-epoch Kaggle run:

```bash
python train.py -cn=lcnn \
  writer.run_name=lcnn_stft_ce_repeat_seed1_full20 \
  writer.project_name=voice-anti-spoofing \
  writer.mode=online \
  writer.log_checkpoints=false \
  trainer.seed=1 \
  trainer.n_epochs=20 \
  trainer.save_period=1 \
  trainer.use_amp=true \
  dataloader.batch_size=8 \
  dataloader.num_workers=4 \
  dataloader.pin_memory=true \
  dataloader.persistent_workers=true
```

The default optimizer is Adam with a learning rate of `3e-4`. StepLR halves
the learning rate every ten epochs. The trainer saves checkpoints under
`saved/<run_name>/` and selects `model_best.pth` using development EER.

## Evaluation and submission

Run inference with the checkpoint selected on the development set:

```bash
python inference.py -cn=lcnn_inference \
  inferencer.from_pretrained=/absolute/path/to/model_best.pth \
  inferencer.submission_filename=your_university_username.csv \
  dataloader.batch_size=8 \
  dataloader.num_workers=4
```

The CSV is written to `data/saved/submissions/`. It has no header and contains
one score for each evaluation trial:

```text
LA_E_2834763,-1.238475
LA_E_8877452,2.193841
```

Before submission, the file can be checked against the evaluation protocol:

```bash
python -m scripts.validate_submission \
  data/saved/submissions/your_university_username.csv \
  "$ASVSPOOF_LA_ROOT/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
```

The final model produced 71,237 scores and achieved an evaluation EER of
**6.418526%**.

## References

1. Lavrentyeva et al., *STC Antispoofing Systems for the ASVspoof2019
   Challenge*, Interspeech 2019.
2. Wu et al., *A Light CNN for Deep Face Representation with Noisy Labels*,
   IEEE TIFS 2018.
3. Wang and Yamagishi, *A Comparative Study on Recent Neural Spoofing
   Countermeasures for Synthetic Speech Detection*, Interspeech 2021.
4. ASVspoof consortium, *ASVspoof 2019 Evaluation Plan*.

This repository is a modified version of the
[PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template)
used in the HSE DLA course.
