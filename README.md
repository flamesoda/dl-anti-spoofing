# Voice anti-spoofing with Light CNN

This repository implements a countermeasure for the Logical Access (LA)
partition of ASVspoof 2019. It is built directly on the course PyTorch Project
Template and keeps its Hydra configuration, trainer, checkpointing, and WandB
integration.

The system is an independent implementation based on the papers listed in
[References](#references). No external LCNN implementation or pretrained model
is used.

## Method

The input waveform is converted to a raw log-power spectrogram with the FFT
front end described by the Speech Technology Center (STC):

- 16 kHz mono audio;
- 1724-point FFT and 1724-sample periodic Blackman window;
- 0.0081 s hop, rounded to 130 samples;
- 863 one-sided frequency bins;
- the first 600 frames are retained and shorter inputs are zero-padded;
- no speech activity detection and no feature normalization.

The classifier follows the STC LCNN topology. Every convolution produces two
competing feature maps per output channel and an MFM 2/1 activation retains
their element-wise maximum. The convolutional stack is:

```text
Conv5x5-MFM(32) -> Pool
Conv1x1-MFM(32) -> BN -> Conv3x3-MFM(48) -> Pool -> BN
Conv1x1-MFM(48) -> BN -> Conv3x3-MFM(64) -> Pool
Conv1x1-MFM(64) -> BN -> Conv3x3-MFM(32) -> BN
Conv1x1-MFM(32) -> BN -> Conv3x3-MFM(32) -> Pool
Flatten -> FC(160) -> MFM(80) -> Dropout(0.75) -> BN -> FC(2)
```

The final dropout is deliberately placed before the last BatchNorm, as required
by the homework. The model has 10,198,818 trainable parameters for a
`1 x 863 x 600` input. Same-padding convolutions resolve the inconsistent
spatial sizes printed in Table 1 of the STC paper.

Training uses cross-entropy instead of A-Softmax. The comparative study found
ordinary sigmoid/cross-entropy competitive with margin-based objectives and
showed that variation between random seeds can exceed the difference between
losses. The default optimizer is Adam with learning rate `3e-4`; the rate is
halved every ten epochs. Every train mini-batch contains the same number of
bona fide and spoof trials; the development and evaluation protocols are never
resampled.

Labels are `spoof=0` and `bonafide=1`. The score used for EER and submission is

```text
bonafide_logit - spoof_logit
```

so a larger value always supports the bona fide hypothesis.

## Repository structure

```text
src/datasets/asvspoof2019.py       protocol parser and FLAC loading
src/transforms/stft.py             Blackman log-power STFT
src/model/lcnn.py                  MFM and STC Light CNN
src/metrics/eer.py                 official pooled EER implementation
src/trainer/                       training, epoch EER, inference and CSV export
src/configs/lcnn*.yaml             train, one-batch and inference configs
scripts/validate_submission.py     grading-format and EER check
scripts/plot_training_history.py   report-ready plots from WandB CSV
tests/                              deterministic unit tests
```

## Installation

Python 3.10 is recommended.

```bash
python3 -m venv project_env
source project_env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pre-commit install
```

Do not put API keys or private GitHub tokens in source files. Authenticate
WandB interactively or through the `WANDB_API_KEY` secret provided by Kaggle or
the execution environment.

## Dataset

Set `ASVSPOOF_LA_ROOT` to the directory containing the official LA split and
protocol directories:

```text
ASVspoof2019_LA_root/
├── ASVspoof2019_LA_train/flac/*.flac
├── ASVspoof2019_LA_dev/flac/*.flac
├── ASVspoof2019_LA_eval/flac/*.flac
└── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019.LA.cm.train.trn.txt
    ├── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2019.LA.cm.eval.trl.txt
```

```bash
export ASVSPOOF_LA_ROOT=/absolute/path/to/ASVspoof2019_LA_root
```

The dataset class indexes only protocol entries. This avoids silently scoring
extra FLAC files included in some redistributed archives. Expected protocol
sizes are 25,380 train, 24,844 development, and 71,237 evaluation trials.

On Kaggle, point the variable at the attached input directory. It can also be
overridden directly through Hydra:

```bash
python train.py -cn=lcnn \
  datasets.train.root_dir=/kaggle/input/.../LA \
  datasets.dev.root_dir=/kaggle/input/.../LA
```

## Tests

Run deterministic unit tests before starting an expensive experiment:

```bash
python -m unittest discover -s tests -v
```

The tests verify the official EER convention, MFM channel pairing, STFT shape,
score direction, and the complete LCNN forward pass.

## One-batch overfitting test

The one-batch config selects four bona fide and four spoof development samples
and uses the same fixed batch for training and evaluation:

```bash
python train.py -cn=lcnn_onebatch \
  writer.run_name=lcnn_onebatch_seed1 \
  trainer.seed=1
```

The loss should approach zero and accuracy should approach 1.0. EER is useful
as an additional check but is very coarse for only eight trials. Passing this
test does not establish evaluation quality; failing it indicates a pipeline or
optimization problem.

## Full training

Authenticate WandB and run:

```bash
python train.py -cn=lcnn \
  writer.run_name=lcnn_stft_ce_seed1 \
  trainer.seed=1
```

Useful overrides include:

```bash
# Smaller batch when GPU memory is limited
python train.py -cn=lcnn dataloader.batch_size=4

# Offline logging
python train.py -cn=lcnn writer.mode=offline

# Resume an interrupted named run
python train.py -cn=lcnn \
  writer.run_name=lcnn_stft_ce_seed1 \
  trainer.resume_from=checkpoint-epoch10.pth
```

The trainer logs `loss_train`, `Accuracy_train`, `loss_dev`, `Accuracy_dev`,
and `EER_dev`. `EER_dev` is computed once from all development scores, not as
an invalid average of per-batch EERs. The best checkpoint is selected with
`min dev_EER` and saved to:

```text
saved/<run_name>/model_best.pth
```

Do not choose checkpoints or hyperparameters using evaluation labels. Because
the comparative study found substantial seed sensitivity, report the seed and,
when compute permits, run more than one seed.

## Evaluation inference and submission

Run the best development-selected checkpoint over the evaluation partition:

```bash
python inference.py -cn=lcnn_inference \
  inferencer.from_pretrained=/absolute/path/to/model_best.pth \
  inferencer.submission_filename=your_university_username.csv
```

The resulting file is:

```text
data/saved/submissions/your_university_username.csv
```

It has no header and contains exactly two comma-separated fields per row:

```text
LA_E_2834763,-1.238475
LA_E_8877452,2.193841
```

Use the official university username in the filename. Validate the file against
the evaluation protocol before submission:

```bash
python scripts/validate_submission.py \
  data/saved/submissions/your_university_username.csv \
  "$ASVSPOOF_LA_ROOT/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
```

The script enforces unique keys, finite soft scores, exact protocol coverage,
and reports EER with the same discrete implementation as the course grader.
EER is printed in percent on the 0--100 scale. The full-performance target is
below 5.3%.

## Report plots

Export the WandB run history as CSV and generate figures rather than using
screenshots:

```bash
python scripts/plot_training_history.py wandb_history.csv \
  --output training_history.png
```

If WandB changes exported column names, pass `--train-loss`, `--dev-loss`, or
`--dev-eer` explicitly. The report should describe the task, EER, LCNN/MFM,
experimental settings, one-batch test, development trajectory, final evaluation
result, limitations, and conclusions.

## References

1. Lavrentyeva et al., *STC Antispoofing Systems for the ASVspoof2019
   Challenge*, Interspeech 2019.
2. Wu et al., *A Light CNN for Deep Face Representation with Noisy Labels*,
   IEEE TIFS 2018.
3. Wang and Yamagishi, *A Comparative Study on Recent Neural Spoofing
   Countermeasures for Synthetic Speech Detection*, Interspeech 2021.
4. ASVspoof consortium, *ASVspoof 2019 Evaluation Plan*.

## Template attribution

This project is based on the
[PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template)
used in the HSE DLA course. Its Hydra, logging, trainer, and checkpointing
structure has been retained and extended for voice anti-spoofing.
