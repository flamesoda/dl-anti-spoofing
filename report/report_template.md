# Voice Anti-spoofing with Light CNN

> Replace every bracketed placeholder with experimental evidence. Export
> plots from WandB as CSV and redraw them; do not paste dashboard screenshots.

## 1. Introduction

[Introduce audio deepfake detection, its relevance to automatic speaker
verification, and the objective of the ASVspoof 2019 LA countermeasure task.]

## 2. Task and methodology

[Define bona fide and spoof trials, explain that LA attacks are produced by
TTS/VC systems, and describe the absence of speaker overlap across splits.]

[Define false rejection rate, false acceptance rate, and pooled EER. State
that scores are continuous, larger values support bona fide, and EER is
reported in percent.]

[Describe the Blackman log-power STFT and the LCNN architecture. Explain MFM
2/1 and why Cross-Entropy was selected instead of A-Softmax based on the
comparative study. Include an original architecture diagram if useful.]

## 3. Experimental setup

- Dataset: ASVspoof 2019 LA
- Train trials: 25,380
- Development trials: 24,844
- Evaluation trials: 71,237
- Input: 863 x 600 log-power spectrogram
- Optimizer: Adam
- Initial learning rate: 3e-4
- Schedule: multiply by 0.5 every 10 epochs
- Batch size: [value]
- Sampling: equal numbers of bona fide and spoof trials in every train batch
- Epochs/steps: [value]
- Random seed: [value]
- Hardware and training time: [value]

[Describe the one-batch test, checkpoint selection by development EER, and
WandB logging. State explicitly that evaluation labels were not used for model
selection.]

## 4. Results

![Training and development loss](path/to/loss_plot.png)

![Development EER](path/to/eer_plot.png)

| Experiment | Dev EER, % | Eval EER, % | Notes |
|---|---:|---:|---|
| One-batch test | [value] | - | [final loss/accuracy] |
| STFT-LCNN-CE | [value] | [value] | [seed/checkpoint] |

[Interpret convergence and compare the result with the STC and comparative
papers. Discuss difficult attacks or unsuccessful changes only when supported
by recorded experiments.]

## 5. Conclusion

[Summarize what was learned, the main engineering/modeling difficulties,
whether the target EER was reached, and the most credible next improvement.]
