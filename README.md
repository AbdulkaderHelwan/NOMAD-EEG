# NOMAD-EEG

**Native Omni-Montage Adaptive Diffusion.** One EEG foundation model for any electrode montage.

[Paper (arXiv:2610.35246)](https://arxiv.org/abs/2610.35246) · [Project page](https://abdulkaderhelwan.github.io/NOMAD-EEG/) · [Pre-trained weights](https://lauedu74602-my.sharepoint.com/:f:/g/personal/abedelkader_helwan_lau_edu_lb/IgBQJQX6xN9QT73Qjxc8KeG7AbqwUOgx-9PocZmKWBN2T4Y?e=wa1ipy)

NOMAD-EEG reads EEG with any number of electrodes, in any order and of any length. It identifies
each electrode only by its 3D scalp position. It is a 12.0 M-parameter diffusion model that alternates
S4D state-space layers along time (no patches) with attention across electrodes. It was pre-trained on
147 h of unlabelled EEG from a single corpus (TUEV).

The model was evaluated with frozen linear probes on ten downstream datasets with 1 to 64 electrodes.
It was compared with seven EEG foundation models (EEGDM, REVE, CBraMod, LaBraM, BIOT, EEGMamba and
DiffEEG), all run by us under one protocol.

## Highlights

| Task | Metric | NOMAD-EEG | Next best |
|---|---|---|---|
| EEGMMIDB motor imagery (4 classes, 64 electrodes) | Balanced accuracy | **0.442** | DiffEEG 0.418 |
| Siena seizure detection | Macro F1 | **0.622** | REVE 0.562 |
| Bonn seizure detection (single channel) | AUROC | **0.999** | REVE 0.998 |
| TUEP epilepsy diagnosis (200 patients) | Patient accuracy | **0.640** | BIOT 0.630 |
| TUEV event classification, fine-tuned | Cohen κ / weighted F1 | **0.635 / 0.814** | EEGDM 0.568 / 0.783 (frozen) |
| BCIAUT-P300, autism, unseen 8-electrode cap | AUROC | **0.718** | EEGDM 0.667 |

On Mumtaz, the mean AUROC of a probe trained on all 19 electrodes falls only from 0.942 to 0.891
when it is given 5 random electrodes. Listing the electrodes in a different order changes no
prediction.

## Results (frozen linear probe)

Primary metric per dataset: AUROC for binary tasks, balanced accuracy for multi-class tasks. 95%
bootstrap CIs and all other metrics are on the [project page](https://abdulkaderhelwan.github.io/NOMAD-EEG/#results).
† = the model's pre-training data included this dataset or the corpus it comes from.

| Model | TUAB | TUSZ | TUEV | Siena | Bonn | Mumtaz | EEGMMIDB | Sleep-EDF |
|---|---|---|---|---|---|---|---|---|
| **NOMAD-EEG** | 0.838 | 0.866 | 0.542† | 0.882 | **0.999** | 0.942 | **0.442** | 0.757 |
| EEGDM | 0.869 | 0.874 | **0.647**† | **0.915** | 0.994 | 0.946 | 0.347 | **0.779** |
| REVE | **0.870** | **0.879**† | 0.565 | 0.906† | 0.998 | **0.960** | 0.371 | 0.760 |
| CBraMod | 0.808† | 0.855† | 0.466† | 0.850 | 0.996 | 0.860 | 0.308 | 0.725 |
| LaBraM | 0.788 | 0.845† | 0.521 | 0.797† | 0.995 | 0.904 | 0.302† | 0.738 |
| BIOT | 0.847† | 0.760 | 0.519† | 0.707 | 0.979 | 0.928 | 0.274 | 0.741 |
| EEGMamba | 0.746† | 0.836† | 0.520† | 0.855† | 0.948 | 0.867 | 0.304 | 0.722 |
| DiffEEG | 0.760 | 0.813† | 0.329 | 0.777 | 0.982 | 0.901 | 0.418 | 0.715 |

## Pre-trained weights

[`nomad_eeg_base.pt`](https://lauedu74602-my.sharepoint.com/:f:/g/personal/abedelkader_helwan_lau_edu_lb/IgBQJQX6xN9QT73Qjxc8KeG7AbqwUOgx-9PocZmKWBN2T4Y?e=wa1ipy) holds the EMA weights
and configuration of the backbone used for every result above (diffusion, 12.0 M parameters, about
74 epochs of pre-training on TUEV). Download it from the [OneDrive folder](https://lauedu74602-my.sharepoint.com/:f:/g/personal/abedelkader_helwan_lau_edu_lb/IgBQJQX6xN9QT73Qjxc8KeG7AbqwUOgx-9PocZmKWBN2T4Y?e=wa1ipy) and pass its path to `nomad_eeg.load_pretrained()`.

## Installation

```bash
git clone https://github.com/AbdulkaderHelwan/NOMAD-EEG.git
cd NOMAD-EEG
pip install -e .            # torch, numpy, scipy, scikit-learn, mne
python tests/test_model.py  # CPU checks: any montage, electrode-order invariance, classifier head
```

## Quick start

```python
import numpy as np
import nomad_eeg

model = nomad_eeg.load_pretrained("nomad_eeg_base.pt", device="cuda")   # the downloaded weights

channels = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]
x_train_uv = ...                                         # [N, 19, T] windows in microvolts at 256 Hz

x = nomad_eeg.preprocess(x_train_uv, fs=256)             # -> 200 Hz
scaler = nomad_eeg.RobustScaler().fit(x)                 # median / IQR from TRAINING data only
feats = nomad_eeg.extract_features(model, scaler.transform(x), channels)   # [N, 10240]
```

Any number of electrodes works, in any order, as long as each has a standard 10-20 / 10-05
name. A bipolar channel is written `"FPZ-CZ"` and placed at the midpoint of its two
electrodes. Drop channels without a scalp position (ECG, EMG) first. `examples/quickstart.py`
embeds a 19-electrode, an 8-electrode and a 2-channel bipolar montage with the same weights.

## Linear probing, fine-tuning and testing on your data

**1. Prepare a dataset folder.** Save subject-disjoint splits as `raw/train.npz`, `raw/val.npz`
and `raw/test.npz`. Each holds `X` [N, E, T] in microvolts, `y` [N] integer labels and,
optionally, `subjects` [N]. Then run:

```bash
python scripts/prepare_dataset.py --raw raw/ --out data/my_task --fs 256 \
    --channels FP1,FP2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8,T3,T4,T5,T6,FZ,CZ,PZ \
    --classes healthy,patient --bandpass 0.3 75 --notch 50
```

This filters the signals, resamples them to 200 Hz, robust-scales each channel with training
statistics (clipped at ±20) and writes `data/my_task/` with a `meta.json`.

**2. Linear probe (the paper's main protocol).**

```bash
python scripts/extract_features.py --data data/my_task --out features/my_task
python scripts/linear_probe.py --features features/my_task
```

This standardises the features and fits a class-balanced logistic regression on train. C is
chosen from 1e-6 to 1e3 on validation, by AUROC for binary tasks and balanced accuracy for
multi-class tasks. The test split is scored once, with 95% bootstrap CIs. The output goes to
`features/my_task/probe.json`.

**3. Fine-tune end to end.**

```bash
python scripts/finetune.py --data data/my_task --out runs/my_task
```

This trains 2 epochs with the encoder frozen, then 20 epochs end to end. It uses the
attention-pooling head from the paper, with the encoder learning rate 10× lower than the head's.
The epoch with the best validation balanced accuracy is kept as `runs/my_task/best.pt`, and
test metrics are written to `runs/my_task/test.json`.

**4. Test a fine-tuned model** on any split or on another dataset folder (it may use a
different montage):

```bash
python scripts/evaluate.py --data data/other_site --checkpoint runs/my_task/best.pt --split test
```

## Repository layout

| Path | Contents |
|---|---|
| `nomad_eeg/model.py` | the backbone (S4D time mixing, electrode attention, 3D coordinates) |
| `nomad_eeg/classifier.py` | the fine-tuning head |
| `nomad_eeg/preprocessing.py` | resampling, filtering and robust scaling |
| `nomad_eeg/features.py`, `hub.py` | frozen embeddings, weight loading |
| `scripts/` | prepare_dataset, extract_features, linear_probe, finetune, evaluate |

## Citation

```bibtex
@article{helwan2026nomadeeg,
  title   = {{NOMAD-EEG}: Native Omni-Montage Adaptive Diffusion},
  author  = {Helwan, Abdulkader and Henni, Khadidja and Abou-Abbas, Lina},
  journal = {arXiv preprint arXiv:2610.35246},
  year    = {2026}
}
```
