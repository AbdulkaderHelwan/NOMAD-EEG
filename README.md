# NOMAD-EEG

**Native Omni-Montage Adaptive Diffusion.** One EEG foundation model for any electrode montage.

[Paper (arXiv:2610.35246)](https://arxiv.org/abs/2610.35246) · [Project page](https://abdulkaderhelwan.github.io/NOMAD-EEG/) · [Pre-trained weights](https://github.com/AbdulkaderHelwan/NOMAD-EEG/releases)

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

The [Releases](https://github.com/AbdulkaderHelwan/NOMAD-EEG/releases) page hosts the EMA weights of
the backbone used for every result above (diffusion, 12.0 M parameters, about 74 epochs of
pre-training on TUEV). Training and evaluation code will follow.

## Citation

```bibtex
@article{helwan2026nomadeeg,
  title   = {{NOMAD-EEG}: Native Omni-Montage Adaptive Diffusion},
  author  = {Helwan, Abdulkader and Henni, Khadidja and Abou-Abbas, Lina},
  journal = {arXiv preprint arXiv:2610.35246},
  year    = {2026}
}
```
