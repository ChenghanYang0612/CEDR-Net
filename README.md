<div align="center">

# CEDR-Net: Change Evidence-guided Description Realization Network

**Review-stage repository for remote-sensing change captioning.**

</div>

---

## Release Status

This repository is a review-stage preview of CEDR-Net. CEDR-Net studies
change-evidence-guided description realization for bi-temporal remote-sensing
images, with the goal of reducing over-realization and under-realization in
change captions.

To protect the unpublished method during peer review, the complete model
implementation, full experiment configuration, trained checkpoints, datasets,
and experiment outputs are not included at this stage. In particular,
`mycode/model.py` and `mycode/config.py` are public placeholders rather than the
complete research implementation.

Currently included:

- Dataset loading code
- Prompt construction and caption decoding utilities
- Two-stage training orchestration code
- Evaluation code and captioning metrics
- Dependency list
- Expected dataset and checkpoint directory layouts
- Placeholders for the withheld model and full configuration

Because the complete model implementation and experiment configuration are
withheld during review, end-to-end training and evaluation are not runnable
from this preview repository.

The full release is planned to include:

- Complete CEDR-Net model implementation
- Complete training configurations and reproduction commands
- Trained checkpoints, when redistribution is permitted

## Overview

CEDR-Net is designed for remote-sensing image change captioning. Given two
images of the same scene acquired at different times, the model generates one
sentence describing whether and how the scene has changed.

The framework contains four main components:

1. A bi-temporal visual evidence encoder based on RemoteCLIP ViT-B/32.
2. A change evidence estimator learned with change/no-change weak supervision.
3. A utility-aware visual evidence adapter that conditions visual tokens on
   learned change evidence.
4. A utility-modulated frozen Qwen decoder for change-description realization.


## Repository Layout

```text
CEDR-Net/
  README.md
  requirements.txt
  LICENSE
  .gitignore
  mycode/
    __init__.py
    config.py                  # review-stage placeholder
    model.py                   # review-stage placeholder
    dataset.py
    prompting.py
    stage_executor.py
    train.py
    evaluate.py
    utils.py
  data/
    LEVIR-CC/
  checkpoint/
    remoteclip/
    Qwen/
      Qwen3-1.7B/
```

## LEVIR-CC Dataset

CEDR-Net is evaluated on
[LEVIR-CC](https://github.com/Chen-Yang-Liu/RSICC). The dataset is not
redistributed in this repository. Please obtain it from the original source
and follow its license and citation requirements.

Place the dataset under:

```text
data/LEVIR-CC/
  images/
    train/
      A/
      B/
    val/
      A/
      B/
    test/
      A/
      B/
  processed/
    official_rsicc_style/
```

`A` and `B` denote the two acquisition times. The released data loader expects
the processed HDF5, caption, caption-length, metadata, and vocabulary files in
`processed/official_rsicc_style/`. The complete release will provide detailed
preprocessing instructions.

## Installation

Create a Python environment and install the dependencies:

```bash
conda create -n cedrnet python=3.10 -y
conda activate cedrnet

# Install a PyTorch build compatible with the local CUDA environment first.
pip install -r requirements.txt
```

The final release will record the exact Python, PyTorch, CUDA, Transformers,
and GPU environment used for the paper experiments.

## Checkpoints and Backbone Weights

The following files are expected by the complete implementation:

```text
checkpoint/
  remoteclip/
    RemoteCLIP-ViT-B-32.pt
  Qwen/
    Qwen3-1.7B/
```

- RemoteCLIP weights should be obtained from the
  [official RemoteCLIP repository](https://github.com/ChenDelong1999/RemoteCLIP).
- Qwen weights should be obtained from the official Qwen distribution or its
  Hugging Face model page.
- CEDR-Net checkpoints are not included during review. The default two-stage
  workflow trains and selects its binary checkpoint inside the experiment
  output directory before starting caption training. A separately pretrained
  binary checkpoint is only needed for the optional caption-only workflow.

No third-party backbone or language-model weights are redistributed by this
repository.

## Acknowledgements

We gratefully acknowledge the following projects:

- [LEVIR-CC / RSICC](https://github.com/Chen-Yang-Liu/RSICC)
- [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [Qwen](https://github.com/QwenLM/Qwen)
- [PyTorch](https://pytorch.org/)
- [pycocoevalcap](https://github.com/salaniz/pycocoevalcap)

## Citation

Citation information will be added after the paper is accepted or assigned a
stable public identifier.

## License

See `LICENSE`. Additional future release artifacts, datasets, pretrained
models, and third-party components may be governed by separate licenses and
terms.
