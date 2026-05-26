# CHESS: Chebyshev Spectral Synthesis for Trajectory Condensation

This is the official implementation of the **ICML 2026** paper: "**CHESS: Chebyshev Spectral Synthesis for Trajectory Condensation.**"

---

## 📢 Project Status

> **[May 2026 Update]**
>
> The author is currently heavily occupied with the preparation of the **Journal Extension** for CHESS and new submissions.
>
> As a result, the codebase is being cleaned and will be released in stages. Responses to Issues and Pull Requests may be delayed during this high-productivity period. Thank you for your patience and interest!

If you have any questions, please feel free to reach out via email **wrt15399487329sdu@163.com**, or via WeChat **15399487329**.

---

## 📖 Introduction

**CHESS** (Chebyshev Spectral Synthesis for Trajectory Condensation) introduces a novel framework that leverages Chebyshev polynomials to synthesize model training trajectories.

Departing from conventional discrete pixel-wise optimization, CHESS shifts the synthesis process from discrete samples to underlying continuous-time signal trajectories. By jointly enforcing **low-rank spatial coherence** and **piecewise Chebyshev polynomial temporal parameterization**, CHESS constrains synthesis to a physically meaningful function manifold, achieving:

- **Extreme compression** — up to **133× per synthetic sample** on high-sampling-rate sensor signals.
- **Cross-architecture generalization** — distilled data transfers reliably across CNN, lightweight, and Transformer backbones.
- **Zero-shot resolution adaptation** — a single distilled set can be analytically resampled to arbitrary sequence lengths without redistillation.

---

## 📦 Code Release

We have currently released the **full code for the ActR dataset** in this repository, including the complete CHESS pipeline (low-rank decomposition, piecewise Chebyshev fitting, manifold-aware initialization, and evaluation scripts).

Code and configurations for the **remaining datasets** — MeR, FacT, NTU-Fi, PAMAP2, and UCI-HAR — will be released after the **Journal Extension** is finalized.

If you urgently need access to the unreleased portions for academic purposes, please feel free to reach out via the email or WeChat above.

### Release Roadmap

- [x] **ActR** — full pipeline released

---

## 📂 Dataset Preparation

Download the **ActR dataset**:

🔗 **[Google Drive — ActR Dataset](https://drive.google.com/drive/folders/1xsHIOvZgS9VQPpZHvlp_sad1wqBP6JVr?usp=sharing)**

After downloading, extract the files into the corresponding data directory:

```bash
mkdir -p ./data/actr
# extract the downloaded archive into ./data/actr/
```

For NTU-Fi, PAMAP2, and UCI-HAR, you may obtain them directly from their official public sources (see Appendix D.2 of the paper for links).

---

## 🚀 Getting Started

### Step 1 — Prepare the Expert Model

You have **two options**:

#### Option 1 — Train from scratch

```bash
python pretrain.py --reproduce
```

#### Option 2 — Download our pre-trained weights

🔗 **[Google Drive — Pre-trained Expert Weights](https://drive.google.com/drive/folders/18-ocCvcvRPwUw0E_ZynAaYFnOtvEM1cn?usp=sharing)**

After downloading, place the weight files into the corresponding checkpoint directory.

### Step 2 — Run Distillation

```bash
python chess_actr.py --ipc 1 --reproduce
```

The `--ipc` argument controls the number of synthetic samples per class (SPC). You can vary it (e.g., `--ipc 5`, `--ipc 10`) to reproduce the results reported in Table 1 of the paper.

---

## 📬 Contact

- **Email:** wrt15399487329sdu@163.com
- **WeChat:** 15399487329

---

## 📝 Citation

If you find CHESS useful for your research, please consider citing our work:

```bibtex
@inproceedings{wu2026chess,
  title={CHESS: Chebyshev Spectral Synthesis for Trajectory Condensation},
  author={Wu, Ruituo and Zhang, Hongyu and Wang, Qiang and Du, Jiawei and Cui, Wei and Zhu, Ce and Li, Bing},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year={2026}
}
```
