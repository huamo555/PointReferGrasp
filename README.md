# PointReferGrasp

**End-to-End Learning of Affordance-Grounded Grasp Detection from Single-View Point Clouds**

[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10.1-ee4c2c.svg)](https://pytorch.org/)
[![Project Status](https://img.shields.io/badge/status-release%20in%20progress-orange.svg)](#release-status)

PointReferGrasp is an end-to-end framework for grounding functional regions on 3D objects and estimating executable 6-DoF grasp poses from a single-view point cloud and a natural-language instruction.

> [!IMPORTANT]
> The **3D AffordanceGrasp dataset is not publicly available yet**. Dataset files, pretrained checkpoints, the complete 6-DoF grasp pipeline, and reproducible evaluation scripts will be released after they are finalized. Please do not request access through the obsolete dataset link that appeared in an earlier version of this README.

## Overview

Conventional affordance-grounded grasping systems commonly use a two-stage pipeline: first segment a functional component, then generate and filter grasp candidates. Treating an entire component as uniformly graspable can produce semantically correct but physically unstable poses, especially near edges and part connections.

PointReferGrasp instead performs point-wise affordance grounding and grasp estimation in a unified architecture. It predicts how suitable each visible point is for the requested interaction and uses this fine-grained affordance map to select functionally relevant 6-DoF grasp poses.

## Highlights

- **Single-stage affordance-grounded grasping:** jointly reasons about point-level affordance and 6-DoF grasp pose estimation.
- **Semantic Modulation Module (SMM):** injects global language semantics into point-cloud features using feature-wise modulation.
- **Hierarchical Semantic-Cascade Affordance Decoding (HSCAD):** progressively moves from coarse object-level localization to fine functional-region refinement.
- **Hierarchical Coarse-to-Fine Contrastive Learning (HCFCL):** uses coarse alignment and hard-negative mining to distinguish geometrically similar but functionally different regions.
- **3D AffordanceGrasp benchmark:** provides natural-language instructions, point-wise soft affordance labels, and point-wise 6-DoF grasp annotations.

## Method

Given a single-view point cloud $P \in \mathbb{R}^{N \times 3}$ and a natural-language instruction $T$, PointReferGrasp contains four main components:

1. **Linguistic-aware point encoding** extracts text features with RoBERTa and geometric features with PointNet++, followed by language-conditioned semantic modulation.
2. **Hierarchical affordance decoding** first localizes a coarse functional region and then refines point-wise affordance predictions inside that region.
3. **Coarse-to-fine contrastive learning** aligns visual and linguistic representations while explicitly suppressing hard negative regions.
4. **6-DoF grasp estimation** predicts approach directions and grasp parameters from local geometry, then filters candidate poses using the predicted affordance map.

The complete model is optimized with a multi-task objective combining hierarchical segmentation, contrastive, and grasp-estimation losses.

## 3D AffordanceGrasp Dataset

Each sample is represented as:

$$
\mathcal{D} = (P, A, T, G),
$$

where $P$ is a single-view object point cloud, $A$ is a point-wise soft affordance label, $T$ is a natural-language instruction, and $G$ contains point-wise 6-DoF grasp ground truths.

Dataset summary:

| Property | Value |
| --- | ---: |
| Instances | 8,837 |
| Object categories | 15 |
| Grasp-related affordances | 5 |
| Training instances | 7,070 |
| Test instances | 1,767 |

The five affordance types are `grasp`, `lift`, `move`, `open`, and `wrap grasp`. Instructions cover simple, intermediate, and complex descriptions generated from structured object-part-affordance triples.

### Dataset availability

The dataset is currently undergoing final organization and validation and is **not public**. When released, this section will contain:

- download links and checksums;
- the license and terms of use;
- directory structure and annotation specification;
- preprocessing and train/test split instructions.

## Release Status

This repository is an early code release. The currently committed code primarily covers the language-guided, point-wise affordance grounding component.

| Component | Status |
| --- | --- |
| Affordance-grounding model | Available |
| Training and grounding utilities | Available |
| 3D AffordanceGrasp dataset | Coming soon |
| Pretrained checkpoints | Coming soon |
| Complete 6-DoF grasp pipeline | Coming soon |
| Reproducible evaluation and robot demos | Coming soon |

## Repository Structure

```text
PointReferGrasp/
├── config/                  # Model and training configuration
├── data_utils/              # Dataset loading and preprocessing
├── model/                   # PointReferGrasp grounding modules
├── utils/                   # Losses, metrics, logging, and visualization
├── train_n.py               # Main grounding training entry point
├── train.py                 # Alternative training entry point
├── inference.py             # Inference utility
├── evalization.py           # Evaluation utility
└── requirements.txt         # Original development environment snapshot
```

## Installation

The original development environment used Python 3.8, PyTorch 1.10.1, torchvision 0.11.2, Transformers 4.30.2, and Open3D 0.16.0.

```bash
git clone https://github.com/huamo555/PointReferGrasp.git
cd PointReferGrasp

conda create -n pointrefergrasp python=3.8 -y
conda activate pointrefergrasp
```

> [!NOTE]
> `requirements.txt` is an environment snapshot and currently includes several machine-local Conda build references. A clean, reproducible environment specification will be provided with the complete release. CUDA extensions such as `pointnet2-cuda` must match the installed PyTorch and CUDA versions.

## Data Preparation

The current grounding loader expects split annotation files, object point clouds, and language instructions. Before training, set `data_root` in `data_utils/shapenetpart.py` to the local dataset directory.

The expected grounding files are:

```text
<data_root>/
├── anno_train.pkl
├── anno_val.pkl
├── anno_test.pkl
├── objects_train.pkl
├── objects_val.pkl
├── objects_test.pkl
└── Affordance-Question.csv
```

These files are part of the forthcoming dataset release and are not included in this repository.

## Training

After preparing the environment and dataset, start grounding training with:

```bash
python train_n.py \
  -v pointrefergrasp \
  -gpu 0 \
  --yaml config/default.yaml \
  --name PointReferGrasp
```

Training outputs are written under `runs/train/` by default. Configuration values such as batch size, learning rate, number of points, and embedding dimensions are defined in `config/default.yaml` and can be overridden by supported command-line arguments.

## Results

On the 3D AffordanceGrasp benchmark, PointReferGrasp improves both physical grasp quality and functional localization over the two-stage PIONEER baseline.

| Split | Method | AP ↑ | AP@0.8 ↑ | AP@0.4 ↑ | DAP (cm) ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| Seen | PIONEER | 50.57 | 56.46 | 45.53 | 4.78 |
| Seen | **PointReferGrasp** | **65.06** | **69.96** | **60.43** | **1.72** |
| Unseen | PIONEER | 46.36 | 51.97 | 43.46 | 5.67 |
| Unseen | **PointReferGrasp** | **60.11** | **66.62** | **56.17** | **2.34** |

Real-robot experiments achieved an average success rate of **91.0%** in single-object scenes and **88.5%** in multi-object scenes. Detailed protocols and evaluation scripts will be included in the complete release.

## Citation

If you find this project useful, please cite the manuscript:

```bibtex
@misc{gao2026pointrefergrasp,
  title  = {PointReferGrasp: End-to-End Learning of Affordance-Grounded Grasp Detection from Single-View Point Cloud},
  author = {Gao, Yuming and Wang, Lichun and Zheng, Jiaqi and Xu, Kai and Yao, Huayang and Yin, Baocai},
  year   = {2026},
  note   = {Manuscript}
}
```

## Acknowledgements

This project builds on ideas and tools from PointNet++, RoBERTa, MinkowskiEngine, GraspNet-1Billion, 3D AffordanceNet, LASO, and related affordance-grounding and grasp-detection research.

## License

A project license will be added before the complete public release. Until then, no permission is granted to redistribute or reuse the code beyond applicable legal exceptions.

## Contact

For research questions, please contact the project authors or open a GitHub issue after the repository becomes public.
