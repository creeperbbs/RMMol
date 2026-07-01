
# Implicit conformational perception via geometry-aware
reciprocal masked molecular learning

[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee8c00.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3+-3c7fb0.svg)](https://pytorch-geometric.readthedocs.io/)

**RMMol** is a self-supervised molecular pre-training framework that fundamentally resolves the trade-off between physical resolution and computational scalability. By reframing representation learning as rigorous inference under complementary observational constraints, RMMol unlocks **implicit 3D perception** (hybridization, steric hindrance, conformational entropy) purely from discrete 2D topological graphs via degree-stratified geometric priors.

## ✨ Key Highlights

- 🧬 **Implicit 3D Perception**: Captures local hybridization ($sp^2$ vs $sp^3$) and mesoscale thermodynamics without explicit 3D coordinates.
- 🔄 **Reciprocal Masking**: Breaks topological smoothness bias by enforcing cross-view structural inference.
- 🚀 **Cross-Domain Generalization**: Achieves zero-shot transferability to Lipid Nanoparticle (LNP) delivery efficiency, outperforming explicit 3D models while accelerating inference by ~100x.
- 🔍 **Mechanistic Interpretability**: The built-in Functional Group Knockout Perturbation (FGKP) system translates black-box embeddings into chemically intuitive SAR rules.

---

## 📂 Repository Structure

```text
RMMol/
├── loader/
│   └── loader.py            # Data loading, graph construction, and fingerprint extraction
├── model/
│   └── rmmol_gnn_model.py   # Core architecture: DMP (Decoupled Message Passing) & Encoder/Decoder
├── trainer/
│   └── pretrain_lightning.py# PyTorch Lightning training loop for self-supervised pre-training
├── utils/
│   ├── metrics.py           # Evaluation metrics (AUROC, RMSE, CDR)
│   └── chem_utils.py        # RDKit wrappers, FGKP implementation, and physicochemical proxies
├── configs/                 # YAML configuration files for pre-training and downstream tasks
├── README.md
└── requirements.txt
```
## ⚙️ Environment Setup
# 1. Create and activate conda environment
conda create -n rmmol python=3.9 -y
conda activate rmmol

# 2. Install PyTorch (adjust CUDA version as needed)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 3. Install PyTorch Geometric
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# 4. Install other dependencies
pip install -r requirements.txt
## ⬇️ Pre-trained Weights
Due to the large size of the pre-trained checkpoints, they are hosted externally.
Option A: Hugging Face Hub (Recommended for automated downloading)
from huggingface_hub import hf_hub_download

# Automatically downloads and caches the checkpoint
ckpt_path = hf_hub_download(
    repo_id="YourUsername/RMMol-Pretrained", 
    filename="rmmol_pretrained_zinc2m.ckpt"
)
## 🚀 Quick Start
#1. Self-Supervised Pre-training
To pre-train RMMol from scratch on your own molecular corpus (e.g., ZINC or ChEMBL):
python trainer/pretrain_lightning.py \
    --config configs/pretrain_zinc.yaml \
    --data_path /path/to/your/smiles_dataset.csv \
    --save_dir checkpoints/
#2. Downstream Fine-tuning / Linear Probing
from model.rmmol_gnn_model import RMMol
import torch

Initialize model
model = RMMol.load_from_checkpoint("checkpoints/rmmol_pretrained.ckpt")
model.eval()

Extract representations
z = model.get_molecular_embedding(graph_data)

## 📊 Reproducing Activity Cliff & LNP Results
To reproduce the 30-target Activity Cliff analysis and the LNP delivery efficiency linear probing results presented in the paper, please refer to the Jupyter notebooks in the notebooks/ directory:
notebooks/01_activity_cliff_30_targets.ipynb
