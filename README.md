
# RMMol

Implicit conformational perception via geometry-aware reciprocal masked molecular learning.
=======

# Implicit conformational perception via geometry-aware
reciprocal masked molecular learning

[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee8c00.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3+-3c7fb0.svg)](https://pytorch-geometric.readthedocs.io/)

<<<<<<< HEAD
RMMol is a self-supervised molecular pre-training framework for learning graph representations that preserve local chemistry and implicit conformational cues from 2D molecular graphs. The repository includes reciprocal masked molecular learning, downstream embedding extraction, activity cliff utilities, lipid physical proxies, and a Functional Group Knockout Perturbation (FGKP) toolkit for interpretable SAR analysis.

## Highlights

- Implicit conformational perception from 2D molecular topology.
- Reciprocal masked reconstruction across complementary molecular graph views.
- RDKit-based activity cliff, USR, FCFP, and lipid descriptor utilities.
- FGKP workflow for functional-group perturbation and activity attribution.
- Parameterized training script with no machine-specific paths.

## Repository Structure

```text
RMMol/
  configs/
    pretrain_zinc.yaml          # Default self-supervised pre-training config
  loader/
    loader.py                   # SMILES-to-PyG graph conversion and masking
    zinc_script.py              # Hugging Face datasets loader for SMILES text files
  model/
    rmmol_gnn_model.py          # Encoder and decoder model components
  trainer/
    pretrain_lightning.py       # PyTorch Lightning training CLI
  utils/
    chem_utils.py               # RDKit, USR, FCFP, cliff, lipid, and FGKP utilities
    loss.py                     # NT-Xent and SCE losses
    metrics.py                  # Regression, classification, and cliff metrics
  requirements.txt
  README.md
```

## Function Groups

The main utility functions are organized in `utils/chem_utils.py`:

- Conformer and USR utilities: `optimize_conformation`, `generate_3d_conformation`, `compute_usr_robust`, `compute_usr_batch`, `compute_conformational_distances`.
- Fingerprint and activity cliff utilities: `canonicalize_safe`, `compute_fcfp`, `identify_activity_cliff_pairs`, `extract_embedding_distances`.
- Lipid physical proxies: `calculate_lipid_physical_descriptors`, `calculate_conformational_entropy_proxy`, `saturate_carbon_double_bonds`.
- FGKP workflow: `FunctionalGroupDetector`, `FunctionalGroupKnockout`, `FeatureBasedActivityChangeQuantifier`, `CausalDiscoveryAnalyzer`, `ActivityFGKPEngine`.
- Model helpers: `EnsembleRegressor`.

## Environment Setup

Create a Python environment:

```bash
conda create -n rmmol python=3.9 -y
conda activate rmmol
```

Install PyTorch and PyTorch Geometric for your CUDA or CPU platform. Example for CUDA 11.8:

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

RDKit can also be installed through conda if the pip package is not available for your platform:

```bash
conda install -c conda-forge rdkit
```

## Data Format

Pre-training expects one or more plain-text, `.smi`, `.txt`, or `.csv` files containing SMILES strings. The loader reads the first whitespace-separated token from each line and skips header lines containing `smiles`.

Example:

```text
CCO
c1ccccc1
CC(=O)O
```

Keep large datasets and checkpoints outside git. The repository intentionally contains code and configuration only.

## Self-Supervised Pre-Training

Run pre-training with a YAML config and a local data file or directory:

```bash
python trainer/pretrain_lightning.py \
  --config configs/pretrain_zinc.yaml \
  --data_path data/smiles \
  --output_dir runs/rmmol_pretrain
```

You can also set `data_path`, `output_dir`, `epochs`, `batch_size`, and device options directly in `configs/pretrain_zinc.yaml`.

Important config fields:

- `data_path`: SMILES file, directory, or comma-separated file list.
- `output_dir`: directory for Lightning logs and checkpoints.
- `batch_size`, `epochs`, `num_workers`: training throughput controls.
- `num_layer`, `emb_dim`, `feat_dim`: model capacity.
- `mask_rate`, `mask_edge_rate`, `num_remasking`: masking and reconstruction controls.
- `contrastive_weight`, `lambda_divergence`: representation regularization controls.

## Embedding Extraction

Use the encoder with `loader.molcae_embed`:

```python
import torch
from loader.loader import molcae_embed
from trainer.pretrain_lightning import MolGATMAE, load_config

config = load_config("configs/pretrain_zinc.yaml")
model = MolGATMAE.load_from_checkpoint("runs/rmmol_pretrain/checkpoints/example.ckpt", config=config)
model.eval()

smiles = ["CCO", "c1ccccc1"]
embeddings = molcae_embed(model.encoder, smiles, device="cuda" if torch.cuda.is_available() else "cpu")
```

## Activity Cliff and USR Example

```python
import numpy as np
from utils.chem_utils import (
    compute_conformational_distances,
    compute_usr_batch,
    identify_activity_cliff_pairs,
)

smiles = ["CCO", "CCCO", "c1ccccc1"]
activity = np.array([5.0, 6.2, 4.1])

pairs = identify_activity_cliff_pairs(smiles, activity, tanimoto_threshold=0.7, delta_potency=1.0)
usr_matrix = compute_usr_batch(smiles, pairs)
usr_distances = compute_conformational_distances(usr_matrix, pairs)
```

## FGKP Example

`ActivityFGKPEngine` expects an encoder object with an `encode_batch(smiles_list, batch_size)` method and a regressor with a `predict(X)` method.

```python
from utils.chem_utils import ActivityFGKPEngine, EnsembleRegressor

regressor = EnsembleRegressor([trained_regressor])
engine = ActivityFGKPEngine(
    encoder=embedding_encoder,
    activity_predictor=regressor,
    alpha=0.05,
    min_samples=5,
)

raw_df, summary_df = engine.run(smiles_list, batch_size=32, max_variants_per_mol=10)
engine.export_results(summary_df, raw_df, output_dir="runs/fgkp", dataset_name="activity")
```

## Lipid Physical Proxies

```python
from utils.chem_utils import calculate_lipid_physical_descriptors

features = calculate_lipid_physical_descriptors("CCCCCCCCCCCCCCCC(=O)O", n_confs=50)
```

The returned dictionary includes `tail_disorder`, `rmsd_std`, `usr_variance`, `cpp_estimate`, `fusion_propensity`, `amphiphilicity_balance`, `tail_carbons`, `head_polar_atoms`, and `tail_length`.

## Checks Before Running Experiments

- Confirm all data paths are passed through config or CLI arguments.
- Keep pretrained checkpoints and large datasets outside the repository.
- Run a repository text scan to confirm that non-English comments and machine-specific absolute paths have not been reintroduced.
- Run a quick syntax check after editing:

```bash
python -m compileall loader model trainer utils
```

## License

This project is released under the MIT License.

=======
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

