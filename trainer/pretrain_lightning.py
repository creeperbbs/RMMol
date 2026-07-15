"""PyTorch Lightning pre-training entry point for RMMol."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

try:
    from ..loader.loader import MoleculeProcessor, molcae_embed
    from ..model.rmmol_gnn_model import GNN, GNNDecoder
    from ..utils.loss import TopologyAwareNTXentLoss, sce_loss
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT))
    from loader.loader import MoleculeProcessor, molcae_embed
    from model.rmmol_gnn_model import GNN, GNNDecoder
    from utils.loss import TopologyAwareNTXentLoss, sce_loss


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 2025,
    "batch_size": 128,
    "epochs": 1,
    "num_workers": 0,
    "pin_memory": True,
    "valid_size": 0.05,
    "init_lr": 3e-4,
    "weight_decay": 1e-5,
    "num_layer": 6,
    "emb_dim": 1024,
    "feat_dim": 768,
    "dropout_ratio": 0.0,
    "gnn_type": "gin",
    "JK": "last",
    "mask_edge": True,
    "temperature": 0.1,
    "use_cosine_similarity": True,
    "alpha_l": 1.0,
    "num_remasking": 2,
    "remask_rate": 0.5,
    "lambda_divergence": 0.0,
    "contrastive_weight": 1.0,
    "output_dir": "runs/rmmol_pretrain",
    "save_top_k": -1,
    "precision": "32-true",
    "accelerator": "auto",
    "devices": "auto",
}

QM9_TARGETS = ["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u298", "h298", "g298", "cv"]


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load YAML config and merge it with defaults."""
    config = DEFAULT_CONFIG.copy()
    if config_path:
        with open(config_path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        config.update(loaded)
    return config


def resolve_smiles_files(data_path: str) -> List[str]:
    """Resolve a data file, directory, or comma-separated file list."""
    if not data_path:
        raise ValueError("data_path is required.")

    parts = [part.strip() for part in data_path.split(",") if part.strip()]
    files: List[Path] = []
    for part in parts:
        path = Path(part)
        if path.is_dir():
            files.extend(sorted(path.glob("*.smi")))
            files.extend(sorted(path.glob("*.txt")))
            files.extend(sorted(path.glob("*.csv")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Data path does not exist: {path}")

    resolved = [str(path) for path in files]
    if not resolved:
        raise FileNotFoundError(f"No SMILES files were found under: {data_path}")
    return resolved


def load_qm9_data(
    csv_path: str,
    smiles_col: str = "smiles",
    target_cols: Optional[Sequence[str]] = None,
) -> Tuple[List[str], np.ndarray]:
    """Read a QM9-style CSV into SMILES and target arrays."""
    target_cols = list(target_cols or QM9_TARGETS)
    df = pd.read_csv(csv_path)
    missing = [col for col in [smiles_col, *target_cols] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    smiles = df[smiles_col].astype(str).tolist()
    targets = df[target_cols].values.astype(np.float32)
    return smiles, targets


def evaluate_linear_probe(
    encoder: torch.nn.Module,
    smiles_train: Sequence[str],
    y_train: np.ndarray,
    smiles_test: Sequence[str],
    y_test: np.ndarray,
    device: str = "cuda",
) -> float:
    """Fit a ridge probe on frozen embeddings and return mean MAE."""
    z_train = molcae_embed(encoder, smiles_train, device=device).cpu().numpy()
    z_test = molcae_embed(encoder, smiles_test, device=device).cpu().numpy()
    maes = []
    for target_idx in range(y_train.shape[1]):
        reg = Ridge(alpha=1.0)
        reg.fit(z_train, y_train[:, target_idx])
        pred = reg.predict(z_test)
        maes.append(mean_absolute_error(y_test[:, target_idx], pred))
    return float(np.mean(maes))


def random_remask(
    dec_mask_token: torch.Tensor,
    rep: torch.Tensor,
    data: Any,
    device: torch.device,
    remask_rate: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replace a random subset of node representations with the decoder mask token."""
    num_nodes = data.num_nodes
    num_remask_nodes = max(1, int(remask_rate * num_nodes))
    remask_nodes = torch.randperm(num_nodes, device=device)[:num_remask_nodes]
    rep_new = rep.clone()
    rep_new[remask_nodes] = dec_mask_token.to(device=device, dtype=rep.dtype)
    return rep_new, remask_nodes


@rank_zero_only
def remove_tree(cache_files: Iterable[str]) -> None:
    """Remove temporary Hugging Face dataset cache files when requested."""
    for cache_file in set(cache_files):
        cache_path = Path(cache_file)
        if cache_path.exists():
            shutil.rmtree(cache_path, ignore_errors=True)


class CheckpointEveryNSteps(pl.Callback):
    """Save a checkpoint every N optimizer steps."""

    def __init__(
        self,
        save_step_frequency: int = -1,
        prefix: str = "step",
        use_modelcheckpoint_filename: bool = False,
    ):
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self.save_step_frequency <= 0 or trainer.global_step % self.save_step_frequency != 0:
            return
        callback = trainer.checkpoint_callback
        if callback is None or callback.dirpath is None:
            return
        filename = callback.filename if self.use_modelcheckpoint_filename else f"{self.prefix}_{trainer.current_epoch}_{trainer.global_step}.ckpt"
        trainer.save_checkpoint(str(Path(callback.dirpath) / filename))


class MoleculeModule(pl.LightningDataModule):
    """Data module for SMILES-only self-supervised pre-training."""

    def __init__(self, data_path: str, config: Dict[str, Any]):
        super().__init__()
        self.data_path = data_path
        self.config = config
        self.data_collector = MoleculeProcessor(
            mask_rate=float(config.get("mask_rate", 0.25)),
            mask_edge=float(config.get("mask_edge_rate", 0.25)),
        )
        self.cache_files: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        files = resolve_smiles_files(self.data_path)
        loader_script = Path(__file__).resolve().parents[1] / "loader" / "zinc_script.py"
        dataset = load_dataset(
            str(loader_script),
            data_files={"train": files},
            cache_dir=self.config.get("cache_dir"),
            split="train",
        )
        self.cache_files = [cache["filename"] for cache in dataset.cache_files]
        split = dataset.train_test_split(test_size=float(self.config["valid_size"]), seed=int(self.config["seed"]))
        self.train_dataset = split["train"]
        self.val_dataset = split["test"]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.config["batch_size"]),
            num_workers=int(self.config["num_workers"]),
            pin_memory=bool(self.config.get("pin_memory", True)),
            collate_fn=self.data_collector.process,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.config["batch_size"]),
            num_workers=int(self.config["num_workers"]),
            pin_memory=bool(self.config.get("pin_memory", True)),
            collate_fn=self.data_collector.process,
            shuffle=False,
            drop_last=True,
        )


class MolGATMAE(pl.LightningModule):
    """RMMol masked molecular autoencoder with reciprocal reconstruction."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.cur_device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss_fn = config.get("loss_fn", "sce")
        self.num_remasking = int(config.get("num_remasking", 2))
        self.remask_rate = float(config.get("remask_rate", 0.5))
        self.lambda_divergence = float(config.get("lambda_divergence", 0.0))
        self.contrastive_weight = float(config.get("contrastive_weight", 1.0))

        num_node_attr = 119 + 4 + 7 + 12 + 10 + 12 + 8
        num_bond_attr = 5 + 3 + 3 + 3
        self.encoder = GNN(
            num_layer=int(config["num_layer"]),
            emb_dim=int(config["emb_dim"]),
            JK=config["JK"],
            feat_dim=int(config["feat_dim"]),
            drop_ratio=float(config["dropout_ratio"]),
            gnn_type=config.get("gnn_type", "gin"),
            degree_list=list(range(11)),
            batch_size=int(config["batch_size"]),
            device=torch.device(self.cur_device_name),
        ).double()
        self.dec_pred_atoms = GNNDecoder(
            int(config["emb_dim"]),
            num_node_attr,
            JK=config["JK"],
            gnn_type=config.get("gnn_type", "gin"),
        ).double()
        self.dec_pred_bonds = GNNDecoder(
            int(config["emb_dim"]),
            num_bond_attr,
            JK=config["JK"],
            gnn_type="linear",
        ).double()
        self.nt_xent_criterion = TopologyAwareNTXentLoss(
            device=torch.device(self.cur_device_name),
            temperature=float(config["temperature"]),
        ).double()
        self.temperature = float(config["temperature"])
        self.criterion = partial(sce_loss, alpha=float(config.get("alpha_l", 1.0)))

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["rng"] = {
            "torch_state": torch.get_rng_state(),
            "cuda_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "numpy_state": np.random.get_state(),
            "python_state": random.getstate(),
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        rng = checkpoint.get("rng")
        if not rng:
            return
        torch.set_rng_state(rng["torch_state"])
        if torch.cuda.is_available() and rng.get("cuda_state") is not None:
            torch.cuda.set_rng_state(rng["cuda_state"])
        np.random.set_state(rng["numpy_state"])
        random.setstate(rng["python_state"])

    def _standard_nt_xent_loss(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent when topology fingerprints are not available."""
        batch_size = z_i.shape[0]
        if batch_size <= 1:
            return z_i.new_tensor(0.0, requires_grad=True)

        reps = torch.cat([z_i, z_j], dim=0)
        sim = torch.matmul(reps, reps.T) / self.temperature
        diag = torch.eye(2 * batch_size, dtype=torch.bool, device=sim.device)
        pos = torch.zeros_like(diag)
        idx = torch.arange(batch_size, device=sim.device)
        pos[idx, idx + batch_size] = True
        pos[idx + batch_size, idx] = True

        positives = sim[pos].view(2 * batch_size, 1)
        negatives = sim[(~diag) & (~pos)].view(2 * batch_size, -1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=sim.device)
        return F.cross_entropy(logits, labels)

    def _contrastive_loss(self, z_i: torch.Tensor, z_j: torch.Tensor, batch: Any) -> torch.Tensor:
        """Use topology-aware NT-Xent only when batch fingerprints are present."""
        fps = getattr(batch, "fp", None)
        if fps is None:
            fps = getattr(batch, "fingerprint", None)
        if fps is not None:
            return self.nt_xent_criterion(z_i, z_j, fps.to(z_i.device))
        return self._standard_nt_xent_loss(z_i, z_j)

    def _shared_step(self, batch: Tuple[Any, Any], stage: str) -> torch.Tensor:
        xis, xjs = batch
        node_rep_i, z_i = self.encoder(xis)
        node_rep_j, z_j = self.encoder(xjs)
        masked_nodes_i = xis.masked_atom_indices
        masked_nodes_j = xjs.masked_atom_indices

        reconstruction_loss = 0.0
        for _ in range(self.num_remasking):
            rep_i = node_rep_i.clone().detach().requires_grad_(True)
            rep_j = node_rep_j.clone().detach().requires_grad_(True)
            with torch.no_grad():
                rep_i_masked, _ = random_remask(self.encoder.dec_mask_token, rep_i, xis, self.device, self.remask_rate)
                rep_j_masked, _ = random_remask(self.encoder.dec_mask_token, rep_j, xjs, self.device, self.remask_rate)
            rep_i_masked = rep_i + (rep_i_masked - rep_i).detach()
            rep_j_masked = rep_j + (rep_j_masked - rep_j).detach()

            recon_i = self.dec_pred_atoms(rep_i_masked, xis.edge_index, xis.edge_attr, masked_nodes_i)
            recon_j = self.dec_pred_atoms(rep_j_masked, xjs.edge_index, xjs.edge_attr, masked_nodes_j)
            reconstruction_loss = reconstruction_loss + self.criterion(
                xis.node_attr_label[masked_nodes_i], recon_i[masked_nodes_i]
            )
            reconstruction_loss = reconstruction_loss + self.criterion(
                xjs.node_attr_label[masked_nodes_j], recon_j[masked_nodes_j]
            )

        pred_i_to_j = self.dec_pred_atoms(node_rep_i, xjs.edge_index, xjs.edge_attr, masked_nodes_j)
        pred_j_to_i = self.dec_pred_atoms(node_rep_j, xis.edge_index, xis.edge_attr, masked_nodes_i)
        latent_loss = self.criterion(xjs.node_attr_label[masked_nodes_j], pred_i_to_j[masked_nodes_j])
        latent_loss = latent_loss + self.criterion(xis.node_attr_label[masked_nodes_i], pred_j_to_i[masked_nodes_i])

        edge_loss = 0.0
        if bool(self.config.get("mask_edge", True)):
            if xjs.connected_edge_indices.numel() > 0:
                edge_index_j = xjs.edge_index[:, xjs.connected_edge_indices]
                edge_rep_j = node_rep_i[edge_index_j[0]] + node_rep_i[edge_index_j[1]]
                pred_edge_j = self.dec_pred_bonds(edge_rep_j, xjs.edge_index, xjs.edge_attr, masked_nodes_j)
                edge_loss = edge_loss + self.criterion(pred_edge_j, xjs.edge_attr_label[xjs.connected_edge_indices])
            if xis.connected_edge_indices.numel() > 0:
                edge_index_i = xis.edge_index[:, xis.connected_edge_indices]
                edge_rep_i = node_rep_j[edge_index_i[0]] + node_rep_j[edge_index_i[1]]
                pred_edge_i = self.dec_pred_bonds(edge_rep_i, xis.edge_index, xis.edge_attr, masked_nodes_i)
                edge_loss = edge_loss + self.criterion(pred_edge_i, xis.edge_attr_label[xis.connected_edge_indices])

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        contrast_loss = self._contrastive_loss(z_i, z_j, xis)
        divergence_loss = torch.norm(z_i - z_j, p=2, dim=1).mean()
        total_loss = (
            reconstruction_loss
            + latent_loss
            + edge_loss
            + self.contrastive_weight * contrast_loss
            + self.lambda_divergence * divergence_loss
        )
        self.log(f"{stage}_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_contrast_loss", contrast_loss, sync_dist=True)
        return total_loss

    def training_step(self, batch: Tuple[Any, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[Any, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.config["init_lr"]),
            weight_decay=float(self.config["weight_decay"]),
            betas=(0.9, 0.99),
        )


def run_qm9_probe_if_requested(model: MolGATMAE, config: Dict[str, Any]) -> Optional[float]:
    """Optionally run a QM9 ridge probe after pre-training."""
    probe = config.get("qm9_probe")
    if not probe:
        return None

    train_csv = probe.get("train_csv")
    test_csv = probe.get("test_csv")
    if not train_csv or not test_csv:
        raise ValueError("qm9_probe requires train_csv and test_csv.")

    smiles_train, y_train_raw = load_qm9_data(train_csv)
    smiles_test, y_test_raw = load_qm9_data(test_csv)
    scaler = StandardScaler()
    y_train = scaler.fit_transform(y_train_raw)
    y_test = scaler.transform(y_test_raw)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return evaluate_linear_probe(model.encoder, smiles_train, y_train, smiles_test, y_test, device=device)


def build_trainer(config: Dict[str, Any]) -> pl.Trainer:
    """Create a Lightning trainer from config."""
    output_dir = Path(config["output_dir"])
    callbacks = [
        ModelCheckpoint(dirpath=str(output_dir / "checkpoints"), save_top_k=int(config["save_top_k"]), every_n_epochs=1),
        LearningRateMonitor(logging_interval="step"),
    ]
    if int(config.get("save_every_n_steps", 0)) > 0:
        callbacks.append(CheckpointEveryNSteps(save_step_frequency=int(config["save_every_n_steps"])))

    return pl.Trainer(
        default_root_dir=str(output_dir),
        max_epochs=int(config["epochs"]),
        accelerator=config.get("accelerator", "auto"),
        devices=config.get("devices", "auto"),
        precision=config.get("precision", "32-true"),
        callbacks=callbacks,
        accumulate_grad_batches=int(config.get("accumulate_grad_batches", 1)),
        log_every_n_steps=int(config.get("log_every_n_steps", 10)),
        val_check_interval=config.get("val_check_interval", 1.0),
        num_sanity_val_steps=int(config.get("num_sanity_val_steps", 0)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-train RMMol on SMILES files.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument("--data_path", type=str, default=None, help="SMILES file, directory, or comma-separated file list.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for logs and checkpoints.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size.")
    parser.add_argument("--devices", type=str, default=None, help="Lightning devices argument, for example 1 or auto.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    for key in ["data_path", "output_dir", "epochs", "batch_size", "devices"]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    if not config.get("data_path"):
        raise ValueError("Set data_path in YAML or pass --data_path.")

    pl.seed_everything(int(config["seed"]), workers=True)
    data_module = MoleculeModule(config["data_path"], config)
    model = MolGATMAE(config)

    checkpoint_path = config.get("resume_from_checkpoint")
    trainer = build_trainer(config)
    trainer.fit(model=model, datamodule=data_module, ckpt_path=checkpoint_path)

    qm9_mae = run_qm9_probe_if_requested(model, config)
    if qm9_mae is not None:
        print(f"QM9 probe mean MAE: {qm9_mae:.6f}")

    if config.get("cleanup_cache", False):
        remove_tree(data_module.cache_files)

if __name__ == "__main__":
    main()
