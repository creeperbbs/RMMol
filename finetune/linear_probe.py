"""Linear and random-forest probes for frozen RMMol embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loader.loader import molcae_embed
from trainer.pretrain_lightning import MolGATMAE, load_config


def load_split_csvs(paths: Sequence[str]) -> pd.DataFrame:
    """Load one or more CSV files and concatenate them."""
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise ValueError("At least one CSV path is required.")
    return pd.concat(frames, ignore_index=True)


def infer_task_columns(df: pd.DataFrame, smiles_col: str, explicit_tasks: Sequence[str] | None) -> List[str]:
    """Resolve task columns for probing."""
    if explicit_tasks:
        missing = [task for task in explicit_tasks if task not in df.columns]
        if missing:
            raise ValueError(f"Missing task columns: {missing}")
        return list(explicit_tasks)
    return [col for col in df.columns if col != smiles_col and pd.api.types.is_numeric_dtype(df[col])]


def embed_smiles(checkpoint: str, config_path: str, smiles: Sequence[str], device: str) -> np.ndarray:
    """Load a checkpoint and embed a SMILES list."""
    config = load_config(config_path)
    model = MolGATMAE.load_from_checkpoint(checkpoint, config=config, map_location=device)
    model.eval()
    model.to(device)
    with torch.no_grad():
        return molcae_embed(model.encoder, list(smiles), device=device).cpu().numpy()


def run_regression_probe(x: np.ndarray, y: np.ndarray, n_splits: int, model_type: str) -> pd.DataFrame:
    """Run cross-validated regression probes for each target."""
    rows = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for target_idx in range(y.shape[1]):
        target = y[:, target_idx]
        valid = np.isfinite(target)
        if valid.sum() < n_splits:
            continue
        x_valid = x[valid]
        y_valid = target[valid]
        for fold, (train_idx, test_idx) in enumerate(kfold.split(x_valid), start=1):
            reg = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1) if model_type == "rf" else Ridge(alpha=1.0)
            reg.fit(x_valid[train_idx], y_valid[train_idx])
            pred = reg.predict(x_valid[test_idx])
            rows.append(
                {
                    "target_index": target_idx,
                    "fold": fold,
                    "mae": mean_absolute_error(y_valid[test_idx], pred),
                    "rmse": mean_squared_error(y_valid[test_idx], pred, squared=False),
                    "r2": r2_score(y_valid[test_idx], pred),
                }
            )
    return pd.DataFrame(rows)


def run_classification_probe(x: np.ndarray, y: np.ndarray, n_splits: int, model_type: str) -> pd.DataFrame:
    """Run cross-validated binary classification probes for each target."""
    rows = []
    for target_idx in range(y.shape[1]):
        target = y[:, target_idx]
        valid = np.isfinite(target)
        y_valid = target[valid].astype(int)
        x_valid = x[valid]
        if len(np.unique(y_valid)) < 2:
            continue
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(x_valid, y_valid), start=1):
            clf = (
                RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
                if model_type == "rf"
                else LogisticRegression(max_iter=5000)
            )
            clf.fit(x_valid[train_idx], y_valid[train_idx])
            score = clf.predict_proba(x_valid[test_idx])[:, 1]
            rows.append(
                {
                    "target_index": target_idx,
                    "fold": fold,
                    "roc_auc": roc_auc_score(y_valid[test_idx], score)
                    if len(np.unique(y_valid[test_idx])) >= 2
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen-embedding probes on molecular datasets.")
    parser.add_argument("--checkpoint", required=True, help="RMMol checkpoint path.")
    parser.add_argument("--config", default="configs/pretrain_zinc.yaml", help="Training config used by the checkpoint.")
    parser.add_argument("--csv", required=True, nargs="+", help="One or more CSV files.")
    parser.add_argument("--smiles_col", default="smiles", help="SMILES column name.")
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional target columns.")
    parser.add_argument("--task_type", choices=["classification", "regression"], required=True)
    parser.add_argument("--model", choices=["linear", "rf"], default="linear", help="Probe model type.")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument("--output", default="runs/linear_probe_results.csv", help="Output CSV path.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_split_csvs(args.csv)
    if args.smiles_col not in df.columns:
        raise ValueError(f"Missing SMILES column: {args.smiles_col}")

    tasks = infer_task_columns(df, args.smiles_col, args.tasks)
    if not tasks:
        raise ValueError("No numeric task columns were found.")

    smiles = df[args.smiles_col].astype(str).tolist()
    x = embed_smiles(args.checkpoint, args.config, smiles, args.device)
    y = df[tasks].to_numpy(dtype=float)

    if args.task_type == "classification":
        results = run_classification_probe(x, y, args.folds, args.model)
    else:
        results = run_regression_probe(x, y, args.folds, args.model)

    target_lookup = {idx: task for idx, task in enumerate(tasks)}
    results["target"] = results["target_index"].map(target_lookup)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(results.groupby("target").mean(numeric_only=True).reset_index())


if __name__ == "__main__":
    main()
