# RMMol Examples

This directory contains lightweight Jupyter demonstrations for the public RMMol repository. The notebooks use small CSV snapshots copied from the internal experiments so users can inspect the evaluation logic without downloading the full benchmark datasets or checkpoints.

## Notebooks

- `conformation_distance_evaluation.ipynb`: compares pairwise representation distance against RDKit USR conformational distance on sampled LNPDB ionizable-lipid pairs. It reports Spearman and Pearson correlations and reproduces the log-transformed hexbin visualization.
- `moleculenet_5fold_results.ipynb`: summarizes MolecularNet-style 5-fold representation learning results, including BBBP, BACE, HIV, the FDA-approved ClinTox label, Tox21, SIDER, ESOL, FreeSolv, Lipophilicity and QM9. It also includes a fold-level BBBP comparison across RMMol, ConfSeq, MoLFormer, MoleBERT, GROVER, UniMol and FCFP.

## Packaged Result Tables

- `data/conformation_distance_lnpdb_common10k_summary.csv`: full 10k-pair conformation-distance summary by model.
- `data/conformation_distance_lnpdb_common10k_pair_sample.csv`: deterministic 600-pair-per-model sample used by the conformation notebook.
- `data/geometry_recoverability_metrics.csv`: geometry recoverability metrics used for downstream figure checks.
- `data/moleculenet_*_source.csv`: task/domain source tables used by the MolecularNet demonstration.
- `data/confseq_moleculenet_5fold_*.csv`: ConfSeq 5-fold fold-level and summary results.
- `data/bbbp_multimodel_5fold_*.csv`: BBBP fold-level and summary comparison across models.

The examples intentionally do not include raw datasets, checkpoints or generated conformer caches.
