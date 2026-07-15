"""Chemistry utilities for RMMol analysis and FGKP experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from scipy import stats
from tqdm import tqdm


N_USR_FEATURES = 12


def canonicalize_safe(smiles: str, isomeric: bool = False) -> Optional[str]:
    """Return a canonical SMILES string, or None when parsing fails."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)
    except Exception:
        return None


def compute_fcfp(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
    use_features: bool = True,
) -> np.ndarray:
    """Compute a feature-class Morgan fingerprint as a dense float array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits, useFeatures=use_features
    )
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def identify_activity_cliff_pairs(
    smiles_list: Sequence[str],
    y_list: Sequence[float],
    tanimoto_threshold: float = 0.7,
    delta_potency: float = 1.0,
    morgan_radius: int = 2,
    morgan_nbits: int = 2048,
) -> List[Tuple[int, int]]:
    """Find structurally similar molecule pairs with large potency differences."""
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = []
    valid_idx = []

    for idx, mol in enumerate(mols):
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, morgan_radius, nBits=morgan_nbits))
        valid_idx.append(idx)

    cliff_pairs = []
    for left in range(len(fps)):
        for right in range(left + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[left], fps[right])
            if sim < tanimoto_threshold:
                continue
            i, j = valid_idx[left], valid_idx[right]
            if abs(float(y_list[i]) - float(y_list[j])) > delta_potency:
                cliff_pairs.append((i, j))
    return cliff_pairs


def optimize_conformation(mol: Chem.Mol, conf_id: int = 0, max_iters: int = 500) -> float:
    """Try MMFF94 and UFF optimization, returning a large sentinel energy on failure."""
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id) if props is not None else None
        if ff is not None:
            ff.Initialize()
            if ff.Minimize(maxIts=max_iters) == 0:
                return float(ff.CalcEnergy())
    except Exception:
        pass

    try:
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
        if ff is not None:
            ff.Initialize()
            if ff.Minimize(maxIts=max_iters) == 0:
                return float(ff.CalcEnergy())
    except Exception:
        pass

    return 1e9


def generate_3d_conformation(mol: Chem.Mol, seed: int = 42) -> Tuple[bool, int]:
    """Generate one 3D conformer using progressively more permissive ETKDG settings."""
    strategies = [
        {"maxIterations": 2000, "enforceChirality": True, "useRandomCoords": False},
        {"maxIterations": 5000, "enforceChirality": False, "useRandomCoords": False},
        {"maxIterations": 5000, "enforceChirality": False, "useRandomCoords": True},
    ]

    for strategy in strategies:
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.numThreads = 1
        params.pruneRmsThresh = -1.0
        params.useSmallRingTorsions = False
        params.useMacrocycleTorsions = False
        params.maxIterations = strategy["maxIterations"]
        params.enforceChirality = strategy["enforceChirality"]
        params.useRandomCoords = strategy["useRandomCoords"]

        conf_id = AllChem.EmbedMolecule(mol, params)
        if conf_id != -1:
            optimize_conformation(mol, conf_id=conf_id)
            return True, int(conf_id)

    return False, -1


def compute_usr_robust(smiles: str) -> np.ndarray:
    """Compute a USR descriptor with molecule sanitization and conformer fallbacks."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.full(N_USR_FEATURES, np.nan)

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        try:
            sanitize_ops = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            Chem.SanitizeMol(mol, sanitizeOps=sanitize_ops)
        except Exception:
            return np.full(N_USR_FEATURES, np.nan)

    try:
        mol = Chem.AddHs(mol, addCoords=True)
    except Exception:
        return np.full(N_USR_FEATURES, np.nan)

    success, conf_id = generate_3d_conformation(mol)
    if not success:
        return np.full(N_USR_FEATURES, np.nan)

    try:
        return np.asarray(rdMolDescriptors.GetUSR(mol, confId=conf_id), dtype=np.float32)
    except Exception:
        return np.full(N_USR_FEATURES, np.nan)


def compute_usr_batch(
    smiles_list: Sequence[str],
    pair_indices: Sequence[Tuple[int, int]],
    show_progress: bool = True,
) -> np.ndarray:
    """Compute USR descriptors only for molecules referenced by pair indices."""
    unique_indices = sorted({idx for pair in pair_indices for idx in pair})
    usr_matrix = np.full((len(smiles_list), N_USR_FEATURES), np.nan, dtype=np.float32)

    iterator = tqdm(unique_indices, desc="USR descriptors") if show_progress else unique_indices
    for idx in iterator:
        usr_matrix[idx] = compute_usr_robust(smiles_list[idx])

    return usr_matrix


def compute_conformational_distances(
    usr_matrix: np.ndarray,
    pair_indices: Sequence[Tuple[int, int]],
) -> np.ndarray:
    """Compute Euclidean distances between paired USR vectors."""
    conf_dist = np.zeros(len(pair_indices), dtype=np.float32)
    for idx, (i, j) in enumerate(pair_indices):
        vec_i = usr_matrix[i]
        vec_j = usr_matrix[j]
        if np.isnan(vec_i).any() or np.isnan(vec_j).any():
            conf_dist[idx] = np.nan
        else:
            conf_dist[idx] = float(np.linalg.norm(vec_i - vec_j))
    return conf_dist


def extract_embedding_distances(
    smiles_list: Sequence[str],
    pair_indices: Sequence[Tuple[int, int]],
    embedding_source: Union[str, Path, Dict[str, np.ndarray]],
) -> np.ndarray:
    """Compute pairwise distances from a SMILES-keyed embedding dictionary or pickle file."""
    if isinstance(embedding_source, (str, Path)):
        import pickle

        with open(embedding_source, "rb") as handle:
            embeddings = pickle.load(handle)
    else:
        embeddings = embedding_source

    valid_idx_map: Dict[int, int] = {}
    graph_emb_list = []
    for original_idx, smiles in enumerate(smiles_list):
        embedding = embeddings.get(smiles)
        if embedding is None:
            canonical = canonicalize_safe(smiles)
            embedding = embeddings.get(canonical) if canonical is not None else None
        if embedding is None:
            continue
        valid_idx_map[original_idx] = len(graph_emb_list)
        graph_emb_list.append(np.asarray(embedding, dtype=np.float32))

    if not graph_emb_list:
        return np.array([], dtype=np.float32)

    z = torch.tensor(np.vstack(graph_emb_list), dtype=torch.float32)
    mapped_pairs = [(valid_idx_map[i], valid_idx_map[j]) for i, j in pair_indices if i in valid_idx_map and j in valid_idx_map]
    if not mapped_pairs:
        return np.array([], dtype=np.float32)

    distances = torch.zeros(len(mapped_pairs), dtype=torch.float32)
    for idx, (i, j) in enumerate(mapped_pairs):
        distances[idx] = torch.norm(z[i] - z[j], p=2)
    return distances.cpu().numpy()


def count_hydrophobic_tail_carbons(mol: Chem.Mol) -> int:
    """Estimate hydrophobic tail carbons with a conservative heavy-atom heuristic."""
    carbon_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
    head_heavy = sum(
        1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in {7, 8, 15, 16} or atom.IsInRing()
    )
    return max(0, len(carbon_atoms) - head_heavy)


def count_polar_headgroup_atoms(mol: Chem.Mol) -> int:
    """Count common polar hetero atoms used as a headgroup proxy."""
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in {7, 8, 15, 16})


def estimate_tail_length(mol: Chem.Mol) -> int:
    """Estimate the longest heavy-atom graph path length."""
    dist = Chem.GetDistanceMatrix(mol)
    return int(np.nanmax(dist)) if dist.size else 0


def calculate_lipid_physical_descriptors(smiles: str, n_confs: int = 50) -> Dict[str, float]:
    """Calculate lightweight physical proxies for lipid-like molecules."""
    base_mol = Chem.MolFromSmiles(smiles)
    if base_mol is None:
        return {
            "tail_disorder": np.nan,
            "rmsd_std": np.nan,
            "usr_variance": np.nan,
            "cpp_estimate": np.nan,
            "fusion_propensity": np.nan,
            "amphiphilicity_balance": np.nan,
            "tail_carbons": np.nan,
            "head_polar_atoms": np.nan,
            "tail_length": np.nan,
        }

    mol = Chem.AddHs(base_mol)
    params = AllChem.ETKDGv3()
    params.numThreads = 0
    params.pruneRmsThresh = 0.5
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params))

    if len(conf_ids) >= 2:
        rmsd_vals = [
            AllChem.GetConformerRMS(mol, i, j)
            for i in range(len(conf_ids))
            for j in range(i + 1, len(conf_ids))
        ]
        rmsd_std = float(np.std(rmsd_vals)) if rmsd_vals else 0.0
        usr_descriptors = [rdMolDescriptors.GetUSR(mol, confId=conf_id) for conf_id in conf_ids]
        usr_variance = float(np.var(usr_descriptors, axis=0).mean())
        conf_entropy_proxy = float(rmsd_std * np.log(len(conf_ids) + 1))
    else:
        rmsd_std = 0.0
        usr_variance = 0.0
        conf_entropy_proxy = 0.0

    tail_carbons = count_hydrophobic_tail_carbons(mol)
    head_polar_atoms = count_polar_headgroup_atoms(mol)
    tail_length = estimate_tail_length(mol)
    volume = tail_carbons * 27.0
    head_area = head_polar_atoms * 10.0 + 20.0
    tail_length_angstrom = tail_length * 1.27 if tail_length > 0 else 10.0
    cpp_estimate = volume / (head_area * tail_length_angstrom) if head_area * tail_length_angstrom > 1e-6 else np.nan

    logp = Descriptors.MolLogP(base_mol)
    tpsa = Descriptors.TPSA(base_mol)
    dipole = 0.0
    if conf_ids:
        try:
            AllChem.ComputeGasteigerCharges(mol)
            charges = [float(atom.GetProp("_GasteigerCharge")) for atom in mol.GetAtoms()]
            coords = mol.GetConformer(conf_ids[0]).GetPositions()
            dipole = float(np.linalg.norm(np.sum([q * coord for q, coord in zip(charges, coords)], axis=0)))
        except Exception:
            dipole = 0.0

    return {
        "tail_disorder": conf_entropy_proxy,
        "rmsd_std": rmsd_std,
        "usr_variance": usr_variance,
        "cpp_estimate": float(cpp_estimate),
        "fusion_propensity": float(0.5 * logp - 0.02 * tpsa + 0.1 * dipole),
        "amphiphilicity_balance": float(head_polar_atoms / (tail_carbons + 1e-6)),
        "tail_carbons": float(tail_carbons),
        "head_polar_atoms": float(head_polar_atoms),
        "tail_length": float(tail_length),
    }


def saturate_carbon_double_bonds(smiles: str) -> Optional[str]:
    """Return a counterfactual molecule with non-aromatic C=C bonds saturated."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    bonds_to_modify = [
        bond.GetIdx()
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE
        and not bond.GetIsAromatic()
        and bond.GetBeginAtom().GetAtomicNum() == 6
        and bond.GetEndAtom().GetAtomicNum() == 6
    ]
    if not bonds_to_modify:
        return smiles

    rw_mol = Chem.RWMol(mol)
    for bond_idx in bonds_to_modify:
        rw_mol.GetBondWithIdx(bond_idx).SetBondType(Chem.rdchem.BondType.SINGLE)

    try:
        new_mol = rw_mol.GetMol()
        Chem.SanitizeMol(new_mol)
        return Chem.MolToSmiles(new_mol)
    except Exception:
        return None


def calculate_conformational_entropy_proxy(smiles: str, n_conformers: int = 50) -> Dict[str, float]:
    """Estimate conformational diversity from generated conformer RMSD and USR spread."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"n_conformers": 0, "rmsd_std": np.nan, "shape_variance": np.nan, "entropy_proxy": np.nan}

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.numThreads = 0
    params.pruneRmsThresh = 0.5
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params))
    if len(conf_ids) < 2:
        return {"n_conformers": 0, "rmsd_std": 0.0, "shape_variance": 0.0, "entropy_proxy": 0.0}

    rmsd_list = [
        AllChem.GetConformerRMS(mol, i, j)
        for i in range(len(conf_ids))
        for j in range(i + 1, len(conf_ids))
    ]
    rmsd_std = float(np.std(rmsd_list)) if rmsd_list else 0.0
    shape_descriptors = [rdMolDescriptors.GetUSR(mol, confId=conf_id) for conf_id in conf_ids]
    shape_variance = float(np.var(shape_descriptors, axis=0).mean())
    return {
        "n_conformers": float(len(conf_ids)),
        "rmsd_std": rmsd_std,
        "shape_variance": shape_variance,
        "entropy_proxy": float(rmsd_std * np.log(len(conf_ids) + 1)),
    }


@dataclass(frozen=True)
class FunctionalGroup:
    """SMARTS definition for one functional group."""

    name: str
    smarts: str
    category: str
    description: str
    priority: int


FUNCTIONAL_GROUPS_LIBRARY = [
    FunctionalGroup("Hydroxyl", "[OX2H;!$([OX2H]-C=O)]", "polar", "Alcohol hydroxyl", 1),
    FunctionalGroup("Carboxylic_Acid", "[CX3](=O)[OX2H1]", "acidic", "Carboxylic acid", 1),
    FunctionalGroup("Primary_Amine", "[NX3;H2;!$(N-C=O)]", "basic", "Primary amine", 1),
    FunctionalGroup("Secondary_Amine", "[NX3;H1;!$(N-C=O)]", "basic", "Secondary amine", 1),
    FunctionalGroup("Tertiary_Amine", "[NX3;H0;!$(N-C=O)]", "basic", "Tertiary amine", 1),
    FunctionalGroup("Amide", "[NX3][CX3](=O)", "polar", "Amide", 2),
    FunctionalGroup("Nitro", "[NX3](=O)=O", "polar", "Nitro group", 1),
    FunctionalGroup("Sulfonamide", "[SX4](=[OX1])(=[OX1])[NX3]", "polar", "Sulfonamide", 1),
    FunctionalGroup("Ether", "[OD2]([#6])[#6]", "polar", "Ether", 3),
    FunctionalGroup("Ester", "[#6][CX3](=O)[OD2][#6]", "polar", "Ester", 2),
    FunctionalGroup("Ketone", "[#6][CX3](=O)[#6]", "polar", "Ketone", 2),
    FunctionalGroup("Aldehyde", "[CX3H1](=O)[#6]", "polar", "Aldehyde", 1),
    FunctionalGroup("Phenyl", "c1ccccc1", "aromatic", "Phenyl ring", 1),
    FunctionalGroup("Pyridine", "c1ccncc1", "aromatic", "Pyridine ring", 1),
    FunctionalGroup("Furan", "c1ccoc1", "aromatic", "Furan ring", 2),
    FunctionalGroup("Thiophene", "c1ccsc1", "aromatic", "Thiophene ring", 2),
    FunctionalGroup("Imidazole", "c1c[nH]cn1", "aromatic", "Imidazole ring", 2),
    FunctionalGroup("Fluorine", "[F;!$([F][CX3](=O))]", "halogen", "Fluorine substituent", 1),
    FunctionalGroup("Chlorine", "[Cl;!$([Cl][CX3](=O))]", "halogen", "Chlorine substituent", 1),
    FunctionalGroup("Bromine", "[Br]", "halogen", "Bromine substituent", 1),
    FunctionalGroup("Iodine", "[I]", "halogen", "Iodine substituent", 1),
    FunctionalGroup("Thiol", "[SX2H]", "polar", "Thiol", 1),
    FunctionalGroup("Thioether", "[#6][SX2][#6]", "polar", "Thioether", 2),
    FunctionalGroup("Cyano", "[CX1]#[NX1]", "polar", "Cyano group", 1),
    FunctionalGroup("Methoxy", "[OX2][CX4H3]", "polar", "Methoxy group", 2),
    FunctionalGroup("Trifluoromethyl", "[CX4](F)(F)F", "halogen", "Trifluoromethyl group", 1),
    FunctionalGroup("Hydroxymethyl", "[CX4H2][OX2H]", "polar", "Hydroxymethyl group", 2),
    FunctionalGroup("Methyl", "[CX4H3]", "hydrophobic", "Methyl group", 3),
    FunctionalGroup("Isopropyl", "[CX4H]([CX4H3])[CX4H3]", "hydrophobic", "Isopropyl group", 2),
    FunctionalGroup("tert-Butyl", "[CX4]([CX4H3])([CX4H3])[CX4H3]", "hydrophobic", "tert-Butyl group", 1),
    FunctionalGroup("Piperazine", "C1CNCCN1", "basic", "Piperazine ring", 1),
    FunctionalGroup("Piperidine", "C1CCCCN1", "basic", "Piperidine ring", 1),
    FunctionalGroup("Morpholine", "C1COCCN1", "polar", "Morpholine ring", 1),
    FunctionalGroup("Cyclopropyl", "C1CC1", "hydrophobic", "Cyclopropyl group", 2),
]


class FunctionalGroupDetector:
    """Detect functional groups in an RDKit molecule using SMARTS patterns."""

    def __init__(self, fg_library: Optional[List[FunctionalGroup]] = None):
        self.fg_library = fg_library or FUNCTIONAL_GROUPS_LIBRARY
        self.patterns = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        for fg in self.fg_library:
            patt = Chem.MolFromSmarts(fg.smarts)
            if patt is None:
                continue
            self.patterns[fg.name] = {
                "pattern": patt,
                "category": fg.category,
                "description": fg.description,
                "priority": fg.priority,
            }

    def detect(self, mol: Chem.Mol) -> List[Dict[str, Any]]:
        if mol is None:
            return []

        detected = []
        matched_atoms = set()
        sorted_items = sorted(self.patterns.items(), key=lambda item: item[1]["priority"])
        for fg_name, fg_info in sorted_items:
            for match in mol.GetSubstructMatches(fg_info["pattern"]):
                match_set = set(match)
                overlap = len(match_set & matched_atoms)
                if match_set and overlap / len(match_set) > 0.5:
                    continue
                detected.append(
                    {
                        "name": fg_name,
                        "category": fg_info["category"],
                        "description": fg_info["description"],
                        "atoms": list(match),
                        "priority": fg_info["priority"],
                        "atom_count": len(match),
                    }
                )
                matched_atoms.update(match_set)
        return detected


class FunctionalGroupKnockout:
    """Generate sanitized functional-group knockout variants."""

    def __init__(self, replacement_strategy: str = "hydrogen"):
        self.replacement_strategy = replacement_strategy
        self.detector = FunctionalGroupDetector()

    @staticmethod
    def has_dative_bond(mol: Optional[Chem.Mol]) -> bool:
        if mol is None:
            return True
        return any(bond.GetBondType() == Chem.BondType.DATIVE for bond in mol.GetBonds())

    def knockout(self, mol: Chem.Mol, fg_atoms: Sequence[int]) -> Optional[Chem.Mol]:
        if mol is None or not fg_atoms:
            return None

        atoms_to_remove = set(fg_atoms)
        rw_mol = Chem.RWMol(mol)
        for atom_idx in sorted(atoms_to_remove, reverse=True):
            rw_mol.RemoveAtom(atom_idx)

        new_mol = rw_mol.GetMol()
        for bond in new_mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DATIVE:
                bond.SetBondType(Chem.BondType.SINGLE)

        try:
            new_mol = Chem.AddHs(new_mol, addCoords=False)
            Chem.SanitizeMol(new_mol)
            return Chem.RemoveHs(new_mol)
        except Exception:
            return None

    def generate_knockout_variants(self, smiles: str, max_variants: int = 10) -> List[Dict[str, Any]]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or self.has_dative_bond(mol):
            return []

        variants = []
        for fg in self.detector.detect(mol):
            knocked_mol = self.knockout(mol, fg["atoms"])
            if knocked_mol is None or self.has_dative_bond(knocked_mol):
                continue
            try:
                knocked_smiles = Chem.MolToSmiles(knocked_mol)
            except Exception:
                continue
            if knocked_smiles == smiles:
                continue
            variants.append(
                {
                    "original_smiles": smiles,
                    "knockout_smiles": knocked_smiles,
                    "functional_group": fg["name"],
                    "category": fg["category"],
                    "description": fg["description"],
                    "knocked_atoms": fg["atoms"],
                    "knocked_mol": knocked_mol,
                    "original_mol": mol,
                }
            )

        priority = {fg.name: fg.priority for fg in FUNCTIONAL_GROUPS_LIBRARY}
        return sorted(variants, key=lambda item: priority.get(item["functional_group"], 999))[:max_variants]


class EnsembleRegressor:
    """Average predictions from several sklearn-like regressors."""

    def __init__(self, models: Union[List[Any], Dict[str, Any]], weights: Optional[List[float]] = None):
        self.model_list = list(models.values()) if isinstance(models, dict) else list(models)
        if not self.model_list:
            raise ValueError("At least one model is required.")

        if weights is None:
            self.weights = [1.0 / len(self.model_list)] * len(self.model_list)
        elif len(weights) != len(self.model_list):
            raise ValueError("weights length must match the number of models.")
        else:
            self.weights = weights

    def predict(self, x: np.ndarray) -> np.ndarray:
        preds = np.asarray([model.predict(x) for model in self.model_list])
        return np.average(preds, axis=0, weights=self.weights)


class FeatureBasedActivityChangeQuantifier:
    """Score the predicted activity impact of a functional-group knockout."""

    def __init__(self, regressor: Any):
        self.regressor = regressor

    @staticmethod
    def compute_fingerprint_distance(fp1: np.ndarray, fp2: np.ndarray) -> float:
        dot = float(np.dot(fp1, fp2))
        norm1 = float(np.sum(fp1))
        norm2 = float(np.sum(fp2))
        sim = dot / (norm1 + norm2 - dot + 1e-8)
        return float(1.0 - sim)

    @staticmethod
    def compute_descriptor_delta(smiles_orig: str, smiles_ko: str) -> Dict[str, float]:
        mol_orig = Chem.MolFromSmiles(smiles_orig)
        mol_ko = Chem.MolFromSmiles(smiles_ko)
        names_and_funcs = {
            "delta_logp": Descriptors.MolLogP,
            "delta_tpsa": Descriptors.TPSA,
            "delta_molwt": Descriptors.MolWt,
            "delta_hbd": Descriptors.NumHDonors,
            "delta_hba": Descriptors.NumHAcceptors,
        }
        if mol_orig is None or mol_ko is None:
            return {name: 0.0 for name in names_and_funcs}
        return {name: float(func(mol_ko) - func(mol_orig)) for name, func in names_and_funcs.items()}

    def score_perturbation(
        self,
        variant: Dict[str, Any],
        features_original: np.ndarray,
        features_knockout: np.ndarray,
    ) -> Dict[str, float]:
        activity_original = float(self.regressor.predict(features_original.reshape(1, -1))[0])
        activity_knockout = float(self.regressor.predict(features_knockout.reshape(1, -1))[0])
        activity_change = activity_knockout - activity_original
        activity_importance = -activity_change
        fp_distance = self.compute_fingerprint_distance(features_original, features_knockout)
        desc_delta = self.compute_descriptor_delta(variant["original_smiles"], variant["knockout_smiles"])
        causal_score = (
            0.5 * abs(activity_importance)
            + 0.2 * fp_distance
            + 0.15 * min(abs(desc_delta["delta_logp"]) / 5.0, 1.0)
            + 0.15 * min(abs(desc_delta["delta_tpsa"]) / 100.0, 1.0)
        )
        return {
            "activity_original": activity_original,
            "activity_knockout": activity_knockout,
            "activity_change": float(activity_change),
            "activity_importance": float(activity_importance),
            "is_positive_contribution": bool(activity_importance > 0),
            "fingerprint_distance": float(fp_distance),
            **desc_delta,
            "causal_score": float(causal_score),
        }


class CausalDiscoveryAnalyzer:
    """Summarize functional-group perturbations with statistical tests."""

    def __init__(self, alpha: float = 0.05, min_samples: int = 5):
        self.alpha = alpha
        self.min_samples = min_samples

    def analyze(self, results: Sequence[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(results)
        if df.empty:
            return pd.DataFrame()

        summary = []
        for fg_name, group in df.groupby("functional_group"):
            if len(group) < self.min_samples:
                continue
            importance = group["activity_importance"].astype(float).values
            causal_scores = group["causal_score"].astype(float).values
            t_stat, p_value = stats.ttest_1samp(importance, 0.0)
            mean_imp = float(np.mean(importance))
            std_imp = float(np.std(importance, ddof=1)) if len(importance) > 1 else 0.0
            effect_size = mean_imp / (std_imp + 1e-8)
            summary.append(
                {
                    "functional_group": fg_name,
                    "category": group["category"].iloc[0],
                    "n_molecules": int(len(group)),
                    "mean_activity_importance": mean_imp,
                    "std_activity_importance": std_imp,
                    "median_activity_importance": float(np.median(importance)),
                    "positive_contribution_ratio": float(np.mean(importance > 0)),
                    "mean_causal_score": float(np.mean(causal_scores)),
                    "median_causal_score": float(np.median(causal_scores)),
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": bool(p_value < self.alpha),
                    "effect_size": float(effect_size),
                    "effect_magnitude": self._effect_magnitude(effect_size),
                    "top_molecules": group.nlargest(3, "activity_importance")["original_smiles"].tolist(),
                }
            )

        summary_df = pd.DataFrame(summary)
        if summary_df.empty:
            return summary_df

        summary_df = summary_df.sort_values("mean_activity_importance", ascending=False).reset_index(drop=True)
        summary_df["p_value_corrected"] = self._benjamini_hochberg(summary_df["p_value"].values)
        summary_df["significant_fdr"] = summary_df["p_value_corrected"] < self.alpha
        return summary_df

    @staticmethod
    def _effect_magnitude(effect_size: float) -> str:
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            return "negligible"
        if abs_effect < 0.5:
            return "small"
        if abs_effect < 0.8:
            return "medium"
        return "large"

    @staticmethod
    def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
        p_values = np.asarray(p_values, dtype=float)
        n = len(p_values)
        order = np.argsort(p_values)
        corrected = np.empty(n, dtype=float)
        for rank, idx in enumerate(order, start=1):
            corrected[idx] = min(p_values[idx] * n / rank, 1.0)
        for pos in range(n - 2, -1, -1):
            corrected[order[pos]] = min(corrected[order[pos]], corrected[order[pos + 1]])
        return corrected

    def generate_report(self, summary_df: pd.DataFrame, top_k: int = 10, dataset_name: str = "Dataset") -> str:
        lines = [
            "# FGKP Activity Report",
            "",
            f"Dataset: {dataset_name}",
            f"Functional groups tested: {len(summary_df)}",
            f"Significant groups after FDR correction: {int(summary_df.get('significant_fdr', pd.Series(dtype=bool)).sum())}",
            "",
            f"## Top {top_k} Functional Groups",
        ]
        for idx, row in summary_df.head(top_k).iterrows():
            lines.extend(
                [
                    "",
                    f"{idx + 1}. {row['functional_group']} ({row['category']})",
                    f"   - Mean activity importance: {row['mean_activity_importance']:.4f}",
                    f"   - Positive contribution ratio: {row['positive_contribution_ratio']:.1%}",
                    f"   - Effect size: {row['effect_magnitude']} (d={row['effect_size']:.3f})",
                    f"   - p-value: {row['p_value']:.2e}",
                    f"   - FDR-adjusted p-value: {row.get('p_value_corrected', np.nan):.2e}",
                    f"   - Molecules tested: {row['n_molecules']}",
                ]
            )
        return "\n".join(lines)


class ActivityFGKPEngine:
    """Run activity-oriented functional-group knockout perturbation."""

    def __init__(
        self,
        encoder: Any,
        activity_predictor: Any,
        alpha: float = 0.05,
        min_samples: int = 5,
    ):
        self.encoder = encoder
        self.predictor = activity_predictor
        self.knockout = FunctionalGroupKnockout()
        self.quantifier = FeatureBasedActivityChangeQuantifier(activity_predictor)
        self.analyzer = CausalDiscoveryAnalyzer(alpha=alpha, min_samples=min_samples)
        self.results_cache: List[Dict[str, Any]] = []

    def run(
        self,
        smiles_list: Sequence[str],
        batch_size: int = 32,
        max_variants_per_mol: int = 10,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run knockout generation, encoding, activity scoring, and summary analysis."""
        all_results = []
        iterator: Iterable[str] = tqdm(smiles_list, desc="FGKP perturbation") if verbose else smiles_list

        for smiles in iterator:
            variants = self.knockout.generate_knockout_variants(smiles, max_variants=max_variants_per_mol)
            if not variants:
                continue

            all_smiles = [smiles] + [variant["knockout_smiles"] for variant in variants]
            embeddings, _ = self.encoder.encode_batch(all_smiles, batch_size=batch_size)
            if len(embeddings) < len(all_smiles):
                continue

            emb_original = embeddings[0]
            for idx, variant in enumerate(variants, start=1):
                scores = self.quantifier.score_perturbation(
                    variant,
                    features_original=emb_original,
                    features_knockout=embeddings[idx],
                )
                all_results.append(
                    {
                        "original_smiles": smiles,
                        "knockout_smiles": variant["knockout_smiles"],
                        "functional_group": variant["functional_group"],
                        "category": variant["category"],
                        **scores,
                    }
                )

        raw_df = pd.DataFrame(all_results)
        summary_df = self.analyzer.analyze(all_results)
        self.results_cache = all_results
        return raw_df, summary_df

    def export_results(
        self,
        summary_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        output_dir: Union[str, Path],
        dataset_name: str = "results",
    ) -> None:
        """Write raw perturbations, summary statistics, and a markdown report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_path / f"{dataset_name}_causal_summary.csv", index=False)
        raw_df.to_csv(output_path / f"{dataset_name}_raw_perturbations.csv", index=False)
        report = self.analyzer.generate_report(summary_df, top_k=15, dataset_name=dataset_name)
        (output_path / f"{dataset_name}_report.md").write_text(report, encoding="utf-8")
