# Standard library
import logging
import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

# Third-party
import numpy as np
import pandas as pd
import torch
from numpy.linalg import pinv
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdPartialCharges
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data
from tqdm.auto import tqdm

# Local imports from refactored modules
from .atom_features import (
    PERIODIC_TABLE,
    RING_COUNT_CATEGORIES,
    RING_SIZE_CATEGORIES,
    PERIOD_CATEGORIES,
    GROUP_CATEGORIES,
    PERMITTED_ATOMS,
    one_hot_encoding,
    get_period,
    get_group,
    get_atom_features,
    get_atom_feature_dim,
    get_pharmacophore_flags,
)
from .bond_features import (
    get_bond_features,
    get_bond_feature_dim,
)


__SMILES = "c1ccccc1"


def get_node_dim() -> int:
    """Return the dimensionality of the node feature vector.

    Returns:
        int: Number of features per node.
    """
    data = get_tensor_data([__SMILES], [0], pe=False)[0]
    return data.x.size(-1)


def get_edge_dim() -> int:
    """Return the dimensionality of the edge feature vector.

    Returns:
        int: Number of features per edge.
    """
    data = get_tensor_data([__SMILES], [0], pe=False)[0]
    return data.edge_attr.size(-1)


def canonicalize_smiles(
    smiles: str,
    keep_stereo: bool = True,
    keep_charges: bool = True,
    keep_largest_fragment: bool = True,
) -> Optional[str]:
    """Canonicalize a SMILES string with optional fragment/stereo/charge handling.

    This function produces a consistent canonical SMILES representation while
    preserving important chemical information like charges and stereochemistry
    by default.

    Args:
        smiles (str): Input SMILES string.
        keep_stereo (bool, optional): Preserve stereochemistry (@, @@, /, \\).
            Defaults to True.
        keep_charges (bool, optional): Preserve formal charges ([NH4+], [O-], etc.).
            Defaults to True.
        keep_largest_fragment (bool, optional): Keep only the largest fragment
            by heavy atom count (removes salts/counterions). Defaults to True.

    Returns:
        Optional[str]: Canonical SMILES string, or None if parsing fails.

    Examples:
        >>> canonicalize_smiles("[NH4+]")  # Preserves charge
        '[NH4+]'
        >>> canonicalize_smiles("C[C@H](O)F")  # Preserves stereo
        'C[C@H](O)F'
        >>> canonicalize_smiles("CCO.[Na+].[Cl-]")  # Removes salts
        'CCO'
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Remove explicit hydrogens (keep implicit)
        mol = Chem.RemoveHs(mol)

        # Strip stereochemistry if requested
        if not keep_stereo:
            Chem.RemoveStereochemistry(mol)

        # Keep largest fragment (remove salts/counterions)
        if keep_largest_fragment:
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            if frags:
                # Select fragment with most heavy atoms
                sizes = [frag.GetNumHeavyAtoms() for frag in frags]
                mol = frags[sizes.index(max(sizes))]

        # Neutralize charges if requested
        if not keep_charges:
            pattern = Chem.MolFromSmarts(
                "[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]"
            )
            at_matches = mol.GetSubstructMatches(pattern)
            at_matches_list = [y[0] for y in at_matches]
            if len(at_matches_list) > 0:
                for at_idx in at_matches_list:
                    atom = mol.GetAtomWithIdx(at_idx)
                    chg = atom.GetFormalCharge()
                    hcount = atom.GetTotalNumHs()
                    atom.SetFormalCharge(0)
                    atom.SetNumExplicitHs(hcount - chg)
                    atom.UpdatePropertyCache()

        # Generate canonical SMILES
        out_smi = Chem.MolToSmiles(mol, isomericSmiles=keep_stereo, canonical=True)
        if not out_smi:
            return None
        return out_smi

    except Exception as e:
        logging.warning(f"Failed to canonicalize SMILES '{smiles}': {e}")
        return None


def clean_smiles_openadmet(
    smiles: str,
    remove_hs: bool = True,
    strip_stereochem: bool = False,
    strip_salts: bool = True,
) -> Optional[str]:
    """Deprecated. Use canonicalize_smiles instead.

    This function is kept for backward compatibility. It wraps canonicalize_smiles
    with parameters mapped to preserve the original behavior (neutralizes charges).

    Args:
        smiles (str): Input SMILES string.
        remove_hs (bool, optional): Removes hydrogens. Defaults to True.
        strip_stereochem (bool, optional): Remove stereochemistry. Defaults to False.
        strip_salts (bool, optional): Remove salt ions. Defaults to True.

    Returns:
        Optional[str]: Cleaned SMILES string, or None if parsing fails.
    """
    warnings.warn(
        "clean_smiles_openadmet is deprecated, use canonicalize_smiles instead",
        DeprecationWarning,
        stacklevel=2,
    )
    # Map old "strip_*" semantics to new "keep_*" semantics
    keep_stereo = not strip_stereochem
    keep_largest_fragment = strip_salts

    return canonicalize_smiles(
        smiles,
        keep_stereo=keep_stereo,
        keep_charges=False,  # Old behavior neutralized charges
        keep_largest_fragment=keep_largest_fragment,
    )


def clean_df(
    df: pd.DataFrame,
    min_num_atoms: int = 0,
    use_largest_fragment: bool = True,
    x_label: str = "Drug",
    y_label: str = "Y",
) -> pd.DataFrame:
    """Clean a DataFrame of SMILES/labels with fragment handling and size filters.

    Notes:
        * Molecules are sanitized and canonicalized while preserving ionization and stereochemistry.
        * If a molecule has one fragment, it is kept as-is (then canonicalized).
        * If ``use_largest_fragment`` is True and a molecule has multiple fragments, the largest
          fragment (by heavy-atom count) is chosen. No neutralization is performed.

    Args:
        df (pd.DataFrame): Input table containing SMILES and labels.
        min_num_atoms (int, optional): Minimum atom count after fragment selection.
            ``0`` disables size filtering. Defaults to ``0``.
        use_largest_fragment (bool, optional): Whether to select the largest fragment
            for multi-fragment inputs. Defaults to ``True``.
        x_label (str, optional): Column name with input SMILES. Defaults to ``"Drug"``.
        y_label (str, optional): Column name to keep with cleaned SMILES. Defaults to ``"Y"``.

    Returns:
        pd.DataFrame: Cleaned table with columns ``[x_label, y_label]``.
    """

    def to_mol(smi: str):
        if not isinstance(smi, str):
            return None
        try:
            return Chem.MolFromSmiles(smi)  # sanitize=True by default
        except Exception:
            return None

    def count_fragments(mol):
        if mol is None:
            return 0
        return len(Chem.GetMolFrags(mol))

    def get_largest_fragment_mol(mol):
        """Return RDKit Mol of the largest fragment (preserve charges/stereo)."""
        if mol is None:
            return None
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if not frags:
            return None
        sizes = [frag.GetNumHeavyAtoms() for frag in frags]
        return frags[sizes.index(max(sizes))]

    def count_atoms(mol):
        if mol is None:
            return 0
        return mol.GetNumAtoms()

    def canonical_smiles(mol):
        """Canonicalize while preserving ionization and stereochemistry."""
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

    for log_level in RDLogger._levels:
        rdBase.DisableLog(log_level)

    df = df.copy()

    # Convert SMILES -> Mol and drop invalids
    df["mol"] = df[x_label].apply(to_mol)
    df = df[df["mol"].notna()].copy()

    # Descriptor columns
    df["num_frags"] = df["mol"].apply(count_fragments)
    df["num_atoms"] = df["mol"].apply(count_atoms)

    # Fragment handling
    initial_len = len(df)
    if use_largest_fragment:
        def select_clean_mol(mol):
            if mol is None:
                return None
            if count_fragments(mol) <= 1:
                return mol
            return get_largest_fragment_mol(mol)

        df["clean_mol"] = df["mol"].apply(select_clean_mol)
        fragments_removed = 0
    else:
        df = df.query("num_frags == 1").copy()
        df["clean_mol"] = df["mol"]
        fragments_removed = initial_len - len(df)
        logging.info(f"Removed {fragments_removed} compounds with >1 fragment.")

    # Recompute atom counts after fragment selection
    df["num_atoms_clean"] = df["clean_mol"].apply(count_atoms)

    # Atom-count filter
    if min_num_atoms > 0:
        before = len(df)
        df = df.query(f"num_atoms_clean >= {min_num_atoms}").copy()
        removed_cmpds = (before - len(df)) + fragments_removed
        logging.info(
            f"Removed {removed_cmpds} compounds that did not meet atom count >= {min_num_atoms}."
        )

    # Final canonical SMILES
    df[x_label] = df["clean_mol"].apply(canonical_smiles)

    return df[[x_label, y_label]].reset_index(drop=True)


def get_data_from_csv(
    filename: str,
    x_label: str,
    y_label: str,
    sep: str = ",",
    min_num_atoms: int = 0,
    use_largest_fragment: bool = True,
) -> pd.DataFrame:
    """Read a CSV and return a cleaned SMILES/label DataFrame.

    Args:
        filename (str): Path to the CSV file.
        x_label (str): Column name to use as the X variable (SMILES).
        y_label (str): Column name to use as the Y variable (label).
        sep (str, optional): CSV separator. Defaults to ``","``.
        min_num_atoms (int, optional): Minimum atom count filter (see ``clean_df``).
            ``0`` disables size filtering. Defaults to ``0``.
        use_largest_fragment (bool, optional): Whether to select the largest fragment
            for multi-fragment inputs (see ``clean_df``). Defaults to ``True``.

    Returns:
        pd.DataFrame: Cleaned DataFrame with columns ``[x_label, y_label]``.

    Example:
        >>> data = get_data_from_csv("data.csv", "X", "Y")
    """
    df = pd.read_csv(filename, sep=sep)
    df = df[[x_label, y_label]]
    return clean_df(
        df,
        min_num_atoms=min_num_atoms,
        use_largest_fragment=use_largest_fragment,
        x_label=x_label,
        y_label=y_label,
    )


def get_pe(mol: Chem.Mol, pe_dim: int = 6, normalized: bool = True) -> np.ndarray:
    """Compute positional encodings via Laplacian eigenvectors.

    Uses the (optionally normalized) graph Laplacian on the molecular graph and
    returns the first ``pe_dim`` non-trivial eigenvectors.

    Args:
        mol (Chem.Mol): Input molecule.
        pe_dim (int, optional): Number of eigenvectors to keep. Defaults to ``6``.
        normalized (bool, optional): Use normalized Laplacian if True. Defaults to ``True``.

    Returns:
        np.ndarray: Array of shape ``[num_atoms, pe_dim]``.
    """
    adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
    degree = np.diag(np.sum(adj, axis=1))
    laplacian = degree - adj
    if normalized:
        degree_inv_sqrt = np.diag(np.sum(adj, axis=1) ** (-0.5))
        laplacian = degree_inv_sqrt @ laplacian @ degree_inv_sqrt
    try:
        val, vec = np.linalg.eig(laplacian)
    except Exception:
        print(Chem.MolToSmiles(mol))
        raise

    vec = vec[:, np.argsort(val)]
    N = vec.shape[1]
    M = pe_dim + 1
    if N < M:
        vec = np.pad(vec, ((0, 0), (0, M - N)), mode="constant")

    return vec[:, 1:M]


def get_ring_membership_stats(
    mol: Chem.Mol,
) -> (Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]):
    """Precompute ring membership statistics for atoms and bonds.

    Returns:
        atom_ring_stats: dict[atom_idx] -> {
            'count': int,
            'min_size': Optional[int],
            'max_size': Optional[int],
            'has_aromatic': bool,
            'has_non_aromatic': bool,
        }
        bond_ring_stats: dict[bond_idx] -> same structure
    """
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()  # tuple of tuples of atom indices
    bond_rings = ring_info.BondRings()  # tuple of tuples of bond indices

    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()

    atom_ring_stats: Dict[int, Dict[str, Any]] = {
        i: {
            "count": 0,
            "min_size": None,
            "max_size": None,
            "has_aromatic": False,
            "has_non_aromatic": False,
        }
        for i in range(num_atoms)
    }

    bond_ring_stats: Dict[int, Dict[str, Any]] = {
        i: {
            "count": 0,
            "min_size": None,
            "max_size": None,
            "has_aromatic": False,
            "has_non_aromatic": False,
        }
        for i in range(num_bonds)
    }

    # RDKit guarantees AtomRings and BondRings correspond ring-by-ring
    for atom_ring, bond_ring in zip(atom_rings, bond_rings):
        size = len(atom_ring)

        # Define a ring as aromatic if ALL its bonds are aromatic
        is_aromatic_ring = True
        for b_idx in bond_ring:
            if not mol.GetBondWithIdx(b_idx).GetIsAromatic():
                is_aromatic_ring = False
                break

        # Update atom stats
        for a_idx in atom_ring:
            st = atom_ring_stats[a_idx]
            st["count"] += 1
            if st["min_size"] is None or size < st["min_size"]:
                st["min_size"] = size
            if st["max_size"] is None or size > st["max_size"]:
                st["max_size"] = size
            if is_aromatic_ring:
                st["has_aromatic"] = True
            else:
                st["has_non_aromatic"] = True

        # Update bond stats
        for b_idx in bond_ring:
            st = bond_ring_stats[b_idx]
            st["count"] += 1
            if st["min_size"] is None or size < st["min_size"]:
                st["min_size"] = size
            if st["max_size"] is None or size > st["max_size"]:
                st["max_size"] = size
            if is_aromatic_ring:
                st["has_aromatic"] = True
            else:
                st["has_non_aromatic"] = True

    return atom_ring_stats, bond_ring_stats


def get_gnn_encodings(mol: Chem.Mol) -> np.ndarray:
    """Compute Gaussian Network Model-style encodings (inverse Kirchhoff).

    Constructs the adjacency matrix, degree matrix, Laplacian (Kirchhoff) matrix,
    then returns its pseudoinverse.

    Args:
        mol (Chem.Mol): Input molecule.

    Returns:
        np.ndarray: Pseudoinverse of the Kirchhoff matrix with shape ``[N, N]``.
    """
    adjacency_matrix = GetAdjacencyMatrix(mol)
    adjacency_np = np.array(adjacency_matrix)
    degree_matrix = np.diag(np.sum(adjacency_np, axis=1))
    kirchhoff_matrix = degree_matrix - adjacency_np
    inv_kirchhoff_matrix = pinv(kirchhoff_matrix)
    return inv_kirchhoff_matrix


def _to_float_sequence(
    y_val: Union[float, int, Sequence[Optional[float]], np.ndarray]
) -> np.ndarray:
    """Convert a per-sample label into a 1D float array.

    * Single numeric -> shape ``(1,)``
    * Sequence/array -> shape ``(T,)``

    ``None``/``np.nan`` values are kept as ``np.nan`` for later masking.

    Args:
        y_val (Union[float, int, Sequence[Optional[float]], np.ndarray]): Label(s).

    Returns:
        np.ndarray: 1D array of dtype ``float32``.
    """
    if isinstance(y_val, (float, int, np.floating, np.integer)):
        return np.array([float(y_val)], dtype=np.float32)
    arr = np.array(y_val, dtype=np.float32)
    if arr.dtype == object:  # handle None values
        arr = np.array([np.nan if v is None else float(v) for v in y_val], dtype=np.float32)
    return arr


def get_tensor_data(
    x_smiles: List[str],
    y: List[Union[float, int, Sequence[Optional[float]], np.ndarray]],
    gnn: bool = True,
    pe: bool = True,
    pe_dim: int = 6,
) -> List[Data]:
    """Build torch_geometric molecular graphs with labels and masks.

    Each sample is constructed from a SMILES string and a label (single- or multi-task).
    Missing task labels can be ``None``/``np.nan``; a mask ``y_mask`` (1.0=present,
    0.0=missing) is included per sample.

    Args:
        x_smiles (List[str]): SMILES strings.
        y (List[Union[float, int, Sequence[Optional[float]], np.ndarray]]): Per-sample labels:
            single float/int (single-task) or a sequence/array (multi-task).
        gnn (bool, optional): If True, append GNN-style diagonal terms to node features.
            Defaults to ``True``.
        pe (bool, optional): If True, include positional encodings in ``Data.pe``.
            Defaults to ``True``.
        pe_dim (int, optional): Number of PE dimensions to keep. Defaults to ``6``.

    Returns:
        List[Data]: One ``Data`` per sample with fields:
            - ``x`` (torch.FloatTensor): Node features ``[N, F]``.
            - ``edge_index`` (torch.LongTensor): COO edges ``[2, E]``.
            - ``edge_attr`` (torch.FloatTensor): Edge features ``[E, D]``.
            - ``pe`` (torch.FloatTensor | None): Positional encodings ``[N, pe_dim]`` if ``pe=True``.
            - ``y`` (torch.FloatTensor): Task targets ``[T]``.
            - ``y_mask`` (torch.FloatTensor): Mask ``[T]`` (1=present, 0=missing).
    """
    data_list: List[Data] = []

    for smiles, y_val in tqdm(zip(x_smiles, y), total=len(x_smiles), desc="Processing data"):
        # Parse and canonicalize SMILES
        smiles = canonicalize_smiles(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        # Compute Gasteiger partial charges for the entire molecule
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
        except Exception:
            # If Gasteiger computation fails, charges will be 0.0 (handled in get_atom_features)
            pass

        # Compute pharmacophore flags for entire molecule
        pharmacophore_flags = get_pharmacophore_flags(mol)

        # Optional GNN-style node augmentation
        dRdR = get_gnn_encodings(mol) if gnn else None

        # Precompute ring membership stats
        atom_ring_stats, bond_ring_stats = get_ring_membership_stats(mol)

        # Node features
        x_feat = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            atom_features = get_atom_features(
                atom,
                use_chirality=True,
                hydrogens_implicit=True,
                atom_ring_stats=atom_ring_stats,
                pharmacophore_flags=pharmacophore_flags,
            )
            if dRdR is not None:
                x_feat.append(atom_features.tolist() + [dRdR[idx][idx]])
            else:
                x_feat.append(atom_features.tolist())
        x = torch.as_tensor(np.asarray(x_feat), dtype=torch.float)

        # Edges
        rows, cols = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([torch_rows, torch_cols], dim=0)

        # Edge attributes
        edge_attr_feat = []
        if pe:
            pe_numpy = get_pe(mol, pe_dim=pe_dim)
            pe_tensor = torch.as_tensor(pe_numpy, dtype=torch.float)
        else:
            pe_tensor = None

        for i, j in zip(rows, cols):
            bond = mol.GetBondBetweenAtoms(int(i), int(j))
            edge_attr_feat.append(
                get_bond_features(
                    bond,
                    use_stereochemistry=True,
                    bond_ring_stats=bond_ring_stats,
                )
            )
        edge_attr = torch.as_tensor(np.asarray(edge_attr_feat), dtype=torch.float)

        # Labels (multi-task friendly)
        y_arr = _to_float_sequence(y_val)  # [T]
        y_mask_arr = np.isfinite(y_arr).astype(np.float32)
        y_tensor = torch.as_tensor(y_arr, dtype=torch.float)
        y_mask_tensor = torch.as_tensor(y_mask_arr, dtype=torch.float)

        data_list.append(
            Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pe=pe_tensor,
                y=y_tensor,          # [num_tasks]
                y_mask=y_mask_tensor # [num_tasks]
            )
        )

    return data_list
