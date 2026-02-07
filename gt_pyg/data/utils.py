# Standard library
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Third-party
import numpy as np
import torch
from numpy.linalg import pinv
from rdkit import Chem
from rdkit.Chem import rdPartialCharges
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


def get_node_dim(gnn: bool = True) -> int:
    """Return the dimensionality of the node feature vector.

    This is equivalent to ``get_atom_feature_dim(gnn=gnn)`` and is provided
    as a convenience alias.

    Args:
        gnn (bool, optional): Whether the GNN (Kirchhoff diagonal) encoding is
            included.  Must match the ``gnn`` flag passed to
            :func:`get_tensor_data`.  Defaults to ``True``.

    Returns:
        int: Number of features per node.
    """
    return get_atom_feature_dim(gnn=gnn)


def get_edge_dim() -> int:
    """Return the dimensionality of the edge feature vector.

    Returns:
        int: Number of features per edge.
    """
    data = get_tensor_data([__SMILES], [0])[0]
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


def get_ring_membership_stats(
    mol: Chem.Mol,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
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

    Returns:
        List[Data]: One ``Data`` per sample with fields:
            - ``x`` (torch.FloatTensor): Node features ``[N, F]``.
            - ``edge_index`` (torch.LongTensor): COO edges ``[2, E]``.
            - ``edge_attr`` (torch.FloatTensor): Edge features ``[E, D]``.
            - ``y`` (torch.FloatTensor): Task targets ``[T]``.
            - ``y_mask`` (torch.FloatTensor): Mask ``[T]`` (1=present, 0=missing).
    """
    data_list: List[Data] = []

    for smiles, y_val in tqdm(zip(x_smiles, y), total=len(x_smiles), desc="Processing data"):
        # Parse and canonicalize SMILES
        raw_smiles = smiles
        smiles = canonicalize_smiles(smiles)
        if smiles is None:
            raise ValueError(f"Failed to canonicalize SMILES: {raw_smiles}")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit failed to parse SMILES: {smiles} (original: {raw_smiles})")
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
            atom_feats = get_atom_features(
                atom,
                use_chirality=True,
                hydrogens_implicit=True,
                atom_ring_stats=atom_ring_stats,
                pharmacophore_flags=pharmacophore_flags,
                gnn_value=dRdR[idx][idx] if dRdR is not None else None,
            )
            x_feat.append(atom_feats.tolist())
        x = torch.as_tensor(np.asarray(x_feat), dtype=torch.float)

        # Edges
        rows, cols = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([torch_rows, torch_cols], dim=0)

        # Edge attributes
        edge_attr_feat = []
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
                y=y_tensor,          # [num_tasks]
                y_mask=y_mask_tensor # [num_tasks]
            )
        )

    return data_list
