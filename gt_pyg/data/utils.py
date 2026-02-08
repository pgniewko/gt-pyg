# Standard library
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Third-party
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdPartialCharges
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data
from tqdm.auto import tqdm

# Local imports from refactored modules
from .atom_features import (
    get_atom_features,
    get_pharmacophore_flags,
)
from .bond_features import (
    get_bond_features,
)




def _canonicalize_mol(
    smiles: str,
    keep_stereo: bool = True,
    keep_charges: bool = True,
    keep_largest_fragment: bool = True,
) -> Optional[Chem.Mol]:
    """Parse and clean a SMILES string, returning the canonicalized Mol object.

    Args:
        smiles (str): Input SMILES string.
        keep_stereo (bool, optional): Preserve stereochemistry. Defaults to True.
        keep_charges (bool, optional): Preserve formal charges. Defaults to True.
        keep_largest_fragment (bool, optional): Keep only the largest fragment.
            Defaults to True.

    Returns:
        Optional[Chem.Mol]: Canonicalized molecule, or None if parsing fails.
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
                    new_hcount = hcount - chg
                    if new_hcount < 0:
                        logging.warning(
                            "Charge neutralization would set negative H count "
                            "(%d) on atom %d; clamping to 0",
                            new_hcount, at_idx,
                        )
                        new_hcount = 0
                    atom.SetNumExplicitHs(new_hcount)
                    atom.UpdatePropertyCache()

        return mol

    except Exception as e:
        logging.warning(f"Failed to canonicalize SMILES '{smiles}': {e}")
        return None


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
    mol = _canonicalize_mol(smiles, keep_stereo, keep_charges, keep_largest_fragment)
    if mol is None:
        return None
    out_smi = Chem.MolToSmiles(mol, isomericSmiles=keep_stereo, canonical=True)
    return out_smi or None


def get_ring_membership_stats(
    mol: Chem.Mol,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """Precompute ring membership statistics for atoms and bonds.

    Args:
        mol (Chem.Mol): RDKit molecule.

    Returns:
        Tuple of (atom_ring_stats, bond_ring_stats). Each is a dict
        mapping index to ``{'count', 'min_size', 'max_size',
        'has_aromatic', 'has_non_aromatic'}``.
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


def get_gnm_encodings(adjacency: np.ndarray) -> np.ndarray:
    """Compute Gaussian Network Model (GNM) diagonal encodings.

    Builds the Kirchhoff (graph Laplacian) matrix from the adjacency matrix
    and returns the diagonal of its pseudoinverse via eigendecomposition.

    Args:
        adjacency (np.ndarray): Adjacency matrix with shape ``[N, N]``.

    Returns:
        np.ndarray: Diagonal of the Kirchhoff pseudoinverse with shape ``[N]``.
    """
    degree = np.diag(adjacency.sum(axis=1))
    kirchhoff = degree - adjacency
    eigenvalues, eigenvectors = np.linalg.eigh(kirchhoff)
    # Invert non-zero eigenvalues (skip the near-zero translational mode)
    mask = np.abs(eigenvalues) > 1e-10
    inv_eigenvalues = np.zeros_like(eigenvalues)
    np.divide(1.0, eigenvalues, out=inv_eigenvalues, where=mask)
    # Diagonal of Q @ diag(inv_eigenvalues) @ Q^T = sum of Q_ij^2 * inv_eigenvalues_j
    return (eigenvectors ** 2) @ inv_eigenvalues


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
    gnm: bool = True,
) -> List[Data]:
    """Build torch_geometric molecular graphs with labels and masks.

    Each sample is constructed from a SMILES string and a label (single- or multi-task).
    Missing task labels can be ``None``/``np.nan``; a mask ``y_mask`` (1.0=present,
    0.0=missing) is included per sample.

    Args:
        x_smiles (List[str]): SMILES strings.
        y (List[Union[float, int, Sequence[Optional[float]], np.ndarray]]): Per-sample labels:
            single float/int (single-task) or a sequence/array (multi-task).
        gnm (bool, optional): If True, compute the GNM (Kirchhoff pseudoinverse
            diagonal) and populate the corresponding node feature.  When False
            the feature is left at ``0.0``.  Defaults to ``True``.

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
        # Parse and canonicalize SMILES (single parse)
        mol = _canonicalize_mol(smiles)
        if mol is None:
            raise ValueError(f"Failed to canonicalize SMILES: {smiles}")
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        # Compute Gasteiger partial charges for the entire molecule
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
        except Exception as e:
            logging.warning("Gasteiger charge computation failed for '%s': %s", smiles, e)

        # Compute pharmacophore flags for entire molecule
        pharmacophore_flags = get_pharmacophore_flags(mol)

        # Compute adjacency matrix once (used by GNM and edge_index)
        adjacency = np.array(GetAdjacencyMatrix(mol))

        # Optional GNM-style node augmentation
        gnm_diag = get_gnm_encodings(adjacency) if gnm else None

        # Precompute ring membership stats
        atom_ring_stats, bond_ring_stats = get_ring_membership_stats(mol)

        # Node features
        x_feat = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            atom_feats = get_atom_features(
                atom,
                use_stereochemistry=True,
                hydrogens_implicit=True,
                atom_ring_stats=atom_ring_stats,
                pharmacophore_flags=pharmacophore_flags,
                gnm_value=gnm_diag[idx] if gnm_diag is not None else None,
            )
            x_feat.append(atom_feats.tolist())
        x = torch.as_tensor(np.asarray(x_feat), dtype=torch.float)

        # Edges
        rows, cols = np.nonzero(adjacency)
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
