# Standard library
import logging
from typing import List, Optional, Sequence, Union

# Third-party
import numpy as np
import pandas as pd
import torch
from numpy.linalg import pinv
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import SaltRemover
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data
from tqdm.auto import tqdm


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


# Copied from the xrx_prf repo
def clean_smiles_openadmet(
    smiles: str,
    remove_hs: bool = True,
    strip_stereochem: bool = False,
    strip_salts: bool = True,
) -> str:
    """Applies preprocessing to SMILES strings, seeking the 'parent' SMILES

    Note that this is different from simply _neutralizing_ the input SMILES - we attempt to get the parent molecule, analogous to a molecular skeleton.
    This is adapted in part from https://rdkit.org/docs/Cookbook.html#neutralizing-molecules

    Args:
        smiles (str): input SMILES
        remove_hs (bool, optional): Removes hydrogens. Defaults to True.
        strip_stereochem (bool, optional): Remove R/S and cis/trans stereochemistry. Defaults to False.
        strip_salts (bool, optional): Remove salt ions. Defaults to True.

    Returns:
        str: cleaned SMILES
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Could not parse SMILES {smiles}"
        if remove_hs:
            mol = Chem.RemoveHs(mol)
        if strip_stereochem:
            Chem.RemoveStereochemistry(mol)
        if strip_salts:
            remover = SaltRemover()  # use default saltremover
            mol = remover.StripMol(mol)  # strip salts

        pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
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
        out_smi = Chem.MolToSmiles(mol, kekuleSmiles=True)  # this also canonicalizes the input
        assert len(out_smi) > 0, f"Could not convert molecule to SMILES {smiles}"
        return out_smi
    except Exception as e:
        print(f"Failed to clean SMILES {smiles} due to {e}")
        return None

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


def one_hot_encoding(x: str, permitted_list: List[str]) -> List[int]:
    """Return a one-hot encoding for ``x`` over a permitted vocabulary.

    Any ``x`` not in ``permitted_list`` is mapped to the last element.

    Args:
        x (str): Input token/value.
        permitted_list (List[str]): Allowed vocabulary.

    Returns:
        List[int]: One-hot vector of length ``len(permitted_list)``.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    return [int(x == s) for s in permitted_list]


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


def get_atom_features(
    atom: Chem.Atom,
    use_chirality: bool = True,
    hydrogens_implicit: bool = True,
) -> np.ndarray:
    """Compute a 1D array of atom features from an RDKit atom.

    Args:
        atom (Chem.Atom): RDKit atom.
        use_chirality (bool, optional): Include chirality (R/S, chiral tag). Defaults to ``True``.
        hydrogens_implicit (bool, optional): Include implicit hydrogen count features.
            Defaults to ``True``.

    Returns:
        np.ndarray: Atom feature vector.
    """
    permitted_list_of_atoms = [
        "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe",
        "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd",
        "Co", "Se", "Ti", "Zn", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn",
        "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown",
    ]
    if not hydrogens_implicit:
        permitted_list_of_atoms = ["H"] + permitted_list_of_atoms

    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    hybridisation_type_enc = one_hot_encoding(
        str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]
    )
    is_in_a_ring_enc = [int(atom.IsInRing())]
    is_aromatic_enc = [int(atom.GetIsAromatic())]

    atom_feature_vector = (
        atom_type_enc
        + n_heavy_neighbors_enc
        + formal_charge_enc
        + hybridisation_type_enc
        + is_in_a_ring_enc
        + is_aromatic_enc
    )

    if use_chirality:
        chirality_type_enc = one_hot_encoding(
            str(atom.GetChiralTag()),
            ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"],
        )
        atom_feature_vector += chirality_type_enc

        cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else "Unknown"
        cip = cip.upper()
        cip_enc = one_hot_encoding(cip, ["R", "S", "UNKNOWN"])
        atom_feature_vector += cip_enc

    if hydrogens_implicit:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)


def get_bond_features(bond: Chem.Bond, use_stereochemistry: bool = True) -> np.ndarray:
    """Compute a 1D array of bond features from an RDKit bond.

    Args:
        bond (Chem.Bond): RDKit bond.
        use_stereochemistry (bool, optional): Include stereo flags (E/Z/any/none).
            Defaults to ``True``.

    Returns:
        np.ndarray: Bond feature vector.
    """
    permitted_list_of_bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]

    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    bond_is_in_ring_enc = [int(bond.IsInRing())]

    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

    if use_stereochemistry:
        stereo_type_enc = one_hot_encoding(
            str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"]
        )
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)


def get_gnn_encodings(mol: Chem.Mol) -> np.ndarray:
    """Compute Gaussian Network Model–style encodings (inverse Kirchhoff).

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
        # Parse SMILES
        smiles = clean_smiles_openadmet(smiles)
        mol = Chem.MolFromSmiles(smiles)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        if mol is None:
            raise ValueError(f"RDKit failed to parse SMILES: {smiles}")

        # Optional GNN-style node augmentation
        dRdR = get_gnn_encodings(mol) if gnn else None

        # Node features
        x_feat = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            atom_features = get_atom_features(atom)
            if dRdR is not None:
                x_feat.append(atom_features + [dRdR[idx][idx]])
            else:
                x_feat.append(atom_features)
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
            edge_attr_feat.append(get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j))))
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

