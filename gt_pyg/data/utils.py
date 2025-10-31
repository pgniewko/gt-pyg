# Standard
import logging
from typing import List
from typing import Tuple

# Third party
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from numpy.linalg import pinv
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
from torch_geometric.data import Data
from rdkit import RDLogger
from rdkit import rdBase


__SMILES = "c1ccccc1"


def get_node_dim() -> int:
    """
    Retrieves the dimensionality of the node feature vector.

    Returns:
        int: The dimensionality of the node feature vector.
    """
    data = get_tensor_data([__SMILES], [0], pe=False)[0]
    return data.x.size(-1)


def get_edge_dim() -> int:
    """
    Retrieves the dimensionality of the edge feature vector.

    Returns:
        int: The dimensionality of the edge feature vector.
    """
    data = get_tensor_data([__SMILES], [0], pe=False)[0]
    return data.edge_attr.size(-1)


def clean_df(
    df: pd.DataFrame,
    min_num_atoms: int = 0,
    use_largest_fragment: bool = True,
    x_label: str = "Drug",
    y_label: str = "Y",
) -> pd.DataFrame:
    """
    Clean a DataFrame containing chemical structures by removing rows that do not meet certain criteria.

    Notes:
      - Molecules are sanitized and canonicalized while preserving ionization state and stereochemistry.
      - If a molecule has only one fragment, the resulting SMILES corresponds to the same molecule
        (possibly only canonicalized, not neutralized or stereochem-flattened).
      - If use_largest_fragment is True and a molecule has multiple fragments, the largest fragment is chosen
        (no neutralization; stereochemistry preserved).

    Args:
        df: The input DataFrame containing chemical structures.
        min_num_atoms: Minimum number of atoms required (0 means no size filtering).
        use_largest_fragment: Whether to use the largest fragment for multi-fragment inputs.
        x_label: Column name containing input SMILES.
        y_label: Column name to keep alongside cleaned SMILES.

    Returns:
        pd.DataFrame with columns [x_label, y_label].
    """

    # --- Helpers ---
    def to_mol(smi: str):
        if not isinstance(smi, str):
            return None
        try:
            # sanitize=True is default; ensures valence checks etc.
            return Chem.MolFromSmiles(smi)
        except Exception:
            return None

    def count_fragments(mol):
        if mol is None:
            return 0
        return len(Chem.GetMolFrags(mol))

    def get_largest_fragment_mol(mol):
        """Return RDKit Mol of the largest fragment (preserve charges/stereo; don't remove Hs)."""
        if mol is None:
            return None
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if not frags:
            return None
        # Use heavy atom count for size
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
        # isomericSmiles=True preserves stereochemistry; canonical ordering is default
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

    # --- Quiet RDKit noise ---
    for log_level in RDLogger._levels:
        rdBase.DisableLog(log_level)

    # --- Work on a copy ---
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
        # For multi-fragment molecules, take largest fragment; for single-fragment, keep the same molecule.
        def select_clean_mol(mol):
            if mol is None:
                return None
            if count_fragments(mol) <= 1:
                return mol  # same molecule (only canonicalized later)
            return get_largest_fragment_mol(mol)

        df["clean_mol"] = df["mol"].apply(select_clean_mol)
        fragments_removed = 0  # not strictly removing rows here
    else:
        # Keep only single-fragment molecules
        df = df.query("num_frags == 1").copy()
        df["clean_mol"] = df["mol"]
        fragments_removed = initial_len - len(df)
        logging.info(f"Removed {fragments_removed} compounds with >1 fragment.")

    # Recompute atom counts on the selected/clean mols (in case fragment choice changed size)
    df["num_atoms_clean"] = df["clean_mol"].apply(count_atoms)

    # Atom-count filter
    if min_num_atoms > 0:
        before = len(df)
        df = df.query(f"num_atoms_clean >= {min_num_atoms}").copy()
        removed_cmpds = (before - len(df)) + fragments_removed
        logging.info(
            f"Removed {removed_cmpds} compounds that did not meet atom count >= {min_num_atoms}."
        )

    # Produce final canonical SMILES, preserving ionization and stereochemistry
    df[x_label] = df["clean_mol"].apply(canonical_smiles)

    # Return just the requested columns
    return df[[x_label, y_label]].reset_index(drop=True)


def get_train_valid_test_data(endpoint: str, min_num_atoms: int = 0, use_largest_fragment: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Retrieves and cleans the train, validation, and test data for a specific endpoint in the ADME dataset.

    Args:
        endpoint (str): The name of the endpoint in the ADME dataset.
        min_num_atoms (int, optional): The minimum number of atoms required for a structure to be considered valid.
            Set to 0 for no size-based filtering. Defaults to 0.
        use_largest_fragment (bool, optional): Whether to use the largest fragment when cleaning the data.
            Defaults to True.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing cleaned train, validation, and test DataFrames.

    """
    try:
        from tdc.single_pred import ADME
    except ImportError:
        raise

    data = ADME(name=endpoint)
    splits = data.get_split()

    train_data = clean_df(splits["train"], min_num_atoms=min_num_atoms, use_largest_fragment=use_largest_fragment)
    valid_data = clean_df(splits["valid"], min_num_atoms=min_num_atoms, use_largest_fragment=use_largest_fragment)
    test_data = clean_df(splits["test"], min_num_atoms=min_num_atoms, use_largest_fragment=use_largest_fragment)

    return (train_data, valid_data, test_data)


def get_molecule_ace_datasets(
    dataset: str,
    training_fraction: float = 1.0,
    valid_fraction: float = 0.0,
    seed: int = 42,
    min_num_atoms: int = 0,
    use_largest_fragment: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Retrieve molecule ACE datasets.

    Args:
        dataset (str): Name of the dataset.
        training_fraction (float, optional): Fraction of data to use for training. Defaults to 1.0.
        valid_fraction (float, optional): Fraction of data to use for validation. Defaults to 0.0.
        seed (int, optional): Random seed for shuffling. Defaults to 42.
        min_num_atoms (int, optional): Minimum number of atoms required for a molecule to be included. Defaults to 0.
        use_largest_fragment (bool, optional): Whether to use the largest fragment of a molecule. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing train, validation, and test datasets as pandas DataFrames.
    """
    try:
        from MoleculeACE import Data
    except ImportError:
        logging.info('Importing MoleculeACE failed')
        raise

    data = Data(dataset)
    df_train_tmp = pd.DataFrame({'SMILES': data.smiles_train, 'Y': data.y_train})
    df_test = pd.DataFrame({'SMILES': data.smiles_test, 'Y': data.y_test})

    shuffled_df = df_train_tmp.sample(frac=1, random_state=seed)

    ratio = training_fraction / (training_fraction + valid_fraction)
    # Calculate the split index based on the ratio
    split_index = int(ratio * len(shuffled_df))

    # Split the shuffled DataFrame into two separate DataFrames
    df_train = shuffled_df.iloc[:split_index]
    df_valid = shuffled_df.iloc[split_index:]

    train_data = clean_df(df_train, min_num_atoms=min_num_atoms, use_largest_fragment=use_largest_fragment, x_label="SMILES")
    valid_data = clean_df(df_valid, min_num_atoms=min_num_atoms, use_largest_fragment=use_largest_fragment, x_label="SMILES")
    test_data = clean_df(df_test, min_num_atoms=min_num_atoms, use_largest_fragment=use_largest_fragment, x_label="SMILES")

    return (train_data, valid_data, test_data)


def get_data_from_csv(filename: str, x_label: str, y_label: str, sep: str = ',', min_num_atoms: int = 0, use_largest_fragment: bool = True) -> pd.DataFrame:
    """
    Reads data from a CSV file and returns a cleaned DataFrame containing specified columns.

    Parameters:
        filename (str): Path to the CSV file.
        x_label (str): Label of the column to be used as the X variable.
        y_label (str): Label of the column to be used as the Y variable.
        sep (str, optional): Separator used in the CSV file. Default is ','.
        min_num_atoms (int, optional): The minimum number of atoms required for a structure to be considered valid.
            Set to 0 for no size-based filtering. Defaults to 0.
        use_largest_fragment (bool, optional): Whether to use the largest fragment when cleaning the data.
            Defaults to True.

    Returns:
        pandas.DataFrame: A cleaned DataFrame containing only the specified X and Y columns.

    Example:
        data = get_data_from_csv('data.csv', 'X', 'Y')
    """
    df = pd.read_csv(filename, sep=sep)
    df = df[[x_label, y_label]]

    data = clean_df(df,
                    min_num_atoms=min_num_atoms,
                    use_largest_fragment=use_largest_fragment,
                    x_label=x_label,
                    y_label=y_label)
    return data


def one_hot_encoding(x: str, permitted_list: List[str]) -> List[int]:
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.

    Args:
        x: The input element to be encoded.
        permitted_list: The list of permitted elements.

    Returns:
        binary_encoding: A list representing the binary encoding of the input element.
    """
    if x not in permitted_list:
        x = permitted_list[-1]

    binary_encoding = [
        int(boolean_value)
        for boolean_value in list(map(lambda s: x == s, permitted_list))
    ]

    return binary_encoding


def get_pe(mol: Chem.Mol, pe_dim: int = 6, normalized: bool = True) -> np.ndarray:
    """
    Calculates the graph signal using the normalized Laplacian.

    Args:
        mol: The input molecule.
        pe_dim: The number of dimensions to keep in the graph signal. Defaults to 6.
        normalized: Specifies whether to use normalized Laplacian. Defaults to True.

    Returns:
        np.ndarray: The graph signal of the molecule.
    """
    adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
    degree = np.diag(np.sum(adj, axis=1))
    laplacian = degree - adj
    if normalized:
        degree_inv_sqrt = np.diag(np.sum(adj, axis=1) ** (-1.0 / 2.0))
        laplacian = degree_inv_sqrt @ laplacian @ degree_inv_sqrt
    try:
        val, vec = np.linalg.eig(laplacian)
    except:
        print(Chem.MolToSmiles(mol))
        raise

    vec = vec[:, np.argsort(val)]
    N = vec.shape[1]
    M = pe_dim + 1
    if N < M:
        vec = np.pad(vec, ((0, 0), (0, M - N)), mode='constant')

    return vec[:, 1:M]


def get_atom_features(atom: Chem.Atom, use_chirality: bool = True, hydrogens_implicit: bool = True) -> np.ndarray:
    """
    Computes a 1D numpy array of atom features from an RDKit atom object.

    Args:
        atom (Chem.Atom): The RDKit atom object.
        use_chirality (bool, optional): Specifies whether to include chirality information. Defaults to True.
        hydrogens_implicit (bool, optional): Specifies whether to include implicit hydrogen count. Defaults to True.

    Returns:
        np.ndarray: A 1D numpy array representing the atom features.
    """
    permitted_list_of_atoms = [
        "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe",
        "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd",
        "Co", "Se", "Ti", "Zn", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn",
        "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown"
    ]

    if not hydrogens_implicit:
        permitted_list_of_atoms = ["H"] + permitted_list_of_atoms

    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)

    n_heavy_neighbors_enc = one_hot_encoding(
        int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"]
    )

    formal_charge_enc = one_hot_encoding(
        int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"]
    )

    hybridisation_type_enc = one_hot_encoding(
        str(atom.GetHybridization()),
        ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"],
    )

    is_in_a_ring_enc = [int(atom.IsInRing())]

    is_aromatic_enc = [int(atom.GetIsAromatic())]

#    atomic_mass_scaled = [float((atom.GetMass() - 10.812) / 116.092)]
#
#    vdw_radius_scaled = [
#        float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6)
#    ]
#
#    covalent_radius_scaled = [
#        float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76)
#    ]

    atom_feature_vector = (
        atom_type_enc
        + n_heavy_neighbors_enc
        + formal_charge_enc
        + hybridisation_type_enc
        + is_in_a_ring_enc
        + is_aromatic_enc
#        + atomic_mass_scaled
#        + vdw_radius_scaled
#        + covalent_radius_scaled
    )

    if use_chirality:
        chirality_type_enc = one_hot_encoding(
            str(atom.GetChiralTag()),
            [
                "CHI_UNSPECIFIED",
                "CHI_TETRAHEDRAL_CW",
                "CHI_TETRAHEDRAL_CCW",
                "CHI_OTHER",
            ],
        )
        atom_feature_vector += chirality_type_enc

    if hydrogens_implicit:
        n_hydrogens_enc = one_hot_encoding(
            int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"]
        )
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)


def get_bond_features(bond: Chem.Bond, use_stereochemistry: bool = True) -> np.ndarray:
    """
    Takes an RDKit bond object as input and gives a 1D numpy array of bond features as output.

    Args:
        bond (Chem.Bond): The RDKit bond object to extract features from.
        use_stereochemistry (bool, optional): Specifies whether to include stereochemistry features.
            Defaults to True.

    Returns:
        np.ndarray: A 1D numpy array of bond features.
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


def get_gnn_encodings(mol):
    # Generate adjacency matrix
    adjacency_matrix = Chem.GetAdjacencyMatrix(mol)

    # Convert adjacency matrix to numpy array
    adjacency_np = np.array(adjacency_matrix)

    # Calculate the degree matrix
    degree_matrix = np.diag(np.sum(adjacency_np, axis=1))

    # Calculate the Laplacian matrix (Kirchhoff matrix)
    kirchhoff_matrix = degree_matrix - adjacency_np

    # Calculate the inverse of the Kirchhoff matrix
    inv_kirchhoff_matrix = pinv(kirchhoff_matrix)

    return inv_kirchhoff_matrix


def get_tensor_data(x_smiles: List[str], y: List[float], gnn: bool = True, pe: bool = True, pe_dim: int = 6) -> List[Data]:
    """
    Constructs labeled molecular graphs in the form of torch_geometric.data.Data objects
    using SMILES strings and associated numerical labels.

    Args:
        x_smiles (List[str]): A list of SMILES strings.
        y (List[float]): A list of numerical labels for the SMILES strings (e.g., associated pKi values).
        gnn (bool, optional): Use Gaussian Network Model style positional encoding.
        pe (bool, optional): Specifies whether to include graph signal (PE) features. Defaults to True.
        pe_dim (int, optional): The number of dimensions to keep in the graph signal. Defaults to 6.

    Returns:
        List[Data]: A list of torch_geometric.data.Data objects representing labeled molecular graphs.
    """

    data_list = []

    for (smiles, y_val) in tqdm(zip(x_smiles, y), desc="Processing data"):
        # convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(smiles)

        if gnn:
            dRdR = get_gnn_encodings(mol)
        else:
            dRdR = None

        # get feature dimensions
        x = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            atom_features = get_atom_features(atom)

            if dRdR is not None:
                x.append(atom_features + [dRdR[idx][idx]])
            else:
                x.append(atom_features)

        x = torch.as_tensor(np.array(x), dtype=torch.float)

        # construct edge index array edge_index of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([torch_rows, torch_cols], dim=0)

        edge_attr = []
        if pe:
            pe_numpy = get_pe(mol, pe_dim=pe_dim)
            pe_tensor = torch.as_tensor(pe_numpy, dtype=torch.float)
        else:
            pe_tensor = None

        for k, (i, j) in enumerate(zip(rows, cols)):
            edge_attr.append(get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j))))

        edge_attr = torch.as_tensor(np.array(edge_attr), dtype=torch.float)

        # construct label tensor
        y_tensor = torch.as_tensor([y_val], dtype=torch.float)

        # construct Pytorch Geometric data object and append to data list
        data_list.append(
            Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pe=pe_tensor,
                y=y_tensor,
            )
        )

    return data_list
