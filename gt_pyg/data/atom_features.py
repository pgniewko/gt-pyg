# Standard library
import logging
from typing import Any, Dict, List, Optional, Union

# Third-party
import numpy as np
from rdkit import Chem


# -----------------------------
# Pharmacophore SMARTS patterns (precompiled at module load)
# -----------------------------
# H-bond donor: N or O with at least one hydrogen
HBD_SMARTS = Chem.MolFromSmarts("[#7,#8;!H0]")
# H-bond acceptor: N or O that is not part of a hetero=hetero bond
HBA_SMARTS = Chem.MolFromSmarts("[#7,#8;!$([#7,#8]=[!#6])]")
# Hydrophobic: Carbon not attached to heteroatoms
HYDROPHOBIC_SMARTS = Chem.MolFromSmarts("[C;!$(C~[#7,#8,#9,#15,#16,#17,#35,#53])]")
# Positive ionizable: Nitrogen with positive charge or protonated amines
POS_IONIZABLE_SMARTS = Chem.MolFromSmarts("[#7;+,H2,H3]")
# Negative ionizable: Carboxylic acid or carboxylate
NEG_IONIZABLE_SMARTS = Chem.MolFromSmarts("[CX3](=O)[O-,OH]")

# -----------------------------
# Global category constants
# -----------------------------
RING_COUNT_CATEGORIES = [0, 1, 2, 3, "MoreThanThree"]
RING_SIZE_CATEGORIES = [3, 4, 5, 6, 7, 8, 9, 10, "MoreThanTen"]
PERIOD_CATEGORIES = [1, 2, 3, 4, 5, 6, 7, "Unknown"]
# 0 is used for "no group / undefined" (e.g. some f-block elements if RDKit returns 0)
GROUP_CATEGORIES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, "Unknown"]

# Permitted list of atoms for one-hot encoding
PERMITTED_ATOMS = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe",
    "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd",
    "Co", "Se", "Ti", "Zn", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn",
    "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown",
]

# Use RDKit's periodic table helper
PERIODIC_TABLE = Chem.GetPeriodicTable()


def one_hot_encoding(x: Union[str, int, Any], permitted_list: List) -> List[int]:
    """Return a one-hot encoding for ``x`` over a permitted vocabulary.

    Any ``x`` not in ``permitted_list`` is mapped to the last element.

    Args:
        x: Input token/value (str/int/etc.).
        permitted_list (List): Allowed vocabulary.

    Returns:
        List[int]: One-hot vector of length ``len(permitted_list)``.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    return [int(x == s) for s in permitted_list]


def get_gasteiger_charge(atom: Chem.Atom, clip: float = 2.0) -> float:
    """Return the clipped, normalized Gasteiger partial charge for an atom.

    The charge is clipped to [-clip, clip] and then divided by clip
    to produce a value in [-1, 1].

    Note:
        Gasteiger charges must be precomputed on the molecule using
        ``rdPartialCharges.ComputeGasteigerCharges(mol)`` before calling
        this function.

    Args:
        atom (Chem.Atom): RDKit atom (must have _GasteigerCharge property).
        clip (float, optional): Clipping range. Defaults to 2.0.

    Returns:
        float: Normalized Gasteiger charge in [-1, 1], or 0.0 if unavailable.
    """
    try:
        charge = float(atom.GetDoubleProp("_GasteigerCharge"))
        if np.isnan(charge) or np.isinf(charge):
            logging.warning(
                "Gasteiger charge is %s for atom %s (idx %d); defaulting to 0.0",
                "NaN" if np.isnan(charge) else "Inf",
                atom.GetSymbol(),
                atom.GetIdx(),
            )
            return 0.0
        return np.clip(charge, -clip, clip) / clip
    except Exception as e:
        logging.warning(
            "Failed to retrieve Gasteiger charge for atom %s (idx %d): %s",
            atom.GetSymbol(),
            atom.GetIdx(),
            e,
        )
        return 0.0


def get_pharmacophore_flags(mol: Chem.Mol) -> Dict[int, List[int]]:
    """Compute pharmacophore flags for all atoms in a molecule.

    Returns a dictionary mapping atom index to a list of 5 binary flags:
        - [0] Is H-bond donor
        - [1] Is H-bond acceptor
        - [2] Is hydrophobic
        - [3] Is positive ionizable
        - [4] Is negative ionizable

    Args:
        mol (Chem.Mol): RDKit molecule.

    Returns:
        Dict[int, List[int]]: Mapping from atom index to pharmacophore flags.
    """
    num_atoms = mol.GetNumAtoms()
    flags = {i: [0, 0, 0, 0, 0] for i in range(num_atoms)}

    # H-bond donors
    if HBD_SMARTS is not None:
        for match in mol.GetSubstructMatches(HBD_SMARTS):
            for idx in match:
                flags[idx][0] = 1

    # H-bond acceptors
    if HBA_SMARTS is not None:
        for match in mol.GetSubstructMatches(HBA_SMARTS):
            for idx in match:
                flags[idx][1] = 1

    # Hydrophobic
    if HYDROPHOBIC_SMARTS is not None:
        for match in mol.GetSubstructMatches(HYDROPHOBIC_SMARTS):
            for idx in match:
                flags[idx][2] = 1

    # Positive ionizable
    if POS_IONIZABLE_SMARTS is not None:
        for match in mol.GetSubstructMatches(POS_IONIZABLE_SMARTS):
            for idx in match:
                flags[idx][3] = 1

    # Negative ionizable
    if NEG_IONIZABLE_SMARTS is not None:
        for match in mol.GetSubstructMatches(NEG_IONIZABLE_SMARTS):
            for idx in match:
                flags[idx][4] = 1

    return flags


def get_period(atomic_num: int) -> int:
    """Map atomic number to periodic table period (row).

    Args:
        atomic_num (int): Atomic number of the element.

    Returns:
        int: Period (row) of the element in the periodic table (1-7).
    """
    # Period boundaries based on atomic number
    # Period 1: H(1), He(2)
    # Period 2: Li(3) to Ne(10)
    # Period 3: Na(11) to Ar(18)
    # Period 4: K(19) to Kr(36)
    # Period 5: Rb(37) to Xe(54)
    # Period 6: Cs(55) to Rn(86)
    # Period 7: Fr(87) onwards
    if atomic_num <= 0:
        return 1
    elif atomic_num <= 2:
        return 1
    elif atomic_num <= 10:
        return 2
    elif atomic_num <= 18:
        return 3
    elif atomic_num <= 36:
        return 4
    elif atomic_num <= 54:
        return 5
    elif atomic_num <= 86:
        return 6
    else:
        return 7


def get_group(atomic_num: int) -> int:
    """Map atomic number to periodic table group (column).

    For lanthanides/actinides, returns 0 (undefined group).

    Args:
        atomic_num (int): Atomic number of the element.

    Returns:
        int: Group (column) of the element in the periodic table (0-18).
    """
    # Group lookup table for common elements
    # Maps atomic number to group (column)
    group_map = {
        1: 1, 2: 18,  # Period 1
        3: 1, 4: 2, 5: 13, 6: 14, 7: 15, 8: 16, 9: 17, 10: 18,  # Period 2
        11: 1, 12: 2, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18,  # Period 3
        # Period 4 (including transition metals)
        19: 1, 20: 2,  # K, Ca
        21: 3, 22: 4, 23: 5, 24: 6, 25: 7, 26: 8, 27: 9, 28: 10, 29: 11, 30: 12,  # Sc-Zn
        31: 13, 32: 14, 33: 15, 34: 16, 35: 17, 36: 18,  # Ga-Kr
        # Period 5
        37: 1, 38: 2,  # Rb, Sr
        39: 3, 40: 4, 41: 5, 42: 6, 43: 7, 44: 8, 45: 9, 46: 10, 47: 11, 48: 12,  # Y-Cd
        49: 13, 50: 14, 51: 15, 52: 16, 53: 17, 54: 18,  # In-Xe
        # Period 6
        55: 1, 56: 2,  # Cs, Ba
        # Lanthanides (57-71) -> group 0
        72: 4, 73: 5, 74: 6, 75: 7, 76: 8, 77: 9, 78: 10, 79: 11, 80: 12,  # Hf-Hg
        81: 13, 82: 14, 83: 15, 84: 16, 85: 17, 86: 18,  # Tl-Rn
        # Period 7
        87: 1, 88: 2,  # Fr, Ra
        # Actinides (89-103) -> group 0
        104: 4, 105: 5, 106: 6, 107: 7, 108: 8, 109: 9, 110: 10, 111: 11, 112: 12,
        113: 13, 114: 14, 115: 15, 116: 16, 117: 17, 118: 18,
    }
    return group_map.get(atomic_num, 0)


def get_atom_features(
    atom: Chem.Atom,
    use_stereochemistry: bool = True,
    hydrogens_implicit: bool = True,
    atom_ring_stats: Optional[Dict[int, Dict[str, Any]]] = None,
    pharmacophore_flags: Optional[Dict[int, List[int]]] = None,
    gnm_value: Optional[float] = None,
) -> np.ndarray:
    """Compute a 1D array of atom features from an RDKit atom.

    Includes:
        - Element, degree, formal charge, hybridization
        - In-ring flag, aromatic flag
        - Atomic number (scalar)
        - Period (row) one-hot
        - Group / column one-hot
        - Ring membership statistics if provided:
            - ring count (one-hot)
            - min ring size (one-hot)
            - max ring size (one-hot)
            - in any aromatic ring (0/1)
            - in any non-aromatic ring (0/1)
        - Gasteiger partial charge (1 continuous, bounded [-1, 1])
        - Pharmacophore flags (5 binary):
            - H-bond donor, H-bond acceptor, hydrophobic,
              positive ionizable, negative ionizable
        - GNM encoding (1 continuous):
            - Kirchhoff pseudoinverse diagonal value (0.0 when GNM is disabled)

    Args:
        atom (Chem.Atom): RDKit atom.
        use_stereochemistry (bool, optional): Include stereochemistry (R/S, chiral tag). Defaults to ``True``.
        hydrogens_implicit (bool, optional): Include implicit hydrogen count features.
            Defaults to ``True``.
        atom_ring_stats (dict, optional): Precomputed ring stats for atoms.
        pharmacophore_flags (dict, optional): Precomputed pharmacophore flags from
            ``get_pharmacophore_flags(mol)``.
        gnm_value (float, optional): Kirchhoff pseudoinverse diagonal value for
            this atom.  Defaults to ``0.0`` when ``None``.

    Returns:
        np.ndarray: Atom feature vector.
    """
    permitted_list_of_atoms = PERMITTED_ATOMS.copy()
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

    # Atomic number, period, and group (column)
    atomic_num = atom.GetAtomicNum()
    atom_feature_vector += [float(atomic_num)]

    period = get_period(atomic_num)
    period_enc = one_hot_encoding(period, PERIOD_CATEGORIES)
    atom_feature_vector += period_enc

    group = get_group(atomic_num)  # may be 0 for undefined
    group_enc = one_hot_encoding(group, GROUP_CATEGORIES)
    atom_feature_vector += group_enc

    if use_stereochemistry:
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
        n_hydrogens_enc = one_hot_encoding(
            int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"]
        )
        atom_feature_vector += n_hydrogens_enc

    # Ring membership statistics
    # Only if precomputed stats are provided; otherwise, all zeros
    ring_count_enc = [0] * len(RING_COUNT_CATEGORIES)
    min_ring_size_enc = [0] * len(RING_SIZE_CATEGORIES)
    max_ring_size_enc = [0] * len(RING_SIZE_CATEGORIES)
    in_any_aromatic_ring = 0
    in_any_non_aromatic_ring = 0

    if atom_ring_stats is not None:
        idx = atom.GetIdx()
        stats = atom_ring_stats.get(idx)
        if stats is not None:
            # Ring count
            count_val = stats["count"]
            if count_val > 3:
                count_val = "MoreThanThree"
            ring_count_enc = one_hot_encoding(count_val, RING_COUNT_CATEGORIES)

            # Min ring size
            if stats["min_size"] is not None:
                min_size_val = stats["min_size"]
                if min_size_val > 10:
                    min_size_val = "MoreThanTen"
                min_ring_size_enc = one_hot_encoding(min_size_val, RING_SIZE_CATEGORIES)

            # Max ring size
            if stats["max_size"] is not None:
                max_size_val = stats["max_size"]
                if max_size_val > 10:
                    max_size_val = "MoreThanTen"
                max_ring_size_enc = one_hot_encoding(max_size_val, RING_SIZE_CATEGORIES)

            in_any_aromatic_ring = int(stats["has_aromatic"])
            in_any_non_aromatic_ring = int(stats["has_non_aromatic"])

    atom_feature_vector += ring_count_enc
    atom_feature_vector += min_ring_size_enc
    atom_feature_vector += max_ring_size_enc
    atom_feature_vector += [in_any_aromatic_ring, in_any_non_aromatic_ring]

    # Gasteiger partial charge (1 continuous, bounded [-1, 1])
    gasteiger = get_gasteiger_charge(atom)
    atom_feature_vector += [gasteiger]

    # Pharmacophore flags (5 binary)
    if pharmacophore_flags is not None:
        flags = pharmacophore_flags.get(atom.GetIdx(), [0, 0, 0, 0, 0])
    else:
        flags = [0, 0, 0, 0, 0]
    atom_feature_vector += flags

    # GNM encoding (Kirchhoff pseudoinverse diagonal)
    atom_feature_vector += [gnm_value if gnm_value is not None else 0.0]

    return np.array(atom_feature_vector)


def get_atom_feature_dim(
    use_stereochemistry: bool = True,
    hydrogens_implicit: bool = True,
) -> int:
    """Return the dimensionality of the atom feature vector.

    Calculates the expected length of the feature vector based on the
    configuration options.  The returned dimension includes the GNM
    (Kirchhoff pseudoinverse diagonal) term appended by
    :func:`get_tensor_data`.

    Args:
        use_stereochemistry (bool, optional): Whether stereochemistry features are included.
            Defaults to ``True``.
        hydrogens_implicit (bool, optional): Whether implicit hydrogen features are included.
            Defaults to ``True``.

    Returns:
        int: Number of features in the atom feature vector.
    """
    # Use a simple test molecule to compute the dimension
    mol = Chem.MolFromSmiles("C")
    atom = mol.GetAtomWithIdx(0)
    features = get_atom_features(
        atom,
        use_stereochemistry=use_stereochemistry,
        hydrogens_implicit=hydrogens_implicit,
        atom_ring_stats=None,
    )
    return len(features)
