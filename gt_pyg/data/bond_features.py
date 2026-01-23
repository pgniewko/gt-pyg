# Standard library
from typing import Any, Dict, Optional

# Third-party
import numpy as np
from rdkit import Chem

# Local imports - shared constants and utilities
from .atom_features import (
    RING_COUNT_CATEGORIES,
    RING_SIZE_CATEGORIES,
    one_hot_encoding,
)


def get_bond_features(
    bond: Chem.Bond,
    use_stereochemistry: bool = True,
    bond_ring_stats: Optional[Dict[int, Dict[str, Any]]] = None,
) -> np.ndarray:
    """Compute a 1D array of bond features from an RDKit bond.

    Includes:
        - Bond type, conjugation, in-ring flag
        - Optional stereo flags
        - Ring membership statistics if provided:
            - ring count (one-hot)
            - min ring size (one-hot)
            - max ring size (one-hot)
            - in any aromatic ring (0/1)
            - in any non-aromatic ring (0/1)

    Args:
        bond (Chem.Bond): RDKit bond.
        use_stereochemistry (bool, optional): Include stereo flags (E/Z/any/none).
            Defaults to ``True``.
        bond_ring_stats (dict, optional): Precomputed ring stats for bonds.

    Returns:
        np.ndarray: Bond feature vector.
    """
    permitted_list_of_bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
        Chem.rdchem.BondType.OTHER,
    ]

    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    bond_is_in_ring_enc = [int(bond.IsInRing())]

    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

    if use_stereochemistry:
        stereo_type_enc = one_hot_encoding(
            str(bond.GetStereo()),
            ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE", "STEREOCIS", "STEREOTRANS", "OTHER"],
        )
        bond_feature_vector += stereo_type_enc

    # Ring membership statistics
    ring_count_enc = [0] * len(RING_COUNT_CATEGORIES)
    min_ring_size_enc = [0] * len(RING_SIZE_CATEGORIES)
    max_ring_size_enc = [0] * len(RING_SIZE_CATEGORIES)
    in_any_aromatic_ring = 0
    in_any_non_aromatic_ring = 0

    if bond_ring_stats is not None:
        idx = bond.GetIdx()
        stats = bond_ring_stats.get(idx)
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

    bond_feature_vector += ring_count_enc
    bond_feature_vector += min_ring_size_enc
    bond_feature_vector += max_ring_size_enc
    bond_feature_vector += [in_any_aromatic_ring, in_any_non_aromatic_ring]

    return np.array(bond_feature_vector)


def get_bond_feature_dim(use_stereochemistry: bool = True) -> int:
    """Return the dimensionality of the bond feature vector.

    Calculates the expected length of the feature vector based on the
    configuration options.

    Args:
        use_stereochemistry (bool, optional): Whether stereochemistry features are included.
            Defaults to ``True``.

    Returns:
        int: Number of features in the bond feature vector.
    """
    # Use a simple test molecule with a bond to compute the dimension
    mol = Chem.MolFromSmiles("CC")
    bond = mol.GetBondWithIdx(0)
    features = get_bond_features(
        bond,
        use_stereochemistry=use_stereochemistry,
        bond_ring_stats=None,
    )
    return len(features)
