"""Data constants"""

from rdkit import Chem


def compile_smarts(pattern: str) -> Chem.Mol:
    """Compile SMARTS, raising ValueError if invalid."""
    query = Chem.MolFromSmarts(pattern)
    if query is None:
        raise ValueError(f"Invalid SMARTS pattern: {pattern}")
    return query


# -----------------------------
# Pharmacophore SMARTS patterns (precompiled at module load)
# -----------------------------
# H-bond donor: N-H (trivalent or protonated), O-H, S-H, aromatic N-H
# Based on RDKit Lipinski/Gobbi donor definition
HBD_SMARTS = compile_smarts(
    "[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]"
)
# H-bond acceptor: divalent O/S, charged O/S, trivalent N (not amide), aromatic heteroatoms
# Adapted from RDKit Lipinski HAcceptorSmarts (rev. Nov 2008)
HBA_SMARTS = compile_smarts(
    "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),"
    "$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]"
)
# Hydrophobic: any neutral carbon not bonded to N, O, or F
# Aligned with RDKit BaseFeatures.fdef Carbon_NonPolar definition
HYDROPHOBIC_SMARTS = compile_smarts("[#6;+0;!$([#6]~[#7,#8,#9])]")
# Positive ionizable: basic amines (not amides/anilines), protonated N,
#   imidazole, guanidine
# Adapted from RDKit BaseFeatures.fdef
POS_IONIZABLE_SMARTS = compile_smarts(
    "[$([N;H2&+0][C;!$(C=O)]),"  # primary amine (not amide)
    "$([N;H1&+0]([C;!$(C=O)])[C;!$(C=O)]),"  # secondary amine (not amide)
    "$([N;H0&+0]([C;!$(C=O)])([C;!$(C=O)])[C;!$(C=O)]),"  # tertiary amine
    "$([#7;+;!$([N+]-[O-])]),"  # already protonated (not nitro)
    "$(c1c[nH]cn1),"  # imidazole
    "$(NC(=N)N)"  # guanidine
    ";!$(N[a])]"  # exclude anilines
)
# Negative ionizable: carboxylic/sulfonic acids, phosphates, tetrazoles,
#   sulfonamide NH, boronic acids
# Extends RDKit BaseFeatures.fdef AcidicGroup
NEG_IONIZABLE_SMARTS = compile_smarts(
    "[$([C,S](=[O,S,P])-[O;H1,H0&-1]),"  # carboxylic, sulfonic, sulfinic acids
    "$([P](=[O])(-[O;H1,H0&-1])(-[O,C])-[O,C]),"  # phosphates/phosphonates
    "$(c1[nH]nnn1),$(c1nn[nH]n1),"  # tetrazole (both tautomers)
    "$([NH]S(=O)(=O)),"  # sulfonamide NH
    "$([B]([O;H1])([O;H1]))]"  # boronic acid
)

NEUTRALIZE_SMARTS = compile_smarts(
    "[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]"
)

# -----------------------------
# Global category constants
# -----------------------------
RING_COUNT_CATEGORIES = (0, 1, 2, 3, "MoreThanThree")
RING_SIZE_CATEGORIES = (3, 4, 5, 6, 7, 8, 9, 10, "MoreThanTen")
PERIOD_CATEGORIES = (0, 1, 2, 3, 4, 5, 6, 7)
# 0 is used for "no group / undefined" (e.g. some f-block elements if RDKit returns 0)
GROUP_CATEGORIES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
DEGREE_CATEGORIES = (0, 1, 2, 3, 4, "MoreThanFour")
FORMAL_CHARGE_CATEGORIES = (-3, -2, -1, 0, 1, 2, 3, "Extreme")
HYBRIDIZATION_CATEGORIES = ("S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER")
CHIRALITY_CATEGORIES = ("CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER")
CIP_CATEGORIES = ("R", "S", "UNKNOWN")
NUM_HS_CATEGORIES = (0, 1, 2, 3, 4, "MoreThanFour")

# Permitted list of atoms for one-hot encoding
PERMITTED_ATOMS = (
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe",
    "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd",
    "Co", "Se", "Ti", "Zn", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn",
    "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown",
)
