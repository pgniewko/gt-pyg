"""Tests for pharmacophore SMARTS patterns.

Validates HBD, HBA, HYDROPHOBIC, POS_IONIZABLE, and NEG_IONIZABLE
patterns against real drug molecules and simple reference compounds.
"""

from rdkit import Chem

from gt_pyg.data.atom_features import (
    HBA_SMARTS,
    HBD_SMARTS,
    HYDROPHOBIC_SMARTS,
    NEG_IONIZABLE_SMARTS,
    POS_IONIZABLE_SMARTS,
    get_pharmacophore_flags,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _matched_indices(smiles, pattern):
    """Return sorted set of atom indices matched by *pattern* in *smiles*."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Failed to parse SMILES: {smiles}"
    matches = mol.GetSubstructMatches(pattern)
    return sorted({idx for match in matches for idx in match})


def _matched_symbols(smiles, pattern):
    """Return sorted list of (idx, symbol) tuples matched by *pattern*."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Failed to parse SMILES: {smiles}"
    matches = mol.GetSubstructMatches(pattern)
    indices = sorted({idx for match in matches for idx in match})
    return [(idx, mol.GetAtomWithIdx(idx).GetSymbol()) for idx in indices]


def _atom_index_by_symbol(smiles, symbol, occurrence=0):
    """Return the atom index of the *occurrence*-th atom with *symbol*."""
    mol = Chem.MolFromSmiles(smiles)
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == symbol:
            if count == occurrence:
                return atom.GetIdx()
            count += 1
    raise ValueError(f"Atom {symbol} (occurrence {occurrence}) not found in {smiles}")


# ---------------------------------------------------------------------------
# SMARTS compilation
# ---------------------------------------------------------------------------

class TestSmartsCompilation:
    """All five patterns must compile to valid query molecules."""

    def test_hbd_compiles(self):
        assert HBD_SMARTS is not None

    def test_hba_compiles(self):
        assert HBA_SMARTS is not None

    def test_hydrophobic_compiles(self):
        assert HYDROPHOBIC_SMARTS is not None

    def test_pos_ionizable_compiles(self):
        assert POS_IONIZABLE_SMARTS is not None

    def test_neg_ionizable_compiles(self):
        assert NEG_IONIZABLE_SMARTS is not None


# ---------------------------------------------------------------------------
# HBD — H-bond donors
# ---------------------------------------------------------------------------

class TestHBD:
    """H-bond donor: N-H (trivalent/protonated), O-H, S-H, aromatic N-H."""

    def test_ethanol_oh(self):
        """Ethanol O-H should be a donor."""
        hits = _matched_symbols("CCO", HBD_SMARTS)
        assert any(sym == "O" for _, sym in hits)

    def test_captopril_thiol(self):
        """Captopril thiol S-H should be a donor."""
        smiles = "OC(=O)[C@@H](CS)C1CCCC(=O)N1C"
        hits = _matched_symbols(smiles, HBD_SMARTS)
        assert any(sym == "S" for _, sym in hits)

    def test_indole_aromatic_nh(self):
        """Indole aromatic NH should be a donor."""
        hits = _matched_symbols("c1ccc2[nH]ccc2c1", HBD_SMARTS)
        assert any(sym == "N" for _, sym in hits)

    def test_primary_amine(self):
        """Primary amine N-H should be a donor."""
        hits = _matched_symbols("CCN", HBD_SMARTS)
        assert any(sym == "N" for _, sym in hits)

    def test_ether_oxygen_not_donor(self):
        """Ether oxygen (no H) should NOT be a donor."""
        hits = _matched_symbols("CCOCC", HBD_SMARTS)
        assert not any(sym == "O" for _, sym in hits)

    def test_ketone_oxygen_not_donor(self):
        """Ketone C=O should NOT be a donor."""
        hits = _matched_symbols("CC(=O)C", HBD_SMARTS)
        assert not any(sym == "O" for _, sym in hits)

    def test_amide_carbonyl_not_donor(self):
        """Amide C=O oxygen should NOT be a donor."""
        smiles = "CC(=O)N"
        mol = Chem.MolFromSmiles(smiles)
        matches = mol.GetSubstructMatches(HBD_SMARTS)
        matched_idx = {idx for match in matches for idx in match}
        # O at idx 2 (the carbonyl) should not be matched
        o_idx = _atom_index_by_symbol(smiles, "O")
        assert o_idx not in matched_idx


# ---------------------------------------------------------------------------
# HBA — H-bond acceptors
# ---------------------------------------------------------------------------

class TestHBA:
    """H-bond acceptor: divalent O/S, charged O/S, trivalent N (not amide),
    aromatic heteroatoms."""

    def test_celecoxib_sulfonyl_oxygen(self):
        """Celecoxib S=O oxygens should be acceptors."""
        smiles = "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"
        hits = _matched_symbols(smiles, HBA_SMARTS)
        o_hits = [idx for idx, sym in hits if sym == "O"]
        assert len(o_hits) >= 2, "S=O oxygens should be matched as acceptors"

    def test_pyridine_nitrogen(self):
        """Pyridine ring nitrogen should be an acceptor."""
        hits = _matched_symbols("c1ccncc1", HBA_SMARTS)
        assert any(sym == "N" for _, sym in hits)

    def test_diethyl_ether_oxygen(self):
        """Diethyl ether oxygen should be an acceptor."""
        hits = _matched_symbols("CCOCC", HBA_SMARTS)
        assert any(sym == "O" for _, sym in hits)

    def test_carboxylate_oxygen(self):
        """Carboxylate O- should be an acceptor."""
        hits = _matched_symbols("CC(=O)[O-]", HBA_SMARTS)
        o_hits = [idx for idx, sym in hits if sym == "O"]
        assert len(o_hits) >= 1

    def test_amide_nitrogen_not_acceptor(self):
        """Amide nitrogen should NOT be an acceptor (lone pair delocalized)."""
        smiles = "CC(=O)N"
        hits = _matched_symbols(smiles, HBA_SMARTS)
        assert not any(sym == "N" for _, sym in hits)

    def test_indole_nh_not_acceptor(self):
        """Indole pyrrole-type NH should NOT be an acceptor."""
        smiles = "c1ccc2[nH]ccc2c1"
        hits = _matched_symbols(smiles, HBA_SMARTS)
        assert not any(sym == "N" for _, sym in hits)

    def test_bortezomib_amide_n_not_acceptor(self):
        """Bortezomib amide nitrogens should NOT be acceptors."""
        smiles = "CC(C)C[C@@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)c1cnccn1)B(O)O"
        mol = Chem.MolFromSmiles(smiles)
        matches = mol.GetSubstructMatches(HBA_SMARTS)
        matched_idx = {idx for match in matches for idx in match}
        # Amide N atoms (idx 5 and 16) should not be matched
        n5 = 5   # first amide N
        n16 = 16  # second amide N
        assert n5 not in matched_idx, "Amide N (idx 5) should not be an acceptor"
        assert n16 not in matched_idx, "Amide N (idx 16) should not be an acceptor"


# ---------------------------------------------------------------------------
# POS_IONIZABLE — positive ionizable
# ---------------------------------------------------------------------------

class TestPosIonizable:
    """Positive ionizable: basic amines, protonated N, imidazole, guanidine."""

    def test_metformin_guanidine(self):
        """Metformin guanidine nitrogens should be positive ionizable."""
        smiles = "CN(C)C(=N)NC(=N)N"
        hits = _matched_indices(smiles, POS_IONIZABLE_SMARTS)
        assert len(hits) >= 2, "Guanidine should have multiple ionizable N"

    def test_histamine_imidazole(self):
        """Histamine imidazole ring should be positive ionizable."""
        smiles = "NCCc1c[nH]cn1"
        hits = _matched_indices(smiles, POS_IONIZABLE_SMARTS)
        assert len(hits) >= 1

    def test_ethylamine(self):
        """Ethylamine NH2 should be positive ionizable."""
        hits = _matched_symbols("CCN", POS_IONIZABLE_SMARTS)
        assert any(sym == "N" for _, sym in hits)

    def test_protonated_ammonium(self):
        """Protonated ammonium [NH3+] should be positive ionizable."""
        hits = _matched_symbols("CC[NH3+]", POS_IONIZABLE_SMARTS)
        assert any(sym == "N" for _, sym in hits)

    def test_acetamide_not_ionizable(self):
        """Acetamide NH2 should NOT be positive ionizable (pKa ~ -1)."""
        smiles = "CC(=O)N"
        hits = _matched_symbols(smiles, POS_IONIZABLE_SMARTS)
        assert not any(sym == "N" for _, sym in hits)

    def test_aniline_not_ionizable(self):
        """Aniline NH2 should NOT be positive ionizable (pKa ~ 4.6)."""
        smiles = "Nc1ccccc1"
        hits = _matched_symbols(smiles, POS_IONIZABLE_SMARTS)
        assert not any(sym == "N" for _, sym in hits)

    def test_nitrobenzene_not_ionizable(self):
        """Nitro group N+ should NOT be positive ionizable."""
        smiles = "[O-][N+](=O)c1ccccc1"
        hits = _matched_symbols(smiles, POS_IONIZABLE_SMARTS)
        assert not any(sym == "N" for _, sym in hits)

    def test_celecoxib_sulfonamide_not_ionizable(self):
        """Celecoxib sulfonamide N should NOT be positive ionizable."""
        smiles = "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"
        mol = Chem.MolFromSmiles(smiles)
        matches = mol.GetSubstructMatches(POS_IONIZABLE_SMARTS)
        matched_idx = {idx for match in matches for idx in match}
        # Sulfonamide N is idx 19
        assert 19 not in matched_idx


# ---------------------------------------------------------------------------
# NEG_IONIZABLE — negative ionizable
# ---------------------------------------------------------------------------

class TestNegIonizable:
    """Negative ionizable: carboxylic/sulfonic acids, phosphates, tetrazoles,
    sulfonamide NH, boronic acids."""

    def test_aspirin_carboxylic_acid(self):
        """Aspirin carboxylic acid should be negative ionizable."""
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        hits = _matched_indices(smiles, NEG_IONIZABLE_SMARTS)
        assert len(hits) >= 1

    def test_tenofovir_phosphonate(self):
        """Tenofovir phosphonate should be negative ionizable."""
        smiles = "C1=NC2=C(N1COCOP(=O)(O)O)N=CN=C2N"
        hits = _matched_symbols(smiles, NEG_IONIZABLE_SMARTS)
        assert any(sym == "P" for _, sym in hits)

    def test_losartan_tetrazole(self):
        """Losartan tetrazole should be negative ionizable."""
        smiles = "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2n[nH]nn2)cc1"
        hits = _matched_indices(smiles, NEG_IONIZABLE_SMARTS)
        assert len(hits) >= 1, "Tetrazole should be matched as neg ionizable"

    def test_bortezomib_boronic_acid(self):
        """Bortezomib boronic acid should be negative ionizable."""
        smiles = "CC(C)C[C@@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)c1cnccn1)B(O)O"
        hits = _matched_symbols(smiles, NEG_IONIZABLE_SMARTS)
        assert any(sym == "B" for _, sym in hits)

    def test_phenol_not_neg_ionizable(self):
        """Phenol O-H should NOT be negative ionizable."""
        hits = _matched_indices("Oc1ccccc1", NEG_IONIZABLE_SMARTS)
        assert len(hits) == 0

    def test_ketone_not_neg_ionizable(self):
        """Ketone C=O should NOT be negative ionizable."""
        hits = _matched_indices("CC(=O)C", NEG_IONIZABLE_SMARTS)
        assert len(hits) == 0

    def test_amide_not_neg_ionizable(self):
        """Amide C=O should NOT be negative ionizable."""
        hits = _matched_indices("CC(=O)N", NEG_IONIZABLE_SMARTS)
        assert len(hits) == 0


# ---------------------------------------------------------------------------
# HYDROPHOBIC
# ---------------------------------------------------------------------------

class TestHydrophobic:
    """Hydrophobic: any neutral carbon not bonded to N, O, or F."""

    def test_cyclohexane_all_carbons(self):
        """Cyclohexane: all 6 carbons should be hydrophobic."""
        hits = _matched_indices("C1CCCCC1", HYDROPHOBIC_SMARTS)
        assert len(hits) == 6

    def test_toluene_all_carbons(self):
        """Toluene: all 7 carbons (methyl + aromatic ring) should be hydrophobic."""
        hits = _matched_indices("Cc1ccccc1", HYDROPHOBIC_SMARTS)
        assert len(hits) == 7

    def test_naphthalene_all_carbons(self):
        """Naphthalene: all 10 aromatic carbons should be hydrophobic."""
        hits = _matched_indices("c1ccc2ccccc2c1", HYDROPHOBIC_SMARTS)
        assert len(hits) == 10

    def test_indole_carbons_not_nitrogen(self):
        """Indole: carbons not bonded to N are hydrophobic; N and its
        direct C neighbors are excluded."""
        smiles = "c1ccc2[nH]ccc2c1"
        mol = Chem.MolFromSmiles(smiles)
        hits = _matched_indices(smiles, HYDROPHOBIC_SMARTS)
        matched_symbols = {mol.GetAtomWithIdx(i).GetSymbol() for i in hits}
        assert "N" not in matched_symbols
        # 8 carbons total, but 2 are bonded to N → 6 hydrophobic
        assert len(hits) == 6

    def test_chlorobenzene_ring_carbons(self):
        """Chlorobenzene: ring carbons should be hydrophobic (Cl is not C)."""
        smiles = "Clc1ccccc1"
        hits = _matched_symbols(smiles, HYDROPHOBIC_SMARTS)
        # All 6 aromatic carbons should match; Cl should not
        c_hits = [(idx, sym) for idx, sym in hits if sym == "C"]
        assert len(c_hits) == 6
        assert not any(sym == "Cl" for _, sym in hits)

    def test_ethanol_methyl_only(self):
        """Ethanol: only the terminal methyl C (not C bonded to O) is hydrophobic."""
        smiles = "CCO"  # idx 0:C, 1:C, 2:O
        hits = _matched_indices(smiles, HYDROPHOBIC_SMARTS)
        assert 0 in hits, "Terminal methyl C should be hydrophobic"
        assert 1 not in hits, "C bonded to O should NOT be hydrophobic"

    def test_phenol_c1_not_hydrophobic(self):
        """Phenol: C bonded to O should NOT be hydrophobic."""
        smiles = "Oc1ccccc1"  # idx 0:O, 1:C(bonded to O), ...
        mol = Chem.MolFromSmiles(smiles)
        hits = _matched_indices(smiles, HYDROPHOBIC_SMARTS)
        # C at idx 1 is bonded to O, should not be hydrophobic
        assert 1 not in hits

    def test_aniline_c1_not_hydrophobic(self):
        """Aniline: C bonded to N should NOT be hydrophobic."""
        smiles = "Nc1ccccc1"  # idx 0:N, 1:C(bonded to N), ...
        hits = _matched_indices(smiles, HYDROPHOBIC_SMARTS)
        assert 1 not in hits

    def test_cf3_carbons_not_hydrophobic(self):
        """CF3 group: carbon bonded to F should NOT be hydrophobic."""
        smiles = "FC(F)(F)C"  # idx 0:F, 1:C(CF3), 2:F, 3:F, 4:C(methyl)
        hits = _matched_indices(smiles, HYDROPHOBIC_SMARTS)
        assert 1 not in hits, "CF3 carbon should NOT be hydrophobic"
        assert 4 in hits, "Methyl carbon should be hydrophobic"

    def test_no_nitrogen_matched(self):
        """Nitrogen should never be flagged hydrophobic."""
        for smiles in ["CCN", "c1ccncc1", "NCCc1c[nH]cn1"]:
            hits = _matched_symbols(smiles, HYDROPHOBIC_SMARTS)
            assert not any(sym == "N" for _, sym in hits), (
                f"N should not be hydrophobic in {smiles}"
            )

    def test_no_oxygen_matched(self):
        """Oxygen should never be flagged hydrophobic."""
        for smiles in ["CCO", "CCOCC", "CC(=O)O"]:
            hits = _matched_symbols(smiles, HYDROPHOBIC_SMARTS)
            assert not any(sym == "O" for _, sym in hits), (
                f"O should not be hydrophobic in {smiles}"
            )


# ---------------------------------------------------------------------------
# Integration: get_pharmacophore_flags
# ---------------------------------------------------------------------------

class TestGetPharmacophoreFlags:
    """Test the high-level get_pharmacophore_flags() function."""

    def test_returns_all_atoms(self):
        """Should return flags for every atom in the molecule."""
        mol = Chem.MolFromSmiles("CCO")
        flags = get_pharmacophore_flags(mol)
        assert len(flags) == mol.GetNumAtoms()

    def test_flag_vector_length(self):
        """Each atom should have exactly 5 flags."""
        mol = Chem.MolFromSmiles("CCO")
        flags = get_pharmacophore_flags(mol)
        for idx, vec in flags.items():
            assert len(vec) == 5, f"Atom {idx} has {len(vec)} flags, expected 5"

    def test_ethanol_flags(self):
        """Ethanol: O is donor+acceptor, terminal C is hydrophobic."""
        mol = Chem.MolFromSmiles("CCO")  # 0:C, 1:C, 2:O
        flags = get_pharmacophore_flags(mol)
        # O (idx 2): HBD=1, HBA=1
        assert flags[2][0] == 1, "O should be HBD"
        assert flags[2][1] == 1, "O should be HBA"
        # Terminal C (idx 0): hydrophobic=1
        assert flags[0][2] == 1, "Terminal C should be hydrophobic"

    def test_pyridine_flags(self):
        """Pyridine N: should be HBA, not HBD, not POS_IONIZABLE."""
        mol = Chem.MolFromSmiles("c1ccncc1")
        flags = get_pharmacophore_flags(mol)
        n_idx = _atom_index_by_symbol("c1ccncc1", "N")
        assert flags[n_idx][0] == 0, "Pyridine N should NOT be HBD"
        assert flags[n_idx][1] == 1, "Pyridine N should be HBA"
        assert flags[n_idx][3] == 0, "Pyridine N should NOT be POS_IONIZABLE"

    def test_naphthalene_all_hydrophobic(self):
        """Naphthalene: all atoms should be hydrophobic."""
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")
        flags = get_pharmacophore_flags(mol)
        for idx in range(mol.GetNumAtoms()):
            assert flags[idx][2] == 1, f"Atom {idx} should be hydrophobic"
