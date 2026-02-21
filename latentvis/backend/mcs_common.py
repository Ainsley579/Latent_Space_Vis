"""
Maximum common substructure (MCS) utilities using RDKit.

Provides a robust, chemistry-expert-approved implementation for computing
the MCS across a set of SMILES, with configurable atom/bond comparison
and ring/valence options. Use this module to avoid overlapping or
ambiguous chemical formula / substructure display in DimeNet (and other)
embedding visualization pipelines.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple, Optional

from rdkit import Chem
from rdkit.Chem import rdFMCS


def common_substructure(
    smiles_list: Iterable[str],
    *,
    timeout_s: int = 30,
    ring_matches_ring_only: bool = True,
    complete_rings_only: bool = True,
    match_valences: bool = True,
    atom_compare: str = "elements",
    bond_compare: str = "order",
) -> Tuple[Chem.Mol, str]:
    """Compute the maximum common substructure (MCS) across a set of SMILES.

    Args:
        smiles_list: An iterable of SMILES strings.
        timeout_s: Maximum time (seconds) RDKit may spend searching for an MCS.
        ring_matches_ring_only: Require ring atoms/bonds to match ring atoms/bonds.
        complete_rings_only: Only allow ring matches that include complete rings.
        match_valences: Consider valence information when matching atoms.
        atom_compare: One of {"elements", "isotopes", "any"}.
        bond_compare: One of {"order", "any"}.

    Returns:
        A tuple of (mcs_mol, mcs_smarts) where:
            mcs_mol is an RDKit Mol for the MCS pattern,
            mcs_smarts is the SMARTS string for the MCS pattern.

    Raises:
        ValueError: If fewer than two valid molecules are provided or if MCS fails.
    """
    smiles_list = list(smiles_list)
    mols: List[Chem.Mol] = []
    bad: List[str] = []

    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            bad.append(s)
        else:
            mols.append(m)

    if bad:
        preview = bad[:5]
        more = "..." if len(bad) > 5 else ""
        raise ValueError(f"Failed to parse {len(bad)} SMILES: {preview}{more}")

    if len(mols) < 2:
        raise ValueError(
            "Need at least two valid molecules to compute a common substructure."
        )

    atom_cmp_map = {
        "elements": rdFMCS.AtomCompare.CompareElements,
        "isotopes": rdFMCS.AtomCompare.CompareIsotopes,
        "any": rdFMCS.AtomCompare.CompareAny,
    }
    bond_cmp_map = {
        "order": rdFMCS.BondCompare.CompareOrder,
        "any": rdFMCS.BondCompare.CompareAny,
    }

    if atom_compare not in atom_cmp_map:
        raise ValueError(
            f"atom_compare must be one of {set(atom_cmp_map)}, got {atom_compare!r}"
        )
    if bond_compare not in bond_cmp_map:
        raise ValueError(
            f"bond_compare must be one of {set(bond_cmp_map)}, got {bond_compare!r}"
        )

    res = rdFMCS.FindMCS(
        mols,
        timeout=timeout_s,
        ringMatchesRingOnly=ring_matches_ring_only,
        completeRingsOnly=complete_rings_only,
        matchValences=match_valences,
        atomCompare=atom_cmp_map[atom_compare],
        bondCompare=bond_cmp_map[bond_compare],
    )

    if not res.smartsString:
        raise ValueError("MCS search returned an empty SMARTS string.")

    mcs_mol = Chem.MolFromSmarts(res.smartsString)
    if mcs_mol is None:
        raise ValueError("Could not convert MCS SMARTS to an RDKit Mol.")

    return mcs_mol, res.smartsString


def common_substructure_smarts_smiles(
    smiles_list: Iterable[str],
    *,
    timeout_s: int = 30,
    ring_matches_ring_only: bool = True,
    complete_rings_only: bool = True,
    match_valences: bool = True,
    atom_compare: str = "elements",
    bond_compare: str = "order",
) -> Tuple[Optional[str], Optional[str]]:
    """Compute MCS and return (mcs_smarts, mcs_smiles). Does not raise; returns (None, None) on failure.

    Convenience wrapper for pipelines that need smarts + smiles and prefer
    graceful degradation over exceptions.
    """
    try:
        mcs_mol, mcs_smarts = common_substructure(
            smiles_list,
            timeout_s=timeout_s,
            ring_matches_ring_only=ring_matches_ring_only,
            complete_rings_only=complete_rings_only,
            match_valences=match_valences,
            atom_compare=atom_compare,
            bond_compare=bond_compare,
        )
        mcs_smiles = Chem.MolToSmiles(mcs_mol) if mcs_mol else None
        return mcs_smarts, mcs_smiles
    except (ValueError, Exception):
        return None, None
