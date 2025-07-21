import pytest
from funcs import substructure_search


def test_valid_match():
    smiles_list = ["CCO", "CCN", "CC(=O)O"]
    substructure = "C=O"
    result = substructure_search(smiles_list, substructure)
    assert result == ["CC(=O)O"]


def test_no_match():
    smiles_list = ["CCO", "CCC", "O"]
    substructure = "N"
    result = substructure_search(smiles_list, substructure)
    assert result == []


def test_multiple_matches():
    smiles_list = ["CCO", "CCN", "NC=O"]
    substructure = "CN"
    result = substructure_search(smiles_list, substructure)
    assert result == ["CCN", "NC=O"]


def test_empty_smiles_list():
    result = substructure_search([], "C")
    assert result == []


def test_invalid_substructure():
    with pytest.raises(ValueError):
        substructure_search(["CCO"], "InvalidSmiles")


def test_invalid_smiles_in_list():
    with pytest.raises(Exception):
        substructure_search(["CCO", "InvalidSmiles"], "C")


def test_aromaticity_handling():
    smiles_list = ["c1ccccc1", "C1CCCC1"]
    substructure = "c1ccccc1"
    result = substructure_search(smiles_list, substructure)
    assert result == ["c1ccccc1"]


def test_stereochemistry_sensitivity():
    smiles_list = ["C[C@@H](O)CC", "C[C@H](O)CC", "CC(O)CC"]
    substructure = "CC(O)C"
    result = substructure_search(smiles_list, substructure)
    assert set(result) == set(smiles_list)


def test_full_molecule_match():
    smiles_list = ["CCO", "CCN"]
    substructure = "CCO"
    result = substructure_search(smiles_list, substructure)
    assert result == ["CCO"]


def test_single_atom_match():
    smiles_list = ["CCO", "CCN", "O"]
    substructure = "O"
    result = substructure_search(smiles_list, substructure)
    assert result == ["CCO", "O"]
