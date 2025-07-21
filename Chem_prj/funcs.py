from rdkit import Chem


def substructure_search(list_of_smiles: list[str], substructure: str) -> list:
    """
    Substructure search function.
    Takes list of SMILES strings and a substructure as a SMILES string.
    Returns list of SMILES strings where substructure was found
    """

    substructure = Chem.MolFromSmiles(substructure)
    if not substructure:
        raise ValueError(f"Invalid substructure SMILES: {substructure}")

    return [smile
            for smile in list_of_smiles
            if Chem.MolFromSmiles(smile).HasSubstructMatch(substructure)]
