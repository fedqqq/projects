from funcs import substructure_search
from rdkit import Chem


class Molecule:
    """
    Molecule class:
    self.mol_id: unique id of molecule (int),
    self.name: name of molecule (str),
    self.formula: formula of molecule (str),
    self.smiles: SMILE structure of molecule (str),
    self.description: description of molecule (str)
    """

    def __init__(self, mol_id: int, name: str, formula: str, smiles: str, description: str | None = None) \
            -> None:
        self.mol_id = mol_id
        self.name = name
        self.formula = formula
        self.smiles = smiles
        if not Chem.MolFromSmiles(smiles):
            raise ValueError(f"Invalid SMILES: {smiles}")
        self.description = description

    def is_substructure(self, list_of_smiles: list):
        """
        Performs substructure search with a molecule object
        """
        return substructure_search(list_of_smiles, self.smiles)
