import json

import rdkit.Chem
import rdkit.Chem.Descriptors
from langchain_core.tools import tool


@tool
def molecular_properties(smiles: str) -> str:
    """Calculate molecular properties of a given SMILES string."""
    try:
        smiles = smiles.split(",")
        properties = [
            "MolWt",
            "HeavyAtomMolWt",
            "ExactMolWt",
            "NumValenceElectrons",
            "NumRadicalElectrons",
            "MaxPartialCharge",
            "MinPartialCharge",
            "MaxAbsPartialCharge",
            "MinAbsPartialCharge",
            "FpDensityMorgan1",
            "FpDensityMorgan2",
            "FpDensityMorgan3",
            "Chi0",
            "Chi0n",
            "Chi0v",
            "Chi1",
            "Chi1n",
            "Chi1v",
            "Chi2n",
            "Chi2v",
            "Chi3n",
            "Chi3v",
            "Chi4n",
            "Chi4v",
            "HallKierAlpha",
            "Kappa1",
            "Kappa2",
            "Kappa3",
            "LabuteASA",
            "TPSA",
            "SlogP_VSA1",
            "SlogP_VSA2",
            "SlogP_VSA3",
            "SlogP_VSA4",
            "SlogP_VSA5",
            "SlogP_VSA6",
            "SlogP_VSA7",
            "SlogP_VSA8",
            "SlogP_VSA9",
            "SlogP_VSA10",
            "SlogP_VSA11",
            "SlogP_VSA12",
            "SlogP_VSA13",
            "SlogP_VSA14",
            "SlogP_VSA15",
            "SlogP_VSA16",
            "SlogP_VSA17",
            "SlogP_VSA18",
            "SlogP_VSA19",
            "SlogP_VSA20",
            "SMR_VSA1",
            "SMR_VSA2",
            "SMR_VSA3",
            "SMR_VSA4",
            "SMR_VSA5",
            "SMR_VSA6",
            "SMR_VSA7",
            "SMR_VSA8",
            "SMR_VSA9",
            "SMR_VSA10",
            "SMR_VSA11",
            "SMR_VSA12",
            "SMR_VSA13",
            "SMR_VSA14",
            "SMR_VSA15",
            "SMR_VSA16",
            "SMR_VSA17",
            "SMR_VSA18",
            "SMR_VSA19",
            "SMR_VSA20",
            "SlogP",
            "MR",
            "BalabanJ",
            "BertzCT",
            "Ipc",
            "HallKierAlpha",
            "Kappa1",
            "Kappa2",
            "Kappa3",
            "Chi0",
            "Chi0n",
            "Chi0v",
            "Chi1",
            "Chi1n",
            "Chi1v",
            "Chi2n",
            "Chi2v",
            "Chi3n",
            "Chi3v",
            "Chi4n",
            "Chi4v",
            "BertzCT",
            "HallKierAlpha",
            "Kappa1",
            "Kappa2",
            "Kappa3",
            "Chi0",
            "Chi0n",
            "Chi0v",
            "Chi1",
            "Chi1n",
            "Chi1v",
            "Chi2n",
            "Chi2v",
            "Chi3n",
        ]
        mols = [rdkit.Chem.MolFromSmiles(smile.strip()) for smile in smiles]
        return json.dumps(
            {
                smile: {
                    prop: round(getattr(rdkit.Chem.Descriptors, prop)(mol), 2)
                    for prop in properties
                    if hasattr(rdkit.Chem.Descriptors, prop)
                }
                for smile, mol in zip(smiles, mols)
                if mol
            }
        )
    except Exception as e:
        return str(e)


@tool
def molecular_atom_properties(smiles: str) -> str:
    """Calculate molecular atom properties of a given SMILES string."""
    try:
        smiles = smiles.split(",")
        mols = [rdkit.Chem.MolFromSmiles(smile.strip()) for smile in smiles]
        return json.dumps(
            {
                smile: [
                    {
                        "symbol": atom.GetSymbol(),
                        "atomic_number": atom.GetAtomicNum(),
                        "degree": atom.GetDegree(),
                        "explicit_valence": atom.GetExplicitValence(),
                        "implicit_valence": atom.GetImplicitValence(),
                        "num_radical_electrons": atom.GetNumRadicalElectrons(),
                        "num_explicit_hs": atom.GetNumExplicitHs(),
                        "num_implicit_hs": atom.GetNumImplicitHs(),
                        "mass": atom.GetMass(),
                    }
                    for atom in mol.GetAtoms()
                ]
                for smile, mol in zip(smiles, mols)
                if mol
            }
        )
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    print(molecular_properties.invoke({"smiles": "Cl2"}))
