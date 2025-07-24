import json
import logging
from time import sleep
from typing import Iterable
from urllib.parse import quote

import pubchempy as pcp
import requests
from langchain_core.tools import tool
from more_itertools import unique

LOGGER = logging.getLogger(__name__)

MAX_COUNT = 5
PREVIOUS_SEARCHES = set()


def fetch_compounds(compound_name: str) -> list[pcp.Compound]:
    try:
        return [pcp.Compound.from_cid(int(compound_name))]
    except ValueError:
        cids = pcp.get_cids(compound_name, "name", "compound", list_return="flat")
        return [pcp.Compound.from_cid(cid) for cid in cids[:MAX_COUNT]]


def search_for_smiles(smiles: str) -> list[dict]:
    try:
        for _ in range(3):
            try:
                compounds = pcp.get_compounds(smiles, "smiles")
                if len(compounds) == 0:
                    compounds = [c for c in fetch_compounds_by_autocomplete(smiles)]
                return format_compounds(compounds)
            except pcp.PubChemHTTPError as e:
                LOGGER.error(e)
                sleep(1)
    except ValueError:
        LOGGER.error(f"Could not find compound for smiles: {smiles}")
        compounds = [c for c in fetch_compounds_by_autocomplete(smiles)]
        return format_compounds(compounds)


def fetch_compounds_by_autocomplete(compound_name: str) -> list[pcp.Compound]:
    compound_name_encoded = quote(compound_name)
    names = (
        requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/compound,gene,taxonomy/{compound_name_encoded}"
        )
        .json()
        .get("dictionary_terms", {})
        .get("compound", [])
    )
    cids = []
    for name in names:
        cids += pcp.get_cids(name, "name", "compound", list_return="flat")
    return [pcp.Compound.from_cid(cid) for cid in cids[:MAX_COUNT]]


def retrieve_description(compound: pcp.Compound) -> (str, str):
    information = requests.get(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{compound.cid}/description/JSON"
    ).json()["InformationList"]["Information"]
    title = "\n".join(i["Title"] for i in information if "Title" in i)
    description = "\n".join(i["Description"] for i in information if "Description" in i)
    return title, description


def format_compounds(compounds: Iterable[pcp.Compound]) -> list[dict]:
    compounds = list(unique(compounds, key=lambda r: r.cid))
    descriptions = [retrieve_description(c) for c in compounds]
    return [
        {
            "Smiles": compound.isomeric_smiles,
            "Compound": title,
            "Source": f"https://pubchem.ncbi.nlm.nih.gov/compound/{compound.cid}",
            "CID": compound.cid,
            "Molecular Formula": compound.molecular_formula,
            "Molecular Weight": compound.molecular_weight,
            "Synonyms": compound.synonyms[:3],
            "Description": description,
        }
        for compound, (title, description) in zip(compounds, descriptions)
    ]


@tool
def pubchem(compound_names: str) -> str:
    """Search the PubChem database for chemical compounds (comma-separated).

    Examples:
    - benzol
    - C6H12O6,aspirin
    """
    PREVIOUS_SEARCHES.add(compound_names)
    compound_names = compound_names.split(",")
    try:
        compounds = [c for cn in compound_names for c in fetch_compounds(cn)]
        if len(compounds) == 0:
            LOGGER.warning(f"No compounds found for compound_names: {compound_names}")
            LOGGER.warning("Trying autocomplete...")
            compounds = [
                c for cn in compound_names for c in fetch_compounds_by_autocomplete(cn)
            ]
        LOGGER.info(
            f"Found {len(compounds)} compounds for compound_names: {compound_names}"
        )
        return json.dumps(format_compounds(compounds))
    except pcp.PubChemHTTPError as e:
        LOGGER.error(e)
        return f"Error: {e}"
    except requests.RequestException as e:
        LOGGER.error(e)
        return f"Error: {e}"
    except KeyError as e:
        LOGGER.error(e)
        return f"Error: {e}"


if __name__ == "__main__":
    # r = fetch_compounds("benzol")[0]
    # pprint(r.to_dict())
    # print(pubchem.invoke("aspir"))
    print(pubchem.invoke("benzol"))
    print(pubchem.invoke("C6H12O6,aspirin"))
