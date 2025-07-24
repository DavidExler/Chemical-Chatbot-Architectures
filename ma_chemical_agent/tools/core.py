import json
import logging
import os
from functools import lru_cache
from time import sleep

import requests
from langchain_core.tools import tool
from requests import RequestException

LOGGER = logging.getLogger(__name__)
CORE_API_KEY = os.environ.get("CORE_API_KEY")


@tool
@lru_cache
def core_search(query: str) -> str:
    """Use this to search the CORE API for open access research papers.

    Use it when you need to find research papers from sources not included in arxiv.
    Input should be a search query
    """
    params = {"q": query, "api_key": CORE_API_KEY}
    try:
        response = None
        for _ in range(5):
            try:
                response = requests.get(
                    "https://api.core.ac.uk/v3/search/works/", params=params
                )
                response.raise_for_status()
                break
            except RequestException as e:
                LOGGER.warning(f"Request failed: {e}, retrying in 1 second.")
                sleep(1)
        if not response:
            return "Error: Request failed."
        LOGGER.info(f"Request to CORE API successful: {response.status_code}")
        results = response.json().get("results")
        if not results:
            return "No results found."

        def get_date(r):
            date_keys = [
                "acceptedDate",
                "updatedDate",
                "publishedDate",
                "createdDate",
                "depositedDate",
            ]
            for key in date_keys:
                if key in r and r[key]:
                    return r[key]
            return ""

        def format_abstract(abstract: str) -> str:
            if not abstract:
                return ""
            return "\n".join(i for i in abstract.split("\n") if len(i) > 5)

        return json.dumps(
            [
                {
                    "ID": r["id"],
                    "Title": r["title"],
                    "Authors": ",".join(a["name"] for a in r["authors"]),
                    "DOI": r["doi"],
                    "Date": get_date(r),
                    "Type": r["documentType"],
                    "Abstract": format_abstract(r["abstract"]),
                }
                for r in results
            ]
        )

    except RequestException as e:
        return f"Error: {e}"


if __name__ == "__main__":
    from pprint import pprint

    pprint(json.loads(core_search.invoke({"query": "aspirin"})))
