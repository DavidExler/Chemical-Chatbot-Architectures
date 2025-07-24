import logging

import docker
from docker.errors import ContainerError
from langchain_core.tools import tool

LOGGER = logging.getLogger(__name__)
IMAGE = "chemllm"

PREVIOUS_CODES = set()


@tool
def python(code: str, requirements: str = "") -> str:
    """Use this tool to run all sorts of calculations."""
    # if code in PREVIOUS_CODES:
    #     return "You have already run this code. Please provide a new code or try a different tool."
    # PREVIOUS_CODES.add(code)
    # if code.startswith("print"):
    #     return "Your code is not allowed to print directly. Please run your calculations and print the results."
    docker_client = docker.from_env()
    commands = []
    if requirements:
        commands.append(f"pip install {requirements} > /dev/null 2>&1")
    code = code.replace("'", '"').replace('"', '\\"')
    commands.append(f'python -c "{code}" || true')  # Corrected line
    command = f"sh -c '{' && '.join(commands)}'"
    LOGGER.info(f"Running commands: {command}")
    try:
        return docker_client.containers.run(
            IMAGE,
            command,
            stdout=True,
            stderr=True,
            auto_remove=True,
            environment=["PYTHONUNBUFFERED=1"],
        ).decode("utf-8")
    except ContainerError as e:
        LOGGER.error(f"Error running Python code: {e.stderr}")
        return e.stderr


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(
        python.invoke(
            {
                "code": 'from rdkit import Chem; from rdkit.Chem import Descriptors; mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"); print(sum([atom.GetNumRadicalElectrons() + atom.GetNumImplicitHs() for atom in mol.GetAtoms()]))',
                "requirements": "rdkit",
            }
        )
    )
