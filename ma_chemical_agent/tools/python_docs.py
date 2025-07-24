import inspect
import json
import pkgutil

from langchain_core.tools import tool


def generate_library_overview(package_name):
    """
    Generates an overview of all classes, methods, and functions in a Python package.

    Args:
        package_name (str): The name of the package to analyze.

    Returns:
        dict: A dictionary containing the overview.
    """
    try:
        package = __import__(package_name)
    except ImportError as e:
        return f"Error: Could not import package '{package_name}'. {e}"

    overview = {}

    for loader, module_name, is_pkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        try:
            module = __import__(module_name, fromlist="dummy")
        except:
            continue

        module_info = {"classes": {}, "functions": []}

        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                # Get class methods
                methods = [
                    m[0] for m in inspect.getmembers(obj, predicate=inspect.isfunction)
                ]
                module_info["classes"][name] = methods
            elif inspect.isfunction(obj) and obj.__module__ == module.__name__:
                # Get standalone functions
                module_info["functions"].append(name)

        if module_info["classes"] or module_info["functions"]:
            overview[module_name] = module_info

    return overview


@tool
def python_docs(package: str) -> str:
    """This tool returns an overview of all classes, methods and functions in a given Python Package.

    Examples:
    - package: 'numpy'
    - package: 'rdkit'
    """
    return json.dumps(generate_library_overview(package))


if __name__ == "__main__":
    print(python_docs.invoke({"package": "rdkit"}))
