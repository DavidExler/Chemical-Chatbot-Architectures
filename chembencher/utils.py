import csv
import json
import logging
import os
from concurrent.futures.thread import ThreadPoolExecutor

from chembench.evaluate import ChemBenchmark
from chembench.prompter import PrompterBuilder
from chembench.task import Task
from chembench.utils import (
    remove_ce,
    remove_math,
    remove_pu,
    remove_smiles,
    remove_rxnsmiles,
)
from langchain_core.runnables import Runnable
from tqdm import tqdm

from chembencher.collect_scores import combine_scores_for_all_models
from chembencher.model_wrapper import ModelWrapper

LOGGER = logging.getLogger(__name__)

ALL_CATEGORIES = [
    "gfk",
    "number_nmr_peaks",
    "point_group",
    # "oxidation_states",
    "oup",
    "reactive_groups",
    "toxicology",
    "LMU_tox",
    "tox_pharma",
    "tox_wwu",
    "dai",
    "pictograms",
    "h_statement",
    "materials_compatibility",
    "chem_chem_comp",
    # "electron_counts",
    "organic_reactivity",
    "polymer_chemistry",
    "stolaf",
    "sci_lab_safety_test",
    "chemical_safety_mcq_exam",
    "analytical_chemistry",
    "periodic_table",
    "ac_faessler_tum",
    "pum_tum",
    "biomolecular",
    "xray",
    "materials_synthesis",
    "func_mats_and_nanomats",
    "molsim",
    "smiles_to_name",
    "name_to_smiles",
    "preference",
]


def benchmark(chain: Runnable, categories: list[str] | None = None):
    LOGGER.info("Benchmarking")
    model = ModelWrapper(chain)
    for category in tqdm(categories or ALL_CATEGORIES):
        LOGGER.info(f"Processing category {category}")
        tasks = [
            (root, file)
            for root, dirs, files in os.walk(f"data/{category}/")
            for file in files
            if file.endswith(".json")
        ]
        # for root, file in tqdm(tasks):
        #     process_task(root, file, model)
        with ThreadPoolExecutor() as executor:
            executor.map(lambda x: process_task(x[0], x[1], model), tasks)
    report = combine_scores_for_all_models(
        "reports/iaichemllm/", "summary.json", "data/"
    )
    return report["fraction_correct"]


def benchmark_topic(
    chain: Runnable, topic: str, force: bool = False, max_tasks: int | None = None
) -> dict:
    LOGGER.info(f"Benchmarking topic {topic}")
    with open("topics.csv") as f:
        topics = list(csv.DictReader(f))
    tasks = sorted(
        t["question"].replace("../", "")
        for t in topics
        if t["topic"] == topic and t["question"].endswith(".json")
    )
    if max_tasks:
        tasks = tasks[: int(max_tasks)]
    tasks = [(os.path.dirname(t), os.path.basename(t)) for t in tasks]
    model = ModelWrapper(chain)
    # tempfolder = tempfile.mkdtemp(dir="tmp")
    tempfolder = os.path.join("tmp", topic)
    os.makedirs(tempfolder, exist_ok=True)
    with ThreadPoolExecutor(max_workers=30) as executor:
        executor.map(
            lambda x: process_task(x[0], x[1], model, force, tempfolder),
            tasks,
            timeout=600,
        )
    # for task in tasks:
    #     process_task(task[0], task[1], model, force, tempfolder)
    report = combine_scores_for_all_models(tempfolder, f"summary_{topic}.json", "data/")
    # if report["fraction_correct"] > 0.0001:
    #     shutil.rmtree(tempfolder)
    return report


def process_task(
    root: str,
    file: str,
    model: ModelWrapper,
    force: bool = False,
    dir: str = "reports/iaichemllm/",
):
    LOGGER.info(f"Processing task {root}{file}")
    if not os.path.exists(os.path.join(root, file)):
        LOGGER.error(f"File {root}{file} does not exist")
        return
    result_file = f"{dir}/{file}"
    if not force and os.path.exists(result_file):
        LOGGER.debug(f"Skipping {result_file}")
        return

    task = Task.from_json(os.path.join(root, file))
    prompter = PrompterBuilder.from_model_object(
        model=model,
        prompt_type="instruction",  # can be 'instruct' or 'completion'
        get_logprobs=True,
        post_process_ce=remove_ce,
        post_process_math=remove_math,
        post_process_pu=remove_pu,
        post_process_smiles=remove_smiles,
        post_process_rxnsmiles=remove_rxnsmiles,
        other=None,
    )
    report = prompter.report(task)
    benchmark = ChemBenchmark()
    benchmark.get_questions_from_directory("data/")
    benchmark.bench(prompter).pr

    LOGGER.info(
        f"Task: {task._name}, Predicted: {report.results[0]['parsed_output']}, Metrics: {report.metrics}"
    )

    LOGGER.debug(f"Writing report to {result_file}")
    with open(result_file, "w") as f:
        json.dump([report.model_dump()], f)
