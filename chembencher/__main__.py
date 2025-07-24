import csv
import json
import logging
import os.path
import shutil
from concurrent.futures import ThreadPoolExecutor
from random import shuffle

from chembench.evaluate import (
    ChemBenchmark,
    aggregated_metrics,
    AggregatedReport,
)
from chembench.prompter import PrompterBuilder
from chembench.report import Report
from chembench.task import Task
from pydantic import TypeAdapter

from chembencher.collect_scores import combine_scores_for_model
from chembencher.model_wrapper import ModelWrapper
from rag.__main__ import build_graph

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


def run_single_task(task: str):
    graph = build_graph()
    model = ModelWrapper(graph)
    prompter = PrompterBuilder.from_model_object(
        model=model, prompt_type="instruction", get_logprobs=True
    )
    task = Task.from_json(task)
    return prompter.report(task)


def main(force: bool = False):
    graph = build_graph()
    all_topics = [
        "Analytical Chemistry",
        "General Chemistry",
        "Technical Chemistry",
        "Organic Chemistry",
        "Inorganic Chemistry",
        "Physical Chemistry",
        "Toxicity and Safety",
        "Materials Science",
        "Chemical Preference",
    ]
    model = ModelWrapper(graph)

    prompter = PrompterBuilder.from_model_object(
        model=model, prompt_type="instruction", get_logprobs=True
    )
    with open("topics.csv") as f:
        questions = list(csv.DictReader(f))

    all_json_files = [
        (q["topic"], q["question"].replace("../", ""))
        for q in questions
        if q["topic"] in all_topics
    ]
    all_json_files = [
        (t, f) for t, f in all_json_files if os.path.exists(f) and f.endswith(".json")
    ]
    shuffle(all_json_files)

    all_reports = []
    for topic in all_topics:
        json_files = [t[1] for t in all_json_files if t[0] == topic]
        print(f"Processing {len(json_files)} questions for {topic}")

        def process_task(json_file: str) -> Report | None:
            task = Task.from_json(json_file)
            path = os.path.join("reports", "multi-agent", f"{task._name}.json")
            if os.path.exists(path) and not force:
                try:
                    with open(path) as f:
                        LOGGER.info(f"Loading {path}")
                        ta = TypeAdapter(list[Report])
                        return ta.validate_json(f.read())[0]
                except Exception as e:
                    LOGGER.error(f"Error loading {path}: {e}")
            LOGGER.info(f"Processing {task._name}")
            try:
                report = prompter.report(task)
            except Exception as e:
                LOGGER.error(f"Error processing {task._name}: {e}")
                return

            os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, "w") as f:
                json.dump([report.model_dump()], f, indent=2)
            return report

        with ThreadPoolExecutor(max_workers=20) as executor:
            all_reports.extend(i for i in executor.map(process_task, json_files) if i)
        # for json_file in json_files:
        #     all_reports.append(process_task(json_file))

    report_files = [
        (
            os.path.join("reports", "multi-agent", f"{r.name}.json"),
            os.path.join("eval", "multi-agent", f"{r.name}.json"),
        )
        for r in all_reports
    ]

    # copy all reports to eval folder
    os.makedirs("eval/multi-agent", exist_ok=True)
    for s, d in report_files:
        os.system(f"cp {s} {d}")

    benchmark = ChemBenchmark()
    benchmark._reportname = "multi-agent"
    try:
        _data = {
            "model_name": "multi-agent",
            "num_questions": len(all_reports),
            "jsonfiles": [f for _, f in all_json_files],
        }
        _score = aggregated_metrics(all_reports)
        _report = {**_data, "aggregated_scores": _score}

        benchmark.save_result(report=_report, name="multi-agent")
        report = AggregatedReport(**_report)
        with open("multi-agent-report.json", "w") as f:
            f.write(report.model_dump_json(indent=2))
        print(combine_scores_for_model("eval", "data", None)["fraction_correct"])
    except Exception as e:
        print(e)
        return

    shutil.rmtree("eval", ignore_errors=True)


if __name__ == "__main__":
    import sys

    force = bool(sys.argv[1]) if len(sys.argv) > 1 else False
    main(force)
    # from pprint import pprint
    #
    # pprint(run_single_task("data/analytical_chemistry/24_4.json").metrics)
