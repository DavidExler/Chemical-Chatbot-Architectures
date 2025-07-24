from typing import TypedDict, Annotated

import dotenv
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import TavilySearchResults
from langchain_community.tools.riza.command import ExecPython, ExecJavaScript
from langchain_core.messages import AnyMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, add_messages
from rich.console import Console
from rich.markdown import Markdown

from ma_core.answerer import Answerer
from ma_core.supervisor import Supervisor
from ma_core.worker import Worker

dotenv.load_dotenv()

llm = ChatOpenAI(
    model_name="llama3.3-70b",
    api_key="EMPTY",
    temperature=0,
    base_url="http://141.52.44.209:7878/v1/",
)


research_worker = Worker(
    "researcher",
    llm,
    load_tools(["arxiv"]) + [TavilySearchResults(max_results=5)],
    "You are a researcher. DO NOT do any maths",
)
code_worker = Worker(
    "coder",
    llm,
    [],
    "You are a coder. DO NOT do any research",
)
review_worker = Worker(
    "tester_and_reviewer",
    llm,
    [ExecPython(), ExecJavaScript()],
    "You are a tester and reviewer. DO NOT do any research or coding",
)

workers = [research_worker, code_worker, review_worker]
supervisor = Supervisor(llm, workers)
answerer = Answerer(llm)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    task: str


def build_graph():
    builder = StateGraph(State)
    builder.add_edge(START, "supervisor")
    builder.add_node("supervisor", supervisor.node)
    builder.add_node(Answerer.name, answerer.node)
    for worker in workers:
        builder.add_node(worker.name, worker.node)
    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()
    user_input = input("Enter your message: ")
    input = {"messages": [("user", user_input)]}
    for _, s in graph.stream(input, subgraphs=True):
        if Answerer.name in s:
            console = Console()
            console.print(Markdown(s[Answerer.name]["messages"][-1].content))
        else:
            print(s)
            print("----")
