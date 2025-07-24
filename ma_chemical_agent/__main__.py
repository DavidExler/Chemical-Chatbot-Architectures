import logging
from typing import TypedDict, Annotated, Literal

import dotenv
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import (
    TavilySearchResults,
    WikipediaQueryRun,
)
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import AnyMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, add_messages
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown

from ma_chemical_agent.collaborator import Collaborator
from ma_chemical_agent.tools.core import core_search
from ma_chemical_agent.tools.pubchem import pubchem
from ma_chemical_agent.tools.python import python
from ma_chemical_agent.tools.rdkit import molecular_properties
from ma_core.answerer import Answerer
from ma_core.validator import Validator

logging.basicConfig(level=logging.INFO)
dotenv.load_dotenv()

llm = ChatOpenAI(
    model_name="llama3.3-70b",
    api_key="EMPTY",
    temperature=0,
    base_url="http://141.52.44.209:7878/v1/",
)

research_collaborator = Collaborator(
    "researcher",
    llm,
    load_tools(["arxiv"])
    + [
        TavilySearchResults(max_results=5),
        pubchem,
        core_search,
        WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=800)
        ),
    ],
    """
You are a researcher.
You MUST research until you have all the information needed to solve the task.
Find multiple sources to cross-verify the information.
    """,
    answer_nodes=[Validator.name],
)
chemist_collaborator = Collaborator(
    "chemist",
    llm,
    [python, molecular_properties],
    """
You are a chemist. Use your knowledge of chemistry to solve the task.
Provide very detailed responses including every piece that might be relevant.
    """,
    answer_nodes=[Validator.name],
)

collaborators = [research_collaborator, chemist_collaborator]
for c in collaborators:
    c.assign_colleagues(collaborators)

answerer = Answerer(llm, short=True)
validator = Validator(llm, research_collaborator.name, Answerer.name)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    answer_structure: str
    task: str | None


def initial_thoughts(state: State) -> Command[Literal[research_collaborator.name]]:
    messages = state.get("messages", [])
    update = {}
    if messages and not state.get("answer_structure"):
        sep = "You MUST include"
        result = messages[-1].content.split(sep)
        if len(result) == 2:
            messages[-1].content, answer_structure = result
            messages[-1].content = messages[-1].content.replace(
                "Please answer by responding with the letter of the correct answer.",
                "",
            )
            state["messages"] = messages
            if answer_structure:
                update["answer_structure"] = sep + answer_structure

    system_prompt = """
You are the entry point to a multi agent system.
Your task is to initiate the conversation and provide the first message.
Write down all thoughts you have about the task and the information you need to solve it.
You MUST NOT solve the task, you ONLY collect initial thoughts and pass the conversation to the researcher.
You can leave a guess for the solution, but it MUST be marked as a guess and the researcher must be advised to verify it.
    """
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    result = llm.invoke(messages)
    return Command(
        update={
            "messages": [result],
            "task": "Research resources that verify these thoughts. And all other resources required to solve the task.",
            **update,
        },
        goto=research_collaborator.name,
    )


def build_graph():
    builder = StateGraph(State)
    builder.add_edge(START, "init")
    builder.add_node("init", initial_thoughts)
    for c in collaborators:
        builder.add_node(c.name, c.node)
    builder.add_node(Validator.name, validator.node)
    builder.add_node(Answerer.name, answerer.node)
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
