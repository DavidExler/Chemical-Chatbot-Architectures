import copy
import functools
import itertools
import logging
import os
from typing import Annotated

import dotenv
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.messages import (
    SystemMessage,
    AIMessage,
    AnyMessage,
    HumanMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph, MessagesState, add_messages
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown

from ma_core.answerer import Answerer

os.environ["LANGCHAIN_PROJECT"] = "ensemble"

NUM_GENERATORS = 5

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
dotenv.load_dotenv()

llm = ChatOpenAI(
    model_name="llama3.3-70b",
    api_key="EMPTY",
    temperature=0,
    max_tokens=512,
    base_url="http://141.52.44.209:7878/v1/",
    # base_url="http://141.52.44.15:7878/v1/",
)
small_llm = ChatOpenAI(
    model_name="llama3.3-70b",
    api_key="EMPTY",
    temperature=0.2,
    max_tokens=8192,
    # base_url="http://141.52.44.209:7878/v1/",
    base_url="http://141.52.44.15:7878/v1/",
)


class EnsembleState(MessagesState):
    answers: Annotated[list[AnyMessage], add_messages]
    verified_answers: Annotated[list[AnyMessage], add_messages]
    unverified_answers: Annotated[list[AnyMessage], add_messages]


def student_node(state: EnsembleState, *, id: int) -> Command:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                """
You are a chemistry student engaging in deep study. You possess knowledge of core chemistry (stoichiometry, balancing equations, basic reactions). You strive to not only solve problems but also understand the underlying principles.

**Analyze and solve the user's chemistry problem, following this structured approach:**

1. **Rephrasing and Key Insights:** Briefly rephrase the problem, highlighting the most critical data and its implications.
2. **Conceptual Foundation:** Explain the key chemical concepts involved, *demonstrating your understanding of their relevance to the problem*.
3. **Strategic Approach:** Outline your solution plan, anticipating potential challenges.
4. **Detailed Solution with Rationale:** Solve the problem step-by-step, *justifying each step with chemical reasoning and showing calculations*.
5. **Answer:** Provide the final answer, ensuring it aligns with the problem statement and your solution.

**Prioritize insightful reasoning and exploration of chemical principles. If uncertain, articulate your thought process and identify knowledge gaps.**
            """
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | small_llm
    res = chain.invoke(state)
    return Command(
        update={"answers": [AIMessage(content=res.content, name=f"student-{id}")]}
    )


class VerifyResponse(BaseModel):
    verified: bool


def validator_node(state: EnsembleState, *, id: int) -> Command:
    prompt = """
You are a reviewer tasked with evaluating the students response by using your own expertise and the provided tools.

First give an short explanation of the problem and the correct answer.
Then evaluate the student's response and determine if it is correct or incorrect.
After your evaluation, provide a final answer.

**As a reviewer your final answer must be one of these:**

- [ANSWER]correct[/ANSWER]
- [ANSWER]incorrect[/ANSWER]
    """
    student_answer = next(a for a in state["answers"] if a.name == f"student-{id}")

    tools = load_tools(["arxiv", "wolfram-alpha"], llm)
    chain = create_react_agent(llm, tools)
    message = chain.invoke(
        {
            "messages": [
                SystemMessage(prompt),
                HumanMessage(
                    content=f"Student Answer: {student_answer.content}", name="student"
                ),
            ]
        }
    )["messages"][-1]
    is_correct = "[ANSWER]correct[/ANSWER]" in message.content
    key = "verified_answers" if is_correct else "unverified_answers"
    return Command(update={key: [student_answer]})


def professor_node(state: EnsembleState) -> Command:
    def answer_to_str(answer: AIMessage, is_verified: bool) -> str:
        return f'<student name="{answer.name}" is_verified="{is_verified}">\n{answer.content}\n</student>'

    def answers_to_str(
        verified_answers: list[AnyMessage], unverified_answers: list[AnyMessage]
    ) -> str:
        answers = itertools.chain(
            zip(verified_answers, itertools.repeat(True)),
            zip(unverified_answers, itertools.repeat(False)),
        )
        return (
            "<answers>\n"
            + "\n\n".join(answer_to_str(a, v) for a, v in answers)
            + "\n</answers>"
        )

    messages = copy.deepcopy(state["messages"])
    messages[-1].content = (
        state["messages"][-1].content
        + "\n\n"
        + answers_to_str(state["verified_answers"], state["unverified_answers"])
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                """
You are a chemistry professor engaged in deep study. You possess knowledge of core chemistry (stoichiometry, balancing equations, basic reactions). You strive to not only solve problems but also understand the underlying principles.

**Follow this structured approach:**

1.  **Independent Solution & Justification:** Solve the problem yourself, showing all steps and explaining the chemical reasoning behind each step. Explain *why* your chosen answer is correct and *why* the other options are incorrect, referencing relevant chemical principles.
2.  **Student Response Analysis:** For each student response:
    *   **Conceptual Understanding:** Evaluate the student's grasp of the core chemical concepts. Identify specific strengths, weaknesses, and any misconceptions.
    *   **Reasoning Depth:** Assess if the student merely selected an answer or if they explored the underlying chemical logic. Did they justify their choice with sound reasoning?
    *   **Error Analysis:** If the student is incorrect, pinpoint the exact error in their reasoning or calculation.
3.  **Synthesis & Final Answer:** Provide a final answer that synthesizes your independent solution with the best aspects of the student responses. Explain why this final answer is the most complete and accurate, highlighting any key insights gained from the analysis.
                """
            ),
        ]
        + messages
    )
    chain = prompt | llm
    res = chain.invoke(state)
    return Command(
        update={"messages": [AIMessage(content=res.content, name="professor")]},
        goto=END,
    )


def to_human_msg(msg: AnyMessage) -> HumanMessage:
    return HumanMessage(content=msg.content, name=msg.name)


def build_graph() -> CompiledGraph:
    builder = StateGraph(EnsembleState)
    builder.set_entry_point("start_generate")
    builder.add_node("start_generate", lambda x: x)
    for i in range(NUM_GENERATORS):
        builder.add_edge("start_generate", f"student_{i}")
        builder.add_node(f"student_{i}", functools.partial(student_node, id=i))
        builder.add_edge(f"student_{i}", f"validator_{i}")
        builder.add_node(f"validator_{i}", functools.partial(validator_node, id=i))
    builder.add_node("professor", professor_node)
    builder.add_edge([f"validator_{i}" for i in range(NUM_GENERATORS)], "professor")

    builder.set_finish_point("professor")

    graph = builder.compile()
    with open("graph.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())
    return graph


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
