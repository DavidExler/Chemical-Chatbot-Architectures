import copy
import logging
import os
from typing import Annotated, Iterable

import dotenv
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
from langgraph.types import Command
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


def student_node(state: EnsembleState) -> Command:
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
    return Command(update={"answers": [AIMessage(content=res.content, name="student")]})


def professor_node(state: EnsembleState) -> Command:
    def answers_to_str(answers: Iterable[AnyMessage]) -> str:
        return (
            "<answers>\n"
            + "\n\n".join(
                f"<student{i}>\n{a.content}\n</student{i}>"
                for i, a in enumerate(answers)
            )
            + "\n</answers>"
        )

    messages = copy.deepcopy(state["messages"])
    messages[-1].content = (
        answers_to_str(state["answers"]) + "\n\n" + state["messages"][-1].content
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
        builder.add_node(f"student_{i}", student_node)
    builder.add_node("professor", professor_node)
    builder.add_edge([f"student_{i}" for i in range(NUM_GENERATORS)], "professor")

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
