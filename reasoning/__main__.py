import logging
import os
from typing import Literal

import dotenv
from langchain_core.messages import (
    SystemMessage,
    AIMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown

from ma_core.answerer import Answerer

NUM_GENERATIONS = 3

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
dotenv.load_dotenv()

PUBMED_API_KEY = os.environ.get("PUBMED_API_KEY")

llm = ChatOpenAI(
    #model_name="llama3.3-70b",
    model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct",
    api_key="EMPTY",
    temperature=0.2,
    n=2,
    max_tokens=8192,
    base_url="http://141.52.44.209:7878/v1/",
    # base_url="http://141.52.44.15:7878/v1/",
)
small_llm = ChatOpenAI(
    #model_name="llama3.3-70b",
    model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct",
    api_key="EMPTY",
    temperature=1.0,
    max_tokens=8192,
    # base_url="http://141.52.44.209:7878/v1/",
    base_url="http://141.52.44.15:7878/v1/",
)


class ReflectState(MessagesState):
    generations: int
    user_question: str
    answer_structure: str


def generate_node(state: ReflectState) -> Command:
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
                update["user_question"] = messages[-1].content

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
5. **Critical Evaluation:** Reflect on your solution. Discuss potential limitations, assumptions made, or areas where further information would improve accuracy or understanding.

**Prioritize insightful reasoning and exploration of chemical principles. If uncertain, articulate your thought process and identify knowledge gaps.**
            """
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    generations = state.get("generations", 0)
    if generations < NUM_GENERATIONS:
        chosen_llm = small_llm
    else:
        chosen_llm = llm
    chain = prompt | chosen_llm
    res = chain.invoke(state)
    return Command(
        update={
            "messages": [AIMessage(content=res.content, name="generate")],
            "generations": generations + 1,
            **update,
        }
    )


def reflection_node(state: ReflectState) -> Command:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                """
You are a distinguished senior researcher, renowned for your rigorous and insightful critiques. You are tasked with evaluating the work of a chemistry student, demanding intellectual rigor and a deep understanding of the subject matter.

**Analyze the student's response with an unyielding critical eye. Produce a comprehensive report that adheres to the following:**

1. **Relentless Scrutiny:**  Examine the student's response with meticulous attention to detail. Verify the accuracy of every claim, assess the depth of understanding demonstrated, and rigorously evaluate the logical coherence of their arguments.
2. **Challenge and Probe:**  Actively challenge the student's conclusions. Identify any leaps in logic, unstated assumptions, or inconsistencies in their reasoning. Demand concrete evidence and specific justifications for each assertion made. Do not accept any statement at face value.
3. **Independent Verification:**  Independently verify all information provided and any steps taken by the student. Never accept the student's response without your own critical assessment and rationalization.
4. **Targeted Feedback:** Provide concise, actionable feedback geared towards substantial improvement. Focus on specific areas where the student's understanding is lacking, their reasoning is flawed, or their approach is incomplete. Highlight opportunities for deeper exploration and more rigorous analysis.
5. **Strictly No Solutions:** Under no circumstances should you provide the correct answer or directly solve the problem for the user. Your role is to guide the student towards a deeper understanding through incisive critique, not to provide solutions. Your goal is a thorough assessment of strengths and weaknesses.

**Your ultimate objective is to deliver a penetrating critique that exposes weaknesses, illuminates areas for improvement, and compels the student to elevate their understanding of the subject matter.**
            """
                + (
                    f'This is the users original question: {state.get("user_question", "")}'
                    if state.get("user_question")
                    else ""
                )
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    generations = state.get("generations", 0)
    if generations < NUM_GENERATIONS:
        chosen_llm = small_llm
    else:
        chosen_llm = llm
    chain = prompt | chosen_llm
    result = chain.invoke(state)
    return Command(
        update={"messages": [AIMessage(content=result.content, name="reflect")]}
    )


def answerer_node(state: ReflectState) -> Command:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                """
You are an answerer tasked with answering a user's question based on the result of other LLMs.
Follow these rules:
- Give a short but precise answer and follow the user guideline on how to answer.
- Base your results on all previous messages
- Structure your answer like the user requested.
            """
                + state.get("answer_structure", "")
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | llm
    res = chain.invoke(state)
    return Command(update={"messages": [res]}, goto=END)


def build_graph(reflective: bool = True) -> CompiledGraph:
    def should_continue(state: ReflectState) -> Literal["reflect", "answer"]:
        LOGGER.info(state)
        generations = state.get("generations", 0)
        LOGGER.info(f"Generation step: {generations}")
        if generations < NUM_GENERATIONS:
            return "reflect"
        return "answer"

    builder = StateGraph(ReflectState)
    builder.set_entry_point("generate")
    builder.add_node("generate", generate_node)
    builder.add_node("answer", answerer_node)
    if reflective:
        builder.add_node("reflect", reflection_node)
        builder.add_edge("reflect", "generate")
        builder.add_conditional_edges("generate", should_continue)
    else:
        builder.add_edge("generate", "answer")
    builder.add_edge("answer", END)
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
