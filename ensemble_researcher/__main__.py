import json
import logging
import os
from typing import Annotated, Iterable, TypedDict

import dotenv
from langchain_community.retrievers import ArxivRetriever
from langchain_core.documents import Document
from langchain_core.messages import (
    SystemMessage,
    AIMessage,
    AnyMessage,
    HumanMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, add_messages
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from ma_chemical_agent.tools.pubchem import search_for_smiles

os.environ["LANGCHAIN_PROJECT"] = "ensemble"

NUM_GENERATORS = 5

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
dotenv.load_dotenv()

llm = ChatOpenAI(
    model_name="llama3.3-70b",
    api_key="EMPTY",
    temperature=0,
    max_tokens=1024,
    base_url="http://141.52.44.209:7878/v1/",
    # base_url="http://141.52.44.15:7878/v1/",
)
small_llm = ChatOpenAI(
    model_name="llama3.3-70b",
    api_key="EMPTY",
    temperature=0.3,
    max_tokens=8192,
    n=2,
    # base_url="http://141.52.44.209:7878/v1/",
    base_url="http://141.52.44.15:7878/v1/",
)


class EnsembleState(MessagesState):
    answers: Annotated[list[AnyMessage], add_messages]
    research: str
    answer_structure: str
    researches: int
    arxiv_queries: set[str]


class ResearchOutput(TypedDict):
    research_thoughts: str
    smiles: str
    arxiv: str


def researcher_node(state: EnsembleState) -> Command:
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
    prompt = """
You are a researcher in the field of chemistry.
You are supposed to write a comma separated list of arXiv queries that you want to search for.
If you need any smiles strings converted to simplify the problem solving you can also write them in a comma separated list.

Your output should be a json object with the following structure:
{
    "research_thoughts": "To research on this topic, I should look into...",
    "smiles": "smiles1,smiles2,smiles3", # this is optional
    "arxiv": "query1,query2,query3" # only the first 3 queries will be used
}
or
{
    "research_thoughts": "I am missing some information about ..., so I will look into that."
    "smiles": "smiles1,smiles2,smiles3", # this is optional
    "arxiv": "query1,query2,query3" # only the first 3 queries will be used
}
    """
    model = llm.with_structured_output(ResearchOutput)
    response = model.invoke([SystemMessage(prompt)] + messages)

    smiles_strings = response["smiles"].split(",")
    smiles_compounds = (search_for_smiles(smiles) for smiles in smiles_strings)
    smiles_compounds = sum((i for i in smiles_compounds if i), [])

    arxiv_queries = set(response["arxiv"].split(",")[:3]) - state.get(
        "arxiv_queries", set()
    )
    arxiv_retriever = ArxivRetriever()
    arxiv_documents = [
        d for query in arxiv_queries for d in arxiv_retriever.invoke(query)
    ]

    def format_document(doc: Document) -> str:
        title = doc.metadata["Title"]
        return f"""
<arxiv-document title="{title}">
    <title>{title}</title>
    <authors>{doc.metadata["Authors"]}</authors>
    <published>{doc.metadata["Published"]}</published>
    <content>{doc.page_content}</content>
</arxiv-document>
        """

    arxiv_documents = "\n\n".join(format_document(doc) for doc in arxiv_documents)
    research_message = HumanMessage(
        content=(
            f"arXiv research results:\n\n{arxiv_documents}"
            f"\n\n"
            f"smiles to compounds:\n\n{json.dumps(smiles_compounds, indent=2)}"
        ),
        name="researcher",
    )
    return Command(
        update={
            "messages": [research_message],
            "research": research_message.content,
            "researches": state.get("researches", 0) + 1,
            "arxiv_queries": state.get("arxiv_queries", set()) | arxiv_queries,
            **update,
        }
    )


def student_node(state: EnsembleState) -> Command:
    prompt = """
You are a chemistry student engaging in deep study. You possess knowledge of core chemistry (stoichiometry, balancing equations, basic reactions). You strive to not only solve problems but also understand the underlying principles.

**Analyze and solve the user's chemistry problem, following this structured approach:**

1. **Rephrasing and Key Insights:** Briefly rephrase the problem, highlighting the most critical data and its implications.
2. **Conceptual Foundation:** Explain the key chemical concepts involved, *demonstrating your understanding of their relevance to the problem*.
3. **Strategic Approach:** Outline your solution plan, anticipating potential challenges.
4. **Detailed Solution with Rationale:** Solve the problem step-by-step, *justifying each step with chemical reasoning and showing calculations*.
5. **Answer:** Provide the final answer, ensuring it aligns with the problem statement and your solution.

**Prioritize insightful reasoning and exploration of chemical principles. If uncertain, articulate your thought process and identify knowledge gaps.**
            """ + state.get("answer_structure", "")
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(prompt),
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
                f'<student id="{i}">\n{a.content}\n</student{i}>'
                for i, a in enumerate(answers)
            )
            + "\n</answers>"
        )

    messages = state["messages"]
    messages += [
        HumanMessage(
            content=answers_to_str(state["answers"])
            + "\n"
            + "<research>"
            + "\n"
            + state["research"]
            + "\n"
            + "</research>",
        )
    ]

    prompt = """
You are a chemistry professor engaged in deep study. You possess knowledge of core chemistry (stoichiometry, balancing equations, basic reactions). You strive to not only solve problems but also understand the underlying principles and how they relate to the given context.

**For each multiple-choice question, follow this structured approach:**

1.  **Contextualized Problem Analysis:** Before attempting a solution, analyze the question within the provided chemical context. What specific concepts are being tested? What are the key relationships or principles at play? What potential challenges or nuances might exist?
2.  **Independent Solution & Justification:** Solve the problem yourself, showing all steps and explaining the chemical reasoning behind each step. Explain *why* your chosen answer is correct and *why* the other options are incorrect, referencing relevant chemical principles and the context established in step 1.
3.  **Student Response Analysis:** For each student response:
    *   **Conceptual Understanding:** Evaluate the student's grasp of the core chemical concepts *in relation to the specific context of the question*. Identify specific strengths, weaknesses, and any misconceptions.
    *   **Reasoning Depth:** Assess if the student's reasoning aligns with the contextual analysis. Did they justify their choice with sound reasoning that demonstrates an understanding of the underlying chemical logic *within the given context*?
    *   **Error Analysis:** If the student is incorrect, pinpoint the exact error in their reasoning or calculation, and explain how it relates to the context and relevant chemical principles.
4.  **Synthesis & Final Answer:** Provide a final answer that synthesizes your independent solution with the best aspects of the student responses. Explain why this final answer is the most complete and accurate, highlighting any key insights gained from the analysis, and how the context influenced the solution.
    """ + state.get("answer_structure", "")
    prompt = ChatPromptTemplate.from_messages([SystemMessage(prompt)] + messages)
    chain = prompt | llm
    res = chain.invoke(state)
    return Command(
        update={"messages": [AIMessage(content=res.content, name="professor")]}
    )


def to_human_msg(msg: AnyMessage) -> HumanMessage:
    return HumanMessage(content=msg.content, name=msg.name)


def modify_input(input: dict) -> dict:
    input_msg = input["messages"][0]
    input_msg.content += (
        "\n\n First explain your answer in detail, then provide the final answer."
    )
    input["messages"] = [input_msg]
    return input


def answerer_node(state: EnsembleState) -> Command:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""
You are the answerer. You are responsible for providing the final answer to the user's question based on the insights and analyses provided by the professor and students.
                """
                + state.get("answer_structure", "")
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | llm
    res = chain.invoke(state)
    return Command(
        update={"messages": [AIMessage(content=res.content, name="answerer")]}
    )


def build_graph() -> CompiledGraph:
    builder = StateGraph(EnsembleState)
    builder.set_entry_point("researcher")
    builder.add_node("researcher", researcher_node)
    for i in range(NUM_GENERATORS):
        builder.add_edge("researcher", f"student_{i}")
        builder.add_node(f"student_{i}", student_node)
    builder.add_node("professor", professor_node)
    builder.add_edge([f"student_{i}" for i in range(NUM_GENERATORS)], "professor")
    builder.add_node("answerer", answerer_node)
    builder.add_edge("professor", "answerer")
    builder.set_finish_point("answerer")

    graph = builder.compile()
    with open("graph.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())
    return graph


if __name__ == "__main__":
    graph = build_graph()
    user_input = input("Enter your message: ")
    input = {"messages": [("user", user_input)]}
    for _, s in graph.stream(input, subgraphs=True):
        print(s)
        print("----")
