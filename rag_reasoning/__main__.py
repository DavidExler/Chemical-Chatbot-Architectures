import json
import logging
import os
from typing import TypedDict

import dotenv
from langchain_community.retrievers import ArxivRetriever
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from ma_chemical_agent.tools.pubchem import search_for_smiles

os.environ["LANGCHAIN_PROJECT"] = "rag"

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
dotenv.load_dotenv()

llm = ChatOpenAI(
    model_name="llama3.3-70b",
    api_key="EMPTY",
    temperature=0,
    max_tokens=1024,
    base_url="http://141.52.44.209:7878/v1/",
)


class RAGState(MessagesState):
    answer_structure: str


class ResearchOutput(TypedDict):
    research_thoughts: str
    smiles: str
    arxiv: str


def researcher_node(state: RAGState) -> Command:
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

    arxiv_queries = set(response["arxiv"].split(",")[:5])
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
    return Command(update={"messages": [research_message], **update})


def generate_node(state: RAGState) -> Command:
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
    if answer_structure_prompt := state.get("answer_structure"):
        prompt += f"\n{answer_structure_prompt}"
    chain = prompt | llm
    response = chain.invoke(state)
    return Command(update={"messages": [response]})


def answerer_node(state: RAGState) -> Command:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                """
You are the answerer. You are responsible for providing the final answer to the user's question based on the insights and analyses provided by the professor and students.
            """
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    if answer_structure_prompt := state.get("answer_structure"):
        prompt += f"\n{answer_structure_prompt}"
    chain = prompt | llm
    response = chain.invoke(state)
    return Command(update={"messages": [response]})


def build_graph() -> CompiledGraph:
    builder = StateGraph(RAGState)
    builder.set_entry_point("researcher")
    builder.add_node("researcher", researcher_node)
    builder.add_node("generate", generate_node)
    builder.add_node("answerer", answerer_node)
    builder.add_edge("researcher", "generate")
    builder.add_edge("generate", "answerer")
    builder.add_edge("answerer", END)

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
