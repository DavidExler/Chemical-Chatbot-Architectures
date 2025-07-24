import logging
from dataclasses import dataclass
from typing import ClassVar

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.types import Command
from typing_extensions import Generic

from ma_core import StateType

LOGGER = logging.getLogger(__name__)


@dataclass
class Answerer(Generic[StateType]):
    llm: ChatOpenAI
    short: bool = False
    name: ClassVar[str] = "Answer"

    def system_prompt(self, state: StateType) -> str:
        if self.short:
            prompt = """
You are an answerer tasked with answering a user's question based on the result of other LLMs.
Give a short but precise answer.
Don't create new results, your output should be based on the results of all previous Messages.
            """
        else:
            prompt = """
You are an answerer tasked with answering a user's question based on the result of other LLMs.
Give a very detailed answer if available include any code, math or references that got generated during the process.
Don't create new results, your output should be based on the results of all previous Messages.
Format your result in Markdown Format.
            """
        LOGGER.info(state)
        if answer_structure_prompt := state.get("answer_structure"):
            prompt += f"\n{answer_structure_prompt}"
        return prompt

    def node(self, state: StateType) -> Command[END]:
        messages = [
            {"role": "system", "content": self.system_prompt(state)},
        ] + state["messages"]
        result = self.llm.invoke(messages)
        return Command(
            update={"messages": [AIMessage(content=result.content, name=self.name)]},
            goto=END,
        )
