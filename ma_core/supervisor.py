from dataclasses import dataclass
from typing import TypedDict, Literal, Generic

from langchain_openai import ChatOpenAI
from langgraph.types import Command

from ma_core import StateType
from ma_core.answerer import Answerer
from ma_core.worker import Worker


@dataclass
class Supervisor(Generic[StateType]):
    llm: ChatOpenAI
    workers: list[Worker[StateType]]

    @property
    def members(self) -> list[str]:
        return [w.name for w in self.workers]

    @property
    def member_description(self):
        worker_tools = {
            w.name: [
                getattr(t, "name")
                for t in getattr(w, "tools", [])
                if hasattr(t, "name")
            ]
            for w in self.workers
        }
        return " ".join(f"{w} ({','.join(worker_tools[w])})" for w in self.members)

    @property
    def options(self):
        return self.members + [Answerer.name]

    def system_prompt(self, state: StateType) -> str:
        return f"""
        You are a supervisor tasked with managing a conversation between the following workers (and tools): {self.member_description}.
        Given the following user request, respond with the worker to act next.
        Never do the same task twice.
        At the end response with {Answerer.name} and an empty task string.
        Last Task: {state.get("task")}.
        """

    @property
    def router(self):
        options = self.options

        class Router(TypedDict):
            """Worker to route to next. If no workers needed, route to FINISH."""

            next: Literal[*options]
            task: str
            reason: str

        return Router

    def node(self, state: StateType) -> Command[str]:
        messages = [
            {"role": "system", "content": self.system_prompt(state)},
        ] + state["messages"]
        response = self.llm.with_structured_output(self.router).invoke(messages)
        return Command(goto=response["next"], update={"task": response["task"]})
