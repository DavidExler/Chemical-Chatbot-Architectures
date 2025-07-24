from dataclasses import dataclass, field
from typing import Literal, Generic

from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from ma_core import StateType


@dataclass
class Worker(Generic[StateType]):
    name: str
    llm: ChatOpenAI
    tools: list[Tool] = field(default_factory=list)
    state_modifier: str | None = None

    def node(self, state: StateType) -> Command[Literal["supervisor"]]:
        if len(self.tools) > 0:
            num_before = len(state["messages"])
            agent = create_react_agent(
                self.llm,
                self.tools,
                state_modifier=f"""
                {self.state_modifier}.
                If you know the answer without any tools, you can answer directly.
                Your task is to {state["task"]}.
                Only do the task that you are assigned to do.
                """,
            )
            result = agent.invoke(state)
            new_messages = [
                HumanMessage(content=m.content, name=self.name)
                for m in result["messages"][num_before:]
            ]
        else:
            prompt = f"You are an {self.name} tasked with {state["task"]}. Only do the task that you are assigned to do."
            messages = [
                {"role": "system", "content": prompt},
            ] + state["messages"]
            result = self.llm.invoke(messages)
            new_messages = [HumanMessage(content=result.content, name=self.name)]
        return Command(
            update={"messages": new_messages},
            goto="supervisor",
        )
