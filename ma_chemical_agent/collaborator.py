import logging
from dataclasses import dataclass, field
from typing import Literal, Generic

from langchain_core.messages import ToolMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel, Field

from ma_core import StateType
from ma_core.validator import Validator

LOGGER = logging.getLogger(__name__)


@dataclass
class Collaborator(Generic[StateType]):
    name: str
    llm: ChatOpenAI
    tools: list[Tool]
    state_modifier: str
    answer_nodes: list[str] = field(default_factory=list)
    colleagues: list["Collaborator"] = field(default_factory=list)

    def assign_colleagues(self, colleagues: list["Collaborator"]):
        self.colleagues = [c for c in colleagues if c != self]

    @property
    def colleague_names(self) -> list[str]:
        return [c.name for c in self.colleagues]

    # @property
    # def colleague_descriptions(self):
    #     colleague_tools = {
    #         c.name: [
    #             getattr(t, "name")
    #             for t in getattr(c, "tools", [])
    #             if hasattr(t, "name")
    #         ]
    #         for c in self.colleagues
    #     }
    #     return " ".join(
    #         f"{c} ({','.join(colleague_tools[c])})" for c in self.colleague_names
    #     )

    @property
    def options(self):
        return self.colleague_names + self.answer_nodes

    @property
    def tool_names(self):
        return [getattr(t, "name") for t in self.tools if hasattr(t, "name")]

    def system_prompt(self, state: StateType) -> str:
        prompt = f"""
{self.state_modifier}
These are your colleagues: {self.colleague_names}.
Use the provided tools {self.tool_names} to progress towards answering the question, make sure you are using correct json syntax for the tools.
Do NOT use any quotation marks like " or ' in your json strings.
You MUST not use the same tool with same parameters twice.
You MUST not provide an answer, only collect information and pass it on.
When you can't progress further or need help, ask your colleagues or proceed to {','.join(self.answer_nodes)}.
        """
        if len(self.answer_nodes) == 1:
            prompt += f"\nWhen you are ready to answer, pass the conversation to {self.answer_nodes[0]}."
        elif len(self.answer_nodes) > 1:
            prompt += f"\nWhen you are ready to answer, pass the conversation to one of the following: {', '.join(self.answer_nodes)}."

        if task := state.get("task"):
            prompt += f"\nYour Task: {task}"
        return prompt

    @property
    def handover_to_colleague(self):
        options = self.options

        class HandoverToColleague(BaseModel):
            """Handover the task to a colleague."""

            state: str = Field(
                description="The current state of the task. Provide all information needed to continue."
            )
            task: str = Field(description="The task for the colleague.")
            next: Literal[*options] = Field(description="The colleague to ask.")

        return HandoverToColleague

    @property
    def answer(self):
        options = self.answer_nodes

        class Answer(BaseModel):
            """Answer to the user's question."""

            next: Literal[*options] = Field(description="The next node to go to.")
            answer: str = Field(
                description="The answer to the user's question. With all details extracted from the tools."
            )

        return Answer

    @property
    def all_tools(self):
        if self.answer_nodes:
            return self.tools + [self.handover_to_colleague, self.answer]
        return self.tools + [self.handover_to_colleague]

    def node(self, state: StateType) -> Command[str]:
        model = self.llm.bind_tools(self.all_tools, tool_choice="auto")
        agent = create_react_agent(
            model, self.all_tools, state_modifier=self.system_prompt(state)
        )
        new_messages = []

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

        for msg in agent.stream(state):
            new_messages = msg["messages"]
            last_message = new_messages[-1]
            if tool_calls := getattr(last_message, "tool_calls", []):
                last_tool_call = tool_calls[-1]
                if (
                    last_tool_call.get("name") == "HandoverToColleague"
                    and last_tool_call["args"]["next"] in self.colleague_names
                ):
                    msg["messages"].append(
                        ToolMessage(
                            content=f"Command: Asking {last_tool_call['args']['next']} for help.",
                            tool_call_id=last_tool_call["id"],
                        )
                    )
                    return Command(
                        update={
                            "messages": msg["messages"],
                            "task": last_tool_call["args"]["task"],
                            **update,
                        },
                        goto=last_tool_call["args"]["next"],
                    )
                if last_tool_call.get("name") == "Answer":
                    msg["messages"].append(
                        ToolMessage(
                            content="Command: Task completed.",
                            tool_call_id=last_tool_call["id"],
                        )
                    )
                    return Command(
                        update={"messages": msg["messages"], "task": None, **update},
                        goto=last_tool_call["args"]["next"],
                    )
        return Command(
            update={"messages": new_messages, "task": None, **update},
            goto=Validator.name,
        )
