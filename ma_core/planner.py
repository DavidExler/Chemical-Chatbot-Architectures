import json
from dataclasses import dataclass
from typing import TypedDict, Generic, Optional, Literal, ClassVar

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from ma_core import StateType
from ma_core.answerer import Answerer


@dataclass
class Planner(Generic[StateType]):
    llm: ChatOpenAI
    goto_options: list[str]
    end_node: str = Answerer.name
    name: ClassVar[str] = "Planner"

    def system_prompt(self, state: StateType) -> str:
        return f"""
        You are a planner. Your task is to plan the work for the researcher and chemist.
        Split the main task into small subtasks, start with tasks for the researcher and later on the chemist can pick up the task with the information provided.
        Make sure to always respond with a new "current_task" if the task is not finished.
        If all tasks are completed you MUST verify them by asking all the collaborators to verify the solution a second time.
        ONLY if all collaborators agree with the solution, pass the conversation to {self.end_node}. 
        If the validator refuses the answer, you MUST create a new plan and pass the conversation back to the researcher or chemist.
        Past Tasks: {state.get("past_tasks", [])}
        """

    @property
    def response_structure(self):
        options = self.goto_options + [self.end_node]

        @dataclass
        class PlanReplanResponse(TypedDict):
            """Structured output for planning."""

            next: Literal[*options]
            current_task: Optional[str]
            reason: Optional[str]

        return PlanReplanResponse

    def node(self, state: StateType) -> Command[str]:
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

        messages = [
            {"role": "system", "content": self.system_prompt(state)},
        ] + state["messages"]
        response = self.llm.with_structured_output(self.response_structure).invoke(
            messages
        )
        return Command(
            goto=response["next"],
            update={
                "messages": [
                    HumanMessage(content=json.dumps(response), name=self.name)
                ],
                "task": response["current_task"],
                "past_tasks": [response["current_task"]],
                "reason": response["reason"],
                **update,
            },
        )
