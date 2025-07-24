import json
import logging
from dataclasses import dataclass
from typing import ClassVar, TypedDict

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from typing_extensions import Generic

from ma_core import StateType

LOGGER = logging.getLogger(__name__)


@dataclass
class Validator(Generic[StateType]):
    llm: ChatOpenAI
    backward_node: str
    forward_node: str
    name: ClassVar[str] = "Validator"

    @property
    def system_prompt(self) -> str:
        return """
You are a reviewer. Validate the results of the previous workers. Be very critical and make sure the information is correct.

**Validate the results by this checklist:**
<checklist>
1. Is the information based on facts and logic that have been retrieved from the tools?
2. Has enough research been done to ground the information?
3. Can the user question be answered with the information provided?
4. Have all collaborators validated the information after the answer was given?
</checklist>

Step through the checklist and validate the results of the previous workers.
Only allow the conversation to proceed if the results are correct.
        """

    @property
    def router(self):
        class Router(TypedDict):
            """Worker to route to next. If no workers needed, route to FINISH."""

            is_information_based_on_facts: bool
            is_enough_research_done: bool
            is_user_question_answerable: bool
            have_all_collaborators_validated: bool

        return Router

    def node(self, state: StateType) -> Command[str]:
        has_tool_messages = any(
            msg
            for msg in state["messages"]
            if isinstance(msg, ToolMessage) and not msg.content.startswith("Command:")
        )
        if not has_tool_messages:
            LOGGER.warning("No tool messages found.")
            return Command(
                goto=self.backward_node,
                update={
                    "messages": [
                        HumanMessage(
                            content="No tool messages found. Please ground your answer using the tools.",
                            name=self.name,
                        )
                    ]
                },
            )
        messages = [
            {"role": "system", "content": self.system_prompt},
        ] + state["messages"]
        response = self.llm.with_structured_output(self.router).invoke(messages)
        return Command(
            goto=self.forward_node if all(response.values()) else self.backward_node,
            update={
                "messages": [HumanMessage(content=json.dumps(response), name=self.name)]
            },
        )
