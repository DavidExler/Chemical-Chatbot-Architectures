from dataclasses import dataclass

from langchain_core.runnables import Runnable


@dataclass
class Generation:
    text: str


@dataclass
class Generations:
    generations: list[list[Generation]]

    def __getitem__(self, index):
        return self.generations[index]


@dataclass
class ModelWrapper:
    chain: Runnable

    def generate(self, prompts: list[str]) -> (Generations, dict):
        for p in prompts:
            input = {"messages": [("user", p)]}
            result = self.chain.invoke(input)["messages"][-1]
            logprobs = result.response_metadata.get("logprobs")
            return Generations([[Generation(text=result.content)]]), logprobs
