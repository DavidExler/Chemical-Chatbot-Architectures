from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable

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

TEMPLATE = """
say hello to {someInput} please
            """
def get_prompt() -> BasePromptTemplate:
    return PromptTemplate(input_variables=["someInput"], template=TEMPLATE)
def get_chain() -> Runnable:
    prompt_template = get_prompt()
    return prompt_template | llm


if __name__ == "__main__":
    chain = get_chain()

    context = "Human Question: What is the peroxidase enzyme? Assistent Answer: Peroxidase is a large group of enzymes that break down peroxides and play a role in various biological processes. There are different types of peroxidases, including glutathione peroxidase, which protects organisms from oxidative damage, and horseradish peroxidase, which is commonly used in biochemistry applications."
    conversation_name = chain.invoke(context)["text"]
    print(conversation_name)


