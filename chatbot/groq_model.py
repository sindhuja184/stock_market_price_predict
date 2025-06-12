from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Define your LLM
llm = ChatGroq(
    groq_api_key="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",
    temperature=0.3,
    model_name="llama3-8b-8192"
)

# Define a prompt template
template = """
You are a smart stock market assistant.

If the user greets you, respond politely.

If user asks for a stock market relatd term and professionally like you have invested in stock marker

If the query is about stock explanations or trends:
- Provide simple insights about trends, technical indicators (like SMA, EMA), or general stock market behavior.

Input: {user_input}
"""

prompt = ChatPromptTemplate.from_template(template)

# Wrap the LLM with the prompt
from langchain.chains import LLMChain

chain = prompt | llm

 
