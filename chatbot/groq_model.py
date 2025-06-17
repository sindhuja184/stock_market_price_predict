from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

 
llm = ChatGroq(
    groq_api_key="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",
    temperature=0.3,
    model_name="llama3-8b-8192"
)

 
template = """
You are a smart stock market assistant.

If the user greets you, respond politely.

If user asks for a stock market relatd term and professionally like you have invested in stock marker

If the query is about stock explanations or trends:
- Provide simple insights about trends, technical indicators (like SMA, EMA), or general stock market behavior.

Input: {user_input}
"""

prompt = ChatPromptTemplate.from_template(template)


chain = prompt | llm

 
