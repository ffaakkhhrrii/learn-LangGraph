import os
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, AnyMessage
from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv

load_dotenv()

model_name = os.getenv("LLM_MODEL")
model_key = os.getenv("LLM_API_KEY")
model_provider = os.getenv("LLM_PROVIDER")

chat = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=0.9,
    google_api_key=model_key,
)

class MessagesState(TypedDict):
    messages: list[AnyMessage]

def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

tools = [multiply]

llm_with_tools = chat.bind_tools(tools)

def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state['messages'])]}

builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", END)
graph = builder.compile()

messages = graph.invoke({"messages": [HumanMessage(content="Hello World! can u multiply 8 and 9 ?")]})

print(messages)
for m in messages['messages']:
    m.pretty_print()