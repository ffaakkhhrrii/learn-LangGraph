import os
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, AnyMessage, SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import MemorySaver

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

def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

tools = [multiply, divide, add]
llm_with_tools = chat.bind_tools(tools)

sys_msg = SystemMessage(content="You are a helpful assistant named ArithmeticBot tasked with writing performing arithmetic on a set of inputs.")

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
memory = MemorySaver()
graph = builder.compile(checkpointer=memory,interrupt_before=["assistant"])

# Input
initial_input = {"messages": "Multiply 2 and 3"}

# Thread
thread = {"configurable": {"thread_id": "1"}}

# Run the graph until the first interruption
for event in graph.stream(initial_input, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()

graph.update_state(
    thread,
    {"messages": [HumanMessage(content="No, actually multiply 3 and 3!")]},
)

# Continue running the graph until the next interruption
for event in graph.stream(None, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()