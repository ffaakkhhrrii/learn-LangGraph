import os
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
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
graph = builder.compile(checkpointer=memory)

initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}
thread = {"configurable": {"thread_id": "thread-1"}}  # misalnya

for event in graph.stream(initial_input, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()

last_state = graph.get_state(thread)
print("Latest checkpoint ID:", last_state.config["configurable"]["checkpoint_id"])


history = list(graph.get_state_history(thread))
print("History checkpoints (latest first):")
for snap in history:
    print("  checkpoint_id:", snap.config["configurable"]["checkpoint_id"],
          " next:", snap.next,
          " values:", snap.values)

target = history[-1]  
print("Will time-travel to checkpoint:", target.config["configurable"]["checkpoint_id"])

new_values = dict(target.values)
new_values["messages"] = [HumanMessage(content="Multiply 3 and 3")]

new_config = graph.update_state(target.config, new_values)
print("Forked new thread/config:", new_config)

for event in graph.stream(None, new_config, stream_mode="values"):
    event['messages'][-1].pretty_print()