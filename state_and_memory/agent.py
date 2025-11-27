import os
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, AnyMessage, SystemMessage, RemoveMessage
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.checkpoint.memory import MemorySaver
import sqlite3 
from langgraph.checkpoint.sqlite import SqliteSaver


from dotenv import load_dotenv

load_dotenv()

db_path = "db/example.db"
conn = sqlite3.connect(db_path, check_same_thread=False)

model_name = os.getenv("LLM_MODEL")
model_key = os.getenv("LLM_API_KEY")
model_provider = os.getenv("LLM_PROVIDER")

chat = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=0.9,
    google_api_key=model_key,
)

class State(MessagesState): 
    summary: str

def call_model(state: State):
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier : {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]

    response = chat.invoke(messages)
    return {"messages": [response]}

def summarize_conversation(state: State):
    # Get any existing summary
    summary = state.get("summary", "")
    # Create our summarization prompt
    if summary:
        summary_message = (
            f"This is summary of the conversation to date : {summary}.\n\n"
            "Extend the summary by taking into account the new messages above"
        )
    else:
        summary_message = (
            "Create a summary of the conversation above"
        )
    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = chat.invoke(messages)

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}
    
# Determine whether to end or summarize the conversation
def should_continue(state: State):
    """Return the next code to execute."""
    messages = state["messages"]
    if len(messages) >= 6:
        return "summarize_conversation"
    
    return END

# Define a new graph
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

# Set the entrpoint as a conversation
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges(
    "conversation",
    should_continue,
)
workflow.add_edge("summarize_conversation", END)

memory = SqliteSaver(conn)
graph = workflow.compile(checkpointer=memory)

# Create a thread
config = {"configurable": {"thread_id": "2"}}

# Start conversation
input_message = HumanMessage(content="hi! I'm Fakhri")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()

input_message = HumanMessage(content="what's my name?")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()

input_message = HumanMessage(content="i like fc barcelona!")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()

input_message = HumanMessage(content="i love messi!")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()

current = graph.get_state(config).values.get("summary", "")
print("\nCurrent Summary:", current)