import os
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import operator
from typing import Annotated
from pydantic import BaseModel
from langgraph.types import Send

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

# Prompts we will use
subjects_prompt = """Generate a list of 3 sub-topics that are all related to this overall topic: {topic}."""
joke_prompt = """Generate a joke about {subject}"""
best_joke_prompt = """Below are a bunch of jokes about {topic}. Select the best one! Return the ID of the best one, starting 0 as the ID for the first joke. Jokes: \n\n  {jokes}"""

# Define the classes for structured output

class Subjects(BaseModel):
    subjects: list[str]

class BestJoke(BaseModel):
    id: int
    
class OverallState(TypedDict):
    topic: str
    subjects: list
    jokes: Annotated[list, operator.add]
    best_selected_joke: str

class JokeState(TypedDict):
    subject: str

class Joke(BaseModel):
    joke: str


# Define the graph nodes
# Map
def generate_topics(state: OverallState):
    prompt = subjects_prompt.format(topic=state["topic"])
    response = chat.with_structured_output(Subjects).invoke(prompt)
    return {"subjects": response.subjects}

def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]


def generate_joke(state: JokeState):
    prompt = joke_prompt.format(subject=state["subject"])
    response = chat.with_structured_output(Joke).invoke(prompt)
    return {"jokes": [response.joke]}

# Reduce
def best_joke(state: OverallState):
    jokes = "\n\n".join(state["jokes"])
    prompt = best_joke_prompt.format(topic=state["topic"], jokes=jokes)
    response = chat.with_structured_output(BestJoke).invoke(prompt)
    return {"best_selected_joke": state["jokes"][response.id]}


graph = StateGraph(OverallState)
graph.add_node("generate_topics", generate_topics)
graph.add_node("generate_joke", generate_joke)
graph.add_node("best_joke", best_joke)
graph.add_edge(START, "generate_topics")
graph.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
graph.add_edge("generate_joke", "best_joke")
graph.add_edge("best_joke", END)
app = graph.compile()

# Run the graph
initial_input = {"topic": "programming"}
for event in app.stream(initial_input, stream_mode="values"):
    print(event)