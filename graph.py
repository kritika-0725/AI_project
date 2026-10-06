from typing import TypedDict

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from core.gemini import generate_response


class NexaMindState(TypedDict):

    question: str
    context: str
    history: list
    answer: str


def generate_answer(
    state: NexaMindState
):

    answer = generate_response(
        question=state["question"],
        context=state["context"],
        history=state["history"],
    )

    return {
        "answer": answer
    }


def create_graph():

    workflow = StateGraph(
        NexaMindState
    )

    workflow.add_node(
        "generate_answer",
        generate_answer
    )

    workflow.add_edge(
        START,
        "generate_answer"
    )

    workflow.add_edge(
        "generate_answer",
        END
    )

    return workflow.compile()
