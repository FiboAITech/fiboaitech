import json

from fiboaitech import Workflow
from fiboaitech.callbacks import TracingCallbackHandler
from fiboaitech.flows import Flow
from fiboaitech.nodes.agents.reflection import ReflectionAgent
from fiboaitech.runnables import RunnableConfig
from fiboaitech.utils import JsonWorkflowEncoder
from examples.llm_setup import setup_llm

# Constants
AGENT_NAME = "Role Agent"
AGENT_ROLE = "professional writer, goal is to produce a well written and informative response with emoji for childrens"
INPUT_QUESTION = "how sin(x) and cos(x) connected in electrodynamics?"


def run_workflow() -> tuple[str, dict]:
    """
    Set up and run a workflow using a ReflexionAgent with OpenAI's language model.

    The workflow processes the input question "What is the capital of France?"
    using a professional writer agent.

    Returns:
        str: The output content generated by the agent, or an empty string if an error occurs.

    Raises:
        Exception: Any exception that occurs during the workflow execution is caught and printed.
    """
    # Set up OpenAI connection and language model
    llm = setup_llm()
    agent = ReflectionAgent(
        name=AGENT_NAME,
        llm=llm,
        role=AGENT_ROLE,
        id="agent",
    )

    # Set up tracing and create the workflow
    tracing = TracingCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent]))

    # Run the workflow and handle the result
    try:
        result = wf.run(
            input_data={"input": INPUT_QUESTION},
            config=RunnableConfig(callbacks=[tracing]),
        )

        # Verify that traces can be serialized to JSON
        json.dumps(
            {"runs": [run.to_dict() for run in tracing.runs.values()]},
            cls=JsonWorkflowEncoder,
        )

        return result.output[agent.id]["output"]["content"], tracing.runs
    except Exception as e:
        print(f"An error occurred: {e}")
        return "", {}


if __name__ == "__main__":
    output, _ = run_workflow()
    print(output)
