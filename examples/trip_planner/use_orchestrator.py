from fiboaitech import Workflow
from fiboaitech.connections import Anthropic as AnthropicConnection
from fiboaitech.connections import OpenAI as OpenAIConnection
from fiboaitech.connections import ScaleSerp
from fiboaitech.flows import Flow
from fiboaitech.nodes.agents.base import Agent
from fiboaitech.nodes.agents.orchestrators.adaptive import AdaptiveOrchestrator
from fiboaitech.nodes.agents.orchestrators.adaptive_manager import AdaptiveAgentManager
from fiboaitech.nodes.agents.react import ReActAgent
from fiboaitech.nodes.llms.anthropic import Anthropic
from fiboaitech.nodes.llms.openai import OpenAI
from fiboaitech.nodes.tools.scale_serp import ScaleSerpTool
from fiboaitech.nodes.types import Behavior
from fiboaitech.utils.logger import logger
from examples.trip_planner.prompts import generate_customer_prompt, generate_simple_customer_prompt

# Please use your own file path
OUTPUT_FILE_PATH = "city_guide_gpt.md"
AGENT_SELECTION_CITY_ROLE = (
    "An expert in analyzing travel data to select ideal destinations. "
    "The goal is to help choose the best city for a trip based on specific criteria, "
    "such as weather patterns, seasonal events, and travel costs."
)

AGENT_CITY_GUIDE_ROLE = (
    "An expert in gathering comprehensive information about a city. "
    "The goal is to compile an in-depth guide for someone traveling to a city, "
    "including key attractions, local customs, special events, and daily activity recommendations."
)

AGENT_WRITER_ROLE = (
    "An expert in creating detailed travel guides. "
    "The goal is to write a comprehensive travel guide for a city, "
    "covering key attractions, local customs, special events, and daily activity recommendations."
)


def choose_provider(model_type, model_name):
    if model_type == "gpt":
        _connection = OpenAIConnection()
        _llm = OpenAI(
            connection=_connection,
            model=model_name,
            temperature=0.1,
            max_tokens=4000,
        )
    elif model_type == "claude":
        _connection = AnthropicConnection()
        _llm = Anthropic(
            connection=_connection,
            model=model_name,
            temperature=0.1,
            max_tokens=4000,
        )
    else:
        raise ValueError("Invalid model provider specified.")
    return _llm


def inference(input_data: dict, model_type="gpt", model_name="gpt-4o-mini", use_advanced_prompt=False) -> dict:
    llm_agent = choose_provider(model_type, model_name)
    search_connection = ScaleSerp()
    tool_search = ScaleSerpTool(connection=search_connection)

    # Create agents
    agent_selection_city = ReActAgent(
        name="City Selection Expert",
        role=AGENT_SELECTION_CITY_ROLE,
        llm=llm_agent,
        tools=[tool_search],
        max_loops=10,
        behavior_on_max_loops=Behavior.RETURN,
    )

    agent_city_guide = ReActAgent(
        name="City Guide Expert",
        role=AGENT_CITY_GUIDE_ROLE,
        llm=llm_agent,
        tools=[tool_search],
        max_loops=10,
        behavior_on_max_loops=Behavior.RETURN,
    )

    agent_writer = Agent(
        name="City Guide Writer",
        role=AGENT_WRITER_ROLE,
        llm=llm_agent,
    )
    agent_manager = AdaptiveAgentManager(
        llm=llm_agent,
    )

    # Create a adaptive orchestrator
    adaptive_orchestrator = AdaptiveOrchestrator(
        manager=agent_manager,
        agents=[agent_city_guide, agent_selection_city, agent_writer],
    )
    # Create a workflow
    workflow = Workflow(flow=Flow(nodes=[adaptive_orchestrator]))

    if use_advanced_prompt:
        user_prompt = generate_customer_prompt(input_data)
    else:
        user_prompt = generate_simple_customer_prompt(input_data)

    result = workflow.run(
        input_data={
            "input": user_prompt,
        }
    )
    logger.info("Workflow completed")
    content = result.output[adaptive_orchestrator.id]
    return content


if __name__ == "__main__":
    user_location = input("Enter your location: ")
    user_cities = input("Enter cities you want to visit: ")
    user_dates = input("Enter dates: ")
    user_interests = input("Enter your interests: ")
    input_data = {
        "location": user_location,
        "cities": user_cities,
        "dates": user_dates,
        "interests": user_interests,
    }
    content = inference(input_data)["output"]["content"]
    print(content)
    with open(OUTPUT_FILE_PATH, "w") as f:
        f.write(content)
