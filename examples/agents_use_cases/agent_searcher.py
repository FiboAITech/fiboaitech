from fiboaitech.connections import Exa
from fiboaitech.nodes.agents.react import ReActAgent
from fiboaitech.nodes.tools.exa_search import ExaTool
from fiboaitech.nodes.types import InferenceMode
from fiboaitech.utils.logger import logger
from examples.llm_setup import setup_llm

if __name__ == "__main__":
    connection_exa = Exa()
    tool_search = ExaTool(connection=connection_exa)
    llm = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", temperature=0)
    agent = ReActAgent(
        name="Agent",
        id="Agent",
        llm=llm,
        tools=[tool_search],
        inference_mode=InferenceMode.XML,
    )
    result = agent.run(input_data={"input": "Who won USA elections 2024?"})
    output_content = result.output.get("content")
    logger.info("RESULT")
    logger.info(output_content)
