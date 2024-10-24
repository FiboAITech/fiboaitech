import io

from fiboaitech.connections import E2B
from fiboaitech.nodes.agents.react import ReActAgent
from fiboaitech.nodes.tools.e2b_sandbox import E2BInterpreterTool
from fiboaitech.nodes.types import InferenceMode
from fiboaitech.utils.logger import logger
from examples.llm_setup import setup_llm

AGENT_ROLE = """
 Senior Data Scientist and Programmer with ability to write a well written
 python code and you have access to python tool.
 You have access to web to search for best solutions for a problem.
 Generally, you follow these rules:
    - ALWAYS FORMAT YOUR RESPONSE IN MARKDOWN
    - Use double quotes for property names
    - Make code correct and runnable test code and reiterate if does not work
"""

PROMPT = """
 Write code in Python that fits linear regression model between
 number of bathrooms and bedrooms) and price of a house from the data.
 Count loss. Return this code. Set a seed that results would be reproducable.
 Provide exact result of MSE.
"""

FILE_PATH = "data.csv"

FILE_DESCRIPTION = """
- It's `data.csv` file
- The CSV file is using , as the delimiter
- It has the following columns (examples included):
    - bedrooms: number of bedrooms
    - bathrooms: number of bathrooms
    - price: price of a house
"""


def read_file_as_bytesio(file_path: str, filename: str = None, description: str = None) -> io.BytesIO:
    """
    Reads the content of a file and returns it as a BytesIO object with custom attributes for filename and description.

    Args:
        file_path (str): The path to the file.
        filename (str, optional): Custom filename for the BytesIO object.
        description (str, optional): Custom description for the BytesIO object.

    Returns:
        io.BytesIO: The file content in a BytesIO object with custom attributes.
    """
    with open(file_path, "rb") as f:
        file_content = f.read()

    file_io = io.BytesIO(file_content)

    file_io.name = filename if filename else "uploaded_file.csv"
    file_io.description = description if description else "No description provided"

    return file_io


def create_agent():
    """
    Create and configure the agent with necessary tools.

    Returns:
        ReActAgent: A configured FiboAITech ReActAgent ready to run.
    """
    tool = E2BInterpreterTool(connection=E2B())
    llm = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", temperature=0.001)

    agent_software = ReActAgent(
        name="React Agent",
        llm=llm,
        tools=[tool],
        role=AGENT_ROLE,
        max_loops=10,
        inference_mode=InferenceMode.XML,
    )

    return agent_software


def run_workflow(prompt: str, files_to_upload: list[io.BytesIO]) -> tuple[str, dict]:
    """
    Main function to set up and run the workflow, handling any exceptions that may occur.

    Args:
        prompt (str): Question/task for the agent to accomplish.
        files_to_upload (List[io.BytesIO]): A list of BytesIO objects representing files to upload.

    Returns:
        tuple[str, dict]: The content generated by the agent and intermediate steps.
    """
    try:
        agent = create_agent()

        result = agent.run(
            input_data={"input": prompt, "files": files_to_upload},
        )

        return result["content"], result.get("intermediate_steps", {})
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return "", {}


if __name__ == "__main__":
    csv_file_io = read_file_as_bytesio(
        file_path=FILE_PATH, filename="custom_house_data.csv", description=FILE_DESCRIPTION
    )

    output, steps = run_workflow(prompt=PROMPT, files_to_upload=[csv_file_io])

    logger.info("---------------------------------Result-------------------------------------")
    logger.info(output)
