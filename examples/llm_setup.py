from fiboaitech.connections import Anthropic as AnthropicConnection
from fiboaitech.connections import Cohere as CohereConnection
from fiboaitech.connections import Gemini as GeminiConnection
from fiboaitech.connections import Groq as GroqConnection
from fiboaitech.connections import OpenAI as OpenAIConnection
from fiboaitech.nodes.llms.anthropic import Anthropic
from fiboaitech.nodes.llms.cohere import Cohere
from fiboaitech.nodes.llms.gemini import Gemini
from fiboaitech.nodes.llms.groq import Groq
from fiboaitech.nodes.llms.openai import OpenAI

MODEL_NAME_GPT = "gpt-4o"
MODEL_NAME_CLAUDE = "claude-3-5-sonnet-20240620"
MODEL_NAME_COHERE = "command-r-plus"
MODEL_NAME_GROQ = "groq/llama3-70b-8192"
MODEL_NAME_GEMINI = "gemini/gemini-1.5-pro-latest"
MODEL_PROVIDER = "gpt"
MODEL_NAME = MODEL_NAME_GPT
TEMPERATURE = 0.1
MAX_TOKENS = 4000


def setup_llm(
    model_provider: str = MODEL_PROVIDER,
    model_name: str = MODEL_NAME,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
):
    """
    Set up and return an LLM based on the specified model provider.

    Args:
        model_provider (str): The model provider to use, either "claude" or "gpt".
        model_name (str): The name of the  model to use.
        temperature (float): The temperature parameter for the LLM.
        max_tokens (int): The maximum number of tokens for the LLM.

    Returns:
        The configured LLM.
    """
    if model_provider == "claude":
        return Anthropic(
            name="Anthropic LLM",
            connection=AnthropicConnection(),
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif model_provider == "gpt":
        return OpenAI(
            name="OpenAI LLM",
            connection=OpenAIConnection(),
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif model_provider == "cohere":
        return Cohere(
            name="Cohere LLM",
            connection=CohereConnection(),
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif model_provider == "groq":
        return Groq(
            name="Groq LLM",
            connection=GroqConnection(),
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif model_provider == "gemini":
        return Gemini(
            name="Gemini LLM",
            connection=GeminiConnection(),
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(f"Invalid model provider: {model_provider}")
