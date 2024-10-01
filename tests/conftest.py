import pytest
from fakeredis import FakeRedis
from litellm import ModelResponse
from litellm.types.utils import EmbeddingResponse
from litellm.utils import Delta

from fiboaitech import connections, prompts
from fiboaitech.cache.backends import RedisCache
from fiboaitech.clients import BaseTracingClient
from fiboaitech.nodes import llms
from fiboaitech.types.document import Document


@pytest.fixture(autouse=True)
def autouse_fixture(
    mock_llm_executor,
    mock_tracing_client,
    mock_redis_backend,
): ...


@pytest.fixture
def mock_llm_response_text():
    return "mocked_response"


@pytest.fixture
def mock_whisper_response_text():
    return "Welcome."


@pytest.fixture
def mock_elevenlabs_response_text():
    return b"Mock text"


@pytest.fixture
def mock_redis():
    return FakeRedis()


@pytest.fixture
def mock_llm_executor(mocker, mock_llm_response_text):
    def mock_completion_streaming_obj(mock_response):
        for chunk in mock_response:
            model_r = ModelResponse(stream=True)
            model_r.choices[0].delta = Delta(**{"role": "assistant", "content": chunk})
            yield model_r

    def response(stream: bool, *args, **kwargs):
        if stream:
            return mock_completion_streaming_obj(mock_response=mock_llm_response_text)

        model_r = ModelResponse()
        model_r["choices"][0]["message"]["content"] = mock_llm_response_text
        return model_r

    mock_llm = mocker.patch("fiboaitech.nodes.llms.base.BaseLLM._completion", side_effect=response)
    yield mock_llm


@pytest.fixture
def mock_embedding_executor(mocker):
    def response(*args, **kwargs):
        embed_r = EmbeddingResponse()
        embedding_data = {"embedding": [0]}
        embed_r["data"] = [embedding_data]
        embed_r["model"] = kwargs.get("model")
        embed_r["usage"] = {"usage": {"prompt_tokens": 6, "completion_tokens": 0, "total_tokens": 6}}
        return embed_r

    mock_llm = mocker.patch(
        "fiboaitech.components.embedders.base.BaseEmbedder._embedding",
        side_effect=response,
    )
    yield mock_llm


@pytest.fixture
def mock_tracing_client(mocker):
    class MockTracingClient(BaseTracingClient):
        def trace(self, runs) -> None:
            pass

    mocker.patch.object(MockTracingClient, "trace")
    yield MockTracingClient


@pytest.fixture
def mock_redis_backend(mocker, mock_redis):
    yield mocker.patch(
        "fiboaitech.cache.backends.RedisCache.from_config",
        return_value=RedisCache(client=mock_redis),
    )


@pytest.fixture()
def ai_prompt():
    return prompts.Prompt(
        messages=[
            prompts.Message(
                role="user",
                content="What is AI?",
            ),
        ],
    )


@pytest.fixture()
def ds_prompt():
    return prompts.Prompt(
        messages=[
            prompts.Message(
                role="user",
                content="What is DS?",
            ),
        ],
    )


@pytest.fixture()
def openai_node(ai_prompt):
    return llms.OpenAI(
        name="OpenAI",
        model="gpt-3.5-turbo",
        connection=connections.OpenAI(
            api_key="test-api-key",
        ),
        prompt=ai_prompt,
        is_postponed_component_init=True,
    )


@pytest.fixture()
def anthropic_node(ds_prompt):
    return llms.Anthropic(
        name="Anthropic",
        model="claude-3-opus-20240229",
        connection=connections.Anthropic(
            api_key="test-api-key",
        ),
        prompt=ds_prompt,
        is_postponed_component_init=True,
    )


@pytest.fixture
def mock_documents():
    return [
        Document(id="1", content="Document 1", embedding=[0.1, 0.1, 0.2], metadata={"file_id": "file_id_1"}),
        Document(id="2", content="Document 2", embedding=[0.1, 0.1, 0.2], metadata={"file_id": "file_id_1"}),
    ]


@pytest.fixture()
def mock_filters():
    return {
        "operator": "AND",
        "conditions": [
            {"field": "years", "operator": "==", "value": "2019"},
            {"field": "companies", "operator": "in", "value": ["BMW", "Mercedes"]},
        ],
    }
