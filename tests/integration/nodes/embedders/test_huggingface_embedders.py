import uuid

from fiboaitech import Workflow, connections
from fiboaitech.callbacks import TracingCallbackHandler
from fiboaitech.flows import Flow
from fiboaitech.nodes.embedders import HuggingFaceDocumentEmbedder, HuggingFaceTextEmbedder
from fiboaitech.runnables import RunnableConfig, RunnableResult, RunnableStatus
from fiboaitech.types import Document


def test_workflow_with_huggingface_text_embedder(mock_embedding_executor):
    connection = connections.HuggingFace(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    model = "huggingface/BAAI/bge-large-zh"
    wf_huggingface_ai = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                HuggingFaceTextEmbedder(
                    name="HuggingFaceTextEmbedder", connection=connection, model=model
                ),
            ],
        ),
    )
    input = {"query": "I love pizza!"}
    response = wf_huggingface_ai.run(
        input_data=input,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input,
        output={"query": "I love pizza!", "embedding": [0]},
    ).to_dict()
    expected_output = {wf_huggingface_ai.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input,
        output=expected_output,
    )
    assert response.output == expected_output
    mock_embedding_executor.assert_called_once_with(
        input=[input["query"]],
        model=model,
        api_key=connection.api_key,
    )


def test_workflow_with_huggingface_document_embedder(mock_embedding_executor):
    connection = connections.HuggingFace(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    model = "huggingface/BAAI/bge-large-zh"
    wf_huggingface_ai = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                HuggingFaceDocumentEmbedder(
                    name="HuggingFaceDocumentEmbedder",
                    connection=connection,
                    model=model,
                ),
            ],
        ),
    )
    document = [Document(content="I love pizza!")]
    input = {"documents": document}
    response = wf_huggingface_ai.run(
        input_data=input,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input,
        output={
            **input,
            "meta": {
                "model": model,
                "usage": {
                    "usage": {
                        "prompt_tokens": 6,
                        "completion_tokens": 0,
                        "total_tokens": 6,
                    }
                },
            },
        },
    ).to_dict()
    expected_output = {wf_huggingface_ai.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input,
        output=expected_output,
    )
    assert response.output == expected_output
    mock_embedding_executor.assert_called_once_with(
        input=[document[0].content],
        model=model,
        api_key=connection.api_key,
    )
