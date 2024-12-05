from fiboaitech.connections import OpenAI as OpenAIConnection
from fiboaitech.connections import Pinecone as PineconeConnection
from fiboaitech.memory import Memory
from fiboaitech.memory.backends import Pinecone
from fiboaitech.nodes.agents.simple import SimpleAgent
from fiboaitech.nodes.embedders import OpenAIDocumentEmbedder
from fiboaitech.storages.vector.pinecone.pinecone import PineconeIndexType
from examples.llm_setup import setup_llm


def setup_agent():
    llm = setup_llm()
    pinecone_connection = PineconeConnection()
    openai_connection = OpenAIConnection()
    embedder = OpenAIDocumentEmbedder(connection=openai_connection)

    backend = Pinecone(
        index_name="test-conv",
        connection=pinecone_connection,
        embedder=embedder,
        index_type=PineconeIndexType.SERVERLESS,
        cloud="aws",
        region="us-east-1",
    )

    memory_pinecone = Memory(backend=backend)

    AGENT_ROLE = "helpful assistant," "goal is to provide useful information and answer questions"
    agent = SimpleAgent(
        name="Agent",
        llm=llm,
        role=AGENT_ROLE,
        id="agent",
        memory=memory_pinecone,
    )
    return agent


def chat_loop(agent):
    print("Welcome to the AI Chat! (Type 'exit' to end)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        response = agent.run({"input": user_input})
        response_content = response.output.get("content")
        print(f"AI: {response_content}")

    print("\nChat History:")
    print(agent.memory.get_all_messages_as_string())


if __name__ == "__main__":
    chat_agent = setup_agent()
    chat_loop(chat_agent)
