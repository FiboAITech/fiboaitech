from fiboaitech.memory import Memory
from fiboaitech.memory.backends.in_memory import InMemory
from fiboaitech.nodes.agents.simple import SimpleAgent
from examples.llm_setup import setup_llm


def setup_agent():
    llm = setup_llm()
    memory = Memory(backend=InMemory())
    AGENT_ROLE = "Helpful assistant with the goal of providing useful information and answering questions."
    agent = SimpleAgent(
        name="Agent",
        llm=llm,
        role=AGENT_ROLE,
        id="agent",
        memory=memory,
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
