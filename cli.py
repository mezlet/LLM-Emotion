# ollama_chat/cli.py

from ollama_chat.client import OllamaChat


def run_chat():
    chat = OllamaChat(
        model="llama3.2:1b",  # change to qwen3:8b if needed
    )

    print("Streaming local chat started.")
    print("Commands:")
    print("  /exit   - quit")
    print("  /quit   - quit")
    print("  /clear  - clear conversation\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_text:
            continue

        if user_text.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("Goodbye.")
            break

        if user_text.lower() == "/clear":
            chat.clear()
            print("Conversation cleared.\n")
            continue

        try:
            print("\nAssistant: ", end="", flush=True)

            for piece in chat.stream(user_text):
                print(piece, end="", flush=True)

            print("\n")

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    run_chat()