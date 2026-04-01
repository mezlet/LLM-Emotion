from ollama import Client


OLLAMA_HOST = "http://127.0.0.1:11434"
MODEL_NAME = "qwen3:8b"


def main() -> None:
    client = Client(host=OLLAMA_HOST)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        }
    ]

    print("Streaming local chat with qwen3:8b started.")
    print("Commands:")
    print("  /exit   - quit")
    print("  /quit   - quit")
    print("  /clear  - clear conversation history\n")

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
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                }
            ]
            print("Conversation cleared.\n")
            continue

        messages.append({"role": "user", "content": user_text})


        try:
            stream = client.chat(
                model=MODEL_NAME,
                messages=messages,
                stream=True,
            )

            print("\nAssistant: ", end="", flush=True)
            full_reply = ""

            for chunk in stream:
                piece = chunk["message"]["content"]
                full_reply += piece
                print(piece, end="", flush=True)

            print("\n")
            messages.append({"role": "assistant", "content": full_reply})

        except Exception as e:
            print(f"\nError talking to local Ollama server: {e}\n")
            messages.pop()


if __name__ == "__main__":
    main()