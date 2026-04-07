# Importing the Ollama client, which allows Python to talk to the local Ollama server
from ollama import Client


# Address of the local Ollama server (running via `rrlab-ollama serve`)
OLLAMA_HOST = "http://127.0.0.1:11434"

# The ollama model
MODEL_NAME = "llama3.2:1b"


def main() -> None:
    # Creating a client that connects to the Ollama server
    client = Client(host=OLLAMA_HOST)

    # This list stores the entire conversation history
    # The "system" message defines the assistant's behavior
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        }
    ]

    # Print startup instructions for the user
    print("Streaming local chat with llama3.2:1b started.")
    print("Commands:")
    print("  /exit   - quit")
    print("  /quit   - quit")
    print("  /clear  - clear conversation history\n")

    #loop for continuous chat
    while True:
        try:
            # Read input from the keyboard
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl+D or Ctrl+C gracefully
            print("\nGoodbye.")
            break

        # If the user entered nothing, skip this iteration
        if not user_text:
            continue

        # Exit commands
        if user_text.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("Goodbye.")
            break

        # Clear conversation history (reset to system prompt only)
        if user_text.lower() == "/clear":
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                }
            ]
            print("Conversation cleared.\n")
            continue

        # Add the user's message to the conversation history
        messages.append({"role": "user", "content": user_text})


        try:
            # Send the conversation to the model
            # stream=True means we receive the response piece by piece (like typing)
            stream = client.chat(
                model=MODEL_NAME,
                messages=messages,
                stream=True,
            )

            # Print assistant label without newline
            print("\nAssistant: ", end="", flush=True)

            # store the full response as we receive it
            full_reply = ""

            # Iterate over streamed chunks from the model
            for chunk in stream:
                # Extract the text piece from the chunk
                piece = chunk["message"]["content"]

                # Build the full response
                full_reply += piece

                # Print each piece immediately (streaming effect)
                print(piece, end="", flush=True)

            # Print a newline after response is complete
            print("\n")

            # Add the assistant's full reply to conversation history
            messages.append({"role": "assistant", "content": full_reply})

        except Exception as e:
            # Handling errors
            print(f"\nError talking to local Ollama server: {e}\n")

            # Remove the last user message to keep history consistent
            messages.pop()


# Standard Python entry point, ensures main() runs only when the script is executed directly
if __name__ == "__main__":
    main()