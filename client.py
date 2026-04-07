# ollama_chat/client.py

from ollama import Client


class OllamaChat:
    """
    A reusable chat client for interacting with a local Ollama server.
    """

    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        model: str = "llama3.2:1b",
        system_prompt: str = "You are a helpful assistant.",
        max_messages: int = 8,
    ) -> None:
        self.client = Client(host=host)
        self.model = model
        self.system_prompt = system_prompt
        self.max_messages = max_messages

        # Initialize conversation history
        self.messages = [{"role": "system", "content": system_prompt}]

    def clear(self) -> None:
        """Reset conversation history."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def _trim_history(self) -> None:
        """
        Keep conversation short to improve performance.
        Keeps system message + last N messages.
        """
        self.messages = [self.messages[0]] + self.messages[-(self.max_messages - 1):]

    def send(self, user_text: str) -> str:
        """
        Send a message and return full response (non-streaming).
        """
        self.messages.append({"role": "user", "content": user_text})
        self._trim_history()

        response = self.client.chat(
            model=self.model,
            messages=self.messages,
        )

        assistant_text = response["message"]["content"]
        self.messages.append({"role": "assistant", "content": assistant_text})

        return assistant_text

    def stream(self, user_text: str):
        """
        Send a message and yield response chunks (streaming).
        """
        self.messages.append({"role": "user", "content": user_text})
        self._trim_history()

        stream = self.client.chat(
            model=self.model,
            messages=self.messages,
            stream=True,
        )

        full_reply = ""

        for chunk in stream:
            piece = chunk["message"]["content"]
            full_reply += piece
            yield piece

        # Save full reply after streaming completes
        self.messages.append({"role": "assistant", "content": full_reply})