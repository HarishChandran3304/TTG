from certifi import contents
from google import genai
from dotenv import load_dotenv
import os
from typing import Optional

from google.genai.chats import Chats, Chat, AsyncChat

load_dotenv()


class KeyManager:
    def __init__(self):
        self.main_key = os.getenv("GEMINI_API_KEY")
        self.fallback_count = int(os.getenv("FALLBACK_COUNT", "0"))
        self.fallback_keys = [
            os.getenv(f"FALLBACK_{i}")
            for i in range(1, self.fallback_count + 1)
            if os.getenv(f"FALLBACK_{i}")
        ]
        self.current_key_index = 0  # Start with main key
        self.tried_keys = set()
        self.client = genai.Client(api_key=self.main_key)

    def get_next_key(self) -> Optional[str]:
        """
        Get the next API key to use.
        If the main key is exhausted, it will try the fallback keys in order.
        Returns:
            The next API key to use, or None if all keys have been exhausted.
        """
        if self.current_key_index == 0:  # If we're on main key
            self.tried_keys.add(self.main_key)
            if self.fallback_keys:  # If we have fallback keys
                self.current_key_index = 1
                next_key = self.fallback_keys[0]
                self.client = genai.Client(api_key=next_key)
                return next_key
        else:  # If we're on a fallback key
            current_key = self.fallback_keys[self.current_key_index - 1]
            self.tried_keys.add(current_key)
            if self.current_key_index < len(self.fallback_keys):
                next_key = self.fallback_keys[self.current_key_index]
                self.current_key_index += 1
                self.client = genai.Client(api_key=next_key)
                return next_key
        return None

    def reset(self):
        """
        Reset the key manager to its initial state.
        This is useful for reinitializing the API client after all keys have been exhausted.
        """
        self.current_key_index = 0
        self.tried_keys.clear()
        self.client = genai.Client(api_key=self.main_key)


# Create a global instance of KeyManager
key_manager = KeyManager()


async def initialize_chat(prompt_context: str) -> AsyncChat:
    while True:
        try:
            chat = key_manager.client.aio.chats.create(model=os.getenv("GEMINI_MODEL"))
            await chat.send_message(prompt_context)
            print(f"Chat {chat} initialized.")
            return chat
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                next_key = key_manager.get_next_key()
                if next_key is None:
                    # Reset the key manager for future requests
                    key_manager.reset()
                    raise ValueError(
                        "OUT_OF_KEYS: All available API keys have been exhausted"
                    )
                # Continue the loop with the new key
                continue
            # If it's not a RESOURCE_EXHAUSTED error, re-raise it
            raise


async def send_chat_message(chat: AsyncChat, prompt: str) -> str:
    while True:
        try:
            response = await chat.send_message(prompt)
            return response.text
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                next_key = key_manager.get_next_key()
                if next_key is None:
                    # Reset the key manager for future requests
                    key_manager.reset()
                    raise ValueError(
                        "OUT_OF_KEYS: All available API keys have been exhausted"
                    )
                # Continue the loop with the new key
                continue
            # If it's not a RESOURCE_EXHAUSTED error, re-raise it
            raise


async def generate_response(prompt: str) -> str:
    """
    Generate a response from the LLM.

    Args:
        prompt: The prompt to generate a response from.

    Returns:
        The response from the LLM (Gemini for now).

    Raises:
        ValueError: If all API keys have been exhausted.
    """
    while True:
        try:
            response = await key_manager.client.aio.models.generate_content(
                model=os.getenv("GEMINI_MODEL"), contents=prompt
            )

            return response.text
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                next_key = key_manager.get_next_key()
                if next_key is None:
                    # Reset the key manager for future requests
                    key_manager.reset()
                    raise ValueError(
                        "OUT_OF_KEYS: All available API keys have been exhausted"
                    )
                # Continue the loop with the new key
                continue
            # If it's not a RESOURCE_EXHAUSTED error, re-raise it
            raise


if __name__ == "__main__":
    import asyncio

    res = asyncio.run(generate_response("What is the capital of France?"))
    print(res)
