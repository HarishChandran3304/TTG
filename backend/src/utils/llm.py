from abc import ABC, abstractmethod
from dotenv import load_dotenv
import os
from typing import Optional, Dict, List, Union

# Import providers
from google import genai  # type: ignore
from openai import AsyncOpenAI, OpenAIError

load_dotenv()


class LLMProviderEnum:
    """Enum-like class for LLM providers"""
    GEMINI = "gemini"
    OPENAI = "openai"


class BaseLLMProvider(ABC):
    """Base abstract class for all LLM providers"""
    
    @abstractmethod
    async def generate_response(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the provider's state"""
        pass


class GeminiProvider(BaseLLMProvider):
    """Gemini LLM provider implementation"""
    
    def __init__(self):
        self.main_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        
        # Fallback keys setup
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
        """Get the next API key to use."""
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
    
    def reset(self) -> None:
        """Reset the provider to its initial state."""
        self.current_key_index = 0
        self.tried_keys.clear()
        self.client = genai.Client(api_key=self.main_key)
    
    async def generate_response(self, prompt: str) -> str:
        """Generate a response using the Gemini model."""
        while True:
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model, contents=prompt
                )
                return response.text
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e):
                    next_key = self.get_next_key()
                    if next_key is None:
                        # Reset for future requests
                        self.reset()
                        raise ValueError(
                            "OUT_OF_KEYS: All available Gemini API keys have been exhausted"
                        )
                    # Continue the loop with the new key
                    continue
                # If it's not a RESOURCE_EXHAUSTED error, re-raise it
                raise


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "4000"))
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    def reset(self) -> None:
        """Reset the provider to its initial state."""
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    async def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response using the OpenAI model."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content or ""
        except OpenAIError as e:
            # Handle OpenAI API errors
            if "rate limit" in str(e).lower():
                self.reset()
                raise ValueError(
                    "OUT_OF_KEYS: OpenAI API rate limit reached. Please try again later."
                )
            raise ValueError(f"OpenAI API error: {str(e)}")


class LLMProviderFactory:
    """Factory class to create appropriate LLM provider instances"""
    
    _instance = None
    _provider_instance = None
    
    @classmethod
    def get_instance(cls) -> 'LLMProviderFactory':
        """Get the singleton instance of the factory"""
        if cls._instance is None:
            cls._instance = LLMProviderFactory()
        return cls._instance
    
    def __init__(self):
        self.provider_type = os.getenv("LLM_PROVIDER", LLMProviderEnum.GEMINI)

    def get_provider(self) -> BaseLLMProvider:
        """Get the appropriate LLM provider based on environment configuration"""
        # Return cached instance if it exists and provider type hasn't changed
        provider_type = os.getenv("LLM_PROVIDER", LLMProviderEnum.GEMINI)
        
        # If provider type changed or we don't have an instance yet, create a new one
        if self._provider_instance is None or self.provider_type != provider_type:
            self.provider_type = provider_type
            
            if provider_type == LLMProviderEnum.OPENAI:
                self._provider_instance = OpenAIProvider()
            else:  # Default to Gemini
                self._provider_instance = GeminiProvider()
                
        return self._provider_instance


# Create a global provider factory
llm_factory = LLMProviderFactory.get_instance()


async def generate_response(prompt: Union[str, List[Dict[str, str]]]) -> str:
    """
    Generate a response from the selected LLM provider.

    Args:
        prompt: The prompt to generate a response from.
            For Gemini: A string prompt
            For OpenAI: A list of message objects with roles and content

    Returns:
        The response from the LLM provider.

    Raises:
        ValueError: If all API keys have been exhausted or another error occurs.
    """
    try:
        provider = llm_factory.get_provider()
        return await provider.generate_response(prompt)
    except ValueError as e:
        # Re-raise any value errors from the providers
        raise
    except Exception as e:
        # Convert any other exceptions to ValueError for consistent error handling
        raise ValueError(f"LLM error: {str(e)}")


if __name__ == "__main__":
    import asyncio
    
    # Simple test
    async def test():
        # Test string prompt for Gemini
        response = await generate_response("What is the capital of France?")
        print(f"Response: {response}")
    
    asyncio.run(test())
