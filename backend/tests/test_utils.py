import pytest
import os
from unittest.mock import patch, AsyncMock
from src.utils import ingest, llm, prompt


@pytest.mark.asyncio
async def test_check_repo_exists_success():
    mock_response = AsyncMock()
    mock_response.status = 200

    with patch("aiohttp.ClientSession.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response

        result = await ingest.check_repo_exists(
            "https://github.com/HarishChandran3304/FCA"
        )
        assert result is True


@pytest.mark.asyncio
async def test_check_repo_exists_failure():
    mock_response = AsyncMock()
    mock_response.status = 404

    with patch("aiohttp.ClientSession.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response

        result = await ingest.check_repo_exists("https://github.com/owner/repo")
        assert result is False


@pytest.mark.asyncio
async def test_check_repo_exists_invalid_url():
    with patch("aiohttp.ClientSession.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = Exception("Invalid URL")
        result = await ingest.check_repo_exists("not_a_url")
        assert result is False or result is None


@pytest.mark.asyncio
async def test_ingest_repo_not_found():
    with patch("src.utils.ingest.check_repo_exists", AsyncMock(return_value=False)):
        with pytest.raises(ValueError) as exc:
            await ingest.ingest_repo("https://github.com/owner/repo")
        assert str(exc.value) == "error:repo_not_found"


@pytest.mark.asyncio
async def test_ingest_repo_too_large():
    async def fake_ingest_async(repo_url, exclude_patterns=None):
        return ("Estimated tokens: 1M", "tree", "content")

    with (
        patch("src.utils.ingest.check_repo_exists", AsyncMock(return_value=True)),
        patch("gitingest.ingest_async", new=fake_ingest_async),
    ):
        with pytest.raises(ValueError) as exc:
            await ingest.ingest_repo("https://github.com/owner/repo")
        assert str(exc.value) == "error:repo_not_found"


@pytest.mark.asyncio
async def test_ingest_repo_network_error():
    with patch(
        "src.utils.ingest.check_repo_exists",
        AsyncMock(side_effect=Exception("Network error")),
    ):
        with pytest.raises(Exception) as exc:
            await ingest.ingest_repo("https://github.com/owner/repo")
        assert "Network error" in str(exc.value)


@pytest.mark.asyncio
async def test_generate_prompt_gemini_basic():
    # Pour ce test, on force l'utilisation du format Gemini (chaîne)
    with patch.object(os, "getenv", return_value="gemini"):
        query = "What does this repo do?"
        history = [("User", "Hello"), ("Bot", "Hi!")]
        tree = "src/\n  main.py"
        content = "def foo(): pass"
        prompt_result = await prompt.generate_prompt(query, history, tree, content)
        # Verify it's a string for Gemini
        assert isinstance(prompt_result, str)
        assert "What does this repo do?" in prompt_result
        assert "src/" in prompt_result
        assert "def foo()" in prompt_result


@pytest.mark.asyncio
async def test_generate_prompt_openai_basic():
    # For this test, we force using the OpenAI format (message list)
    with patch.object(os, "getenv", return_value="openai"):
        query = "What does this repo do?"
        history = [("User", "Hello"), ("Bot", "Hi!")]
        tree = "src/\n  main.py"
        content = "def foo(): pass"
        prompt_result = await prompt.generate_prompt(query, history, tree, content)
        # Verify it's a list of messages for OpenAI
        assert isinstance(prompt_result, list)
        assert len(prompt_result) > 0
        # Verify the last message contains the query
        assert prompt_result[-1]["role"] == "user"
        assert prompt_result[-1]["content"] == query
        
        # Verify code information is in the system message
        system_message = prompt_result[0]["content"]
        assert "src/" in system_message
        assert "def foo()" in system_message


@pytest.mark.asyncio
async def test_generate_prompt_gemini_empty_content():
    # For this test, we force using the Gemini format (string)
    with patch.object(os, "getenv", return_value="gemini"):
        query = "Explain the repo."
        history = []
        tree = ""
        content = ""
        prompt_str = await prompt.generate_prompt(query, history, tree, content)
        assert isinstance(prompt_str, str)
        assert query in prompt_str
        assert "File Content:" in prompt_str


@pytest.mark.asyncio
async def test_generate_prompt_openai_empty_content():
    # For this test, we force using the OpenAI format (message list)
    with patch.object(os, "getenv", return_value="openai"):
        query = "Explain the repo."
        history = []
        tree = ""
        content = ""
        prompt_result = await prompt.generate_prompt(query, history, tree, content)
        # Verify it's a list of messages for OpenAI
        assert isinstance(prompt_result, list)
        assert len(prompt_result) > 0
        # Verify the last message contains the query
        assert prompt_result[-1]["role"] == "user"
        assert prompt_result[-1]["content"] == query
        
        # Verify code information is in the system message
        system_message = prompt_result[0]["content"]
        assert "File Content:" in system_message
        # We don't check for specific code since content is empty


@pytest.mark.asyncio
async def test_generate_response_gemini_success():
    # Mock for the Gemini provider
    class MockGeminiProvider:
        def __init__(self):
            self.client = None
            
        async def generate_response(self, prompt):
            return "gemini response"
            
        def reset(self):
            pass
    
    # Patch le factory pour qu'il retourne notre mock
    with patch.object(llm.llm_factory, "get_provider", return_value=MockGeminiProvider()):
        # Test avec un prompt string (format Gemini)
        resp = await llm.generate_response("prompt")
        assert resp == "gemini response"

@pytest.mark.asyncio
async def test_generate_response_openai_success():
    # Mock pour le provider OpenAI
    class MockOpenAIProvider:
        def __init__(self):
            self.client = None
            
        async def generate_response(self, messages):
            return "openai response"
            
        def reset(self):
            pass
    
    # Patch le factory pour qu'il retourne notre mock
    with patch.object(llm.llm_factory, "get_provider", return_value=MockOpenAIProvider()):
        # Test avec une liste de messages (format OpenAI)
        resp = await llm.generate_response([{"role": "user", "content": "prompt"}])
        assert resp == "openai response"


@pytest.mark.asyncio
async def test_generate_response_error_handling():
    # Mock for a provider that raises an exception
    class MockErrorProvider:
        def __init__(self):
            self.client = None
            
        async def generate_response(self, prompt):
            raise ValueError("OUT_OF_KEYS: Test error")
            
        def reset(self):
            pass
    
    # Patch the factory to return our mock
    with patch.object(llm.llm_factory, "get_provider", return_value=MockErrorProvider()):
        with pytest.raises(ValueError, match="OUT_OF_KEYS: Test error"):
            await llm.generate_response("prompt")


@pytest.mark.asyncio
async def test_generate_response_invalid_prompt():
    # Mock for a provider that raises an invalid prompt error
    class MockInvalidPromptProvider:
        def __init__(self):
            self.client = None
            
        async def generate_response(self, prompt):
            # Raise an error to simulate an invalid prompt
            raise ValueError("LLM error: INVALID_PROMPT")
            
        def reset(self):
            pass
    
    # Patch the factory to return our mock
    with patch.object(llm.llm_factory, "get_provider", return_value=MockInvalidPromptProvider()):
        with pytest.raises(ValueError, match="LLM error: INVALID_PROMPT"):
            await llm.generate_response("")
