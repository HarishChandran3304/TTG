from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.utils.cache import load_repo_cache, save_repo_cache
from src.utils.ingest import ingest_repo  # type: ignore
from src.utils.llm import generate_response  # type: ignore
from src.utils.prompt import generate_prompt  # type: ignore

from typing import Any

import os
from dotenv import load_dotenv
import logging


load_dotenv()

IS_PROD = os.getenv("ENV") == "production"


app = FastAPI(
    title="Talk to GitHub",
    description="A simple chat app to interact with GitHub repositories",
    version="0.1.0",
    contact={"name": "Harish", "email": "harish3304.work@gmail.com"},
    license_info={"name": "MIT License"},
    openapi_url=None if IS_PROD else "/openapi.json",
    docs_url=None if IS_PROD else "/docs",
    redoc_url=None if IS_PROD else "/redoc",
)

if not IS_PROD:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: dict[str, dict[str, Any]] = {}
        """
		{
			"client_id": {
				"websocket": websocket object,
				"history": [(prompt1, response1), (prompt2, response2), ...],
				"owner": owner,
				"repo": repo,
				"summary": summary,
				"tree": tree,
				"content": content
			}
		}
		"""

    async def connect(
        self, websocket: WebSocket, client_id: str, owner: str, repo: str
    ) -> None:
        await websocket.accept()

        if client_id in self.active_connections:
            await self.active_connections[client_id]["websocket"].close()
            del self.active_connections[client_id]

        repo_url = f"https://github.com/{owner}/{repo}"
        logging.info(f"Processing repo: {repo_url}...")
        # Try to load from cache first
        cached = load_repo_cache(owner, repo)
        if cached:
            summary, tree, content = cached["summary"], cached["tree"], cached["content"]
            logging.info(f"Loaded repo from cache: {repo_url}")
        else:
            try:
                summary, tree, content = await ingest_repo(repo_url)
                logging.info(f"Repo processed - {repo_url}!")
                logging.info(f"Repository Summary:\n{summary}")
                save_repo_cache(owner, repo, summary, tree, content)
            except ValueError as e:
                error_msg = str(e)
                if error_msg == "error:repo_too_large":
                    await websocket.send_text("error:repo_too_large")
                elif error_msg == "error:repo_not_found":
                    await websocket.send_text("error:repo_not_found")
                elif error_msg == "error:repo_private":
                    await websocket.send_text("error:repo_private")
                else:
                    raise
                await websocket.close()
                return

        self.active_connections[client_id] = {
            "websocket": websocket,
            "history": [],
            "owner": owner,
            "repo": repo,
            "summary": summary,
            "tree": tree,
            "content": content,
        }

        # Send confirmation that repo is processed
        await websocket.send_text("repo_processed")

    async def disconnect(self, client_id: str) -> None:
        if client_id in self.active_connections:
            await self.active_connections[client_id]["websocket"].close()
            del self.active_connections[client_id]

    async def handle_message(self, client_id: str, text: str) -> None:
        if text == "ping":
            await self.active_connections[client_id]["websocket"].send_text("pong")
        else:
            query = text
            owner = self.active_connections[client_id]["owner"]  # noqa: F841
            repo = self.active_connections[client_id]["repo"]  # noqa: F841
            summary = self.active_connections[client_id]["summary"]  # noqa: F841
            tree = self.active_connections[client_id]["tree"]
            content = self.active_connections[client_id]["content"]
            history = self.active_connections[client_id]["history"]

            logging.info(f"Generating prompt for query: {query}...")
            prompt_data = await generate_prompt(query, history, tree, content)
            logging.info(f"Prompt generated successfully")
            try:
                response = await generate_response(prompt_data)
                logging.info(f"Response generated: {response}")
                await self.active_connections[client_id]["websocket"].send_text(
                    response
                )
                self.active_connections[client_id]["history"].append((query, response))
            except ValueError as e:
                if "OUT_OF_KEYS" in str(e):
                    error_msg = "All API keys have been exhausted. Please try again in a few minutes."
                    await self.active_connections[client_id]["websocket"].send_text(
                        error_msg
                    )
                else:
                    logging.error(f"Error generating response: {str(e)}")
                    error_msg = f"An error occurred: {str(e)}"
                    await self.active_connections[client_id]["websocket"].send_text(
                        error_msg
                    )

    async def get_history(self, client_id: str) -> list[tuple[str, str]]:
        if client_id in self.active_connections:
            return self.active_connections[client_id]["history"]
        return []


manager = ConnectionManager()


@app.websocket("/{owner}/{repo}/{client_id}")
async def websocket_endpoint(
    websocket: WebSocket, owner: str, repo: str, client_id: str
) -> None:
    try:
        await manager.connect(websocket, client_id, owner, repo)
        print(f"Client {client_id} connected")

        while True:
            try:
                text = await websocket.receive_text()
                if text == "ping":
                    print(f"Received ping from {client_id}")
                    continue
                await manager.handle_message(client_id, text)
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"Error in websocket_endpoint: {e}")
                break

    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        await manager.disconnect(client_id)
        print(f"Client {client_id} disconnected")

@app.get("/healthcheck")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}

async def main():
    summary, tree, content = await ingest_repo("https://github.com/EnhancedJax/Bagels")
    prompt = await generate_prompt(
        "How does this codebase work? What is it built using?", [], tree, content
    )
    response = await generate_response(prompt)
    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
