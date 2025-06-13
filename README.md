<h1 align="center">TalkToGitHub</h1>

[![Image](./frontend/public/header.png "TalkToGitHub main page")](https://talktogithub.com)

<!-- <a href="https://www.producthunt.com/posts/talktogithub?embed=true&utm_source=badge-featured&utm_medium=badge&utm_souce=badge-talktogithub" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=957930&theme=light&t=1745741597937" alt="TalkToGitHub - Turn&#0032;GitHub&#0032;repositories&#0032;into&#0032;conversations | Product Hunt" style="width: 200px; height: 45px;" width="200" height="45" /></a> -->

[![Discord](https://dcbadge.limes.pink/api/server/https://discord.com/invite/JKtHeSn4S7)](https://discord.com/invite/JKtHeSn4S7)

Chat with any GitHub repository within seconds.

You can also add the `talkto` prefix to any GitHub repository URL.

<h2>Watch TTG in Action</h2>

https://github.com/user-attachments/assets/9b1b9dd3-546f-41ca-817f-531314452073

## Roadmap
- [x] Repo and query caching
- [ ] Support for individual files and folders
- [ ] Support for local files
- [ ] Support for larger repositories (>1M tokens)
- [ ] Streaming responses
- [ ] Save queries to user account
- [ ] BYOK support

## How to run locally

Prerequisites:
- [Python 3.13+](https://www.python.org/downloads/release/python-3130/)
- [uv](https://docs.astral.sh/uv/)
- [node 23.6.0+](https://nodejs.org/en/download)

1. Clone the forked repository:

  ```bash
  git clone https://github.com/HarishChandran3304/TTG.git
  cd TTG
  ```

2. Set up the backend:

  ```bash
  uv sync 
  source ./.venv/bin/activate
  ```

3. Add the following to your `.env` file (choose either Gemini or OpenAI):
  ```bash
  # Environment (development or production)
  ENV=development
  
  # Choose which LLM provider to use (gemini or openai)
  LLM_PROVIDER=gemini  # or openai
  
  # Gemini Configuration (if using Gemini)
  GEMINI_API_KEY=<your-gemini-api-key>
  GEMINI_MODEL=gemini-2.0-flash
  
  # OpenAI Configuration (if using OpenAI)
  # OPENAI_API_KEY=<your-openai-api-key>
  # OPENAI_MODEL=gpt-4o-mini  # or gpt-4o, gpt-3.5-turbo, etc.
  # OPENAI_MAX_TOKENS=4000
  # OPENAI_TEMPERATURE=0.1
  
  # Fallback keys (for Gemini only)
  FALLBACK_COUNT=0
  ```

4. Run the server:
  ```bash
  fastapi dev src/main.py
  ```

5. This should start the server at localhost:8000.

6. Set up the frontend:
  ```bash
  cd frontend
  npm install
  npm run dev
  ```

7. This should start the frontend at localhost:5173.

## Contributing
Check out the [contributing guide](./CONTRIBUTING.md) to learn how to contribute to TalkToGitHub.

## Tech Stack
### Frontend
- [Vite](https://vitejs.dev/)
- [React](https://reactjs.org/)
- [TailwindCSS](https://tailwindcss.com/)
- [Shadcn UI](https://ui.shadcn.com/)

### Backend
- [Python](https://www.python.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Gemini](https://gemini.google.com/)
- [OpenAI](https://openai.com/) (alternative LLM provider)

## Support

If you need help, please join our [Discord](https://discord.com/invite/JKtHeSn4S7) server. We will be happy to help!

To request a feature or report an issue, please [create an issue](https://github.com/HarishChandran3304/TTG/issues/new) on GitHub.

## ⭐ Star History

<a href="https://star-history.com/#HarishChandran3304/TTG&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=HarishChandran3304/TTG&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=HarishChandran3304/TTG&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=HarishChandran3304/TTG&type=Date" />
  </picture>
</a>

[![Image](./frontend/public/footer.png)](https://x.com/harishchan3304)
