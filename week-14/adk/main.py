"""Week 14 — wrap the campus_helper_agent in a real FastAPI service with ADK's get_fast_api_app.

Every sub-folder of this directory that holds an agent (here campus_helper_agent/) becomes an app of the
service. Run from this directory with: python main.py  (serves on http://localhost:8000)

1. Create a session:  POST /apps/campus_helper_agent/users/u1/sessions/s1      body: {}
2. Ask, all at once:  POST /run      body: {"app_name": "campus_helper_agent", "user_id": "u1",
                                            "session_id": "s1", "new_message": {"role": "user",
                                            "parts": [{"text": "What does week 9 cover?"}]}}
3. Ask, streaming:    POST /run_sse  same body plus "streaming": true  (Server-Sent Events, use curl -N)
"""

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app

AGENTS_DIR = Path(__file__).resolve().parent
load_dotenv(AGENTS_DIR / ".env")

# Comma-separated list of the frontends allowed to call the API. Never "*" in a real deployment.
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()]

app: FastAPI = get_fast_api_app(
    agents_dir=str(AGENTS_DIR),
    session_service_uri=f"sqlite+aiosqlite:///{(AGENTS_DIR / 'sessions.db').as_posix()}",  # gitignored (*.db)
    allow_origins=ALLOWED_ORIGINS,
    web=False,  # no dev chat UI — this is the production entry point
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # no --reload: this is not a dev server
