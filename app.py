from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from env import TaxiEnv, Action, TASKS

# -------------------------------
# App Initialization
# -------------------------------
app = FastAPI(title="Taxi Dispatch OpenEnv", version="1.0.0")

ENV = TaxiEnv()


# -------------------------------
# Request Models
# -------------------------------
class ResetRequest(BaseModel):
    task_id: str = Field(default="ride_matching")
    seed: int | None = Field(default=42)


class StepRequest(BaseModel):
    action: Action


# -------------------------------
# Routes
# -------------------------------

@app.get("/")
def root() -> dict[str, Any]:
    return {
        "status": "ok",
        "env_id": TaxiEnv.ENV_ID,
        "tasks": list(TASKS.keys()),
    }


# ✅ Supports BOTH GET and POST (important)
@app.get("/reset")
@app.post("/reset")
def reset(req: ResetRequest | None = None) -> dict[str, Any]:
    try:
        if req is None:
            obs = ENV.reset(task_id="ride_matching", seed=42)
        else:
            obs = ENV.reset(task_id=req.task_id, seed=req.seed)
        return obs.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step")
def step(req: StepRequest) -> dict[str, Any]:
    try:
        result = ENV.step(req.action)
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state")
def state() -> dict[str, Any]:
    return ENV.state()


@app.get("/score")
def score() -> dict[str, Any]:
    return {
        "score": ENV.score(),
        "metrics": ENV.episode_metrics,
    }


@app.get("/tasks")
def tasks() -> dict[str, Any]:
    return {k: v.model_dump() for k, v in TASKS.items()}


@app.get("/openenv.yaml")
def openenv_yaml() -> str:
    path = Path(__file__).with_name("openenv.yaml")
    return path.read_text(encoding="utf-8")


# -------------------------------
# Entry Point 
# -------------------------------
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()