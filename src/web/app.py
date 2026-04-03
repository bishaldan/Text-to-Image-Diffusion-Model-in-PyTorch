from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.inference import InferenceService, image_to_base64


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

app = FastAPI(title="Tiny Diffusion Demo")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

service = InferenceService()


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=180)
    checkpoint_path: str
    guidance_scale: float | None = Field(default=None, ge=1.0, le=10.0)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/checkpoints")
async def checkpoints() -> dict[str, object]:
    checkpoint_summaries = service.list_checkpoints()
    if not checkpoint_summaries:
        return {"default_checkpoint": None, "checkpoints": []}

    return {
        "default_checkpoint": checkpoint_summaries[0].checkpoint_path,
        "checkpoints": [
            {
                "run_id": checkpoint.run_id,
                "checkpoint_path": checkpoint.checkpoint_path,
                "config_path": checkpoint.config_path,
                "training_prompts": checkpoint.training_prompts,
                "guidance_scale": checkpoint.guidance_scale,
                "image_size": checkpoint.image_size,
                "timesteps": checkpoint.timesteps,
            }
            for checkpoint in checkpoint_summaries
        ],
    }


@app.post("/api/generate")
async def generate(request: GenerateRequest) -> dict[str, object]:
    checkpoint = Path(request.checkpoint_path)
    if not checkpoint.exists():
        raise HTTPException(status_code=404, detail="Checkpoint not found.")

    image, meta = service.generate(
        checkpoint_path=checkpoint,
        prompt=request.prompt.strip(),
        guidance_scale=request.guidance_scale,
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path("outputs") / "web"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_path = output_dir / f"web_{timestamp}.png"
    image.save(saved_path)

    return {
        "prompt": request.prompt.strip(),
        "saved_path": str(saved_path),
        "image_base64": image_to_base64(image),
        "meta": meta,
    }
