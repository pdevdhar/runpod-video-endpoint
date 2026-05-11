from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()


class GenerateRequest(BaseModel):
    prompt: str = ""
    image_base64: str = ""


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/generate")
def generate(req: GenerateRequest):
    return {
        "status": "success",
        "message": "endpoint working",
        "prompt": req.prompt
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
