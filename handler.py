import runpod

import os
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"
os.environ["TORCH_HOME"] = "/workspace/torch_cache"

pipe = None  # IMPORTANT: no loading at startup


def load_model():
    global pipe
    if pipe is None:
        from diffusers import StableVideoDiffusionPipeline
        import torch

        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        pipe.to("cuda")


def handler(job):
    load_model()

    input_data = job["input"]
    image = input_data["image"]  # we'll wire this next step

    # placeholder for now
    return {
        "status": "model_loaded",
        "message": "SVD initialized safely"
    }


runpod.serverless.start({"handler": handler})
