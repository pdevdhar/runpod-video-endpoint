import runpod
import base64
from io import BytesIO
from PIL import Image
import imageio
import uuid
import os

# -----------------------------
# Lazy-loaded SVD pipeline
# -----------------------------

svd_pipe = None

def load_svd():
    global svd_pipe

    if svd_pipe is None:
        import torch
        from diffusers import StableVideoDiffusionPipeline

        svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            torch_dtype=torch.float16,
            variant="fp16"
        )

        svd_pipe.to("cuda")

    return svd_pipe


# -----------------------------
# Decode incoming base64 image
# -----------------------------

def decode_image(image_base64):

    # remove data:image/... prefix
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    image_bytes = base64.b64decode(image_base64)

    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    return image


# -----------------------------
# Main handler
# -----------------------------

def handler(job):

    input_data = job.get("input", {})

    prompt = input_data.get("prompt", "")
    image_base64 = input_data.get("image_base64")
    use_svd = input_data.get("use_svd", False)

    if image_base64 is None:
        return {
            "status": "error",
            "message": "No image received"
        }

    # decode image
    image = decode_image(image_base64)

    # legacy test path
    if not use_svd:
        return {
            "status": "legacy_mode",
            "prompt": prompt,
            "image_size": image.size
        }

    # -----------------------------
    # Load SVD lazily
    # -----------------------------

    pipe = load_svd()

    # generate frames
    result = pipe(image)

    frames = result.frames[0]

    # -----------------------------
    # Save MP4
    # -----------------------------

    video_id = str(uuid.uuid4())

    video_path = f"/tmp/{video_id}.mp4"

    imageio.mimsave(
        video_path,
        frames,
        fps=7
    )

    return {
        "status": "success",
        "mode": "svd",
        "prompt": prompt,
        "image_size": image.size,
        "num_frames": len(frames),
        "video_path": video_path
    }


# -----------------------------
# Runpod entrypoint
# -----------------------------

runpod.serverless.start({
    "handler": handler
})
