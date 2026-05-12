import runpod
import base64
from PIL import Image
import io
import torch
import imageio

from diffusers import StableVideoDiffusionPipeline


# -------------------------------------------------
# LOAD MODEL ONCE
# -------------------------------------------------

print("Loading Stable Video Diffusion model...")

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16
)

pipe.to("cuda")

print("Model loaded.")


# -------------------------------------------------
# DECODE IMAGE
# -------------------------------------------------

def decode_image(image_base64):

    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    img_bytes = base64.b64decode(image_base64)

    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    return image


# -------------------------------------------------
# GENERATE VIDEO
# -------------------------------------------------

def generate_video(image):

    image = image.resize((1024, 576))

    frames = pipe(
        image,
        decode_chunk_size=8
    ).frames[0]

    output_path = "/tmp/output.mp4"

    imageio.mimsave(
        output_path,
        frames,
        fps=7
    )

    with open(output_path, "rb") as f:
        video_bytes = f.read()

    video_base64 = base64.b64encode(video_bytes).decode("utf-8")

    return video_base64


# -------------------------------------------------
# RUNPOD HANDLER
# -------------------------------------------------

def handler(job):

    job_input = job["input"]

    prompt = job_input.get("prompt", "")

    image_base64 = job_input.get("image_base64")

    image = decode_image(image_base64)

    video_base64 = generate_video(image)

    return {
        "received": True,
        "prompt": prompt,
        "image_size": image.size,
        "video": video_base64
    }


# -------------------------------------------------
# START WORKER
# -------------------------------------------------

runpod.serverless.start({
    "handler": handler
})
