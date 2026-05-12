import runpod
import os
import base64
from PIL import Image
import io
import imageio

# -------------------------
# 1. LOAD MODEL ONCE
# -------------------------
MODEL = None

def load_model():
    global MODEL
    print("Loading WAN2.2 model... (placeholder)")
    
    # Later we replace this with actual WAN2.2 / Seedance pipeline:
    # from diffusers import DiffusionPipeline
    # MODEL = DiffusionPipeline.from_pretrained(...)
    
    MODEL = "dummy_model"
    return MODEL


load_model()


# -------------------------
# 2. IMAGE DECODER
# -------------------------
def decode_image(image_base64):
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    img_bytes = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


# -------------------------
# 3. DUMMY VIDEO GENERATOR (TEMP)
# -------------------------
def generate_dummy_video(image):
    frames = []

    for i in range(10):
        frames.append(image)

    output_path = "/tmp/output.mp4"
    imageio.mimsave(output_path, frames, fps=5)

    with open(output_path, "rb") as f:
        video_bytes = f.read()

    return base64.b64encode(video_bytes).decode("utf-8")


# -------------------------
# 4. HANDLER
# -------------------------
def handler(job):
    inp = job["input"]

    image = decode_image(inp["image_base64"])

    video_b64 = generate_dummy_video(image)

    return {
        "received": True,
        "image_size": image.size,
        "video": video_b64
    }


runpod.serverless.start({"handler": handler})
