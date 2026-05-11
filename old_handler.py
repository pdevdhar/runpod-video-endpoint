import runpod
import base64
import os
from PIL import Image
from io import BytesIO


def decode_image(image_base64: str):
    # remove prefix if present
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    image_bytes = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return image


def run_inference(image, prompt, steps=10):
    """
    PLACEHOLDER: replace with WAN / Seedance pipeline later
    """
    output_path = "/tmp/output.mp4"

    # fake placeholder for now
    with open(output_path, "wb") as f:
        f.write(b"FAKE_VIDEO")

    return output_path


def handler(event):
    input_data = event["input"]

    prompt = input_data.get("prompt", "")
    image_base64 = input_data.get("image_base64")

    image = decode_image(image_base64)

    output_path = run_inference(image, prompt)

    # encode result (temporary simple version)
    with open(output_path, "rb") as f:
        video_bytes = f.read()

    video_base64 = base64.b64encode(video_bytes).decode("utf-8")

    return {
        "video": f"data:video/mp4;base64,{video_base64}"
    }


runpod.serverless.start({
    "handler": handler
})
