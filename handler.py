import runpod
import base64
from io import BytesIO
from PIL import Image

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
        ).to("cuda")

    return svd_pipe


def decode_image(image_base64):
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    image_bytes = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def handler(job):
    input_data = job.get("input", {})

    prompt = input_data.get("prompt")
    image_base64 = input_data.get("image_base64")
    use_svd = input_data.get("use_svd", False)

    if not image_base64:
        return {"error": "No image received"}

    image = decode_image(image_base64)

    if not use_svd:
        return {
            "status": "legacy_mode",
            "prompt": prompt,
            "image_size": image.size
        }

    pipe = load_svd()

    result = pipe(image)

    return {
        "status": "success",
        "mode": "svd",
        "prompt": prompt,
        "image_size": image.size,
        "output": str(result)
    }


runpod.serverless.start({"handler": handler})
