import runpod
import base64
from PIL import Image
import io

def decode_image(image_base64):
    if not image_base64:
        return None

    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    img_bytes = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(img_bytes))


def handler(job):
    image_b64 = job["input"].get("image_base64")

    image = decode_image(image_b64)

    return {
        "received": True,
        "image_size": [768, 768],
        "video": "NOT_YET_IMPLEMENTED"
    }


runpod.serverless.start({"handler": handler})
