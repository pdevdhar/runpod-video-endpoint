import runpod
import base64
from PIL import Image
import io
import imageio

# -----------------------------------
# MODEL PLACEHOLDER (LOAD ONCE)
# -----------------------------------
MODEL = "dummy_model_loaded"


# -----------------------------------
# DECODE BASE64 IMAGE
# -----------------------------------
def decode_image(image_base64):

    # remove data:image/jpeg;base64,
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    img_bytes = base64.b64decode(image_base64)

    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    return image


# -----------------------------------
# CREATE DUMMY VIDEO
# -----------------------------------
def generate_dummy_video(image):

    frames = []

    width, height = image.size

    for i in range(20):

        # move image slightly each frame
        shifted = Image.new("RGB", (width, height))

        offset = i * 5

        shifted.paste(image, (offset, 0))

        frames.append(shifted)

    output_path = "/tmp/output.mp4"

    imageio.mimsave(
        output_path,
        frames,
        fps=10
    )

    with open(output_path, "rb") as f:
        video_bytes = f.read()

    video_base64 = base64.b64encode(video_bytes).decode("utf-8")

    return video_base64

# -----------------------------------
# RUNPOD HANDLER
# -----------------------------------
def handler(job):

    job_input = job["input"]

    prompt = job_input.get("prompt", "")

    image_base64 = job_input.get("image_base64")

    image = decode_image(image_base64)

    video_base64 = generate_dummy_video(image)

    return {
        "received": True,
        "prompt": prompt,
        "image_size": image.size,
        "video": video_base64
    }


# -----------------------------------
# START WORKER
# -----------------------------------
runpod.serverless.start({
    "handler": handler
})
