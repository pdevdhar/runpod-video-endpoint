import runpod
import base64
from io import BytesIO
from PIL import Image
import imageio
import uuid

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
# Decode incoming image
# -----------------------------

def decode_image(image_base64):

    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    image_bytes = base64.b64decode(image_base64)

    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    return image


# -----------------------------
# Preserve aspect ratio
# inside widescreen frame
# -----------------------------

def prepare_image_for_svd(image):

    target_width = 1024
    target_height = 576

    # preserve aspect ratio
    image.thumbnail((target_width, target_height))

    background = Image.new(
        "RGB",
        (target_width, target_height),
        (0, 0, 0)
    )

    x = (target_width - image.width) // 2
    y = (target_height - image.height) // 2

    background.paste(image, (x, y))

    return background


# -----------------------------
# Main handler
# -----------------------------

def handler(job):

    input_data = job.get("input", {})

    prompt = input_data.get("prompt", "")

    image_base64 = input_data.get("image_base64")

    use_svd = input_data.get("use_svd", False)

    # decimal duration support
    duration = float(input_data.get("duration", 2.0))

    # keep quality stable
    num_frames = 14

    # derive fps from desired duration
    fps = max(2, min(10, round(num_frames / duration)))

    if image_base64 is None:
        return {
            "status": "error",
            "message": "No image received"
        }

    # decode image
    image = decode_image(image_base64)

    # preserve aspect ratio
    image = prepare_image_for_svd(image)

    # fallback mode
    if not use_svd:
        return {
            "status": "legacy_mode",
            "prompt": prompt,
            "image_size": image.size
        }

    # load pipeline lazily
    pipe = load_svd()

    # generate frames
    result = pipe(
        image,
        num_frames=num_frames,
        num_inference_steps=30,
        decode_chunk_size=8,
        motion_bucket_id=60
    )

    frames = result.frames[0]

    # -----------------------------
    # Save MP4
    # -----------------------------

    video_id = str(uuid.uuid4())

    video_path = f"/tmp/{video_id}.mp4"

    imageio.mimsave(
        video_path,
        frames,
        fps=fps
    )

    # read mp4 bytes
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    # encode base64 video
    video_base64 = base64.b64encode(video_bytes).decode("utf-8")

    return {
        "status": "success",
        "mode": "svd",
        "prompt": prompt,
        "image_size": image.size,
        "duration": duration,
        "fps": fps,
        "num_frames": len(frames),
        "video_base64": video_base64
    }


# -----------------------------
# Runpod entrypoint
# -----------------------------

runpod.serverless.start({
    "handler": handler
})
