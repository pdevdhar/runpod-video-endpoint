import runpod
import base64
from io import BytesIO
from PIL import Image
import imageio
import uuid

# -----------------------------
# Lazy pipelines
# -----------------------------

svd_pipe = None
wan_pipe = None


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


def load_wan():
    """
    Placeholder loader for WAN-style model.
    Replace with actual model once available in your setup.
    """
    global wan_pipe
    if wan_pipe is None:
        # Example structure only (model name depends on your deployment)
        from diffusers import DiffusionPipeline

        wan_pipe = DiffusionPipeline.from_pretrained(
            "WAN_MODEL_PATH_OR_REPO"
        ).to("cuda")

    return wan_pipe


# -----------------------------
# Image decode + padding
# -----------------------------

def decode_image(image_base64):
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    return Image.open(
        BytesIO(base64.b64decode(image_base64))
    ).convert("RGB")


def prepare_image(image):
    image = image.copy()

    target = (1024, 576)
    image.thumbnail(target)

    canvas = Image.new("RGB", target, (0, 0, 0))
    x = (target[0] - image.width) // 2
    y = (target[1] - image.height) // 2

    canvas.paste(image, (x, y))
    return canvas


# -----------------------------
# FAST MODE (SVD)
# -----------------------------

def run_svd(image, duration):

    pipe = load_svd()

    num_frames = 14
    fps = 7

    result = pipe(
        image,
        num_frames=num_frames,
        num_inference_steps=30,
        motion_bucket_id=70
    )

    return result.frames[0], fps


# -----------------------------
# CINEMATIC MODE (WAN)
# -----------------------------

def run_wan(image, duration):

    pipe = load_wan()

    # WAN-style models typically support longer sequences better
    fps = 6
    num_frames = min(int(duration * fps), 90)  # allow longer clips safely

    result = pipe(
        image,
        num_frames=num_frames
    )

    return result.frames[0], fps


# -----------------------------
# Main handler
# -----------------------------

def handler(job):

    inp = job.get("input", {})

    prompt = inp.get("prompt", "")
    image_base64 = inp.get("image_base64")
    duration = float(inp.get("duration", 2.0))
    mode = inp.get("mode", "fast")

    if not image_base64:
        return {"status": "error", "message": "no image"}

    image = prepare_image(decode_image(image_base64))

    # -------------------------
    # ROUTER
    # -------------------------

    if mode == "cinematic":
        frames, fps = run_wan(image, duration)
    else:
        frames, fps = run_svd(image, duration)

    # -------------------------
    # Encode video
    # -------------------------

    vid = str(uuid.uuid4())
    path = f"/tmp/{vid}.mp4"

    imageio.mimsave(path, frames, fps=fps)

    with open(path, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "status": "success",
        "mode": mode,
        "prompt": prompt,
        "duration": duration,
        "fps": fps,
        "num_frames": len(frames),
        "video_base64": video_base64
    }


runpod.serverless.start({"handler": handler})
