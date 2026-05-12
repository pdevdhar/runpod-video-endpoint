import runpod
import base64
from io import BytesIO
from PIL import Image
import imageio
import uuid

# -----------------------------
# Pipelines (cached)
# -----------------------------

svd_pipe = None
wan_pipe = None


# -----------------------------
# SVD loader (FAST MODE)
# -----------------------------

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


# -----------------------------
# WAN loader (CINEMATIC MODE)
# -----------------------------
def load_wan():
    global wan_pipe

    if wan_pipe is None:
        """
        REAL PLUG-IN POINT:
        Replace with actual WAN-compatible model checkpoint when available.
        """

        from diffusers import DiffusionPipeline

        wan_pipe = DiffusionPipeline.from_pretrained(
            "PUT_WAN_MODEL_HERE"
        ).to("cuda")

    return wan_pipe


# -----------------------------
# Image utils
# -----------------------------

def decode_image(b64):
    if "," in b64:
        b64 = b64.split(",")[1]

    return Image.open(
        BytesIO(base64.b64decode(b64))
    ).convert("RGB")


def pad_image(img):
    target = (1024, 576)

    img = img.copy()
    img.thumbnail(target)

    canvas = Image.new("RGB", target, (0, 0, 0))

    x = (target[0] - img.width) // 2
    y = (target[1] - img.height) // 2

    canvas.paste(img, (x, y))
    return canvas


# -----------------------------
# FAST MODE (SVD)
# -----------------------------

def run_fast(image, duration):

    pipe = load_svd()

    # fixed high-quality regime
    num_frames = 14

    # FPS controls playback only
    fps = max(4, min(8, round(num_frames / duration)))

    result = pipe(
        image,
        num_frames=num_frames,
        num_inference_steps=30,
        motion_bucket_id=75
    )

    return result.frames[0], fps


# -----------------------------
# CINEMATIC MODE (WAN)
# -----------------------------

def run_cinematic(image, duration):

    pipe = load_wan()

    fps = 6
    num_frames = min(int(duration * fps), 90)

    result = pipe(
        image,
        num_frames=num_frames
    )

    return result.frames[0], fps


# -----------------------------
# MAIN HANDLER
# -----------------------------

def handler(job):

    inp = job.get("input", {})

    prompt = inp.get("prompt", "")
    mode = inp.get("mode", "fast")
    duration = float(inp.get("duration", 3.0))

    image_b64 = inp.get("image_base64")

    if not image_b64:
        return {
            "status": "error",
            "message": "missing image"
        }

    image = pad_image(decode_image(image_b64))

    # -------------------------
    # ROUTER
    # -------------------------

    if mode == "cinematic":
        frames, fps = run_cinematic(image, duration)
    else:
        frames, fps = run_fast(image, duration)

    # -------------------------
    # encode video
    # -------------------------

    vid = str(uuid.uuid4())
    path = f"/tmp/{vid}.mp4"

    imageio.mimsave(path, frames, fps=fps)

    with open(path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "status": "success",
        "mode": mode,
        "duration": duration,
        "fps": fps,
        "frames": len(frames),
        "video_base64": video_b64
    }


runpod.serverless.start({"handler": handler})
