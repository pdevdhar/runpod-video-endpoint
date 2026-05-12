import runpod
import base64
from io import BytesIO
from PIL import Image
import imageio
import uuid
import numpy as np

# -----------------------------
# Model
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
        ).to("cuda")

    return svd_pipe


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
# SIMPLE FRAME INTERPOLATION
# (lightweight blending)
# -----------------------------

def interpolate_frames(frames, factor=2):

    if factor <= 1:
        return frames

    new_frames = []

    for i in range(len(frames) - 1):

        f1 = np.array(frames[i])
        f2 = np.array(frames[i + 1])

        new_frames.append(frames[i])

        # generate intermediate frames
        for j in range(1, factor):
            alpha = j / factor
            blended = (1 - alpha) * f1 + alpha * f2
            new_frames.append(Image.fromarray(blended.astype(np.uint8)))

    new_frames.append(frames[-1])

    return new_frames


# -----------------------------
# FAST / CINEMATIC CORE
# -----------------------------

def generate(image, duration, quality):

    pipe = load_svd()

    # base frames (stable SVD range)
    num_frames = min(max(int(14 + duration * 2.5), 14), 28)

    motion_bucket_id = 75 if quality != "low" else 90

    result = pipe(
        image,
        num_frames=num_frames,
        num_inference_steps=30,
        motion_bucket_id=motion_bucket_id
    )

    frames = result.frames[0]

    # -------------------------
    # INTERPOLATION (NEW)
    # -------------------------

    # 2x interpolation for 5s+ videos
    if duration >= 4.0:
        frames = interpolate_frames(frames, factor=2)

    fps = max(5, min(12, round(len(frames) / duration)))

    return frames, fps


# -----------------------------
# MAIN HANDLER
# -----------------------------

def handler(job):

    inp = job.get("input", {})

    image_b64 = inp.get("image_base64")
    duration = float(inp.get("duration", 3.0))
    quality = inp.get("quality", "medium")

    if not image_b64:
        return {"status": "error", "message": "missing image"}

    image = pad_image(decode_image(image_b64))

    frames, fps = generate(image, duration, quality)

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
        "duration": duration,
        "fps": fps,
        "frames": len(frames),
        "interpolation": True,
        "video_base64": video_b64
    }


runpod.serverless.start({"handler": handler})
