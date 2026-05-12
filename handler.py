import runpod
import base64
from io import BytesIO
from PIL import Image
import imageio
import uuid

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
# Motion presets (NEW)
# -----------------------------

def get_motion_config(motion, quality):

    base = {
        "static": 60,
        "zoom_in": 85,
        "zoom_out": 80,
        "pan_left": 75,
        "pan_right": 75,
        "handheld": 110
    }

    mb = base.get(motion, 75)

    # quality adjusts stability
    # higher quality = more stable, less chaotic motion
    if quality == "high":
        mb -= 10
    elif quality == "low":
        mb += 15

    return max(40, min(120, mb))


# -----------------------------
# FAST / CINEMATIC SHARED CORE
# -----------------------------

def generate_svd(image, duration, motion, quality):

    pipe = load_svd()

    motion_bucket_id = get_motion_config(motion, quality)

    # -------------------------
    # frame strategy (NEW)
    # -------------------------

    # smooth scaling instead of fixed 14 frames
    num_frames = min(max(int(12 + duration * 2.5), 14), 24)

    # fps derived from duration
    fps = max(4, min(8, round(num_frames / duration)))

    result = pipe(
        image,
        num_frames=num_frames,
        num_inference_steps=30 if quality == "high" else 25,
        motion_bucket_id=motion_bucket_id
    )

    return result.frames[0], fps, num_frames


# -----------------------------
# MAIN HANDLER
# -----------------------------

def handler(job):

    inp = job.get("input", {})

    image_b64 = inp.get("image_base64")
    mode = inp.get("mode", "fast")
    duration = float(inp.get("duration", 3.0))

    motion = inp.get("motion", "static")
    quality = inp.get("quality", "medium")

    if not image_b64:
        return {"status": "error", "message": "missing image"}

    image = pad_image(decode_image(image_b64))

    # -------------------------
    # SINGLE ENGINE (stable)
    # -------------------------

    frames, fps, num_frames = generate_svd(
        image,
        duration,
        motion,
        quality
    )

    # -------------------------
    # Encode video
    # -------------------------

    vid = str(uuid.uuid4())
    path = f"/tmp/{vid}.mp4"

    imageio.mimsave(path, frames, fps=fps)

    with open(path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "status": "success",
        "mode": mode,
        "motion": motion,
        "quality": quality,
        "duration": duration,
        "fps": fps,
        "frames": num_frames,
        "motion_bucket_id_used": get_motion_config(motion, quality),
        "video_base64": video_b64
    }


runpod.serverless.start({"handler": handler})
