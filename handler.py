import runpod
import base64
from io import BytesIO
from PIL import Image
import imageio
import uuid

# -----------------------------
# Pipelines
# -----------------------------

svd_pipe = None
anim_pipe = None


# -----------------------------
# FAST MODE (SVD)
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
# CINEMATIC MODE (AnimateDiff)
# -----------------------------

def load_animdiff():
    global anim_pipe

    if anim_pipe is None:
        import torch
        from diffusers import AnimateDiffPipeline

        # motion module approach (standard AnimateDiff setup)
        anim_pipe = AnimateDiffPipeline.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-2"
        ).to("cuda")

    return anim_pipe


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

    num_frames = 14
    fps = max(4, min(8, round(num_frames / duration)))

    result = pipe(
        image,
        num_frames=num_frames,
        num_inference_steps=30,
        motion_bucket_id=75
    )

    return result.frames[0], fps


# -----------------------------
# CINEMATIC MODE (AnimateDiff)
# -----------------------------

def run_cinematic(image, duration):

    try:
        pipe = load_animdiff()

        fps = 6
        num_frames = min(int(duration * fps), 48)

        result = pipe(
            prompt="cinematic smooth camera motion, high quality, filmic lighting",
            num_frames=num_frames,
            guidance_scale=7.5
        )

        return result.frames[0], fps

    except Exception as e:
        print("AnimateDiff failed, falling back to SVD:", e)
        return run_fast(image, duration)


# -----------------------------
# MAIN HANDLER
# -----------------------------

def handler(job):

    inp = job.get("input", {})

    image_b64 = inp.get("image_base64")
    mode = inp.get("mode", "fast")
    duration = float(inp.get("duration", 3.0))

    if not image_b64:
        return {"status": "error", "message": "missing image"}

    image = pad_image(decode_image(image_b64))

    # -------------------------
    # ROUTER
    # -------------------------

    if mode == "cinematic":
        frames, fps = run_cinematic(image, duration)
    else:
        frames, fps = run_fast(image, duration)

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
        "duration": duration,
        "fps": fps,
        "frames": len(frames),
        "video_base64": video_b64,
        "backend_used": "animdiff" if mode == "cinematic" else "svd"
    }


runpod.serverless.start({"handler": handler})
