import runpod

# ----------------------------
# Lazy-loaded SVD pipeline
# ----------------------------

svd_pipe = None

def load_svd():
    """
    Lazily load Stable Video Diffusion only when needed.
    This prevents rollout/startup delays.
    """
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


# ----------------------------
# YOUR EXISTING WORKING LOGIC
# ----------------------------

def legacy_video_pipeline(input_data):
    """
    Replace this with your CURRENT working code
    (the sideways frames version that already worked).
    """
    return {
        "status": "legacy_pipeline_used",
        "message": "This is your working pre-SVD pipeline"
    }


# ----------------------------
# MAIN HANDLER
# ----------------------------

def handler(job):
    input_data = job.get("input", {})

    use_svd = input_data.get("use_svd", False)

    # ------------------------
    # PATH 1: LEGACY (SAFE)
    # ------------------------
    if not use_svd:
        return legacy_video_pipeline(input_data)

    # ------------------------
    # PATH 2: SVD (LAZY LOAD)
    # ------------------------
    pipe = load_svd()

    image = input_data.get("image")

    if image is None:
        return {
            "error": "No image provided for SVD"
        }

    # NOTE: actual SVD call (you may refine this later)
    result = pipe(image)

    return {
        "status": "success",
        "mode": "svd",
        "output": str(result)
    }


# ----------------------------
# RUNPOD ENTRYPOINT
# ----------------------------

runpod.serverless.start({
    "handler": handler
})
