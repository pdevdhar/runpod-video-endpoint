import runpod

def handler(job):
    image = job["input"].get("image_base64")

    return {
        "received": image is not None,
        "size": len(image) if image else 0
    }

runpod.serverless.start({"handler": handler})
