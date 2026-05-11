import runpod

def handler(event):
    return {"status": "ok"}

runpod.serverless.start({
    "handler": handler
})
