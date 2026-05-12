import runpod

def handler(job):
    return {"ok": True}

runpod.serverless.start({
    "handler": handler
})
