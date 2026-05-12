import runpod

# This is the function that will process your input
def handler(job):
    # 'job' is a dictionary containing the input data
    job_input = job['input']
    
    # Do your logic here
    result = f"Hello, {job_input.get('name', 'World')}!"
    
    return result

# Use this instead of uvicorn.run()
runpod.serverless.start({"handler": handler})

