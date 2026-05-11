FROM runpod/pytorch:3.10-2.1.1-cuda11.8.0-devel

WORKDIR /workspace

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "handler.py"]
