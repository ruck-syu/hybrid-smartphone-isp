FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY isp ./isp
COPY app ./app
COPY models ./models
COPY samples ./samples
COPY eval ./eval
EXPOSE 7860
CMD ["python", "app/server.py"]
