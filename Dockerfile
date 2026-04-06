FROM python:3.11-slim

RUN useradd -m -u 1000 envuser

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod -R 755 repo_templates/
RUN mkdir -p /tmp/openenv_work && chmod 777 /tmp/openenv_work

USER envuser

EXPOSE 7860

# Entry point: Gradio app that also mounts FastAPI endpoints
CMD ["python", "app.py"]
