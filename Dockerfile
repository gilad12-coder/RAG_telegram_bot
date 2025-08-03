FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1


RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*


COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt


COPY main.py /app/main.py
COPY guide_HR.pdf /app/guide_HR.pdf


ENV HOST=0.0.0.0 PORT=8080
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
