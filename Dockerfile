#python
FROM python:3.13-slim
#dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    zenity \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
#interpreter
COPY requirements.txt .
#install interpreter
RUN pip install --no-cache-dir -r requirements.txt
#source code
COPY . .
#run
CMD ["python", "ImageSegmentation.py"]