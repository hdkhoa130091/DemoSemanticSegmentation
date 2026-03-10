# Sử dụng Python 3.13 mỏng nhẹ
FROM python:3.13-slim

# Cài đặt thư viện hệ thống cho OpenCV và giao diện
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy các file cần thiết
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Lệnh chạy mặc định
CMD ["python", "ImageSegmentation.py"]