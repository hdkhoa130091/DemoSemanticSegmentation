# Sử dụng Python 3.13 theo đúng môi trường bạn đang học
FROM python:3.13-slim

# Cài đặt thư viện hệ thống (Đã cập nhật gói libgl1 để sửa lỗi)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY . .

# Lệnh chạy app
CMD ["python", "ImageSegmentation.py"]