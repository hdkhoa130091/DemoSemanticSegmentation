# Sử dụng Python 3.13 cho đồ án
FROM python:3.13-slim

# Sửa lỗi: Dùng libgl1 thay cho libgl1-mesa-glx vì bản cũ đã bị xóa
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn
COPY . .

# Chạy app chính của bạn
CMD ["python", "ImageSegmentation.py"]