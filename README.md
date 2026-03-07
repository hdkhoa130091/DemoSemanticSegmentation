# 📡 RF Signal Semantic Segmentation (4G LTE & 5G NR)

Dự án sử dụng Deep Learning để phân mảng phổ tín hiệu di động (Spectrogram) giúp nhận diện và phân loại tín hiệu 4G (LTE) và 5G (NR) theo thời gian thực.

## 📦 Tài nguyên (Large Files)
Do giới hạn kích thước file của GitHub, các dữ liệu nặng được lưu trữ tại Hugging Face:
* **Models (.onnx, .pth, .tflite):** [Hugging Face Models](https://huggingface.co/hdkhoa130091/RF-Signal-Segmentation-4G-5G)
* **Dataset (Spectrogram images):** [Hugging Face Dataset](https://huggingface.co/datasets/hdkhoa130091/LTENRSpectrogramVisualize)

## 🛠 Cài đặt
1. Clone dự án: `git clone https://github.com/hdkhoa130091/DemoSemanticSegmentation.git`
2. Cài đặt thư viện: `pip install -r requirements.txt`
3. Tải model từ link trên và đặt vào thư mục `models/`.

## 🚀 Chạy Demo
* Phân mảng ảnh: `python ImageSegmentation.py`
* Phân mảng màn hình thời gian thực: `python VideoSegmentation.py`
