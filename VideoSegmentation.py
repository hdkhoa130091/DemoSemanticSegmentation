import cv2 as cv
import numpy as np
import os
import mss
from PIL import Image

# --- CẤU HÌNH HỆ THỐNG ---
ZOOM_FACTOR = 1.2
CLASSES = ["LTE", "NR", "Background"]
CLASS_COLORS = [(0, 0, 0), (0, 0, 128), (191, 163, 248)]
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model_spectrogram.onnx')


def preprocess_image(img, target_size=(256, 256)):
    # Chuyển từ BGRA (mss) sang RGB
    img_rgb = cv.cvtColor(img, cv.COLOR_BGRA2RGB)
    img_res = cv.resize(img_rgb, target_size, interpolation=cv.INTER_LINEAR)
    img_float = img_res.astype(np.float32) / 255.0
    img_norm = (img_float - MEAN) / STD
    return cv.dnn.blobFromImage(img_norm, 1.0, target_size, (0, 0, 0), swapRB=False)


def main():
    net = cv.dnn.readNetFromONNX(MODEL_PATH)

    # Cấu hình vùng màn hình muốn chụp (Ví dụ: 800x600 ở góc trên bên trái)
    with mss.mss() as sct:
        # Bạn có thể điều chỉnh 'top', 'left', 'width', 'height' khớp với phần mềm hiện phổ tín hiệu
        monitor = {"top": 100, "left": 100, "width": 800, "height": 600}

        print(f"🚀 Đang phân mảng màn hình vùng {monitor}... Nhấn 'q' để thoát.")

        while True:
            # 1. Chụp màn hình (Screen Grab)
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)

            H_orig, W_orig = frame.shape[:2]
            new_W, new_H = int(W_orig * ZOOM_FACTOR), int(H_orig * ZOOM_FACTOR)

            # 2. Tiền xử lý và Dự đoán
            input_blob = preprocess_image(frame)
            net.setInput(input_blob)
            preds = net.forward()

            # 3. Xử lý Mask phân mảng
            mask_id = np.argmax(preds[0], axis=0).astype(np.uint8)
            mask_id_resized = cv.resize(mask_id, (W_orig, H_orig), interpolation=cv.INTER_NEAREST)
            mask_id_zoom = cv.resize(mask_id, (new_W, new_H), interpolation=cv.INTER_NEAREST)

            colored_mask = np.zeros((new_H, new_W, 3), dtype=np.uint8)
            for i, color in enumerate(CLASS_COLORS):
                colored_mask[mask_id_zoom == i] = color

            # 4. Trộn kết quả và hiển thị
            frame_zoom = cv.resize(frame, (new_W, new_H))
            # Loại bỏ kênh Alpha của mss để trộn với mask 3 kênh
            frame_zoom_rgb = cv.cvtColor(frame_zoom, cv.COLOR_BGRA2BGR)
            overlay = cv.addWeighted(frame_zoom_rgb, 0.6, colored_mask, 0.4, 0)

            # Vẽ nhãn LTE/NR lên các vùng phát hiện được
            for cls_idx in [0, 1]:
                class_mask = (mask_id_resized == cls_idx).astype(np.uint8) * 255
                contours, _ = cv.findContours(class_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv.contourArea(cnt) > 500:
                        x, y, w, h = cv.boundingRect(cnt)
                        cv.putText(overlay, CLASSES[cls_idx], (int(x * ZOOM_FACTOR), int(y * ZOOM_FACTOR)),
                                   cv.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

            cv.imshow('Screen Semantic Segmentation - 4G/5G Signals', overlay)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()