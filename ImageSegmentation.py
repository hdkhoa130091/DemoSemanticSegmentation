import cv2 as cv
import numpy as np
import os
import subprocess
import sys

# --- Cấu hình hiển thị ---
ZOOM_FACTOR = 2
FONT_SCALE = 0.8
LINE_THICKNESS = 2
CLASSES = ["LTE", "NR", "Background"]
CLASS_COLORS = [(0, 0, 0), (0, 0, 128), (191, 163, 248)]
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
# (1) Define path
BASE_DIR = './archive'
MODEL_PATH = './model_spectrogram.onnx'


def calculate_signal_miou(pred, target):
    ious = []
    for cls in [0, 1]:
        inter = np.logical_and(pred == cls, target == cls).sum()
        union = np.logical_or(pred == cls, target == cls).sum()
        if union > 0:
            ious.append(inter / (union + 1e-7))
    return np.mean(ious) if ious else 0.0


def get_id_from_label_robust(mask_rgb):
    # Sử dụng np.zeros_like để khởi tạo mask mặc định là Background (ID 2)
    h, w = mask_rgb.shape[:2]
    mask_id = np.ones((h, w), dtype=np.uint8) * 2

    lte_mask = cv.inRange(mask_rgb, np.array([0, 0, 0]), np.array([30, 30, 30]))
    mask_id[lte_mask > 0] = 0

    nr_mask = cv.inRange(mask_rgb, np.array([0, 0, 80]), np.array([60, 60, 180]))
    mask_id[nr_mask > 0] = 1
    return mask_id


def preprocess_image(img, target_size=(256, 256)):
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_res = cv.resize(img_rgb, target_size, interpolation=cv.INTER_LINEAR)
    img_float = img_res.astype(np.float32) / 255.0
    img_norm = (img_float - MEAN) / STD
    return cv.dnn.blobFromImage(img_norm, 1.0, target_size, (0, 0, 0), swapRB=False)


def main():
    #Chose Image
    try:
        result = subprocess.run(
            ['zenity', '--file-selection', '--title=Select your spectrogram'],
            capture_output=True, text=True, check=True
        )
        IMG_PATH = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error")
        return
    # (2) Load Image
    filename = os.path.basename(IMG_PATH)
    MASK_PATH = os.path.join(BASE_DIR, 'testSet', 'label', filename)

    img = cv.imread(IMG_PATH)
    gt_rgb = cv.imread(MASK_PATH)

    # (3) Load model
    net = cv.dnn.readNetFromONNX(MODEL_PATH)

    # Generate Rectangle
    h_orig, w_orig = img.shape[:2]
    overlay_orig = img.copy()
    x1, y1 = np.random.randint(0, w_orig), np.random.randint(0, h_orig)
    x2, y2 = np.random.randint(x1, w_orig), np.random.randint(y1, h_orig)
    color_rand = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    cv.rectangle(overlay_orig, (x1, y1), (x2, y2), color_rand, thickness=-1)
    img_processed = cv.addWeighted(overlay_orig, 0.3, img, 0.7, 0)

# (5) Inference
    input_blob = preprocess_image(img_processed)
    net.setInput(input_blob)
    preds = net.forward()
    mask_id = np.argmax(preds[0], axis=0).astype(np.uint8)

# (6) mIoU
    gt_id = get_id_from_label_robust(gt_rgb if gt_rgb is not None else np.zeros_like(img))
    mask_id_resized = cv.resize(mask_id, (w_orig, h_orig), interpolation=cv.INTER_NEAREST)
    miou_score = calculate_signal_miou(mask_id_resized, gt_id)

# (7) Visualization
    new_W, new_H = int(w_orig * ZOOM_FACTOR), int(h_orig * ZOOM_FACTOR)
    mask_id_zoom = cv.resize(mask_id, (new_W, new_H), interpolation=cv.INTER_NEAREST)
    colored_mask = np.zeros((new_H, new_W, 3), dtype=np.uint8)
    for i, color in enumerate(CLASS_COLORS):
        colored_mask[mask_id_zoom == i] = color

    img_zoom = cv.resize(img_processed, (new_W, new_H))
    final_view = cv.addWeighted(img_zoom, 0.7, colored_mask, 0.3, 0)

#Labels LTE/NR
    for cls_idx in [0, 1]:
        class_mask = (mask_id_resized == cls_idx).astype(np.uint8) * 255
        contours, _ = cv.findContours(class_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv.contourArea(cnt) > 500:
                x, y, w, h = cv.boundingRect(cnt)
                x_z, y_z = x * ZOOM_FACTOR, y * ZOOM_FACTOR
                cv.putText(final_view, CLASSES[cls_idx], (x_z + 5, y_z + 25),
                           cv.FONT_HERSHEY_DUPLEX, FONT_SCALE, (255, 255, 255), LINE_THICKNESS)

    # Hiển thị kết quả mIoU
    cv.putText(final_view, f"Signal mIoU: {miou_score:.4f}", (20, new_H - 20),
               cv.FONT_HERSHEY_DUPLEX, FONT_SCALE, (255, 255, 255), LINE_THICKNESS)

    cv.imshow('Semantic Segmentation Spectrogram', final_view)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()