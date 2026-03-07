import cv2 as cv
import numpy as np
import os
import subprocess

ZOOM_FACTOR = 2
FONT_SCALE = 0.8
LINE_THICKNESS = 2

CLASSES = ["LTE", "NR", "Background"]
CLASS_COLORS = [
    (0, 0, 0),
    (0, 0, 128),
    (191, 163, 248)
]

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
#(1) Define Paths
BASE_DIR = './archive'
MODEL_PATH = './model_spectrogram.onnx'

result = subprocess.run(
    ['zenity', '--file-selection', f'--filename={os.path.abspath(os.path.join(BASE_DIR, "testSet/input/"))}', '--title=Select Spectrogram Image'],
    capture_output=True, text=True
)
IMG_PATH = result.stdout.strip()

filename = os.path.basename(IMG_PATH)
MASK_PATH = os.path.join(BASE_DIR, 'testSet', 'label', filename)

def calculate_signal_miou(pred, target):
    ious = []
    for cls in [0, 1]:
        inter = np.logical_and(pred == cls, target == cls).sum()
        union = np.logical_or(pred == cls, target == cls).sum()
        if union > 0:
            ious.append(inter / (union + 1e-7))
    return np.mean(ious) if ious else 0.0

def get_id_from_label_robust(mask_rgb):
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
#(2) Load image
    img = cv.imread(IMG_PATH)
    h, w = img.shape[:2]
    overlay = img.copy()

    x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
    x2, y2 = np.random.randint(x1, w), np.random.randint(y1, h)

    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    alpha = np.random.uniform(0.1, 0.9)

    img = cv.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)

    gt_rgb = cv.imread(MASK_PATH)

#(3) Load model
    net = cv.dnn.readNetFromONNX(MODEL_PATH)

    H_orig, W_orig = img.shape[:2]
    new_W, new_H = int(W_orig * ZOOM_FACTOR), int(H_orig * ZOOM_FACTOR)

#(4) convert image
    input_blob = preprocess_image(img)
    net.setInput(input_blob)
#(5) get masks
    preds = net.forward()
    mask_id = np.argmax(preds[0], axis=0).astype(np.uint8)

    gt_id = get_id_from_label_robust(gt_rgb)
    mask_id_resized = cv.resize(mask_id, (W_orig, H_orig), interpolation=cv.INTER_NEAREST)
    miou_score = calculate_signal_miou(mask_id_resized, gt_id)

    mask_id_zoom = cv.resize(mask_id, (new_W, new_H), interpolation=cv.INTER_NEAREST)
#(6) draw masks
    colored_mask = np.zeros((new_H, new_W, 3), dtype=np.uint8)
    for i, color in enumerate(CLASS_COLORS):
        colored_mask[mask_id_zoom == i] = color
#(7) Visualization
    overlay = cv.addWeighted(cv.resize(img, (new_W, new_H)), 0.7, colored_mask, 0.3, 0)

    for cls_idx in [0, 1]:
        class_mask = (mask_id_resized == cls_idx).astype(np.uint8) * 255
        contours, _ = cv.findContours(class_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv.contourArea(cnt) > 500:
                x, y, w, h = cv.boundingRect(cnt)
                x_z, y_z = x * ZOOM_FACTOR, y * ZOOM_FACTOR

                text = CLASSES[cls_idx]
                cv.putText(overlay, text, (x_z + 5, y_z + 25), cv.FONT_HERSHEY_DUPLEX,
                            FONT_SCALE, (0, 0, 0), LINE_THICKNESS + 1)
                cv.putText(overlay, text, (x_z + 5, y_z + 25), cv.FONT_HERSHEY_DUPLEX,
                            FONT_SCALE, (255, 255, 255), LINE_THICKNESS)

    miou_text = f"Signal mIoU: {miou_score:.4f}"
    (t_w, t_h), _ = cv.getTextSize(miou_text, cv.FONT_HERSHEY_DUPLEX, FONT_SCALE, LINE_THICKNESS)

    text_x = new_W - t_w - 15
    text_y = new_H - 15

    cv.putText(overlay, miou_text, (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                FONT_SCALE, (255, 255, 255), LINE_THICKNESS)

    cv.imshow('Semantic Segmentation Spectrogram', overlay)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()