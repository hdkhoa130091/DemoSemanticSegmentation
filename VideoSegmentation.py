import cv2 as cv
import numpy as np
import os

ZOOM_FACTOR = 1.5
FONT_SCALE = 0.7
LINE_THICKNESS = 2

CLASSES = ["LTE", "NR", "Background"]
CLASS_COLORS = [(0, 0, 0), (0, 0, 128), (191, 163, 248)]

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model_spectrogram.onnx')


def preprocess_image(img, target_size=(256, 256)):
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_res = cv.resize(img_rgb, target_size, interpolation=cv.INTER_LINEAR)
    img_float = img_res.astype(np.float32) / 255.0
    img_norm = (img_float - MEAN) / STD
    return cv.dnn.blobFromImage(img_norm, 1.0, target_size, (0, 0, 0), swapRB=False)


def main():
    cap = cv.VideoCapture(0)
    net = cv.dnn.readNetFromONNX(MODEL_PATH)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H_orig, W_orig = frame.shape[:2]
        new_W, new_H = int(W_orig * ZOOM_FACTOR), int(H_orig * ZOOM_FACTOR)

        input_blob = preprocess_image(frame)
        net.setInput(input_blob)
        preds = net.forward()

        mask_id = np.argmax(preds[0], axis=0).astype(np.uint8)
        mask_id_resized = cv.resize(mask_id, (W_orig, H_orig), interpolation=cv.INTER_NEAREST)
        mask_id_zoom = cv.resize(mask_id, (new_W, new_H), interpolation=cv.INTER_NEAREST)

        colored_mask = np.zeros((new_H, new_W, 3), dtype=np.uint8)
        for i, color in enumerate(CLASS_COLORS):
            colored_mask[mask_id_zoom == i] = color

        frame_zoom = cv.resize(frame, (new_W, new_H))
        overlay = cv.addWeighted(frame_zoom, 0.6, colored_mask, 0.4, 0)

        for cls_idx in [0, 1]:  # LTE (0) và NR (1)
            class_mask = (mask_id_resized == cls_idx).astype(np.uint8) * 255
            contours, _ = cv.findContours(class_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv.contourArea(cnt) > 500:
                    x, y, w, h = cv.boundingRect(cnt)
                    x_z, y_z = int(x * ZOOM_FACTOR), int(y * ZOOM_FACTOR)

                    text = CLASSES[cls_idx]
                    cv.putText(overlay, text, (x_z + 5, y_z + 25), cv.FONT_HERSHEY_DUPLEX,
                                FONT_SCALE, (0, 0, 0), LINE_THICKNESS + 1)
                    cv.putText(overlay, text, (x_z + 5, y_z + 25), cv.FONT_HERSHEY_DUPLEX,
                                FONT_SCALE, (255, 255, 255), LINE_THICKNESS)

        cv.imshow('Live Semantic Segmentation - Mobile Signals', overlay)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()