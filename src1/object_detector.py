import cv2
import numpy as np
from ultralytics import YOLO

class ObjectDetectorYOLO:
    """
    YOLOv8 object detector wrapper.
    Outputs detections as rows (dicts) suitable for CSV.
    """
    def __init__(self, model_name="yolov8n.pt", conf=0.25, iou=0.5):
        self.model = YOLO(model_name)
        self.conf = conf
        self.iou = iou

    def infer(self, frame_bgr):
        # Ultralytics expects BGR numpy ok
        results = self.model.predict(frame_bgr, conf=self.conf, iou=self.iou, verbose=False)
        return results[0]  # first image result

    def extract_detections(self, yolo_result):
        rows = []
        if yolo_result.boxes is None:
            return rows

        names = yolo_result.names  # class_id -> label

        for b in yolo_result.boxes:
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            x1, y1, x2, y2 = [float(x) for x in b.xyxy[0].tolist()]

            rows.append({
                "label": names.get(cls_id, str(cls_id)),
                "class_id": cls_id,
                "confidence": conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })
        return rows

    def draw(self, frame_bgr, det_rows):
        out = frame_bgr.copy()
        for r in det_rows:
            x1, y1, x2, y2 = map(int, [r["x1"], r["y1"], r["x2"], r["y2"]])
            label = r["label"]
            conf = r["confidence"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, f"{label} {conf:.2f}", (x1, max(20, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return out


def small_object_proposals(frame_bgr, min_area=40, max_area=5000):
    """
    Optional heuristic for jewellery: propose small high-contrast regions.
    Returns list of bboxes [x1,y1,x2,y2] that might contain small objects.
    This is NOT a detector, just proposals.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # edge/contrast
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    props = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < min_area or area > max_area:
            continue
        # ignore super-thin boxes
        if bw < 5 or bh < 5:
            continue
        # ignore boxes too close to borders (often noise)
        if x <= 1 or y <= 1 or x + bw >= w - 2 or y + bh >= h - 2:
            continue
        props.append([x, y, x + bw, y + bh])

    # keep top-N largest proposals to avoid spam
    props = sorted(props, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)[:30]
    return props