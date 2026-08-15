"""
capture_dataset.py
Session 1 tool: capture and auto-crop labeled coin images for dataset building.
Press the number key shown for each class to save the currently detected
coin crop into that class's train/val/test folder.
Press 'q' to quit.
"""
import cv2
import os
 
CLASSES = {
    ord('1'): "1_piso",
    ord('2'): "5_piso",
    ord('3'): "10_piso",
    ord('4'): "20_piso",
    ord('5'): "25_centavo",
}
BASE_DIR = "dataset/raw"
 
def ensure_folders():
    for cls in CLASSES.values():
        os.makedirs(os.path.join(BASE_DIR, cls), exist_ok=True)
 
def detect_largest_circle(gray_blurred):
    circles = cv2.HoughCircles(
        gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80,
        param1=100, param2=45, minRadius=40, maxRadius=200
    )
    if circles is None:
        return None
    circles = circles[0]
    return max(circles, key=lambda c: c[2])
 
def main():
    ensure_folders()
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check the connection or index.")
 
    counters = {}

    for cls in CLASSES.values():
        folder = os.path.join(BASE_DIR, cls)
        counters[cls] = len([
            f for f in os.listdir(folder)
            if f.endswith(".jpg")
        ])

    print("Place ONE coin at a time in view. Keys: 1=P1 2=P5 3=P10 4=P20 5=25c  q=quit")
 
    while True:
        ok, frame = cap.read()
        if not ok:
            break
 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray_detect = cv2.equalizeHist(gray) #to capture 20 peso
        blurred = cv2.GaussianBlur(gray, (11, 11), 2)
        circle = detect_largest_circle(blurred)
 
        display = frame.copy()
        crop = None
        if circle is not None:
            x, y, r = circle.astype(int)
            pad = int(r * 0.15)
            x1, y1 = max(0, x - r - pad), max(0, y - r - pad)
            x2, y2 = x + r + pad, y + r + pad
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                preview = cv2.resize(crop, (224, 224))
                cv2.imshow("Crop Preview", preview)
            cv2.circle(display, (x, y), r, (0, 255, 0), 2)
 
        cv2.putText(display, "1:P1 2:P5 3:P10 4:P20 5:25c q:quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y = 55
        for cls, count in counters.items():
            cv2.putText(
                display,
                f"{cls}: {count}/200",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            y += 25

        cv2.imshow("Dataset Capture", display)
 
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key in CLASSES and crop is not None and crop.size > 0:
            cls_name = CLASSES[key]
            folder = os.path.join(BASE_DIR, cls_name)

            existing = [
                f for f in os.listdir(folder)
                if f.lower().endswith(".jpg")
            ]

            if existing:
                numbers = [
                    int(os.path.splitext(f)[0].split("_")[-1])
                    for f in existing
                ]
                next_num = max(numbers) + 1
            else:
                next_num = 1

            fname = f"{cls_name}_{next_num:04d}.jpg"
            out_path = os.path.join(folder, fname)

            crop = cv2.resize(crop, (224, 224))
            cv2.imwrite(out_path, crop)

            counters[cls_name] = len(existing) + 1

            print(f"Saved: {out_path}")
            print(counters)
 
    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == "__main__":
    main()
