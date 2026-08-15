import cv2
import numpy as np
import tensorflow as tf
from collections import deque, Counter

MODEL_PATH = "models/coin_classifier_finetuned.keras"

IMG_SIZE = (224,224)
CONFIDENCE_THRESHOLD = 0.50

model = tf.keras.models.load_model(
    MODEL_PATH
)

CLASS_INFO = {
    0: ("10-Piso", 10.00),
    1: ("1-Piso", 1.00),
    2: ("20-Piso", 20.00),
    3: ("25-Centavo", 0.25),
    4: ("5-Piso", 5.00)
}


CALIBRATION_SCALE_MM_PER_PX = 0.20
COIN_SPECS = [
    ("25-Centavo",0.25,20.0),
    ("1-Piso",1.00,23.0),
    ("5-Piso",5.00,25.0),
    ("10-Piso",10.00,27.0),
    ("20-Piso",20.00,30.0)
]


DIAMETER_TOLERANCE_MM = 1.3
HOUGH_DP = 1.2
HOUGH_MIN_DIST_PX = 40
HOUGH_PARAM1 = 100
HOUGH_PARAM2 = 44
HOUGH_MIN_RADIUS_PX = 25
HOUGH_MAX_RADIUS_PX = 160

TRACK_DISTANCE = 40
POSITION_ALPHA = 0.35
LABEL_HISTORY = 15
MIN_CONFIRM_FRAMES = 8
MAX_MISSED_FRAMES = 3

GOLD_HSV_LOWER = np.array(
    [15,60,90]
)

GOLD_HSV_UPPER = np.array(
    [35,255,255]
)

GOLD_RING_MIN_FRACTION = 0.10



def detect_coins(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(
        gray
    )

    gray = cv2.GaussianBlur(
        gray,
        (9,9),
        2
    )

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST_PX,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_RADIUS_PX,
        maxRadius=HOUGH_MAX_RADIUS_PX
    )

    if circles is None:
        return []

    circles = np.round(
        circles[0]
    ).astype(int)

    detected=[]

    for x,y,r in circles:

        detected.append(
            (
                x,
                y,
                r
            )
        )


    return detected

def validate_size(radius):

    diameter_mm = (
        2 *
        radius *
        CALIBRATION_SCALE_MM_PER_PX
    )

    candidates=[]

    for label,value,expected in COIN_SPECS:


        error = abs(
            diameter_mm -
            expected
        )

        if error <= DIAMETER_TOLERANCE_MM:

            candidates.append(
                (
                    label,
                    value
                )
            )
    return candidates

def has_gold_ring(frame,x,y,r):

    h,w = frame.shape[:2]

    x1=max(0,x-r)
    y1=max(0,y-r)

    x2=min(w,x+r)
    y2=min(h,y+r)

    roi = frame[
        y1:y2,
        x1:x2
    ]

    if roi.size==0:
        return False

    mask=np.zeros(
        roi.shape[:2],
        dtype=np.uint8
    )

    cx=x-x1
    cy=y-y1

    cv2.circle(
        mask,
        (cx,cy),
        r,
        255,
        -1
    )

    cv2.circle(
        mask,
        (cx,cy),
        int(r*0.55),
        0,
        -1
    )

    hsv=cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    gold=cv2.inRange(
        hsv,
        GOLD_HSV_LOWER,
        GOLD_HSV_UPPER
    )

    ring_pixels=cv2.countNonZero(mask)
    
    gold_pixels=cv2.countNonZero(
        cv2.bitwise_and(
            gold,
            mask
        )
    )

    if ring_pixels==0:
        return False

    return (
        gold_pixels /
        ring_pixels
    ) >= GOLD_RING_MIN_FRACTION

def classify_coin(frame,x,y,r):

    candidates = validate_size(r)

    if len(candidates)==0:
        return None,0.0

    h,w=frame.shape[:2]

    x1=max(0,x-r)
    y1=max(0,y-r)

    x2=min(w,x+r)
    y2=min(h,y+r)

    crop=frame[
        y1:y2,
        x1:x2
    ]

    if crop.size==0:
        return None,0.0

    crop=cv2.resize(
        crop,
        IMG_SIZE
    )

    crop=cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    img=np.expand_dims(
        crop.astype("float32")/255.0,
        axis=0
    )

    prediction=model.predict(
        img,
        verbose=0
    )[0]

    class_id=np.argmax(
        prediction
    )

    confidence=float(
        prediction[class_id]
    )

    if confidence < CONFIDENCE_THRESHOLD:
        return None,0.0

    label,value = CLASS_INFO[class_id]

    if label=="20-Piso":
        if not has_gold_ring(
            frame,
            x,
            y,
            r
        ):
            return None,0.0



    return label,value

class CoinTrack:
    def __init__(
        self,
        track_id,
        x,
        y,
        r,
        label,
        value
    ):
        
        self.id = track_id
        self.x = float(x)
        self.y = float(y)
        self.r = float(r)

        self.labels = deque(
            maxlen=LABEL_HISTORY
        )

        self.frames_seen = 0
        self.missed = 0

        self.push_label(
            label,
            value
        )

    def push_label(
        self,
        label,
        value
    ):
        self.labels.append(
            (
                label,
                value
            )
        )
        self.frames_seen += 1

    def update(
        self,
        x,
        y,
        r,
        label,
        value
    ):

        a = POSITION_ALPHA

        self.x = (
            a*x +
            (1-a)*self.x
        )

        self.y = (
            a*y +
            (1-a)*self.y
        )

        self.r = (
            a*r +
            (1-a)*self.r
        )

        self.push_label(
            label,
            value
        )
        self.missed = 0

    def miss(self):
        self.missed += 1

    def get_label(self):
        valid = [
            item
            for item in self.labels
            if item[0] is not None
        ]

        if len(valid)==0:
            return None,0.0

        result = Counter(
            valid
        ).most_common(1)[0][0]

        return result

    def confirmed(self):
        return (
            self.frames_seen >=
            MIN_CONFIRM_FRAMES
        )

    def position(self):
        return (
            int(self.x),
            int(self.y),
            int(self.r)
        )

class CoinTracker:

    def __init__(self):
        self.tracks = {}
        self.next_id = 0

    def update(
        self,
        detections
    ):

        matched_tracks=set()
        matched_detections=set()

        for tid,track in list(
            self.tracks.items()
        ):

            best=None
            best_distance=9999

            for i,d in enumerate(detections):
                if i in matched_detections:
                    continue

                x,y,r,label,value=d

                distance=np.hypot(
                    track.x-x,
                    track.y-y
                )

                if (
                    distance <
                    TRACK_DISTANCE
                    and
                    distance <
                    best_distance
                ):

                    best_distance=distance
                    best=i

            if best is not None:
                x,y,r,label,value = detections[best]
                track.update(
                    x,
                    y,
                    r,
                    label,
                    value
                )

                matched_tracks.add(tid)
                matched_detections.add(best)

        for tid in list(self.tracks.keys()):
            if tid not in matched_tracks:
                self.tracks[tid].miss()

            if (
                self.tracks[tid].missed
                >
                MAX_MISSED_FRAMES
            ):
                del self.tracks[tid]


        for i,d in enumerate(detections):
            if i not in matched_detections:
                x,y,r,label,value=d

                self.tracks[
                    self.next_id
                ] = CoinTrack(
                    self.next_id,

                    x,
                    y,
                    r,
                    label,
                    value
                )
                self.next_id += 1
        return self.tracks

def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError(
            "Camera not detected."
        )

    tracker = CoinTracker()


    fps = 0
    prev_time = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cv2.getTickCount()
        fps = (
            cv2.getTickFrequency()
            /
            (current_time-prev_time)
        )

        prev_time=current_time
        detections=[]

        circles = detect_coins(frame)

        for x,y,r in circles:
            label,value = classify_coin(
                frame,
                x,
                y,
                r
            )

            detections.append(
                (
                    x,
                    y,
                    r,
                    label,
                    value
                )
            )

        tracks = tracker.update(
            detections
        )

        counts={
            spec[0]:0
            for spec in COIN_SPECS
        }

        total_value=0.0

        for track in tracks.values():
            x,y,r = track.position()
            label,value = track.get_label()
            if not track.confirmed():

                cv2.circle(
                    frame,
                    (x,y),
                    r,
                    (0,165,255),
                    1
                )
                continue

            if label is None:

                cv2.circle(
                    frame,
                    (x,y),
                    r,
                    (0,0,255),
                    2
                )
                continue

            counts[label]+=1
            total_value+=value

            cv2.circle(
                frame,
                (x,y),
                r,
                (0,255,0),
                2
            )

            cv2.putText(
                frame,
                label,
                (
                    x-r,
                    y-r-10
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0,255,0),
                2
            )
        y=25

        for label,_,_ in COIN_SPECS:
            cv2.putText(
                frame,
                f"{label}: {counts[label]}",
                (10,y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,0),
                2
            )
            y+=25

        cv2.putText(
            frame,
            f"TOTAL: PHP {total_value:.2f}",
            (10,y+10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (
                frame.shape[1]-150,
                30
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,255),
            2
        )

        cv2.imshow(
            "NGC Coin Recognition CNN",
            frame
        )

        if (
            cv2.waitKey(1)
            &
            0xFF
        ) == ord('q'):
            
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()