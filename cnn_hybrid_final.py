import cv2
import numpy as np
import tensorflow as tf
import time
from collections import deque, Counter

MODEL_PATH = "models/coin_classifier.keras"
IMG_SIZE = (224, 224)

CONFIDENCE_THRESHOLD = 0.50

CNN_WEIGHT = 0.50 
SIZE_WEIGHT = 0.50

CLASS_INFO = {
    0: ("10-Piso", 10.00, 27.0, False),
    1: ("1-Piso", 1.00, 23.0, False),
    2: ("20-Piso", 20.00, 30.0, True),
    3: ("25-Centavo", 0.25, 20.0, False),
    4: ("5-Piso", 5.00, 25.0, False),
}

model = tf.keras.models.load_model(
    MODEL_PATH
)

CALIBRATION_SCALE_MM_PER_PX = 0.20

COIN_SPECS = [
    ("25-Centavo", 0.25, 20.0, False),
    ("1-Piso", 1.00, 23.0, False),
    ("5-Piso", 5.00, 25.0, False),
    ("10-Piso", 10.00, 27.0, False),
    ("20-Piso", 20.00, 30.0, True),
]

DIAMETER_TOLERANCE_MM = 1.3

GOLD_HSV_LOWER = np.array(
    [15, 60, 90]
)

GOLD_HSV_UPPER = np.array(
    [35, 255, 255]
)

GOLD_RING_MIN_FRACTION = 0.10

HOUGH_DP = 1.2
HOUGH_MIN_DIST_PX = 40
HOUGH_PARAM1 = 100
HOUGH_PARAM2 = 44
HOUGH_MIN_RADIUS_PX = 25
HOUGH_MAX_RADIUS_PX = 160



TRACK_MATCH_MAX_DIST_PX = 35
POSITION_SMOOTHING_ALPHA = 0.35
LABEL_HISTORY_LEN = 15
MIN_CONFIRM_FRAMES = 8
MAX_MISSED_FRAMES = 6

def validate_coin_size(r):
    diameter_mm = (
        2 *
        r *
        CALIBRATION_SCALE_MM_PER_PX
    )

    candidates = []

    for spec in COIN_SPECS:
        label = spec[0]
        value = spec[1]
        expected = spec[2]
        bimetallic = spec[3]

        error = abs(
            diameter_mm - expected
        )

        if error <= DIAMETER_TOLERANCE_MM:

            candidates.append(
                (
                    label,
                    value,
                    expected,
                    bimetallic
                )
            )

    return candidates

def calculate_size_score(r):
    diameter_mm = (
        2 *
        r *
        CALIBRATION_SCALE_MM_PER_PX
    )

    scores = []

    for class_id, info in CLASS_INFO.items():

        expected = info[2]

        error = abs(
            diameter_mm - expected
        )

        score = np.exp(
            -(error ** 2) /
            (2 * DIAMETER_TOLERANCE_MM ** 2)
        )

        scores.append(score)

    return np.array(scores)


def _has_gold_ring(frame_bgr, x, y, r):

    h, w = frame_bgr.shape[:2]

    x1 = max(
        0,
        x - r
    )

    y1 = max(
        0,
        y - r
    )

    x2 = min(
        w,
        x + r
    )

    y2 = min(
        h,
        y + r
    )

    roi = frame_bgr[
        y1:y2,
        x1:x2
    ]

    if roi.size == 0:
        return False


    mask = np.zeros(
        roi.shape[:2],
        dtype=np.uint8
    )


    cx = x - x1
    cy = y - y1


    cv2.circle(
        mask,
        (cx, cy),
        r,
        255,
        -1
    )


    cv2.circle(
        mask,
        (cx, cy),
        int(r * 0.55),
        0,
        -1
    )


    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )


    gold_mask = cv2.inRange(
        hsv,
        GOLD_HSV_LOWER,
        GOLD_HSV_UPPER
    )


    ring_pixels = cv2.countNonZero(
        mask
    )


    gold_pixels = cv2.countNonZero(
        cv2.bitwise_and(
            gold_mask,
            mask
        )
    )


    if ring_pixels == 0:
        return False


    fraction = (
        gold_pixels /
        ring_pixels
    )


    return (
        fraction >= GOLD_RING_MIN_FRACTION
    )



def classify_coin(model, frame, x, y, r):

    candidates = validate_coin_size(
        r
    )

    if len(candidates) == 0:
        return None, 0.0



    h, w = frame.shape[:2]


    x1 = max(
        0,
        x-r
    )

    y1 = max(
        0,
        y-r
    )

    x2 = min(
        w,
        x+r
    )

    y2 = min(
        h,
        y+r
    )


    crop = frame[
        y1:y2,
        x1:x2
    ]


    if crop.size == 0:
        return None, 0.0



    crop = cv2.resize(
        crop,
        IMG_SIZE
    )


    crop = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )


    img = np.expand_dims(
        crop.astype("float32") / 255.0,
        axis=0
    )


    prediction = model.predict(
        img,
        verbose=0
    )[0]


    size_scores = calculate_size_score(r)


    final_scores = (
        CNN_WEIGHT * prediction
        +
        SIZE_WEIGHT * size_scores
    )


    class_id = np.argmax(
        final_scores
    )


    confidence = float(
        final_scores[class_id]
    )


    if confidence < CONFIDENCE_THRESHOLD:
        return None, 0.0


    label, value, _, bimetallic = CLASS_INFO[class_id]

    if label == "20-Piso":

        if not _has_gold_ring(
            frame,
            x,
            y,
            r
        ):

            return None, 0.0



    return label, value
def detect_coins(frame_bgr):

    gray = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2GRAY
    )


    gray = cv2.medianBlur(
        gray,
        5
    )


    blurred = cv2.GaussianBlur(
        gray,
        (9, 9),
        2
    )


    circles = cv2.HoughCircles(
        blurred,
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


    return [
        (x, y, r)
        for x, y, r in circles
    ]



def filter_duplicates(circles):

    filtered = []


    for x, y, r in circles:

        duplicate = False


        for fx, fy, fr in filtered:

            distance = np.hypot(
                x - fx,
                y - fy
            )


            if distance < min(r, fr):

                duplicate = True
                break



        if not duplicate:

            filtered.append(
                (
                    x,
                    y,
                    r
                )
            )


    return sorted(
        filtered,
        key=lambda c: c[2],
        reverse=True
    )
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


        self.label_history = deque(
            maxlen=LABEL_HISTORY_LEN
        )


        self.frames_seen = 0
        self.missed = 0


        self._push_label(
            label,
            value
        )



    def _push_label(
        self,
        label,
        value
    ):

        self.label_history.append(
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

        a = POSITION_SMOOTHING_ALPHA


        self.x = (
            a * x +
            (1 - a) * self.x
        )


        self.y = (
            a * y +
            (1 - a) * self.y
        )


        self.r = (
            a * r +
            (1 - a) * self.r
        )


        self._push_label(
            label,
            value
        )


        self.missed = 0



    def mark_missed(self):

        self.missed += 1



    def voted_label(self):

        votes = [
            item
            for item in self.label_history
            if item[0] is not None
        ]


        if not votes:
            return None, 0.0


        (label, value), _ = Counter(
            votes
        ).most_common(1)[0]


        return label, value



    def is_confirmed(self):

        return (
            self.frames_seen >= MIN_CONFIRM_FRAMES
        )



    def pos(self):

        return (
            int(round(self.x)),
            int(round(self.y)),
            int(round(self.r))
        )



class CoinTracker:


    def __init__(self):

        self.tracks = {}

        self._next_id = 0



    def update(self, detections):

        unmatched_dets = list(
            range(len(detections))
        )


        unmatched_tracks = list(
            self.tracks.keys()
        )


        pairs = []


        for tid in unmatched_tracks:

            track = self.tracks[tid]


            for di in unmatched_dets:

                x, y, r, label, value = detections[di]


                distance = np.hypot(
                    track.x - x,
                    track.y - y
                )


                if distance <= TRACK_MATCH_MAX_DIST_PX:

                    pairs.append(
                        (
                            distance,
                            tid,
                            di
                        )
                    )



        pairs.sort(
            key=lambda x: x[0]
        )


        matched_tracks = set()
        matched_dets = set()



        for _, tid, di in pairs:


            if tid in matched_tracks:
                continue


            if di in matched_dets:
                continue



            x, y, r, label, value = detections[di]


            self.tracks[tid].update(
                x,
                y,
                r,
                label,
                value
            )


            matched_tracks.add(tid)
            matched_dets.add(di)



        for tid in unmatched_tracks:

            if tid not in matched_tracks:

                self.tracks[tid].mark_missed()



        for tid in list(self.tracks.keys()):

            if (
                self.tracks[tid].missed >
                MAX_MISSED_FRAMES
            ):

                del self.tracks[tid]



        for di in unmatched_dets:

            if di not in matched_dets:

                x, y, r, label, value = detections[di]


                self.tracks[self._next_id] = CoinTrack(
                    self._next_id,
                    x,
                    y,
                    r,
                    label,
                    value
                )


                self._next_id += 1



        return self.tracks
def main():

    cap = cv2.VideoCapture(1)


    if not cap.isOpened():

        raise RuntimeError(
            "Could not open webcam."
        )

    tracker = CoinTracker()
    prev_time = time.time()

    while True:

        ok, frame = cap.read()

        if not ok:
            break


        current_time = time.time()

        fps = 1.0 / (current_time - prev_time)

        prev_time = current_time

        detections = []

        circles = filter_duplicates(
            detect_coins(frame)
        )



        for x, y, r in circles:


            label, value = classify_coin(
                model,
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



        counts = {
            spec[0]: 0
            for spec in COIN_SPECS
        }


        total_value = 0.0



        for track in tracks.values():


            x, y, r = track.pos()


            label, value = track.voted_label()



            if not track.is_confirmed():

                cv2.circle(
                    frame,
                    (x, y),
                    r,
                    (0,165,255),
                    1
                )

                continue



            if label is None:


                cv2.circle(
                    frame,
                    (x, y),
                    r,
                    (0,0,255),
                    2
                )


                cv2.putText(
                    frame,
                    "?",
                    (x-10,y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,0,255),
                    2
                )

                continue



            counts[label] += 1

            total_value += value



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
                    y-r-8
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0,255,0),
                2
            )



        y_offset = 25



        for label, _, _, _ in COIN_SPECS:


            cv2.putText(
                frame,
                f"{label}: {counts[label]}",
                (
                    10,
                    y_offset
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,0),
                2
            )


            y_offset += 25



        cv2.putText(
            frame,
            f"TOTAL: PHP {total_value:.2f}",
            (
                10,
                y_offset+10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, y_offset + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
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
        ) == ord("q"):

            break



    cap.release()

    cv2.destroyAllWindows()



if __name__ == "__main__":

    main()