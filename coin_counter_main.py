
import cv2
import numpy as np
from collections import deque, Counter


CALIBRATION_SCALE_MM_PER_PX = 0.20


COIN_SPECS = [
    ("25-Centavo",   0.25,      20.0,        False),
    ("1-Piso",       1.00,      23.0,        False),
    ("5-Piso",       5.00,      25.0,        False),
    ("10-Piso",      10.00,     27.0,        False),
    ("20-Piso",      20.00,     30.0,        True),
]
DIAMETER_TOLERANCE_MM = 1.3


GOLD_HSV_LOWER = np.array([15, 60, 90])
GOLD_HSV_UPPER = np.array([35, 255, 255])
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


def classify_by_size_and_color(frame_bgr, x, y, r):
    diameter_mm = 2.0 * r * CALIBRATION_SCALE_MM_PER_PX

    candidates = [
        spec for spec in COIN_SPECS
        if abs(diameter_mm - spec[2]) <= DIAMETER_TOLERANCE_MM
    ]
    if not candidates:
        return None, 0.0

    largest_candidate = max(candidates, key=lambda s: s[2])
    if largest_candidate[3]:
        if _has_gold_ring(frame_bgr, x, y, r):
            return largest_candidate[0], largest_candidate[1]
        else:
            candidates = [c for c in candidates if not c[3]]
            if not candidates:
                return None, 0.0

    best = min(candidates, key=lambda s: abs(diameter_mm - s[2]))
    return best[0], best[1]


def _has_gold_ring(frame_bgr, x, y, r):
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x - r), max(0, y - r)
    x2, y2 = min(w, x + r), min(h, y + r)
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cx, cy = x - x1, y - y1
    cv2.circle(mask, (cx, cy), r, 255, -1)
    cv2.circle(mask, (cx, cy), int(r * 0.55), 0, -1)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gold_mask = cv2.inRange(hsv, GOLD_HSV_LOWER, GOLD_HSV_UPPER)
    ring_pixels = cv2.countNonZero(mask)
    gold_pixels = cv2.countNonZero(cv2.bitwise_and(gold_mask, mask))
    if ring_pixels == 0:
        return False
    return (gold_pixels / ring_pixels) >= GOLD_RING_MIN_FRACTION


def detect_coins(frame_bgr):
    """Return a list of (x, y, r) circles detected in the frame."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=HOUGH_DP, minDist=HOUGH_MIN_DIST_PX,
        param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_RADIUS_PX, maxRadius=HOUGH_MAX_RADIUS_PX,
    )
    if circles is None:
        return []
    circles = np.round(circles[0]).astype(int)
    return [(x, y, r) for (x, y, r) in circles]


class CoinTrack:
    """One persistent coin identity across frames."""

    def __init__(self, track_id, x, y, r, label, value):
        self.id = track_id
        self.x, self.y, self.r = float(x), float(y), float(r)
        self.label_history = deque(maxlen=LABEL_HISTORY_LEN)
        self.frames_seen = 0
        self.missed = 0
        self._push_label(label, value)

    def _push_label(self, label, value):
        self.label_history.append((label, value))
        self.frames_seen += 1

    def update(self, x, y, r, label, value):
        a = POSITION_SMOOTHING_ALPHA
        self.x = a * x + (1 - a) * self.x
        self.y = a * y + (1 - a) * self.y
        self.r = a * r + (1 - a) * self.r
        self._push_label(label, value)
        self.missed = 0

    def mark_missed(self):
        self.missed += 1

    def voted_label(self):
        """Majority-vote label/value over recent history (ignores None votes
        unless that's all there is)."""
        real_votes = [lv for lv in self.label_history if lv[0] is not None]
        pool = real_votes if real_votes else list(self.label_history)
        if not pool:
            return None, 0.0
        (label, value), _count = Counter(pool).most_common(1)[0]
        return label, value

    def is_confirmed(self):
        return self.frames_seen >= MIN_CONFIRM_FRAMES

    def pos(self):
        return int(round(self.x)), int(round(self.y)), int(round(self.r))


class CoinTracker:

    def __init__(self):
        self.tracks = {}
        self._next_id = 0

    def update(self, detections):
        """detections: list of (x, y, r, label, value) for this frame."""
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())

        # Build all (distance, track_id, det_idx) pairs within range, then
        # greedily assign closest pairs first.
        pairs = []
        for tid in unmatched_tracks:
            t = self.tracks[tid]
            for di in unmatched_dets:
                x, y, r, label, value = detections[di]
                dist = np.hypot(t.x - x, t.y - y)
                if dist <= TRACK_MATCH_MAX_DIST_PX:
                    pairs.append((dist, tid, di))
        pairs.sort(key=lambda p: p[0])

        matched_tracks, matched_dets = set(), set()
        for dist, tid, di in pairs:
            if tid in matched_tracks or di in matched_dets:
                continue
            x, y, r, label, value = detections[di]
            self.tracks[tid].update(x, y, r, label, value)
            matched_tracks.add(tid)
            matched_dets.add(di)

        # Unmatched existing tracks: mark missed, drop if too stale.
        for tid in unmatched_tracks:
            if tid not in matched_tracks:
                self.tracks[tid].mark_missed()

        for tid in list(self.tracks.keys()):
            if self.tracks[tid].missed > MAX_MISSED_FRAMES:
                del self.tracks[tid]

        # Unmatched detections: start new tracks.
        for di in unmatched_dets:
            if di not in matched_dets:
                x, y, r, label, value = detections[di]
                self.tracks[self._next_id] = CoinTrack(self._next_id, x, y, r, label, value)
                self._next_id += 1

        return self.tracks


def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check the connection or index.")

    tracker = CoinTracker()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 1) Raw per-frame detection + per-frame classification (as before)
        raw_detections = []
        for (x, y, r) in detect_coins(frame):
            label, value = classify_by_size_and_color(frame, x, y, r)
            raw_detections.append((x, y, r, label, value))

        # 2) Feed raw detections into the tracker to get stable identities
        tracks = tracker.update(raw_detections)

        # 3) Only confirmed tracks (seen consistently) count toward totals
        counts = {spec[0]: 0 for spec in COIN_SPECS}
        total_value = 0.0

        for t in tracks.values():
            x, y, r = t.pos()
            label, value = t.voted_label()

            if not t.is_confirmed():
                # Still "proving itself" - show as pending, don't count yet
                cv2.circle(frame, (x, y), r, (0, 165, 255), 1)
                continue

            if label is None:
                cv2.circle(frame, (x, y), r, (0, 0, 255), 2)
                cv2.putText(frame, "?", (x - 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                continue

            counts[label] += 1
            total_value += value
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.putText(frame, label, (x - r, y - r - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # --- Overlay counts and total value ---
        y_offset = 25
        for label, *_ in COIN_SPECS:
            cv2.putText(frame, f"{label}: {counts[label]}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25

        cv2.putText(frame, f"TOTAL: PHP {total_value:.2f}", (10, y_offset + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("NGC Coin Recognition and Value Counter By RANOCO, ALENBERT B.", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()