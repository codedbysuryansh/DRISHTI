import cv2
import mediapipe as mp

class HandDetector:
    """
    MediaPipe Hands wrapper for frame-wise inference.
    Returns landmarks in normalized coords (0-1) plus pixel coords.
    """
    def __init__(
        self,
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def infer(self, frame_bgr):
        """
        Input: BGR image (OpenCV)
        Output: MediaPipe results object
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)

    def extract_landmarks(self, frame_bgr, results):
        """
        Output: list of dict rows (ready for CSV/DF)
        Includes normalized and pixel coordinates.
        """
        h, w = frame_bgr.shape[:2]
        rows = []

        if not results.multi_hand_landmarks:
            return rows

        handedness_list = results.multi_handedness or [None] * len(results.multi_hand_landmarks)

        for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handed = None
            if handedness_list[hand_id] is not None:
                handed = handedness_list[hand_id].classification[0].label  # 'Left' or 'Right'

            for lm_id, lm in enumerate(hand_landmarks.landmark):
                rows.append({
                    "hand_id": hand_id,
                    "handedness": handed,
                    "landmark_id": lm_id,
                    "x_norm": float(lm.x),
                    "y_norm": float(lm.y),
                    "z_norm": float(lm.z),
                    "x_px": int(lm.x * w),
                    "y_px": int(lm.y * h)
                })
        return rows

    def draw(self, frame_bgr, results):
        annotated = frame_bgr.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    annotated, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        return annotated