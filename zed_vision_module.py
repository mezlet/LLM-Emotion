import os
import time
import threading

import cv2
import numpy as np

from gaze_speaker_utils import (
    open_zed_uvc,
    crop_eye_from_sbs,
    RESOLUTION_MAP,
)

# DeepFace/TensorFlow are only imported if emotion analysis is actually
# enabled. Importing tensorflow unconditionally (even when unused) is a
# common cause of native segfaults when this process also loads torch /
# sentence-transformers / mediapipe, so the import is deferred.
_DEEPFACE_IMPORTED = False
DeepFace = None


def _ensure_deepface_imported() -> bool:
    global _DEEPFACE_IMPORTED, DeepFace
    if _DEEPFACE_IMPORTED:
        return DeepFace is not None
    _DEEPFACE_IMPORTED = True
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        from deepface import DeepFace as _DeepFace
        DeepFace = _DeepFace
        return True
    except Exception as exc:
        print("[VISION] DeepFace/TensorFlow not available, emotion analysis disabled:", exc)
        DeepFace = None
        return False


class ZedVisionModule:
    def __init__(
        self,
        video_index=0,
        resolution="HD2K",
        fps=15,
        view="LEFT",
        no_mjpeg=False,
        show_window=False,
        enable_emotion_analysis=False,
    ):
        self.video_index = video_index
        self.resolution = resolution
        self.fps = fps
        self.view = view
        self.no_mjpeg = no_mjpeg
        self.show_window = show_window

        # DeepFace emotion analysis is OFF by default. The ZED camera feed is
        # also consumed by the Qwen2.5-VL facial-emotion sampler in the main
        # pipeline, so running DeepFace here too is redundant and adds extra
        # CPU/GPU load to every frame for no benefit.
        self.enable_emotion_analysis = bool(enable_emotion_analysis)

        self.latest_emotion = "disabled" if not self.enable_emotion_analysis else "unknown"
        self.latest_frame = None
        self.latest_frame_ts = 0.0
        self.frame_lock = threading.Lock()
        self.stop_flag = False

        if self.enable_emotion_analysis:
            _ensure_deepface_imported()

    def get_latest_frame(self, max_age_seconds=None):
        """
        Thread-safe accessor for the most recent camera frame. Returns a copy
        of the frame (BGR ndarray) or None if no frame has been captured yet,
        or if max_age_seconds is given and the frame is older than that.

        This lets other consumers (e.g. the Qwen2.5-VL facial-emotion
        sampler) read from this module's single camera handle instead of
        opening their own competing cv2.VideoCapture on the same physical
        ZED camera.
        """
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            if max_age_seconds is not None and (time.time() - self.latest_frame_ts) > max_age_seconds:
                return None
            return self.latest_frame.copy()

    def process_frame(self, frame):
        if not self.enable_emotion_analysis or DeepFace is None:
            return frame

        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False
            )

            if isinstance(result, list):
                result = result[0]

            dominant_emotion = result.get("dominant_emotion", "unknown")
            self.latest_emotion = dominant_emotion

            print("[EMOTION]", dominant_emotion)

            if self.show_window:
                cv2.putText(
                    frame,
                    f"Emotion: {dominant_emotion}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            return frame

        except Exception as e:
            print("[EMOTION] Error:", str(e))
            return frame

    def save_latest_frame(self, folder="logs/vqa"):
        print("[VISION] save_latest_frame called")

        frame = self.get_latest_frame()
        if frame is None:
            print("[VISION] latest_frame is None: True")
            return None

        os.makedirs(folder, exist_ok=True)

        image_path = os.path.join(
            folder,
            f"zed_frame_{int(time.time())}.jpg"
        )

        ok = cv2.imwrite(image_path, frame)

        print("[VISION] cv2.imwrite =", ok)
        print("[VISION] image_path =", image_path)

        return image_path

    def start(self):
        sbs_w, sbs_h, _, _ = RESOLUTION_MAP[self.resolution]

        cap = open_zed_uvc(
            device_index=self.video_index,
            sbs_w=sbs_w,
            sbs_h=sbs_h,
            fps=self.fps,
            use_mjpeg=(not self.no_mjpeg)
        )

        if cap is None or not cap.isOpened():
            print(f"[VISION] Could not open ZED camera index {self.video_index}")
            return

        print("[VISION] ZED camera opened.")
        print(f"[VISION] Emotion analysis (DeepFace): {'enabled' if self.enable_emotion_analysis else 'disabled'}")

        frame_counter = 0
        last_processed_frame = None

        while not self.stop_flag:
            ret, sbs = cap.read()

            if not ret or sbs is None:
                print("[VISION] Could not read ZED frame.")
                break

            if self.view == "RIGHT":
                frame = crop_eye_from_sbs(sbs, "RIGHT")
            elif self.view == "LEFT":
                frame = crop_eye_from_sbs(sbs, "LEFT")
            else:
                frame = sbs

            with self.frame_lock:
                self.latest_frame = frame.copy()
                self.latest_frame_ts = time.time()

            frame_counter += 1

            if self.enable_emotion_analysis and frame_counter % 30 == 0:
                last_processed_frame = self.process_frame(frame.copy())

            if self.show_window:
                display_frame = last_processed_frame if last_processed_frame is not None else frame
                cv2.imshow("ZED Vision Module", display_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(0.01)

        cap.release()

        if self.show_window:
            cv2.destroyAllWindows()

        print("[VISION] ZED vision module stopped.")

    def stop(self):
        self.stop_flag = True


if __name__ == "__main__":
    module = ZedVisionModule(
        video_index=0,
        resolution="HD2K",
        fps=15,
        view="LEFT",
        no_mjpeg=False,
        show_window=True,
        enable_emotion_analysis=False,
    )
    module.start()