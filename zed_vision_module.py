import os
import time
import threading

import cv2
import numpy as np
import tensorflow as tf
from deepface import DeepFace

from gaze_speaker_utils import (
    open_zed_uvc,
    crop_eye_from_sbs,
    RESOLUTION_MAP,
)

tf.config.set_visible_devices([], "GPU")


class ZedVisionModule:
    def __init__(self, video_index=0, resolution="HD2K", fps=15, view="LEFT", no_mjpeg=False, show_window=False):
        self.video_index = video_index
        self.resolution = resolution
        self.fps = fps
        self.view = view
        self.no_mjpeg = no_mjpeg
        self.show_window = show_window

        self.latest_emotion = "unknown"
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.stop_flag = False

    def process_frame(self, frame):
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

        with self.frame_lock:
            print("[VISION] latest_frame is None:", self.latest_frame is None)

            if self.latest_frame is None:
                return None

            frame = self.latest_frame.copy()

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

            frame_counter += 1

            if frame_counter % 30 == 0:
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
        show_window=True
    )
    module.start()