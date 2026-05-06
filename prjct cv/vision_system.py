# ============================================================
#  VISION SYSTEM - OpenCV + Tkinter
#  Triggered by Processing radar (hooter_trigger.txt)
#  Detects objects via MobileNet SSD:
#    - Human / Machine / Robot  -> RED box + hooter
#    - Animal / Bird / Insect   -> ORANGE box (no hooter)
#    - Military Weapon          -> RED box (same urgency as human)
#    - Military Vehicle         -> BLUE box
#  Writes cv_result.txt for Processing sidebar display
# ============================================================

import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pygame
import os
import time
import threading
import numpy as np

# NO torch / ultralytics needed at runtime — uses OpenCV DNN + ONNX directly

# MediaPipe for advanced human pose detection (optional, falls back to simple overlay)
try:
    import mediapipe as mp
    # Verify that the 'solutions' sub-module actually exists (newer mp versions removed it)
    if not hasattr(mp, 'solutions'):
        raise ImportError("mediapipe installed but 'solutions' API not available")
except ImportError as _mp_err:
    mp = None
    print(f"[WARN] mediapipe disabled: {_mp_err}")

# ------------------------------------------------------------------ #
#  PATHS  (all files live in the same folder as this script)
# ------------------------------------------------------------------ #
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ONNX_MODEL   = os.path.join(BASE_DIR, "yolov8n.onnx")   # exported once; no torch needed
HOOTER_MP3   = os.path.join(BASE_DIR, "hooter.mp3")
TRIGGER_FILE = os.path.join(BASE_DIR, "sketch_260428a", "hooter_trigger.txt")
CV_RESULT    = os.path.join(BASE_DIR, "sketch_260428a", "cv_result.txt")

# ------------------------------------------------------------------ #
#  YOLOv8 uses COCO class names — no manual list needed (model.names)

# ------------------------------------------------------------------ #
#  Category mapping
# ------------------------------------------------------------------ #
# COCO/YOLO label names (differ from old Caffe: airplane, motorcycle, tv)
HUMAN_MACHINE = {
    "person", "car", "bus", "motorcycle", "airplane",
    "bicycle", "train", "tv"
}
ANIMAL = {
    "bird", "cat", "cow", "dog", "horse", "sheep",
    "elephant", "bear", "zebra", "giraffe"
}
# Additional military categories
MILITARY_WEAPONS = {
    "knife", "gun", "rifle", "sniper", "pistol",
    "grenade", "rpg", "rocket", "weapon", "scissors"
}
MILITARY_VEHICLES = {
    "tank", "truck", "jeep", "armored", "military", "vehicle"
}

def categorize(label: str) -> str:
    if label in HUMAN_MACHINE:
        return "HUMAN / MACHINE"
    elif label in ANIMAL:
        return "ANIMAL / BIRD"
    elif label in MILITARY_WEAPONS:
        return "MILITARY WEAPON"
    elif label in MILITARY_VEHICLES:
        return "MILITARY VEHICLE"
    else:
        return "UNKNOWN OBJECT"

def box_color(label: str):
    """Return BGR colour based on the detected category."""
    cat = categorize(label)
    if cat == "HUMAN / MACHINE":
        return (0, 0, 255)          # RED
    elif cat == "ANIMAL / BIRD":
        return (0, 165, 255)       # ORANGE
    elif cat == "MILITARY WEAPON":
        return (0, 0, 255)          # RED (same as human)
    elif cat == "MILITARY VEHICLE":
        return (255, 0, 0)          # BLUE
    else:
        return (0, 200, 200)        # CYAN (unknown)

# ------------------------------------------------------------------ #
#  COCO class names (YOLOv8 / ONNX output order)
# ------------------------------------------------------------------ #
COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

# ------------------------------------------------------------------ #
#  Load YOLOv8 ONNX via OpenCV DNN  (NO torch / ultralytics needed)
# ------------------------------------------------------------------ #
print("[INIT] Loading YOLOv8 ONNX model via OpenCV DNN …")
net = cv2.dnn.readNet(ONNX_MODEL)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print("[INIT] Model loaded.")

# ------------------------------------------------------------------ #
#  Hooter
# ------------------------------------------------------------------ #
pygame.mixer.init()

def play_hooter():
    def _play():
        if os.path.exists(HOOTER_MP3):
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.load(HOOTER_MP3)
                pygame.mixer.music.play()
        else:
            print("[WARN] hooter.mp3 not found, skipping audio.")
    threading.Thread(target=_play, daemon=True).start()

# ------------------------------------------------------------------ #
#  Write results for Processing sidebar
# ------------------------------------------------------------------ #
def write_cv_result(text: str):
    try:
        with open(CV_RESULT, "w") as f:
            f.write(text)
    except Exception:
        pass

# ------------------------------------------------------------------ #
#  Main GUI class
# ------------------------------------------------------------------ #
class VisionSystem:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CV Vision System — Radar Object Detector")
        self.root.configure(bg="#0a0c10")
        self.root.resizable(True, True)

        self._build_ui()

        self.cap = None
        self.running = False
        self.trigger_active = False

        # watcher thread
        self._watcher_thread = threading.Thread(
            target=self._watch_trigger, daemon=True
        )
        self._watcher_thread.start()

        # MediaPipe pose (optional)
        if mp:
            try:
                self.pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                )
            except Exception as e:
                print(f"[WARN] MediaPipe Pose init failed: {e}")
                self.pose = None
        else:
            self.pose = None

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -------------------------------------------------------------- #
    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#0d1117", pady=6)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text="⬡  RADAR VISION SYSTEM",
            font=("Consolas", 16, "bold"),
            fg="#00ff60",
            bg="#0d1117",
        ).pack(side=tk.LEFT, padx=16)

        self.status_lbl = tk.Label(
            header,
            text="● STANDBY",
            font=("Consolas", 13, "bold"),
            fg="#555",
            bg="#0d1117",
        )
        self.status_lbl.pack(side=tk.RIGHT, padx=16)

        # Main layout
        main = tk.Frame(self.root, bg="#0a0c10")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        # Camera feed (left)
        cam_frame = tk.Frame(main, bg="#111418", bd=2, relief=tk.GROOVE)
        cam_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            cam_frame,
            text="LIVE CAMERA FEED",
            font=("Consolas", 11),
            fg="#00cc50",
            bg="#111418",
        ).pack(pady=(6, 2))

        self.video_lbl = tk.Label(cam_frame, bg="#0a0c10")
        self.video_lbl.pack(padx=4, pady=4, fill=tk.BOTH, expand=True)

        # Info panel (right)
        info_frame = tk.Frame(main, bg="#0d1117", width=260)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        info_frame.pack_propagate(False)

        tk.Label(
            info_frame,
            text="DETECTION INFO",
            font=("Consolas", 12, "bold"),
            fg="#00cc50",
            bg="#0d1117",
        ).pack(pady=(10, 4))

        ttk.Separator(info_frame, orient="horizontal").pack(fill=tk.X, padx=8)

        def info_row(label, var_name):
            row = tk.Frame(info_frame, bg="#0d1117", pady=2)
            row.pack(fill=tk.X, padx=10)
            tk.Label(
                row,
                text=label,
                width=12,
                anchor="w",
                font=("Consolas", 11),
                fg="#778899",
                bg="#0d1117",
            ).pack(side=tk.LEFT)
            v = tk.StringVar(value="--")
            lbl = tk.Label(
                row,
                textvariable=v,
                anchor="w",
                font=("Consolas", 11, "bold"),
                fg="#ffffff",
                bg="#0d1117",
            )
            lbl.pack(side=tk.LEFT)
            return v, lbl

        self.var_object,    self.lbl_object    = info_row("Object  :", "object")
        self.var_category,  self.lbl_category  = info_row("Category:", "cat")
        self.var_confidence,self.lbl_confidence= info_row("Confidence:", "conf")
        self.var_action,    self.lbl_action    = info_row("Action  :", "action")
        self.var_time,      self.lbl_time      = info_row("Time    :", "time")

        ttk.Separator(info_frame, orient="horizontal").pack(fill=tk.X, padx=8, pady=6)

        # Detection log
        tk.Label(
            info_frame,
            text="DETECTION LOG",
            font=("Consolas", 11, "bold"),
            fg="#00cc50",
            bg="#0d1117",
        ).pack(pady=(0, 4))

        log_frame = tk.Frame(info_frame, bg="#0d1117")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=6)

        scrollbar = tk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_box = tk.Text(
            log_frame,
            width=28,
            height=12,
            bg="#07090d",
            fg="#aabb88",
            font=("Consolas", 9),
            yscrollcommand=scrollbar.set,
            state=tk.DISABLED,
            relief=tk.FLAT,
        )
        self.log_box.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.log_box.yview)

        # Manual trigger button
        ttk.Separator(info_frame, orient="horizontal").pack(fill=tk.X, padx=8, pady=4)
        tk.Button(
            info_frame,
            text="▶  MANUAL SCAN",
            font=("Consolas", 11, "bold"),
            bg="#1a3f1f",
            fg="#00ff60",
            activebackground="#2a5a2f",
            relief=tk.FLAT,
            bd=0,
            pady=6,
            command=self._manual_trigger,
        ).pack(fill=tk.X, padx=10, pady=(0, 8))

        # Stop button
        tk.Button(
            info_frame,
            text="■  STOP CAMERA",
            font=("Consolas", 10),
            bg="#2a1a1a",
            fg="#ff6060",
            activebackground="#4a2a2a",
            relief=tk.FLAT,
            bd=0,
            pady=4,
            command=self._stop_camera,
        ).pack(fill=tk.X, padx=10, pady=(0, 10))

    # -------------------------------------------------------------- #
    def _watch_trigger(self):
        """Background thread: polls hooter_trigger.txt"""
        while True:
            try:
                if os.path.exists(TRIGGER_FILE):
                    with open(TRIGGER_FILE, "r") as f:
                        content = f.read().strip()
                    if content == "DETECTED" and not self.trigger_active:
                        self.trigger_active = True
                        os.remove(TRIGGER_FILE)   # consume trigger
                        self.root.after(0, self._start_camera)
            except Exception:
                pass
            time.sleep(0.3)

    # -------------------------------------------------------------- #
    def _manual_trigger(self):
        if not self.running:
            self._start_camera()

    # -------------------------------------------------------------- #
    def _start_camera(self):
        if self.running:
            return
        self.running = True
        self.status_lbl.config(text="● SCANNING", fg="#00ff60")
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self._update_frame()

    # -------------------------------------------------------------- #
    def _stop_camera(self):
        self.running = False
        self.trigger_active = False
        self.status_lbl.config(text="● STANDBY", fg="#555")
        if self.cap:
            self.cap.release()
            self.cap = None
        # Blank frame
        blank = Image.new("RGB", (640, 480), color=(10, 12, 16))
        photo = ImageTk.PhotoImage(blank)
        self.video_lbl.config(image=photo)
        self.video_lbl.image = photo

    # -------------------------------------------------------------- #
    def _update_frame(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(100, self._update_frame)
            return

        frame, detections = self._detect(frame)

        # Show in Tkinter
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=img)
        self.video_lbl.config(image=photo)
        self.video_lbl.image = photo

        # Info panel
        if detections:
            best = max(detections, key=lambda d: d["confidence"])
            self._update_info(best)
            write_cv_result(
                f"{best['label'].upper()} ({best['category']}) {best['confidence']:.0%}"
            )
        else:
            write_cv_result("-- No Detection --")

        self.root.after(15, self._update_frame)

    # -------------------------------------------------------------- #
    def _detect(self, frame):
        h, w = frame.shape[:2]
        INPUT_W, INPUT_H = 640, 640

        # --- Pre-process ---
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (INPUT_W, INPUT_H),
            swapRB=True, crop=False
        )
        net.setInput(blob)
        raw = net.forward()          # shape: (1, 84, 8400)
        raw = raw[0].T               # → (8400, 84)

        # --- Decode boxes ---
        cx    = raw[:, 0]; cy = raw[:, 1]
        bw    = raw[:, 2]; bh = raw[:, 3]
        scores = raw[:, 4:]          # (8400, 80)
        class_ids  = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        mask = confidences >= 0.45
        cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
        class_ids   = class_ids[mask]
        confidences = confidences[mask]

        # Scale back to original image size
        sx_arr = ((cx - bw / 2) * w / INPUT_W).astype(int)
        sy_arr = ((cy - bh / 2) * h / INPUT_H).astype(int)
        ex_arr = ((cx + bw / 2) * w / INPUT_W).astype(int)
        ey_arr = ((cy + bh / 2) * h / INPUT_H).astype(int)

        # NMS
        boxes_xywh = np.stack([
            sx_arr, sy_arr, ex_arr - sx_arr, ey_arr - sy_arr
        ], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh, confidences.tolist(), 0.45, 0.40
        )
        if len(indices) == 0:
            return frame, []

        det_results  = []
        hooter_needed = False

        for i in (indices.flatten() if hasattr(indices, 'flatten') else indices):
            confidence = float(confidences[i])
            label = COCO_NAMES[int(class_ids[i])]
            cat   = categorize(label)
            color = box_color(label)

            sx = max(0, int(sx_arr[i]))
            sy = max(0, int(sy_arr[i]))
            ex = min(w - 1, int(ex_arr[i]))
            ey = min(h - 1, int(ey_arr[i]))

            # Bounding box + corner accents
            cv2.rectangle(frame, (sx, sy), (ex, ey), color, 2)
            corner_len = 12
            cv2.line(frame, (sx, sy), (sx + corner_len, sy), color, 3)
            cv2.line(frame, (sx, sy), (sx, sy + corner_len), color, 3)
            cv2.line(frame, (ex, sy), (ex - corner_len, sy), color, 3)
            cv2.line(frame, (ex, sy), (ex, sy + corner_len), color, 3)
            cv2.line(frame, (sx, ey), (sx + corner_len, ey), color, 3)
            cv2.line(frame, (sx, ey), (sx, ey - corner_len), color, 3)
            cv2.line(frame, (ex, ey), (ex - corner_len, ey), color, 3)
            cv2.line(frame, (ex, ey), (ex, ey - corner_len), color, 3)

            # Label pill
            label_text = f"{label.upper()}  {confidence:.0%}  [{cat}]"
            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
            )
            pill_y1 = max(sy - th - 8, 0)
            pill_y2 = max(sy, th + 8)
            cv2.rectangle(frame, (sx, pill_y1), (sx + tw + 10, pill_y2), color, -1)
            cv2.putText(
                frame, label_text, (sx + 5, pill_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA,
            )

            # Pose / skeleton overlay for person
            if label == "person":
                if self.pose:
                    self._draw_pose_landmark(frame, sx, sy, ex, ey)
                else:
                    self._draw_person_overlay(frame, sx, sy, ex, ey)

            if cat in ("HUMAN / MACHINE", "MILITARY WEAPON"):
                hooter_needed = True

            det_results.append({
                "label": label, "category": cat,
                "confidence": confidence, "box": (sx, sy, ex, ey), "color": color,
            })

        if hooter_needed:
            play_hooter()

        return frame, det_results

    # -------------------------------------------------------------- #
    def _draw_pose_landmark(self, frame, sx, sy, ex, ey):
        """Draw MediaPipe pose landmarks inside the given bounding box."""
        if not mp or not self.pose:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if not results.pose_landmarks:
            return

        overlay = frame.copy()
        for lm in results.pose_landmarks.landmark:
            lx = int(lm.x * frame.shape[1])
            ly = int(lm.y * frame.shape[0])
            if sx <= lx <= ex and sy <= ly <= ey:
                cv2.circle(overlay, (lx, ly), 3, (0, 255, 0), -1)

        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # -------------------------------------------------------------- #
    def _draw_person_overlay(self, frame, sx, sy, ex, ey):
        """Simple geometric skeleton overlay (fallback)."""
        bw = ex - sx
        bh = ey - sy
        cx = sx + bw // 2
        head = sy + int(bh * 0.08)
        neck = sy + int(bh * 0.18)
        shlL = sx + int(bw * 0.20)
        shlR = sx + int(bw * 0.80)
        shlY = sy + int(bh * 0.25)
        elbL = sx + int(bw * 0.08)
        elbR = sx + int(bw * 0.92)
        elbY = sy + int(bh * 0.45)
        wstL = sx + int(bw * 0.10)
        wstR = sx + int(bw * 0.90)
        wstY = sy + int(bh * 0.60)
        hip = sy + int(bh * 0.60)
        knL = sx + int(bw * 0.30)
        knR = sx + int(bw * 0.70)
        knY = sy + int(bh * 0.78)
        ftL = sx + int(bw * 0.28)
        ftR = sx + int(bw * 0.72)
        ftY = ey

        sk_color = (0, 255, 200)
        r, th = 4, 2

        cv2.circle(frame, (cx, head), r + 4, sk_color, th)
        cv2.line(frame, (cx, neck), (cx, hip), sk_color, th)
        cv2.line(frame, (shlL, shlY), (shlR, shlY), sk_color, th)
        cv2.line(frame, (cx, neck), (shlL, shlY), sk_color, th)
        cv2.line(frame, (cx, neck), (shlR, shlY), sk_color, th)
        cv2.line(frame, (shlL, shlY), (elbL, elbY), sk_color, th)
        cv2.line(frame, (elbL, elbY), (wstL, wstY), sk_color, th)
        cv2.line(frame, (shlR, shlY), (elbR, elbY), sk_color, th)
        cv2.line(frame, (elbR, elbY), (wstR, wstY), sk_color, th)
        cv2.line(frame, (cx, hip), (knL, knY), sk_color, th)
        cv2.line(frame, (cx, hip), (knR, knY), sk_color, th)
        cv2.line(frame, (knL, knY), (ftL, ftY), sk_color, th)
        cv2.line(frame, (knR, knY), (ftR, ftY), sk_color, th)

        for pt in [
            (cx, neck),
            (shlL, shlY),
            (shlR, shlY),
            (elbL, elbY),
            (elbR, elbY),
            (wstL, wstY),
            (wstR, wstY),
            (cx, hip),
            (knL, knY),
            (knR, knY),
            (ftL, ftY),
            (ftR, ftY),
        ]:
            cv2.circle(frame, pt, r, sk_color, -1)

    # -------------------------------------------------------------- #
    def _update_info(self, det: dict):
        self.var_object.set(det["label"].upper())
        self.var_category.set(det["category"])
        self.var_confidence.set(f"{det['confidence']:.1%}")

        if det["category"] == "HUMAN / MACHINE":
            self.var_action.set("HOOTER FIRED")
            self.lbl_action.config(fg="#ff4444")
            self.lbl_category.config(fg="#ff4444")
        else:
            self.var_action.set("MONITORING")
            self.lbl_action.config(fg="#ffaa00")
            self.lbl_category.config(fg="#ffaa00")

        ts = time.strftime("%H:%M:%S")
        self.var_time.set(ts)

        entry = f"[{ts}] {det['label'].upper()} {det['confidence']:.0%} {det['category']}\n"
        self.log_box.config(state=tk.NORMAL)
        self.log_box.insert(tk.END, entry)
        self.log_box.see(tk.END)
        self.log_box.config(state=tk.DISABLED)

    # -------------------------------------------------------------- #
    def _on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


# ------------------------------------------------------------------ #
#  Entry point
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("980x660")
    app = VisionSystem(root)
    root.mainloop()
