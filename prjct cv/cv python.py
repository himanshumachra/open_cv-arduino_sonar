import cv2
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
import datetime

class MPSS_MultiModelSystem:
    def __init__(self, window):
        self.window = window
        self.window.title("MPSS | MULTI-MODAL TACTICAL SURVEILLANCE")
        self.window.geometry("1400x850")
        self.window.configure(bg="#020202")

        # --- AI INITIALIZATION ---
        # Model 1: Pose Estimation (Humans + Skeletons)
        self.pose_model = YOLO('yolov8n-pose.pt') 
        
        # Model 2: Specialized Detection (Weapons/Drones)
        # Note: In a real project, you'd use your own 'best.pt' here.
        # For now, we use standard nano weights as a placeholder.
        self.object_model = YOLO('yolov8n.pt') 

        self.setup_ui()
        self.cap = cv2.VideoCapture(0)
        self.update_loop()

    def setup_ui(self):
        # Layout: 80% Left (Camera), 20% Right (Data)
        self.window.columnconfigure(0, weight=8)
        self.window.columnconfigure(1, weight=2)
        self.window.rowconfigure(0, weight=1)

        # Camera Panel
        self.left_panel = tk.Frame(self.window, bg="black", highlightbackground="#00FF41", highlightthickness=1)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.video_label = tk.Label(self.left_panel, bg="black")
        self.video_label.pack(expand=True, fill="both")

        # Command Panel
        self.right_panel = tk.Frame(self.window, bg="#0a0a0a")
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)

        tk.Label(self.right_panel, text="TACTICAL FEED", font=("Courier", 18, "bold"), 
                 fg="#00FF41", bg="#0a0a0a").pack(pady=20)

        self.stats = {
            "HUMANS": tk.StringVar(value="0"),
            "WEAPONS": tk.StringVar(value="0"),
            "DRONES": tk.StringVar(value="0"),
            "THREAT": tk.StringVar(value="SCANNING")
        }

        for key, var in self.stats.items():
            f = tk.Frame(self.right_panel, bg="#111", pady=10)
            f.pack(fill="x", padx=15, pady=5)
            tk.Label(f, text=key, font=("Courier", 10), fg="#888", bg="#111").pack(side="left", padx=10)
            tk.Label(f, textvariable=var, font=("Courier", 14, "bold"), fg="#00FF41", bg="#111").pack(side="right", padx=10)

        # Log
        self.log = tk.Text(self.right_panel, height=15, bg="black", fg="#00FF41", font=("Courier", 8), bd=0)
        self.log.pack(padx=10, pady=20, fill="x")

    def update_loop(self):
        success, frame = self.cap.read()
        if success:
            # 1. Pose & Human Detection
            pose_results = self.pose_model(frame, verbose=False)[0]
            annotated_frame = pose_results.plot(boxes=True, labels=False) # Plot skeletons
            
            # 2. Weapon & Drone Detection
            obj_results = self.object_model(frame, verbose=False)[0]
            
            w_count = 0
            d_count = 0
            
            for box in obj_results.boxes:
                cls = int(box.cls[0])
                name = self.object_model.names[cls]
                
                # Logic: Define which YOLO classes count as Threats
                # Standard YOLOv8 IDs: 0=person, 4=airplane(drone proxy), 43=knife
                if name in ["knife", "scissors", "suitcase"]: # Proxy for weapons
                    w_count += 1
                    self.draw_target(annotated_frame, box, "WEAPON DETECTED", (0, 0, 255))
                
                if name in ["airplane", "bird"]: # Proxy for drones
                    d_count += 1
                    self.draw_target(annotated_frame, box, "UAV / DRONE", (255, 100, 0))

            # Update Counters
            self.stats["HUMANS"].set(str(len(pose_results.boxes)))
            self.stats["WEAPONS"].set(str(w_count))
            self.stats["DRONES"].set(str(d_count))
            
            # Threat Level Logic
            if w_count > 0: self.stats["THREAT"].set("CRITICAL")
            elif len(pose_results.boxes) > 5: self.stats["THREAT"].set("ELEVATED")
            else: self.stats["THREAT"].set("NORMAL")

            # Final HUD Overlays
            self.apply_hud_accents(annotated_frame)

            # Display
            img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.video_label.imgtk = img_tk
            self.video_label.configure(image=img_tk)

        self.window.after(10, self.update_loop)

    def draw_target(self, frame, box, label, color):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def apply_hud_accents(self, frame):
        h, w, _ = frame.shape
        color = (0, 255, 65)
        # Center Crosshair
        cv2.line(frame, (w//2-20, h//2), (w//2+20, h//2), color, 1)
        cv2.line(frame, (w//2, h//2-20), (w//2, h//2+20), color, 1)

if __name__ == "__main__":
    root = tk.Tk()
    app = MPSS_MultiModelSystem(root)
    root.mainloop()