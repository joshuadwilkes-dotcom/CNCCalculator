import tkinter as tk
from tkinter import filedialog, messagebox
import sqlite3
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import trimesh
import cascadio

class VCMLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VC — Direct STEP Quoting Engine")
        self.root.geometry("550x700")
        self.root.configure(padx=20, pady=20)
        
        self.db_file = "vc_history.db"
        self.current_step_data = None
        
        self.setup_database()
        self.create_ui()

    def setup_database(self):
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS job_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_name TEXT,
                        total_area REAL,
                        total_layers INTEGER,
                        removed_vol REAL,
                        actual_time REAL
                    )''')
        conn.commit()
        conn.close()

    def create_ui(self):
        # --- SECTION 1: LOAD STEP ---
        tk.Label(self.root, text="1. Load Geometry", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0,5))
        tk.Button(self.root, text="Upload STEP File...", command=self.load_step, width=25, bg="#2E86C1", fg="white").pack(anchor="w")
        
        self.file_lbl = tk.Label(self.root, text="No file loaded", fg="gray", justify="left")
        self.file_lbl.pack(anchor="w", pady=(5,15))

        # --- SECTION 2: AI QUOTE ---
        tk.Label(self.root, text="2. Cycle Time Quote", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0,5))
        
        self.pred_frame = tk.Frame(self.root, bg="#f0f0f0", padx=15, pady=15)
        self.pred_frame.pack(fill="x", pady=(0,15))
        
        self.pred_lbl = tk.Label(self.pred_frame, text="Load a STEP to generate quote...", font=("Arial", 11), bg="#f0f0f0")
        self.pred_lbl.pack()

        # --- SECTION 3: TRAINING/LOGGING ---
        tk.Label(self.root, text="3. AI Training (Post-Production)", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0,5))
        tk.Label(self.root, text="If this was a past job, log the actual floor time below to train the AI.", font=("Arial", 9, "italic"), fg="gray").pack(anchor="w", pady=(0,5))
        
        input_frame = tk.Frame(self.root)
        input_frame.pack(fill="x", pady=(0,10))
        
        tk.Label(input_frame, text="Actual Floor Time (mins):").pack(side="left")
        self.actual_time_entry = tk.Entry(input_frame, width=10)
        self.actual_time_entry.pack(side="left", padx=10)
        
        tk.Button(self.root, text="SAVE TO DATABASE & TRAIN", command=self.save_and_train, bg="#27AE60", fg="white", font=("Arial", 10, "bold")).pack(fill="x", pady=10)
        
        self.stats_lbl = tk.Label(self.root, text="", fg="gray")
        self.stats_lbl.pack(pady=20)
        self.update_stats()

    def process_step(self, filepath):
        scene = trimesh.load(filepath)
        
        # Handle both single bodies and multi-body assemblies
        if isinstance(scene, trimesh.Scene):
            meshes = list(scene.geometry.values())
        else:
            meshes = [scene]
            
        stock_sizes = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
        
        total_area = 0
        total_removed_vol = 0
        layer_breakdown = {}
        
        for mesh in meshes:
            # FIX: Force the CAD kernel to scale the raw coordinates from Meters back to Inches
            mesh.apply_scale(39.3701)
            
            # (If it still calculates too small, it might be reading meters. 
            # Comment out the line above and uncomment the line below to test meters.)
            # mesh.apply_scale(39.3701)

            extents = mesh.extents
            x, y, z = extents[0], extents[1], extents[2]
            
            # Categorize the Z height based on your foam stock
            categorized_z = 6.0
            for size in stock_sizes:
                if z <= size:
                    categorized_z = size
                    break
                    
            stock_vol = x * y * categorized_z
            # Prevent negative volume anomalies from mesh conversions
            removed_vol = max(0, stock_vol - mesh.volume) 
            
            total_area += (x * y)
            total_removed_vol += removed_vol
            layer_breakdown[categorized_z] = layer_breakdown.get(categorized_z, 0) + 1
            
        breakdown_str = ", ".join([f"{count}x {z}\" layer" for z, count in layer_breakdown.items()])
        
        return {
            "file_name": os.path.basename(filepath),
            "total_area": total_area,
            "total_layers": len(meshes),
            "removed_vol": total_removed_vol,
            "breakdown": breakdown_str
        }

    def load_step(self):
        filepath = filedialog.askopenfilename(filetypes=[("STEP Files", "*.step *.stp")])
        if not filepath: return
        
        self.file_lbl.config(text="Processing geometry... please wait.", fg="orange")
        self.root.update()
        
        try:
            self.current_step_data = self.process_step(filepath)
            
            info_text = (f"Loaded: {self.current_step_data['file_name']}\n"
                         f"Layers Detected: {self.current_step_data['total_layers']} ({self.current_step_data['breakdown']})\n"
                         f"Material Removed: {self.current_step_data['removed_vol']:.2f} in³")
            
            self.file_lbl.config(text=info_text, fg="blue")
            self.run_prediction()
            
        except Exception as e:
            messagebox.showerror("CAD Engine Error", f"Could not process STEP file:\n{str(e)}")
            self.file_lbl.config(text="No file loaded", fg="gray")

    def run_prediction(self):
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        c.execute("SELECT total_area, total_layers, removed_vol, actual_time FROM job_history")
        rows = c.fetchall()
        conn.close()

        data = self.current_step_data
        current_features = np.array([[data["total_area"], data["total_layers"], data["removed_vol"]]])

        if len(rows) < 3:
            # Fallback mathematical guess based purely on volume and setups
            fallback_time = (data["total_layers"] * 2.5) + (data["removed_vol"] * 0.08)
            self.pred_lbl.config(text=f"Not enough data to use AI yet ({len(rows)}/3 jobs).\n\nStandard Formula Estimate: {fallback_time:.2f} mins", fg="black")
            return

        # 3 Input Features (X) and 1 Output (y)
        X = np.array([row[0:3] for row in rows])
        y = np.array([row[3] for row in rows])

        model = LinearRegression()
        model.fit(X, y)

        prediction = model.predict(current_features)[0]
        
        # Ensure it never predicts a negative number
        final_quote = max(1.0, prediction)
        
        self.pred_lbl.config(text=f"AI QUOTED CYCLE TIME:\n{final_quote:.2f} minutes", fg="green", font=("Arial", 14, "bold"))

    def save_and_train(self):
        if not self.current_step_data:
            messagebox.showwarning("Hold Up", "Please load a STEP file first.")
            return
            
        actual_time_str = self.actual_time_entry.get()
        if not actual_time_str:
            messagebox.showwarning("Hold Up", "Please enter the actual time the CNC took.")
            return
            
        try:
            actual_time = float(actual_time_str)
        except ValueError:
            messagebox.showerror("Error", "Actual time must be a number.")
            return

        data = self.current_step_data
        
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        c.execute('''INSERT INTO job_history 
                     (file_name, total_area, total_layers, removed_vol, actual_time) 
                     VALUES (?, ?, ?, ?, ?)''', 
                  (data["file_name"], data["total_area"], data["total_layers"], data["removed_vol"], actual_time))
        conn.commit()
        conn.close()

        messagebox.showinfo("Success", "Data logged! The geometry model just got smarter.")
        self.actual_time_entry.delete(0, 'end')
        self.current_step_data = None
        self.file_lbl.config(text="No file loaded", fg="gray")
        self.pred_lbl.config(text="Load a STEP to generate quote...", font=("Arial", 11), fg="black")
        
        self.update_stats()

    def update_stats(self):
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM job_history")
        count = c.fetchone()[0]
        conn.close()
        
        self.stats_lbl.config(text=f"Database Size: {count} recorded jobs")

if __name__ == "__main__":
    root = tk.Tk()
    app = VCMLApp(root)
    root.mainloop()