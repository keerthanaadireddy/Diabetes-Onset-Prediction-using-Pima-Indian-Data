import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
import os

# --- Load Models ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "svm_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Load logic
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except:
    print("Error: Model files not found. Run training script first.")

class DiabetesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Diagnostics Tool v1.2")
        self.root.geometry("700x450")
        self.root.configure(bg="#f4f7f6")
        self.root.resizable(False, False)

        # Custom Colors
        self.primary_color = "#2c3e50"  # Dark Blue/Grey
        self.accent_color = "#3498db"   # Medical Blue
        self.bg_color = "#f4f7f6"
        self.text_color = "#333333"

        # Sidebar
        self.sidebar = tk.Frame(self.root, bg=self.primary_color, width=200, height=450)
        self.sidebar.pack(side="left", fill="y")

        tk.Label(self.sidebar, text="DIAGNOSTIC\nSYSTEM", fg="white", bg=self.primary_color, 
                 font=("Verdana", 14, "bold")).pack(pady=30)
        
        tk.Label(self.sidebar, text="Pima Indians\nDataset Analysis", fg="#bdc3c7", bg=self.primary_color, 
                 font=("Verdana", 9)).pack(side="bottom", pady=20)

        # Main Header
        self.header = tk.Frame(self.root, bg="white", height=60)
        self.header.pack(side="top", fill="x")
        tk.Label(self.header, text="Patient Health Assessment Input", bg="white", fg=self.primary_color,
                 font=("Arial", 12, "bold")).pack(side="left", padx=20, pady=15)

        # Input Grid Container
        self.container = tk.Frame(self.root, bg=self.bg_color, padx=30, pady=20)
        self.container.pack(expand=True, fill="both")

        self.labels = [
            "Pregnancies", "Glucose (mg/dL)", "Blood Pressure", "Skin Thickness",
            "Insulin (mu U/ml)", "Body Mass Index", "Pedigree Function", "Current Age"
        ]
        
        self.entries = []
        
        # Build 2-column grid manually for a "human-structured" feel
        for i, text in enumerate(self.labels):
            row = i // 2
            col = i % 2
            
            sub_frame = tk.Frame(self.container, bg=self.bg_color)
            sub_frame.grid(row=row, column=col, padx=15, pady=10, sticky="w")
            
            tk.Label(sub_frame, text=text, font=("Arial", 9), fg=self.text_color, bg=self.bg_color).pack(anchor="w")
            e = tk.Entry(sub_frame, font=("Arial", 11), width=22, relief="flat", highlightthickness=1)
            e.config(highlightbackground="#dcdde1", highlightcolor=self.accent_color)
            e.pack(pady=5)
            self.entries.append(e)

        # Action Buttons
        self.btn_frame = tk.Frame(self.root, bg=self.bg_color)
        self.btn_frame.pack(side="bottom", fill="x", pady=20)

        self.predict_btn = tk.Button(self.btn_frame, text="RUN ANALYSIS", command=self.run_prediction, 
                                    bg=self.accent_color, fg="white", font=("Arial", 10, "bold"),
                                    width=20, relief="flat", cursor="hand2")
        self.predict_btn.pack(side="right", padx=45)

        self.result_label = tk.Label(self.btn_frame, text="Ready for Analysis", font=("Arial", 11, "italic"),
                                     bg=self.bg_color, fg="#7f8c8d")
        self.result_label.pack(side="left", padx=45)

    def run_prediction(self):
        try:
            raw_data = [float(e.get()) for e in self.entries]
            features = np.array([raw_data])
            scaled = scaler.transform(features)
            
            prediction = model.predict(scaled)[0]
            prob = model.predict_proba(scaled)[0][1] * 100
            
            if prediction == 1:
                res_text = f"High Risk Detected ({prob:.1f}%)"
                res_color = "#c0392b"
            else:
                res_text = f"Low Risk Identified ({prob:.1f}%)"
                res_color = "#27ae60"
                
            self.result_label.config(text=res_text, fg=res_color, font=("Arial", 12, "bold"))
            
        except ValueError:
            messagebox.showwarning("Incomplete Data", "Please fill all fields with numeric values.")

if __name__ == "__main__":
    root = tk.Tk()
    app = DiabetesApp(root)
    root.mainloop()