import streamlit as st
import sqlite3
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import trimesh
import cascadio
import pandas as pd

DB_FILE = "vc_history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
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

init_db()

st.set_page_config(page_title="Source Made — Quoting Engine", layout="centered")

st.title("Source Made — Direct STEP Quoting Engine")
st.write("Upload a STEP assembly to instantly calculate layer breakdowns, material removal, and AI-predicted cycle times.")

# Sidebar Navigation for Modes
mode = st.sidebar.selectbox("Navigation", ["Quote / Train", "Manage Training Data"])

# --- PROCESS STEP FUNCTION ---
def process_step(uploaded_file):
    # Streamlit saves uploaded files temporarily
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    try:
        scene = trimesh.load(temp_path)
        if isinstance(scene, trimesh.Scene):
            meshes = list(scene.geometry.values())
        else:
            meshes = [scene]
            
        stock_sizes = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
        total_area = 0
        total_removed_vol = 0
        layer_breakdown = {}
        
        for mesh in meshes:
            # Scale Meters to Inches (adjust if needed, based on previous testing)
            mesh.apply_scale(39.3701)

            extents = mesh.extents
            x, y, z = extents[0], extents[1], extents[2]
            
            categorized_z = 6.0
            for size in stock_sizes:
                if z <= size:
                    categorized_z = size
                    break
                    
            stock_vol = x * y * categorized_z
            removed_vol = max(0, stock_vol - mesh.volume) 
            
            total_area += (x * y)
            total_removed_vol += removed_vol
            layer_breakdown[categorized_z] = layer_breakdown.get(categorized_z, 0) + 1
            
        breakdown_str = ", ".join([f"{count}x {z}\" layer" for z, count in layer_breakdown.items()])
        
        data = {
            "file_name": uploaded_file.name,
            "total_area": total_area,
            "total_layers": len(meshes),
            "removed_vol": total_removed_vol,
            "breakdown": breakdown_str
        }
        return data
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- MODE 1: QUOTE & TRAIN ---
if mode == "Quote / Train":
    st.header("1. Upload Geometry")
    uploaded_file = st.file_uploader("Choose a STEP file", type=["step", "stp"])
    
    if uploaded_file is not None:
        with st.spinner("Analyzing 3D assembly and layers..."):
            step_data = process_step(uploaded_file)
            
        st.success("Geometry processed successfully!")
        
        col1, col2 = st.columns(2)
        col1.metric("Layers Detected", f"{step_data['total_layers']}")
        col2.metric("Material Removed", f"{step_data['removed_vol']:.2f} in³")
        st.write(f"**Stock Breakdown:** {step_data['breakdown']}")
        
        # Run Prediction
        conn = sqlite3.connect(DB_FILE)
        df_history = pd.read_sql("SELECT * FROM job_history", conn)
        conn.close()
        
        st.divider()
        st.header("2. Cycle Time Quote")
        
        current_features = np.array([[step_data["total_area"], step_data["total_layers"], step_data["removed_vol"]]])
        
        if len(df_history) < 3:
            fallback_time = (step_data["total_layers"] * 2.5) + (step_data["removed_vol"] * 0.08)
            st.warning(f"Not enough training data yet ({len(df_history)}/3 jobs logged). Showing standard formula estimate.")
            st.metric("Estimated Cycle Time", f"{fallback_time:.2f} minutes")
        else:
            X = df_history[["total_area", "total_layers", "removed_vol"]].values
            y = df_history["actual_time"].values
            
            model = LinearRegression()
            model.fit(X, y)
            prediction = max(1.0, model.predict(current_features)[0])
            
            st.metric("AI Quoted Cycle Time", f"{prediction:.2f} minutes")
            
        st.divider()
        st.header("3. AI Training (Post-Production Log)")
        st.write("If this was a completed job, input the actual floor time to train the AI model for future quotes.")
        
        with st.form("training_form"):
            actual_time = st.number_input("Actual Floor Time (mins)", min_value=0.1, step=0.5)
            submit_training = st.form_submit_button("Save to Database & Train")
            
            if submit_training:
                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                c.execute('''INSERT INTO job_history 
                             (file_name, total_area, total_layers, removed_vol, actual_time) 
                             VALUES (?, ?, ?, ?, ?)''', 
                          (step_data["file_name"], step_data["total_area"], step_data["total_layers"], step_data["removed_vol"], actual_time))
                conn.commit()
                conn.close()
                st.success("Successfully logged! The AI model has been updated.")

# --- MODE 2: MANAGE TRAINING DATA (DELETE BAD DATA) ---
elif mode == "Manage Training Data":
    st.header("Manage Training Database")
    st.write("Review past logged jobs. If you see a typo or an outlier, you can delete it directly from the database here.")
    
    conn = sqlite3.connect(DB_FILE)
    df_history = pd.read_sql("SELECT * FROM job_history", conn)
    conn.close()
    
    if df_history.empty:
        st.info("No training data logged yet.")
    else:
        st.dataframe(df_history, use_container_width=True)
        
        st.subheader("Delete a Record")
        record_ids = df_history["id"].tolist()
        selected_id = st.selectbox("Select Record ID to Delete", record_ids)
        
        if st.button("Delete Selected Record", type="primary"):
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("DELETE FROM job_history WHERE id = ?", (selected_id,))
            conn.commit()
            conn.close()
            st.success(f"Record ID {selected_id} deleted successfully. AI model updated.")
            st.rerun()

# Sidebar stats footer
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()
c.execute("SELECT COUNT(*) FROM job_history")
db_count = c.fetchone()[0]
conn.close()
st.sidebar.divider()
st.sidebar.text(f"Database Size: {db_count} jobs")