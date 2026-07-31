import os
import sqlite3
import math
import streamlit as st
import trimesh

# Import your standalone CAM simulator module
from cam_engine import FoamCAMEngine

# -------------------------------------------------------------------
# 1. DATABASE SETUP (vc_history.db)
# -------------------------------------------------------------------
DB_PATH = "vc_history.db"

def init_db():
    """Initializes the SQLite database table for training history."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS job_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            total_area REAL,
            total_layers INTEGER,
            removed_vol REAL,
            simulated_cam_time REAL,
            actual_time REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_job_to_db(file_name, total_area, total_layers, removed_vol, sim_time, actual_time):
    """Saves a completed floor job to SQLite for future AI training."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO job_history (file_name, total_area, total_layers, removed_vol, simulated_cam_time, actual_time)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (file_name, total_area, total_layers, removed_vol, sim_time, actual_time))
    conn.commit()
    conn.close()

def fetch_all_jobs():
    """Retrieves recorded jobs from database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT file_name, total_area, total_layers, removed_vol, simulated_cam_time, actual_time, created_at FROM job_history ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows

# Initialize DB on start
init_db()

# Initialize CAM Simulation Engine
cam_engine = FoamCAMEngine()

# -------------------------------------------------------------------
# 2. CAD PROCESSING & CAM SIMULATION LOGIC
# -------------------------------------------------------------------
def process_step_file(uploaded_file, dual_head_mode=False):
    """
    Slices CAD mesh geometry, scales units, and runs feature-aware CAM simulation.
    """
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        scene = trimesh.load(temp_path)
        meshes = list(scene.geometry.values()) if isinstance(scene, trimesh.Scene) else [scene]

        stock_sizes = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
        total_area = 0.0
        total_removed_vol = 0.0
        accumulated_cam_time = 0.0
        total_swaps = 0
        all_tools = set()

        for mesh in meshes:
            # Unit Fix: Scale from Meters to Inches (STEP standard export fix)
            mesh.apply_scale(39.3701)

            extents = mesh.extents
            x, y, z = extents[0], extents[1], extents[2]

            # Categorize layer thickness
            categorized_z = 6.0
            for size in stock_sizes:
                if z <= size:
                    categorized_z = size
                    break

            stock_vol = x * y * categorized_z
            removed_vol = max(0.0, stock_vol - mesh.volume)

            total_area += (x * y)
            total_removed_vol += removed_vol

            # Detect complex 3D surface features (e.g., parallel rastering)
            is_3d = mesh.is_watertight and (len(mesh.faces) > 5000)

            # RUN FEATURE-AWARE CAM SIMULATION
            sim_res = cam_engine.estimate_layer_cam_time(
                mesh=mesh,
                layer_z_height=categorized_z,
                is_3d_surface=is_3d
            )

            accumulated_cam_time += sim_res["total_time_min"]
            total_swaps += sim_res["tool_changes"]
            all_tools.update(sim_res["tools_used"])

        # Apply Dual-Head setup factor if toggled
        final_sim_time = accumulated_cam_time * 0.5 if dual_head_mode else accumulated_cam_time

        return {
            "file_name": uploaded_file.name,
            "total_area": round(total_area, 2),
            "total_layers": len(meshes),
            "removed_vol": round(total_removed_vol, 2),
            "simulated_time_min": round(final_sim_time, 2),
            "tool_changes": total_swaps,
            "tools_list": ", ".join(all_tools) if all_tools else "5/8_POCKET"
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# -------------------------------------------------------------------
# 3. STREAMLIT USER INTERFACE
# -------------------------------------------------------------------
st.set_page_config(page_title="SourceMade CAM Estimator", page_icon="⚙️", layout="wide")

st.title("⚙️ SourceMade Feature-Aware CAM Estimator")
st.caption("Automated Foam Insert Pocketing, Perimeter, & Ramping Cycle Time Simulation")

# Top Controls Bar
col_cfg1, col_cfg2 = st.columns([1, 2])
with col_cfg1:
    dual_head_mode = st.checkbox("Dual-Head Spindle Setup (0.5x Time Multiplier)", value=False)

st.markdown("---")

# Main Upload Zone
uploaded_file = st.file_uploader("Upload STEP CAD File (.step, .stp)", type=["step", "stp"])

if uploaded_file is not None:
    with st.spinner("Analyzing CAD geometry & running kinematic CAM simulator..."):
        res = process_step_file(uploaded_file, dual_head_mode=dual_head_mode)

    st.success("Simulation Complete!")

    # Display Calculated CAD & CAM Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Simulated Cut Time", f"{res['simulated_time_min']} mins")
    m2.metric("Material Removed", f"{res['removed_vol']} in³")
    m3.metric("Total Layers", res['total_layers'])
    m4.metric("Tool Swaps", res['tool_changes'])

    st.info(f"**Tools Called in Simulation:** {res['tools_list']}")

    st.markdown("---")

    # Real-World Floor Time Logger for Database Training
    st.subheader("📝 Record Actual Floor Time")
    st.write("Log actual machine runs to feed the central database history (`vc_history.db`).")

    with st.form("log_actual_time_form"):
        actual_floor_time = st.number_input("Actual Floor Run Time (Minutes):", min_value=0.1, value=float(res['simulated_time_min']), step=0.5)
        submit_btn = st.form_submit_button("Save Job to Database")

        if submit_btn:
            save_job_to_db(
                file_name=res['file_name'],
                total_area=res['total_area'],
                total_layers=res['total_layers'],
                removed_vol=res['removed_vol'],
                sim_time=res['simulated_time_min'],
                actual_time=actual_floor_time
            )
            st.success(f"Saved **{res['file_name']}** to database!")

# -------------------------------------------------------------------
# 4. DATABASE HISTORY VIEW
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Logged Job History (`vc_history.db`)")

history_data = fetch_all_jobs()
if history_data:
    st.dataframe(
        history_data,
        column_config={
            "0": "File Name",
            "1": "Total Area (in²)",
            "2": "Layers",
            "3": "Removed Vol (in³)",
            "4": "Simulated Time (min)",
            "5": "Actual Floor Time (min)",
            "6": "Logged At"
        },
        use_container_width=True
    )
else:
    st.caption("No floor jobs logged in database yet.")
