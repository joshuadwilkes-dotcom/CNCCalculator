import os
import sqlite3
import math
import streamlit as st
import trimesh

from cam_engine import FoamCAMEngine

# -------------------------------------------------------------------
# 1. DATABASE SETUP & AUTO-MIGRATION
# -------------------------------------------------------------------
DB_PATH = "vc_history.db"

def init_db():
    """Initializes and safely updates database schema if missing columns exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS job_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            total_area REAL,
            total_layers INTEGER,
            removed_vol REAL,
            simulated_cam_time REAL DEFAULT 0.0,
            actual_time REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    c.execute("PRAGMA table_info(job_history)")
    existing_columns = [col[1] for col in c.fetchall()]
    
    if "simulated_cam_time" not in existing_columns:
        c.execute("ALTER TABLE job_history ADD COLUMN simulated_cam_time REAL DEFAULT 0.0")
        
    if "created_at" not in existing_columns:
        c.execute("ALTER TABLE job_history ADD COLUMN created_at TIMESTAMP")
        
    conn.commit()
    conn.close()

def save_job_to_db(file_name, total_area, total_layers, removed_vol, sim_time, actual_time):
    """Saves completed floor jobs to SQLite."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO job_history (file_name, total_area, total_layers, removed_vol, simulated_cam_time, actual_time)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (file_name, total_area, total_layers, removed_vol, sim_time, actual_time))
    conn.commit()
    conn.close()

def fetch_all_jobs():
    """Retrieves recorded jobs safely."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("SELECT file_name, total_area, total_layers, removed_vol, simulated_cam_time, actual_time, created_at FROM job_history ORDER BY id DESC")
        rows = c.fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()
    return rows

# Initialize DB on app boot
init_db()
cam_engine = FoamCAMEngine()

# -------------------------------------------------------------------
# 2. CAD PROCESSING & CAM SIMULATION LOGIC
# -------------------------------------------------------------------
def process_step_file(uploaded_file, unit_choice):
    """
    Loads STEP mesh geometry, flattens assemblies, scales to Inches, and runs CAM simulation.
    """
    temp_path = f"temp_{uploaded_file.name}"
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        try:
            scene = trimesh.load(temp_path, file_type='step')
        except Exception as e:
            st.error(f"STEP parser failed. Ensure Streamlit Cloud is set to Python 3.11 or 3.12. Error: {e}")
            return None

        # 1. FLATTEN MULTI-BODY ASSEMBLIES
        # If the STEP file has multiple parts (e.g. lid, base, cutouts), merge them into one 
        # single unified mesh. This preserves their exact positions and gives us the true bounding box.
        if isinstance(scene, trimesh.Scene):
            mesh = scene.dump(concatenate=True)
        else:
            mesh = scene

        if mesh.is_empty:
            st.error("No valid 3D geometry could be extracted from the uploaded file.")
            return None

        # 2. APPLY UNIT SCALING TO INCHES
        scale_factor = 1.0
        if unit_choice == "Millimeters (Standard for STEP)":
            scale_factor = 1.0 / 25.4
        elif unit_choice == "Meters":
            scale_factor = 39.3701
        # If "Inches", scale remains 1.0

        mesh.apply_scale(scale_factor)

        # 3. GET PHYSICAL DIMENSIONS
        extents = mesh.extents
        final_x, final_y, z = extents[0], extents[1], extents[2]

        stock_sizes = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0]
        categorized_z = 9.0
        for size in stock_sizes:
            if z <= size:
                categorized_z = size
                break

        stock_vol = final_x * final_y * categorized_z
        removed_vol = max(0.0, stock_vol - mesh.volume)
        total_area = final_x * final_y

        # Detect complex 3D surface features
        is_3d = getattr(mesh, "is_watertight", False) and (len(getattr(mesh, "faces", [])) > 5000)

        # 4. RUN CAM SIMULATION
        sim_res = cam_engine.estimate_layer_cam_time(
            mesh=mesh,
            layer_z_height=categorized_z,
            is_3d_surface=is_3d
        )

        return {
            "file_name": uploaded_file.name,
            "total_area": round(total_area, 2),
            "total_layers": 1, # Flattened to a single stock piece
            "removed_vol": round(removed_vol, 2),
            "simulated_time_min": round(sim_res.get("total_time_min", 0.0), 2),
            "tool_changes": sim_res.get("tool_changes", 0),
            "tools_list": ", ".join(sim_res.get("tools_used", [])) if sim_res.get("tools_used") else "5/8_POCKET",
            "detected_x": round(final_x, 2),
            "detected_y": round(final_y, 2)
        }

    except Exception as e:
        st.error(f"Error parsing geometry: {e}")
        return None

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# -------------------------------------------------------------------
# 3. STREAMLIT USER INTERFACE
# -------------------------------------------------------------------
st.set_page_config(page_title="SourceMade CAM Estimator", page_icon="⚙️", layout="wide")

st.title("⚙️ SourceMade Feature-Aware CAM Estimator")
st.caption("Automated Foam Insert Pocketing, Perimeter, & Ramping Cycle Time Simulation")

st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    # Most CAD software (Fusion/Onshape) writes STEPs in Millimeters by default
    unit_choice = st.radio(
        "1. STEP Export Units", 
        ["Millimeters (Standard for STEP)", "Inches", "Meters"],
        index=0, 
        help="STEP files often default to mm behind the scenes. Adjust this if your detected part size is incorrect."
    )

with col2:
    uploaded_file = st.file_uploader("2. Upload STEP File (.step, .stp)", type=["step", "stp"])

if uploaded_file is not None:
    with st.spinner("Analyzing CAD geometry & running kinematic CAM simulator..."):
        res = process_step_file(uploaded_file, unit_choice)

    if res is not None:
        st.success(f"Simulation Complete! (Detected Outer Bounds: **{res['detected_x']}\" x {res['detected_y']}\"**)")

        # Calculated CAD & CAM Metrics
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
            safe_default = max(0.1, float(res['simulated_time_min']))
            actual_floor_time = st.number_input("Actual Floor Run Time (Minutes):", min_value=0.1, value=safe_default, step=0.5)
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
