import os
import sqlite3
import math
import numpy as np
import pandas as pd
import streamlit as st
import trimesh

from cam_engine import FoamCAMEngine

# -------------------------------------------------------------------
# 1. DATABASE SETUP & AUTO-MIGRATION
# -------------------------------------------------------------------
DB_PATH = "vc_history.db"

def init_db():
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

def save_job_to_db(file_name, total_area, total_layers, removed_vol, raw_sim_time, actual_time):
    # CRITICAL: We always save the RAW sim time to the database to prevent recursive drift.
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO job_history (file_name, total_area, total_layers, removed_vol, simulated_cam_time, actual_time)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (file_name, total_area, total_layers, removed_vol, raw_sim_time, actual_time))
    conn.commit()
    conn.close()

def fetch_all_jobs():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("SELECT id, file_name, total_area, total_layers, removed_vol, simulated_cam_time, actual_time, created_at FROM job_history ORDER BY id DESC")
        rows = c.fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()
    return rows

def delete_job_by_id(job_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM job_history WHERE id = ?", (job_id,))
    conn.commit()
    conn.close()

init_db()
cam_engine = FoamCAMEngine()

# -------------------------------------------------------------------
# 2. MACHINE LEARNING / CALIBRATION LOGIC
# -------------------------------------------------------------------
def calculate_calibration_factor(df):
    """
    Looks at historical jobs and creates a dynamic time multiplier based on floor reality.
    """
    if df is None or df.empty:
        return 1.0

    # Only look at valid jobs where time is recorded
    valid_jobs = df[(df["Actual (min)"] > 0) & (df["Simulated (min)"] > 0)]
    if valid_jobs.empty:
        return 1.0
    
    # Calculate the ratio (e.g., if actual is 24 mins and simulated was 20, ratio is 1.2)
    ratios = valid_jobs["Actual (min)"] / valid_jobs["Simulated (min)"]
    
    # Filter out crazy outliers for the learning math so a single mistake doesn't ruin the algorithm
    # (Only learn from jobs that are between 50% and 200% of the estimate)
    valid_ratios = ratios[(ratios >= 0.5) & (ratios <= 2.0)]
    
    if valid_ratios.empty:
        return 1.0
        
    # Return the average multiplier
    return float(valid_ratios.mean())

# -------------------------------------------------------------------
# 3. CAD PROCESSING & CAM SIMULATION LOGIC
# -------------------------------------------------------------------
def process_step_file(uploaded_file, unit_choice):
    temp_path = f"temp_{uploaded_file.name}"
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        try:
            scene = trimesh.load(temp_path, file_type='step')
        except Exception as e:
            st.error(f"STEP parser failed. Ensure Streamlit Cloud is set to Python 3.11 or 3.12. Error: {e}")
            return None

        # FLATTEN MULTI-BODY ASSEMBLIES
        if isinstance(scene, trimesh.Scene):
            mesh = scene.dump(concatenate=True)
        else:
            mesh = scene

        if mesh.is_empty:
            st.error("No valid 3D geometry could be extracted from the uploaded file.")
            return None

        # APPLY UNIT SCALING TO INCHES
        scale_factor = 1.0
        if unit_choice == "Millimeters (Standard for STEP)":
            scale_factor = 1.0 / 25.4
        elif unit_choice == "Meters":
            scale_factor = 39.3701

        mesh.apply_scale(scale_factor)

        # FIX CAD ORIENTATION (AUTO-LAY FLAT)
        extents = mesh.extents
        min_axis = np.argmin(extents)
        
        if min_axis == 0: 
            rot_matrix = trimesh.transformations.rotation_matrix(math.pi/2, [0, 1, 0])
            mesh.apply_transform(rot_matrix)
        elif min_axis == 1: 
            rot_matrix = trimesh.transformations.rotation_matrix(math.pi/2, [1, 0, 0])
            mesh.apply_transform(rot_matrix)
            
        extents = mesh.extents
        final_x, final_y, z = extents[0], extents[1], extents[2]

        # Ensure the part sits clean on the virtual origin
        center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
        mesh.apply_translation([-center[0], -center[1], -mesh.bounds[0][2]])

        # DETERMINE Z STOCK SIZE
        stock_sizes = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0]
        categorized_z = 9.0
        for size in stock_sizes:
            if z <= size:
                categorized_z = size
                break

        # FOOLPROOF MATERIAL REMOVED CALCULATION
        stock_vol = final_x * final_y * categorized_z
        try:
            actual_mesh_vol = abs(mesh.volume)
        except Exception:
            actual_mesh_vol = stock_vol 
            
        removed_vol = stock_vol - actual_mesh_vol
        
        if math.isnan(removed_vol) or removed_vol <= 0.5:
            removed_vol = stock_vol * 0.35

        total_area = final_x * final_y
        is_3d = getattr(mesh, "is_watertight", False) and (len(getattr(mesh, "faces", [])) > 5000)

        # RUN CAM SIMULATION (This is the RAW math baseline)
        sim_res = cam_engine.estimate_layer_cam_time(
            mesh=mesh,
            layer_z_height=categorized_z,
            removed_vol=removed_vol,
            is_3d_surface=is_3d
        )

        return {
            "file_name": uploaded_file.name,
            "total_area": round(total_area, 2),
            "total_layers": 1,
            "removed_vol": round(removed_vol, 2),
            "raw_simulated_time_min": sim_res.get("total_time_min", 0.0), # Storing unadjusted raw math
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
# 4. STREAMLIT USER INTERFACE
# -------------------------------------------------------------------
st.set_page_config(page_title="SourceMade CAM Estimator", page_icon="⚙️", layout="wide")

st.title("⚙️ SourceMade Feature-Aware CAM Estimator")
st.caption("Automated Foam Insert Pocketing, Perimeter, & Ramping Cycle Time Simulation")

# Fetch database history and calculate learning multiplier on boot
history_data = fetch_all_jobs()
df = pd.DataFrame(history_data, columns=[
    "ID", "File Name", "Area (in²)", "Layers", "Removed Vol (in³)", 
    "Simulated (min)", "Actual (min)", "Logged At"
]) if history_data else pd.DataFrame()

machine_calibration_factor = calculate_calibration_factor(df)

# Show calibration status to the user
if machine_calibration_factor != 1.0:
    st.info(f"🧠 **Learning Engine Active:** The system has learned that your machine runs **{machine_calibration_factor:.2f}x** compared to raw kinematic math. Future estimates are now being auto-adjusted.")

st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    unit_choice = st.radio(
        "1. STEP Export Units", 
        ["Millimeters (Standard for STEP)", "Inches", "Meters"],
        index=2,
        help="STEP files often default to mm behind the scenes. Adjust this if your detected part size is incorrect."
    )

with col2:
    uploaded_file = st.file_uploader("2. Upload STEP File (.step, .stp)", type=["step", "stp"])

if uploaded_file is not None:
    with st.spinner("Analyzing CAD geometry & running kinematic CAM simulator..."):
        res = process_step_file(uploaded_file, unit_choice)

    if res is not None:
        st.success(f"Simulation Complete! (Detected Outer Bounds: **{res['detected_x']}\" x {res['detected_y']}\"**)")

        # APPLY THE CALIBRATION MULTIPLIER
        adjusted_sim_time = res['raw_simulated_time_min'] * machine_calibration_factor

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Simulated Cut Time", f"{adjusted_sim_time:.2f} mins", delta=f"{machine_calibration_factor:.2f}x modifier" if machine_calibration_factor != 1.0 else None, delta_color="off")
        m2.metric("Material Removed", f"{res['removed_vol']} in³")
        m3.metric("Total Layers", res['total_layers'])
        m4.metric("Tool Swaps", res['tool_changes'])

        st.info(f"**Tools Called in Simulation:** {res['tools_list']}")

        st.markdown("---")

        st.subheader("📝 Record Actual Floor Time")
        st.write("Log actual machine runs to feed the central database history (`vc_history.db`).")

        with st.form("log_actual_time_form"):
            safe_default = max(0.1, float(adjusted_sim_time))
            actual_floor_time = st.number_input("Actual Floor Run Time (Minutes):", min_value=0.1, value=safe_default, step=0.5)
            submit_btn = st.form_submit_button("Save Job to Database")

            if submit_btn:
                # Save the RAW time to the database so the baseline math never drifts
                save_job_to_db(
                    file_name=res['file_name'],
                    total_area=res['total_area'],
                    total_layers=res['total_layers'],
                    removed_vol=res['removed_vol'],
                    raw_sim_time=res['raw_simulated_time_min'], 
                    actual_time=actual_floor_time
                )
                st.success(f"Saved **{res['file_name']}** to database! The learning engine will update on your next upload.")

# -------------------------------------------------------------------
# 5. DATABASE HISTORY VIEW & OUTLIER DETECTION
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Logged Job History & Outlier Tracking")

if not df.empty:
    safe_sim = df["Simulated (min)"].replace(0, 0.1)
    df["Error %"] = ((df["Actual (min)"] - df["Simulated (min)"]).abs() / safe_sim) * 100

    def flag_outlier(row):
        if row["Error %"] > 25.0 and abs(row["Actual (min)"] - row["Simulated (min)"]) >= 2.0:
            return "⚠️ Outlier"
        return "✅ Accurate"

    df["Status"] = df.apply(flag_outlier, axis=1)

    def highlight_outliers(row):
        color = 'background-color: rgba(255, 75, 75, 0.15)' if row["Status"] == "⚠️ Outlier" else ''
        return [color] * len(row)

    styled_df = df.style.apply(highlight_outliers, axis=1).format({
        "Simulated (min)": "{:.2f}",
        "Actual (min)": "{:.2f}",
        "Error %": "{:.1f}%",
        "Removed Vol (in³)": "{:.2f}",
        "Area (in²)": "{:.1f}"
    })

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    st.markdown("#### Manage Database")
    del_col1, del_col2 = st.columns([3, 1])
    
    with del_col1:
        delete_options = df["ID"].astype(str) + " - " + df["File Name"]
        selected_job = st.selectbox("Select a specific job to remove:", delete_options)
        
    with del_col2:
        st.write("") 
        st.write("") 
        if st.button("🗑️ Delete Job", use_container_width=True):
            if selected_job:
                job_id = int(selected_job.split(" - ")[0])
                delete_job_by_id(job_id)
                st.rerun() 

else:
    st.caption("No floor jobs logged in database yet.")
