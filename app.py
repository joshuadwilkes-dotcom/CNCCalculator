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

def save_job_to_db(file_name, total_area, total_layers, removed_vol, sim_time, actual_time):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO job_history (file_name, total_area, total_layers, removed_vol, simulated_cam_time, actual_time)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (file_name, total_area, total_layers, removed_vol, sim_time, actual_time))
    conn.commit()
    conn.close()

def fetch_all_jobs():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        # Added 'id' to the SELECT statement so we can target rows for deletion
        c.execute("SELECT id, file_name, total_area, total_layers, removed_vol, simulated_cam_time, actual_time, created_at FROM job_history ORDER BY id DESC")
        rows = c.fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()
    return rows

def delete_job_by_id(job_id):
    """Deletes a specific job from the database by its ID."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM job_history WHERE id = ?", (job_id,))
    conn.commit()
    conn.close()

init_db()
cam_engine = FoamCAMEngine()

# -------------------------------------------------------------------
# 2. CAD PROCESSING & CAM SIMULATION LOGIC
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
        
        # Failsafe: if the STEP exported solid without pockets, or objects filled the assembly
        if math.isnan(removed_vol) or removed_vol <= 0.5:
            # Estimate a standard ~35% volume removal for heavy custom foam cases
            removed_vol = stock_vol * 0.35

        total_area = final_x * final_y
        is_3d = getattr(mesh, "is_watertight", False) and (len(getattr(mesh, "faces", [])) > 5000)

        # RUN CAM SIMULATION
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
    unit_choice = st.radio(
        "1. STEP Export Units", 
        ["Millimeters (Standard for STEP)", "Inches", "Meters"],
        index=2, # Meters is now set as default
        help="STEP files often default to mm behind the scenes. Adjust this if your detected part size is incorrect."
    )

with col2:
    uploaded_file = st.file_uploader("2. Upload STEP File (.step, .stp)", type=["step", "stp"])

if uploaded_file is not None:
    with st.spinner("Analyzing CAD geometry & running kinematic CAM simulator..."):
        res = process_step_file(uploaded_file, unit_choice)

    if res is not None:
        st.success(f"Simulation Complete! (Detected Outer Bounds: **{res['detected_x']}\" x {res['detected_y']}\"**)")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Simulated Cut Time", f"{res['simulated_time_min']} mins")
        m2.metric("Material Removed", f"{res['removed_vol']} in³")
        m3.metric("Total Layers", res['total_layers'])
        m4.metric("Tool Swaps", res['tool_changes'])

        st.info(f"**Tools Called in Simulation:** {res['tools_list']}")

        st.markdown("---")

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
# 4. DATABASE HISTORY VIEW & OUTLIER DETECTION
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Logged Job History & Outlier Tracking")

history_data = fetch_all_jobs()
if history_data:
    # Convert SQL data into a Pandas DataFrame for analytics
    df = pd.DataFrame(history_data, columns=[
        "ID", "File Name", "Area (in²)", "Layers", "Removed Vol (in³)", 
        "Simulated (min)", "Actual (min)", "Logged At"
    ])

    # Calculate the Error Percentage
    # Formula: Absolute Difference / Simulated Time * 100
    safe_sim = df["Simulated (min)"].replace(0, 0.1)
    df["Error %"] = ((df["Actual (min)"] - df["Simulated (min)"]).abs() / safe_sim) * 100

    # Define Outlier Logic: Flag if error is > 25% AND off by more than 2 minutes
    def flag_outlier(row):
        if row["Error %"] > 25.0 and abs(row["Actual (min)"] - row["Simulated (min)"]) >= 2.0:
            return "⚠️ Outlier"
        return "✅ Accurate"

    df["Status"] = df.apply(flag_outlier, axis=1)

    # Style function to highlight outlier rows with a soft red background
    def highlight_outliers(row):
        color = 'background-color: rgba(255, 75, 75, 0.15)' if row["Status"] == "⚠️ Outlier" else ''
        return [color] * len(row)

    # Format numbers for clean display
    styled_df = df.style.apply(highlight_outliers, axis=1).format({
        "Simulated (min)": "{:.2f}",
        "Actual (min)": "{:.2f}",
        "Error %": "{:.1f}%",
        "Removed Vol (in³)": "{:.2f}",
        "Area (in²)": "{:.1f}"
    })

    # Render the table
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # UI tool for deleting rows one-by-one
    st.markdown("#### Manage Database")
    del_col1, del_col2 = st.columns([3, 1])
    
    with del_col1:
        # Create a dropdown mapping formatted as "12 - insert_base.step"
        delete_options = df["ID"].astype(str) + " - " + df["File Name"]
        selected_job = st.selectbox("Select a specific job to remove:", delete_options)
        
    with del_col2:
        st.write("") # Formatting spacer
        st.write("") # Formatting spacer
        if st.button("🗑️ Delete Job", use_container_width=True):
            if selected_job:
                # Extract just the ID number from the string
                job_id = int(selected_job.split(" - ")[0])
                delete_job_by_id(job_id)
                st.rerun() # Refresh page instantly

else:
    st.caption("No floor jobs logged in database yet.")
