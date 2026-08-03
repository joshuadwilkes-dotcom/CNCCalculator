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
        
    if "est_outline" not in existing_columns:
        c.execute("ALTER TABLE job_history ADD COLUMN est_outline REAL DEFAULT 0.0")
    if "est_pocket" not in existing_columns:
        c.execute("ALTER TABLE job_history ADD COLUMN est_pocket REAL DEFAULT 0.0")
    if "est_3d" not in existing_columns:
        c.execute("ALTER TABLE job_history ADD COLUMN est_3d REAL DEFAULT 0.0")
        
    if "created_at" not in existing_columns:
        c.execute("ALTER TABLE job_history ADD COLUMN created_at TIMESTAMP")
        
    conn.commit()
    conn.close()

def save_job_to_db(file_name, total_area, total_layers, removed_vol, est_outline, est_pocket, est_3d, total_sim, actual_time):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO job_history 
        (file_name, total_area, total_layers, removed_vol, est_outline, est_pocket, est_3d, simulated_cam_time, actual_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (file_name, total_area, total_layers, removed_vol, est_outline, est_pocket, est_3d, total_sim, actual_time))
    conn.commit()
    conn.close()

def fetch_all_jobs():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("SELECT id, file_name, removed_vol, est_outline, est_pocket, est_3d, simulated_cam_time, actual_time FROM job_history ORDER BY id DESC")
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
# 2. FEATURE-BASED MULTIPLE LINEAR REGRESSION
# -------------------------------------------------------------------
def calculate_feature_multipliers(df):
    base_mults = {"outline": 1.0, "pocket": 1.0, "3d": 1.0}
    
    if df is None or df.empty:
        return base_mults

    valid = df[(df["Actual (min)"] > 0) & (df["Total Simulated (min)"] > 0)]
    if len(valid) < 3: 
        return base_mults 

    X = valid[["est_outline", "est_pocket", "est_3d"]].values
    y = valid["Actual (min)"].values

    prior_X = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0]
    ])
    prior_y = np.array([10.0, 10.0, 10.0])

    X_reg = np.vstack([X, prior_X])
    y_reg = np.concatenate([y, prior_y])

    coeffs = np.linalg.lstsq(X_reg, y_reg, rcond=None)[0]

    base_mults["outline"] = max(0.25, min(4.0, coeffs[0]))
    base_mults["pocket"] = max(0.25, min(4.0, coeffs[1]))
    base_mults["3d"] = max(0.25, min(4.0, coeffs[2]))

    return base_mults

history_data = fetch_all_jobs()
df = pd.DataFrame(history_data, columns=[
    "ID", "File Name", "Removed Vol (in³)", 
    "est_outline", "est_pocket", "est_3d", "Total Simulated (min)", "Actual (min)"
]) if history_data else pd.DataFrame()

multipliers = calculate_feature_multipliers(df)

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
            st.error(f"STEP parser failed. Error: {e}")
            return None

        # Break the assembly apart into distinct bodies/layers
        meshes = list(scene.geometry.values()) if isinstance(scene, trimesh.Scene) else [scene]
        if not meshes: return None
        
        scale_factor = 1.0 / 25.4 if "Millimeters" in unit_choice else 39.3701 if unit_choice == "Meters" else 1.0

        # Accumulators for the entire assembly
        total_outline_time = 0.0
        total_pocket_time = 0.0
        total_3d_time = 0.0
        total_raw_time = 0.0
        total_removed_vol = 0.0
        total_area = 0.0
        overall_tools = set()
        
        valid_layers = 0
        max_detected_x = 0.0
        max_detected_y = 0.0

        for mesh in meshes:
            if mesh.is_empty: continue
            
            mesh.apply_scale(scale_factor)

            # Isolate and orient THIS specific layer
            extents = mesh.extents
            
            # Skip micro-bodies (e.g. tiny screws or artifacts accidentally left in the STEP)
            if max(extents) < 1.0: 
                continue 

            min_axis = np.argmin(extents)
            if min_axis == 0: mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2, [0, 1, 0]))
            elif min_axis == 1: mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2, [1, 0, 0]))
                
            extents = mesh.extents
            final_x, final_y, z = extents[0], extents[1], extents[2]
            
            max_detected_x = max(max_detected_x, final_x)
            max_detected_y = max(max_detected_y, final_y)

            # Center the individual layer on the Z origin
            center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
            mesh.apply_translation([-center[0], -center[1], -mesh.bounds[0][2]])

            categorized_z = 9.0
            for size in [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0]:
                if z <= size:
                    categorized_z = size
                    break

            stock_vol = final_x * final_y * categorized_z
            actual_mesh_vol = abs(mesh.volume) if not math.isnan(getattr(mesh, "volume", float('nan'))) else stock_vol 
            removed_vol = max(0.0, stock_vol - actual_mesh_vol)
            if removed_vol <= 0.5: removed_vol = stock_vol * 0.35

            is_3d = getattr(mesh, "is_watertight", False) and (len(getattr(mesh, "faces", [])) > 5000)

            # Generate math for THIS layer
            sim_res = cam_engine.estimate_layer_cam_time(
                mesh=mesh, layer_z_height=categorized_z, removed_vol=removed_vol, is_3d_surface=is_3d
            )

            # Accumulate totals
            total_outline_time += sim_res["est_outline"]
            total_pocket_time += sim_res["est_pocket"]
            total_3d_time += sim_res["est_3d"]
            total_raw_time += sim_res["total_time_min"]
            total_removed_vol += removed_vol
            total_area += (final_x * final_y)
            overall_tools.update(sim_res.get("tools_used", []))
            valid_layers += 1

        if valid_layers == 0:
            st.error("No valid foam layers found in assembly.")
            return None

        return {
            "file_name": uploaded_file.name,
            "total_area": round(total_area, 2),
            "total_layers": valid_layers,
            "removed_vol": round(total_removed_vol, 2),
            "raw_outline": total_outline_time,
            "raw_pocket": total_pocket_time,
            "raw_3d": total_3d_time,
            "raw_total": total_raw_time,
            "tool_changes": max(0, len(overall_tools) - 1),
            "tools_list": ", ".join(overall_tools) if overall_tools else "5/8_POCKET",
            "detected_x": round(max_detected_x, 2),
            "detected_y": round(max_detected_y, 2)
        }

    except Exception as e:
        st.error(f"Error parsing geometry: {e}")
        return None
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

# -------------------------------------------------------------------
# 4. STREAMLIT USER INTERFACE
# -------------------------------------------------------------------
st.set_page_config(page_title="SourceMade CAM Estimator", page_icon="⚙️", layout="wide")

st.title("⚙️ SourceMade Feature-Aware CAM Estimator")
st.caption("Automated Foam Insert Pocketing, Perimeter, & Ramping Cycle Time Simulation")

st.info(f"🧠 **Live ML Kinematic Modifiers:** Outline (`{multipliers['outline']:.2f}x`) | Pocket (`{multipliers['pocket']:.2f}x`) | 3D Surface (`{multipliers['3d']:.2f}x`)")
st.markdown("---")

col1, col2 = st.columns([1, 2])
with col1:
    unit_choice = st.radio("1. STEP Export Units", ["Millimeters (Standard for STEP)", "Inches", "Meters"], index=2)
with col2:
    uploaded_file = st.file_uploader("2. Upload STEP File (.step, .stp)", type=["step", "stp"])

if uploaded_file is not None:
    with st.spinner("Analyzing multi-body CAD geometry & predicting cycle times..."):
        res = process_step_file(uploaded_file, unit_choice)

    if res is not None:
        st.success(f"Simulation Complete! (Processed {res['total_layers']} Bodies | Largest Profile: **{res['detected_x']}\" x {res['detected_y']}\"**)")

        adj_outline = res['raw_outline'] * multipliers['outline']
        adj_pocket = res['raw_pocket'] * multipliers['pocket']
        adj_3d = res['raw_3d'] * multipliers['3d']
        final_adjusted_time = adj_outline + adj_pocket + adj_3d

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Floor Time", f"{final_adjusted_time:.2f} mins")
        m2.metric("Material Removed", f"{res['removed_vol']} in³")
        m3.metric("Total Layers", res['total_layers'])
        m4.metric("Tool Swaps", res['tool_changes'])

        st.info(f"**Tools Called:** {res['tools_list']} | **Adjusted Breakdown:** {adj_outline:.1f}m Outline, {adj_pocket:.1f}m Pocket, {adj_3d:.1f}m 3D")

        st.markdown("---")
        st.subheader("📝 Record Actual Floor Time")
        
        with st.form("log_actual_time_form"):
            safe_default = max(0.1, float(final_adjusted_time))
            actual_floor_time = st.number_input("Actual Floor Run Time (Minutes):", min_value=0.1, value=safe_default, step=0.5)
            submit_btn = st.form_submit_button("Save Job to Database")

            if submit_btn:
                save_job_to_db(
                    file_name=res['file_name'], total_area=res['total_area'], total_layers=res['total_layers'], removed_vol=res['removed_vol'],
                    est_outline=res['raw_outline'], est_pocket=res['raw_pocket'], est_3d=res['raw_3d'], 
                    total_sim=res['raw_total'], actual_time=actual_floor_time
                )
                st.success(f"Saved to database! The ML engine will recalibrate on your next upload.")

# -------------------------------------------------------------------
# 5. DATABASE HISTORY VIEW & OUTLIER DETECTION
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Logged Job History & ML Target Tracking")

if not df.empty:
    df["Adj. Predicted"] = (df["est_outline"] * multipliers['outline']) + \
                           (df["est_pocket"] * multipliers['pocket']) + \
                           (df["est_3d"] * multipliers['3d'])

    df["Error %"] = ((df["Actual (min)"] - df["Adj. Predicted"]).abs() / df["Adj. Predicted"].replace(0, 0.1)) * 100

    def flag_outlier(row):
        if row["Error %"] > 25.0 and abs(row["Actual (min)"] - row["Adj. Predicted"]) >= 2.0:
            return "⚠️ Outlier"
        return "✅ Accurate"

    df["Status"] = df.apply(flag_outlier, axis=1)

    def highlight_outliers(row):
        return ['background-color: rgba(255, 75, 75, 0.15)' if row["Status"] == "⚠️ Outlier" else ''] * len(row)

    display_df = df[["ID", "File Name", "Removed Vol (in³)", "Total Simulated (min)", "Adj. Predicted", "Actual (min)", "Error %", "Status"]]
    
    styled_df = display_df.style.apply(highlight_outliers, axis=1).format({
        "Total Simulated (min)": "{:.2f}",
        "Adj. Predicted": "{:.2f}",
        "Actual (min)": "{:.2f}",
        "Error %": "{:.1f}%",
        "Removed Vol (in³)": "{:.2f}"
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
                delete_job_by_id(int(selected_job.split(" - ")[0]))
                st.rerun() 
else:
    st.caption("No floor jobs logged in database yet.")
