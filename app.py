import math
import numpy as np
import trimesh
from shapely.geometry import Polygon

CAM_CONFIG = {
    "tool_change_penalty_min": 1.0,
    "tools": {
        "5/8_POCKET": {
            "diameter": 0.625,
            "cut_ipm": 350.0,
            "ramp_ipm": 150.0,
            "ramp_angle_deg": 45.0,
            "max_stepdown": 1.5,
            "stepover_ratio": 0.65
        },
        "5/8_PROFILE": {
            "diameter": 0.625,
            "cut_ipm": 150.0,
            "ramp_ipm": 150.0,
            "ramp_angle_deg": 45.0,
            "max_stepdown": 1.0,
            "stepover_ratio": 1.0
        },
        "3/8_FOAM": {
            "diameter": 0.375,
            "cut_ipm": 350.0,
            "ramp_ipm": 13.333,
            "ramp_angle_deg": 89.0,
            "max_stepdown": 1.0,
            "stepover_ratio": 0.60
        },
        "1/4_3D_SURFER": {
            "diameter": 0.250,
            "cut_ipm": 150.0,
            "ramp_ipm": 150.0,
            "ramp_angle_deg": 20.0,
            "max_stepdown": 0.5,
            "lookahead_penalty": 1.35
        }
    }
}

class FoamCAMEngine:
    def __init__(self, config=CAM_CONFIG):
        self.cfg = config
        self.tools = config["tools"]

    def select_pocket_tool(self, min_slot_width):
        if min_slot_width >= 0.625:
            return "5/8_POCKET"
        elif min_slot_width >= 0.375:
            return "3/8_FOAM"
        else:
            return "1/4_3D_SURFER"

    def estimate_layer_cam_time(self, mesh, layer_z_height, removed_vol=0.0, is_3d_surface=False):
        
        # New Feature Buckets
        time_outline = 0.0
        time_pocket = 0.0
        time_3d = 0.0
        
        unique_tools_used = set()
        
        try:
            top_z = mesh.bounds[1][2]
            slice_plane_origin = [0, 0, top_z - 0.1]
            slice_2d = mesh.section(plane_origin=slice_plane_origin, plane_normal=[0, 0, 1])

            if slice_2d is None:
                center_z = (mesh.bounds[0][2] + mesh.bounds[1][2]) / 2.0
                slice_2d = mesh.section(plane_origin=[0, 0, center_z], plane_normal=[0, 0, 1])

            if slice_2d is not None:
                path_2d, _ = slice_2d.to_planar()
                polygons = path_2d.polygons_full
            else:
                polygons = []
                
        except Exception:
            polygons = []

        if polygons:
            prof_tool = self.tools["5/8_PROFILE"]
            unique_tools_used.add("5/8_PROFILE")

            for poly in polygons:
                if poly is None or poly.is_empty:
                    continue

                # 1. OUTLINE PROCESSING (Exteriors)
                ext_length = poly.exterior.length
                prof_passes = math.ceil(layer_z_height / prof_tool["max_stepdown"])
                
                ext_cut_time = (ext_length / prof_tool["cut_ipm"]) * prof_passes
                ramp_dist = layer_z_height / math.sin(math.radians(prof_tool["ramp_angle_deg"]))
                ext_ramp_time = (ramp_dist / prof_tool["ramp_ipm"]) * prof_passes
                
                time_outline += (ext_cut_time + ext_ramp_time)

                # 2. POCKET & 3D PROCESSING (Interiors)
                for interior in poly.interiors:
                    pocket_poly = Polygon(interior)
                    pocket_area = pocket_poly.area

                    min_bounds_dim = min(
                        pocket_poly.bounds[2] - pocket_poly.bounds[0], 
                        pocket_poly.bounds[3] - pocket_poly.bounds[1]
                    )

                    pocket_tool_key = self.select_pocket_tool(min_bounds_dim)
                    p_tool = self.tools[pocket_tool_key]
                    unique_tools_used.add(pocket_tool_key)

                    effective_stepover = p_tool["diameter"] * p_tool["stepover_ratio"]
                    clearing_distance = pocket_area / effective_stepover
                    pocket_passes = math.ceil(layer_z_height / p_tool["max_stepdown"])

                    pocket_cut_time = (clearing_distance / p_tool["cut_ipm"]) * pocket_passes
                    pocket_ramp_dist = layer_z_height / math.sin(math.radians(p_tool["ramp_angle_deg"]))
                    pocket_ramp_time = (pocket_ramp_dist / p_tool["ramp_ipm"]) * pocket_passes

                    total_interior_time = pocket_cut_time + pocket_ramp_time

                    if is_3d_surface or pocket_tool_key == "1/4_3D_SURFER":
                        lookahead_multiplier = p_tool.get("lookahead_penalty", 1.35)
                        time_3d += (total_interior_time * lookahead_multiplier)
                    else:
                        time_pocket += total_interior_time

        # 3. VOLUMETRIC MRR FAILSAFE
        total_raw_time = time_outline + time_pocket + time_3d
        if total_raw_time == 0.0 and removed_vol > 0.1:
            estimated_cut_time = (removed_vol / 125.0) * 1.25
            unique_tools_used.add("5/8_POCKET")
            if is_3d_surface:
                time_3d += estimated_cut_time
            else:
                time_pocket += estimated_cut_time

        # Distribute Tool Swap Penalty proportionally across the buckets
        tool_changes = max(0, len(unique_tools_used) - 1)
        tool_swap_penalty = tool_changes * self.cfg["tool_change_penalty_min"]
        
        if total_raw_time > 0:
            time_outline += tool_swap_penalty * (time_outline / total_raw_time)
            time_pocket += tool_swap_penalty * (time_pocket / total_raw_time)
            time_3d += tool_swap_penalty * (time_3d / total_raw_time)

        return {
            "total_time_min": time_outline + time_pocket + time_3d,
            "est_outline": time_outline,
            "est_pocket": time_pocket,
            "est_3d": time_3d,
            "tool_changes": tool_changes,
            "tools_used": list(unique_tools_used)
        }
